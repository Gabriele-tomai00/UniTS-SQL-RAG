from pathlib import Path
import re
import time

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.prompts import PromptTemplate
from sqlalchemy import create_engine
import chromadb
from llama_index.core import Settings, SQLDatabase, VectorStoreIndex
from llama_index.core.indices.struct_store.sql_query import SQLTableRetrieverQueryEngine
from llama_index.core.objects import ObjectIndex, SQLTableNodeMapping, SQLTableSchema
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

def get_prompt_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

Settings.llm = OpenAILike(
    model="ggml-org/gpt-oss-120b-GGUF",
    api_base=os.getenv("LLM_API_BASE"),
    api_key=os.getenv("LLM_API_KEY", "not_necessary"),
    context_window=8192,
    max_tokens=2048,
    temperature=0.1,
    is_chat_model=True,
    system_prompt=get_prompt_from_file("prompt_for_llm.txt"),
    timeout=30,
)


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

# Mapping of special apostrophe/quote variants to standard ASCII apostrophe.
# We then replace the apostrophe itself with a space to avoid SQL issues.
_APOSTROPHE_VARIANTS = str.maketrans({
    "\u2019": "'",   # right single quotation mark  →  '
    "\u2018": "'",   # left single quotation mark   →  '
    "\u02BC": "'",   # modifier letter apostrophe   →  '
    "\u0060": "'",   # grave accent                 →  '
    "\u00B4": "'",   # acute accent                 →  '
})

def normalize_text(value) -> str | None:
    """
    Normalize a text value before inserting into the DB:
      1. Skip non-string values unchanged (None, int, bool...)
      2. Strip leading/trailing whitespace
      3. Normalize all apostrophe variants to standard ASCII apostrophe
      4. Replace apostrophe with a space  (e.g. "DELL'AMBIENTE" → "DELL AMBIENTE")
         This avoids SQL quoting issues while preserving searchability for embeddings.
      5. Collapse multiple spaces into one
    """
    if not isinstance(value, str):
        return value

    value = value.strip()
    value = value.translate(_APOSTROPHE_VARIANTS)
    value = value.replace("'", " ")
    value = re.sub(r" {2,}", " ", value)  # collapse multiple spaces
    return value

# ---------------------------------------------------------------------------
# Logging wrapper for column retrievers
# ---------------------------------------------------------------------------

class LoggingRetriever:
    """
    Wraps a LlamaIndex retriever and logs the nodes it returns,
    showing what DB values were matched for the user query.
    """

    def __init__(self, inner_retriever, column_label: str):
        self._inner = inner_retriever
        self._label = column_label

    def retrieve(self, query: str):
        nodes = self._inner.retrieve(query)
        if nodes:
            print(f"\n  [COLUMN RETRIEVER — {self._label}]")
            print(f"  Query : '{query}'")
            print(f"  Matches ({len(nodes)}):")
            for i, n in enumerate(nodes, 1):
                score = f"{n.score:.4f}" if n.score is not None else "n/a"
                print(f"    {i}. '{n.node.get_content()}' (score: {score})")
        else:
            print(f"\n  [COLUMN RETRIEVER — {self._label}] No matches for '{query}'")
        return nodes

    def __getattr__(self, name):
        return getattr(self._inner, name)


# ---------------------------------------------------------------------------
# Load a persisted ChromaDB collection as a LlamaIndex retriever
# ---------------------------------------------------------------------------

def load_column_retriever(
    collection_name: str,
    chroma_client: chromadb.PersistentClient,
    top_k: int = 3,
    label: str = "",
    similarity_cutoff: float | None = None,  # NEW
) -> LoggingRetriever:
    try:
        chroma_collection = chroma_client.get_collection(collection_name)
    except Exception as exc:
        raise ValueError(
            f"Collection '{collection_name}' not found. Run 03_build_index.py first."
        ) from exc

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store)

    retriever_kwargs = {"similarity_top_k": top_k}
    if similarity_cutoff is not None:
        retriever_kwargs["vector_store_query_mode"] = "default"
    
    retriever = index.as_retriever(**retriever_kwargs)

    # Apply cutoff post-hoc if specified (most reliable with ChromaDB)
    if similarity_cutoff is not None:
        from llama_index.core.postprocessor import SimilarityPostprocessor
        retriever._node_postprocessors = [
            SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)
        ]

    return LoggingRetriever(retriever, label or collection_name)

# ---------------------------------------------------------------------------
# Table router — selects relevant tables for a given query
# ---------------------------------------------------------------------------

def build_table_router(embed_model) -> VectorStoreIndex:
    """
    Builds an in-memory VectorStoreIndex over TABLE_DOMAINS descriptions.
    Used at query time to select which tables are relevant.
    Each node carries the table name in its metadata.
    """
    nodes = [
        TextNode(text=description, metadata={"table": table_name})
        for table_name, description in TABLE_DOMAINS.items()
    ]
    return VectorStoreIndex(nodes, embed_model=embed_model)


def route_tables(query: str, table_router_index: VectorStoreIndex) -> list[str]:
    """
    Returns the list of table names most relevant to the query.
    Prints routing decisions for transparency.
    """
    retriever = table_router_index.as_retriever(similarity_top_k=TABLE_ROUTER_TOP_K)
    matched = retriever.retrieve(query)
    selected = [n.metadata["table"] for n in matched]

    print(f"\n  [TABLE ROUTER] Query: '{query}'")
    for n in matched:
        score = f"{n.score:.4f}" if n.score is not None else "n/a"
        print(f"    → {n.metadata['table']} (score: {score})")

    return selected



# ---------------------------------------------------------------------------
# Global prompt — rules for SQL generation applied to all tables
# ---------------------------------------------------------------------------

TEXT_TO_SQL_PROMPT = PromptTemplate(
    "You are a SQL generation assistant for the University of Trieste (UniTS) database.\n"
    "Given the database schema and the user's question, generate a correct SQLite query.\n"
    "\n"
    "## Column hints\n"
    "- ALWAYS use the exact values suggested by the column hints to build WHERE conditions.\n"
    "  Never rephrase the user's original text. If the hint suggests 'TREVISAN MARTINO',\n"
    "  use LIKE '%TREVISAN MARTINO%', even if the user wrote 'Martino Trevisan'.\n"
    "- Use only column names present in the provided schema.\n"
    "\n"
    "## Table routing\n"
    "- Lesson times, dates, schedules → table 'lezione'\n"
    "- Exams, room bookings, or any non-lesson event → table 'evento_aula'\n"
    "- Who teaches a course, Teams code → table 'insegnamento'\n"
    "- Person info (email, role, department) → table 'personale'\n"
    "- Degree program info (not single subject) → table 'corso_di_laurea'\n"
    "- Room static info (capacity, equipment, floor) → table 'info_aula'\n"
    "  If not found in 'evento_aula', also try 'lezione'.\n"
    "\n"
    "## SQL rules\n"
    "- Always use UPPER() with LIKE when comparing text columns.\n"
    "- Dates are stored in ISO format YYYY-MM-DD.\n"
    "- Never filter by academic year or semester period unless explicitly requested.\n"
    "- Never use SQL JOINs — tables are not normalized and have no reliable foreign keys.\n"
    "- Default LIMIT is 40 rows unless the user specifies otherwise.\n"
    "- For week or date range queries, SELECT only essential columns:\n"
    "  date, start_time, end_time, subject_name, room_name, site_name\n"
    "- For relative dates (today, yesterday, tomorrow), compute the explicit ISO date\n"
    "  and use it directly in the WHERE clause.\n"
    "- When searching by professor and the column hints return multiple distinct names\n"
    "  (e.g. 'DE LORENZO ANDREA' and 'DE LORENZO GIUDITTA'), use a separate LIKE per name\n"
    "  combined with OR. Never use a generic LIKE on surname alone.\n"
    "- Always include the 'professors' column in SELECT when querying 'insegnamento',\n"
    "  so that results from different professors can be distinguished in the final answer.\n"
    "- When the user mentions a department to identify a professor, treat it as\n"
    "  disambiguation context only — do NOT add filters on degree_program_name.\n"
    "- For exam queries in 'evento_aula', always filter by subject name using\n"
    "  name_event (e.g. UPPER(name_event) LIKE '%DATABASE%'), never by degree \n"
    "  program name since that column does not exist in this table.\n"
    "  Always filter by event_type: UPPER(event_type) LIKE '%ESAME%'.\n"
    "  If the user asks for exams of a degree program instead of a specific \n"
    "  subject, explain that the exam calendar can only be searched by subject\n"
    "  name, not by degree program.\n"
    "\n"
    "## Output\n"
    "- Return only the SQL query, no comments, no explanation, no markdown.\n"
    "\n"
    "Available schema:\n"
    "{schema}\n"
    "\n"
    "Question: {query_str}\n"
    "\n"
    "SQL:"
)

# ---------------------------------------------------------------------------
# Routed query engine — wraps SQLTableRetrieverQueryEngine with dynamic routing
# ---------------------------------------------------------------------------

class RoutedSQLQueryEngine:
    """
    At each query:
      1. Routes the query to the 1-2 most relevant tables (table router)
      2. Passes only those tables' column retrievers to a fresh
         SQLTableRetrieverQueryEngine instance
      3. Delegates query execution to that engine

    Re-creating SQLTableRetrieverQueryEngine per query is cheap —
    it is a plain Python object with no re-embedding cost.
    """

    def __init__(self, sql_database, obj_index, all_cols_retrievers, table_router_index):
        self._sql_database = sql_database
        self._obj_index = obj_index
        self._all_cols_retrievers = all_cols_retrievers
        self._table_router_index = table_router_index


    def query(self, query_str: str):
        t_total = time.time()

        # Stage 1: table routing
        t0 = time.time()
        relevant_tables = route_tables(query_str, self._table_router_index)
        t_routing = time.time() - t0

        # Stage 2: build routed engine
        routed_cols = {
            table: (
                self._all_cols_retrievers[table] if table in relevant_tables else {}
            )
            for table in self._all_cols_retrievers
        }

        engine = SQLTableRetrieverQueryEngine(
            self._sql_database,
            self._obj_index.as_retriever(similarity_top_k=5),
            cols_retrievers=routed_cols,
            llm=Settings.llm,
            text_to_sql_prompt=TEXT_TO_SQL_PROMPT,
        )

        # Stage 3: full pipeline (column retrieval + SQL gen + DB exec + answer synthesis)
        t0 = time.time()
        response = engine.query(query_str)
        t_pipeline = time.time() - t0

        timings = {
            "table_routing": t_routing,
            "pipeline":      t_pipeline,
            "total":         time.time() - t_total,
        }

        print(f"\n  [TIMING SUMMARY]")
        print(f"    Table routing : {t_routing:.3f}s")
        print(f"    Pipeline      : {t_pipeline:.3f}s")
        print(f"    TOTAL         : {timings['total']:.3f}s")

        return response, timings
# ---------------------------------------------------------------------------
# Table router — semantic description used to pick relevant tables per query
# ---------------------------------------------------------------------------

# Each entry describes the domain of the table in natural language.
# The table router embeds these descriptions and finds the closest ones
# to the user query, so only those tables' column retrievers are used.
TABLE_DOMAINS = {
    "personale": (
        "staff person professor docente employee contact "
        "personale nome email phone telefono ruolo dipartimento"
    ),
    "insegnamento": (
        "course insegnamento subject materia degree laurea professor docente "
        "teams code codice periodo semestre academic-year anno accademico"
    ),
    "corso_di_laurea": (
        "name url department type duration location language "
        "corsi di laurea triennale, magistrale e a ciclo unico con relativo dipartimento, durata, lingua di erogazione, tipo di corso (triennale, magistrale, a ciclo unico)"
    ),
    "lezione": (
        "lesson lecture lezione schedule orario timetable "
        "quando date data ora start end aula edificio cancelled annullata "
        "subject materia professor docente degree"
    ),
    "evento_aula": (
        "event evento booking prenotazione aula room occupancy occupazione "
        "calendar calendario edificio building schedule orario tipo di evento esame exam"
    ),
    "info_aula": (
        "classroom aula room info details dettagli building edificio "
        "floor piano capacity capienza equipment attrezzature wifi "
        "proiettore accessible accessibile maps mappa indirizzo address"
    ),
}

# How many tables to select per query (increase to 3 for cross-table queries)
TABLE_ROUTER_TOP_K = 3



# ---------------------------------------------------------------------------
# Build the full query engine
# ---------------------------------------------------------------------------

def build_query_engine(db_path: Path, chroma_dir: Path) -> RoutedSQLQueryEngine:
    engine = create_engine(f"sqlite:///{db_path}")
    sql_database = SQLDatabase(
        engine,
        include_tables=["personale", "insegnamento", "lezione", "evento_aula", "info_aula"],
    )

    # --- Table schema index (used by SQLTableRetrieverQueryEngine internally) ---
    table_node_mapping = SQLTableNodeMapping(sql_database)
    table_schema_objs = [
        SQLTableSchema(
            table_name="personale",
            context_str=(
                "Contains university staff and professors. "
                "Key columns: nome (full name), role, department, email, phone."
            ),
        ),
        SQLTableSchema(
            table_name="insegnamento",
            context_str=(
                "Contains university courses (subjects) and their academic details. "
                "Use for questions about: which professor teaches a course, "
                "Teams code of a course, degree program, academic year, semester period. "
                "Key columns: "
                "subject_name (course name), "
                "professors (professor names, can be more then one) and the relative main_professor_id, "
                "degree_program_name and "
                "degree_program_name_eng (official degree program name in English), "
                "degree_program_code (e.g. 'IN22'), "
                "academic_year (e.g. '2025/2026' but take into account the user can write for example 2025-2026), "
                "period (semester: 'S1' = first, 'S2' = second), "
                "teams_code (Microsoft Teams code), "
                "study_code (course code, e.g. '472MI-1'). "
                "last_update of these information."
            ),
        ),
        SQLTableSchema(
            table_name="corso_di_laurea",
            context_str=(
                "Contains university courses (subjects) and their academic details. "
                "Use for questions about: which professor teaches a course, "
                "Teams code of a course, degree program, academic year, semester period. "
                "Key columns: "
                "name (course name), "
                "url (link of the webpage course), "
                "department, "
                "type (if it's bachelor or master or 5 years course of a 6 years course) "
                "duration (years), "
                "location (the site), "
                "language (language of the lessons), "
            ),
        ),
        SQLTableSchema(
            table_name="lezione",
            context_str=(
                "Contains scheduled lessons and academic calendar events. "
                "Use for questions about: lesson times, start hour, end hour, "
                "lesson date, classroom, building, which lessons happen on a specific day, "
                "whether a lesson is cancelled. "
                "Key columns: "
                "subject_name (course name), "
                "date (iso format), "
                "start_time (lesson start, e.g. '10:00'), "
                "end_time (lesson end, e.g. '13:00'), "
                "department, "
                "room_name (classroom like Aula 301) and site_name (building like Edificio Gorizia), "
                "room_code and site_code are the relative IDs of the room and site/building, "
                "professors (professor names, may contain multiple separated by comma), "
                "degree_program_name (degree program name), "
                "degree_program_code (e.g. 'IN22'), "
                "cancelled ('yes' if cancelled, 'no' otherwise)."
            ),
        ),
        SQLTableSchema(
            table_name="evento_aula",
            context_str=(
                "Contains classroom booking events and room occupancy schedule. "
                "Use for questions about: which events or activities are scheduled in a room, "
                "room availability, event times, who is involved in a room event. "
                "Key columns: "
                "room_name (classroom like Aula 301) and site_name (building like Edificio Gorizia), "
                "room_code and site_code are the relative IDs of the room and site/building, "
                "name_event (event or activity name), "
                "date (iso format), "
                "start_time (event start, e.g. '10:00'), "
                "end_time (event end, e.g. '13:00'), "
                "professors (professors or persons involved). "
                "cancelled (yes or not) "
                "event_type (usually lesson or exam, or other type of activity)"
            ),
        ),
        SQLTableSchema(
            table_name="info_aula",
            context_str=(
                "Contains static info about a classroom. "
                "Use for questions about: where is the room, how big it is, "
                "is there wifi in that room. "
                "Key columns: "
                "room_name (classroom like Aula 301) and site_name (building like Edificio Gorizia), "
                "room_code and site_code are the relative IDs of the room and site/building, "
                "floor of the building, "
                "room_type (if it's considered small, big, if it's aula magna...), "
                "capacity (how many students can stay there), "
                "accessible (it's not about disability. If no maybe there are works in progress, specify only if the value is no), "
                "maps_url (you can see on google maps the position of the building where the room is), "
                "equipment (if there is wifi, Proiettore, lavagne...), "
                "url (url with more info about the room)."
            ),
        ),
    ]

    obj_index = ObjectIndex.from_objects(
        table_schema_objs,
        table_node_mapping,
        VectorStoreIndex,
    )

    # --- Column retrievers loaded from ChromaDB on disk ---
    chroma_client = chromadb.PersistentClient(path=str(chroma_dir))

    all_cols_retrievers = {
        "personale": {
            "nome_and_surname": load_column_retriever("personale__nome_and_surname", chroma_client, top_k=5),
            "role":             load_column_retriever("personale__role",             chroma_client, top_k=5),
            "department":       load_column_retriever("personale__department",       chroma_client, top_k=5),
        },
        "insegnamento": {
            "degree_program_name":     load_column_retriever("insegnamento__degree_program_name",     chroma_client, top_k=5),
            "degree_program_name_eng": load_column_retriever("insegnamento__degree_program_name_eng", chroma_client, top_k=5),
            "subject_name":            load_column_retriever("insegnamento__subject_name",            chroma_client, top_k=5, similarity_cutoff=0.5),
            "professors":              load_column_retriever("insegnamento__professors",              chroma_client, top_k=5),
            "period":                  load_column_retriever("insegnamento__period",                  chroma_client, top_k=1),
        },
        "corso_di_laurea": {
            "name":             load_column_retriever("corso_di_laurea__name",                chroma_client, top_k=5),
            "department":       load_column_retriever("corso_di_laurea__department",          chroma_client, top_k=5),
            "type":             load_column_retriever("corso_di_laurea__type",                chroma_client, top_k=5),
            "duration":         load_column_retriever("corso_di_laurea__duration",            chroma_client, top_k=5),
            "language":         load_column_retriever("corso_di_laurea__language",            chroma_client, top_k=5),
        },
        "lezione": {
            "degree_program_name": load_column_retriever("lezione__degree_program_name", chroma_client, top_k=5),
            "subject_name":        load_column_retriever("lezione__subject_name",        chroma_client, top_k=5, similarity_cutoff=0.5),
            "study_year_code":     load_column_retriever("lezione__study_year_code",     chroma_client, top_k=5),
            "curriculum":          load_column_retriever("lezione__curriculum",          chroma_client, top_k=5),
            "date":                load_column_retriever("lezione__date",                chroma_client, top_k=1),
            "department":          load_column_retriever("lezione__department",          chroma_client, top_k=5),
            "room_name":           load_column_retriever("lezione__room_name",           chroma_client, top_k=5),
            "site_name":           load_column_retriever("lezione__site_name",           chroma_client, top_k=5),
            "address":             load_column_retriever("lezione__address",             chroma_client, top_k=5),
            "professors":          load_column_retriever("lezione__professors",          chroma_client, top_k=5),
        },
        "evento_aula": {
            "site_name":  load_column_retriever("evento_aula__site_name",  chroma_client, top_k=5),
            "room_name":  load_column_retriever("evento_aula__room_name",  chroma_client, top_k=5),
            "name_event": load_column_retriever("evento_aula__name_event", chroma_client, top_k=5),
            "professors": load_column_retriever("evento_aula__professors", chroma_client, top_k=5),
            "event_type": load_column_retriever("evento_aula__event_type", chroma_client, top_k=5),

        },
        "info_aula": {
            "site_name": load_column_retriever("info_aula__site_name", chroma_client, top_k=5),
            "room_name": load_column_retriever("info_aula__room_name", chroma_client, top_k=5),
            "address":   load_column_retriever("info_aula__address",   chroma_client, top_k=5),
            "floor":     load_column_retriever("info_aula__floor",     chroma_client, top_k=5),
            "room_type": load_column_retriever("info_aula__room_type", chroma_client, top_k=5),
        },
    }

    # --- Table router (in-memory, built from TABLE_DOMAINS descriptions) ---
    table_router_index = build_table_router(embed_model=Settings.embed_model)

    return RoutedSQLQueryEngine(
        sql_database=sql_database,
        obj_index=obj_index,
        all_cols_retrievers=all_cols_retrievers,
        table_router_index=table_router_index,
    )