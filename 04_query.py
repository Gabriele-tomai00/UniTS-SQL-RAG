"""
04_query.py
-----------
Natural Language → SQL → Answer pipeline for the university database.

Uses:
  - LlamaIndex SQLTableRetrieverQueryEngine
  - ChromaDB (persistent) for column-level fuzzy value matching

Flow:
  1. User query arrives in natural language
  2. Column retrievers scan Chroma indexes to find the best-matching
     DB values (e.g. "ingegneria informatica" → "Ingegneria Elettronica e Informatica")
  3. Those values are injected as context hints for the LLM
  4. LLM generates the correct SQL query
  5. SQL is executed against the SQLite DB
  6. LLM synthesizes the final natural language answer

Usage:
    python 04_query.py
    python 04_query.py --query "A che ora è la lezione di Robotics?"
"""

import argparse
from pathlib import Path

import chromadb
from llama_index.core import Settings, SQLDatabase, StorageContext, VectorStoreIndex
from llama_index.core.indices.struct_store.sql_query import SQLTableRetrieverQueryEngine
from llama_index.core.objects import ObjectIndex, SQLTableNodeMapping, SQLTableSchema
from llama_index.core.prompts import PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from sqlalchemy import create_engine

from index_utils import *

DEFAULT_DB = "2025-2026_data/university.db"
DEFAULT_CHROMA_DIR = "2025-2026_data/chroma_store"

# ---------------------------------------------------------------------------
# Global prompt — rules for SQL generation applied to all tables
# ---------------------------------------------------------------------------

TEXT_TO_SQL_PROMPT = PromptTemplate(
    "You are a university assistant for the University of Trieste (UniTS).\n"
    "Given the database schema and the user's question, generate a correct SQLite SQL query.\n"
    "\n"
    "Rules:\n"
    "- Use only column names that appear in the provided schema."
    " - When the question does not specify a limit, return at most 40 rows using LIMIT."
    "- Always use UPPER() or LIKE when comparing text columns\n"
    "- Dates in the database are stored in ISO format YYYY-MM-DD, "
    "- Never filter by academic_year or period unless explicitly requested\n"
    "- If the question concerns lesson schedules or lesson dates, use the 'lezione' table\n"
    "- If the question concerns a classroom, search in the 'evento_aula' table (this is the classroom occupancy calendar) "
    "and if nothing is found try searching in the 'lezione' table\n"
    "- If the question concerns who teaches a course or the Teams code, use the 'insegnamento' table\n"
    "- If the question concerns information about a person (email, role, department), use the 'personale' table\n"
    "- Do not perform SQL joins because the columns are often not normalized (except for ISO date format and times). "
    "Instead search in different tables, first in one and then in another "
    "(for example find an event in 'evento_aula' and if the user wants more information, search by date, time and classroom "
    "(note that the name might be slightly different) in 'lezione')\n"
    "- For queries over an entire week or a date range, use SELECT with only the essential columns: "
    "date, start_time, end_time, subject_name, room_name, building_name, url. "
    "Do not use SELECT * for queries that could return many rows.\n"
    "- There are no relationships between tables due to non-normalized data\n"
    "- Always respond in Italian in the final synthesis phase\n"
    "- If the question refers to a relative date (for example today, yesterday, tomorrow...), "
    "when answering also include the explicit date in the appropriate format\n"
    "\n"
    "Available schema:\n"
    "{schema}\n"
    "\n"
    "Question: {query_str}\n"
    "\n"
    "SQL (only the query, no comments):"
)

# ---------------------------------------------------------------------------
# Load a persisted ChromaDB collection as a LlamaIndex retriever
# ---------------------------------------------------------------------------

def load_column_retriever(
    collection_name: str,
    chroma_client: chromadb.PersistentClient,
    similarity_top_k: int = 3,
):
    """
    Loads an existing Chroma collection and returns a retriever.
    Raises ValueError if the collection does not exist (run 03_build_index.py first).
    """
    try:
        chroma_collection = chroma_client.get_collection(collection_name)
    except Exception as exc:
        raise ValueError(
            f"Collection '{collection_name}' not found in ChromaDB. "
            "Run 03_build_index.py first."
        ) from exc

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store)
    return index.as_retriever(similarity_top_k=similarity_top_k)


# ---------------------------------------------------------------------------
# Build the full query engine
# ---------------------------------------------------------------------------

def build_query_engine(db_path: Path, chroma_dir: Path):
    engine = create_engine(f"sqlite:///{db_path}")
    sql_database = SQLDatabase(
        engine,
        include_tables=["personale", "insegnamento", "lezione", "evento_aula"],
    )

    # --- Table schema index (in-memory) ---
    # context_str is critical: the table retriever uses it to decide
    # which table to query. Words here must match the user's natural language.
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
                "professors (professor names, can be more then one) and the relative main_professor_id"
                "degree_program_name and"
                "degree_program_name_eng (official degree program name in English), "
                "degree_program_code (e.g. 'IN22'), "
                "academic_year (e.g. '2025/2026' but take into accout the user can write for exemple 2025-2026), "
                "period (semester: 'S1' = first, 'S2' = second), "
                "teams_code (Microsoft Teams code), "
                "study_code (course code, e.g. '472MI-1')."
                "last_update of these information"
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
                "room_name (classroom like Aula 301) and  site_name (building like Edificio Gorizia), "
                "room_code and site_code are the relatives ID of the room and site/building"
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

                "room_name (classroom like Aula 301) and  site_name (building like Edificio Gorizia), "
                "room_code and site_code are the relatives ID of the room and site/building"
                "room_code and site_code are the relatives ID of the room and site/building"

                "name_event (event or activity name), "
                "date (iso format), "
                "start_time (event start, e.g. '10:00'), "
                "end_time (event end, e.g. '13:00'), "
                "professors (professors or persons involved), "
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

    cols_retrievers = {
            "personale": {
                "nome_and_surname": load_column_retriever(
                    "personale__nome_and_surname", chroma_client, similarity_top_k=5
                ),
                "role": load_column_retriever(
                    "personale__role", chroma_client, similarity_top_k=5
                ),
                "department": load_column_retriever(
                    "personale__department", chroma_client, similarity_top_k=5
                ),
            },

            "insegnamento": {
                "degree_program_name": load_column_retriever(
                    "insegnamento__degree_program_name", chroma_client, similarity_top_k=5
                ),
                "degree_program_name_eng": load_column_retriever(
                    "insegnamento__degree_program_name_eng", chroma_client, similarity_top_k=5
                ),
                "subject_name": load_column_retriever(
                    "insegnamento__subject_name", chroma_client, similarity_top_k=5
                ),
                "professors": load_column_retriever(
                    "insegnamento__professors", chroma_client, similarity_top_k=5
                ),
                "period": load_column_retriever(
                    "insegnamento__period", chroma_client, similarity_top_k=1
                ),
            },

            "lezione": {
                "degree_program_name": load_column_retriever(
                    "lezione__degree_program_name", chroma_client, similarity_top_k=5
                ),
                "subject_name": load_column_retriever(
                    "lezione__subject_name", chroma_client, similarity_top_k=5
                ),
                "study_year_code": load_column_retriever(
                    "lezione__study_year_code", chroma_client, similarity_top_k=5
                ),
                "curriculum": load_column_retriever(
                    "lezione__curriculum", chroma_client, similarity_top_k=5
                ),
                "date": load_column_retriever(
                    "lezione__date", chroma_client, similarity_top_k=1
                ),
                "department": load_column_retriever(
                    "lezione__department", chroma_client, similarity_top_k=5
                ),
                "room_name": load_column_retriever(
                    "lezione__room_name", chroma_client, similarity_top_k=5
                ),
                "site_name": load_column_retriever(
                    "lezione__site_name", chroma_client, similarity_top_k=5
                ),
                "address": load_column_retriever(
                    "lezione__address", chroma_client, similarity_top_k=5
                ),
                "professors": load_column_retriever(
                    "lezione__professors", chroma_client, similarity_top_k=5
                ),
            },
            "evento_aula": {
                "site_name": load_column_retriever(
                    "evento_aula__site_name", chroma_client, similarity_top_k=5
                ),
                "room_name": load_column_retriever(
                    "evento_aula__room_name", chroma_client, similarity_top_k=5
                ),
                "name_event": load_column_retriever(
                    "evento_aula__name_event", chroma_client, similarity_top_k=5
                ),
                "professors": load_column_retriever(
                    "evento_aula__professors", chroma_client, similarity_top_k=5
                ),
            },
    }

    query_engine = SQLTableRetrieverQueryEngine(
        sql_database,
        obj_index.as_retriever(similarity_top_k=5),
        cols_retrievers=cols_retrievers,
        llm=Settings.llm,                          # always use the globally configured LLM
        text_to_sql_prompt=TEXT_TO_SQL_PROMPT,     # custom prompt with UniTS-specific rules
    )

    return query_engine


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------

def interactive_loop(query_engine) -> None:
    print("\nUniversity Query System — type 'exit' to quit\n")
    while True:
        user_input = input("Query> ").strip()
        if user_input.lower() in ("exit", "quit", "q"):
            break
        if not user_input:
            continue
        try:
            response = query_engine.query(user_input)
            print(f"\nAnswer: {response}\n")
            # Show the generated SQL for transparency
            if hasattr(response, "metadata") and response.metadata:
                sql = response.metadata.get("sql_query")
                if sql:
                    print(f"[SQL] {sql}\n")
        except Exception as exc:
            print(f"[ERROR] {exc}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="University NL query engine")
    parser.add_argument("--query", default=None, help="Single query (non-interactive)")
    args = parser.parse_args()

    qe = build_query_engine(Path(DEFAULT_DB), Path(DEFAULT_CHROMA_DIR))

    if args.query:
        response = qe.query(args.query)
        print(f"\nAnswer: {response}")
        if hasattr(response, "metadata") and response.metadata:
            sql = response.metadata.get("sql_query")
            if sql:
                print(f"[SQL] {sql}")
    else:
        interactive_loop(qe)