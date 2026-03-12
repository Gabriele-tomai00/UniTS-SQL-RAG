"""
test_llm.py
-----------
Runs a pool of test queries against the university NL-to-SQL engine
and prints detailed logs: retrieved chunks, generated SQL, and final answer.
Results (questions + answers only) are also written to a markdown file.

Usage:
    python test_llm.py
    python test_llm.py --db university.db --chroma-dir ./chroma_store
    python test_llm.py --output results.md
"""

import argparse
import time
from pathlib import Path
from datetime import datetime

import chromadb
from llama_index.core import Settings, SQLDatabase, StorageContext, VectorStoreIndex
from llama_index.core.indices.struct_store.sql_query import SQLTableRetrieverQueryEngine
from llama_index.core.objects import ObjectIndex, SQLTableNodeMapping, SQLTableSchema
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.chroma import ChromaVectorStore
from sqlalchemy import create_engine
from index_utils import *

DEFAULT_DB = "2025-2026_data/university.db"
DEFAULT_CHROMA_DIR = "2025-2026_data/chroma_store"
DEFAULT_OUTPUT_MD = "2025-2026_data/results.md"

# ---------------------------------------------------------------------------
# Test question pool
# ---------------------------------------------------------------------------

TEST_QUESTIONS = [
    "Chi è Martino Trevisan?",
    "Chi è Trevisan Martino?",
    "Chi è Trevisan?",
    "Chi è il prof Trevisan di Ingegneria Informatica?",
    "Quale insegnamento tiene il prof Paolo Vercesi?",
    "Quali insegnamenti sono tenuti da Vercesi Paolo",
    "Quali insegnamenti sono tenuti dal prof De Lorenzo?"
]

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

SEPARATOR = "=" * 70
SUBSEP    = "-" * 50


def log_section(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def log_sub(title: str) -> None:
    print(f"\n{SUBSEP}")
    print(f"  {title}")
    print(SUBSEP)


# ---------------------------------------------------------------------------
# Retriever wrapper — intercepts retrieve() to log returned chunks
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

    # Proxy all other attribute accesses to the inner retriever
    def __getattr__(self, name):
        return getattr(self._inner, name)


# ---------------------------------------------------------------------------
# Load ChromaDB collection as a retriever (with logging wrapper)
# ---------------------------------------------------------------------------

def load_column_retriever(
    collection_name: str,
    chroma_client: chromadb.PersistentClient,
    similarity_top_k: int = 3,
    label: str = "",
) -> LoggingRetriever:
    try:
        chroma_collection = chroma_client.get_collection(collection_name)
    except Exception as exc:
        raise ValueError(
            f"Collection '{collection_name}' not found. Run 03_build_index.py first."
        ) from exc

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store)
    retriever = index.as_retriever(similarity_top_k=similarity_top_k)
    return LoggingRetriever(retriever, label or collection_name)


# ---------------------------------------------------------------------------
# Build query engine
# ---------------------------------------------------------------------------

def build_query_engine(db_path: Path, chroma_dir: Path):
    engine = create_engine(f"sqlite:///{db_path}")
    sql_database = SQLDatabase(
        engine,
        include_tables=["personale", "insegnamento", "lezione", "evento_aula"],
    )

    table_node_mapping = SQLTableNodeMapping(sql_database)
    table_schema_objs = [
        SQLTableSchema(
            table_name="insegnamento",
            context_str=(
                "Contains university courses. "
                "Key columns: "
                "course_name (course name, stored UPPERCASE — use UPPER() or LIKE for comparisons), "
                "professors (professor names, may contain multiple separated by '#' or other symbol and may be contained an ID), "
                "degree_program (degree program name, UPPERCASE), "
                "academic_year (format 'YYYY/YYYY', e.g. '2025/2026'), "
                "period (semester code: 'S1' = first semester, 'S2' = second semester), "
                "teams_code (Microsoft Teams code for the course), "
                "course_code."
            ),
        ),
        SQLTableSchema(
            table_name="personale",
            context_str=(
                "Contains university staff and professors. "
                "Key columns: nome (full name), role, department, email, phone."
            ),
        ),
    ]

    obj_index = ObjectIndex.from_objects(
        table_schema_objs,
        table_node_mapping,
        VectorStoreIndex,
    )

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
    )

    return query_engine


# ---------------------------------------------------------------------------
# Run test suite
# ---------------------------------------------------------------------------

def run_tests(query_engine, output_md: Path) -> None:
    total = len(TEST_QUESTIONS)
    errors = 0

    # Collect results for markdown
    md_results = []

    for idx, question in enumerate(TEST_QUESTIONS, 1):
        log_section(f"TEST {idx}/{total}: {question}")

        start = time.time()
        try:
            response = query_engine.query(question)
            elapsed = time.time() - start

            # Generated SQL
            sql = None
            if hasattr(response, "metadata") and response.metadata:
                sql = response.metadata.get("sql_query")

            log_sub("Generated SQL")
            if sql:
                print(f"  {sql}")
            else:
                print("  (no SQL metadata available)")

            log_sub("Answer")
            print(f"  {response}")

            print(f"\n  ⏱  Elapsed: {elapsed:.2f}s")

            md_results.append({
                "question": question,
                "answer": str(response),
                "sql": sql,
                "elapsed": elapsed,
                "error": None,
            })

        except Exception as exc:
            elapsed = time.time() - start
            errors += 1
            log_sub("ERROR")
            print(f"  {type(exc).__name__}: {exc}")
            print(f"\n  ⏱  Elapsed: {elapsed:.2f}s")

            md_results.append({
                "question": question,
                "answer": None,
                "elapsed": elapsed,
                "error": f"{type(exc).__name__}: {exc}",
            })

    print(f"\n{SEPARATOR}")
    print(f"  Results: {total - errors}/{total} passed  |  {errors} errors")
    print(SEPARATOR)

    # Write markdown file
    write_markdown(output_md, md_results, total, errors)
    print(f"\n  📄 Results saved to: {output_md}")


# ---------------------------------------------------------------------------
# Write markdown report (questions + answers only)
# ---------------------------------------------------------------------------

def write_markdown(output_path: Path, results: list, total: int, errors: int) -> None:
    lines = []
    lines.append("# Test Results — University NL Query Engine")
    lines.append(f"\n**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    lines.append(f"**Passed:** {total - errors}/{total}  |  **Errors:** {errors}\n")
    lines.append("---\n")

    for idx, r in enumerate(results, 1):
        lines.append(f"## {idx}. {r['question']}\n")
        if r["error"]:
            lines.append(f"⚠️ **Error:** `{r['error']}`\n")
        else:
            if r.get("sql"):
                lines.append(f"```sql\n{r['sql']}\n```\n")
            lines.append(f"**Risposta:** {r['answer']}\n")
        lines.append(f"*Tempo: {r['elapsed']:.2f}s*\n")
        lines.append("---\n")

    output_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test suite for university NL query engine")
    parser.add_argument("--db", default=DEFAULT_DB, help="SQLite file path")
    parser.add_argument("--log", default=DEFAULT_DB, help="print additional information about retrieved chunks and generated SQL")
    parser.add_argument(
        "--chroma-dir", default=DEFAULT_CHROMA_DIR, help="ChromaDB persistence directory"
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT_MD, help="Markdown output file path (default: results.md)"
    )
    args = parser.parse_args()

    print("\nLoading query engine...")
    qe = build_query_engine(Path(args.db), Path(args.chroma_dir))
    print("Query engine ready.\n")

    run_tests(qe, Path(args.output))