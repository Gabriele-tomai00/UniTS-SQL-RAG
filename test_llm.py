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
from utils import *

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
    "Quali insegnamenti sono tenuti dal prof De Lorenzo?",
    "Quali insegnamenti sono tenuti dal prof De Lorenzo del dipartimento di ingegneria?",
    "Quali insegnamenti sono tenuti dalla prof De Lorenzo?",
    "Quali insegnamenti sono tenuti dai professori con cognome De Lorenzo?",
    "Che lezioni ci sono venerdì 13 marzo, per gli studenti di Computer Engineering curriculum Informatics?",
    "Che lezioni c'erano ieri, per gli studenti di Computer Engineering curriculum Informatics?",
    "Ci sono lezioni di geometria di giovedì?",
    "Dimmi tutti i corsi di laurea (nome e tipo) del dipartimento di ingegneria e architettura, ",
    "Dimmi tutti gli esami in calendario del corso di Computer Engineering che si sono tenuti nel mese di febbraio 2026"
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
# Run test suite
# ---------------------------------------------------------------------------

def run_tests(query_engine, output_md: Path) -> None:
    total = len(TEST_QUESTIONS)
    md_results = []

    for idx, question in enumerate(TEST_QUESTIONS, 1):
        log_section(f"TEST {idx}/{total}: {question}")

        start = time.time()
        try:
            response, timings = query_engine.query(question)

            sql = None
            if hasattr(response, "metadata") and response.metadata:
                sql = response.metadata.get("sql_query")

            log_sub("Generated SQL")
            print(f"  {sql}" if sql else "  (no SQL metadata available)")
            log_sub("Answer")
            print(f"  {response}")

            md_results.append({
                "question": question,
                "answer": str(response),
                "sql": sql,
                "timings": timings,
                "error": None,
            })


        except Exception as exc:
            elapsed = time.time() - start
            log_sub("ERROR")
            print(f"  {type(exc).__name__}: {exc}")
            md_results.append({
                "question": question,
                "answer": None,
                "timings": {"total": elapsed},
                "error": f"{type(exc).__name__}: {exc}",
            })
    
    write_markdown(output_md, md_results, total)
    print(f"\n  📄 Results saved to: {output_md}")


# ---------------------------------------------------------------------------
# Write markdown report (questions + answers only)
# ---------------------------------------------------------------------------

def write_markdown(output_path: Path, results: list, total: int) -> None:
    lines = []
    lines.append("# Test Results — University NL Query Engine")
    lines.append(f"\n<sub>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</sub>")  # MODIFIED
    lines.append("\n---\n")

    for idx, r in enumerate(results, 1):
        lines.append(f"## {idx}. {r['question']}\n")
        if r["error"]:
            lines.append(f"⚠️ **Error:** `{r['error']}`\n")
        else:
            if r.get("sql"):
                lines.append(f"```sql\n{r['sql']}\n```\n")
            lines.append(f"**Risposta:** {r['answer']}\n")

        t = r.get("timings", {})
        lines.append(
            f"<sub>"
            f"Table routing: **{t.get('table_routing', 0):.2f}s** | "
            f"Pipeline (retrieval+SQL+DB+answer): **{t.get('pipeline', 0):.2f}s** | "
            f"Total: **{t.get('total', 0):.2f}s**"
            f"</sub>\n"
        )
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