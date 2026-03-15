"""
03_build_index.py
-----------------
Builds and persists ChromaDB embedding indexes for columns used as
search filters in natural language queries.

Indexed columns:
  - insegnamento.course_name       (e.g. "sw dev" → "SOFTWARE DEVELOPMENT METHODS")
  - insegnamento.professors    (e.g. "Paolo Vercesi" → "VERCESI PAOLO (014095)")
  - insegnamento.degree_program_name    (e.g. "informatica" → "COMPUTER ENGINEERING")
  - insegnamento.academic_year     (e.g. "2025-2026" → "2025/2026")
  - insegnamento.period            (e.g. "primo semestre" → "S1")
  - personale.nome                 (fuzzy professor name matching)

Usage:
    python 03_build_index.py
    python 03_build_index.py
"""

import sqlite3
from pathlib import Path

import chromadb
from llama_index.core import Settings

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore

DEFAULT_DB = "2025-2026_data/university.db"
DEFAULT_CHROMA_DIR = "2025-2026_data/chroma_store"

from utils import *


def build_column_index(
    values: list[str],
    collection_name: str,
    chroma_client: chromadb.PersistentClient,
) -> VectorStoreIndex:
    """
    Creates (or recreates) a ChromaDB collection and builds a VectorStoreIndex
    from a list of string values. Deletes existing collection first to avoid
    stale embeddings after a DB update.
    """
    # Delete existing collection to ensure fresh embeddings on rebuild
    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass  # Collection did not exist yet — fine

    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # One TextNode per distinct value
    nodes = [TextNode(text=v) for v in values if v and v.strip()]

    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        show_progress=True,
    )
    print(f"  [{collection_name}] indexed {len(nodes)} values")
    return index


def build_all_indexes(db_path: Path, chroma_dir: Path) -> None:
    chroma_dir.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
    con = sqlite3.connect(db_path)



# ------------ PERSONALE ---------------

    print("\nBuilding index: personale.nome_and_surname")
    rows = con.execute(
        "SELECT DISTINCT nome_and_surname FROM personale WHERE nome_and_surname IS NOT NULL"
    ).fetchall()
    build_column_index([r[0] for r in rows], "personale__nome_and_surname", chroma_client)

    print("\nBuilding index: personale.role")
    rows = con.execute(
        "SELECT DISTINCT role FROM personale WHERE role IS NOT NULL"
    ).fetchall()
    build_column_index([r[0] for r in rows], "personale__role", chroma_client)

    print("\nBuilding index: personale.department")
    rows = con.execute(
        "SELECT DISTINCT department FROM personale WHERE department IS NOT NULL"
    ).fetchall()
    build_column_index([r[0] for r in rows], "personale__department", chroma_client)

# ----------- INSEGNAMENTO ---------------

    print("\nBuilding index: insegnamento.degree_program_name")
    rows = con.execute(
        "SELECT DISTINCT degree_program_name FROM insegnamento WHERE degree_program_name IS NOT NULL"
    ).fetchall()
    build_column_index([r[0] for r in rows], "insegnamento__degree_program_name", chroma_client)

    print("\nBuilding index: insegnamento.degree_program_name_eng")
    rows = con.execute(
        "SELECT DISTINCT degree_program_name_eng FROM insegnamento WHERE degree_program_name_eng IS NOT NULL"
    ).fetchall()
    build_column_index([r[0] for r in rows], "insegnamento__degree_program_name_eng", chroma_client)

    print("\nBuilding index: insegnamento.subject_name")
    rows = con.execute(
        "SELECT DISTINCT subject_name FROM lezione WHERE subject_name IS NOT NULL"
    ).fetchall()
    build_column_index([r[0] for r in rows], "insegnamento__subject_name", chroma_client)

    print("\nBuilding index: insegnamento.professors")
    rows = con.execute(
        "SELECT DISTINCT professors FROM insegnamento WHERE professors IS NOT NULL"
    ).fetchall()
    build_column_index([r[0] for r in rows], "insegnamento__professors", chroma_client)

    print("\nBuilding index: insegnamento.period")
    rows = con.execute(
        "SELECT DISTINCT period FROM insegnamento WHERE period IS NOT NULL"
    ).fetchall()
    build_column_index([r[0] for r in rows], "insegnamento__period", chroma_client)


# ------------ LEZIONE ---------------

    print("\nBuilding index: lezione.degree_program_name")
    rows = con.execute(
        "SELECT DISTINCT degree_program_name FROM lezione WHERE degree_program_name IS NOT NULL"
    ).fetchall()
    build_column_index([r[0] for r in rows], "lezione__degree_program_name", chroma_client)

    print("\nBuilding index: lezione.subject_name")
    rows = con.execute(
        "SELECT DISTINCT subject_name FROM lezione WHERE subject_name IS NOT NULL"
    ).fetchall()
    build_column_index([r[0] for r in rows], "lezione__subject_name", chroma_client)

    print("\nBuilding index: lezione.study_year_code")
    rows = con.execute(
        "SELECT DISTINCT study_year_code FROM lezione WHERE study_year_code IS NOT NULL"
    ).fetchall()
    build_column_index([r[0] for r in rows], "lezione__study_year_code", chroma_client)

    print("\nBuilding index: lezione.curriculum")
    rows = con.execute(
        "SELECT DISTINCT curriculum FROM lezione WHERE curriculum IS NOT NULL"
    ).fetchall()
    build_column_index([r[0] for r in rows], "lezione__curriculum", chroma_client)

    print("\nBuilding index: lezione.date")
    rows = con.execute(
        "SELECT DISTINCT date FROM lezione WHERE date IS NOT NULL"
    ).fetchall()
    build_column_index([r[0] for r in rows], "lezione__date", chroma_client)

    print("\nBuilding index: lezione.department")
    rows = con.execute(
        "SELECT DISTINCT department FROM lezione WHERE department IS NOT NULL"
    ).fetchall()
    build_column_index([r[0] for r in rows], "lezione__department", chroma_client)

    print("\nBuilding index: lezione.room_name")
    rows = con.execute(
        "SELECT DISTINCT room_name FROM lezione WHERE room_name IS NOT NULL"
    ).fetchall()
    build_column_index([r[0] for r in rows], "lezione__room_name", chroma_client)

    print("\nBuilding index: lezione.site_name")
    rows = con.execute(
        "SELECT DISTINCT site_name FROM lezione WHERE site_name IS NOT NULL"
    ).fetchall()
    build_column_index([r[0] for r in rows], "lezione__site_name", chroma_client)

    print("\nBuilding index: lezione.address")
    rows = con.execute(
        "SELECT DISTINCT address FROM lezione WHERE address IS NOT NULL"
    ).fetchall()
    build_column_index([r[0] for r in rows], "lezione__address", chroma_client)

    print("\nBuilding index: lezione.professors")
    rows = con.execute(
        "SELECT DISTINCT professors FROM lezione WHERE professors IS NOT NULL"
    ).fetchall()
    build_column_index([r[0] for r in rows], "lezione__professors", chroma_client)


# ------------ EVENTO AULA ---------------

    print("\nBuilding index: evento_aula.site_name")
    rows = con.execute(
        "SELECT DISTINCT site_name FROM evento_aula WHERE site_name IS NOT NULL"
    ).fetchall()
    build_column_index([r[0] for r in rows], "evento_aula__site_name", chroma_client)

    print("\nBuilding index: evento_aula.room_name")
    rows = con.execute(
        "SELECT DISTINCT room_name FROM evento_aula WHERE room_name IS NOT NULL"
    ).fetchall()
    build_column_index([r[0] for r in rows], "evento_aula__room_name", chroma_client)

    print("\nBuilding index: evento_aula.name_event")
    rows = con.execute(
        "SELECT DISTINCT name_event FROM evento_aula WHERE name_event IS NOT NULL"
    ).fetchall()
    build_column_index([r[0] for r in rows], "evento_aula__name_event", chroma_client) 

    print("\nBuilding index: evento_aula.professors")
    rows = con.execute(
        "SELECT DISTINCT professors FROM evento_aula WHERE professors IS NOT NULL"
    ).fetchall()
    build_column_index([r[0] for r in rows], "evento_aula__professors", chroma_client)

    print("\nBuilding index: evento_aula.event_type")
    rows = con.execute(
        "SELECT DISTINCT event_type FROM evento_aula WHERE event_type IS NOT NULL"
    ).fetchall()
    build_column_index([r[0] for r in rows], "evento_aula__event_type", chroma_client)

# ------------ INFO AULA ---------------

    print("\nBuilding index: info_aula.site_name")
    rows = con.execute(
        "SELECT DISTINCT site_name FROM info_aula WHERE site_name IS NOT NULL"
    ).fetchall()
    build_column_index([r[0] for r in rows], "info_aula__site_name", chroma_client)

    print("\nBuilding index: info_aula.room_name")
    rows = con.execute(
        "SELECT DISTINCT room_name FROM info_aula WHERE room_name IS NOT NULL"
    ).fetchall()
    build_column_index([r[0] for r in rows], "info_aula__room_name", chroma_client)

    print("\nBuilding index: info_aula.address")
    rows = con.execute(
        "SELECT DISTINCT address FROM info_aula WHERE address IS NOT NULL"
    ).fetchall()
    build_column_index([r[0] for r in rows], "info_aula__address", chroma_client)

    print("\nBuilding index: info_aula.floor")
    rows = con.execute(
        "SELECT DISTINCT floor FROM info_aula WHERE floor IS NOT NULL"
    ).fetchall()
    build_column_index([r[0] for r in rows], "info_aula__floor", chroma_client)

    print("\nBuilding index: info_aula.room_type")
    rows = con.execute(
        "SELECT DISTINCT room_type FROM info_aula WHERE room_type IS NOT NULL"
    ).fetchall()
    build_column_index([r[0] for r in rows], "info_aula__room_type", chroma_client)

    con.close()
    print(f"\nAll indexes persisted to: {chroma_dir.resolve()}")


if __name__ == "__main__":
    build_all_indexes(Path(DEFAULT_DB), Path(DEFAULT_CHROMA_DIR))