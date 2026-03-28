# UniTS University Database — NL-to-SQL Query Engine

A natural language query system for the University of Trieste (UniTS) database.
It allows users to ask questions in plain Italian or English and get accurate answers
by combining vector search (RAG) with LLM-generated SQL queries.

## Architecture

- **SQLite** database storing university data (courses, staff, lessons, classrooms, etc.)
- **ChromaDB** vector store for semantic column retrieval (column-level RAG)
- **LlamaIndex** `SQLTableRetrieverQueryEngine` for text-to-SQL generation
- **Local LLM** served via llama.cpp / LiteLLM (OpenAI-compatible API)
- **HuggingFace Embeddings** (`BAAI/bge-m3`) for multilingual semantic search

## Embedding Model

The system uses [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3) (570M parameters) for all vector similarity searches.

This model was chosen over lighter alternatives (e.g. `intfloat/multilingual-e5-small`, 118M params)
because smaller models tend to match words based on surface-level lexical similarity rather than
true semantic meaning. A concrete example of this failure: when a user asks about *"geometria"*
(mathematics), a small model would also retrieve *"GEOMORFOLOGIA"* (geology) as a strong match,
simply because both words share the `GEO` prefix and have similar character sequences.
`bge-m3` handles these cases correctly by understanding the semantic difference between the two domains.

`bge-m3` is currently state-of-the-art for multilingual embeddings and handles mixed
Italian/English content robustly, which is important given that UniTS course names and
degree programs appear in both languages.
## Setup

### 1. Create the database schema
Creates all tables: `personale`, `insegnamento`, `lezione`, `evento_aula`, `info_aula`, `corso_di_laurea`.
```bash
python 01_create_schema.py
```

### 2. Populate the database
Loads data from JSON files into the database, with text normalization (apostrophe handling, whitespace collapsing).
```bash
python 02_populate_db.py
```

### 3. Build the RAG index
Embeds the values of key columns into ChromaDB collections.
This allows the system to match user queries to exact DB values even when phrasing differs
(e.g. "Martino Trevisan" → `TREVISAN MARTINO`).
```bash
python 03_create_rag_index.py
```

### 4. Run the query engine
Starts the interactive query interface. Requires a local LLM running on an OpenAI-compatible endpoint.
```bash
python 04_query.py
```

## Testing

Run a predefined set of test questions and inspect retrieved chunks, generated SQL, and final answers.
Results are saved to a markdown report with per-query timing breakdown.
```bash
python test_llm.py
# With custom paths:
python test_llm.py --db path/to/university.db --chroma-dir path/to/chroma_store --output results.md
```

## Requirements

- Python 3.11+
- A local LLM server exposing an OpenAI-compatible `/v1` endpoint (e.g. llama.cpp, LiteLLM)
- See `requirements.txt` for Python dependencies