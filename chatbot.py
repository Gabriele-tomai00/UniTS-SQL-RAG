"""
chatbot.py
----------
Agentic RAG Chatbot for UniTS.

Pipeline per ogni query:

  User query
    │
    ▼
  RouterQueryEngine  (LLMMultiSelector)
    ├─► [Tool] sql_rag     → RoutedSQLQueryEngine (two-stage table routing + col retrievers)
    └─► [Tool] general_rag → VectorStoreIndex (unstructured docs, optional)
    │
    ▼
  Natural language answer

Usage:
    python chatbot.py
    python chatbot.py --query "A che ora è la lezione di Robotics domani?"
    python chatbot.py --no-multi      # single-tool routing
"""

import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Import utils FIRST — sets Settings.llm and Settings.embed_model globally
# ---------------------------------------------------------------------------
from utils import build_query_engine, RoutedSQLQueryEngine  # noqa: E402

from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMMultiSelector, LLMSingleSelector
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import io
import sys
import contextlib

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_DB         = Path("2025-2026_data/university.db")
DEFAULT_CHROMA_DIR = Path("2025-2026_data/chroma_store")

# Optional unstructured docs collection — set to None to disable general RAG
GENERAL_CHROMA_DIR        = Path("2025-2026_data/chroma_store_docs")
GENERAL_CHROMA_COLLECTION = "units_docs"


# ---------------------------------------------------------------------------
# Adapter: RoutedSQLQueryEngine → LlamaIndex QueryEngineTool interface
# ---------------------------------------------------------------------------

class SQLQueryEngineTool:
    """
    Thin wrapper so that RoutedSQLQueryEngine satisfies the interface
    expected by LlamaIndex's QueryEngineTool (needs a .query() method).
    """

    def __init__(self, routed_engine: RoutedSQLQueryEngine):
        self._engine = routed_engine

    def query(self, query_str: str):
        return self._engine.query(query_str)

    def retrieve(self, query_str: str):
        return self._engine.query(query_str)


# ---------------------------------------------------------------------------
# Optional: General RAG over unstructured documents
# ---------------------------------------------------------------------------

def build_general_rag(
    chroma_dir: Path,
    collection_name: str,
    similarity_cutoff: float = 0.45,
    top_k: int = 6,
):
    """
    Loads an existing ChromaDB collection (unstructured docs) and returns
    a query engine. Returns None silently if the collection does not exist.
    """
    try:
        client     = chromadb.PersistentClient(path=str(chroma_dir))
        collection = client.get_collection(collection_name)
    except Exception:
        print(
            f"[INFO] General RAG collection '{collection_name}' not found at "
            f"'{chroma_dir}'. Skipping general RAG tool."
        )
        return None

    vector_store    = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index           = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
    )
    return index.as_query_engine(
        similarity_top_k=top_k,
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)
        ],
    )


# ---------------------------------------------------------------------------
# Build the router engine
# ---------------------------------------------------------------------------

def build_router_engine(
    db_path: Path = DEFAULT_DB,
    chroma_dir: Path = DEFAULT_CHROMA_DIR,
    general_chroma_dir: Path = GENERAL_CHROMA_DIR,
    general_collection: str = GENERAL_CHROMA_COLLECTION,
    multi_select: bool = True,
) -> RouterQueryEngine:
    """
    Assembles the RouterQueryEngine from:
      1. RoutedSQLQueryEngine  (always present)
      2. General RAG engine    (optional, skipped if collection not found)

    multi_select=True  → LLMMultiSelector: LLM can pick both tools for
                          composite queries, activating SubQuestionQueryEngine
    multi_select=False → LLMSingleSelector: simpler, one tool per query
    """
    print("Loading SQL query engine...")
    routed_sql  = build_query_engine(db_path, chroma_dir)
    sql_adapter = SQLQueryEngineTool(routed_sql)

    sql_tool = QueryEngineTool(
        query_engine=sql_adapter,
        metadata=ToolMetadata(
            name="sql_rag",
            description=(
                "Answers questions that require structured data from the UniTS database. "
                "Use for: lesson schedules, lecture times, classroom info, room bookings, "
                "exam dates, course details (professor, Teams code, semester), "
                "staff contacts (email, role, department), enrollment stats. "
                "Any question involving specific times, dates, rooms, names, or numbers."
            ),
        ),
    )

    tools = [sql_tool]

    print("Loading general RAG engine...")
    general_engine = build_general_rag(general_chroma_dir, general_collection)
    if general_engine is not None:
        general_tool = QueryEngineTool(
            query_engine=general_engine,
            metadata=ToolMetadata(
                name="general_rag",
                description=(
                    "Answers open-ended or descriptive questions about UniTS using "
                    "unstructured documents: regulations, admission procedures, "
                    "degree program descriptions, scholarship info, campus services, "
                    "university policies, and any content not stored in a database table."
                ),
            ),
        )
        tools.append(general_tool)
    else:
        print("[INFO] Running with SQL RAG only.")

    selector = (
        LLMMultiSelector.from_defaults(llm=Settings.llm)
        if multi_select
        else LLMSingleSelector.from_defaults(llm=Settings.llm)
    )

    router = RouterQueryEngine(
        selector=selector,
        query_engine_tools=tools,
        verbose=True,   # kept True so the captured output contains tool names
    )

    print("Router ready.\n")
    return router


# ---------------------------------------------------------------------------
# Chatbot
# ---------------------------------------------------------------------------

class UniTSChatbot:
    """
    Stateful chatbot with rolling conversation history.

    History is prepended to each query so the LLM has context for
    follow-up questions (e.g. "E invece il martedì?").
    """

    SYSTEM_PROMPT = (
        "Sei un assistente dell'Università degli Studi di Trieste (UniTS). "
        "Rispondi nella stessa lingua dell'utente. "
        "Sii preciso e conciso. "
        "Se non hai abbastanza informazioni per rispondere, dillo chiaramente."
    )

    def __init__(
        self,
        router: RouterQueryEngine,
        history_turns: int = 3,
    ):
        self._router          = router
        self._history: list[dict] = []
        self._max_turns       = history_turns

    def chat(self, user_query: str) -> str:
        if self._history:
            past = "\n".join(
                f"{'Utente' if t['role'] == 'user' else 'Assistente'}: {t['content']}"
                for t in self._history[-(self._max_turns * 2):]
            )
            prompt = (
                f"{self.SYSTEM_PROMPT}\n\n"
                f"Conversazione precedente:\n{past}\n\n"
                f"Utente: {user_query}"
            )
        else:
            prompt = f"{self.SYSTEM_PROMPT}\n\nUtente: {user_query}"

        # Suppress all debug output from LoggingRetriever, table router,
        # and RouterQueryEngine verbose mode; capture it to detect which
        # tools were actually selected.
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            response = self._router.query(prompt)

        debug_output = captured.getvalue()

        # Detect which tools were invoked from the captured verbose output
        used_sql     = "sql_rag"     in debug_output
        used_general = "general_rag" in debug_output
        if used_sql and used_general:
            print("[routing] SQL database + General RAG")
        elif used_sql:
            print("[routing] SQL database")
        elif used_general:
            print("[routing] General RAG")

        answer = str(response)

        self._history.append({"role": "user",      "content": user_query})
        self._history.append({"role": "assistant", "content": answer})

        return answer

    def reset(self):
        self._history.clear()
        print("[History cleared]\n")

    @property
    def history(self) -> list[dict]:
        return list(self._history)


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------

def interactive_loop(bot: UniTSChatbot) -> None:
    print("UniTS Chatbot — type 'exit' to quit, 'reset' to clear history.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            break
        if user_input.lower() == "reset":
            bot.reset()
            continue

        answer = bot.chat(user_input)
        print(f"Bot: {answer}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UniTS Agentic RAG Chatbot")
    parser.add_argument("--query",         default=None,        help="Single query (non-interactive)")
    parser.add_argument("--no-multi",      action="store_true", help="Use single-select router")
    parser.add_argument("--history-turns", type=int, default=3, help="Past exchanges to include in context")
    args = parser.parse_args()

    router = build_router_engine(multi_select=not args.no_multi)
    bot    = UniTSChatbot(router, history_turns=args.history_turns)

    if args.query:
        answer = bot.chat(args.query)
        print(f"\nBot: {answer}\n")
    else:
        interactive_loop(bot)