"""
app.py
------
Chainlit front-end for the University NL → SQL → Answer pipeline.

Run with:
    chainlit run app.py
"""

import os
from pathlib import Path

import chainlit as cl
from index_utils import (
    TABLE_DOMAINS,
    TABLE_ROUTER_TOP_K,
    TEXT_TO_SQL_PROMPT,
    RoutedSQLQueryEngine,
    build_query_engine,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_DB         = Path("2025-2026_data/university.db")
DEFAULT_CHROMA_DIR = Path("2025-2026_data/chroma_store")


@cl.on_chat_start
async def on_chat_start():
    """
    Called once when a new chat session opens.
    Builds the query engine and stores it in the user session so it is
    reused across messages without rebuilding every time.
    """
    await cl.Message(
        content=(
            "🎓 Benvenuto nel sistema di informazioni universitarie UniTS!\n"
            "Puoi chiedermi orari, docenti, aule e molto altro."
        ),
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    _query_engine: RoutedSQLQueryEngine = build_query_engine(DEFAULT_DB, DEFAULT_CHROMA_DIR)
    async with cl.Step(name="Elaborazione query...") as step:
        response = await cl.make_async(_query_engine.query)(message.content)

        # Extract the generated SQL for transparency (if available)
        sql = None
        if hasattr(response, "metadata") and response.metadata:
            sql = response.metadata.get("sql_query")

        if sql:
            step.output = f"```sql\n{sql}\n```"

    # Send the final natural language answer
    await cl.Message(content=str(response)).send()


@cl.on_stop
def on_stop():
    print("The user stopped the task.")


@cl.on_chat_end
def on_chat_end():
    print("The user disconnected.")