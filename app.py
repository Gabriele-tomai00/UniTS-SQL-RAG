"""
Run with:
    chainlit run app.py
"""

import os
from pathlib import Path

import chainlit as cl
from utils import (
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
    query_engine = build_query_engine(DEFAULT_DB, DEFAULT_CHROMA_DIR)
    cl.user_session.set("query_engine", query_engine)
    
    await cl.Message(
        content=(
            "🎓 Welcome to the UniTS university information system!\n"
            "You can ask me about schedules, professors, classrooms, and much more."
        ),
    ).send()
            
@cl.on_message
async def on_message(message: cl.Message):
    query_engine: RoutedSQLQueryEngine = cl.user_session.get("query_engine")

    async with cl.Step(name="Query elaboration...") as step:
        response, timings = await cl.make_async(query_engine.query)(message.content)

        sql = None
        if hasattr(response, "metadata") and response.metadata:
            sql = response.metadata.get("sql_query")

        if sql:
            step.output = f"```sql\n{sql}\n```"

    await cl.Message(content=str(response)).send()


@cl.on_stop
def on_stop():
    print("The user stopped the task.")


@cl.on_chat_end
def on_chat_end():
    print("The user disconnected.")




@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="info about exams dates for Machine",
            message="tell me all the exams in the calendar for the Machine Learning subject that were held in February 2026",
        ),

        cl.Starter(
            label="Info about a professor",
            message="Who is Trevisan Martino",
        )
    ]
# ...