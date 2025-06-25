import os
from textwrap import dedent
from typing import Any, Callable, Optional

from databricks_ai_bridge.genie import GenieResponse
from databricks_langchain.genie import Genie
from langchain_core.tools import StructuredTool

from dao_ai.config import (
    GenieRoomModel,
)


def create_genie_tool(
    genie_room: GenieRoomModel | dict[str, Any],
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Callable[[str], GenieResponse]:
    """
    Create a tool for interacting with Databricks Genie for natural language queries to databases.

    This factory function generates a tool that leverages Databricks Genie to translate natural
    language questions into SQL queries and execute them against retail databases. This enables
    answering questions about inventory, sales, and other structured retail data.

    Args:
        space_id: Databricks workspace ID where Genie is configured. If None, tries to
                get it from DATABRICKS_GENIE_SPACE_ID environment variable.

    Returns:
        A callable tool function that processes natural language queries through Genie
    """

    if isinstance(genie_room, dict):
        genie_room = GenieRoomModel(**genie_room)

    space_id: str = genie_room.space_id or os.environ.get("DATABRICKS_GENIE_SPACE_ID")

    genie: Genie = Genie(
        space_id=space_id,
        client=genie_room.workspace_client,
    )

    default_description: str = dedent("""
    This tool lets you have a conversation and chat with tabular data about <topic>. You should ask
    questions about the data and the tool will try to answer them.
    Please ask simple clear questions that can be answer by sql queries. If you need to do statistics or other forms of testing defer to using another tool.
    Try to ask for aggregations on the data and ask very simple questions.
    Prefer to call this tool multiple times rather than asking a complex question.
    """)

    if description is None:
        description = default_description

    doc_signature: str = dedent("""
    Args:
        question (str): The question to ask to ask Genie

    Returns:
        response (GenieResponse): An object containing the Genie response
    """)

    doc: str = description + "\n" + doc_signature

    def genie_tool(question: str) -> GenieResponse:
        response: GenieResponse = genie.ask_question(question)
        return response

    name: str = name if name else genie_tool.__name__

    structured_tool: StructuredTool = StructuredTool.from_function(
        func=genie_tool, name=name, description=doc, parse_docstring=False
    )

    return structured_tool
