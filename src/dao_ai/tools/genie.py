"""
Genie tool for natural language queries to databases.

This module provides the tool factory for creating LangGraph tools that
interact with Databricks Genie.

For the core Genie service and cache implementations, see:
- dao_ai.genie: GenieService, GenieServiceBase
- dao_ai.genie.cache: LRUCacheService, SemanticCacheService
"""

import json
import os
from textwrap import dedent
from typing import Annotated, Any, Callable

import pandas as pd
from databricks_ai_bridge.genie import Genie, GenieResponse
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from loguru import logger
from pydantic import BaseModel

from dao_ai.config import (
    AnyVariable,
    CompositeVariableModel,
    GenieLRUCacheParametersModel,
    GenieRoomModel,
    GenieSemanticCacheParametersModel,
    value_of,
)
from dao_ai.genie import GenieService, GenieServiceBase
from dao_ai.genie.cache import LRUCacheService, SemanticCacheService
from dao_ai.state import AgentState, Context


class GenieToolInput(BaseModel):
    """Input schema for Genie tool - only includes user-facing parameters."""

    question: str


def _response_to_json(response: GenieResponse) -> str:
    """Convert GenieResponse to JSON string, handling DataFrame results."""
    # Convert result to string if it's a DataFrame
    result: str | pd.DataFrame = response.result
    if isinstance(result, pd.DataFrame):
        result = result.to_markdown()

    data: dict[str, Any] = {
        "result": result,
        "query": response.query,
        "description": response.description,
        "conversation_id": response.conversation_id,
    }
    return json.dumps(data)


def create_genie_tool(
    genie_room: GenieRoomModel | dict[str, Any],
    name: str | None = None,
    description: str | None = None,
    persist_conversation: bool = True,
    truncate_results: bool = False,
    lru_cache_parameters: GenieLRUCacheParametersModel | dict[str, Any] | None = None,
    semantic_cache_parameters: GenieSemanticCacheParametersModel
    | dict[str, Any]
    | None = None,
) -> Callable[..., Command]:
    """
    Create a tool for interacting with Databricks Genie for natural language queries to databases.

    This factory function generates a tool that leverages Databricks Genie to translate natural
    language questions into SQL queries and execute them against retail databases. This enables
    answering questions about inventory, sales, and other structured retail data.

    Args:
        genie_room: GenieRoomModel or dict containing Genie configuration
        name: Optional custom name for the tool. If None, uses default "genie_tool"
        description: Optional custom description for the tool. If None, uses default description
        persist_conversation: Whether to persist conversation IDs across tool calls for
            multi-turn conversations within the same Genie space
        truncate_results: Whether to truncate large query results to fit token limits
        lru_cache_parameters: Optional LRU cache configuration for SQL query caching
        semantic_cache_parameters: Optional semantic cache configuration using pg_vector
            for similarity-based query matching

    Returns:
        A LangGraph tool that processes natural language queries through Genie
    """
    logger.debug("create_genie_tool")
    logger.debug(f"genie_room type: {type(genie_room)}")
    logger.debug(f"genie_room: {genie_room}")
    logger.debug(f"persist_conversation: {persist_conversation}")
    logger.debug(f"truncate_results: {truncate_results}")
    logger.debug(f"name: {name}")
    logger.debug(f"description: {description}")
    logger.debug(f"lru_cache_parameters: {lru_cache_parameters}")
    logger.debug(f"semantic_cache_parameters: {semantic_cache_parameters}")

    if isinstance(genie_room, dict):
        genie_room = GenieRoomModel(**genie_room)

    if isinstance(lru_cache_parameters, dict):
        lru_cache_parameters = GenieLRUCacheParametersModel(**lru_cache_parameters)

    if isinstance(semantic_cache_parameters, dict):
        semantic_cache_parameters = GenieSemanticCacheParametersModel(
            **semantic_cache_parameters
        )

    space_id: AnyVariable = genie_room.space_id or os.environ.get(
        "DATABRICKS_GENIE_SPACE_ID"
    )
    if isinstance(space_id, dict):
        space_id = CompositeVariableModel(**space_id)
    space_id = value_of(space_id)

    default_description: str = dedent("""
    This tool lets you have a conversation and chat with tabular data about <topic>. You should ask
    questions about the data and the tool will try to answer them.
    Please ask simple clear questions that can be answer by sql queries. If you need to do statistics or other forms of testing defer to using another tool.
    Try to ask for aggregations on the data and ask very simple questions.
    Prefer to call this tool multiple times rather than asking a complex question.
    """)

    tool_description: str = (
        description if description is not None else default_description
    )
    tool_name: str = name if name is not None else "genie_tool"

    function_docs = """

Args:
question (str): The question to ask to ask Genie about your data. Ask simple, clear questions about your tabular data. For complex analysis, ask multiple simple questions rather than one complex question.

Returns:
GenieResponse: A response object containing the conversation ID and result from Genie."""
    tool_description = tool_description + function_docs

    genie: Genie = Genie(
        space_id=space_id,
        client=genie_room.workspace_client,
        truncate_results=truncate_results,
    )

    genie_service: GenieServiceBase = GenieService(genie)

    # Wrap with semantic cache first (checked second due to decorator pattern)
    if semantic_cache_parameters is not None:
        genie_service = SemanticCacheService(
            impl=genie_service,
            parameters=semantic_cache_parameters,
            genie_space_id=space_id,
        ).initialize()  # Eagerly initialize to fail fast and create table

    # Wrap with LRU cache last (checked first - fast O(1) exact match)
    if lru_cache_parameters is not None:
        genie_service = LRUCacheService(
            impl=genie_service,
            parameters=lru_cache_parameters,
        )

    @tool(
        name_or_callable=tool_name,
        description=tool_description,
    )
    def genie_tool(
        question: Annotated[str, "The question to ask Genie about your data"],
        runtime: ToolRuntime[Context, AgentState],
    ) -> Command:
        """Process a natural language question through Databricks Genie.

        Uses ToolRuntime to access state and context in a type-safe way.
        """
        # Access state through runtime
        state: AgentState = runtime.state
        tool_call_id: str = runtime.tool_call_id

        # Get existing conversation mapping and retrieve conversation ID for this space
        conversation_ids: dict[str, str] = state.get("genie_conversation_ids", {})
        existing_conversation_id: str | None = conversation_ids.get(space_id)
        logger.debug(
            f"Existing conversation ID for space {space_id}: {existing_conversation_id}"
        )

        response: GenieResponse = genie_service.ask_question(
            question, conversation_id=existing_conversation_id
        )

        current_conversation_id: str = response.conversation_id
        logger.debug(
            f"Current conversation ID for space {space_id}: {current_conversation_id}"
        )

        # Update the conversation mapping with the new conversation ID for this space

        update: dict[str, Any] = {
            "messages": [
                ToolMessage(_response_to_json(response), tool_call_id=tool_call_id)
            ],
        }

        if persist_conversation:
            updated_conversation_ids: dict[str, str] = conversation_ids.copy()
            updated_conversation_ids[space_id] = current_conversation_id
            update["genie_conversation_ids"] = updated_conversation_ids

        return Command(update=update)

    return genie_tool
