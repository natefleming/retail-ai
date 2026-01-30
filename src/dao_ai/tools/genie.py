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
    GenieInMemorySemanticCacheParametersModel,
    GenieLRUCacheParametersModel,
    GenieRoomModel,
    GenieSemanticCacheParametersModel,
    value_of,
)
from dao_ai.genie import GenieService, GenieServiceBase
from dao_ai.genie.cache import (
    CacheResult,
    InMemorySemanticCacheService,
    LRUCacheService,
    SemanticCacheService,
)
from dao_ai.state import AgentState, Context, SessionState


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
    in_memory_semantic_cache_parameters: GenieInMemorySemanticCacheParametersModel
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
            for similarity-based query matching (requires PostgreSQL/Lakebase)
        in_memory_semantic_cache_parameters: Optional in-memory semantic cache configuration
            for similarity-based query matching (no database required)

    Returns:
        A LangGraph tool that processes natural language queries through Genie
    """
    logger.debug(
        "Creating Genie tool",
        genie_room_type=type(genie_room).__name__,
        persist_conversation=persist_conversation,
        truncate_results=truncate_results,
        name=name,
        has_lru_cache=lru_cache_parameters is not None,
        has_semantic_cache=semantic_cache_parameters is not None,
        has_in_memory_semantic_cache=in_memory_semantic_cache_parameters is not None,
    )

    if isinstance(genie_room, dict):
        genie_room = GenieRoomModel(**genie_room)

    if isinstance(lru_cache_parameters, dict):
        lru_cache_parameters = GenieLRUCacheParametersModel(**lru_cache_parameters)

    if isinstance(semantic_cache_parameters, dict):
        semantic_cache_parameters = GenieSemanticCacheParametersModel(
            **semantic_cache_parameters
        )

    if isinstance(in_memory_semantic_cache_parameters, dict):
        in_memory_semantic_cache_parameters = GenieInMemorySemanticCacheParametersModel(
            **in_memory_semantic_cache_parameters
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

    # Cache for genie service - created lazily on first call
    # This allows us to use workspace_client_from with runtime context for OBO
    _cached_genie_service: GenieServiceBase | None = None

    def _get_genie_service(context: Context | None) -> GenieServiceBase:
        """Get or create the Genie service, using context for OBO auth if available."""
        nonlocal _cached_genie_service

        # Use cached service if available (for non-OBO or after first call)
        # For OBO, we need fresh workspace client each time to use the user's token
        if _cached_genie_service is not None and not genie_room.on_behalf_of_user:
            return _cached_genie_service

        # Get workspace client using context for OBO support
        from databricks.sdk import WorkspaceClient

        workspace_client: WorkspaceClient = genie_room.workspace_client_from(context)

        genie: Genie = Genie(
            space_id=space_id,
            client=workspace_client,
            truncate_results=truncate_results,
        )

        genie_service: GenieServiceBase = GenieService(genie)

        # Wrap with semantic cache first (checked second/third due to decorator pattern)
        if semantic_cache_parameters is not None:
            genie_service = SemanticCacheService(
                impl=genie_service,
                parameters=semantic_cache_parameters,
                workspace_client=workspace_client,
            ).initialize()

        # Wrap with in-memory semantic cache (alternative to PostgreSQL semantic cache)
        if in_memory_semantic_cache_parameters is not None:
            genie_service = InMemorySemanticCacheService(
                impl=genie_service,
                parameters=in_memory_semantic_cache_parameters,
                workspace_client=workspace_client,
            ).initialize()

        # Wrap with LRU cache last (checked first - fast O(1) exact match)
        if lru_cache_parameters is not None:
            genie_service = LRUCacheService(
                impl=genie_service,
                parameters=lru_cache_parameters,
            )

        # Cache for non-OBO scenarios
        if not genie_room.on_behalf_of_user:
            _cached_genie_service = genie_service

        return genie_service

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
        context: Context | None = runtime.context

        # Get genie service with OBO support via context
        genie_service: GenieServiceBase = _get_genie_service(context)

        # Ensure space_id is a string for state keys
        space_id_str: str = str(space_id)

        # Get session state (or create new one)
        session: SessionState = state.get("session", SessionState())

        # Get existing conversation ID from session
        existing_conversation_id: str | None = session.genie.get_conversation_id(
            space_id_str
        )
        logger.trace(
            "Using existing conversation ID",
            space_id=space_id_str,
            conversation_id=existing_conversation_id,
        )

        # Log the prompt being sent to Genie
        logger.trace(
            "Sending prompt to Genie",
            space_id=space_id_str,
            conversation_id=existing_conversation_id,
            prompt=question[:500] + "..." if len(question) > 500 else question,
        )

        # Call ask_question which always returns CacheResult with cache metadata
        cache_result: CacheResult = genie_service.ask_question(
            question, conversation_id=existing_conversation_id
        )
        genie_response: GenieResponse = cache_result.response
        cache_hit: bool = cache_result.cache_hit
        cache_key: str | None = cache_result.served_by

        current_conversation_id: str = genie_response.conversation_id
        logger.debug(
            "Genie question answered",
            space_id=space_id_str,
            conversation_id=current_conversation_id,
            cache_hit=cache_hit,
            cache_key=cache_key,
        )

        # Log truncated response for debugging
        result_preview: str = str(genie_response.result)
        if len(result_preview) > 500:
            result_preview = result_preview[:500] + "..."
        logger.trace(
            "Genie response content",
            question=question[:100] + "..." if len(question) > 100 else question,
            query=genie_response.query,
            description=(
                genie_response.description[:200] + "..."
                if genie_response.description and len(genie_response.description) > 200
                else genie_response.description
            ),
            result_preview=result_preview,
        )

        # Update session state with cache information
        if persist_conversation:
            session.genie.update_space(
                space_id=space_id_str,
                conversation_id=current_conversation_id,
                cache_hit=cache_hit,
                cache_key=cache_key,
                last_query=question,
            )

        # Build update dict with response and session
        update: dict[str, Any] = {
            "messages": [
                ToolMessage(
                    _response_to_json(genie_response), tool_call_id=tool_call_id
                )
            ],
        }

        if persist_conversation:
            update["session"] = session

        return Command(update=update)

    return genie_tool
