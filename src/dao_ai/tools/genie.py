"""
Genie tool and toolkit for natural language queries to databases.

This module provides two factory functions for creating LangGraph tools that
interact with Databricks Genie:

- ``create_genie_tool``: Simple factory returning a single uncached Genie query tool.
- ``create_genie_toolkit``: Returns a ``GenieToolkit`` bundling a cached Genie query
  tool with an implicit feedback/invalidation tool. Both tools share one
  ``GenieServiceBase`` stack (and thus one LRU cache).

For the core Genie service and cache implementations, see:
- dao_ai.genie: GenieService, GenieServiceBase
- dao_ai.genie.cache: LRUCacheService, PostgresContextAwareGenieService, InMemoryContextAwareGenieService
"""

import json
import os
from textwrap import dedent
from typing import Annotated, Any, Callable

import pandas as pd
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, BaseToolkit
from langgraph.types import Command
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from dao_ai.config import (
    AnyVariable,
    CompositeVariableModel,
    GenieContextAwareCacheParametersModel,
    GenieInMemoryContextAwareCacheParametersModel,
    GenieLRUCacheParametersModel,
    GenieRoomModel,
    value_of,
)
from dao_ai.genie import GenieService, GenieServiceBase
from dao_ai.genie.cache import (
    CacheResult,
    InMemoryContextAwareGenieService,
    LRUCacheService,
    PostgresContextAwareGenieService,
)
from dao_ai.genie.core import Genie, GenieResponse
from dao_ai.state import AgentState, Context, SessionState
from dao_ai.tools.tracing import ResourceInfo, set_resource_attributes


class GenieToolInput(BaseModel):
    """Input schema for Genie tool - only includes user-facing parameters."""

    question: str


class GenieToolkit(BaseToolkit):
    """Toolkit bundling a cached Genie query tool with an implicit feedback tool.

    Both tools share one ``GenieServiceBase`` instance (and thus one LRU cache),
    so calling the feedback tool to invalidate a cache entry directly affects
    the query tool's next call.

    Created by :func:`create_genie_toolkit`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    tools: list[BaseTool] = Field(default_factory=list)

    def get_tools(self) -> list[BaseTool]:
        return self.tools


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _response_to_json(response: GenieResponse) -> str:
    """Convert GenieResponse to JSON string, handling DataFrame results."""
    result: str | pd.DataFrame = response.result
    if isinstance(result, pd.DataFrame):
        result = result.to_markdown()

    data: dict[str, Any] = {
        "result": result,
        "query": response.query,
        "description": response.description,
        "conversation_id": response.conversation_id,
        "statement_id": response.statement_id,
    }
    return json.dumps(data)


def _response_to_json_with_cache(
    response: GenieResponse, cache_hit: bool = False
) -> str:
    """Convert GenieResponse to JSON string, including cache metadata."""
    result: str | pd.DataFrame = response.result
    if isinstance(result, pd.DataFrame):
        result = result.to_markdown()

    data: dict[str, Any] = {
        "result": result,
        "query": response.query,
        "description": response.description,
        "conversation_id": response.conversation_id,
        "statement_id": response.statement_id,
        "cache_hit": cache_hit,
    }
    return json.dumps(data)


_DEFAULT_DESCRIPTION: str = dedent("""
    This tool lets you have a conversation and chat with tabular data about <topic>. You should ask
    questions about the data and the tool will try to answer them.
    Please ask simple clear questions that can be answer by sql queries. If you need to do statistics or other forms of testing defer to using another tool.
    Try to ask for aggregations on the data and ask very simple questions.
    Prefer to call this tool multiple times rather than asking a complex question.
    """)

_FUNCTION_DOCS: str = """

Args:
question (str): The question to ask to ask Genie about your data. Ask simple, clear questions about your tabular data. For complex analysis, ask multiple simple questions rather than one complex question.

Returns:
GenieResponse: A response object containing the conversation ID and result from Genie."""


def _resolve_genie_room(
    genie_room: GenieRoomModel | dict[str, Any],
) -> tuple[GenieRoomModel, str]:
    """Normalize genie_room to a model and resolve the space_id."""
    if isinstance(genie_room, dict):
        genie_room = GenieRoomModel(**genie_room)

    space_id: AnyVariable = genie_room.space_id or os.environ.get(
        "DATABRICKS_GENIE_SPACE_ID"
    )
    if isinstance(space_id, dict):
        space_id = CompositeVariableModel(**space_id)
    space_id = value_of(space_id)
    return genie_room, str(space_id)


# ---------------------------------------------------------------------------
# create_genie_tool  (simple, uncached)
# ---------------------------------------------------------------------------


def create_genie_tool(
    genie_room: GenieRoomModel | dict[str, Any],
    name: str | None = None,
    description: str | None = None,
    persist_conversation: bool = True,
    truncate_results: bool = False,
) -> Callable[..., Command]:
    """Create a simple Genie query tool with no caching.

    For cached queries with feedback/invalidation support, use
    :func:`create_genie_toolkit` instead.

    Args:
        genie_room: GenieRoomModel or dict containing Genie configuration.
        name: Custom tool name visible to the LLM. Defaults to ``"genie_tool"``.
        description: Custom tool description. Defaults to a generic prompt.
        persist_conversation: Persist conversation IDs across calls for multi-turn.
        truncate_results: Truncate large query results.

    Returns:
        A LangGraph tool that processes natural language queries through Genie.
    """
    logger.debug(
        "Creating Genie tool (uncached)",
        genie_room_type=type(genie_room).__name__,
        persist_conversation=persist_conversation,
        name=name,
    )

    genie_room_model, space_id_str = _resolve_genie_room(genie_room)

    tool_name: str = name if name is not None else "genie_tool"
    tool_description: str = (
        description if description is not None else _DEFAULT_DESCRIPTION
    ) + _FUNCTION_DOCS

    _cached_genie_service: GenieServiceBase | None = None

    def _get_genie_service(context: Context | None) -> GenieServiceBase:
        nonlocal _cached_genie_service
        if _cached_genie_service is not None and not genie_room_model.on_behalf_of_user:
            return _cached_genie_service

        from databricks.sdk import WorkspaceClient

        workspace_client: WorkspaceClient = genie_room_model.workspace_client_from(
            context
        )
        genie: Genie = Genie(
            space_id=space_id_str,
            client=workspace_client,
            truncate_results=truncate_results,
        )
        genie_service: GenieServiceBase = GenieService(genie)

        if not genie_room_model.on_behalf_of_user:
            _cached_genie_service = genie_service
        return genie_service

    @tool(name_or_callable=tool_name, description=tool_description)
    def genie_tool(
        question: Annotated[str, "The question to ask Genie about your data"],
        runtime: ToolRuntime[Context, AgentState],
    ) -> Command:
        """Process a natural language question through Databricks Genie."""
        state: AgentState = runtime.state
        tool_call_id: str = runtime.tool_call_id
        context: Context | None = runtime.context

        set_resource_attributes(
            ResourceInfo("genie", genie_room_model.on_behalf_of_user, space_id_str)
        )

        genie_service: GenieServiceBase = _get_genie_service(context)
        session: SessionState = state.get("session", SessionState())

        existing_conversation_id: str | None = (
            session.genie.get_conversation_id(space_id_str)
            if persist_conversation
            else None
        )

        cache_result: CacheResult = genie_service.ask_question(
            question, conversation_id=existing_conversation_id
        )
        genie_response: GenieResponse = cache_result.response

        if persist_conversation:
            session.genie.update_space(
                space_id=space_id_str,
                conversation_id=genie_response.conversation_id,
                cache_hit=False,
                last_query=question,
                message_id=cache_result.message_id,
            )

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


# ---------------------------------------------------------------------------
# create_genie_toolkit  (cached + implicit feedback tool)
# ---------------------------------------------------------------------------


def create_genie_toolkit(
    genie_room: GenieRoomModel | dict[str, Any],
    name: str | None = None,
    description: str | None = None,
    persist_conversation: bool = True,
    truncate_results: bool = False,
    lru_cache_parameters: GenieLRUCacheParametersModel | dict[str, Any] | None = None,
    context_aware_cache_parameters: (
        GenieContextAwareCacheParametersModel | dict[str, Any] | None
    ) = None,
    in_memory_context_aware_cache_parameters: (
        GenieInMemoryContextAwareCacheParametersModel | dict[str, Any] | None
    ) = None,
) -> GenieToolkit:
    """Create a cached Genie toolkit with query and implicit feedback tools.

    Returns a :class:`GenieToolkit` containing:

    * A **query tool** (named *name*) that translates natural-language questions
      into SQL via Databricks Genie, with results served from cache when possible.
    * A **feedback tool** (named ``{name}_feedback``) that lets the LLM invalidate
      a stale or incorrect cached result so the next query goes fresh to Genie.

    Both tools share one ``GenieServiceBase`` stack, so invalidation on the
    feedback tool directly clears the query tool's LRU cache.

    Args:
        genie_room: GenieRoomModel or dict containing Genie configuration.
        name: Custom tool name visible to the LLM. Defaults to ``"genie_tool"``.
        description: Custom tool description. Defaults to a generic prompt.
        persist_conversation: Persist conversation IDs across calls for multi-turn.
        truncate_results: Truncate large query results.
        lru_cache_parameters: LRU cache config for fast exact-match SQL caching.
        context_aware_cache_parameters: PostgreSQL/Lakebase context-aware cache config.
        in_memory_context_aware_cache_parameters: In-memory context-aware cache config.

    Returns:
        A GenieToolkit containing the query tool and feedback tool.
    """
    logger.debug(
        "Creating Genie toolkit",
        genie_room_type=type(genie_room).__name__,
        persist_conversation=persist_conversation,
        truncate_results=truncate_results,
        name=name,
        has_lru_cache=lru_cache_parameters is not None,
        has_context_aware_cache=context_aware_cache_parameters is not None,
        has_in_memory_context_aware_cache=in_memory_context_aware_cache_parameters
        is not None,
    )

    genie_room_model, space_id_str = _resolve_genie_room(genie_room)

    if isinstance(lru_cache_parameters, dict):
        lru_cache_parameters = GenieLRUCacheParametersModel(**lru_cache_parameters)
    if isinstance(context_aware_cache_parameters, dict):
        context_aware_cache_parameters = GenieContextAwareCacheParametersModel(
            **context_aware_cache_parameters
        )
    if isinstance(in_memory_context_aware_cache_parameters, dict):
        in_memory_context_aware_cache_parameters = (
            GenieInMemoryContextAwareCacheParametersModel(
                **in_memory_context_aware_cache_parameters
            )
        )

    tool_name: str = name if name is not None else "genie_tool"
    tool_description: str = (
        description if description is not None else _DEFAULT_DESCRIPTION
    ) + _FUNCTION_DOCS

    # ---- Shared service stack (one LRU, one closure) ----

    _cached_genie_service: GenieServiceBase | None = None

    def _get_genie_service(context: Context | None) -> GenieServiceBase:
        nonlocal _cached_genie_service
        if _cached_genie_service is not None and not genie_room_model.on_behalf_of_user:
            return _cached_genie_service

        from databricks.sdk import WorkspaceClient

        workspace_client: WorkspaceClient = genie_room_model.workspace_client_from(
            context
        )
        genie: Genie = Genie(
            space_id=space_id_str,
            client=workspace_client,
            truncate_results=truncate_results,
        )
        genie_service: GenieServiceBase = GenieService(genie)

        if context_aware_cache_parameters is not None:
            genie_service = PostgresContextAwareGenieService(
                impl=genie_service,
                parameters=context_aware_cache_parameters,
                workspace_client=workspace_client,
            ).initialize()

        if in_memory_context_aware_cache_parameters is not None:
            genie_service = InMemoryContextAwareGenieService(
                impl=genie_service,
                parameters=in_memory_context_aware_cache_parameters,
                workspace_client=workspace_client,
            ).initialize()

        if lru_cache_parameters is not None:
            genie_service = LRUCacheService(
                impl=genie_service,
                parameters=lru_cache_parameters,
            )

        if not genie_room_model.on_behalf_of_user:
            _cached_genie_service = genie_service
        return genie_service

    # Eagerly initialize the service stack (including Postgres pool) at
    # tool-creation time so the cost is paid during model load rather than
    # on the first inference request (which competes with the 5-min serving
    # worker timeout).
    if not genie_room_model.on_behalf_of_user:
        try:
            _get_genie_service(context=None)
            logger.debug(
                "Eagerly initialized Genie service stack",
                tool=tool_name,
                space_id=space_id_str,
            )
        except Exception as e:
            logger.warning(
                "Eager Genie service initialization failed; will retry on first call",
                tool=tool_name,
                error=str(e)[:200],
            )

    # ---- Query tool ----

    @tool(name_or_callable=tool_name, description=tool_description)
    def genie_tool(
        question: Annotated[str, "The question to ask Genie about your data"],
        runtime: ToolRuntime[Context, AgentState],
    ) -> Command:
        """Process a natural language question through Databricks Genie."""
        state: AgentState = runtime.state
        tool_call_id: str = runtime.tool_call_id
        context: Context | None = runtime.context

        set_resource_attributes(
            ResourceInfo("genie", genie_room_model.on_behalf_of_user, space_id_str)
        )

        genie_service: GenieServiceBase = _get_genie_service(context)
        session: SessionState = state.get("session", SessionState())

        existing_conversation_id: str | None = (
            session.genie.get_conversation_id(space_id_str)
            if persist_conversation
            else None
        )

        logger.trace(
            "Sending prompt to Genie",
            space_id=space_id_str,
            conversation_id=existing_conversation_id,
            prompt=question[:500] + "..." if len(question) > 500 else question,
        )

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

        if persist_conversation:
            session.genie.update_space(
                space_id=space_id_str,
                conversation_id=current_conversation_id,
                cache_hit=cache_hit,
                cache_key=cache_key,
                last_query=question,
                message_id=cache_result.message_id,
            )

        update: dict[str, Any] = {
            "messages": [
                ToolMessage(
                    _response_to_json_with_cache(genie_response, cache_hit=cache_hit),
                    tool_call_id=tool_call_id,
                )
            ],
        }
        if persist_conversation:
            update["session"] = session

        return Command(update=update)

    # ---- Feedback tool (shares same _get_genie_service) ----

    feedback_name: str = f"{tool_name}_feedback"
    feedback_desc: str = (
        f"Provide feedback on results from {tool_name}. "
        "Call with rating 'negative' when results are wrong, empty, or missing "
        "expected columns to invalidate the cached result so the next query "
        "gets fresh data from Genie."
    )

    from dao_ai.genie import GenieFeedbackRating

    @tool(name_or_callable=feedback_name, description=feedback_desc)
    def genie_feedback_tool(
        rating: Annotated[
            str, "'negative' if the result was wrong/unhelpful, 'positive' if correct"
        ],
        runtime: ToolRuntime[Context, AgentState],
    ) -> Command:
        """Send feedback and invalidate cached results for a Genie query."""
        state: AgentState = runtime.state
        tool_call_id: str = runtime.tool_call_id
        context: Context | None = runtime.context

        genie_service: GenieServiceBase = _get_genie_service(context)
        session: SessionState = state.get("session", SessionState())

        space_state = session.genie.spaces.get(space_id_str)
        if space_state is None:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            json.dumps(
                                {
                                    "status": "error",
                                    "detail": "No prior query found for this Genie space.",
                                }
                            ),
                            tool_call_id=tool_call_id,
                        )
                    ],
                }
            )

        conversation_id: str = space_state.conversation_id
        message_id: str | None = space_state.message_id
        was_cache_hit: bool = space_state.cache_hit
        last_query: str | None = space_state.last_query

        feedback_rating: GenieFeedbackRating = (
            GenieFeedbackRating.NEGATIVE
            if rating.lower().strip() == "negative"
            else GenieFeedbackRating.POSITIVE
        )

        genie_service.send_feedback(
            conversation_id=conversation_id,
            rating=feedback_rating,
            message_id=message_id,
            was_cache_hit=was_cache_hit,
        )

        invalidated: bool = False
        if last_query and feedback_rating == GenieFeedbackRating.NEGATIVE:
            invalidated = genie_service.invalidate(last_query, conversation_id)

        logger.info(
            "Genie feedback sent",
            space_id=space_id_str,
            conversation_id=conversation_id,
            rating=rating,
            cache_invalidated=invalidated,
            last_query=last_query[:80] if last_query else None,
        )

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        json.dumps(
                            {
                                "status": "ok",
                                "rating": rating,
                                "cache_invalidated": invalidated,
                                "detail": (
                                    "Cache entry invalidated. Your next query will get fresh results."
                                    if invalidated
                                    else "Feedback recorded."
                                ),
                            }
                        ),
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    return GenieToolkit(tools=[genie_tool, genie_feedback_tool])
