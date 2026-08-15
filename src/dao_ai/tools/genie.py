"""
Genie tool factory for natural language queries to databases.

This module exposes a single unified factory:

- :func:`create_genie_tool` — returns either a single uncached Genie query tool
  or a :class:`GenieToolkit` (query + feedback tools) depending on whether any
  caching is configured or feedback is explicitly enabled.

For backward compatibility, :func:`create_genie_toolkit` is preserved as a thin
shim that always returns a :class:`GenieToolkit` (equivalent to calling
:func:`create_genie_tool` with ``enable_feedback=True``).

For the core Genie service and cache implementations, see:
- dao_ai.genie: GenieService, GenieServiceBase
- dao_ai.genie.cache: LRUCacheService, PostgresContextAwareGenieService, InMemoryContextAwareGenieService
"""

import hashlib
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

    Returned by :func:`create_genie_tool` when caching is configured or
    ``enable_feedback=True``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    tools: list[BaseTool] = Field(default_factory=list)

    def get_tools(self) -> list[BaseTool]:
        return self.tools


# ---------------------------------------------------------------------------
# Cache hit tracker (Circuit Breaker pattern)
# ---------------------------------------------------------------------------


class _CacheHitTracker:
    """Tracks consecutive identical cache hits per Genie space.

    Implements the Circuit Breaker pattern: after N consecutive cache hits
    returning the same SQL query, the caller can auto-invalidate and re-query
    Genie directly, breaking out of stale cache loops.

    The tracker is closure-scoped (not stored in session state) to avoid
    serialization overhead and ``merge_session`` conflicts.
    """

    def __init__(self, max_history: int = 20) -> None:
        self._history: dict[str, list[str | None]] = {}
        self._max_history = max_history

    def record(self, space_id: str, cache_hit: bool, sql_query: str | None) -> int:
        """Record a query result and return the consecutive identical cache hit count.

        Args:
            space_id: The Genie space ID.
            cache_hit: Whether the result was served from cache.
            sql_query: The SQL query string from the response (used for hashing).

        Returns:
            Number of consecutive cache hits with the same SQL hash.
            Returns 0 for cache misses.
        """
        sql_hash: str | None = (
            hashlib.md5(sql_query.encode()).hexdigest() if sql_query else None
        )
        history: list[str | None] = self._history.setdefault(space_id, [])
        entry: str | None = sql_hash if cache_hit else None
        history.append(entry)

        if len(history) > self._max_history:
            history[:] = history[-self._max_history :]
            logger.trace(
                "Cache hit tracker history trimmed",
                space_id=space_id,
                trimmed_to=self._max_history,
            )

        count: int = self._consecutive_count(history)
        logger.trace(
            "Cache hit tracker recorded",
            space_id=space_id,
            cache_hit=cache_hit,
            sql_hash=sql_hash[:12] if sql_hash else None,
            consecutive=count,
        )
        return count

    @staticmethod
    def _consecutive_count(history: list[str | None]) -> int:
        """Count consecutive identical entries from the end of the history."""
        if not history or history[-1] is None:
            return 0
        target: str | None = history[-1]
        count: int = 0
        for entry in reversed(history):
            if entry == target:
                count += 1
            else:
                break
        return count


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
    response: GenieResponse,
    cache_hit: bool = False,
    consecutive_cache_hits: int = 0,
    feedback_tool_name: str | None = None,
    auto_invalidated: bool = False,
) -> str:
    """Convert GenieResponse to JSON string, including cache metadata.

    Args:
        response: The GenieResponse to serialize.
        cache_hit: Whether the result was served from cache.
        consecutive_cache_hits: How many consecutive identical cache hits
            have occurred. Included in response when > 1 to signal the LLM.
        feedback_tool_name: Name of the feedback tool (for cache-hit hints).
        auto_invalidated: Whether the result was auto-invalidated via
            the circuit breaker and re-queried from Genie.
    """
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

    if cache_hit and feedback_tool_name:
        data["_hint"] = (
            "This result was served from cache. If it doesn't match your question, "
            f"call {feedback_tool_name} with rating='negative' then re-ask."
        )

    if consecutive_cache_hits > 1:
        data["consecutive_cache_hits"] = consecutive_cache_hits

    if auto_invalidated:
        data["auto_invalidated"] = True
        data["auto_invalidation_reason"] = (
            "Circuit breaker: consecutive identical cache hits exceeded threshold"
        )

    return json.dumps(data)


_DEFAULT_DESCRIPTION_TEMPLATE: str = dedent("""
    This tool lets you have a conversation and chat with tabular data{topic}. You should ask
    questions about the data and the tool will try to answer them.
    Please ask simple clear questions that can be answer by sql queries. If you need to do statistics or other forms of testing defer to using another tool.
    Try to ask for aggregations on the data and ask very simple questions.
    Prefer to call this tool multiple times rather than asking a complex question.
    """)


def _default_description(topic: str | None) -> str:
    """Last-resort tool description, naming ``topic`` when one is known.

    Reached only when neither the tool nor the Genie space supplies a
    description. The topic slot is filled from the space's display name; with
    no name the clause is dropped entirely rather than left as an unfilled
    placeholder.
    """
    return _DEFAULT_DESCRIPTION_TEMPLATE.format(
        topic=f" about {topic}" if topic else ""
    )


_FUNCTION_DOCS: str = """

Args:
question (str): The question to ask to ask Genie about your data. Ask simple, clear questions about your tabular data. For complex analysis, ask multiple simple questions rather than one complex question.

Returns:
GenieResponse: A response object containing the conversation ID and result from Genie."""

# Question-argument annotations. The default nudges the model to shape the
# question; the preserve variant tells it to pass the user's own wording through
# untouched for the portion routed to THIS tool — while still allowing a
# genuinely multi-tool request to be split across tools.
_QUESTION_ARG_DEFAULT: str = "The question to ask Genie about your data"
_QUESTION_ARG_PRESERVE: str = (
    "The user's question for this tool, word-for-word and complete. Keep every "
    "qualifier, scope, and constraint; do not rephrase, summarize, or trim. If "
    "the request spans several tools, route each part to its tool but send this "
    "part unchanged."
)

# Appended to the tool description when ``preserve_question=True``. Deliberately
# only a pointer: this clause and the question-argument annotation above land in
# the *same* serialized tool schema, re-sent on every LLM call for the whole
# conversation, so restating the full instruction here bought redundancy rather
# than reinforcement.
_PRESERVE_QUESTION_TOOL_CLAUSE: str = (
    "Pass this tool the user's own words for its part of the request — complete "
    "and unedited (see the `question` argument)."
)


_MAX_EXAMPLE_QUESTIONS: int = 10

_EXAMPLE_QUESTIONS_HEADER: str = "Example questions this tool can answer:"


def _example_questions(genie_room: GenieRoomModel | None) -> list[str]:
    """Few-shot example questions to advertise, in preference order.

    Prefers the space's ``sample_questions``, falling back to the *questions*
    of its trusted ``example_sqls`` pairs. SQL bodies are deliberately never
    included: this text is read by the calling LLM to decide routing, and Genie
    already holds the SQL on its own space, so the bodies would cost hundreds
    of tokens on every LLM call for no routing gain.
    """
    if genie_room is None:
        return []

    candidates: list[Any] = list(genie_room.sample_questions or [])
    if not candidates:
        candidates = [
            getattr(example, "question", None)
            for example in genie_room.example_sqls or []
        ]

    questions: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        question: str = candidate.strip()
        if question and question not in questions:
            questions.append(question)
        if len(questions) == _MAX_EXAMPLE_QUESTIONS:
            break
    return questions


def _build_tool_description(
    description: str | None,
    preserve_question: bool,
    genie_room: GenieRoomModel | None = None,
    include_example_questions: bool = False,
) -> str:
    """Assemble the ``@tool`` description string.

    Base text precedence: the caller-supplied ``description``, else the Genie
    space's own ``description``, else :func:`_default_description`. Falling back
    to the space description is what lets a supervisor tell two Genie tools
    apart — with neither set, every description-less Genie tool in a config
    advertises byte-identical, subject-free text and routing becomes a coin
    flip.

    Example questions are appended only when ``include_example_questions`` is
    set: they are prompt surface, so the config author opts in rather than
    inheriting up to ``_MAX_EXAMPLE_QUESTIONS`` of them in every LLM call. The
    base text above is chosen independently, so adding a ``description`` no
    longer silently removes the questions block.

    Then appends the preserve-question clause when ``preserve_question`` is set
    — even for a user-supplied description, so the flag always takes effect —
    and finally the function-args docs. Shared by both tool-builder impls.
    """
    room_description: str | None = genie_room.description if genie_room else None
    room_name: str | None = genie_room.name if genie_room else None
    base: str = (
        description
        if description is not None
        else room_description or _default_description(room_name)
    )

    questions: list[str] = _example_questions(genie_room)
    if not include_example_questions:
        if questions and genie_room is not None:
            # ``name`` is None for a room referenced by bare ``space_id`` until
            # it resolves, so fall back to the id — an unnamed space would make
            # the line unactionable.
            logger.info(
                "Genie space has example questions that are not in the tool "
                "description; set include_example_questions: true to append them "
                "as routing hints",
                space=room_name or value_of(genie_room.space_id),
            )
        questions = []
    if questions:
        base = "\n".join(
            [
                base.rstrip("\n") + "\n",
                _EXAMPLE_QUESTIONS_HEADER,
                *(f"- {q}" for q in questions),
            ]
        )

    if preserve_question:
        base = base + "\n" + _PRESERVE_QUESTION_TOOL_CLAUSE
    return base + _FUNCTION_DOCS


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
# Public factory
# ---------------------------------------------------------------------------


def create_genie_tool(
    genie_room: GenieRoomModel | dict[str, Any],
    name: str | None = None,
    description: str | None = None,
    persist_conversation: bool = True,
    truncate_results: bool = False,
    preserve_question: bool = False,
    include_example_questions: bool = False,
    lru_cache_parameters: GenieLRUCacheParametersModel | dict[str, Any] | None = None,
    context_aware_cache_parameters: (
        GenieContextAwareCacheParametersModel | dict[str, Any] | None
    ) = None,
    in_memory_context_aware_cache_parameters: (
        GenieInMemoryContextAwareCacheParametersModel | dict[str, Any] | None
    ) = None,
    max_consecutive_cache_hits: int | None = None,
    enable_feedback: bool = False,
) -> Callable[..., Command] | GenieToolkit:
    """Create a Genie tool, optionally with caching and a feedback tool.

    Returns a single uncached query tool when no caching is configured and
    ``enable_feedback`` is ``False``. When any cache is configured *or*
    ``enable_feedback=True`` is set, returns a :class:`GenieToolkit` containing:

    * A **query tool** that translates natural-language questions into SQL via
      Databricks Genie, with results served from cache when configured.
    * A **feedback tool** (named ``{name}_feedback``) that lets the LLM
      invalidate a stale or incorrect cached result so the next query goes
      fresh to Genie.

    Both tools share one ``GenieServiceBase`` stack, so invalidation on the
    feedback tool directly clears the query tool's cache.

    When ``max_consecutive_cache_hits`` is set, the toolkit implements a
    **Circuit Breaker** pattern: if the same cached SQL is returned N times
    consecutively, the cache entry is automatically invalidated and a fresh
    query is sent to Genie. This prevents the agent from getting stuck in a
    loop where the LLM normalizes different user phrasings into the same
    tool call, repeatedly hitting the same stale cache entry.

    Args:
        genie_room: GenieRoomModel or dict containing Genie configuration.
        name: Custom tool name visible to the LLM. Defaults to ``"genie_tool"``.
        description: Custom tool description. Defaults to a generic prompt.
        persist_conversation: Persist conversation IDs across calls for multi-turn.
        truncate_results: Truncate large query results.
        preserve_question: When ``True``, instruct the calling LLM (via the tool
            description and the ``question`` argument annotation) to pass the
            user's question through exactly as asked — no rephrasing,
            decomposition, or added qualifiers. Default ``False`` preserves the
            existing behavior.
        include_example_questions: When ``True``, append the Genie space's
            example questions to the tool description as few-shot routing hints.
            Default ``False`` never appends them, whether or not a
            ``description`` was supplied.
        lru_cache_parameters: LRU cache config for fast exact-match SQL caching.
        context_aware_cache_parameters: PostgreSQL/Lakebase context-aware cache config.
        in_memory_context_aware_cache_parameters: In-memory context-aware cache config.
        max_consecutive_cache_hits: Auto-invalidate after this many consecutive
            identical cache hits (Circuit Breaker). ``None`` disables (default).
            Suggested value: ``3``. Only meaningful when caching is configured.
        enable_feedback: Force toolkit mode (with feedback tool) even when no
            cache is configured. Implicitly ``True`` whenever any cache is set.

    Returns:
        A single LangGraph tool callable when no caching and no feedback are
        configured. Otherwise a :class:`GenieToolkit` bundling the query and
        feedback tools.
    """
    has_cache: bool = any(
        [
            lru_cache_parameters is not None,
            context_aware_cache_parameters is not None,
            in_memory_context_aware_cache_parameters is not None,
        ]
    )
    toolkit_mode: bool = has_cache or enable_feedback

    if not toolkit_mode:
        return _create_simple_genie_tool(
            genie_room=genie_room,
            name=name,
            description=description,
            persist_conversation=persist_conversation,
            truncate_results=truncate_results,
            preserve_question=preserve_question,
            include_example_questions=include_example_questions,
        )

    return _create_genie_toolkit_impl(
        genie_room=genie_room,
        name=name,
        description=description,
        persist_conversation=persist_conversation,
        truncate_results=truncate_results,
        preserve_question=preserve_question,
        include_example_questions=include_example_questions,
        lru_cache_parameters=lru_cache_parameters,
        context_aware_cache_parameters=context_aware_cache_parameters,
        in_memory_context_aware_cache_parameters=in_memory_context_aware_cache_parameters,
        max_consecutive_cache_hits=max_consecutive_cache_hits,
    )


def create_genie_toolkit(
    genie_room: GenieRoomModel | dict[str, Any],
    name: str | None = None,
    description: str | None = None,
    persist_conversation: bool = True,
    truncate_results: bool = False,
    preserve_question: bool = False,
    include_example_questions: bool = False,
    lru_cache_parameters: GenieLRUCacheParametersModel | dict[str, Any] | None = None,
    context_aware_cache_parameters: (
        GenieContextAwareCacheParametersModel | dict[str, Any] | None
    ) = None,
    in_memory_context_aware_cache_parameters: (
        GenieInMemoryContextAwareCacheParametersModel | dict[str, Any] | None
    ) = None,
    max_consecutive_cache_hits: int | None = None,
) -> GenieToolkit:
    """Backward-compatible shim. Prefer :func:`create_genie_tool` directly.

    Always returns a :class:`GenieToolkit` (with the implicit feedback tool),
    matching the historical behavior of this factory. Equivalent to calling
    ``create_genie_tool(..., enable_feedback=True)``.
    """
    result = create_genie_tool(
        genie_room=genie_room,
        name=name,
        description=description,
        persist_conversation=persist_conversation,
        truncate_results=truncate_results,
        preserve_question=preserve_question,
        include_example_questions=include_example_questions,
        lru_cache_parameters=lru_cache_parameters,
        context_aware_cache_parameters=context_aware_cache_parameters,
        in_memory_context_aware_cache_parameters=in_memory_context_aware_cache_parameters,
        max_consecutive_cache_hits=max_consecutive_cache_hits,
        enable_feedback=True,
    )
    assert isinstance(result, GenieToolkit)
    return result


# ---------------------------------------------------------------------------
# Internal builders
# ---------------------------------------------------------------------------


def _create_simple_genie_tool(
    genie_room: GenieRoomModel | dict[str, Any],
    name: str | None,
    description: str | None,
    persist_conversation: bool,
    truncate_results: bool,
    preserve_question: bool = False,
    include_example_questions: bool = False,
) -> Callable[..., Command]:
    """Build the uncached single-tool variant (no cache, no feedback tool)."""
    logger.debug(
        "Creating Genie tool (uncached)",
        genie_room_type=type(genie_room).__name__,
        persist_conversation=persist_conversation,
        name=name,
        preserve_question=preserve_question,
        include_example_questions=include_example_questions,
    )

    genie_room_model, space_id_str = _resolve_genie_room(genie_room)

    tool_name: str = name if name is not None else "genie_tool"
    tool_description: str = _build_tool_description(
        description, preserve_question, genie_room_model, include_example_questions
    )
    question_desc: str = (
        _QUESTION_ARG_PRESERVE if preserve_question else _QUESTION_ARG_DEFAULT
    )

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
        question: Annotated[str, question_desc],
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


def _create_genie_toolkit_impl(
    genie_room: GenieRoomModel | dict[str, Any],
    name: str | None,
    description: str | None,
    persist_conversation: bool,
    truncate_results: bool,
    lru_cache_parameters: GenieLRUCacheParametersModel | dict[str, Any] | None,
    context_aware_cache_parameters: (
        GenieContextAwareCacheParametersModel | dict[str, Any] | None
    ),
    in_memory_context_aware_cache_parameters: (
        GenieInMemoryContextAwareCacheParametersModel | dict[str, Any] | None
    ),
    max_consecutive_cache_hits: int | None,
    preserve_question: bool = False,
    include_example_questions: bool = False,
) -> GenieToolkit:
    """Build the cached toolkit variant (query + feedback, optional cache layers)."""
    logger.debug(
        "Creating Genie toolkit",
        genie_room_type=type(genie_room).__name__,
        persist_conversation=persist_conversation,
        truncate_results=truncate_results,
        name=name,
        preserve_question=preserve_question,
        include_example_questions=include_example_questions,
        has_lru_cache=lru_cache_parameters is not None,
        has_context_aware_cache=context_aware_cache_parameters is not None,
        has_in_memory_context_aware_cache=in_memory_context_aware_cache_parameters
        is not None,
        max_consecutive_cache_hits=max_consecutive_cache_hits,
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
    tool_description: str = _build_tool_description(
        description, preserve_question, genie_room_model, include_example_questions
    )
    question_desc: str = (
        _QUESTION_ARG_PRESERVE if preserve_question else _QUESTION_ARG_DEFAULT
    )

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

    # ---- Cache hit tracker (Circuit Breaker) ----

    tracker: _CacheHitTracker = _CacheHitTracker()
    logger.debug(
        "Cache hit tracker initialized",
        tool=tool_name,
        max_consecutive_cache_hits=max_consecutive_cache_hits,
    )

    # ---- Query tool ----

    feedback_name: str = f"{tool_name}_feedback"

    @tool(name_or_callable=tool_name, description=tool_description)
    def genie_tool(
        question: Annotated[str, question_desc],
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
        auto_invalidated: bool = False

        consecutive: int = tracker.record(space_id_str, cache_hit, genie_response.query)

        if (
            max_consecutive_cache_hits is not None
            and cache_hit
            and consecutive >= max_consecutive_cache_hits
        ):
            logger.warning(
                "Circuit breaker: auto-invalidating after consecutive identical cache hits",
                space_id=space_id_str,
                consecutive=consecutive,
                threshold=max_consecutive_cache_hits,
                sql=genie_response.query[:80] if genie_response.query else None,
            )
            invalidated: bool = genie_service.invalidate(
                question, existing_conversation_id
            )
            logger.info(
                "Circuit breaker invalidation result",
                space_id=space_id_str,
                invalidated=invalidated,
                question=question[:80],
            )
            cache_result = genie_service.ask_question(
                question, conversation_id=existing_conversation_id
            )
            genie_response = cache_result.response
            cache_hit = False
            cache_key = None
            auto_invalidated = True
            tracker.record(space_id_str, False, genie_response.query)
            logger.info(
                "Circuit breaker fresh query result",
                space_id=space_id_str,
                fresh_sql=genie_response.query[:80] if genie_response.query else None,
                has_result=genie_response.result is not None,
            )
            logger.debug(
                "Circuit breaker tracker reset",
                space_id=space_id_str,
            )

        current_conversation_id: str = genie_response.conversation_id
        logger.debug(
            "Genie question answered",
            space_id=space_id_str,
            conversation_id=current_conversation_id,
            cache_hit=cache_hit,
            cache_key=cache_key,
            auto_invalidated=auto_invalidated,
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
                    _response_to_json_with_cache(
                        genie_response,
                        cache_hit=cache_hit,
                        consecutive_cache_hits=consecutive,
                        feedback_tool_name=feedback_name,
                        auto_invalidated=auto_invalidated,
                    ),
                    tool_call_id=tool_call_id,
                )
            ],
        }
        if persist_conversation:
            update["session"] = session

        return Command(update=update)

    # ---- Feedback tool (shares same _get_genie_service) ----

    feedback_desc: str = (
        f"Provide feedback on results from {tool_name}. "
        "You MUST call this tool with rating 'negative' when:\n"
        "1. Results are wrong, empty, or missing expected columns\n"
        "2. You received a cache_hit=true response that doesn't answer the question\n"
        "3. You've asked similar questions multiple times and keep getting the same incorrect data\n"
        "After calling with 'negative', re-ask your question to get fresh results from Genie."
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
