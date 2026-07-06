"""
State definitions for DAO AI agents.

This module defines the state schemas used by DAO AI agents,
compatible with both LangChain v1's create_agent and LangGraph's StateGraph.

State Schema:
- AgentState: Primary state schema for all agent operations
- Context: Runtime context passed via ToolRuntime[Context] or Runtime[Context]
- GenieSpaceState: Per-space state for Genie conversations
- SessionState: Accumulated state that flows between requests
"""

from datetime import datetime
from typing import Annotated, Any, Optional

from langgraph.graph import MessagesState
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import NotRequired


class GenieSpaceState(BaseModel):
    """State for a single Genie space/conversation.

    This tracks the conversation state and metadata for a Genie space,
    allowing multi-turn conversations and caching information to be preserved.
    """

    conversation_id: str = Field(description="Genie conversation ID for this space")
    cache_hit: bool = Field(
        default=False, description="Whether the last query was a cache hit"
    )
    cache_key: Optional[str] = Field(default=None, description="Cache key if cached")
    follow_up_questions: list[str] = Field(
        default_factory=list, description="Suggested follow-up questions from Genie"
    )
    last_query: Optional[str] = Field(
        default=None, description="The last query sent to Genie"
    )
    last_query_time: Optional[datetime] = Field(
        default=None, description="When the last query was made"
    )
    message_id: Optional[str] = Field(
        default=None, description="Message ID from the last Genie response"
    )


class GenieState(BaseModel):
    """State for all Genie spaces.

    Maps space_id to GenieSpaceState for each Genie space the user has interacted with.
    """

    spaces: dict[str, GenieSpaceState] = Field(
        default_factory=dict, description="Map of space_id to space state"
    )

    def get_conversation_id(self, space_id: str) -> Optional[str]:
        """Get conversation ID for a space, if it exists."""
        if space_id in self.spaces:
            return self.spaces[space_id].conversation_id
        return None

    def update_space(
        self,
        space_id: str,
        conversation_id: str,
        cache_hit: bool = False,
        cache_key: Optional[str] = None,
        follow_up_questions: Optional[list[str]] = None,
        last_query: Optional[str] = None,
        message_id: Optional[str] = None,
    ) -> None:
        """Update or create state for a Genie space."""
        self.spaces[space_id] = GenieSpaceState(
            conversation_id=conversation_id,
            cache_hit=cache_hit,
            cache_key=cache_key,
            follow_up_questions=follow_up_questions or [],
            last_query=last_query,
            last_query_time=datetime.now() if last_query else None,
            message_id=message_id,
        )


class SessionState(BaseModel):
    """Accumulated state that flows between requests.

    This is the "paste from previous output" portion of the request.
    Users can copy the session from custom_outputs and paste it back
    as custom_inputs.session to restore state.
    """

    genie: GenieState = Field(
        default_factory=GenieState, description="Genie conversation state per space"
    )

    # Future: Add other stateful tool state here
    # other_tool_state: OtherToolState = Field(default_factory=OtherToolState)


def _merge_basemodel(current: BaseModel, new: BaseModel) -> BaseModel:
    """Recursively merge two same-typed BaseModel instances.

    Rules:
    - Nested BaseModel of the same type: recurse.
    - Dict-valued fields: union with ``new`` winning on key conflict.
    - All other fields: ``new`` wins (last-update semantics).

    Different runtime types short-circuit to ``new``, since merging unlike
    schemas can't be done coherently.
    """
    if type(current) is not type(new):
        return new

    updates: dict[str, Any] = {}
    for field_name in type(current).model_fields:
        cur_val = getattr(current, field_name)
        new_val = getattr(new, field_name)
        if (
            isinstance(cur_val, BaseModel)
            and isinstance(new_val, BaseModel)
            and type(cur_val) is type(new_val)
        ):
            updates[field_name] = _merge_basemodel(cur_val, new_val)
        elif isinstance(cur_val, dict) and isinstance(new_val, dict):
            updates[field_name] = {**cur_val, **new_val}
        else:
            updates[field_name] = new_val
    return type(current)(**updates)


def merge_session(current: SessionState, new: SessionState) -> SessionState:
    """Reducer that merges SessionState values from concurrent tool updates.

    When multiple tools (e.g., parallel Genie calls) write to ``session`` in
    the same LangGraph step, the default ``LastValue`` channel would raise
    ``InvalidUpdateError``. This reducer recursively walks ``SessionState``,
    union-merging dict fields and recursing into nested ``BaseModel`` fields,
    so newly added stateful tool state automatically participates in merging
    without bespoke reducer code.
    """
    merged = _merge_basemodel(current, new)
    return merged  # type: ignore[return-value]


def concat_parallel_dispatches(
    current: Optional[list[str]], new: Optional[list[str]]
) -> list[str]:
    """Reducer that concatenates parallel dispatch targets across concurrent writes.

    Parallel fan-out handoff tools each write ``__parallel_dispatches__``
    with a single target name via their ``Command.update``. LangGraph's
    default ``LastValue`` channel would only keep the last, but N parallel
    tools in one turn must ALL be recorded so the source-agent wrapper
    can dispatch a ``Send`` per fired sibling.

    We use a dedicated reducer rather than piggybacking on ``add_messages``
    because the ToolMessage content gets stripped by
    ``extract_agent_response`` before the source wrapper sees it — a
    first-class state field survives regardless of output_mode.

    The wrapper is responsible for clearing this field once it has
    dispatched, so the next turn starts empty.
    """
    if not new:
        return list(current or [])
    return list(current or []) + list(new)


def last_active_agent(current: Optional[str], new: Optional[str]) -> Optional[str]:
    """Reducer that tolerates concurrent writes to ``active_agent``.

    In swarm configs that mix deterministic edges with agentic handoff
    tools (e.g. ``deterministic_handoff_pattern.yaml``), an agentic
    ``Command(goto=X, graph=PARENT)`` and the parent graph's static
    ``add_edge`` from the same source can both fire in one step. Each
    writes ``active_agent``, and the default ``LastValue`` channel raises
    ``InvalidUpdateError: At key 'active_agent': Can receive only one
    value per step.``

    The Command path is the source of truth (it carries the LLM's chosen
    target); the static-edge update is bookkeeping. We resolve concurrent
    writes by preferring whichever value is non-None, falling back to the
    new value.
    """
    if new is not None:
        return new
    return current


class AgentState(MessagesState, total=False):
    """
    Primary state schema for DAO AI agents.

    Extends MessagesState to include the messages channel with proper
    add_messages reducer, plus additional fields for DAO AI functionality.

    Used for:
    - state_schema in create_agent calls
    - state_schema in StateGraph for orchestration
    - Type parameter in ToolRuntime[Context, AgentState]
    - Type parameter in AgentMiddleware[AgentState, Context]
    - API input/output contracts

    Fields:
        messages: Conversation history with add_messages reducer (from MessagesState)
        context: Short/long term memory context
        active_agent: Name of currently active agent in multi-agent workflows
        is_valid: Message validation status
        message_error: Error message if validation failed
        session: Accumulated session state (genie conversations, etc.)
        structured_response: Structured output from response_format (populated by LangChain)
    """

    context: NotRequired[str]
    active_agent: NotRequired[Annotated[str, last_active_agent]]
    is_valid: NotRequired[bool]
    message_error: NotRequired[str]
    session: NotRequired[Annotated[SessionState, merge_session]]
    structured_response: NotRequired[Any]
    # Ephemeral field used by the parallel fan-out feature to collect
    # target agent names from N parallel-handoff tools invoked in one LLM
    # turn. The source-agent wrapper reads this to build the fan-out
    # ``Send`` list, then clears it. See ``concat_parallel_dispatches``.
    parallel_dispatches: NotRequired[
        Annotated[list[str], concat_parallel_dispatches]
    ]


class Context(BaseModel):
    """
    Runtime context for DAO AI agents.

    This is passed to tools and middleware via the runtime parameter.
    Access via ToolRuntime[Context] in tools or Runtime[Context] in middleware.

    Additional fields beyond user_id and thread_id can be added dynamically
    and will be available as top-level attributes on the context object.
    These fields are:
    - Used as template parameters in prompts (all fields are applied)
    - Validated by middleware (check for specific fields like "store_num")
    - Accessible as direct attributes (e.g., context.store_num)

    Example:
        @tool
        def my_tool(runtime: ToolRuntime[Context]) -> str:
            user_id = runtime.context.user_id
            store_num = runtime.context.store_num  # Direct attribute access
            return f"Hello, {user_id} at store {store_num}!"

        class MyMiddleware(AgentMiddleware[AgentState, Context]):
            def before_model(
                self,
                state: AgentState,
                runtime: Runtime[Context]
            ) -> dict[str, Any] | None:
                user_id = runtime.context.user_id
                store_num = getattr(runtime.context, "store_num", None)
                return None
    """

    model_config = ConfigDict(
        extra="allow"
    )  # Allow extra fields as top-level attributes

    user_id: str | None = None
    thread_id: str | None = None
    headers: dict[str, Any] | None = None

    @classmethod
    def from_runnable_config(cls, config: dict[str, Any]) -> "Context":
        """
        Create Context from LangChain RunnableConfig.

        This method is called by LangChain when context_schema is provided to create_agent.
        It extracts the 'configurable' dict from the config and uses it to instantiate Context.
        """
        configurable = config.get("configurable", {})
        return cls(**configurable)
