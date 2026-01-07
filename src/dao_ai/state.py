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
from typing import Any, Optional

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
    ) -> None:
        """Update or create state for a Genie space."""
        self.spaces[space_id] = GenieSpaceState(
            conversation_id=conversation_id,
            cache_hit=cache_hit,
            cache_key=cache_key,
            follow_up_questions=follow_up_questions or [],
            last_query=last_query,
            last_query_time=datetime.now() if last_query else None,
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
    active_agent: NotRequired[str]
    is_valid: NotRequired[bool]
    message_error: NotRequired[str]
    session: NotRequired[SessionState]
    structured_response: NotRequired[Any]


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
