"""
Base classes and types for DAO AI middleware.

This module re-exports LangChain's middleware types for convenience.
Use LangChainAgentMiddleware directly with DAO AI's state and context types.

Example:
    from langchain.agents.middleware import AgentMiddleware
    from dao_ai.state import AgentState, Context
    from langgraph.runtime import Runtime

    class MyMiddleware(AgentMiddleware[AgentState, Context]):
        def before_model(
            self,
            state: AgentState,
            runtime: Runtime[Context]
        ) -> dict[str, Any] | None:
            print(f"About to call model with {len(state['messages'])} messages")
            return None
"""

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelRequest,
    after_agent,
    after_model,
    before_agent,
    before_model,
    dynamic_prompt,
    wrap_model_call,
    wrap_tool_call,
)
from langchain.agents.middleware.types import ModelResponse

# Re-export LangChain types for convenience
__all__ = [
    # Base middleware class
    "AgentMiddleware",
    # Types
    "ModelRequest",
    "ModelResponse",
    # Decorators
    "before_agent",
    "before_model",
    "after_agent",
    "after_model",
    "wrap_model_call",
    "wrap_tool_call",
    "dynamic_prompt",
]
