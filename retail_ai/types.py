from typing import Any, Callable, TypeAlias

from retail_ai.state import AgentState, AgentConfig


AgentCallable: TypeAlias = Callable[[AgentState, AgentConfig], dict[str, Any]]
