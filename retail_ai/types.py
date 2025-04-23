from typing import Any, Callable, TypeAlias

from retail_ai.state import AgentConfig, AgentState

AgentCallable: TypeAlias = Callable[[AgentState, AgentConfig], dict[str, Any]]
