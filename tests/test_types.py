import pytest

from retail_ai.config import AppConfig
from retail_ai.state import AgentConfig, AgentState
from retail_ai.types import AgentCallable


@pytest.mark.key("unit")
def test_agent_callable_type_signature(config: AppConfig) -> None:
    """Test that AgentCallable type works as expected."""

    def sample_agent_function(state: AgentState, config: AgentConfig) -> dict[str, str]:
        """Sample function that matches AgentCallable signature."""
        return {"route": "test_route", "context": "test_context"}

    # This should work without type errors
    agent_func: AgentCallable = sample_agent_function

    # Test that we can call it
    test_state = AgentState(messages=[])
    test_config = AgentConfig()

    result = agent_func(test_state, test_config)

    assert isinstance(result, dict)
    assert "route" in result
    assert result["route"] == "test_route"


@pytest.mark.key("unit")
def test_agent_callable_return_type_flexibility() -> None:
    """Test that AgentCallable allows flexible return types."""

    def agent_with_list_return(
        state: AgentState, config: AgentConfig
    ) -> dict[str, list]:
        """Agent function that returns a list in the dict."""
        return {"messages": ["new message"]}

    def agent_with_mixed_return(
        state: AgentState, config: AgentConfig
    ) -> dict[str, any]:
        """Agent function that returns mixed types."""
        return {
            "route": "search",
            "messages": ["msg1", "msg2"],
            "context": {"key": "value"},
            "count": 42,
        }

    # Both should be valid AgentCallable types
    agent1: AgentCallable = agent_with_list_return
    agent2: AgentCallable = agent_with_mixed_return

    test_state = AgentState(messages=[])
    test_config = AgentConfig()

    result1 = agent1(test_state, test_config)
    result2 = agent2(test_state, test_config)

    assert isinstance(result1, dict)
    assert isinstance(result2, dict)
    assert "messages" in result1
    assert len(result2) == 4
