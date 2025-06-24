import pytest
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.base import RunnableLike

from dao_ai.config import AppConfig
from dao_ai.state import SharedState


@pytest.mark.unit
def test_agent_callable_type_signature(config: AppConfig) -> None:
    """Test that RunnableLike type works as expected."""

    def sample_agent_function(
        state: SharedState, config: RunnableConfig
    ) -> dict[str, str]:
        """Sample function that matches RunnableLike signature."""
        return {"route": "test_route", "context": "test_context"}

    # This should work without type errors
    agent_func: RunnableLike = sample_agent_function

    # Test that we can call it
    test_state = SharedState(messages=[])
    test_config = RunnableConfig()

    result = agent_func(test_state, test_config)

    assert isinstance(result, dict)
    assert "route" in result
    assert result["route"] == "test_route"


@pytest.mark.unit
def test_agent_callable_return_type_flexibility() -> None:
    """Test that RunnableLike allows flexible return types."""

    def agent_with_list_return(
        state: SharedState, config: RunnableConfig
    ) -> dict[str, list]:
        """Agent function that returns a list in the dict."""
        return {"messages": ["new message"]}

    def agent_with_mixed_return(
        state: SharedState, config: RunnableConfig
    ) -> dict[str, any]:
        """Agent function that returns mixed types."""
        return {
            "route": "search",
            "messages": ["msg1", "msg2"],
            "context": {"key": "value"},
            "count": 42,
        }

    # Both should be valid RunnableLike types
    agent1: RunnableLike = agent_with_list_return
    agent2: RunnableLike = agent_with_mixed_return

    test_state = SharedState(messages=[])
    test_config = RunnableConfig()

    result1 = agent1(test_state, test_config)
    result2 = agent2(test_state, test_config)

    assert isinstance(result1, dict)
    assert isinstance(result2, dict)
    assert "messages" in result1
    assert len(result2) == 4
