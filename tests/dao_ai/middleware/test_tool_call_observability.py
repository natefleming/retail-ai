"""
Unit tests for ToolCallObservabilityMiddleware.

These tests verify that:
1. The middleware correctly tracks tool call patterns
2. Parallel vs sequential calls are identified correctly
3. Statistics are computed correctly
"""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from dao_ai.middleware.tool_call_observability import (
    ToolCallObservabilityMiddleware,
    create_tool_call_observability_middleware,
)
from dao_ai.state import AgentState, Context


class TestToolCallObservabilityMiddleware:
    """Tests for ToolCallObservabilityMiddleware."""

    @pytest.mark.unit
    def test_middleware_creation(self) -> None:
        """Test that middleware can be created with default parameters."""
        middleware = ToolCallObservabilityMiddleware()

        assert middleware.log_level == "INFO"
        assert middleware.include_args is False
        assert middleware.track_timing is True

    @pytest.mark.unit
    def test_middleware_creation_with_custom_params(self) -> None:
        """Test that middleware can be created with custom parameters."""
        middleware = ToolCallObservabilityMiddleware(
            log_level="DEBUG",
            include_args=True,
            track_timing=False,
        )

        assert middleware.log_level == "DEBUG"
        assert middleware.include_args is True
        assert middleware.track_timing is False

    @pytest.mark.unit
    def test_factory_function(self) -> None:
        """Test the factory function creates middleware correctly."""
        middleware = create_tool_call_observability_middleware(
            log_level="WARNING",
            include_args=True,
            track_timing=True,
        )

        assert isinstance(middleware, ToolCallObservabilityMiddleware)
        assert middleware.log_level == "WARNING"
        assert middleware.include_args is True

    @pytest.mark.unit
    def test_before_agent_resets_state(self) -> None:
        """Test that before_agent resets all tracking state."""
        middleware = ToolCallObservabilityMiddleware()

        # Simulate some previous state
        middleware._total_model_calls = 5
        middleware._total_tool_calls = 10
        middleware._parallel_batches = 2
        middleware._sequential_calls = 3

        # Mock runtime
        runtime = MagicMock()
        runtime.context = Context(thread_id="test", user_id="test_user")

        # Call before_agent
        state: AgentState = {"messages": []}
        result = middleware.before_agent(state, runtime)

        # Verify state was reset
        assert middleware._total_model_calls == 0
        assert middleware._total_tool_calls == 0
        assert middleware._parallel_batches == 0
        assert middleware._sequential_calls == 0
        assert middleware._run_start_time is not None
        assert result is None

    @pytest.mark.unit
    def test_after_model_detects_parallel_calls(self) -> None:
        """Test that after_model correctly detects parallel tool calls."""
        middleware = ToolCallObservabilityMiddleware()

        # Create an AIMessage with multiple tool calls
        ai_message = AIMessage(
            content="",
            tool_calls=[
                {"name": "tool_a", "args": {"x": 1}, "id": "1"},
                {"name": "tool_b", "args": {"y": 2}, "id": "2"},
                {"name": "tool_c", "args": {"z": 3}, "id": "3"},
            ],
        )

        state: AgentState = {"messages": [ai_message]}
        runtime = MagicMock()
        runtime.context = Context(thread_id="test", user_id="test_user")

        # Call after_model
        result = middleware.after_model(state, runtime)

        # Verify parallel detection
        assert middleware._total_model_calls == 1
        assert middleware._total_tool_calls == 3
        assert middleware._parallel_batches == 1
        assert middleware._sequential_calls == 0
        assert result is None

    @pytest.mark.unit
    def test_after_model_detects_sequential_calls(self) -> None:
        """Test that after_model correctly detects sequential tool calls."""
        middleware = ToolCallObservabilityMiddleware()

        # Create an AIMessage with a single tool call
        ai_message = AIMessage(
            content="",
            tool_calls=[
                {"name": "tool_a", "args": {"x": 1}, "id": "1"},
            ],
        )

        state: AgentState = {"messages": [ai_message]}
        runtime = MagicMock()
        runtime.context = Context(thread_id="test", user_id="test_user")

        # Call after_model
        result = middleware.after_model(state, runtime)

        # Verify sequential detection
        assert middleware._total_model_calls == 1
        assert middleware._total_tool_calls == 1
        assert middleware._parallel_batches == 0
        assert middleware._sequential_calls == 1
        assert result is None

    @pytest.mark.unit
    def test_after_model_no_tool_calls(self) -> None:
        """Test that after_model handles messages without tool calls."""
        middleware = ToolCallObservabilityMiddleware()

        # Create an AIMessage without tool calls
        ai_message = AIMessage(content="Just a regular response")

        state: AgentState = {"messages": [ai_message]}
        runtime = MagicMock()
        runtime.context = Context(thread_id="test", user_id="test_user")

        # Call after_model
        result = middleware.after_model(state, runtime)

        # Verify no tool calls counted
        assert middleware._total_model_calls == 1
        assert middleware._total_tool_calls == 0
        assert middleware._parallel_batches == 0
        assert middleware._sequential_calls == 0
        assert result is None

    @pytest.mark.unit
    def test_after_model_with_human_message(self) -> None:
        """Test that after_model ignores non-AI messages."""
        middleware = ToolCallObservabilityMiddleware()

        # Create a HumanMessage (should be ignored)
        human_message = HumanMessage(content="Hello")

        state: AgentState = {"messages": [human_message]}
        runtime = MagicMock()
        runtime.context = Context(thread_id="test", user_id="test_user")

        # Call after_model
        result = middleware.after_model(state, runtime)

        # Verify nothing counted for human messages
        assert middleware._total_model_calls == 1
        assert middleware._total_tool_calls == 0
        assert result is None

    @pytest.mark.unit
    def test_after_agent_computes_statistics(self) -> None:
        """Test that after_agent computes correct statistics."""
        middleware = ToolCallObservabilityMiddleware()

        # Set up state
        middleware._total_model_calls = 5
        middleware._total_tool_calls = 10
        middleware._parallel_batches = 2
        middleware._sequential_calls = 3
        middleware._run_start_time = 0  # Mock start time

        runtime = MagicMock()
        runtime.context = Context(thread_id="test", user_id="test_user")

        state: AgentState = {"messages": []}

        # Call after_agent (this logs the summary)
        result = middleware.after_agent(state, runtime)

        # Verify result is None (no state changes)
        assert result is None

    @pytest.mark.unit
    def test_parallelism_ratio_calculation(self) -> None:
        """Test that parallelism ratio is calculated correctly."""
        middleware = ToolCallObservabilityMiddleware()

        # 2 parallel batches, 2 sequential = 50% parallelism
        middleware._parallel_batches = 2
        middleware._sequential_calls = 2

        total = middleware._parallel_batches + middleware._sequential_calls
        ratio = middleware._parallel_batches / total * 100

        assert ratio == 50.0

    @pytest.mark.unit
    def test_parallelism_ratio_100_percent(self) -> None:
        """Test 100% parallelism ratio."""
        middleware = ToolCallObservabilityMiddleware()

        middleware._parallel_batches = 5
        middleware._sequential_calls = 0

        total = middleware._parallel_batches + middleware._sequential_calls
        ratio = middleware._parallel_batches / total * 100 if total > 0 else 0

        assert ratio == 100.0

    @pytest.mark.unit
    def test_parallelism_ratio_0_percent(self) -> None:
        """Test 0% parallelism ratio."""
        middleware = ToolCallObservabilityMiddleware()

        middleware._parallel_batches = 0
        middleware._sequential_calls = 5

        total = middleware._parallel_batches + middleware._sequential_calls
        ratio = middleware._parallel_batches / total * 100 if total > 0 else 0

        assert ratio == 0.0

    @pytest.mark.unit
    def test_parallelism_ratio_no_calls(self) -> None:
        """Test parallelism ratio with no tool calls."""
        middleware = ToolCallObservabilityMiddleware()

        middleware._parallel_batches = 0
        middleware._sequential_calls = 0

        total = middleware._parallel_batches + middleware._sequential_calls
        ratio = middleware._parallel_batches / total * 100 if total > 0 else 0

        assert ratio == 0.0

    @pytest.mark.unit
    def test_log_level_case_insensitive(self) -> None:
        """Test that log level is normalized to uppercase."""
        middleware = ToolCallObservabilityMiddleware(log_level="debug")
        assert middleware.log_level == "DEBUG"

        middleware = ToolCallObservabilityMiddleware(log_level="Info")
        assert middleware.log_level == "INFO"

        middleware = ToolCallObservabilityMiddleware(log_level="WARNING")
        assert middleware.log_level == "WARNING"
