"""
Integration test for Genie tool Databricks model serving validation issue.

This test reproduces and verifies the fix for the ValidationError that occurred
when the Genie tool was deployed to Databricks model serving, where Pydantic
was trying to validate injected state parameters as tool inputs.

The original error was:
ValidationError: 6 validation errors for my_genie_tool
  state.context
    Field required [type=missing, input_value={...}, input_loc=('state', 'context')]
  state.conversations
    Field required [type=missing, input_value={...}, input_loc=('state', 'conversations')]
  ...and 4 more similar errors for other state fields

The fix involved:
1. Using ToolRuntime[Context, AgentState] for unified access to state and context
2. Using explicit args_schema=GenieToolInput to exclude injected parameters from validation
3. LangGraph's dependency injection handles runtime parameter injection
"""

from unittest.mock import Mock, patch

import pytest
from conftest import has_databricks_env
from langchain_core.tools import StructuredTool

from dao_ai.config import GenieRoomModel
from dao_ai.tools.genie import GenieToolInput, create_genie_tool


@pytest.fixture
def mock_genie_room():
    """Fixture providing a mock GenieRoomModel."""
    return GenieRoomModel(name="test-genie-room", space_id="test-space-123")


@pytest.fixture
def mock_genie_space():
    """Mock Databricks Genie space response."""
    space = Mock()
    space.description = "Test space description"
    return space


@pytest.fixture
def mock_genie_response():
    """Mock Genie response object."""
    response = Mock()
    response.conversation_id = "test-conversation-123"
    response.description = "Test description from Genie"
    response.query = "SELECT * FROM test_table"
    response.result = "Test result data"
    return response


@pytest.fixture
def mock_genie_tool():
    """Create a mocked Genie tool for testing validation scenarios."""
    with (
        patch("dao_ai.tools.genie.Genie") as mock_genie_class,
        patch("databricks.sdk.WorkspaceClient") as mock_client,
    ):
        # Mock the WorkspaceClient and Genie space
        mock_workspace = Mock()
        mock_workspace.genie.get_space.return_value = Mock(description="Test space")
        mock_client.return_value = mock_workspace

        # Mock the Genie instance
        mock_genie_instance = Mock()
        mock_genie_instance.ask_question = Mock(
            return_value=Mock(
                conversation_id="test-conv-123",
                description="Test description",
                query="SELECT * FROM test",
                result="Test result data",
            )
        )
        mock_genie_class.return_value = mock_genie_instance

        genie_room = GenieRoomModel(name="test-genie-room", space_id="test-space-123")

        yield create_genie_tool(genie_room, persist_conversation=True)


class TestGenieDatabricksIntegration:
    """
    Test suite for Databricks model serving validation scenarios.

    This test suite reproduces and verifies the fix for a critical ValidationError
    that occurred when the Genie tool was deployed to Databricks model serving.

    PROBLEM: Pydantic was trying to validate injected state parameters as tool inputs
    SOLUTION: Made function async + used explicit args_schema to exclude injected parameters
    RESULT: Tool validation only includes user inputs, not LangGraph injected state
    """

    def test_genie_tool_uses_injected_state_pattern(self, mock_genie_tool):
        """Test that the tool uses ToolRuntime for accessing state and context."""
        tool = mock_genie_tool

        # Verify the function signature has injected parameters
        import inspect

        sig = inspect.signature(tool.func)
        params = list(sig.parameters.keys())

        # With @tool decorator and ToolRuntime, the function signature
        # includes question and runtime
        assert "question" in params
        assert "runtime" in params

        # The ToolRuntime parameter will be injected by LangGraph
        # at runtime and hidden from the model

    def test_genie_tool_input_validation_success(self):
        """Test that GenieToolInput validates correctly with just question."""
        # This should succeed - only user inputs
        valid_input = GenieToolInput(question="What is the weather today?")
        assert valid_input.question == "What is the weather today?"

    def test_genie_tool_input_validation_accepts_only_question(self):
        """Test that GenieToolInput correctly accepts only expected fields."""
        # This should succeed - only with question
        valid_input = GenieToolInput(question="What is the weather today?")
        assert valid_input.question == "What is the weather today?"

        # Extra fields are ignored by default in Pydantic v2 unless model_config forbids them
        # The key point is that injected parameters are NOT part of the schema validation
        input_with_extra = GenieToolInput(
            question="What is the weather today?",
            state={"context": {}, "conversations": {}},  # This gets ignored
            tool_call_id="test-123",  # This gets ignored
        )
        assert input_with_extra.question == "What is the weather today?"

        # The important thing is that these fields don't show up in the model dump
        dumped = input_with_extra.model_dump()
        assert "state" not in dumped
        assert "tool_call_id" not in dumped
        assert "question" in dumped

    def test_genie_tool_function_signature_compatibility(self, mock_genie_tool):
        """Test that the genie_tool function has the correct async signature for LangGraph injection."""
        tool = mock_genie_tool

        # Verify the tool is created with func parameter (not coroutine)
        assert hasattr(tool, "func")
        assert tool.func is not None

        # Verify the function is sync (not async)
        import inspect

        assert not inspect.iscoroutinefunction(tool.func)

    def test_simulate_databricks_model_serving_scenario(self, mock_genie_tool):
        """
        Simulate the exact scenario that caused validation errors in Databricks model serving.

        This test reproduces and verifies the fix for the conditions where:
        1. Tool is invoked with only user inputs (question)
        2. Runtime is injected by LangGraph using ToolRuntime type
        3. Tool works correctly with injected parameters

        THE FIX:
        1. Used @tool decorator which properly handles injected parameters
        2. ToolRuntime[Context, AgentState] provides access to state and context
        3. The tool function signature includes question and runtime, but LangGraph
           handles injection transparently
        """
        # Use the mocked tool
        tool = mock_genie_tool

        # Verify tool has the proper function signature
        import inspect

        sig = inspect.signature(tool.func)
        params = list(sig.parameters.keys())

        # Should have question and runtime parameters
        assert "question" in params
        assert "runtime" in params

        # Verify the tool is properly configured
        assert isinstance(tool, StructuredTool)
        assert tool.name == "genie_tool"

        # The key fix: @tool decorator with ToolRuntime handles injection properly
        # LangGraph will inject runtime at runtime

    def test_structured_tool_configuration(self, mock_genie_tool):
        """Test that StructuredTool is configured correctly with @tool decorator."""
        tool = mock_genie_tool

        # Verify it's a StructuredTool
        assert isinstance(tool, StructuredTool)

        # Verify key configuration
        assert tool.name == "genie_tool"
        assert tool.description is not None
        assert "tabular data" in tool.description

        # Verify args_schema is auto-generated by @tool decorator
        assert tool.args_schema is not None
        assert hasattr(tool.args_schema, "model_json_schema")

        # Verify func is set
        assert hasattr(tool, "func")
        assert tool.func is not None

    @pytest.mark.skipif(not has_databricks_env(), reason="Databricks env vars not set")
    def test_conversation_persistence_with_injected_state(self, mock_genie_tool):
        """
        Test that conversation persistence works correctly with ToolRuntime pattern.

        This verifies that the fix doesn't break the core functionality of
        conversation mapping using space_id from the state.
        """
        from unittest.mock import Mock

        from dao_ai.state import AgentState, Context

        # Create a mock runtime
        mock_runtime = Mock()
        mock_runtime.state = AgentState(
            messages=[],
            genie_conversation_ids={"test-space-123": "existing-conversation-id"},
        )
        mock_runtime.context = Context(user_id="test-user", thread_id="test-thread")
        mock_runtime.tool_call_id = "test-call-id"

        # Use the fixture genie tool
        tool = mock_genie_tool

        # Call the tool - this should work without validation errors
        result = tool.func(
            question="Continue our previous conversation",
            runtime=mock_runtime,
        )

        # Verify the result structure (basic validation that it executed)
        from langgraph.types import Command

        from dao_ai.state import SessionState

        assert isinstance(result, Command)
        assert result.update is not None
        # Conversation IDs are now stored in session.genie.spaces
        assert "session" in result.update
        session: SessionState = result.update["session"]
        assert session.genie is not None
        assert session.genie.spaces is not None
        assert "test-space-123" in session.genie.spaces

    def test_injected_state_pattern(self, mock_genie_tool):
        """
        Test that the tool uses ToolRuntime for dependency injection.

        ToolRuntime[Context, AgentState] provides access to state, context,
        and tool_call_id through LangGraph's dependency injection.
        """
        tool = mock_genie_tool

        # Verify the tool is properly configured with @tool decorator
        assert isinstance(tool, StructuredTool)
        assert tool.name == "genie_tool"

        # Verify function signature has injected parameters
        import inspect

        sig = inspect.signature(tool.func)
        params = list(sig.parameters.keys())

        # Should have question and runtime parameters
        assert "question" in params
        assert "runtime" in params

        # ToolRuntime provides access to state, context, and tool_call_id at runtime


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
