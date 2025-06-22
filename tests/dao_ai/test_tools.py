from typing import Sequence
from unittest.mock import patch

import pytest
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.tools import tool as create_tool
from langgraph.prebuilt.interrupt import HumanInterruptConfig

from dao_ai.config import AppConfig, FunctionType, ToolModel
from dao_ai.tools import create_tools
from dao_ai.tools.core import add_human_in_the_loop

excluded_tools: Sequence[str] = [
    "vector_search",
    "genie",
    "find_product_details_by_description",
]


@pytest.mark.unit
def test_create_tools(config: AppConfig) -> None:
    tool_models: list[ToolModel] = config.find_tools(
        lambda tool: not any(excluded in tool.name for excluded in excluded_tools)
        and tool.function.type != FunctionType.UNITY_CATALOG
    )

    tools = create_tools(tool_models)

    assert tools is not None


@pytest.mark.unit
def test_add_human_in_the_loop_with_callable():
    """Test add_human_in_the_loop with a callable function."""

    @create_tool("test_tool", description="A test tool")
    def test_function(input_text: str) -> str:
        """A simple test function that returns modified input."""
        return f"processed: {input_text}"

    # Mock the interrupt function
    with patch("dao_ai.tools.core.interrupt") as mock_interrupt:
        # Test accept scenario
        mock_interrupt.return_value = [{"type": "accept"}]

        wrapped_tool = add_human_in_the_loop(test_function)

        assert isinstance(wrapped_tool, BaseTool)
        assert wrapped_tool.name == "test_tool"
        assert wrapped_tool.description == "A test tool"

        # Test tool execution with accept
        config = RunnableConfig()
        result = wrapped_tool.invoke({"input_text": "hello"}, config=config)
        assert result == "processed: hello"

        # Verify interrupt was called with correct parameters
        mock_interrupt.assert_called_once()
        interrupt_args = mock_interrupt.call_args[0][0][0]
        assert interrupt_args["action_request"]["action"] == "test_tool"
        assert interrupt_args["action_request"]["args"] == {"input_text": "hello"}


@pytest.mark.unit
def test_add_human_in_the_loop_with_base_tool():
    """Test add_human_in_the_loop with an existing BaseTool."""

    @create_tool("existing_tool", description="An existing tool")
    def existing_function(value: int) -> int:
        """Multiply input by 2."""
        return value * 2

    with patch("dao_ai.tools.core.interrupt") as mock_interrupt:
        mock_interrupt.return_value = [{"type": "accept"}]

        wrapped_tool = add_human_in_the_loop(existing_function)

        assert isinstance(wrapped_tool, BaseTool)
        assert wrapped_tool.name == "existing_tool"

        # Test execution
        config = RunnableConfig()
        result = wrapped_tool.invoke({"value": 5}, config=config)
        assert result == 10


@pytest.mark.unit
def test_add_human_in_the_loop_edit_response():
    """Test human-in-the-loop with edit response type."""

    @create_tool("edit_test_tool", description="Tool for testing edit functionality")
    def edit_test_function(message: str) -> str:
        """Echo the message with prefix."""
        return f"echo: {message}"

    with patch("dao_ai.tools.core.interrupt") as mock_interrupt:
        # Mock edit response
        mock_interrupt.return_value = [
            {"type": "edit", "args": {"args": {"message": "edited_message"}}}
        ]

        wrapped_tool = add_human_in_the_loop(edit_test_function)

        config = RunnableConfig()
        result = wrapped_tool.invoke({"message": "original_message"}, config=config)

        # Should use the edited message
        assert result == "echo: edited_message"


@pytest.mark.unit
def test_add_human_in_the_loop_response_type():
    """Test human-in-the-loop with direct response type."""

    @create_tool(
        "response_test_tool", description="Tool for testing response functionality"
    )
    def response_test_function(query: str) -> str:
        """Process query."""
        return f"processed: {query}"

    with patch("dao_ai.tools.core.interrupt") as mock_interrupt:
        # Mock direct response
        mock_interrupt.return_value = [
            {"type": "response", "args": "custom human response"}
        ]

        wrapped_tool = add_human_in_the_loop(response_test_function)

        config = RunnableConfig()
        result = wrapped_tool.invoke({"query": "test query"}, config=config)

        # Should return the human response directly
        assert result == "custom human response"


@pytest.mark.unit
def test_add_human_in_the_loop_custom_interrupt_config():
    """Test add_human_in_the_loop with custom interrupt configuration."""

    custom_config: HumanInterruptConfig = {
        "allow_accept": True,
        "allow_edit": False,
        "allow_respond": True,
    }

    @create_tool("config_test_tool", description="Tool for testing custom config")
    def config_test_function(data: str) -> str:
        """Process data."""
        return f"result: {data}"

    with patch("dao_ai.tools.core.interrupt") as mock_interrupt:
        mock_interrupt.return_value = [{"type": "accept"}]

        wrapped_tool = add_human_in_the_loop(
            config_test_function, interrupt_config=custom_config
        )

        config = RunnableConfig()
        wrapped_tool.invoke({"data": "test"}, config=config)

        # Verify custom config was passed
        interrupt_args = mock_interrupt.call_args[0][0][0]
        assert interrupt_args["config"] == custom_config


@pytest.mark.unit
def test_add_human_in_the_loop_invalid_response_type():
    """Test add_human_in_the_loop with invalid response type raises ValueError."""

    @create_tool("invalid_test_tool", description="Tool for testing invalid responses")
    def invalid_test_function(input_data: str) -> str:
        """Process input."""
        return f"output: {input_data}"

    with patch("dao_ai.tools.core.interrupt") as mock_interrupt:
        # Mock invalid response type
        mock_interrupt.return_value = [{"type": "unknown_type"}]

        wrapped_tool = add_human_in_the_loop(invalid_test_function)

        config = RunnableConfig()

        with pytest.raises(
            ValueError, match="Unknown interrupt response type: unknown_type"
        ):
            wrapped_tool.invoke({"input_data": "test"}, config=config)


@pytest.mark.unit
def test_add_human_in_the_loop_default_interrupt_config():
    """Test that default interrupt config is used when none provided."""

    @create_tool("default_config_tool", description="Tool for testing default config")
    def default_config_function(text: str) -> str:
        """Process text."""
        return f"default: {text}"

    with patch("dao_ai.tools.core.interrupt") as mock_interrupt:
        mock_interrupt.return_value = [{"type": "accept"}]

        wrapped_tool = add_human_in_the_loop(default_config_function)

        config = RunnableConfig()
        wrapped_tool.invoke({"text": "test"}, config=config)

        # Verify default config was used
        interrupt_args = mock_interrupt.call_args[0][0][0]
        expected_default_config = {
            "allow_accept": True,
            "allow_edit": True,
            "allow_respond": True,
        }
        assert interrupt_args["config"] == expected_default_config
