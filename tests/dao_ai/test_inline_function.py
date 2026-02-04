"""
Unit tests for InlineFunctionModel - inline tool definitions in YAML.

These tests verify that:
1. InlineFunctionModel can be created with valid code
2. Tools are correctly created from inline code
3. The created tools work as expected
4. Error handling works for invalid code
"""

import pytest
from langchain_core.tools import BaseTool

from dao_ai.config import (
    FunctionType,
    InlineFunctionModel,
    ToolModel,
)


class TestInlineFunctionModel:
    """Tests for InlineFunctionModel configuration."""

    @pytest.mark.unit
    def test_inline_function_model_creation(self) -> None:
        """Test that InlineFunctionModel can be created with valid code."""
        code = """
from langchain.tools import tool

@tool
def test_tool(x: str) -> str:
    '''A test tool.'''
    return f"Result: {x}"
"""
        model = InlineFunctionModel(type=FunctionType.INLINE, code=code)

        assert model.type == FunctionType.INLINE
        assert model.code == code

    @pytest.mark.unit
    def test_inline_function_model_type_default(self) -> None:
        """Test that InlineFunctionModel defaults to INLINE type."""
        code = """
from langchain.tools import tool

@tool
def test_tool(x: str) -> str:
    '''A test tool.'''
    return f"Result: {x}"
"""
        model = InlineFunctionModel(code=code)

        assert model.type == FunctionType.INLINE

    @pytest.mark.unit
    def test_inline_function_model_as_tools_single(self) -> None:
        """Test that as_tools() returns a list with one tool."""
        code = """
from langchain.tools import tool

@tool
def calculator(expression: str) -> str:
    '''Evaluate a mathematical expression.'''
    return str(eval(expression))
"""
        model = InlineFunctionModel(code=code)
        tools = model.as_tools()

        assert len(tools) == 1
        assert isinstance(tools[0], BaseTool)
        assert tools[0].name == "calculator"

    @pytest.mark.unit
    def test_inline_function_model_as_tools_multiple(self) -> None:
        """Test that as_tools() can return multiple tools defined in one code block."""
        code = """
from langchain.tools import tool

@tool
def add(a: int, b: int) -> int:
    '''Add two numbers.'''
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    '''Multiply two numbers.'''
    return a * b
"""
        model = InlineFunctionModel(code=code)
        tools = model.as_tools()

        assert len(tools) == 2
        tool_names = {t.name for t in tools}
        assert "add" in tool_names
        assert "multiply" in tool_names

    @pytest.mark.unit
    def test_inline_function_model_tool_execution(self) -> None:
        """Test that the created tool can be invoked and returns correct results."""
        code = """
from langchain.tools import tool

@tool
def calculator(expression: str) -> str:
    '''Evaluate a mathematical expression.'''
    return str(eval(expression))
"""
        model = InlineFunctionModel(code=code)
        tools = model.as_tools()
        calc_tool = tools[0]

        result = calc_tool.invoke({"expression": "2 + 3"})
        assert result == "5"

        result = calc_tool.invoke({"expression": "10 * 5"})
        assert result == "50"

    @pytest.mark.unit
    def test_inline_function_model_tool_description(self) -> None:
        """Test that the tool description comes from the docstring."""
        code = """
from langchain.tools import tool

@tool
def my_tool(x: str) -> str:
    '''This is my custom description.'''
    return x
"""
        model = InlineFunctionModel(code=code)
        tools = model.as_tools()

        assert tools[0].description == "This is my custom description."

    @pytest.mark.unit
    def test_inline_function_model_invalid_code(self) -> None:
        """Test that invalid Python code raises an error."""
        code = """
this is not valid python code!!!
"""
        model = InlineFunctionModel(code=code)

        with pytest.raises(ValueError, match="Failed to execute inline tool code"):
            model.as_tools()

    @pytest.mark.unit
    def test_inline_function_model_no_tool_decorator(self) -> None:
        """Test that code without @tool decorator raises an error."""
        code = """
def my_function(x: str) -> str:
    '''A function without @tool decorator.'''
    return x
"""
        model = InlineFunctionModel(code=code)

        with pytest.raises(
            ValueError, match="must define at least one function decorated with @tool"
        ):
            model.as_tools()

    @pytest.mark.unit
    def test_inline_function_model_with_imports(self) -> None:
        """Test that inline code can use standard library imports."""
        code = """
from langchain.tools import tool
import random

@tool
def random_number(min_val: int = 1, max_val: int = 100) -> int:
    '''Generate a random number.'''
    return random.randint(min_val, max_val)
"""
        model = InlineFunctionModel(code=code)
        tools = model.as_tools()

        assert len(tools) == 1
        result = tools[0].invoke({"min_val": 1, "max_val": 10})
        assert 1 <= result <= 10

    @pytest.mark.unit
    def test_inline_function_model_serialization(self) -> None:
        """Test that InlineFunctionModel can be serialized and deserialized."""
        code = """
from langchain.tools import tool

@tool
def test_tool(x: str) -> str:
    '''A test tool.'''
    return f"Result: {x}"
"""
        model = InlineFunctionModel(code=code)
        serialized = model.model_dump()

        assert serialized["type"] == "inline"
        assert serialized["code"] == code

        # Recreate from serialized
        recreated = InlineFunctionModel(**serialized)
        assert recreated.type == FunctionType.INLINE
        assert recreated.code == code


class TestToolModelWithInline:
    """Tests for ToolModel with inline function type."""

    @pytest.mark.unit
    def test_tool_model_with_inline_function(self) -> None:
        """Test that ToolModel can be created with InlineFunctionModel."""
        tool_config = {
            "name": "my_calculator",
            "function": {
                "type": "inline",
                "code": """
from langchain.tools import tool

@tool
def calculator(expression: str) -> str:
    '''Evaluate a mathematical expression.'''
    return str(eval(expression))
""",
            },
        }

        tool_model = ToolModel(**tool_config)

        assert tool_model.name == "my_calculator"
        assert isinstance(tool_model.function, InlineFunctionModel)
        assert tool_model.function.type == FunctionType.INLINE

    @pytest.mark.unit
    def test_tool_model_inline_as_tools(self) -> None:
        """Test that ToolModel with inline function creates working tools."""
        tool_config = {
            "name": "greeting",
            "function": {
                "type": "inline",
                "code": """
from langchain.tools import tool

@tool
def greet(name: str) -> str:
    '''Greet someone by name.'''
    return f"Hello, {name}!"
""",
            },
        }

        tool_model = ToolModel(**tool_config)
        tools = tool_model.function.as_tools()

        assert len(tools) == 1
        result = tools[0].invoke({"name": "World"})
        assert result == "Hello, World!"


class TestInlineFunctionHumanInTheLoop:
    """Tests for InlineFunctionModel with human-in-the-loop configuration."""

    @pytest.mark.unit
    def test_inline_function_with_hitl(self) -> None:
        """Test that inline functions can have human-in-the-loop config."""
        tool_config = {
            "name": "dangerous_tool",
            "function": {
                "type": "inline",
                "code": """
from langchain.tools import tool

@tool
def dangerous_action(target: str) -> str:
    '''A dangerous action that requires approval.'''
    return f"Action performed on {target}"
""",
                "human_in_the_loop": {
                    "review_prompt": "This action is dangerous. Approve?",
                    "allowed_decisions": ["approve", "reject"],
                },
            },
        }

        tool_model = ToolModel(**tool_config)

        assert tool_model.function.human_in_the_loop is not None
        assert (
            tool_model.function.human_in_the_loop.review_prompt
            == "This action is dangerous. Approve?"
        )
        assert tool_model.function.human_in_the_loop.allowed_decisions == [
            "approve",
            "reject",
        ]
