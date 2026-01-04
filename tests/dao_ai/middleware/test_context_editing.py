"""
Tests for context editing middleware factory.
"""

import pytest
from langchain.agents.middleware import ClearToolUsesEdit, ContextEditingMiddleware

from dao_ai.config import PythonFunctionModel, ToolModel
from dao_ai.middleware import (
    create_clear_tool_uses_edit,
    create_context_editing_middleware,
)


class TestCreateClearToolUsesEdit:
    """Tests for the create_clear_tool_uses_edit factory function."""

    def test_create_with_defaults(self):
        """Test creating edit with default parameters."""
        edit = create_clear_tool_uses_edit()

        assert isinstance(edit, ClearToolUsesEdit)
        assert edit.trigger == 100000
        assert edit.keep == 3
        assert edit.clear_at_least == 0
        assert edit.clear_tool_inputs is False
        assert edit.exclude_tools == []
        assert edit.placeholder == "[cleared]"

    def test_create_with_custom_trigger(self):
        """Test creating edit with custom trigger threshold."""
        edit = create_clear_tool_uses_edit(trigger=50000)

        assert edit.trigger == 50000

    def test_create_with_custom_keep(self):
        """Test creating edit with custom keep count."""
        edit = create_clear_tool_uses_edit(keep=5)

        assert edit.keep == 5

    def test_create_with_clear_at_least(self):
        """Test creating edit with clear_at_least set."""
        edit = create_clear_tool_uses_edit(clear_at_least=1000)

        assert edit.clear_at_least == 1000

    def test_create_with_clear_tool_inputs(self):
        """Test creating edit with clear_tool_inputs enabled."""
        edit = create_clear_tool_uses_edit(clear_tool_inputs=True)

        assert edit.clear_tool_inputs is True

    def test_create_with_string_exclude_tools(self):
        """Test creating edit with string tool exclusions."""
        edit = create_clear_tool_uses_edit(
            exclude_tools=["important_tool", "critical_search"],
        )

        assert edit.exclude_tools == ["important_tool", "critical_search"]

    def test_create_with_custom_placeholder(self):
        """Test creating edit with custom placeholder."""
        edit = create_clear_tool_uses_edit(placeholder="[output removed]")

        assert edit.placeholder == "[output removed]"

    def test_create_with_all_parameters(self):
        """Test creating edit with all parameters specified."""
        edit = create_clear_tool_uses_edit(
            trigger=75000,
            keep=10,
            clear_at_least=500,
            clear_tool_inputs=True,
            exclude_tools=["tool1", "tool2"],
            placeholder="[cleared for context]",
        )

        assert edit.trigger == 75000
        assert edit.keep == 10
        assert edit.clear_at_least == 500
        assert edit.clear_tool_inputs is True
        assert edit.exclude_tools == ["tool1", "tool2"]
        assert edit.placeholder == "[cleared for context]"

    def test_create_with_tool_model_exclude(self):
        """Test creating edit with ToolModel exclusions."""
        tool_model = ToolModel(
            name="my_tool",
            function=PythonFunctionModel(name="dao_ai.tools.say_hello_tool"),
        )

        edit = create_clear_tool_uses_edit(exclude_tools=[tool_model])

        # Should resolve to tool names
        assert isinstance(edit.exclude_tools, list)

    def test_create_with_dict_exclude(self):
        """Test creating edit with dict exclusions."""
        tool_dict = {
            "name": "my_tool",
            "function": {"name": "dao_ai.tools.say_hello_tool"},
        }

        edit = create_clear_tool_uses_edit(exclude_tools=[tool_dict])

        assert isinstance(edit.exclude_tools, list)


class TestCreateContextEditingMiddleware:
    """Tests for the create_context_editing_middleware factory function."""

    def test_create_with_defaults(self):
        """Test creating middleware with default parameters."""
        middleware = create_context_editing_middleware()

        # Middleware is single instance
        assert middleware is not None
        middleware = middleware
        assert isinstance(middleware, ContextEditingMiddleware)
        assert middleware.token_count_method == "approximate"
        assert len(middleware.edits) == 1
        assert isinstance(middleware.edits[0], ClearToolUsesEdit)

    def test_create_with_custom_trigger(self):
        """Test creating middleware with custom trigger."""
        middleware = create_context_editing_middleware(trigger=50000)

        # Middleware is single instance
        assert middleware is not None
        middleware = middleware
        assert isinstance(middleware, ContextEditingMiddleware)
        assert middleware.edits[0].trigger == 50000

    def test_create_with_custom_keep(self):
        """Test creating middleware with custom keep count."""
        middleware = create_context_editing_middleware(keep=10)

        assert middleware.edits[0].keep == 10

    def test_create_with_model_token_count(self):
        """Test creating middleware with model token counting."""
        middleware = create_context_editing_middleware(
            token_count_method="model",
        )

        assert middleware.token_count_method == "model"

    def test_create_with_all_parameters(self):
        """Test creating middleware with all parameters."""
        middleware = create_context_editing_middleware(
            trigger=50000,
            keep=5,
            clear_at_least=1000,
            clear_tool_inputs=True,
            exclude_tools=["important_tool"],
            placeholder="[removed]",
            token_count_method="model",
        )

        # Middleware is single instance
        assert middleware is not None
        middleware = middleware
        assert isinstance(middleware, ContextEditingMiddleware)
        assert middleware.token_count_method == "model"

        edit = middleware.edits[0]
        assert edit.trigger == 50000
        assert edit.keep == 5
        assert edit.clear_at_least == 1000
        assert edit.clear_tool_inputs is True
        assert edit.exclude_tools == ["important_tool"]
        assert edit.placeholder == "[removed]"

    def test_create_with_invalid_dict_exclude(self):
        """Test that invalid dict raises helpful error."""
        invalid_dict = {"invalid": "data"}

        with pytest.raises(ValueError, match="Failed to construct ToolModel from dict"):
            create_context_editing_middleware(exclude_tools=[invalid_dict])

    def test_returns_list_for_composition(self):
        """Test that factory returns list for easy composition."""
        middleware = create_context_editing_middleware()

        # Middleware is single instance
        assert middleware is not None
