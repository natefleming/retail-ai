"""
Tests for LLM tool selector middleware factory.
"""

import pytest
from langchain.agents.middleware import LLMToolSelectorMiddleware

from dao_ai.config import PythonFunctionModel, ToolModel
from dao_ai.middleware import create_llm_tool_selector_middleware


class TestCreateLLMToolSelectorMiddleware:
    """Tests for the create_llm_tool_selector_middleware factory function."""

    @pytest.fixture
    def test_model(self):
        """Create a test LLM model string that LangChain can initialize."""
        # Use a simple model string that init_chat_model can parse
        return "gpt-4o-mini"

    def test_create_with_defaults(self, test_model):
        """Test creating middleware with default parameters."""
        middleware = create_llm_tool_selector_middleware(model=test_model)

        assert isinstance(middleware, LLMToolSelectorMiddleware)
        assert middleware.model is not None  # Model gets initialized
        assert middleware.max_tools == 3  # Default
        # When None is passed, middleware stores it as empty list
        assert middleware.always_include == [] or middleware.always_include is None

    def test_create_with_max_tools(self, test_model):
        """Test creating middleware with custom max_tools."""
        middleware = create_llm_tool_selector_middleware(model=test_model, max_tools=5)

        assert isinstance(middleware, LLMToolSelectorMiddleware)
        assert middleware.max_tools == 5

    def test_create_with_always_include_strings(self, test_model):
        """Test creating middleware with always_include as tool name strings."""
        middleware = create_llm_tool_selector_middleware(
            model=test_model,
            max_tools=3,
            always_include=["search_web", "database_query"],
        )

        assert isinstance(middleware, LLMToolSelectorMiddleware)
        assert middleware.always_include == ["search_web", "database_query"]

    def test_create_with_always_include_tool_models(self, test_model):
        """Test creating middleware with ToolModel instances in always_include."""
        tool_model_1 = ToolModel(
            name="tool_one",
            function=PythonFunctionModel(name="dao_ai.tools.tool_one"),
        )
        tool_model_2 = ToolModel(
            name="tool_two",
            function=PythonFunctionModel(name="dao_ai.tools.tool_two"),
        )

        middleware = create_llm_tool_selector_middleware(
            model=test_model,
            always_include=[tool_model_1, tool_model_2],
        )

        assert isinstance(middleware, LLMToolSelectorMiddleware)
        assert middleware.always_include == ["tool_one", "tool_two"]

    def test_create_with_always_include_dicts(self, test_model):
        """Test creating middleware with tool dicts in always_include."""
        tool_dict_1 = {
            "name": "dict_tool_one",
            "function": {"name": "dao_ai.tools.dict_tool_one"},
        }
        tool_dict_2 = {
            "name": "dict_tool_two",
            "function": {"name": "dao_ai.tools.dict_tool_two"},
        }

        middleware = create_llm_tool_selector_middleware(
            model=test_model,
            always_include=[tool_dict_1, tool_dict_2],
        )

        assert isinstance(middleware, LLMToolSelectorMiddleware)
        assert middleware.always_include == ["dict_tool_one", "dict_tool_two"]

    def test_create_with_mixed_always_include(self, test_model):
        """Test creating middleware with mixed types in always_include."""
        tool_model = ToolModel(
            name="model_tool",
            function=PythonFunctionModel(name="dao_ai.tools.model_tool"),
        )
        tool_dict = {
            "name": "dict_tool",
            "function": {"name": "dao_ai.tools.dict_tool"},
        }

        middleware = create_llm_tool_selector_middleware(
            model=test_model,
            always_include=["string_tool", tool_model, tool_dict],
        )

        assert isinstance(middleware, LLMToolSelectorMiddleware)
        assert middleware.always_include == ["string_tool", "model_tool", "dict_tool"]

    def test_create_with_empty_always_include(self, test_model):
        """Test creating middleware with empty always_include list."""
        middleware = create_llm_tool_selector_middleware(
            model=test_model,
            always_include=[],
        )

        assert isinstance(middleware, LLMToolSelectorMiddleware)
        # When empty list is passed, middleware stores it as empty list
        assert middleware.always_include == [] or middleware.always_include is None

    def test_resolve_tool_names_handles_missing_name_in_dict(self, test_model):
        """Test that dicts missing 'name' field are skipped with warning."""
        invalid_dict = {"description": "No name field"}

        middleware = create_llm_tool_selector_middleware(
            model=test_model,
            always_include=[invalid_dict, "valid_tool"],
        )

        # Should only include the valid tool
        assert middleware.always_include == ["valid_tool"]

    def test_resolve_tool_names_handles_unknown_type(self, test_model):
        """Test that unknown types in always_include are skipped with warning."""
        middleware = create_llm_tool_selector_middleware(
            model=test_model,
            always_include=["valid_tool", 123, "another_valid"],  # 123 is invalid
        )

        # Should only include the valid tools
        assert middleware.always_include == ["valid_tool", "another_valid"]

    def test_returns_single_instance(self, test_model):
        """Test that factory returns a single middleware instance."""
        middleware = create_llm_tool_selector_middleware(
            model=test_model,
            max_tools=5,
        )

        assert isinstance(middleware, LLMToolSelectorMiddleware)
        # Not a list
        assert not isinstance(middleware, list)

    def test_factory_accepts_all_parameters(self, test_model):
        """Test that factory accepts all supported parameters."""
        middleware = create_llm_tool_selector_middleware(
            model=test_model,
            max_tools=7,
            always_include=["tool1", "tool2"],
        )

        assert isinstance(middleware, LLMToolSelectorMiddleware)
        assert middleware.model is not None  # Model gets initialized
        assert middleware.max_tools == 7
        assert middleware.always_include == ["tool1", "tool2"]

    def test_use_case_large_tool_set(self, test_model):
        """
        Test typical use case: agent with 20+ tools where most aren't relevant.

        This middleware helps reduce context size and improve accuracy by
        having a smaller, cheaper model pre-select the most relevant tools.
        """
        middleware = create_llm_tool_selector_middleware(
            model=test_model,  # Typically a cheap, fast model like gpt-4o-mini
            max_tools=5,  # Select top 5 most relevant tools
            always_include=["search_web"],  # Always keep search available
        )

        assert isinstance(middleware, LLMToolSelectorMiddleware)
        assert middleware.max_tools == 5
        assert "search_web" in middleware.always_include

    def test_use_case_cost_optimization(self, test_model):
        """
        Test use case: optimize cost by reducing tokens in main model calls.

        By filtering down to only relevant tools, we reduce the size of the
        system prompt and save tokens on every agent turn.
        """
        middleware = create_llm_tool_selector_middleware(
            model=test_model,
            max_tools=3,  # Very selective to minimize tokens
        )

        assert isinstance(middleware, LLMToolSelectorMiddleware)
        assert middleware.max_tools == 3
