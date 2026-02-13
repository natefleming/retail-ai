"""
Tests for Deep Agents middleware factory functions.

This module tests the factory functions for TodoListMiddleware (from LangChain)
and the Deep Agents middleware factories (FilesystemMiddleware,
SubAgentMiddleware, MemoryMiddleware, SkillsMiddleware,
SummarizationMiddleware).

Run with:
    pytest tests/dao_ai/test_deepagents_middleware.py -v -m unit
"""

import pytest

# =============================================================================
# TodoListMiddleware Tests
# =============================================================================


@pytest.mark.unit
class TestTodoListMiddleware:
    """Tests for TodoListMiddleware factory."""

    def test_create_default(self) -> None:
        """Test creating TodoListMiddleware with default settings."""
        from dao_ai.middleware.todo import create_todo_list_middleware

        middleware = create_todo_list_middleware()
        assert middleware is not None
        assert middleware.tools is not None
        assert len(middleware.tools) == 1
        assert middleware.tools[0].name == "write_todos"

    def test_create_with_custom_system_prompt(self) -> None:
        """Test creating TodoListMiddleware with a custom system prompt."""
        from dao_ai.middleware.todo import create_todo_list_middleware

        custom_prompt = "Always create a todo list before starting."
        middleware = create_todo_list_middleware(
            system_prompt=custom_prompt,
        )
        assert middleware is not None
        assert middleware.system_prompt == custom_prompt

    def test_create_with_custom_tool_description(self) -> None:
        """Test creating TodoListMiddleware with custom tool description."""
        from dao_ai.middleware.todo import create_todo_list_middleware

        custom_desc = "Custom write_todos tool description."
        middleware = create_todo_list_middleware(
            tool_description=custom_desc,
        )
        assert middleware is not None
        assert middleware.tool_description == custom_desc

    def test_create_with_all_custom(self) -> None:
        """Test creating TodoListMiddleware with all custom settings."""
        from dao_ai.middleware.todo import create_todo_list_middleware

        middleware = create_todo_list_middleware(
            system_prompt="Custom prompt",
            tool_description="Custom desc",
        )
        assert middleware is not None
        assert middleware.system_prompt == "Custom prompt"
        assert middleware.tool_description == "Custom desc"

    def test_todo_type_exported(self) -> None:
        """Test that Todo TypedDict is exported."""
        from dao_ai.middleware.todo import Todo

        # Verify Todo has expected keys
        todo: Todo = {"content": "Test task", "status": "pending"}
        assert todo["content"] == "Test task"
        assert todo["status"] == "pending"

    def test_planning_state_exported(self) -> None:
        """Test that PlanningState is exported."""
        from dao_ai.middleware.todo import PlanningState

        assert PlanningState is not None

    def test_import_from_init(self) -> None:
        """Test imports from dao_ai.middleware."""
        from dao_ai.middleware import (
            PlanningState,
            Todo,
            TodoListMiddleware,
            create_todo_list_middleware,
        )

        assert TodoListMiddleware is not None
        assert Todo is not None
        assert PlanningState is not None
        assert create_todo_list_middleware is not None


# =============================================================================
# Backend Resolution Tests
# =============================================================================


@pytest.mark.unit
class TestBackendResolution:
    """Tests for the shared backend resolution utility."""

    def test_resolve_state_backend(self) -> None:
        """Test resolving the state backend."""
        from deepagents.backends import StateBackend

        from dao_ai.middleware._backends import resolve_backend

        backend = resolve_backend("state")
        assert backend is StateBackend

    def test_resolve_store_backend(self) -> None:
        """Test resolving the store backend."""
        from dao_ai.middleware._backends import resolve_backend

        backend = resolve_backend("store")
        assert backend is not None

    def test_resolve_filesystem_backend(self) -> None:
        """Test resolving the filesystem backend."""
        from dao_ai.middleware._backends import resolve_backend

        backend = resolve_backend("filesystem", root_dir="/tmp")
        assert backend is not None

    def test_resolve_filesystem_missing_root_dir(self) -> None:
        """Test that filesystem backend requires root_dir."""
        from dao_ai.middleware._backends import resolve_backend

        with pytest.raises(ValueError, match="root_dir is required"):
            resolve_backend("filesystem")

    def test_resolve_unknown_backend(self) -> None:
        """Test that unknown backend type raises ValueError."""
        from dao_ai.middleware._backends import resolve_backend

        with pytest.raises(ValueError, match="Unknown backend_type"):
            resolve_backend("unknown_backend")


# =============================================================================
# FilesystemMiddleware Tests
# =============================================================================


@pytest.mark.unit
class TestFilesystemMiddleware:
    """Tests for FilesystemMiddleware factory."""

    def test_create_default(self) -> None:
        """Test creating FilesystemMiddleware with defaults."""
        from dao_ai.middleware.filesystem import (
            create_filesystem_middleware,
        )

        middleware = create_filesystem_middleware()
        assert middleware is not None
        assert middleware.tools is not None
        # Should have ls, read_file, write_file, edit_file, glob, grep, execute
        assert len(middleware.tools) >= 6

    def test_create_with_custom_eviction(self) -> None:
        """Test creating with custom eviction threshold."""
        from dao_ai.middleware.filesystem import (
            create_filesystem_middleware,
        )

        middleware = create_filesystem_middleware(
            tool_token_limit_before_evict=10000,
        )
        assert middleware is not None

    def test_create_with_no_eviction(self) -> None:
        """Test creating with eviction disabled."""
        from dao_ai.middleware.filesystem import (
            create_filesystem_middleware,
        )

        middleware = create_filesystem_middleware(
            tool_token_limit_before_evict=None,
        )
        assert middleware is not None

    def test_import_from_init(self) -> None:
        """Test filesystem factory is importable from init."""
        from dao_ai.middleware import create_filesystem_middleware

        assert create_filesystem_middleware is not None


# =============================================================================
# SubAgentMiddleware Tests
# =============================================================================


@pytest.mark.unit
class TestSubAgentMiddleware:
    """Tests for SubAgentMiddleware factory."""

    def test_create_with_subagents(self) -> None:
        """Test creating with subagent specs."""
        from dao_ai.middleware.subagent import (
            create_subagent_middleware,
        )

        middleware = create_subagent_middleware(
            subagents=[
                {
                    "name": "researcher",
                    "description": "Research agent",
                    "system_prompt": "You are a researcher.",
                    "model": "openai:gpt-4o-mini",
                    "tools": [],
                }
            ],
        )
        assert middleware is not None
        assert middleware.tools is not None
        assert len(middleware.tools) == 1
        assert middleware.tools[0].name == "task"

    def test_import_from_init(self) -> None:
        """Test subagent factory is importable from init."""
        from dao_ai.middleware import create_subagent_middleware

        assert create_subagent_middleware is not None


# =============================================================================
# Subagent Model Resolution Tests
# =============================================================================


@pytest.mark.unit
class TestSubAgentModelResolution:
    """Tests for _resolve_subagent_model and model resolution in the factory."""

    def test_resolve_none(self) -> None:
        """Test that None is returned as-is (no model override)."""
        from dao_ai.middleware.subagent import _resolve_subagent_model

        assert _resolve_subagent_model(None) is None

    def test_resolve_string(self) -> None:
        """Test that a string model is passed through unchanged."""
        from dao_ai.middleware.subagent import _resolve_subagent_model

        result = _resolve_subagent_model("openai:gpt-4o-mini")
        assert result == "openai:gpt-4o-mini"

    def test_resolve_base_chat_model(self) -> None:
        """Test that a BaseChatModel instance is passed through directly."""
        from unittest.mock import MagicMock

        from langchain_core.language_models import BaseChatModel

        from dao_ai.middleware.subagent import _resolve_subagent_model

        mock_model = MagicMock(spec=BaseChatModel)
        result = _resolve_subagent_model(mock_model)
        assert result is mock_model

    def test_resolve_dict_calls_llm_model(self) -> None:
        """Test that a dict is converted via LLMModel(**dict).as_chat_model()."""
        from unittest.mock import MagicMock, patch

        from dao_ai.middleware.subagent import _resolve_subagent_model

        mock_chat_model = MagicMock()
        mock_llm_model_instance = MagicMock()
        mock_llm_model_instance.as_chat_model.return_value = mock_chat_model

        with patch(
            "dao_ai.config.LLMModel",
            return_value=mock_llm_model_instance,
        ) as mock_llm_cls:
            result = _resolve_subagent_model(
                {"name": "my-endpoint", "temperature": 0.1}
            )

        mock_llm_cls.assert_called_once_with(name="my-endpoint", temperature=0.1)
        mock_llm_model_instance.as_chat_model.assert_called_once()
        assert result is mock_chat_model

    def test_resolve_llm_model_instance(self) -> None:
        """Test that an LLMModel instance is converted via as_chat_model()."""
        from unittest.mock import MagicMock, patch

        from dao_ai.middleware.subagent import _resolve_subagent_model

        mock_chat_model = MagicMock()
        mock_llm_model = MagicMock()
        mock_llm_model.name = "my-endpoint"
        mock_llm_model.as_chat_model.return_value = mock_chat_model

        # Patch LLMModel so isinstance check succeeds
        with patch("dao_ai.config.LLMModel", type(mock_llm_model)):
            result = _resolve_subagent_model(mock_llm_model)

        mock_llm_model.as_chat_model.assert_called_once()
        assert result is mock_chat_model

    def test_resolve_invalid_type_raises(self) -> None:
        """Test that an unsupported type raises TypeError."""
        from dao_ai.middleware.subagent import _resolve_subagent_model

        with pytest.raises(TypeError, match="Unsupported subagent model type"):
            _resolve_subagent_model(12345)

    def test_factory_resolves_dict_model_in_subagent(self) -> None:
        """Test that create_subagent_middleware resolves dict models."""
        from unittest.mock import MagicMock, patch

        from dao_ai.middleware.subagent import create_subagent_middleware

        mock_chat_model = MagicMock()
        mock_llm_model_instance = MagicMock()
        mock_llm_model_instance.as_chat_model.return_value = mock_chat_model

        with patch(
            "dao_ai.config.LLMModel",
            return_value=mock_llm_model_instance,
        ):
            middleware = create_subagent_middleware(
                subagents=[
                    {
                        "name": "analyst",
                        "description": "Data analyst",
                        "system_prompt": "You are an analyst.",
                        "model": {"name": "my-endpoint", "temperature": 0.1},
                        "tools": [],
                    }
                ],
            )

        assert middleware is not None
        assert middleware.tools is not None
        assert len(middleware.tools) == 1
        assert middleware.tools[0].name == "task"

    def test_factory_passes_through_string_model(self) -> None:
        """Test that string models pass through without modification."""
        from dao_ai.middleware.subagent import create_subagent_middleware

        middleware = create_subagent_middleware(
            subagents=[
                {
                    "name": "researcher",
                    "description": "Research agent",
                    "system_prompt": "You are a researcher.",
                    "model": "openai:gpt-4o-mini",
                    "tools": [],
                }
            ],
        )
        assert middleware is not None

    def test_factory_does_not_mutate_original_specs(self) -> None:
        """Test that the original subagent spec dicts are not mutated."""
        from dao_ai.middleware.subagent import create_subagent_middleware

        original_spec: dict = {
            "name": "researcher",
            "description": "Research agent",
            "system_prompt": "You are a researcher.",
            "model": "openai:gpt-4o-mini",
            "tools": [],
        }
        original_model = original_spec["model"]

        create_subagent_middleware(subagents=[original_spec])

        # Original dict should be unmodified
        assert original_spec["model"] is original_model


# =============================================================================
# MemoryMiddleware Tests
# =============================================================================


@pytest.mark.unit
class TestAgentsMemoryMiddleware:
    """Tests for AGENTS.md MemoryMiddleware factory."""

    def test_create_with_sources(self) -> None:
        """Test creating with source paths."""
        from dao_ai.middleware.memory_agents import (
            create_agents_memory_middleware,
        )

        middleware = create_agents_memory_middleware(
            sources=["~/.deepagents/AGENTS.md"],
        )
        assert middleware is not None
        assert middleware.sources == ["~/.deepagents/AGENTS.md"]

    def test_create_with_multiple_sources(self) -> None:
        """Test creating with multiple source paths."""
        from dao_ai.middleware.memory_agents import (
            create_agents_memory_middleware,
        )

        sources = [
            "~/.deepagents/AGENTS.md",
            "./.deepagents/AGENTS.md",
        ]
        middleware = create_agents_memory_middleware(sources=sources)
        assert middleware is not None
        assert middleware.sources == sources

    def test_empty_sources_raises(self) -> None:
        """Test that empty sources list raises ValueError."""
        from dao_ai.middleware.memory_agents import (
            create_agents_memory_middleware,
        )

        with pytest.raises(ValueError, match="At least one source"):
            create_agents_memory_middleware(sources=[])

    def test_import_from_init(self) -> None:
        """Test memory factory is importable from init."""
        from dao_ai.middleware import create_agents_memory_middleware

        assert create_agents_memory_middleware is not None


# =============================================================================
# SkillsMiddleware Tests
# =============================================================================


@pytest.mark.unit
class TestSkillsMiddleware:
    """Tests for SkillsMiddleware factory."""

    def test_create_with_sources(self) -> None:
        """Test creating with source paths."""
        from dao_ai.middleware.skills import create_skills_middleware

        middleware = create_skills_middleware(
            sources=["/skills/user/"],
        )
        assert middleware is not None
        assert middleware.sources == ["/skills/user/"]

    def test_create_with_multiple_sources(self) -> None:
        """Test creating with multiple source paths."""
        from dao_ai.middleware.skills import create_skills_middleware

        sources = [
            "/skills/base/",
            "/skills/user/",
            "/skills/project/",
        ]
        middleware = create_skills_middleware(sources=sources)
        assert middleware is not None
        assert middleware.sources == sources

    def test_empty_sources_raises(self) -> None:
        """Test that empty sources list raises ValueError."""
        from dao_ai.middleware.skills import create_skills_middleware

        with pytest.raises(ValueError, match="At least one source"):
            create_skills_middleware(sources=[])

    def test_import_from_init(self) -> None:
        """Test importable from dao_ai.middleware."""
        from dao_ai.middleware import create_skills_middleware

        assert create_skills_middleware is not None


# =============================================================================
# Deep Agents SummarizationMiddleware Tests
# =============================================================================


@pytest.mark.unit
class TestDeepSummarizationMiddleware:
    """Tests for Deep Agents SummarizationMiddleware factory."""

    def test_create_with_model_string(self) -> None:
        """Test creating with model string."""
        from dao_ai.middleware.summarization import (
            create_deep_summarization_middleware,
        )

        middleware = create_deep_summarization_middleware(
            model="openai:gpt-4o-mini",
            trigger=("tokens", 100000),
        )
        assert middleware is not None

    def test_create_with_custom_keep(self) -> None:
        """Test creating with custom keep settings."""
        from dao_ai.middleware.summarization import (
            create_deep_summarization_middleware,
        )

        middleware = create_deep_summarization_middleware(
            model="openai:gpt-4o-mini",
            trigger=("tokens", 50000),
            keep=("messages", 10),
        )
        assert middleware is not None

    def test_create_with_arg_truncation(self) -> None:
        """Test creating with argument truncation enabled."""
        from dao_ai.middleware.summarization import (
            create_deep_summarization_middleware,
        )

        middleware = create_deep_summarization_middleware(
            model="openai:gpt-4o-mini",
            trigger=("tokens", 100000),
            truncate_args_trigger=("messages", 50),
            truncate_args_max_length=1000,
        )
        assert middleware is not None

    def test_import_from_init(self) -> None:
        """Test importable from dao_ai.middleware."""
        from dao_ai.middleware import (
            create_deep_summarization_middleware,
        )

        assert create_deep_summarization_middleware is not None


# =============================================================================
# Import-from-init Smoke Tests
# =============================================================================


@pytest.mark.unit
class TestInitImports:
    """Verify all factory functions are importable from dao_ai.middleware."""

    def test_todo_factory_importable(self) -> None:
        """Test that todo factory is importable."""
        from dao_ai.middleware import create_todo_list_middleware

        assert callable(create_todo_list_middleware)

    def test_filesystem_factory_importable(self) -> None:
        """Test that filesystem factory is importable."""
        from dao_ai.middleware import create_filesystem_middleware

        assert callable(create_filesystem_middleware)

    def test_subagent_factory_importable(self) -> None:
        """Test that subagent factory is importable."""
        from dao_ai.middleware import create_subagent_middleware

        assert callable(create_subagent_middleware)

    def test_memory_factory_importable(self) -> None:
        """Test that agents memory factory is importable."""
        from dao_ai.middleware import create_agents_memory_middleware

        assert callable(create_agents_memory_middleware)

    def test_skills_factory_importable(self) -> None:
        """Test that skills factory is importable."""
        from dao_ai.middleware import create_skills_middleware

        assert callable(create_skills_middleware)

    def test_deep_summarization_factory_importable(self) -> None:
        """Test that deep summarization factory is importable."""
        from dao_ai.middleware import (
            create_deep_summarization_middleware,
        )

        assert callable(create_deep_summarization_middleware)
