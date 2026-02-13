"""
Subagent middleware for DAO AI agents.

This module provides a factory function for creating SubAgentMiddleware instances
from the Deep Agents library. SubAgentMiddleware adds a ``task`` tool that allows
agents to spawn subagents with isolated context windows for handling complex,
multi-step tasks.

This complements DAO AI's existing supervisor/swarm orchestration patterns by
enabling agent-initiated delegation (the agent decides when to spawn subagents)
rather than graph-level orchestration (the graph topology determines routing).

The ``model`` field in each subagent spec supports multiple input types:

- **str**: A ``"provider:model"`` identifier (e.g., ``"openai:gpt-4o"``), passed
  directly to deepagents.
- **dict**: An ``LLMModel``-style mapping (e.g.,
  ``{"name": "my-endpoint", "temperature": 0.1}``), converted to a
  ``ChatDatabricks`` instance via ``LLMModel.as_chat_model()``.
- **LLMModel**: A DAO AI ``LLMModel`` instance, converted via
  ``as_chat_model()``.
- **BaseChatModel**: A LangChain chat model instance (e.g., ``ChatDatabricks``),
  passed through directly.

Example:
    from dao_ai.middleware import create_subagent_middleware

    middleware = create_subagent_middleware(
        subagents=[
            {
                "name": "researcher",
                "description": "Research agent for complex queries",
                "system_prompt": "You are a research assistant.",
                "model": "openai:gpt-4o",
                "tools": [],
            }
        ],
    )

    # Using a Databricks endpoint via LLMModel dict:
    middleware = create_subagent_middleware(
        subagents=[
            {
                "name": "analyst",
                "description": "Data analysis agent",
                "system_prompt": "You are a data analyst.",
                "model": {
                    "name": "databricks-meta-llama-3-3-70b-instruct",
                    "temperature": 0.1,
                },
            }
        ],
    )

YAML Config:
    middleware:
      - name: dao_ai.middleware.subagent.create_subagent_middleware
        args:
          subagents:
            - name: researcher
              description: "Research agent for complex queries"
              system_prompt: "You are a research assistant."
              model: "openai:gpt-4o"
            - name: analyst
              description: "Data analysis agent"
              system_prompt: "You are a data analyst."
              model:
                name: "databricks-meta-llama-3-3-70b-instruct"
                temperature: 0.1
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from deepagents.middleware.subagents import SubAgentMiddleware
from langchain_core.language_models import BaseChatModel
from loguru import logger

from dao_ai.middleware._backends import resolve_backend

if TYPE_CHECKING:
    from dao_ai.config import LLMModel, VolumePathModel

__all__ = [
    "create_subagent_middleware",
]


def _resolve_subagent_model(
    model: str | dict[str, Any] | LLMModel | BaseChatModel | None,
) -> str | BaseChatModel | None:
    """Resolve a subagent model value to a type accepted by deepagents.

    deepagents ``SubAgent.model`` accepts ``str | BaseChatModel``.  This
    helper bridges DAO AI configuration types so users can provide:

    - ``str`` -- passed through (e.g., ``"openai:gpt-4o"``)
    - ``dict`` -- treated as ``LLMModel`` kwargs, converted via
      ``LLMModel(**dict).as_chat_model()`` which returns a
      ``ChatDatabricks`` instance.
    - ``LLMModel`` -- converted via ``as_chat_model()``.
    - ``BaseChatModel`` -- passed through (e.g., ``ChatDatabricks``).
    - ``None`` -- returned as-is (deepagents uses the parent model).

    Args:
        model: The raw model value from a subagent specification.

    Returns:
        A resolved model value compatible with deepagents.

    Raises:
        TypeError: If *model* is an unsupported type.
    """
    if model is None:
        return None

    if isinstance(model, str):
        return model

    if isinstance(model, BaseChatModel):
        return model

    # Import LLMModel at runtime to avoid circular imports.
    from dao_ai.config import LLMModel

    if isinstance(model, dict):
        logger.debug(
            "Resolving subagent model from dict",
            model_name=model.get("name", "unknown"),
        )
        return LLMModel(**model).as_chat_model()

    if isinstance(model, LLMModel):
        logger.debug(
            "Resolving subagent model from LLMModel",
            model_name=model.name,
        )
        return model.as_chat_model()

    msg = f"Unsupported subagent model type: {type(model).__name__}"
    raise TypeError(msg)


def _resolve_subagent_models(
    subagents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Iterate subagent specs and resolve each ``model`` field in place.

    Returns a new list of spec dicts with resolved model values.  The
    original dicts are not mutated.
    """
    resolved: list[dict[str, Any]] = []
    for spec in subagents:
        spec = dict(spec)  # defensive copy
        if "model" in spec:
            spec["model"] = _resolve_subagent_model(spec["model"])
        resolved.append(spec)
    return resolved


def create_subagent_middleware(
    subagents: list[dict[str, Any]] | None = None,
    backend_type: str = "state",
    root_dir: str | None = None,
    volume_path: str | VolumePathModel | None = None,
    system_prompt: str | None = None,
    task_description: str | None = None,
) -> SubAgentMiddleware:
    """
    Create a SubAgentMiddleware for spawning subagents via a ``task`` tool.

    This factory function creates a SubAgentMiddleware from the Deep Agents
    library that provides agents with the ability to spawn subagents for
    handling complex, multi-step tasks with isolated context windows.

    Subagents are useful for:
        - Complex research tasks that need focused, isolated context
        - Parallel execution of independent subtasks
        - Domain-specific expertise requiring a narrow tool subset
        - Offloading heavy token/context usage from the main thread

    Each subagent specification must include:
        - ``name``: Unique identifier
        - ``description``: What the subagent does (used for delegation)
        - ``system_prompt``: Instructions for the subagent

    Optional subagent fields:
        - ``model``: Override model.  Accepts a ``str`` (e.g.,
          ``"openai:gpt-4o"``), a ``dict`` of ``LLMModel`` kwargs (e.g.,
          ``{"name": "my-endpoint", "temperature": 0.1}``), an ``LLMModel``
          instance, or a ``BaseChatModel`` instance (e.g.,
          ``ChatDatabricks``).  Dicts and ``LLMModel`` values are
          automatically converted to ``ChatDatabricks`` via
          ``LLMModel.as_chat_model()``.
        - ``tools``: Tools the subagent can use
        - ``middleware``: Additional middleware for the subagent

    Args:
        subagents: List of subagent specification dicts. Each must include
            ``name``, ``description``, and ``system_prompt``. If None, only
            a general-purpose subagent is created.
        backend_type: Backend for file storage. One of ``"state"``
            (ephemeral, default), ``"filesystem"`` (real disk),
            ``"store"`` (persistent), or ``"volume"`` (Databricks
            Unity Catalog Volume).
        root_dir: Root directory for filesystem backend. Required when
            ``backend_type="filesystem"``.
        volume_path: Volume path for Databricks Volume backend.
            Required when ``backend_type="volume"``.
        system_prompt: Custom system prompt for guiding task tool usage.
        task_description: Custom description for the ``task`` tool.

    Returns:
        A configured SubAgentMiddleware instance.

    Example:
        from dao_ai.middleware import create_subagent_middleware

        # String model (passed through to deepagents):
        middleware = create_subagent_middleware(
            subagents=[
                {
                    "name": "code-reviewer",
                    "description": "Reviews code for bugs",
                    "system_prompt": "You are a code reviewer.",
                    "model": "openai:gpt-4o",
                }
            ],
        )

        # LLMModel dict (converted to ChatDatabricks):
        middleware = create_subagent_middleware(
            subagents=[
                {
                    "name": "analyst",
                    "description": "Analyzes data",
                    "system_prompt": "You are a data analyst.",
                    "model": {
                        "name": "databricks-meta-llama-3-3-70b-instruct",
                        "temperature": 0.1,
                    },
                }
            ],
        )
    """
    backend = resolve_backend(
        backend_type=backend_type,
        root_dir=root_dir,
        volume_path=volume_path,
    )

    subagent_names = [s.get("name", "unnamed") for s in (subagents or [])]
    logger.debug(
        "Creating SubAgentMiddleware",
        backend_type=backend_type,
        subagent_count=len(subagents or []),
        subagent_names=subagent_names,
    )

    kwargs: dict[str, Any] = {
        "backend": backend,
    }
    if subagents is not None:
        kwargs["subagents"] = _resolve_subagent_models(subagents)
    if system_prompt is not None:
        kwargs["system_prompt"] = system_prompt
    if task_description is not None:
        kwargs["task_description"] = task_description

    middleware = SubAgentMiddleware(**kwargs)

    logger.info(
        "SubAgentMiddleware created",
        backend_type=backend_type,
        subagent_count=len(subagents or []),
    )
    return middleware
