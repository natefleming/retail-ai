"""
Todo list middleware for DAO AI agents.

This module provides a factory function for creating TodoListMiddleware instances
that equip agents with task planning and tracking capabilities via a `write_todos`
tool.

The TodoListMiddleware comes from LangChain and is also used by Deep Agents as
a core planning component. It adds a `write_todos` tool and injects system prompts
guiding agents to break complex tasks into manageable steps.

Example:
    from dao_ai.middleware import create_todo_list_middleware

    middleware = create_todo_list_middleware()

    # Or with custom prompts:
    middleware = create_todo_list_middleware(
        system_prompt="Use write_todos for multi-step tasks...",
    )

YAML Config:
    middleware:
      - name: dao_ai.middleware.todo.create_todo_list_middleware
        args:
          system_prompt: "Custom planning instructions..."
"""

from __future__ import annotations

from langchain.agents.middleware import TodoListMiddleware
from langchain.agents.middleware.todo import PlanningState, Todo
from loguru import logger

from dao_ai.config import PromptModel
from dao_ai.middleware._prompt_utils import resolve_prompt

__all__ = [
    "TodoListMiddleware",
    "Todo",
    "PlanningState",
    "create_todo_list_middleware",
]


def create_todo_list_middleware(
    system_prompt: str | PromptModel | None = None,
    tool_description: str | None = None,
) -> TodoListMiddleware:
    """
    Create a TodoListMiddleware for agent task planning and tracking.

    This factory function creates a TodoListMiddleware that provides agents with
    a ``write_todos`` tool for creating and managing structured task lists. The
    middleware automatically injects system prompts guiding the agent on when and
    how to use the todo functionality effectively.

    The ``write_todos`` tool is enforced to be called at most once per model turn,
    since it replaces the entire todo list and parallel calls would create ambiguity.

    Each todo item has:
        - ``content``: The task description
        - ``status``: One of ``"pending"``, ``"in_progress"``, ``"completed"``

    Args:
        system_prompt: Custom system prompt to guide todo usage. If None, uses
            the built-in prompt from LangChain. Accepts a plain string or a
            ``PromptModel`` from the prompt registry.
        tool_description: Custom description for the ``write_todos`` tool. If None,
            uses the built-in description from LangChain.

    Returns:
        A configured TodoListMiddleware instance.

    Example:
        from dao_ai.middleware import create_todo_list_middleware

        # Default configuration
        middleware = create_todo_list_middleware()

        # Custom system prompt
        middleware = create_todo_list_middleware(
            system_prompt="Always create a todo list before starting work."
        )
    """
    kwargs: dict[str, str] = {}
    if system_prompt is not None:
        kwargs["system_prompt"] = resolve_prompt(system_prompt)
    if tool_description is not None:
        kwargs["tool_description"] = tool_description

    logger.debug(
        "Creating TodoListMiddleware",
        custom_system_prompt=system_prompt is not None,
        custom_tool_description=tool_description is not None,
    )

    middleware = TodoListMiddleware(**kwargs)

    logger.info("TodoListMiddleware created")
    return middleware
