"""Tests for ``create_search_user_profile_tool``.

The tool must resolve ``user_id`` from the LangChain ``ToolRuntime`` when the
LLM omits the argument (the common case — the runtime context already carries
the authenticated user, and the LLM shouldn't have to learn it). An explicit
LLM-supplied ``user_id`` continues to override for cross-user lookups.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Sequence

import pytest
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from dao_ai.state import Context
from dao_ai.tools.memory import create_search_user_profile_tool


@dataclass
class _FakeRuntime:
    """Minimal stand-in for ``ToolRuntime`` — duck-typing the only attribute
    ``search_user_profile_wrapper`` reads (``context``)."""

    context: Context | None


def _seed(store: BaseStore, namespace: Sequence[str], payload: dict) -> None:
    asyncio.run(store.aput(tuple(namespace), "default", payload))


def _invoke(tool, **kwargs):
    return asyncio.run(tool.coroutine(**kwargs))


@pytest.mark.unit
def test_runtime_user_id_resolves_namespace_when_llm_arg_empty() -> None:
    store = InMemoryStore()
    _seed(
        store,
        ["memory", "nate_fleming@databricks_com"],
        {"name": "Nate", "preferences": ["likes brunch"]},
    )
    tool = create_search_user_profile_tool(
        store=store, namespace=("memory", "{user_id}")
    )
    runtime = _FakeRuntime(context=Context(user_id="nate_fleming@databricks_com"))

    result = _invoke(tool, user_id="", runtime=runtime)

    assert result == {"name": "Nate", "preferences": ["likes brunch"]}


@pytest.mark.unit
def test_explicit_user_id_overrides_runtime_context() -> None:
    store = InMemoryStore()
    _seed(store, ["memory", "alice"], {"name": "Alice"})
    _seed(store, ["memory", "bob"], {"name": "Bob"})
    tool = create_search_user_profile_tool(
        store=store, namespace=("memory", "{user_id}")
    )
    runtime = _FakeRuntime(context=Context(user_id="alice"))

    result = _invoke(tool, user_id="bob", runtime=runtime)

    assert result == {"name": "Bob"}


@pytest.mark.unit
def test_no_runtime_and_empty_arg_returns_placeholder() -> None:
    store = InMemoryStore()
    tool = create_search_user_profile_tool(
        store=store, namespace=("memory", "{user_id}")
    )

    # Both runtime and arg empty — preserves prior behavior.
    result = _invoke(tool, user_id="", runtime=None)

    assert isinstance(result, str)
    assert "No user profile found" in result


@pytest.mark.unit
def test_runtime_with_null_user_id_returns_placeholder() -> None:
    """``Context.user_id`` may be ``None`` when no user identity is attached
    (e.g. an unauthenticated test harness). The tool must not blow up — it
    falls back to the empty-namespace miss."""
    store = InMemoryStore()
    tool = create_search_user_profile_tool(
        store=store, namespace=("memory", "{user_id}")
    )
    runtime = _FakeRuntime(context=Context(user_id=None))

    result = _invoke(tool, user_id="", runtime=runtime)

    assert isinstance(result, str)
    assert "No user profile found" in result


@pytest.mark.unit
def test_tool_metadata_unchanged() -> None:
    """LLM-facing tool name + args schema must not change — only the internal
    wrapper signature gains the (LLM-invisible) ``runtime`` parameter."""
    store = InMemoryStore()
    tool = create_search_user_profile_tool(
        store=store, namespace=("memory", "{user_id}")
    )

    assert tool.name == "search_user_profile"
    # Single user-visible field: user_id (optional override).
    schema = tool.args_schema.model_json_schema()
    assert list(schema["properties"].keys()) == ["user_id"]
