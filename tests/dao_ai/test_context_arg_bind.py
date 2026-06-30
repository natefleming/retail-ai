"""Tests for ContextArgBindMiddleware.

Pins behavior across the scenarios this middleware MUST handle cleanly:

1. Multiple tools per agent (UC fn + VS + MCP + REST + handoff)
2. Multiple tool calls in one turn
3. Concurrent (parallel asyncio) tool calls
4. LLM-provided value wins (no override)
5. Tool without ``args_schema`` is never touched
6. Tool's schema doesn't declare the bound arg → no-op for that tool
7. Multi-placeholder templates render correctly
8. Missing context field is logged + skipped, not raised
9. Bindings against extra Context fields (extra="allow") work
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from dao_ai.middleware.context_arg_bind import (
    ContextArgBindMiddleware,
    create_context_arg_bind_middleware,
)
from dao_ai.state import Context


# ---------------------------------------------------------------------------
# Helpers — fake tools + a minimal ToolCallRequest stand-in
# ---------------------------------------------------------------------------


class _OrderHistoryArgs(BaseModel):
    customer_id: str = Field(description="customer id")
    row_limit: int = Field(default=10)


class _ProductSearchArgs(BaseModel):
    query: str
    k: int = 3


class _NoSchemaTool(BaseTool):
    """A tool that exposes no args_schema (legacy / dynamic StructuredTool)."""

    name: str = "no_schema_tool"
    description: str = "tool without args_schema"

    def _run(self, **kwargs: Any) -> str:
        return "ok"


def _make_tool(name: str, schema: type[BaseModel] | None) -> BaseTool:
    t = MagicMock(spec=BaseTool)
    t.name = name
    t.args_schema = schema
    return t


def _make_request(
    *,
    tool: BaseTool,
    tool_call: dict[str, Any],
    context: Context | None,
) -> Any:
    runtime = MagicMock()
    runtime.context = context
    req = MagicMock()
    req.tool = tool
    req.tool_call = tool_call
    req.runtime = runtime
    # override() returns a new request with a patched tool_call
    def _override(**updates: Any) -> Any:
        new_req = _make_request(
            tool=tool,
            tool_call=updates.get("tool_call", tool_call),
            context=context,
        )
        new_req._is_override = True
        return new_req
    req.override = _override
    return req


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_fills_missing_arg_from_context() -> None:
    """Core case: LLM left customer_id blank, middleware fills from
    runtime.context.user_id."""
    mw = ContextArgBindMiddleware(bindings={"customer_id": "{user_id}"})
    tool = _make_tool("get_order_history_uc", _OrderHistoryArgs)
    ctx = Context(user_id="C0042", thread_id="t1")
    req = _make_request(
        tool=tool,
        tool_call={"name": "get_order_history_uc", "args": {"row_limit": 10}, "id": "c1"},
        context=ctx,
    )

    captured: dict[str, Any] = {}

    def handler(r: Any) -> ToolMessage:
        captured.update(r.tool_call.get("args") or {})
        return ToolMessage(content="ok", tool_call_id="c1")

    mw.wrap_tool_call(req, handler)
    assert captured["customer_id"] == "C0042"
    assert captured["row_limit"] == 10


@pytest.mark.unit
def test_does_not_override_llm_value() -> None:
    """If the LLM provided a value, the middleware never overrides it."""
    mw = ContextArgBindMiddleware(bindings={"customer_id": "{user_id}"})
    tool = _make_tool("get_order_history_uc", _OrderHistoryArgs)
    ctx = Context(user_id="C0042")
    req = _make_request(
        tool=tool,
        tool_call={"name": "get_order_history_uc", "args": {"customer_id": "C9999"}, "id": "c1"},
        context=ctx,
    )
    captured: dict[str, Any] = {}

    def handler(r: Any) -> ToolMessage:
        captured.update(r.tool_call.get("args") or {})
        return ToolMessage(content="ok", tool_call_id="c1")

    mw.wrap_tool_call(req, handler)
    assert captured["customer_id"] == "C9999"


@pytest.mark.unit
def test_skips_tool_whose_schema_does_not_declare_the_arg() -> None:
    """product_search has no ``customer_id`` field → middleware no-ops."""
    mw = ContextArgBindMiddleware(bindings={"customer_id": "{user_id}"})
    tool = _make_tool("product_search", _ProductSearchArgs)
    ctx = Context(user_id="C0042")
    req = _make_request(
        tool=tool,
        tool_call={"name": "product_search", "args": {"query": "cake", "k": 5}, "id": "c1"},
        context=ctx,
    )
    captured: dict[str, Any] = {}

    def handler(r: Any) -> ToolMessage:
        captured.update(r.tool_call.get("args") or {})
        return ToolMessage(content="ok", tool_call_id="c1")

    mw.wrap_tool_call(req, handler)
    assert "customer_id" not in captured
    assert captured == {"query": "cake", "k": 5}


@pytest.mark.unit
def test_skips_tool_without_args_schema() -> None:
    """If the tool has no declared schema, we can't safely add args — skip."""
    mw = ContextArgBindMiddleware(bindings={"customer_id": "{user_id}"})
    tool = _make_tool("no_schema_tool", None)
    ctx = Context(user_id="C0042")
    req = _make_request(
        tool=tool,
        tool_call={"name": "no_schema_tool", "args": {"foo": "bar"}, "id": "c1"},
        context=ctx,
    )
    captured: dict[str, Any] = {}

    def handler(r: Any) -> ToolMessage:
        captured.update(r.tool_call.get("args") or {})
        return ToolMessage(content="ok", tool_call_id="c1")

    mw.wrap_tool_call(req, handler)
    assert "customer_id" not in captured


@pytest.mark.unit
def test_multi_placeholder_template() -> None:
    """Templates can reference multiple Context fields."""

    class _Args(BaseModel):
        idempotency_key: str

    mw = ContextArgBindMiddleware(
        bindings={"idempotency_key": "{user_id}-{thread_id}"}
    )
    tool = _make_tool("ucp_action", _Args)
    ctx = Context(user_id="C0042", thread_id="t-1234")
    req = _make_request(
        tool=tool,
        tool_call={"name": "ucp_action", "args": {}, "id": "c1"},
        context=ctx,
    )
    captured: dict[str, Any] = {}

    def handler(r: Any) -> ToolMessage:
        captured.update(r.tool_call.get("args") or {})
        return ToolMessage(content="ok", tool_call_id="c1")

    mw.wrap_tool_call(req, handler)
    assert captured["idempotency_key"] == "C0042-t-1234"


@pytest.mark.unit
def test_missing_context_field_is_skipped_not_raised() -> None:
    """Template references {foo} but Context has no field foo → skip, no raise."""

    class _Args(BaseModel):
        thing: str

    mw = ContextArgBindMiddleware(bindings={"thing": "{nonexistent_field}"})
    tool = _make_tool("some_tool", _Args)
    ctx = Context(user_id="C0042")  # no nonexistent_field
    req = _make_request(
        tool=tool,
        tool_call={"name": "some_tool", "args": {}, "id": "c1"},
        context=ctx,
    )
    captured: dict[str, Any] = {}

    def handler(r: Any) -> ToolMessage:
        captured.update(r.tool_call.get("args") or {})
        return ToolMessage(content="ok", tool_call_id="c1")

    # Must not raise.
    mw.wrap_tool_call(req, handler)
    # Tool call still runs with what the LLM provided (nothing here).
    assert "thing" not in captured


@pytest.mark.unit
def test_works_against_extra_context_fields() -> None:
    """Context allows extra fields (e.g. store_num). Bindings against them resolve."""

    class _Args(BaseModel):
        store_num: int

    mw = ContextArgBindMiddleware(bindings={"store_num": "{store_num}"})
    tool = _make_tool("store_tool", _Args)
    ctx = Context(user_id="C0042")  # add extra at runtime
    ctx_dict = ctx.model_dump()
    # Build a Context with extra field via the same model_config (extra="allow")
    ctx_with_store = Context.model_validate({**ctx_dict, "store_num": "101"})
    req = _make_request(
        tool=tool,
        tool_call={"name": "store_tool", "args": {}, "id": "c1"},
        context=ctx_with_store,
    )
    captured: dict[str, Any] = {}

    def handler(r: Any) -> ToolMessage:
        captured.update(r.tool_call.get("args") or {})
        return ToolMessage(content="ok", tool_call_id="c1")

    mw.wrap_tool_call(req, handler)
    assert captured["store_num"] == "101"


@pytest.mark.unit
def test_concurrent_tool_calls_are_independent() -> None:
    """Run many tool_calls through the SAME middleware instance in parallel
    via asyncio.gather. Each call should fill independently — no shared state."""

    mw = ContextArgBindMiddleware(bindings={"customer_id": "{user_id}"})
    tool = _make_tool("get_order_history_uc", _OrderHistoryArgs)

    async def _one_call(uid: str) -> str:
        ctx = Context(user_id=uid)
        req = _make_request(
            tool=tool,
            tool_call={"name": "get_order_history_uc", "args": {"row_limit": 10}, "id": f"c-{uid}"},
            context=ctx,
        )
        result: dict[str, Any] = {}

        async def handler(r: Any) -> ToolMessage:
            # Yield to event loop to allow interleaving
            await asyncio.sleep(0.001)
            result["args"] = dict(r.tool_call.get("args") or {})
            return ToolMessage(content="ok", tool_call_id=r.tool_call["id"])

        await mw.awrap_tool_call(req, handler)
        return result["args"]["customer_id"]

    async def _run() -> list[str]:
        return await asyncio.gather(*[_one_call(f"U{i:04d}") for i in range(25)])

    seen = asyncio.run(_run())
    expected = [f"U{i:04d}" for i in range(25)]
    assert seen == expected, "concurrent calls leaked state across requests"


@pytest.mark.unit
def test_empty_bindings_is_noop() -> None:
    """No bindings configured → middleware is a transparent pass-through."""
    mw = ContextArgBindMiddleware(bindings={})
    tool = _make_tool("any_tool", _OrderHistoryArgs)
    req = _make_request(
        tool=tool,
        tool_call={"name": "any_tool", "args": {"customer_id": "X"}, "id": "c1"},
        context=Context(user_id="C0042"),
    )
    captured: dict[str, Any] = {}

    def handler(r: Any) -> ToolMessage:
        captured.update(r.tool_call.get("args") or {})
        return ToolMessage(content="ok", tool_call_id="c1")

    mw.wrap_tool_call(req, handler)
    assert captured == {"customer_id": "X"}


@pytest.mark.unit
def test_factory_returns_configured_instance() -> None:
    """The dao-ai FQN factory pattern: YAML-declared name + args → instance."""
    mw = create_context_arg_bind_middleware(bindings={"customer_id": "{user_id}"})
    assert isinstance(mw, ContextArgBindMiddleware)
