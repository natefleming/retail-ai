"""Tests for ``_DaoAiStreamCollector`` — the per-request LangChain callback
handler that captures MCP notifications *and* tool-call lifecycle events onto a
queue for the streaming agent to drain into SSE items.

Tool timing is computed by the collector itself (LangChain callbacks do not
carry span durations), keyed by the tool run's ``run_id``.

The project has no pytest-asyncio configured, so async handler methods are
driven with ``asyncio.run()`` (matching the existing test suite convention).
"""

import asyncio
from typing import Any, Awaitable, Callable
from uuid import uuid4

import pytest

from dao_ai.models import _DaoAiStreamCollector


def _run(coro: Awaitable[Any]) -> Any:
    """Drive a coroutine on a fresh loop, then restore a fresh current loop.

    ``asyncio.run`` closes its loop and leaves no current loop, which breaks
    sibling tests that use the deprecated ``asyncio.get_event_loop()`` idiom.
    Restoring a fresh open loop keeps the suite order-independent.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        asyncio.set_event_loop(asyncio.new_event_loop())


def _drain(queue: "asyncio.Queue") -> list[dict]:
    out: list[dict] = []
    while not queue.empty():
        out.append(queue.get_nowait())
    return out


def _collect(
    scenario: Callable[[_DaoAiStreamCollector], Awaitable[None]],
) -> list[dict]:
    """Run an async scenario against a fresh collector and return its envelopes."""

    async def _driver() -> list[dict]:
        queue: asyncio.Queue = asyncio.Queue()
        collector = _DaoAiStreamCollector(queue)
        await scenario(collector)
        return _drain(queue)

    return _run(_driver())


class TestDaoAiStreamCollectorMcp:
    """The MCP behavior of the original collector must be preserved."""

    @pytest.mark.unit
    def test_mcp_custom_event_enqueued(self) -> None:
        async def scenario(c: _DaoAiStreamCollector) -> None:
            await c.on_custom_event(
                "mcp.progress",
                {"channel": "mcp.progress", "server_name": "fs", "progress": 0.5},
                run_id=uuid4(),
            )

        envelopes = _collect(scenario)
        assert len(envelopes) == 1
        assert envelopes[0]["channel"] == "mcp.progress"

    @pytest.mark.unit
    def test_non_mcp_custom_event_ignored(self) -> None:
        async def scenario(c: _DaoAiStreamCollector) -> None:
            await c.on_custom_event(
                "other.event", {"channel": "other.event"}, run_id=uuid4()
            )

        assert _collect(scenario) == []


class TestDaoAiStreamCollectorToolLifecycle:
    """Tool start/end/error are captured as dao_ai.tool.* envelopes."""

    @pytest.mark.unit
    def test_tool_start_enqueues_call_with_args(self) -> None:
        run_id = uuid4()

        async def scenario(c: _DaoAiStreamCollector) -> None:
            await c.on_tool_start(
                {"name": "search_docs"},
                "query string",
                run_id=run_id,
                inputs={"query": "hello"},
            )

        envelopes = _collect(scenario)
        assert len(envelopes) == 1
        env = envelopes[0]
        assert env["channel"] == "dao_ai.tool.start"
        assert env["call_id"] == str(run_id)
        assert env["name"] == "search_docs"
        assert env["arguments"] == {"query": "hello"}
        assert "started_at" in env

    @pytest.mark.unit
    def test_tool_end_reports_duration_and_matches_call_id(self) -> None:
        run_id = uuid4()

        async def scenario(c: _DaoAiStreamCollector) -> None:
            await c.on_tool_start(
                {"name": "search_docs"}, "q", run_id=run_id, inputs={}
            )
            await c.on_tool_end("the tool result", run_id=run_id)

        envelopes = _collect(scenario)
        end = [e for e in envelopes if e["channel"] == "dao_ai.tool.end"]
        assert len(end) == 1
        assert end[0]["call_id"] == str(run_id)
        assert isinstance(end[0]["duration_ms"], float)
        assert end[0]["duration_ms"] >= 0.0
        assert "the tool result" in end[0]["result_summary"]

    @pytest.mark.unit
    def test_tool_error_reports_error_and_duration(self) -> None:
        run_id = uuid4()

        async def scenario(c: _DaoAiStreamCollector) -> None:
            await c.on_tool_start({"name": "flaky"}, "q", run_id=run_id, inputs={})
            await c.on_tool_error(ValueError("boom"), run_id=run_id)

        envelopes = _collect(scenario)
        err = [e for e in envelopes if e["channel"] == "dao_ai.tool.error"]
        assert len(err) == 1
        assert err[0]["call_id"] == str(run_id)
        assert "boom" in err[0]["error"]
        assert isinstance(err[0]["duration_ms"], float)
