"""Tests for the server-side capabilities module (PR 2).

Covers two concerns:

1. `register_resources` and `register_prompts` bind resources/prompts on the
   FastMCP instance and return the URIs/names for `/healthz` advertisement.
2. `_heartbeat_progress` emits `ctx.report_progress` at fixed intervals with
   monotonically increasing values, halts on cancel, and never overshoots
   the 95 ceiling reserved for the terminal `agent_complete` step.

No Databricks credentials required — everything is in-process.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest
from mcp.server.fastmcp import FastMCP

from dao_ai.config import (
    McpPromptArgumentModel,
    McpPromptModel,
    McpResourceModel,
)
from dao_ai.mcp.agent_tool import _heartbeat_progress
from dao_ai.mcp.server_capabilities import (
    register_prompts,
    register_resources,
)


def _mcp() -> FastMCP:
    return FastMCP("test-server")


class TestRegisterResources:
    def test_returns_registered_uris(self) -> None:
        mcp = _mcp()
        uris = register_resources(
            mcp,
            [
                McpResourceModel(
                    uri="dao-ai://prompts/system",
                    name="system",
                    content="You are helpful.",
                ),
                McpResourceModel(
                    uri="dao-ai://prompts/style",
                    name="style",
                    content="Answer concisely.",
                ),
            ],
        )
        assert uris == ["dao-ai://prompts/system", "dao-ai://prompts/style"]

    def test_empty_list_registers_nothing(self) -> None:
        mcp = _mcp()
        uris = register_resources(mcp, [])
        assert uris == []


class TestRegisterPrompts:
    def test_returns_registered_names(self) -> None:
        mcp = _mcp()
        names = register_prompts(
            mcp,
            [
                McpPromptModel(
                    name="greet",
                    template="Hello, {name}!",
                    arguments=[McpPromptArgumentModel(name="name", required=True)],
                ),
                McpPromptModel(
                    name="analyze",
                    template="Analyze: {query}",
                    arguments=[McpPromptArgumentModel(name="query", required=True)],
                ),
            ],
        )
        assert names == ["greet", "analyze"]

    def test_empty_list_registers_nothing(self) -> None:
        mcp = _mcp()
        names = register_prompts(mcp, [])
        assert names == []


class TestHeartbeatProgress:
    def test_emits_ascending_values_up_to_ceiling(self) -> None:
        """The heartbeat must emit monotonic progress and never exceed 95."""
        ctx = MagicMock()
        emitted: list[tuple[float, float | None, str | None]] = []

        async def _report(
            progress: float, total: float | None = None, message: str | None = None
        ) -> None:
            emitted.append((progress, total, message))

        ctx.report_progress = _report

        async def _run() -> None:
            task = asyncio.create_task(_heartbeat_progress(ctx, "probe"))
            # Wait long enough for ~3 heartbeats (interval = 2s → give it 7s).
            await asyncio.sleep(7)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(_run())
        assert emitted, "no heartbeats emitted"
        progresses = [e[0] for e in emitted]
        # Monotonic
        for a, b in zip(progresses, progresses[1:]):
            assert b >= a
        # Ceiling
        assert max(progresses) <= 95.0
        # Message tag
        assert all(e[2] == "agent_in_flight" for e in emitted)

    def test_cancels_cleanly(self) -> None:
        """Cancelling the heartbeat must raise CancelledError, not swallow it."""
        ctx = MagicMock()

        async def _report(*a: Any, **kw: Any) -> None:
            return None

        ctx.report_progress = _report

        async def _run() -> None:
            task = asyncio.create_task(_heartbeat_progress(ctx, "probe"))
            await asyncio.sleep(0.1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(_run())

    def test_silences_report_progress_failures(self) -> None:
        """If ctx.report_progress raises, the heartbeat returns quietly."""
        ctx = MagicMock()

        async def _report(*a: Any, **kw: Any) -> None:
            raise RuntimeError("channel closed")

        ctx.report_progress = _report

        async def _run() -> None:
            task = asyncio.create_task(_heartbeat_progress(ctx, "probe"))
            # Give it time to hit the first sleep + raise on first report.
            await asyncio.sleep(2.5)
            # Task should have exited on its own.
            assert task.done()

        asyncio.run(_run())
