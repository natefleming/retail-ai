"""Tier 2 integration tests for MCP advanced capabilities.

Spins up a real ``FastMCP`` server in a background thread on an ephemeral
port and drives ``create_mcp_tools()`` at it. Verifies that our
callback + interceptor wiring receives progress / structured-content
notifications from a real MCP transport, and that the default
(``capabilities=None``) path stays silent.

No Databricks credentials required — everything is localhost.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from contextlib import closing
from typing import Iterator

import pytest
import uvicorn
from mcp.server.fastmcp import Context, FastMCP
from pydantic import BaseModel

from dao_ai.config import McpCapabilitiesModel, McpFunctionModel
from dao_ai.tools.mcp import acreate_mcp_tools


class _StructuredAnswer(BaseModel):
    answer: str
    confidence: float


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _build_probe_server(name: str) -> FastMCP:
    # Stateful HTTP: notifications channel is preserved across the call.
    # With stateless_http=True the session terminates before progress /
    # logging notifications reach the client.
    mcp = FastMCP(name)

    @mcp.tool()
    async def long_task(steps: int, ctx: Context) -> str:
        for i in range(1, steps + 1):
            await ctx.report_progress(
                progress=float(i),
                total=float(steps),
                message=f"step {i}/{steps}",
            )
        return f"completed {steps} steps"

    @mcp.tool()
    async def structured_result(query: str, ctx: Context) -> _StructuredAnswer:
        # Typed return: FastMCP populates CallToolResult.structuredContent.
        return _StructuredAnswer(answer=f"echo:{query}", confidence=0.87)

    @mcp.tool()
    async def raise_tool_error(ctx: Context) -> str:
        raise ValueError("intentional error")

    return mcp


class _ServerThread:
    def __init__(self, port: int, app) -> None:
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            lifespan="on",
        )
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self) -> None:
        self.thread.start()
        deadline = time.time() + 10.0
        while time.time() < deadline:
            if self.server.started:
                return
            time.sleep(0.05)
        raise RuntimeError("uvicorn did not start in 10s")

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5.0)


@pytest.fixture(scope="module")
def probe_url() -> Iterator[str]:
    port = _free_port()
    mcp = _build_probe_server("caps_probe")
    server = _ServerThread(port, mcp.streamable_http_app())
    server.start()
    try:
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        server.stop()


def _fn(url: str, capabilities: McpCapabilitiesModel | None) -> McpFunctionModel:
    return McpFunctionModel(url=url, capabilities=capabilities)


class _RecordingProgress:
    """Drop-in for DaoAiProgressCallback that also records calls."""

    events: list[tuple] = []

    def __init__(self) -> None:
        pass

    async def __call__(self, progress, total, message, context) -> None:
        _RecordingProgress.events.append(
            (progress, total, message, context.tool_name)
        )


class _RecordingStructured:
    """Interceptor spy that records the CallToolResult it sees."""

    seen: list[dict] = []

    def __init__(self) -> None:
        pass

    async def __call__(self, request, handler):
        result = await handler(request)
        from mcp.types import CallToolResult

        if isinstance(result, CallToolResult):
            _RecordingStructured.seen.append(
                {
                    "has_structured": getattr(result, "structuredContent", None)
                    is not None,
                    "structured": result.structuredContent,
                    "server_name": request.server_name,
                    "tool_name": request.name,
                }
            )
        return result


class TestClassicPathParity:
    def test_tools_discovered(self, probe_url: str) -> None:
        async def run() -> None:
            tools = await acreate_mcp_tools(_fn(probe_url, None))
            names = {t.name for t in tools}
            assert "long_task" in names
            assert "structured_result" in names

        asyncio.run(run())

    def test_long_task_invokable(self, probe_url: str) -> None:
        async def run() -> None:
            tools = await acreate_mcp_tools(_fn(probe_url, None))
            long_task = next(t for t in tools if t.name == "long_task")
            result = await long_task.ainvoke({"steps": 3})
            assert "completed" in str(result)

        asyncio.run(run())


class TestProgressCapability:
    def test_progress_callback_receives_notifications(
        self, probe_url: str, monkeypatch
    ) -> None:
        """Server emits 5 progress events; our progress callback should be
        invoked exactly 5 times with monotonically increasing progress."""
        _RecordingProgress.events = []
        # Swap the class used by _call_tool_with_capabilities.
        monkeypatch.setattr(
            "dao_ai.tools.mcp.DaoAiProgressCallback",
            _RecordingProgress,
            raising=False,
        )
        # It's imported lazily inside _call_tool_with_capabilities from
        # dao_ai.tools.mcp_callbacks — patch there too.
        monkeypatch.setattr(
            "dao_ai.tools.mcp_callbacks.DaoAiProgressCallback",
            _RecordingProgress,
        )

        caps = McpCapabilitiesModel(progress=True, structured_output=False)

        async def run() -> None:
            tools = await acreate_mcp_tools(_fn(probe_url, caps))
            long_task = next(t for t in tools if t.name == "long_task")
            result = await long_task.ainvoke({"steps": 5})
            assert "completed 5" in str(result)

        asyncio.run(run())
        assert (
            len(_RecordingProgress.events) == 5
        ), f"expected 5 progress events, got {len(_RecordingProgress.events)}: {_RecordingProgress.events}"
        progresses = [e[0] for e in _RecordingProgress.events]
        assert progresses == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert all(e[1] == 5.0 for e in _RecordingProgress.events)
        assert all(e[3] == "long_task" for e in _RecordingProgress.events)


class TestStructuredOutputCapability:
    def test_structured_content_returned(self, probe_url: str, monkeypatch) -> None:
        _RecordingStructured.seen = []
        monkeypatch.setattr(
            "dao_ai.tools.mcp_interceptors.DaoAiStructuredOutputInterceptor",
            _RecordingStructured,
        )

        caps = McpCapabilitiesModel(structured_output=True)

        async def run() -> None:
            tools = await acreate_mcp_tools(_fn(probe_url, caps))
            sr = next(t for t in tools if t.name == "structured_result")
            await sr.ainvoke({"query": "ping"})

        asyncio.run(run())
        assert _RecordingStructured.seen, "interceptor never ran"
        record = _RecordingStructured.seen[-1]
        assert record["tool_name"] == "structured_result"
        assert record["has_structured"] is True
        assert record["structured"] is not None


class TestTraceContextMeta:
    def test_meta_trace_context_merged_on_each_call(
        self, probe_url: str, monkeypatch
    ) -> None:
        """When capabilities is set, W3C trace context is merged into _meta
        for every tool call — proves the SEP-414 inject site fires end-to-end
        over a real transport."""
        calls: list[dict | None] = []

        def _fake_merge(meta):
            merged = dict(meta or {})
            merged.setdefault("traceparent", "00-" + "a" * 32 + "-" + "b" * 16 + "-01")
            calls.append(merged)
            return merged

        monkeypatch.setattr(
            "dao_ai.tools.mcp_trace_context.merge_trace_context_meta",
            _fake_merge,
        )

        caps = McpCapabilitiesModel(structured_output=True)

        async def run() -> None:
            tools = await acreate_mcp_tools(_fn(probe_url, caps))
            sr = next(t for t in tools if t.name == "structured_result")
            await sr.ainvoke({"query": "hi"})

        asyncio.run(run())
        assert calls, "trace-context merge never ran on the capabilities path"
        assert calls[-1]["traceparent"].startswith("00-")


class TestErrorHandling:
    def test_isError_surfaces_as_text_on_classic_path(self, probe_url: str) -> None:
        """dao-ai's custom wrapper has always returned server-side error text
        rather than raising. Verify that contract persists on the 0.3.0 bump."""

        async def run() -> str:
            tools = await acreate_mcp_tools(_fn(probe_url, None))
            err = next(t for t in tools if t.name == "raise_tool_error")
            return await err.ainvoke({})

        result = str(asyncio.run(run()))
        assert "intentional error" in result.lower()

    def test_isError_surfaces_as_text_on_capabilities_path(
        self, probe_url: str
    ) -> None:
        caps = McpCapabilitiesModel(progress=True, structured_output=True)

        async def run() -> str:
            tools = await acreate_mcp_tools(_fn(probe_url, caps))
            err = next(t for t in tools if t.name == "raise_tool_error")
            return await err.ainvoke({})

        result = str(asyncio.run(run()))
        assert "intentional error" in result.lower()
