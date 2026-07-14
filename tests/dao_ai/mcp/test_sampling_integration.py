"""Tier 2 integration tests for PR 3 sampling + roots.

Spins up a FastMCP server on a background thread whose tools:
  - request sampling via ``ctx.session.create_message(...)`` mid-execution
  - request roots via ``ctx.session.list_roots()``

Points dao-ai's raw-ClientSession path (``SamplingRootsMCPClient``) at the
server with a stubbed ``InferenceEndpointModel`` chat model so no real
LLM is hit. Asserts:

  - `sampling_callback` receives the request and returns a valid completion.
  - `max_iterations` cap surfaces as an ErrorData response after N calls.
  - `list_roots_callback` returns the URIs declared in config.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from contextlib import closing
from typing import Any, Iterator
from unittest.mock import patch

import pytest
import uvicorn
from langchain_core.messages import AIMessage
from mcp.server.fastmcp import Context, FastMCP
from mcp.types import SamplingMessage, TextContent

from dao_ai.config import (
    InferenceEndpointModel,
    McpCapabilitiesModel,
    McpFunctionModel,
    McpRootModel,
    McpSamplingCapabilityModel,
)
from dao_ai.tools.mcp import acreate_mcp_tools
from dao_ai.tools.mcp_sampling import (
    DaoAiListRootsCallback,
    DaoAiSamplingCallback,
    _mcp_messages_to_langchain,
    sampling_or_roots_active,
)


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _build_sampling_server(name: str) -> FastMCP:
    """FastMCP server whose tool requests sampling from the client."""
    mcp = FastMCP(name)

    @mcp.tool()
    async def summarize(user_text: str, ctx: Context) -> str:
        """Ask the client's LLM to summarize the user's text (server-driven)."""
        result = await ctx.session.create_message(
            messages=[
                SamplingMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=f"Please summarize: {user_text}",
                    ),
                )
            ],
            max_tokens=200,
            system_prompt="You are a terse summarizer.",
        )
        text = result.content.text if hasattr(result.content, "text") else str(result.content)
        return f"summary: {text}"

    @mcp.tool()
    async def sample_n_times(n: int, ctx: Context) -> str:
        """Fire ``n`` sampling requests back-to-back — used to verify the
        ``max_iterations`` cap."""
        outputs: list[str] = []
        for i in range(n):
            r = await ctx.session.create_message(
                messages=[
                    SamplingMessage(
                        role="user",
                        content=TextContent(type="text", text=f"iteration {i}"),
                    )
                ],
                max_tokens=50,
            )
            content = r.content
            outputs.append(getattr(content, "text", str(content)))
        return " | ".join(outputs)

    @mcp.tool()
    async def get_client_roots(ctx: Context) -> str:
        """Ask the client to enumerate its roots — returns the count."""
        result = await ctx.session.list_roots()
        uris = [str(r.uri) for r in result.roots]
        return f"roots({len(uris)}): {uris}"

    return mcp


class _ServerThread:
    def __init__(self, port: int, app: Any) -> None:
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
def sampling_url() -> Iterator[str]:
    port = _free_port()
    mcp = _build_sampling_server("sampling_probe")
    server = _ServerThread(port, mcp.streamable_http_app())
    server.start()
    try:
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        server.stop()


class _FakeChatModel:
    """Stub chat model that echoes the last user message. Stands in for
    ``InferenceEndpointModel.as_chat_model()`` output so tests don't hit
    a real LLM.
    """

    def __init__(self, reply: str = "stubbed-summary") -> None:
        self.reply = reply
        self.received_messages: list[Any] = []

    async def ainvoke(self, messages: list[Any], config: dict | None = None) -> AIMessage:
        self.received_messages = messages
        return AIMessage(content=self.reply)


def _fn_with_sampling(url: str, max_iter: int = 3) -> McpFunctionModel:
    return McpFunctionModel(
        url=url,
        capabilities=McpCapabilitiesModel(
            sampling=McpSamplingCapabilityModel(
                endpoint=InferenceEndpointModel(
                    name="fake-endpoint",
                    ai_gateway=False,
                ),
                max_iterations=max_iter,
            ),
        ),
    )


def _fn_with_roots(url: str, uris: list[str]) -> McpFunctionModel:
    return McpFunctionModel(
        url=url,
        capabilities=McpCapabilitiesModel(
            roots=[McpRootModel(uri=u) for u in uris],
        ),
    )


class TestRouteSelection:
    def test_sampling_or_roots_active_negatives(self) -> None:
        assert not sampling_or_roots_active(McpFunctionModel(url="http://x/mcp"))
        assert not sampling_or_roots_active(
            McpFunctionModel(
                url="http://x/mcp",
                capabilities=McpCapabilitiesModel(progress=True),
            )
        )

    def test_sampling_or_roots_active_positives(self) -> None:
        assert sampling_or_roots_active(_fn_with_sampling("http://x/mcp"))
        assert sampling_or_roots_active(_fn_with_roots("http://x/mcp", ["file:///a"]))


class TestSamplingCallbackUnit:
    """Callback unit — exercised without a live MCP server."""

    def _params(self, systemPrompt: str | None = None) -> Any:
        from mcp.types import CreateMessageRequestParams

        return CreateMessageRequestParams(
            messages=[
                SamplingMessage(
                    role="user",
                    content=TextContent(type="text", text="hello"),
                )
            ],
            maxTokens=100,
            systemPrompt=systemPrompt,
        )

    def test_message_translation_includes_system_prompt(self) -> None:
        params = self._params(systemPrompt="be brief")
        lc = _mcp_messages_to_langchain(params)
        assert len(lc) == 2  # SystemMessage + HumanMessage
        assert lc[0].content == "be brief"
        assert lc[1].content == "hello"

    def test_message_translation_without_system_prompt(self) -> None:
        params = self._params()
        lc = _mcp_messages_to_langchain(params)
        assert len(lc) == 1
        assert lc[1 - 1].content == "hello"

    def test_returns_completion_from_stubbed_chat_model(self) -> None:
        fn = _fn_with_sampling("http://x/mcp")
        cb = DaoAiSamplingCallback(fn)

        fake = _FakeChatModel(reply="stubbed reply")

        async def _run() -> Any:
            with patch.object(
                type(fn.capabilities.sampling.endpoint),
                "as_chat_model",
                return_value=fake,
            ):
                result = await cb(context=None, params=self._params())
            return result

        result = asyncio.run(_run())
        assert result.role == "assistant"
        assert result.content.text == "stubbed reply"
        assert result.model == "fake-endpoint"
        assert result.stopReason == "endTurn"

    def test_max_iterations_returns_error_data_after_cap(self) -> None:
        fn = _fn_with_sampling("http://x/mcp", max_iter=2)
        cb = DaoAiSamplingCallback(fn)
        fake = _FakeChatModel()

        async def _run() -> list[Any]:
            results = []
            with patch.object(
                type(fn.capabilities.sampling.endpoint),
                "as_chat_model",
                return_value=fake,
            ):
                # Fire max_iter+1 requests; last must be ErrorData.
                for _ in range(3):
                    results.append(await cb(context=None, params=self._params()))
            return results

        results = asyncio.run(_run())
        # First 2 succeed
        from mcp.types import CreateMessageResult, ErrorData

        assert isinstance(results[0], CreateMessageResult)
        assert isinstance(results[1], CreateMessageResult)
        # Third exceeds cap
        assert isinstance(results[2], ErrorData)
        assert "iteration cap exceeded" in results[2].message.lower()


class TestListRootsCallbackUnit:
    def test_returns_configured_roots(self) -> None:
        fn = _fn_with_roots("http://x/mcp", ["file:///workspace", "file:///data"])
        cb = DaoAiListRootsCallback(fn)
        result = asyncio.run(cb(context=None))
        # ListRootsResult
        uris = {str(r.uri) for r in result.roots}
        assert "file:///workspace" in uris
        assert "file:///data" in uris

    def test_invalid_uri_is_skipped_not_raised(self) -> None:
        fn = _fn_with_roots("http://x/mcp", ["not a valid uri", "file:///ok"])
        cb = DaoAiListRootsCallback(fn)
        result = asyncio.run(cb(context=None))
        # Only the valid one survives; no exception.
        uris = {str(r.uri) for r in result.roots}
        assert any("file:///ok" in u for u in uris)


class TestLiveSampling:
    """Full round-trip: real FastMCP fixture requests sampling; our callback
    fields it via a stubbed chat model."""

    def test_server_summarize_receives_sampled_reply(self, sampling_url: str) -> None:
        fn = _fn_with_sampling(sampling_url)

        fake = _FakeChatModel(reply="TL;DR")

        async def _run() -> str:
            with patch.object(
                type(fn.capabilities.sampling.endpoint),
                "as_chat_model",
                return_value=fake,
            ):
                tools = await acreate_mcp_tools(fn)
                summarize_tool = next(t for t in tools if t.name == "summarize")
                text = await summarize_tool.ainvoke({"user_text": "A long paragraph"})
                return str(text)

        text = asyncio.run(_run())
        assert "TL;DR" in text
        # Confirm the stub received the server's user prompt
        received = [
            m.content
            for m in fake.received_messages
            if getattr(m, "type", "").lower() in ("system", "human")
        ]
        assert any("summarize" in str(c).lower() for c in received)

    def test_max_iterations_cap_enforced_live(self, sampling_url: str) -> None:
        # Server fires N=4 sampling calls; cap is 2.
        fn = _fn_with_sampling(sampling_url, max_iter=2)
        fake = _FakeChatModel(reply="ok")

        async def _run() -> str:
            with patch.object(
                type(fn.capabilities.sampling.endpoint),
                "as_chat_model",
                return_value=fake,
            ):
                tools = await acreate_mcp_tools(fn)
                sn = next(t for t in tools if t.name == "sample_n_times")
                # Server-side .create_message raises when the client returns
                # ErrorData — the tool call will fail. Assert we get an
                # observable failure text.
                try:
                    text = await sn.ainvoke({"n": 4})
                except Exception as exc:
                    return f"raised: {exc}"
                return str(text)

        result = asyncio.run(_run())
        # We either raise or return with an error text depending on how the
        # server handles our ErrorData response. Either way we should see
        # evidence the cap fired.
        assert "cap" in result.lower() or "raised" in result.lower() or "error" in result.lower()


class TestLiveRoots:
    def test_server_reads_client_roots(self, sampling_url: str) -> None:
        fn = _fn_with_roots(sampling_url, ["file:///workspace/one", "file:///workspace/two"])

        async def _run() -> str:
            tools = await acreate_mcp_tools(fn)
            gr = next(t for t in tools if t.name == "get_client_roots")
            return str(await gr.ainvoke({}))

        result = asyncio.run(_run())
        assert "roots(2)" in result
        assert "file:///workspace/one" in result
        assert "file:///workspace/two" in result
