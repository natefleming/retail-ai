"""Pin the contract: tool dispatch under streaming.

We characterized a hard incompatibility between
``CompiledStateGraph.astream_events(version="v3")`` (the experimental
v3 typed-projection API in LangGraph 1.2.x) and tool dispatch from
``langchain.agents.create_agent`` when the underlying chat model is
``ChatDatabricks``/``ChatUnityAIGateway`` against any
Databricks-served LLM (verified with ``databricks-gpt-oss-120b`` and
``databricks-claude-sonnet-4-5``): the LLM emits a tool call, but the
graph never invokes the tool. Same agent, same prompt, same model:

* ``ainvoke``                                  → tool fires ✅
* ``astream(stream_mode=...)``                 → tool fires ✅
* ``astream_events(version="v3")``             → tool **never fires** ❌

These tests use offline mock chat models so they can run in CI without
hitting Databricks. They lock the regression in place and verify the
fix path so we don't accidentally re-introduce the v3 swap.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Iterator

import pytest
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from langchain.agents import create_agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _SkuArgs(BaseModel):
    sku: str = Field(description="Product SKU")


def _check_stock_tool(calls: list[dict[str, Any]]):
    """Return a tool that records every call into ``calls``."""

    @tool(
        "check_stock_uc",
        args_schema=_SkuArgs,
        description="Check inventory by SKU across all locations.",
    )
    def _check_stock_uc(sku: str) -> str:
        """Check inventory by SKU."""
        calls.append({"sku": sku})
        return f"sku={sku} -> 7 in stock"

    return _check_stock_uc


class _FakeToolCallingChatModel(BaseChatModel):
    """Fake chat model. First invocation emits a tool call; second emits
    a plain text reply. Supports both blocking and streaming paths so we
    can drive create_agent across every stream mode.
    """

    call_index: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling"

    def _canned(self) -> AIMessage:
        if self.call_index == 0:
            self.__dict__["call_index"] = 1
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "check_stock_uc",
                        "args": {"sku": "FRZ-CAKE-001"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(content="7 in stock at LA, 12 at Dallas.")

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._canned())])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._canned())])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        canned: AIMessage = self._canned()
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content=canned.content,
                tool_calls=canned.tool_calls,
                tool_call_chunks=[
                    {
                        "name": tc["name"],
                        "args": "{\"sku\": \"FRZ-CAKE-001\"}",
                        "id": tc["id"],
                        "index": 0,
                        "type": "tool_call_chunk",
                    }
                    for tc in canned.tool_calls
                ],
            )
        )

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        for chunk in self._stream(messages, stop, None, **kwargs):
            yield chunk

    def bind_tools(self, tools, **_kwargs):  # type: ignore[override]
        return self


@pytest.fixture
def fake_agent_with_tool() -> tuple[Any, list[dict[str, Any]]]:
    """Return ``(agent, calls_list)``. ``calls_list`` is mutated by the
    tool when (and only when) the graph actually dispatches it."""
    calls: list[dict[str, Any]] = []
    llm = _FakeToolCallingChatModel()
    agent = create_agent(
        model=llm,
        tools=[_check_stock_tool(calls)],
        system_prompt="Call check_stock_uc when the user gives a SKU.",
    )
    return agent, calls


# ---------------------------------------------------------------------------
# Live offline tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_ainvoke_dispatches_tool(fake_agent_with_tool: tuple[Any, list[dict[str, Any]]]) -> None:
    agent, calls = fake_agent_with_tool

    async def _run() -> None:
        await agent.ainvoke(
            {"messages": [HumanMessage(content="Do you have FRZ-CAKE-001?")]}
        )

    asyncio.run(_run())
    assert calls == [{"sku": "FRZ-CAKE-001"}], (
        "Non-stream invoke must dispatch the tool. If this fails the LLM "
        "fixture or create_agent is wrong, not the streaming code."
    )


@pytest.mark.unit
def test_astream_messages_dispatches_tool(
    fake_agent_with_tool: tuple[Any, list[dict[str, Any]]],
) -> None:
    """The fix path: ``stream_mode=['messages','updates']`` dispatches
    tools correctly. This is what ``models.py`` MUST use."""
    agent, calls = fake_agent_with_tool

    async def _run() -> None:
        async for _kind, _payload in agent.astream(
            {"messages": [HumanMessage(content="Do you have FRZ-CAKE-001?")]},
            stream_mode=["messages", "updates"],
        ):
            pass

    asyncio.run(_run())
    assert calls == [{"sku": "FRZ-CAKE-001"}], (
        "The fix path (stream_mode=['messages','updates']) must dispatch "
        "the tool — this is the contract models.py relies on."
    )


@pytest.mark.unit
def test_astream_updates_dispatches_tool(
    fake_agent_with_tool: tuple[Any, list[dict[str, Any]]],
) -> None:
    """Equivalent contract for ``stream_mode='updates'`` alone."""
    agent, calls = fake_agent_with_tool

    async def _run() -> None:
        async for _ in agent.astream(
            {"messages": [HumanMessage(content="Do you have FRZ-CAKE-001?")]},
            stream_mode="updates",
        ):
            pass

    asyncio.run(_run())
    assert calls == [{"sku": "FRZ-CAKE-001"}]


@pytest.mark.unit
def test_astream_messages_yields_tuple_with_metadata(
    fake_agent_with_tool: tuple[Any, list[dict[str, Any]]],
) -> None:
    """``stream_mode='messages'`` must emit ``(AIMessageChunk, metadata)``
    tuples so the streaming consumer can attribute deltas to a node."""
    agent, _ = fake_agent_with_tool

    seen_nodes: set[str] = set()

    async def _run() -> None:
        async for chunk_kind, payload in agent.astream(
            {"messages": [HumanMessage(content="Do you have FRZ-CAKE-001?")]},
            stream_mode=["messages", "updates"],
        ):
            if chunk_kind == "messages":
                _msg, metadata = payload
                node: str | None = metadata.get("langgraph_node")
                if node:
                    seen_nodes.add(node)

    asyncio.run(_run())
    assert "model" in seen_nodes, (
        f"Expected ``model`` node in metadata, got {seen_nodes!r}. The "
        "surface_to_user filter relies on metadata.langgraph_node + "
        "chunk.name to map deltas to agents."
    )
