"""Unit tests for dao_ai.mcp.agent_tool.

These exercise the tool-registration surface without hitting live
Databricks resources: we stub :class:`AppConfig` with a fake
``as_responses_agent`` that returns a captured request, and drive the tool
directly to verify input normalization, header propagation, and final-text
extraction.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest
from mcp.server.fastmcp import FastMCP
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

from dao_ai.mcp.agent_tool import (
    _extract_final_assistant_text,
    _normalize_input,
    _slugify,
    register_agent_as_tool,
)


class _StubAgent:
    """Records the last ResponsesAgentRequest received and returns a canned reply."""

    def __init__(self, reply_text: str = "hello world") -> None:
        self.reply_text: str = reply_text
        self.last_request: ResponsesAgentRequest | None = None

    async def apredict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        self.last_request = request
        return ResponsesAgentResponse(
            output=[
                {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": self.reply_text, "annotations": []}
                    ],
                }
            ]
        )


class _StubAppModel:
    def __init__(self, name: str, description: str | None) -> None:
        self.name = name
        self.description = description


class _StubConfig:
    def __init__(
        self,
        *,
        name: str = "mcp-dao-ai-test",
        description: str | None = "Ask about our retail catalog.",
        agent: _StubAgent | None = None,
    ) -> None:
        self.app = _StubAppModel(name, description)
        self._agent = agent or _StubAgent()

    def as_responses_agent(self) -> _StubAgent:
        return self._agent


def _drop_none(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


def _list_tools(mcp: FastMCP) -> list[Any]:
    return asyncio.run(mcp.list_tools())


def _call_tool(
    mcp: FastMCP, name: str, arguments: dict[str, Any]
) -> Any:
    return asyncio.run(mcp.call_tool(name, arguments))


@pytest.mark.unit
def test_slugify_normalizes_app_names() -> None:
    assert _slugify("mcp-dao-ai-test") == "mcp_dao_ai_test"
    assert _slugify("My Fancy App!") == "my_fancy_app"
    assert _slugify("  ") == "invoke_agent"


@pytest.mark.unit
def test_normalize_input_string_becomes_user_turn() -> None:
    assert _normalize_input("hi") == [{"role": "user", "content": "hi"}]


@pytest.mark.unit
def test_normalize_input_array_passes_through() -> None:
    payload = [{"role": "user", "content": "hi"}]
    assert _normalize_input(payload) is payload


@pytest.mark.unit
def test_extract_final_assistant_text_joins_output_text_parts() -> None:
    response = ResponsesAgentResponse(
        output=[
            {
                "id": "msg_a",
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "one ", "annotations": []},
                    {"type": "output_text", "text": "two", "annotations": []},
                ],
            }
        ]
    )
    assert _extract_final_assistant_text(response) == "one two"


@pytest.mark.unit
def test_extract_final_assistant_text_takes_last_assistant_message() -> None:
    response = ResponsesAgentResponse(
        output=[
            {
                "id": "m1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "first"}],
            },
            {
                "id": "m2",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "second"}],
            },
        ]
    )
    assert _extract_final_assistant_text(response) == "second"


@pytest.mark.unit
def test_register_agent_as_tool_names_from_app_config() -> None:
    mcp = FastMCP("test-server", stateless_http=True)
    config = _StubConfig(name="mcp-dao-ai-test", description="Ask retail.")
    tool_name = register_agent_as_tool(mcp, config)  # type: ignore[arg-type]
    assert tool_name == "mcp_dao_ai_test"

    tools = _list_tools(mcp)
    assert [t.name for t in tools] == ["mcp_dao_ai_test"]
    assert tools[0].description == "Ask retail."


@pytest.mark.unit
def test_register_agent_as_tool_falls_back_to_default_description() -> None:
    mcp = FastMCP("test-server", stateless_http=True)
    config = _StubConfig(name="mcp-dao-ai-test", description=None)
    register_agent_as_tool(mcp, config)  # type: ignore[arg-type]
    tool = _list_tools(mcp)[0]
    assert "Invoke the mcp-dao-ai-test dao-ai agent" in (tool.description or "")


@pytest.mark.unit
def test_register_agent_as_tool_requires_app_name() -> None:
    mcp = FastMCP("test-server", stateless_http=True)

    class _NoApp:
        app = None

        def as_responses_agent(self) -> _StubAgent:  # pragma: no cover
            raise AssertionError("should not be called")

    with pytest.raises(ValueError, match="config.app.name"):
        register_agent_as_tool(mcp, _NoApp())  # type: ignore[arg-type]


@pytest.mark.unit
def test_invoke_agent_string_input_is_wrapped_and_headers_forwarded() -> None:
    mcp = FastMCP("test-server", stateless_http=True)
    agent = _StubAgent(reply_text="reply")
    config = _StubConfig(agent=agent)
    register_agent_as_tool(mcp, config)  # type: ignore[arg-type]

    fake_headers = {"x-forwarded-access-token": "abc123", "x-request-id": "r1"}
    with patch(
        "dao_ai.mcp.agent_tool.current_request_headers", return_value=fake_headers
    ), patch("dao_ai.mcp.agent_tool.current_request_id", return_value="r1"):
        result = _call_tool(mcp, "mcp_dao_ai_test", {"input": "hello"})

    assert agent.last_request is not None
    dumped_input = [
        _drop_none(msg.model_dump() if hasattr(msg, "model_dump") else msg)
        for msg in agent.last_request.input
    ]
    assert dumped_input == [{"role": "user", "content": "hello", "type": "message"}]
    assert agent.last_request.custom_inputs == {
        "configurable": {"headers": fake_headers}
    }
    # New surface: tool returns a CallToolResult with content + structured
    # content + _meta. Assert all three.
    assert result.isError is False
    assert result.content[0].text == "reply"
    assert result.structuredContent["final_message"] == "reply"
    assert result.meta is not None
    assert "databricks.latency_ms" in result.meta
    assert result.meta.get("databricks.request_id") == "r1"
    # x-forwarded-access-token was in the fake headers → obo_present=True.
    assert result.meta.get("databricks.obo_present") is True


@pytest.mark.unit
def test_invoke_agent_reports_obo_absent_when_no_forwarded_token() -> None:
    """Without x-forwarded-access-token, _meta.databricks.obo_present is False."""
    mcp = FastMCP("test-server", stateless_http=True)
    config = _StubConfig(agent=_StubAgent())
    register_agent_as_tool(mcp, config)  # type: ignore[arg-type]

    with patch(
        "dao_ai.mcp.agent_tool.current_request_headers",
        return_value={"x-request-id": "r2"},
    ), patch("dao_ai.mcp.agent_tool.current_request_id", return_value="r2"):
        result = _call_tool(mcp, "mcp_dao_ai_test", {"input": "hi"})

    assert result.meta is not None
    assert result.meta.get("databricks.obo_present") is False


@pytest.mark.unit
def test_invoke_agent_array_input_passes_through() -> None:
    mcp = FastMCP("test-server", stateless_http=True)
    agent = _StubAgent()
    config = _StubConfig(agent=agent)
    register_agent_as_tool(mcp, config)  # type: ignore[arg-type]

    payload = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "prior"},
        {"role": "user", "content": "follow-up"},
    ]
    with patch("dao_ai.mcp.agent_tool.current_request_headers", return_value={}):
        _call_tool(mcp, "mcp_dao_ai_test", {"input": payload})

    assert agent.last_request is not None
    dumped_input = [
        _drop_none(msg.model_dump() if hasattr(msg, "model_dump") else msg)
        for msg in agent.last_request.input
    ]
    assert dumped_input == [
        {"role": "user", "content": "first", "type": "message"},
        {"role": "assistant", "content": "prior", "type": "message"},
        {"role": "user", "content": "follow-up", "type": "message"},
    ]
    # Empty headers → no configurable block.
    assert agent.last_request.custom_inputs is None


class _StubAgentWithTrace(_StubAgent):
    """Stub that populates ``custom_outputs.trace_id`` like the real agent."""

    async def apredict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        self.last_request = request
        return ResponsesAgentResponse(
            output=[
                {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": self.reply_text, "annotations": []}
                    ],
                }
            ],
            custom_outputs={"trace_id": "trace:/cat.schema.prefix/deadbeef"},
        )


@pytest.mark.unit
def test_invoke_agent_returns_structured_content_with_trace_id() -> None:
    """Phase 2 Change 2: response must include structuredContent + _meta.trace_id."""
    mcp = FastMCP("test-server", stateless_http=True)
    agent = _StubAgentWithTrace(reply_text="answer")
    config = _StubConfig(agent=agent)
    register_agent_as_tool(mcp, config)  # type: ignore[arg-type]

    with patch("dao_ai.mcp.agent_tool.current_request_headers", return_value={}), patch(
        "dao_ai.mcp.agent_tool.current_request_id", return_value="req-1"
    ):
        result = _call_tool(mcp, "mcp_dao_ai_test", {"input": "hi"})

    assert result.isError is False
    # Structured content matches AgentInvocationResult schema.
    assert result.structuredContent == {
        "final_message": "answer",
        "trace_id": "trace:/cat.schema.prefix/deadbeef",
        "confidence": None,
    }
    # Plain-text fallback for legacy clients.
    assert result.content[0].text == "answer"
    # _meta carries observability fields.
    assert result.meta is not None
    assert result.meta["databricks.trace_id"] == "trace:/cat.schema.prefix/deadbeef"
    assert "databricks.latency_ms" in result.meta
    assert result.meta["databricks.request_id"] == "req-1"


class _FailingAgent(_StubAgent):
    async def apredict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        raise RuntimeError("simulated agent failure")


@pytest.mark.unit
def test_invoke_agent_surfaces_error_via_is_error_flag() -> None:
    """Agent-side failure must surface as isError:true, not raise."""
    mcp = FastMCP("test-server", stateless_http=True)
    config = _StubConfig(agent=_FailingAgent())
    register_agent_as_tool(mcp, config)  # type: ignore[arg-type]

    with patch("dao_ai.mcp.agent_tool.current_request_headers", return_value={}), patch(
        "dao_ai.mcp.agent_tool.current_request_id", return_value="req-err"
    ):
        result = _call_tool(mcp, "mcp_dao_ai_test", {"input": "boom"})

    assert result.isError is True
    assert "simulated agent failure" in result.content[0].text
    assert result.structuredContent is not None
    assert "simulated agent failure" in result.structuredContent["final_message"]
    assert result.meta is not None
    assert result.meta["databricks.request_id"] == "req-err"
