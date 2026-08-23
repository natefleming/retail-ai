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
from mcp.types import RequestParams
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

from dao_ai.mcp.agent_tool import (
    _extract_conversation_id,
    _extract_final_assistant_text,
    _normalize_input,
    _resolve_conversation_id,
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
                        {
                            "type": "output_text",
                            "text": self.reply_text,
                            "annotations": [],
                        }
                    ],
                }
            ]
        )


class _StubAppModel:
    def __init__(self, name: str, description: str | None) -> None:
        self.name = name
        self.description = description
        # register_agent_as_tool reads mcp_tool_description first, falling back to
        # description; the real AppModel always defines it (default None).
        self.mcp_tool_description = None


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


def _call_tool(mcp: FastMCP, name: str, arguments: dict[str, Any]) -> Any:
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


def _meta(**extras: Any) -> RequestParams.Meta:
    """Build a real ``RequestParams.Meta`` with the given extra fields.

    ``Meta.model_config = ConfigDict(extra="allow")`` accepts arbitrary
    keys, so passing ``conversation_id="..."`` mirrors what a real client
    sends over the wire on a ``tools/call`` request's ``_meta`` block.
    """
    return RequestParams.Meta.model_validate(extras)


@pytest.mark.unit
def test_resolve_conversation_id_precedence_meta_over_header() -> None:
    """_meta.conversation_id > X-Databricks-Conversation-Id header > None.

    The tool-argument channel is intentionally NOT part of the resolver —
    exposing a session key on the tool's inputSchema would make it
    model-controlled and prompt-injection-attackable. Identity belongs on
    the transport, so only _meta (MCP-native) and the HTTP header are
    accepted.
    """
    # 1. _meta.conversation_id wins over header when both present.
    got, src = _resolve_conversation_id(
        meta=_meta(conversation_id="c"),
        headers={"x-databricks-conversation-id": "d"},
    )
    assert (got, src) == ("c", "meta")

    # 2. Header is the fallback channel.
    got, src = _resolve_conversation_id(
        meta=None,
        headers={"x-databricks-conversation-id": "d"},
    )
    assert (got, src) == ("d", "header")

    # 3. Nothing supplied → agent will generate downstream.
    got, src = _resolve_conversation_id(
        meta=None,
        headers={},
    )
    assert (got, src) == (None, None)

    # 4. meta carrying only progressToken (no conversation_id) skips
    # cleanly to the header channel — protects against spurious meta wins
    # when the client only meant to send a progress token.
    got, src = _resolve_conversation_id(
        meta=_meta(progressToken="tok-1"),
        headers={"x-databricks-conversation-id": "hdr"},
    )
    assert (got, src) == ("hdr", "header")


@pytest.mark.unit
def test_extract_conversation_id_prefers_session_over_configurable() -> None:
    """Response echo path reads session.conversation_id first, thread_id as fallback."""
    both = ResponsesAgentResponse(
        output=[],
        custom_outputs={
            "session": {"conversation_id": "from-session"},
            "configurable": {"thread_id": "from-configurable"},
        },
    )
    assert _extract_conversation_id(both) == "from-session"

    only_thread = ResponsesAgentResponse(
        output=[],
        custom_outputs={"configurable": {"thread_id": "just-thread"}},
    )
    assert _extract_conversation_id(only_thread) == "just-thread"

    neither = ResponsesAgentResponse(output=[], custom_outputs={"trace_id": "t"})
    assert _extract_conversation_id(neither) is None

    no_outputs = ResponsesAgentResponse(output=[])
    assert _extract_conversation_id(no_outputs) is None


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
    with (
        patch(
            "dao_ai.mcp.agent_tool.current_request_headers", return_value=fake_headers
        ),
        patch("dao_ai.mcp.agent_tool.current_request_id", return_value="r1"),
    ):
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

    with (
        patch(
            "dao_ai.mcp.agent_tool.current_request_headers",
            return_value={"x-request-id": "r2"},
        ),
        patch("dao_ai.mcp.agent_tool.current_request_id", return_value="r2"),
    ):
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
                        {
                            "type": "output_text",
                            "text": self.reply_text,
                            "annotations": [],
                        }
                    ],
                }
            ],
            custom_outputs={"trace_id": "trace:/cat.schema.prefix/deadbeef"},
        )


class _StubAgentWithSession(_StubAgent):
    """Stub that populates ``custom_outputs.session.conversation_id``.

    Mirrors how :meth:`LanggraphResponsesAgent._build_custom_outputs_async`
    surfaces the resolved conversation key on the real agent response.
    Echoes back whatever the caller supplied on
    ``custom_inputs.configurable.conversation_id`` (or generates a stable
    stub value when nothing was supplied) so tests can verify the MCP
    tool's response-side echo without wiring a live Lakebase checkpoint.
    """

    def __init__(
        self, reply_text: str = "hello world", generated_id: str = "srv-generated"
    ) -> None:
        super().__init__(reply_text)
        self.generated_id: str = generated_id

    async def apredict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        self.last_request = request
        supplied_id: str | None = None
        custom_inputs = request.custom_inputs or {}
        if isinstance(custom_inputs, dict):
            configurable = custom_inputs.get("configurable") or {}
            if isinstance(configurable, dict):
                candidate = configurable.get("conversation_id")
                if isinstance(candidate, str):
                    supplied_id = candidate
        resolved_id: str = supplied_id or self.generated_id
        return ResponsesAgentResponse(
            output=[
                {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": self.reply_text,
                            "annotations": [],
                        }
                    ],
                }
            ],
            custom_outputs={
                "trace_id": "trace:/cat.schema.prefix/deadbeef",
                "configurable": {"thread_id": resolved_id},
                "session": {"conversation_id": resolved_id},
            },
        )


@pytest.mark.unit
def test_invoke_agent_returns_structured_content_with_trace_id() -> None:
    """Phase 2 Change 2: response must include structuredContent + _meta.trace_id."""
    mcp = FastMCP("test-server", stateless_http=True)
    agent = _StubAgentWithTrace(reply_text="answer")
    config = _StubConfig(agent=agent)
    register_agent_as_tool(mcp, config)  # type: ignore[arg-type]

    with (
        patch("dao_ai.mcp.agent_tool.current_request_headers", return_value={}),
        patch("dao_ai.mcp.agent_tool.current_request_id", return_value="req-1"),
    ):
        result = _call_tool(mcp, "mcp_dao_ai_test", {"input": "hi"})

    assert result.isError is False
    # Structured content matches AgentInvocationResult schema. conversation_id
    # / thread_id are None because this stub does not populate
    # custom_outputs.session — kept explicit so schema drift is caught.
    assert result.structuredContent == {
        "final_message": "answer",
        "trace_id": "trace:/cat.schema.prefix/deadbeef",
        "conversation_id": None,
        "thread_id": None,
        "confidence": None,
    }
    # Plain-text fallback for legacy clients.
    assert result.content[0].text == "answer"
    # _meta carries observability fields.
    assert result.meta is not None
    assert result.meta["databricks.trace_id"] == "trace:/cat.schema.prefix/deadbeef"
    assert "databricks.latency_ms" in result.meta
    assert result.meta["databricks.request_id"] == "req-1"
    # No conversation_id supplied AND stub didn't surface one → key absent.
    assert "conversation_id" not in result.meta


@pytest.mark.unit
def test_invoke_agent_header_flows_into_custom_inputs_and_echoes() -> None:
    """Header → custom_inputs.configurable.conversation_id → structuredContent + _meta.

    `X-Databricks-Conversation-Id` on the HTTP transport is the primary
    transport-level channel for supplying a conversation id (the tool-arg
    channel has been removed for prompt-injection reasons).
    """
    mcp = FastMCP("test-server", stateless_http=True)
    agent = _StubAgentWithSession(reply_text="hi")
    config = _StubConfig(agent=agent)
    register_agent_as_tool(mcp, config)  # type: ignore[arg-type]

    with (
        patch(
            "dao_ai.mcp.agent_tool.current_request_headers",
            return_value={"x-databricks-conversation-id": "hdr-999"},
        ),
        patch("dao_ai.mcp.agent_tool.current_request_id", return_value="req-c3"),
    ):
        result = _call_tool(mcp, "mcp_dao_ai_test", {"input": "hi"})

    assert agent.last_request is not None
    configurable = agent.last_request.custom_inputs["configurable"]
    assert configurable["conversation_id"] == "hdr-999"
    # Headers still forwarded for OBO propagation.
    assert configurable["headers"] == {"x-databricks-conversation-id": "hdr-999"}
    assert result.isError is False
    assert result.structuredContent["conversation_id"] == "hdr-999"
    assert result.structuredContent["thread_id"] == "hdr-999"
    assert result.meta is not None
    assert result.meta["conversation_id"] == "hdr-999"


@pytest.mark.unit
def test_invoke_agent_input_schema_does_not_expose_conversation_id() -> None:
    """Regression guard: the MCP tool's advertised inputSchema must not carry
    ``conversation_id`` or ``thread_id`` as parameters.

    Placing session identity on the tool's inputSchema would make it
    model-controlled per MCP semantics (the LLM would populate it on each
    call), creating a prompt-injection surface. This test locks the
    contract so a future re-introduction fails CI.
    """
    mcp = FastMCP("test-server", stateless_http=True)
    register_agent_as_tool(mcp, _StubConfig())  # type: ignore[arg-type]
    (tool,) = _list_tools(mcp)
    schema = tool.inputSchema
    props: set[str] = set((schema or {}).get("properties", {}).keys())
    assert "conversation_id" not in props, (
        f"conversation_id must not appear on the tool inputSchema; got {props}"
    )
    assert "thread_id" not in props, (
        f"thread_id must not appear on the tool inputSchema; got {props}"
    )
    # Positive assertion: `input` is still the sole caller-facing property.
    assert props == {"input"}, (
        f"expected exactly {{'input'}} on inputSchema.properties; got {props}"
    )


@pytest.mark.unit
def test_invoke_agent_echoes_server_generated_id_when_none_supplied() -> None:
    """Nothing supplied → agent generates → MCP tool echoes for the caller."""
    mcp = FastMCP("test-server", stateless_http=True)
    agent = _StubAgentWithSession(generated_id="srv-uuid-1")
    config = _StubConfig(agent=agent)
    register_agent_as_tool(mcp, config)  # type: ignore[arg-type]

    with (
        patch("dao_ai.mcp.agent_tool.current_request_headers", return_value={}),
        patch("dao_ai.mcp.agent_tool.current_request_id", return_value="req-c5"),
    ):
        result = _call_tool(mcp, "mcp_dao_ai_test", {"input": "hi"})

    # No id on request → the MCP layer does not fabricate one; the agent
    # generates it and it comes back on custom_outputs.
    assert agent.last_request is not None
    assert agent.last_request.custom_inputs is None
    assert result.structuredContent["conversation_id"] == "srv-uuid-1"
    assert result.meta is not None
    assert result.meta["conversation_id"] == "srv-uuid-1"


class _FailingAgent(_StubAgent):
    async def apredict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        raise RuntimeError("simulated agent failure")


@pytest.mark.unit
def test_invoke_agent_surfaces_error_via_is_error_flag() -> None:
    """Agent-side failure must surface as isError:true, not raise."""
    mcp = FastMCP("test-server", stateless_http=True)
    config = _StubConfig(agent=_FailingAgent())
    register_agent_as_tool(mcp, config)  # type: ignore[arg-type]

    with (
        patch("dao_ai.mcp.agent_tool.current_request_headers", return_value={}),
        patch("dao_ai.mcp.agent_tool.current_request_id", return_value="req-err"),
    ):
        result = _call_tool(mcp, "mcp_dao_ai_test", {"input": "boom"})

    assert result.isError is True
    assert "simulated agent failure" in result.content[0].text
    assert result.structuredContent is not None
    assert "simulated agent failure" in result.structuredContent["final_message"]
    assert result.meta is not None
    assert result.meta["databricks.request_id"] == "req-err"
