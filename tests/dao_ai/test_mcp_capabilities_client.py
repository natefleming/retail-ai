"""Unit tests for the MCP capabilities client-side wiring (PR 1).

Covers three concerns:

1. Pydantic capability models load and validate correctly.
2. ``_build_mcp_client`` constructs a plain ``MultiServerMCPClient`` when
   ``capabilities`` is None (regression guard) and attaches callbacks +
   interceptors when it's set.
3. Callback + interceptor classes conform to their Protocols and fire the
   expected MLflow span events (via patched ``mlflow.get_current_active_span``).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_mcp_adapters.callbacks import (
    CallbackContext,
    ElicitationCallback,
    ProgressCallback,
)
from langchain_mcp_adapters.interceptors import (
    MCPToolCallRequest,
    ToolCallInterceptor,
)
from mcp.types import (
    CallToolResult,
    ElicitRequestFormParams,
    ElicitResult,
    LoggingMessageNotification,
    LoggingMessageNotificationParams,
    ResourceLink,
    ServerNotification,
    TextContent,
)

from dao_ai.config import (
    McpCapabilitiesModel,
    McpFunctionModel,
    McpRootModel,
)
from dao_ai.tools.mcp_callbacks import (
    DaoAiElicitationCallback,
    DaoAiNotificationCallback,
    DaoAiProgressCallback,
    _resume_value_to_elicit_result,
)
from dao_ai.tools.mcp_interceptors import (
    DaoAiStructuredOutputInterceptor,
    DaoAiTraceInterceptor,
)


class TestCapabilityModels:
    def test_defaults(self) -> None:
        caps = McpCapabilitiesModel()
        assert caps.progress is False
        assert caps.logging is False
        assert caps.elicitation is None
        assert caps.structured_output is True
        assert caps.sampling is None
        assert caps.roots == []

    def test_full_shape(self) -> None:
        caps = McpCapabilitiesModel(
            progress=True,
            logging=True,
            elicitation="hitl",
            structured_output=True,
            roots=[McpRootModel(uri="file:///workspace", name="workspace")],
        )
        assert caps.progress is True
        assert caps.logging is True
        assert caps.elicitation == "hitl"
        assert caps.roots[0].uri == "file:///workspace"

    def test_invalid_elicitation_mode_rejected(self) -> None:
        with pytest.raises(ValueError):
            McpCapabilitiesModel(elicitation="prompt")  # type: ignore[arg-type]

    def test_mcpfunctionmodel_accepts_capabilities(self) -> None:
        model = McpFunctionModel(
            url="http://example.com/mcp",
            capabilities=McpCapabilitiesModel(progress=True, logging=True),
        )
        assert model.capabilities is not None
        assert model.capabilities.progress is True
        assert model.capabilities.logging is True


class TestBuildMcpClient:
    """``_build_mcp_client`` is the single construction site for the client."""

    def _make_fn(self, capabilities=None) -> McpFunctionModel:
        return McpFunctionModel(url="http://example.com/mcp", capabilities=capabilities)

    def test_classic_path_when_capabilities_none(self) -> None:
        from dao_ai.tools import mcp as mcp_mod

        with patch.object(mcp_mod, "MultiServerMCPClient") as mock_cls, patch.object(
            mcp_mod, "_build_connection_config", return_value={"url": "x"}
        ):
            mcp_mod._build_mcp_client(self._make_fn(capabilities=None))
            _, kwargs = mock_cls.call_args
            assert "callbacks" not in kwargs or kwargs.get("callbacks") is None
            assert "tool_interceptors" not in kwargs or kwargs.get("tool_interceptors") is None
            assert kwargs["handle_tool_errors"] is False

    def test_capabilities_path_attaches_callbacks_and_interceptors(self) -> None:
        from dao_ai.tools import mcp as mcp_mod

        caps = McpCapabilitiesModel(
            progress=True,
            logging=True,
            elicitation="reject",
            structured_output=True,
        )

        with patch.object(mcp_mod, "MultiServerMCPClient") as mock_cls, patch.object(
            mcp_mod, "_build_connection_config", return_value={"url": "x"}
        ):
            mcp_mod._build_mcp_client(self._make_fn(capabilities=caps))
            args, kwargs = mock_cls.call_args
            assert kwargs["callbacks"] is not None
            assert kwargs["handle_tool_errors"] is False
            interceptors = kwargs["tool_interceptors"]
            assert len(interceptors) == 2
            assert isinstance(interceptors[0], DaoAiTraceInterceptor)
            assert isinstance(interceptors[1], DaoAiStructuredOutputInterceptor)
            # logging=True → message_handler injected via session_kwargs
            from dao_ai.tools.mcp_callbacks import DaoAiNotificationCallback

            connections = args[0]
            session_kwargs = connections["mcp_function"].get("session_kwargs") or {}
            assert isinstance(
                session_kwargs.get("message_handler"), DaoAiNotificationCallback
            )

    def test_capabilities_path_skips_structured_interceptor_when_disabled(self) -> None:
        from dao_ai.tools import mcp as mcp_mod

        caps = McpCapabilitiesModel(progress=True, structured_output=False)
        with patch.object(mcp_mod, "MultiServerMCPClient") as mock_cls, patch.object(
            mcp_mod, "_build_connection_config", return_value={"url": "x"}
        ):
            mcp_mod._build_mcp_client(self._make_fn(capabilities=caps))
            _, kwargs = mock_cls.call_args
            interceptors = kwargs["tool_interceptors"]
            assert len(interceptors) == 1
            assert isinstance(interceptors[0], DaoAiTraceInterceptor)

    def test_capabilities_path_omits_callbacks_when_all_disabled(self) -> None:
        from dao_ai.tools import mcp as mcp_mod

        caps = McpCapabilitiesModel(structured_output=True)
        with patch.object(mcp_mod, "MultiServerMCPClient") as mock_cls, patch.object(
            mcp_mod, "_build_connection_config", return_value={"url": "x"}
        ):
            mcp_mod._build_mcp_client(self._make_fn(capabilities=caps))
            _, kwargs = mock_cls.call_args
            assert kwargs["callbacks"] is None


class TestClientRoutingCallsites:
    """Every ``MultiServerMCPClient`` construction site in tools/mcp.py MUST
    flow through ``_build_mcp_client`` so capabilities propagate uniformly.
    """

    def _make_fn(self, capabilities=None) -> McpFunctionModel:
        return McpFunctionModel(url="http://example.com/mcp", capabilities=capabilities)

    def test_afetch_tools_from_server_uses_build_mcp_client(self) -> None:
        from dao_ai.tools import mcp as mcp_mod

        fake_client = MagicMock()

        class _FakeSession:
            async def list_tools(self):
                return SimpleNamespace(tools=[])

        class _FakeCtx:
            async def __aenter__(self_inner):
                return _FakeSession()

            async def __aexit__(self_inner, exc_type, exc, tb):
                return False

        fake_client.session.return_value = _FakeCtx()

        caps = McpCapabilitiesModel(progress=True)
        with patch.object(
            mcp_mod, "_build_mcp_client", return_value=fake_client
        ) as mock_build:
            asyncio.run(mcp_mod._afetch_tools_from_server(self._make_fn(caps)))

        mock_build.assert_called_once()
        # Pass-through — the same function object we handed in.
        passed_fn, *_ = mock_build.call_args.args
        assert passed_fn.capabilities is caps

    def test_acreate_tool_wrapper_uses_build_mcp_client(self) -> None:
        """The wrapper produced by acreate_mcp_tools() must route through
        _build_mcp_client on invocation, threading context + capabilities."""
        from dao_ai.tools import mcp as mcp_mod
        from mcp.types import Tool

        # Skip discovery — return one fake Tool so a wrapper is built.
        fake_tool = Tool(
            name="probe",
            description="probe",
            inputSchema={"type": "object", "properties": {}},
        )

        # Fake session that returns a benign text result. Accepts arbitrary
        # kwargs so we don't need to track SDK signature changes.
        class _FakeSession:
            async def call_tool(self, name, args, **_kwargs):
                return CallToolResult(
                    content=[TextContent(type="text", text="ok")]
                )

        class _FakeCtx:
            async def __aenter__(self_inner):
                return _FakeSession()

            async def __aexit__(self_inner, exc_type, exc, tb):
                return False

        fake_client = MagicMock()
        fake_client.session.return_value = _FakeCtx()

        caps = McpCapabilitiesModel(progress=True, structured_output=True)
        fn = self._make_fn(capabilities=caps)

        async def _run() -> str:
            with patch.object(
                mcp_mod, "_afetch_tools_from_server", return_value=[fake_tool]
            ), patch.object(
                mcp_mod, "_build_mcp_client", return_value=fake_client
            ) as mock_build, patch.object(
                mcp_mod, "set_resource_attributes"
            ):
                tools = await mcp_mod.acreate_mcp_tools(fn)
                assert len(tools) == 1
                result = await tools[0].ainvoke({})
                # Assert build was called at least once inside the wrapper.
                mock_build.assert_called()
                # Confirm capabilities on the invoked function are ours.
                call_fn, *_ = mock_build.call_args.args
                assert call_fn.capabilities is caps
                return result

        assert asyncio.run(_run()) == "ok"


class TestProtocolConformance:
    def test_progress_callback_is_protocol(self) -> None:
        assert isinstance(DaoAiProgressCallback(), ProgressCallback)

    def test_elicitation_callback_is_protocol(self) -> None:
        assert isinstance(DaoAiElicitationCallback("reject"), ElicitationCallback)

    def test_trace_interceptor_is_protocol(self) -> None:
        assert isinstance(DaoAiTraceInterceptor(), ToolCallInterceptor)

    def test_structured_output_interceptor_is_protocol(self) -> None:
        assert isinstance(DaoAiStructuredOutputInterceptor(), ToolCallInterceptor)


class TestProgressCallback:
    def test_forwards_to_span_event(self) -> None:
        span = MagicMock()
        with patch(
            "dao_ai.tools.mcp_callbacks.mlflow.get_current_active_span",
            return_value=span,
        ):
            cb = DaoAiProgressCallback()
            asyncio.run(
                cb(
                    progress=0.5,
                    total=1.0,
                    message="halfway",
                    context=CallbackContext(server_name="genie", tool_name="query"),
                )
            )
        span.add_event.assert_called_once()
        args, kwargs = span.add_event.call_args
        # MLflow LiveSpan.add_event(event: SpanEvent) — single positional arg.
        span_event = args[0]
        assert span_event.name == "mcp.progress"
        attrs = span_event.attributes
        assert attrs["channel"] == "mcp.progress"
        assert attrs["progress"] == 0.5
        assert attrs["tool_name"] == "query"
        assert attrs["server_name"] == "genie"

    def test_silent_when_no_active_span(self) -> None:
        with patch(
            "dao_ai.tools.mcp_callbacks.mlflow.get_current_active_span",
            return_value=None,
        ):
            cb = DaoAiProgressCallback()
            # Must not raise.
            asyncio.run(
                cb(
                    progress=0.5,
                    total=None,
                    message=None,
                    context=CallbackContext(server_name="x"),
                )
            )


class TestNotificationCallback:
    @staticmethod
    def _log_notification(level: str) -> ServerNotification:
        return ServerNotification(
            LoggingMessageNotification(
                method="notifications/message",
                params=LoggingMessageNotificationParams(
                    level=level, data="boom", logger="genie"
                ),
            )
        )

    def test_forwards_log_notification_generically(self) -> None:
        span = MagicMock()
        with patch(
            "dao_ai.tools.mcp_callbacks.mlflow.get_current_active_span",
            return_value=span,
        ):
            cb = DaoAiNotificationCallback(server_name="genie")
            asyncio.run(cb(self._log_notification("error")))
        span.add_event.assert_called_once()
        span_event = span.add_event.call_args[0][0]
        # Span-event name equals the channel. For notifications/message
        # frames MCP guarantees level+data on params; those are also lifted
        # to top-level of the envelope for consumer convenience.
        assert span_event.name == "mcp.log"
        attrs = span_event.attributes
        assert attrs["channel"] == "mcp.log"
        assert attrs["method"] == "notifications/message"
        assert attrs["server_name"] == "genie"
        assert attrs["level"] == "error"
        assert attrs["logger"] == "genie"
        assert attrs["data"] == "boom"
        # Raw params also carried through for consumers that want the
        # original shape.
        assert attrs["params"]["level"] == "error"

    def test_ignores_exceptions_and_requests(self) -> None:
        span = MagicMock()
        with patch(
            "dao_ai.tools.mcp_callbacks.mlflow.get_current_active_span",
            return_value=span,
        ):
            cb = DaoAiNotificationCallback()
            asyncio.run(cb(RuntimeError("transport blew up")))
            asyncio.run(cb(SimpleNamespace(this_is_a="request-responder")))
        span.add_event.assert_not_called()

    def test_skips_progress_notification(self) -> None:
        from mcp.types import ProgressNotification, ProgressNotificationParams

        span = MagicMock()
        with patch(
            "dao_ai.tools.mcp_callbacks.mlflow.get_current_active_span",
            return_value=span,
        ):
            cb = DaoAiNotificationCallback()
            note = ServerNotification(
                ProgressNotification(
                    method="notifications/progress",
                    params=ProgressNotificationParams(
                        progressToken="t1", progress=0.5, total=1.0
                    ),
                )
            )
            asyncio.run(cb(note))
        span.add_event.assert_not_called()


class TestElicitationCallback:
    def test_reject_mode_returns_cancel(self) -> None:
        span = MagicMock()
        with patch(
            "dao_ai.tools.mcp_callbacks.mlflow.get_current_active_span",
            return_value=span,
        ):
            cb = DaoAiElicitationCallback("reject")
            result = asyncio.run(
                cb(
                    mcp_context=SimpleNamespace(),  # type: ignore[arg-type]
                    params=ElicitRequestFormParams(
                        message="please provide", requestedSchema={}
                    ),
                    context=CallbackContext(server_name="s"),
                )
            )
        assert isinstance(result, ElicitResult)
        assert result.action == "cancel"

    def test_hitl_mode_delegates_to_langgraph_interrupt(self) -> None:
        # In hitl mode the callback surfaces the elicitation as a LangGraph
        # interrupt whose resume value becomes the ElicitResult. Verify by
        # patching the interrupt shim to return an accept payload directly.
        cb = DaoAiElicitationCallback("hitl")
        with patch(
            "dao_ai.tools.mcp_callbacks.langgraph_interrupt",
            return_value={"action": "accept", "content": {"answer": "42"}},
        ) as mock_interrupt:
            result = asyncio.run(
                cb(
                    mcp_context=SimpleNamespace(),  # type: ignore[arg-type]
                    params=ElicitRequestFormParams(
                        message="need info", requestedSchema={"type": "object"}
                    ),
                    context=CallbackContext(server_name="s", tool_name="t"),
                )
            )
        mock_interrupt.assert_called_once()
        payload = mock_interrupt.call_args.args[0]
        assert payload["type"] == "mcp.elicitation"
        assert payload["server_name"] == "s"
        assert payload["tool_name"] == "t"
        assert payload["message"] == "need info"
        assert result.action == "accept"
        assert result.content == {"answer": "42"}


class TestResumeValueMapping:
    def test_none_maps_to_cancel(self) -> None:
        r = _resume_value_to_elicit_result(None)
        assert r.action == "cancel"

    def test_action_dict_pass_through(self) -> None:
        r = _resume_value_to_elicit_result({"action": "decline"})
        assert r.action == "decline"

    def test_bare_dict_maps_to_accept(self) -> None:
        r = _resume_value_to_elicit_result({"name": "Nate"})
        assert r.action == "accept"
        assert r.content == {"name": "Nate"}


class TestStructuredOutputInterceptor:
    def test_resource_link_expanded_to_span_event(self) -> None:
        span = MagicMock()
        request = MCPToolCallRequest(
            name="tool_a", args={}, server_name="server_a"
        )
        result = CallToolResult(
            content=[
                TextContent(type="text", text="hello"),
                ResourceLink(
                    type="resource_link",
                    uri="https://example.com/foo.txt",
                    name="foo.txt",
                    mimeType="text/plain",
                ),
            ],
            structuredContent={"final_message": "done"},
        )

        async def handler(req: MCPToolCallRequest) -> CallToolResult:
            return result

        with patch(
            "dao_ai.tools.mcp_interceptors.mlflow.get_current_active_span",
            return_value=span,
        ):
            interceptor = DaoAiStructuredOutputInterceptor()
            out = asyncio.run(interceptor(request, handler))
        assert out is result
        span.set_attribute.assert_any_call("mcp.structured_output", True)
        event_calls = [
            c for c in span.add_event.call_args_list if c[0][0] == "mcp.resource_link"
        ]
        assert len(event_calls) == 1
        attrs = event_calls[0][1]["attributes"]
        assert attrs["mcp.resource_link.count"] == 1
        assert any(
            "example.com/foo.txt" in uri for uri in attrs["mcp.resource_link.uris"]
        )

    def test_no_resource_link_no_event(self) -> None:
        span = MagicMock()
        request = MCPToolCallRequest(name="t", args={}, server_name="s")
        result = CallToolResult(content=[TextContent(type="text", text="hi")])

        async def handler(req: MCPToolCallRequest) -> CallToolResult:
            return result

        with patch(
            "dao_ai.tools.mcp_interceptors.mlflow.get_current_active_span",
            return_value=span,
        ):
            interceptor = DaoAiStructuredOutputInterceptor()
            asyncio.run(interceptor(request, handler))
        assert not any(
            c[0][0] == "mcp.resource_link" for c in span.add_event.call_args_list
        )


class TestTraceInterceptor:
    def test_injects_trace_id_header(self) -> None:
        span = MagicMock()
        span.request_id = "req-abc"
        span.trace_id = None
        request = MCPToolCallRequest(name="t", args={}, server_name="s")
        captured: dict = {}

        async def handler(req: MCPToolCallRequest) -> CallToolResult:
            captured["headers"] = req.headers
            return CallToolResult(content=[TextContent(type="text", text="x")])

        with patch(
            "dao_ai.tools.mcp_interceptors.mlflow.get_current_active_span",
            return_value=span,
        ):
            interceptor = DaoAiTraceInterceptor()
            asyncio.run(interceptor(request, handler))
        assert captured["headers"] == {"x-dao-ai-trace-id": "req-abc"}

    def test_no_span_no_header(self) -> None:
        request = MCPToolCallRequest(
            name="t", args={}, server_name="s", headers={"a": "b"}
        )
        captured: dict = {}

        async def handler(req: MCPToolCallRequest) -> CallToolResult:
            captured["headers"] = req.headers
            return CallToolResult(content=[TextContent(type="text", text="x")])

        with patch(
            "dao_ai.tools.mcp_interceptors.mlflow.get_current_active_span",
            return_value=None,
        ):
            interceptor = DaoAiTraceInterceptor()
            asyncio.run(interceptor(request, handler))
        assert captured["headers"] == {"a": "b"}
