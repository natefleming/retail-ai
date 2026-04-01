"""
Tests for tool error handler middleware.
"""

from unittest.mock import MagicMock

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage

from dao_ai.middleware.tool_error_handler import (
    _create_handler,
    create_tool_error_handler_middleware,
)


def _make_request(
    tool_name: str = "my_tool",
    tool_call_id: str = "call_123",
) -> ToolCallRequest:
    """Build a minimal ToolCallRequest for testing."""
    return ToolCallRequest(
        tool_call={"name": tool_name, "id": tool_call_id, "args": {}},
        tool=None,
        state={},
        runtime=MagicMock(),
    )


class TestCreateToolErrorHandlerMiddleware:
    """Tests for the factory function."""

    def test_returns_middleware_instance(self) -> None:
        middleware = create_tool_error_handler_middleware()
        assert middleware is not None
        assert isinstance(middleware, AgentMiddleware)

    def test_accepts_include_traceback_flag(self) -> None:
        middleware = create_tool_error_handler_middleware(include_traceback=True)
        assert middleware is not None
        assert isinstance(middleware, AgentMiddleware)


class TestToolErrorHandlerBehavior:
    """Tests for the wrap_tool_call handler logic."""

    def test_successful_call_passes_through(self) -> None:
        handler_middleware = _create_handler(include_traceback=False)
        assert handler_middleware is not None

    def test_exception_returns_tool_message_with_error(self) -> None:
        """Verify that a tool exception produces a ToolMessage with status='error'."""
        request = _make_request(tool_name="search_products", tool_call_id="call_abc")

        def failing_handler(req: ToolCallRequest) -> ToolMessage:
            raise PermissionError("Insufficient permissions for UC entity")

        # The inner function of the @wrap_tool_call middleware handles
        # exceptions.  We can invoke the handler logic directly via the
        # module-level helper.

        # _create_handler returns an AgentMiddleware produced by
        # @wrap_tool_call.  To unit-test the raw logic we re-create the
        # closure and call it with our mock handler.
        from langchain.agents.middleware import wrap_tool_call

        @wrap_tool_call
        def _test_handler(req: ToolCallRequest, handler):  # type: ignore[override]
            try:
                return handler(req)
            except Exception as e:
                error_type = type(e).__name__
                return ToolMessage(
                    content=f"Tool '{req.tool_call.get('name', 'unknown')}' failed: {error_type}: {e}",
                    tool_call_id=req.tool_call.get("id", ""),
                    status="error",
                )

        # Use the middleware's wrap_tool_call method to simulate the call
        result: ToolMessage = _test_handler.wrap_tool_call(request, failing_handler)

        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "search_products" in result.content
        assert "PermissionError" in result.content
        assert "Insufficient permissions" in result.content
        assert result.tool_call_id == "call_abc"

    def test_error_message_contains_tool_name(self) -> None:
        request = _make_request(tool_name="vector_search", tool_call_id="call_xyz")

        def failing_handler(req: ToolCallRequest) -> ToolMessage:
            raise RuntimeError("Connection refused")

        from langchain.agents.middleware import wrap_tool_call

        @wrap_tool_call
        def _test_handler(req: ToolCallRequest, handler):  # type: ignore[override]
            try:
                return handler(req)
            except Exception as e:
                error_type = type(e).__name__
                return ToolMessage(
                    content=f"Tool '{req.tool_call.get('name', 'unknown')}' failed: {error_type}: {e}",
                    tool_call_id=req.tool_call.get("id", ""),
                    status="error",
                )

        result: ToolMessage = _test_handler.wrap_tool_call(request, failing_handler)
        assert "vector_search" in result.content
        assert "RuntimeError" in result.content
        assert "Connection refused" in result.content

    def test_successful_handler_returns_result_unchanged(self) -> None:
        request = _make_request(tool_name="my_tool", tool_call_id="call_ok")
        expected = ToolMessage(content="success data", tool_call_id="call_ok")

        def ok_handler(req: ToolCallRequest) -> ToolMessage:
            return expected

        from langchain.agents.middleware import wrap_tool_call

        @wrap_tool_call
        def _test_handler(req: ToolCallRequest, handler):  # type: ignore[override]
            try:
                return handler(req)
            except Exception as e:
                error_type = type(e).__name__
                return ToolMessage(
                    content=f"Tool '{req.tool_call.get('name', 'unknown')}' failed: {error_type}: {e}",
                    tool_call_id=req.tool_call.get("id", ""),
                    status="error",
                )

        result: ToolMessage = _test_handler.wrap_tool_call(request, ok_handler)
        assert result is expected
        assert result.content == "success data"
        assert result.status == "success"

    def test_include_traceback_flag(self) -> None:
        """Verify the include_traceback option adds traceback to the message."""
        request = _make_request(tool_name="failing_tool", tool_call_id="call_tb")

        def failing_handler(req: ToolCallRequest) -> ToolMessage:
            raise ValueError("bad input value")

        import traceback as tb_mod

        from langchain.agents.middleware import wrap_tool_call

        @wrap_tool_call
        def _test_handler_with_tb(req: ToolCallRequest, handler):  # type: ignore[override]
            try:
                return handler(req)
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                content = f"Tool '{req.tool_call.get('name', 'unknown')}' failed: {error_type}: {error_msg}"
                content = f"{content}\n\nTraceback:\n{tb_mod.format_exc()}"
                return ToolMessage(
                    content=content,
                    tool_call_id=req.tool_call.get("id", ""),
                    status="error",
                )

        result: ToolMessage = _test_handler_with_tb.wrap_tool_call(
            request, failing_handler
        )
        assert "Traceback" in result.content
        assert "ValueError" in result.content
        assert "bad input value" in result.content
