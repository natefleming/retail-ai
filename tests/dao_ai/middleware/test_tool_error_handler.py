"""
Tests for tool error handler middleware.
"""

import asyncio
from unittest.mock import MagicMock

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage

from dao_ai.middleware.tool_error_handler import (
    ToolErrorHandlerMiddleware,
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
        assert isinstance(middleware, ToolErrorHandlerMiddleware)

    def test_accepts_include_traceback_flag(self) -> None:
        middleware = create_tool_error_handler_middleware(include_traceback=True)
        assert middleware is not None
        assert isinstance(middleware, ToolErrorHandlerMiddleware)

    def test_overrides_both_sync_and_async(self) -> None:
        """Both wrap_tool_call and awrap_tool_call must be overridden."""
        mw = create_tool_error_handler_middleware()
        assert type(mw).wrap_tool_call is not AgentMiddleware.wrap_tool_call
        assert type(mw).awrap_tool_call is not AgentMiddleware.awrap_tool_call


class TestSyncWrapToolCall:
    """Tests for the synchronous wrap_tool_call path."""

    def test_successful_handler_returns_result_unchanged(self) -> None:
        mw = ToolErrorHandlerMiddleware()
        request = _make_request(tool_call_id="call_ok")
        expected = ToolMessage(content="success data", tool_call_id="call_ok")

        result = mw.wrap_tool_call(request, lambda req: expected)

        assert result is expected
        assert result.content == "success data"

    def test_exception_returns_error_tool_message(self) -> None:
        mw = ToolErrorHandlerMiddleware()
        request = _make_request(tool_name="search_products", tool_call_id="call_abc")

        def failing(req: ToolCallRequest) -> ToolMessage:
            raise PermissionError("Insufficient permissions for UC entity")

        result = mw.wrap_tool_call(request, failing)

        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "search_products" in result.content
        assert "PermissionError" in result.content
        assert "Insufficient permissions" in result.content
        assert result.tool_call_id == "call_abc"

    def test_error_message_contains_tool_name(self) -> None:
        mw = ToolErrorHandlerMiddleware()
        request = _make_request(tool_name="vector_search", tool_call_id="call_xyz")

        def failing(req: ToolCallRequest) -> ToolMessage:
            raise RuntimeError("Connection refused")

        result = mw.wrap_tool_call(request, failing)
        assert "vector_search" in result.content
        assert "RuntimeError" in result.content
        assert "Connection refused" in result.content

    def test_include_traceback(self) -> None:
        mw = ToolErrorHandlerMiddleware(include_traceback=True)
        request = _make_request(tool_name="failing_tool", tool_call_id="call_tb")

        def failing(req: ToolCallRequest) -> ToolMessage:
            raise ValueError("bad input value")

        result = mw.wrap_tool_call(request, failing)
        assert "Traceback" in result.content
        assert "ValueError" in result.content
        assert "bad input value" in result.content


class TestAsyncWrapToolCall:
    """Tests for the asynchronous awrap_tool_call path."""

    def test_successful_handler_returns_result_unchanged(self) -> None:
        mw = ToolErrorHandlerMiddleware()
        request = _make_request(tool_call_id="call_ok")
        expected = ToolMessage(content="async success", tool_call_id="call_ok")

        async def ok_handler(req: ToolCallRequest) -> ToolMessage:
            return expected

        result = asyncio.run(mw.awrap_tool_call(request, ok_handler))
        assert result is expected
        assert result.content == "async success"

    def test_exception_returns_error_tool_message(self) -> None:
        mw = ToolErrorHandlerMiddleware()
        request = _make_request(tool_name="genie_query", tool_call_id="call_async")

        async def failing(req: ToolCallRequest) -> ToolMessage:
            raise TimeoutError("Genie room timed out")

        result = asyncio.run(mw.awrap_tool_call(request, failing))

        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "genie_query" in result.content
        assert "TimeoutError" in result.content
        assert "Genie room timed out" in result.content
        assert result.tool_call_id == "call_async"

    def test_include_traceback(self) -> None:
        mw = ToolErrorHandlerMiddleware(include_traceback=True)
        request = _make_request(tool_name="uc_func", tool_call_id="call_tb_async")

        async def failing(req: ToolCallRequest) -> ToolMessage:
            raise ValueError("bad async input")

        result = asyncio.run(mw.awrap_tool_call(request, failing))
        assert "Traceback" in result.content
        assert "ValueError" in result.content
        assert "bad async input" in result.content
