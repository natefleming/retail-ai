"""Per-request HTTP context shared across the FastMCP tool surface.

FastMCP tools receive a ``Context`` object, but Streamable-HTTP-transport
request headers (notably ``x-forwarded-access-token``, ``x-request-id``) are
not part of the MCP protocol envelope. We capture them in a contextvar via a
Starlette middleware and expose typed accessors here so tools can read
headers without coupling to a specific transport.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable, Mapping
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

RequestResponseEndpoint = Callable[[Request], Awaitable[Response]]

_EMPTY_HEADERS: Mapping[str, str] = {}
_request_headers: ContextVar[Mapping[str, str]] = ContextVar(
    "dao_ai_mcp_request_headers", default=_EMPTY_HEADERS
)
_request_id: ContextVar[str] = ContextVar("dao_ai_mcp_request_id", default="")


def current_request_headers() -> Mapping[str, str]:
    """Return the headers of the in-flight HTTP request (lowercased keys)."""
    return _request_headers.get()


def current_request_id() -> str:
    """Return the request id assigned to the in-flight request."""
    return _request_id.get()


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Capture request headers + assign a request_id for every HTTP request."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        headers = {k.lower(): v for k, v in request.headers.items()}
        req_id = headers.get("x-request-id") or uuid.uuid4().hex
        headers_token = _request_headers.set(headers)
        req_id_token = _request_id.set(req_id)
        try:
            response: Response = await call_next(request)
            response.headers["x-request-id"] = req_id
            return response
        finally:
            _request_headers.reset(headers_token)
            _request_id.reset(req_id_token)
