"""MCP tool-call interceptors that observe tool-call responses in the
langchain-mcp-adapters onion pipeline.

Wired into ``MultiServerMCPClient(tool_interceptors=[...])`` when an
``McpFunctionModel`` declares an ``McpCapabilitiesModel``. Named classes
(not closures) so their execution shows up in MLflow trace spans.

W3C trace-context propagation (``traceparent``/``tracestate``/``baggage``)
is handled separately via ``_meta`` on ``session.call_tool`` — see
``dao_ai.tools.mcp_trace_context``.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

import mlflow
from langchain_mcp_adapters.interceptors import (
    MCPToolCallRequest,
    MCPToolCallResult,
)
from loguru import logger
from mcp.types import CallToolResult

_HandlerT = Callable[[MCPToolCallRequest], Awaitable[MCPToolCallResult]]


def _active_span_add_attribute(key: str, value: Any) -> None:
    span = mlflow.get_current_active_span()
    if span is None:
        return
    try:
        span.set_attribute(key, value)
    except Exception as exc:
        logger.debug(f"mcp interceptor: failed to set attr {key!r}: {exc}")


def _active_span_add_event(name: str, attributes: dict[str, Any]) -> None:
    span = mlflow.get_current_active_span()
    if span is None:
        return
    try:
        span.add_event(name, attributes=attributes)
    except Exception as exc:
        logger.debug(f"mcp interceptor: failed to add event {name!r}: {exc}")


class DaoAiStructuredOutputInterceptor:
    """Expands ``resource_link`` items from a ``CallToolResult`` into MLflow
    span attributes, and records the presence of ``structuredContent`` so
    downstream consumers can rely on schema-typed output when present.

    The interceptor does NOT swallow errors or transform the result payload —
    it is observation-only. The adapter still returns the same CallToolResult
    to the LangChain tool.
    """

    async def __call__(
        self,
        request: MCPToolCallRequest,
        handler: _HandlerT,
    ) -> MCPToolCallResult:
        result = await handler(request)

        if not isinstance(result, CallToolResult):
            return result

        if getattr(result, "structuredContent", None) is not None:
            _active_span_add_attribute("mcp.structured_output", True)

        resource_links: list[dict[str, Any]] = []
        for item in result.content or []:
            if getattr(item, "type", None) == "resource_link":
                resource_links.append(
                    {
                        "uri": getattr(item, "uri", None),
                        "name": getattr(item, "name", None),
                        "mimeType": getattr(item, "mimeType", None),
                    }
                )

        if resource_links:
            _active_span_add_event(
                "mcp.resource_link",
                {
                    "mcp.server_name": request.server_name,
                    "mcp.tool_name": request.name,
                    "mcp.resource_link.count": len(resource_links),
                    "mcp.resource_link.uris": [
                        str(link.get("uri", "")) for link in resource_links
                    ],
                },
            )
        return result
