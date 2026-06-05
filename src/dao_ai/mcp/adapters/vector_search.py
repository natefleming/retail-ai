"""Adapter for ``dao_ai.tools.create_vector_search_tool``.

The dao-ai factory returns a LangChain ``StructuredTool`` whose runtime is
optional (``vector_search.py:625``), so we can invoke it from MCP directly.
We lift the tool's ``name``, ``description`` and ``args_schema`` straight from
the factory output — no MCP-side hardcoding.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import mlflow
from databricks.sdk import WorkspaceClient
from loguru import logger
from mcp.server.fastmcp import Context, FastMCP

from dao_ai._tracing import to_thread_in_context
from dao_ai.mcp._request_context import current_request_headers, current_request_id
from dao_ai.mcp.adapters import McpAdapter, register_adapter
from dao_ai.tools.vector_search import create_vector_search_tool

VECTOR_SEARCH_FACTORY_NAME = "dao_ai.tools.create_vector_search_tool"

AnyContext = Context[Any, Any, Any]


def register_vector_search(
    mcp: FastMCP,
    tool_name: str,
    args: dict[str, Any],
    workspace_client: WorkspaceClient,
) -> None:
    """Invoke the factory + wrap its StructuredTool as a single MCP tool."""

    lc_tool = create_vector_search_tool(**args)

    # Lift name + description from the factory's output (which already encodes
    # the dao-ai-supplied description from args.description, possibly with the
    # _FUNCTION_DOCS suffix the factory adds for LangGraph consumers).
    mcp_tool_name = getattr(lc_tool, "name", None) or tool_name
    mcp_tool_description = getattr(lc_tool, "description", None) or (
        args.get("description") or f"dao-ai vector search tool '{tool_name}'."
    )

    @mcp.tool(name=mcp_tool_name, description=mcp_tool_description)
    async def vector_search_tool(
        query: str,
        filters: list[dict[str, Any]] | None = None,
        ctx: AnyContext | None = None,
    ) -> dict[str, Any]:
        headers = current_request_headers()
        request_id = current_request_id()
        with logger.contextualize(
            request_id=request_id,
            tool=mcp_tool_name,
            obo_present=bool(headers.get("x-forwarded-access-token")),
        ):
            logger.info(
                "mcp.vs.start",
                query_chars=len(query),
                filter_count=len(filters or []),
            )
            start = time.perf_counter()
            try:
                # to_thread_in_context propagates the caller's contextvars so
                # MLflow's active-span ContextVar reaches the worker thread
                # and the LangChain tool's autolog spans nest correctly.
                raw: Any = await to_thread_in_context(
                    lc_tool.invoke, {"query": query, "filters": filters or []}
                )
            except Exception as exc:
                latency_ms = int((time.perf_counter() - start) * 1000)
                logger.exception("mcp.vs.failed", latency_ms=latency_ms)
                raise RuntimeError(
                    f"{mcp_tool_name} failed after {latency_ms}ms: {exc}"
                ) from exc

            latency_ms = int((time.perf_counter() - start) * 1000)
            (logger.warning if latency_ms > 5000 else logger.info)(
                "mcp.vs.done", latency_ms=latency_ms
            )

            docs = _to_docs(raw)
            return {
                "results": docs,
                "_meta": {
                    "tool_name": mcp_tool_name,
                    "result_count": len(docs) if isinstance(docs, list) else None,
                    "latency_ms": latency_ms,
                    "trace_id": _current_trace_id(),
                },
            }

    logger.info("mcp.adapter.vector_search.registered", tool_name=mcp_tool_name)


def _to_docs(raw: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return raw
    return raw


def _current_trace_id() -> str | None:
    try:
        span = mlflow.get_current_active_span()
    except Exception:
        return None
    if span is None:
        return None
    try:
        return span.request_id
    except Exception:
        return getattr(span, "trace_id", None)


register_adapter(
    McpAdapter(factory_name=VECTOR_SEARCH_FACTORY_NAME, register=register_vector_search)
)
