"""Adapter for ``dao_ai.tools.create_genie_toolkit``.

dao-ai's Genie tools require LangGraph's ``ToolRuntime`` injection (see
``dao_ai/tools/genie.py:314,683``), so we can't invoke them via
``lc_tool.invoke({...})`` from an MCP context. Instead we rebuild the cache
chain directly from the factory's ``args`` using the same composition order as
``create_genie_toolkit`` itself (``Genie → GenieService →
PostgresContextAwareGenieService → InMemoryContextAwareGenieService →
LRUCacheService``), then surface ``ask_question`` and ``send_feedback`` as
two MCP tools.

Descriptions come from the factory ``args`` verbatim — we do not add any
MCP-side prefix or suffix.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Literal

import mlflow
import pandas as pd  # type: ignore[import-untyped]
from databricks.sdk import WorkspaceClient
from loguru import logger
from mcp.server.fastmcp import Context, FastMCP

from dao_ai.config import (
    DatabaseModel,
    GenieContextAwareCacheParametersModel,
    GenieInMemoryContextAwareCacheParametersModel,
    GenieLRUCacheParametersModel,
    GenieRoomModel,
)
from dao_ai.genie import (
    CacheResult,
    Genie,
    GenieFeedbackRating,
    GenieService,
    GenieServiceBase,
    InMemoryContextAwareGenieService,
    LRUCacheService,
    PostgresContextAwareGenieService,
)
from dao_ai.mcp._request_context import current_request_headers, current_request_id
from dao_ai.mcp.adapters import McpAdapter, register_adapter

GENIE_FACTORY_NAME = "dao_ai.tools.create_genie_toolkit"

AnyContext = Context[Any, Any, Any]

_RATING_MAP: dict[str, GenieFeedbackRating] = {
    "POSITIVE": GenieFeedbackRating.POSITIVE,
    "NEGATIVE": GenieFeedbackRating.NEGATIVE,
    "NONE": GenieFeedbackRating.NONE,
}


def register_genie(
    mcp: FastMCP,
    tool_name: str,
    args: dict[str, Any],
    workspace_client: WorkspaceClient,
) -> None:
    """Register one MCP query tool and one feedback tool for a Genie toolkit entry."""

    description = args.get("description") or f"dao-ai Genie tool '{tool_name}'."
    service, space_id = _build_service(tool_name, args, workspace_client)

    feedback_tool_name = f"{tool_name}_feedback"

    @mcp.tool(name=tool_name, description=description)
    async def query_tool(question: str, ctx: AnyContext) -> dict[str, Any]:
        return await _invoke_query(
            service=service,
            tool_name=tool_name,
            space_id=space_id,
            question=question,
            ctx=ctx,
        )

    @mcp.tool(
        name=feedback_tool_name,
        description=(
            f"Submit feedback on a prior {tool_name} response. NEGATIVE "
            "invalidates the matching cache entry so the next call re-asks Genie. "
            "Provide message_id + conversation_id + cache_hit values from the "
            f"prior {tool_name} _meta."
        ),
    )
    async def feedback_tool(
        conversation_id: str,
        rating: Literal["POSITIVE", "NEGATIVE", "NONE"],
        message_id: str | None = None,
        was_cache_hit: bool = False,
        ctx: AnyContext | None = None,
    ) -> dict[str, Any]:
        return await _invoke_feedback(
            service=service,
            tool_name=tool_name,
            conversation_id=conversation_id,
            rating=rating,
            message_id=message_id,
            was_cache_hit=was_cache_hit,
        )

    logger.info(
        "mcp.adapter.genie.registered",
        tool_name=tool_name,
        feedback_tool=feedback_tool_name,
        space_id=space_id,
    )


def _build_service(
    tool_name: str,
    args: dict[str, Any],
    workspace_client: WorkspaceClient,
) -> tuple[GenieServiceBase, str]:
    """Rebuild the cache chain matching ``create_genie_toolkit``'s composition order."""
    genie_room = _coerce(args.get("genie_room"), GenieRoomModel, "genie_room")
    if not genie_room.space_id:
        raise ValueError(
            f"tools.{tool_name}: genie_room.space_id is required at startup."
        )
    space_id = str(genie_room.space_id)

    lru_params = _coerce_opt(
        args.get("lru_cache_parameters"),
        GenieLRUCacheParametersModel,
        "lru_cache_parameters",
    )
    ctx_params = _coerce_opt(
        args.get("context_aware_cache_parameters"),
        GenieContextAwareCacheParametersModel,
        "context_aware_cache_parameters",
    )
    in_memory_params = _coerce_opt(
        args.get("in_memory_context_aware_cache_parameters"),
        GenieInMemoryContextAwareCacheParametersModel,
        "in_memory_context_aware_cache_parameters",
    )

    # Side-validate that any database dict on the ctx_params is well-formed
    if ctx_params is not None and isinstance(ctx_params.database, dict):
        ctx_params.database = DatabaseModel(**ctx_params.database)

    logger.info(
        "mcp.adapter.genie.build",
        tool_name=tool_name,
        space_id=space_id,
        has_lru=lru_params is not None,
        has_context_aware=ctx_params is not None,
        has_in_memory_context_aware=in_memory_params is not None,
    )

    genie = Genie(space_id=space_id, client=workspace_client)
    service: GenieServiceBase = GenieService(
        genie=genie, workspace_client=workspace_client
    )

    if ctx_params is not None:
        service = PostgresContextAwareGenieService(
            impl=service,
            parameters=ctx_params,
            workspace_client=workspace_client,
            name=f"{tool_name}-postgres-semantic",
        ).initialize()

    if in_memory_params is not None:
        service = InMemoryContextAwareGenieService(
            impl=service,
            parameters=in_memory_params,
            workspace_client=workspace_client,
            name=f"{tool_name}-in-memory-semantic",
        ).initialize()

    if lru_params is not None:
        service = LRUCacheService(
            impl=service, parameters=lru_params, name=f"{tool_name}-lru"
        )

    return service, space_id


async def _invoke_query(
    *,
    service: GenieServiceBase,
    tool_name: str,
    space_id: str,
    question: str,
    ctx: AnyContext,
) -> dict[str, Any]:
    headers = current_request_headers()
    request_id = current_request_id()

    client_meta = _client_meta(ctx)
    conversation_id = client_meta.get("dao-ai/conversation_id")
    disable_cache = bool(client_meta.get("dao-ai/disable_cache", False))

    with logger.contextualize(
        request_id=request_id,
        tool=tool_name,
        space_id=space_id,
        conversation_id=conversation_id,
        obo_present=bool(headers.get("x-forwarded-access-token")),
    ):
        logger.info(
            "mcp.genie.query.start",
            question_chars=len(question),
            disable_cache=disable_cache,
        )

        try:
            await ctx.report_progress(progress=0.0, total=1.0, message="dispatching")
        except Exception:
            logger.debug("mcp.genie.progress.skip", reason="no_progress_token")

        target = service if not disable_cache else _innermost(service)
        start = time.perf_counter()
        try:
            result: CacheResult = await asyncio.to_thread(
                target.ask_question, question, conversation_id
            )
        except Exception as exc:
            latency_ms = int((time.perf_counter() - start) * 1000)
            logger.exception("mcp.genie.query.failed", latency_ms=latency_ms)
            raise RuntimeError(
                f"{tool_name} failed after {latency_ms}ms: {exc}"
            ) from exc

        latency_ms = int((time.perf_counter() - start) * 1000)
        (logger.warning if latency_ms > 5000 else logger.info)(
            "mcp.genie.query.done",
            cache_hit=result.cache_hit,
            served_by=result.served_by,
            latency_ms=latency_ms,
        )

        return _structured_payload(
            result, latency_ms=latency_ms, tool_name=tool_name, space_id=space_id
        )


async def _invoke_feedback(
    *,
    service: GenieServiceBase,
    tool_name: str,
    conversation_id: str,
    rating: str,
    message_id: str | None,
    was_cache_hit: bool,
) -> dict[str, Any]:
    headers = current_request_headers()
    request_id = current_request_id()
    rating_enum = _RATING_MAP[rating]

    with logger.contextualize(
        request_id=request_id,
        tool=tool_name,
        conversation_id=conversation_id,
        message_id=message_id,
        rating=rating,
        obo_present=bool(headers.get("x-forwarded-access-token")),
    ):
        logger.info("mcp.genie.feedback.start", was_cache_hit=was_cache_hit)
        try:
            await asyncio.to_thread(
                service.send_feedback,
                conversation_id,
                rating_enum,
                message_id,
                was_cache_hit,
            )
        except Exception as exc:
            logger.exception("mcp.genie.feedback.failed")
            raise RuntimeError(f"{tool_name}_feedback failed: {exc}") from exc

        logger.info("mcp.genie.feedback.done")
        return {
            "ok": True,
            "_meta": {
                "tool_name": tool_name,
                "conversation_id": conversation_id,
                "message_id": message_id,
                "rating": rating,
                "was_cache_hit": was_cache_hit,
            },
        }


def _client_meta(ctx: AnyContext) -> dict[str, Any]:
    """Best-effort access to the client's ``_meta`` block on ``tools/call``."""
    try:
        meta = getattr(ctx.request_context, "meta", None)
    except Exception:
        return {}
    if meta is None:
        return {}
    if hasattr(meta, "model_dump"):
        try:
            return meta.model_dump(by_alias=True, exclude_none=True)  # type: ignore[no-any-return]
        except Exception:
            pass
    if isinstance(meta, dict):
        return meta
    return {}


def _innermost(service: GenieServiceBase) -> GenieServiceBase:
    """Walk to the underlying ``GenieService`` (no-cache target)."""
    inner: GenieServiceBase = service
    while hasattr(inner, "impl"):
        inner = inner.impl  # type: ignore[attr-defined]
    return inner


def _structured_payload(
    result: CacheResult,
    *,
    latency_ms: int,
    tool_name: str,
    space_id: str,
) -> dict[str, Any]:
    response = result.response
    return {
        "sql": response.query,
        "description": response.description,
        "result_preview": _render_result(response.result),
        "_meta": {
            "tool_name": tool_name,
            "space_id": space_id,
            "cache_hit": result.cache_hit,
            "served_by": result.served_by,
            "latency_ms": latency_ms,
            "message_id": result.message_id,
            "cache_entry_id": result.cache_entry_id,
            "conversation_id": response.conversation_id,
            "trace_id": _current_trace_id(),
        },
    }


def _render_result(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, pd.DataFrame):
        return (
            result.head(50).to_markdown(index=False)
            if not result.empty
            else "(empty)"
        )
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, default=str)
    except Exception:
        return str(result)


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


def _coerce(value: Any, model_cls: type, field_name: str) -> Any:
    if value is None:
        raise ValueError(f"required field {field_name!r} is missing")
    if isinstance(value, model_cls):
        return value
    if isinstance(value, dict):
        return model_cls(**value)
    raise TypeError(
        f"field {field_name!r}: expected {model_cls.__name__} or dict, "
        f"got {type(value).__name__}"
    )


def _coerce_opt(value: Any, model_cls: type, field_name: str) -> Any:
    if value is None:
        return None
    return _coerce(value, model_cls, field_name)


# Self-register at import time.
register_adapter(McpAdapter(factory_name=GENIE_FACTORY_NAME, register=register_genie))
