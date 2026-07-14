"""
Agent-facing tools for querying the dao-ai audit trail.

Exposes three LangChain tools that agents can invoke to inspect the
tamper-evident receipts written by ``AuditReceiptMiddleware`` and
``AuditedHumanInTheLoopMiddleware``:

- ``query_audit_receipts`` — filtered listing (thread_id, tool_name,
  decision, receipt_kind, approver_sub, time range).
- ``get_audit_receipt_by_id`` — single-receipt lookup by ``receipt_id``.
- ``verify_audit_hash_chain`` — walks the per-thread chain and reports
  any prev_hash / this_hash mismatch as evidence of tampering.

All three tools are produced by ``create_audit_query_tools`` (or the
individual ``create_*_tool`` factories) which takes an ``AuditModel`` —
either as an already-constructed model instance or as a plain dict that
will be validated into one. This lets a YAML config either reuse an
existing anchor (``args.audit: *audit_config``) or inline the block.

Register via ``FactoryFunctionModel``::

    tools:
      audit_query:
        name: query_audit_receipts
        function:
          type: factory
          name: dao_ai.tools.audit_query.create_query_audit_receipts_tool
          args:
            audit: *audit_config

The tools read from the same Lakebase configured on the ``AuditModel`` —
so read-back tooling and write-side auditing share a single storage
identity + connection pool.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional, Sequence, Union

from langchain_community.agent_toolkits.base import BaseToolkit
from langchain_core.tools import BaseTool, tool
from loguru import logger
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from pydantic import ConfigDict, Field

from dao_ai.audit import AuditSinkManager, LakebaseAuditSink
from dao_ai.config import AuditModel

__all__ = [
    "AuditToolkit",
    "create_audit_query_tools",
    "create_audit_toolkit",
    "create_get_audit_receipt_by_id_tool",
    "create_query_audit_receipts_tool",
    "create_verify_audit_hash_chain_tool",
]


AuditConfigInput = Union[AuditModel, dict[str, Any]]


def _coerce_audit_model(audit: AuditConfigInput) -> AuditModel:
    """Accept either an ``AuditModel`` instance or a dict and return a model."""
    if isinstance(audit, AuditModel):
        return audit
    if isinstance(audit, dict):
        return AuditModel.model_validate(audit)
    raise TypeError(
        f"`audit` must be an AuditModel or dict, got {type(audit).__name__}"
    )


def _sink_for(audit: AuditConfigInput) -> LakebaseAuditSink:
    """Resolve (or reuse) the sink for this audit config."""
    return AuditSinkManager.for_config(_coerce_audit_model(audit))


_QUERY_SAFE_COLUMNS: tuple[str, ...] = (
    "receipt_id",
    "schema_version",
    "receipt_kind",
    "thread_id",
    "agent_id",
    "mlflow_trace_id",
    "tool_call_id",
    "tool_name",
    "args_hash",
    "args_hash_at_interrupt",
    "args_hash_at_resume",
    "displayed_summary",
    "decision",
    "decision_detail",
    "approver_sub",
    "approver_email",
    "confirmed_via",
    "obo_token_sub",
    "obo_token_exp",
    "execution_status",
    "execution_error",
    "recorded_at",
    "prev_hash",
    "this_hash",
)


def _row_to_receipt(row: dict[str, Any]) -> dict[str, Any]:
    """Normalise a psycopg row for agent consumption (drop sensitive fields, ISO dates)."""
    out: dict[str, Any] = {}
    for key, value in row.items():
        if key == "args_jcs":
            # args_jcs can be large; expose args_hash as the identity key instead.
            continue
        if key == "obo_access_token":
            # Sensitive raw JWT — never surface to agents; use obo_token_sub / obo_token_exp
            # for attribution without leaking the credential material.
            continue
        if isinstance(value, datetime):
            out[key] = value.isoformat()
        elif isinstance(value, str) and key == "decision_detail":
            # decision_detail column is JSONB, psycopg returns dict; but if we got str, parse.
            try:
                out[key] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                out[key] = value
        else:
            out[key] = value
    return out


# ----------------------------------------------------------------------
# query_audit_receipts
# ----------------------------------------------------------------------


def create_query_audit_receipts_tool(audit: AuditConfigInput) -> BaseTool:
    """Return a LangChain tool that lists audit receipts with optional filters."""
    audit_model: AuditModel = _coerce_audit_model(audit)
    receipts_table: str = audit_model.table

    @tool
    async def query_audit_receipts(
        thread_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        decision: Optional[str] = None,
        receipt_kind: Optional[str] = None,
        approver_sub: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        List dao-ai audit receipts for tool invocations.

        Every audited tool call writes exactly one receipt. Use this tool to
        answer questions like "who approved the last refund?", "show
        rejections in the past hour", or "list every audited call in this
        thread".

        The raw OBO JWT is never returned — use ``obo_token_sub`` /
        ``obo_token_exp`` for attribution.

        Args:
            thread_id: LangGraph thread identifier to filter to a specific
                conversation.
            tool_name: Restrict to invocations of a specific tool.
            decision: HITL decision — "approve", "reject", "edit", or "respond".
            receipt_kind: "execution" (tool ran), "rejection" (blocked), or
                "approval".
            approver_sub: Filter by approver identity (matches the
                X-Forwarded-User principal at approval time).
            since: ISO-8601 lower bound on recorded_at (inclusive).
            until: ISO-8601 upper bound on recorded_at (exclusive).
            limit: Maximum number of rows to return, 1-200 (default 20).

        Returns:
            List of receipts ordered by recorded_at DESC. Each row includes
            the hash-chain link fields (prev_hash, this_hash) so downstream
            code can verify integrity.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if thread_id is not None:
            clauses.append("thread_id = %s")
            params.append(thread_id)
        if tool_name is not None:
            clauses.append("tool_name = %s")
            params.append(tool_name)
        if decision is not None:
            clauses.append("decision = %s")
            params.append(decision)
        if receipt_kind is not None:
            clauses.append("receipt_kind = %s")
            params.append(receipt_kind)
        if approver_sub is not None:
            clauses.append("approver_sub = %s")
            params.append(approver_sub)
        if since is not None:
            clauses.append("recorded_at >= %s")
            params.append(_parse_iso(since))
        if until is not None:
            clauses.append("recorded_at < %s")
            params.append(_parse_iso(until))
        where: str = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        capped_limit: int = max(1, min(int(limit), 200))
        columns: str = ", ".join(_QUERY_SAFE_COLUMNS)
        sql: str = (
            f"SELECT {columns} FROM {receipts_table} "
            f"{where} ORDER BY recorded_at DESC LIMIT %s"
        )
        params.append(capped_limit)

        sink: LakebaseAuditSink = _sink_for(audit_model)
        await sink.ensure_schema()
        pool: AsyncConnectionPool = await sink._pool()
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(sql, tuple(params))
                rows: list[dict[str, Any]] = list(await cur.fetchall())
        logger.info(
            "query_audit_receipts",
            filters={
                "thread_id": thread_id,
                "tool_name": tool_name,
                "decision": decision,
                "receipt_kind": receipt_kind,
                "approver_sub": approver_sub,
                "since": since,
                "until": until,
            },
            row_count=len(rows),
        )
        return [_row_to_receipt(row) for row in rows]

    return query_audit_receipts


# ----------------------------------------------------------------------
# get_audit_receipt_by_id
# ----------------------------------------------------------------------


def create_get_audit_receipt_by_id_tool(audit: AuditConfigInput) -> BaseTool:
    """Return a LangChain tool that fetches a single audit receipt by receipt_id."""
    audit_model: AuditModel = _coerce_audit_model(audit)
    receipts_table: str = audit_model.table

    @tool
    async def get_audit_receipt_by_id(receipt_id: str) -> Optional[dict[str, Any]]:
        """
        Fetch a single audit receipt by its ``receipt_id``.

        Returns ``None`` when no receipt matches. The raw OBO JWT is not
        included — use ``obo_token_sub`` for attribution.

        Args:
            receipt_id: UUID-hex identifier from a prior ``query_audit_receipts``
                result or an MLflow span attribute ``dao_ai.audit.receipt_id``.

        Returns:
            The receipt as a JSON-serialisable dict, or ``None`` if not found.
        """
        columns: str = ", ".join(_QUERY_SAFE_COLUMNS)
        sql: str = (
            f"SELECT {columns} FROM {receipts_table} WHERE receipt_id = %s LIMIT 1"
        )
        sink: LakebaseAuditSink = _sink_for(audit_model)
        await sink.ensure_schema()
        pool: AsyncConnectionPool = await sink._pool()
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(sql, (receipt_id,))
                row: Optional[dict[str, Any]] = await cur.fetchone()
        if row is None:
            return None
        return _row_to_receipt(row)

    return get_audit_receipt_by_id


# ----------------------------------------------------------------------
# verify_audit_hash_chain
# ----------------------------------------------------------------------


def create_verify_audit_hash_chain_tool(audit: AuditConfigInput) -> BaseTool:
    """Return a tool that walks the per-thread hash chain and reports tampering."""
    audit_model: AuditModel = _coerce_audit_model(audit)
    receipts_table: str = audit_model.table

    @tool
    async def verify_audit_hash_chain(thread_id: str) -> dict[str, Any]:
        """
        Walk the audit hash chain for a thread and report any tampering.

        For every receipt on ``thread_id`` in ``recorded_at`` order, verify
        that ``prev_hash`` matches the previous receipt's ``this_hash``.
        Any mismatch is evidence that a receipt was inserted, mutated, or
        deleted post-hoc — the append-only trigger blocks the SQL surface,
        but this check also catches out-of-band manipulation of the
        underlying storage.

        Args:
            thread_id: LangGraph thread whose chain should be verified.

        Returns:
            Dict with:
              - ``thread_id``: the thread queried
              - ``receipts_checked``: count of receipts examined
              - ``valid``: True when the chain has no gaps or mismatches
              - ``breaks``: list of {index, receipt_id, expected_prev_hash,
                actual_prev_hash} for every break detected
        """
        sql: str = (
            f"SELECT receipt_id, prev_hash, this_hash, recorded_at "
            f"FROM {receipts_table} "
            f"WHERE thread_id = %s ORDER BY recorded_at ASC"
        )
        sink: LakebaseAuditSink = _sink_for(audit_model)
        await sink.ensure_schema()
        pool: AsyncConnectionPool = await sink._pool()
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(sql, (thread_id,))
                rows: list[dict[str, Any]] = list(await cur.fetchall())

        breaks: list[dict[str, Any]] = []
        expected_prev: Optional[str] = None
        for idx, row in enumerate(rows):
            actual_prev: Optional[str] = row["prev_hash"]
            if actual_prev != expected_prev:
                breaks.append(
                    {
                        "index": idx,
                        "receipt_id": row["receipt_id"],
                        "expected_prev_hash": expected_prev,
                        "actual_prev_hash": actual_prev,
                    }
                )
            expected_prev = row["this_hash"]
        return {
            "thread_id": thread_id,
            "receipts_checked": len(rows),
            "valid": not breaks,
            "breaks": breaks,
        }

    return verify_audit_hash_chain


# ----------------------------------------------------------------------
# Bundled factory: all three tools at once
# ----------------------------------------------------------------------


def create_audit_query_tools(audit: AuditConfigInput) -> Sequence[BaseTool]:
    """Return every audit-query tool bound to the given ``AuditModel``.

    Prefer :func:`create_audit_toolkit` in YAML configs — it wraps the same
    tools in a LangChain :class:`BaseToolkit` that dao-ai's factory-tool
    resolver expands automatically (see
    ``dao_ai.tools.python.create_factory_tool``). Use this list-returning
    variant only when you need to plug tools into a codepath that expects
    a raw list.
    """
    return [
        create_query_audit_receipts_tool(audit),
        create_get_audit_receipt_by_id_tool(audit),
        create_verify_audit_hash_chain_tool(audit),
    ]


class AuditToolkit(BaseToolkit):
    """
    LangChain toolkit bundling every dao-ai audit-query tool.

    Follows the same shape as :class:`dao_ai.tools.genie.GenieToolkit`: the
    ``tools`` list is populated at construction time by
    :func:`create_audit_toolkit`, and ``get_tools`` returns the bundle
    verbatim. dao-ai's factory-tool resolver invokes ``get_tools`` when it
    sees a :class:`BaseToolkit` return, so registering this toolkit as a
    single factory tool wires up the whole audit-query surface.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    tools: list[BaseTool] = Field(default_factory=list)

    def get_tools(self) -> list[BaseTool]:
        return list(self.tools)


def create_audit_toolkit(audit: AuditConfigInput) -> AuditToolkit:
    """Return an :class:`AuditToolkit` bound to the given ``AuditModel``.

    Register in YAML with ``type: factory``::

        tools:
          audit_toolkit:
            name: audit_toolkit
            function:
              type: factory
              name: dao_ai.tools.audit_query.create_audit_toolkit
              args:
                audit: *audit_config

    dao-ai's ``create_factory_tool`` sees ``BaseToolkit`` and expands to
    ``get_tools()`` so the agent gets ``query_audit_receipts``,
    ``get_audit_receipt_by_id``, and ``verify_audit_hash_chain`` in one
    registration.
    """
    return AuditToolkit(tools=list(create_audit_query_tools(audit)))


def _parse_iso(value: str) -> datetime:
    """Parse an ISO-8601 timestamp; accept 'Z' as UTC."""
    normalised: str = value.replace("Z", "+00:00") if value.endswith("Z") else value
    return datetime.fromisoformat(normalised)
