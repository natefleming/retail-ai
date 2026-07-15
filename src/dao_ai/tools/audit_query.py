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
from psycopg import sql
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from pydantic import ConfigDict, Field

from dao_ai.audit import AuditSinkManager, LakebaseAuditSink
from dao_ai.config import AuditModel

__all__ = [
    "AuditToolkit",
    "ToolLike",
    "as_tool_list",
    "as_toolkit",
    "create_audit_query_tools",
    "create_audit_toolkit",
    "create_find_security_incidents_tool",
    "create_get_approver_activity_tool",
    "create_get_audit_receipt_by_id_tool",
    "create_get_thread_timeline_tool",
    "create_query_audit_receipts_tool",
    "create_summarize_audit_activity_tool",
    "create_verify_audit_hash_chain_tool",
]


AuditConfigInput = Union[AuditModel, dict[str, Any]]
ToolLike = Union[BaseTool, Sequence[BaseTool], BaseToolkit]


def _coerce_audit_model(audit: AuditConfigInput) -> AuditModel:
    """Accept either an ``AuditModel`` instance or a dict and return a model."""
    if isinstance(audit, AuditModel):
        return audit
    if isinstance(audit, dict):
        return AuditModel.model_validate(audit)
    raise TypeError(
        f"`audit` must be an AuditModel or dict, got {type(audit).__name__}"
    )


def as_tool_list(items: Optional[ToolLike]) -> list[BaseTool]:
    """Normalise ``BaseTool | Sequence[BaseTool] | BaseToolkit | None`` to ``list[BaseTool]``.

    Public shape-adapter for callers composing tool factories. Accepts:

    - a single ``BaseTool`` → wrapped in a one-element list.
    - a ``Sequence[BaseTool]`` (list, tuple) → copied to a fresh list;
      strings are rejected explicitly so ``str`` won't sneak through as a
      "sequence of chars".
    - a ``BaseToolkit`` → expanded via ``get_tools()``.
    - ``None`` → empty list.

    Raises ``TypeError`` on any other input so misuse fails loudly at
    the composition site rather than downstream in the middleware.
    """
    if items is None:
        return []
    if isinstance(items, BaseToolkit):
        return list(items.get_tools())
    if isinstance(items, BaseTool):
        return [items]
    if isinstance(items, str) or isinstance(items, bytes):
        raise TypeError(
            f"as_tool_list expected BaseTool | Sequence[BaseTool] | BaseToolkit; "
            f"got {type(items).__name__}"
        )
    if isinstance(items, Sequence):
        collected: list[BaseTool] = []
        for entry in items:
            if not isinstance(entry, BaseTool):
                raise TypeError(
                    f"as_tool_list Sequence must contain BaseTool instances; "
                    f"got {type(entry).__name__}"
                )
            collected.append(entry)
        return collected
    raise TypeError(
        f"as_tool_list expected BaseTool | Sequence[BaseTool] | BaseToolkit; "
        f"got {type(items).__name__}"
    )


def as_toolkit(items: ToolLike) -> BaseToolkit:
    """Normalise any of the three shapes into an :class:`AuditToolkit`-shaped ``BaseToolkit``.

    Useful when a caller needs to hand a downstream API a ``BaseToolkit``
    but has only individual tools or a list. Any existing ``BaseToolkit``
    passed in is returned unchanged so identity is preserved (no wrapping
    around a wrapping).
    """
    if isinstance(items, BaseToolkit):
        return items
    return AuditToolkit(tools=as_tool_list(items))


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
    "hitl_involved",
)


def _row_to_receipt(row: dict[str, Any]) -> dict[str, Any]:
    """Normalise a psycopg row for agent consumption (drop sensitive fields, ISO dates).

    The receipts table exposes a first-class ``hitl_involved`` GENERATED
    column (see ``src/dao_ai/audit/ddl.sql``) that always populates —
    the normaliser simply passes it through.
    """
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
        hitl_involved: Optional[bool] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        List dao-ai audit receipts for tool invocations.

        Every audited tool call writes exactly one receipt. Use this tool to
        answer questions like "who approved the last refund?", "show
        rejections in the past hour", "which invocations required human
        approval?", or "list every audited call in this thread".

        Every returned row includes a synthesised ``hitl_involved``
        boolean — true when the receipt came from a HITL-audited tool
        call (approval, edit, rejection, respond, or args-mismatch),
        false when it's a pure audit-only execution receipt.

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
            hitl_involved: When True, only receipts from HITL-audited tool
                calls (any decision + all rejections). When False, only
                pure audit-only execution receipts. When None (default),
                no filter.
            since: ISO-8601 lower bound on recorded_at (inclusive).
            until: ISO-8601 upper bound on recorded_at (exclusive).
            limit: Maximum number of rows to return, 1-200 (default 20).

        Returns:
            List of receipts ordered by recorded_at DESC. Each row includes
            the hash-chain link fields (prev_hash, this_hash) so downstream
            code can verify integrity, plus a synthesised ``hitl_involved``
            boolean.
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
        if hitl_involved is True:
            clauses.append("hitl_involved = TRUE")
        elif hitl_involved is False:
            clauses.append("hitl_involved = FALSE")
        if since is not None:
            clauses.append("recorded_at >= %s")
            params.append(_parse_iso(since))
        if until is not None:
            clauses.append("recorded_at < %s")
            params.append(_parse_iso(until))
        where_sql: sql.Composable = (
            sql.SQL("WHERE ") + sql.SQL(" AND ").join(sql.SQL(c) for c in clauses)
            if clauses
            else sql.SQL("")
        )
        capped_limit: int = max(1, min(int(limit), 200))
        query: sql.Composed = sql.SQL(
            "SELECT {cols} FROM {table} {where} "
            "ORDER BY recorded_at DESC LIMIT %s"
        ).format(
            cols=sql.SQL(", ").join(sql.Identifier(c) for c in _QUERY_SAFE_COLUMNS),
            table=sql.Identifier(receipts_table),
            where=where_sql,
        )
        params.append(capped_limit)

        sink: LakebaseAuditSink = _sink_for(audit_model)
        await sink.ensure_schema()
        rows: list[dict[str, Any]] = await audit_model.database.aexecute_query(
            query, tuple(params)
        )
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
        query: sql.Composed = sql.SQL(
            "SELECT {cols} FROM {table} WHERE receipt_id = %s LIMIT 1"
        ).format(
            cols=sql.SQL(", ").join(sql.Identifier(c) for c in _QUERY_SAFE_COLUMNS),
            table=sql.Identifier(receipts_table),
        )
        sink: LakebaseAuditSink = _sink_for(audit_model)
        await sink.ensure_schema()
        rows: list[dict[str, Any]] = await audit_model.database.aexecute_query(
            query, (receipt_id,)
        )
        if not rows:
            return None
        return _row_to_receipt(rows[0])

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
        query: sql.Composed = sql.SQL(
            "SELECT receipt_id, prev_hash, this_hash, recorded_at "
            "FROM {table} "
            "WHERE thread_id = %s ORDER BY recorded_at ASC"
        ).format(table=sql.Identifier(receipts_table))
        sink: LakebaseAuditSink = _sink_for(audit_model)
        await sink.ensure_schema()
        rows: list[dict[str, Any]] = await audit_model.database.aexecute_query(
            query, (thread_id,)
        )

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


def create_audit_query_tools(
    audit: AuditConfigInput,
    extra_tools: Optional[ToolLike] = None,
) -> list[BaseTool]:
    """Return every audit-query tool bound to the given ``AuditModel``.

    When ``extra_tools`` is provided (as a single ``BaseTool``, a
    ``Sequence[BaseTool]``, or another ``BaseToolkit``) they are appended
    to the returned list so callers can bundle custom tools alongside the
    audit-query surface in a single registration.

    Prefer :func:`create_audit_toolkit` in YAML configs — it wraps the
    same tools in a LangChain :class:`BaseToolkit` that dao-ai's
    factory-tool resolver expands automatically (see
    ``dao_ai.tools.python.create_factory_tool``). Use this list-returning
    variant only when you need to plug tools into a codepath that expects
    a raw list.
    """
    tools: list[BaseTool] = [
        create_query_audit_receipts_tool(audit),
        create_get_audit_receipt_by_id_tool(audit),
        create_verify_audit_hash_chain_tool(audit),
        create_summarize_audit_activity_tool(audit),
        create_find_security_incidents_tool(audit),
        create_get_thread_timeline_tool(audit),
        create_get_approver_activity_tool(audit),
    ]
    tools.extend(as_tool_list(extra_tools))
    return tools


# ----------------------------------------------------------------------
# summarize_audit_activity
# ----------------------------------------------------------------------


def create_summarize_audit_activity_tool(audit: AuditConfigInput) -> BaseTool:
    """Return a tool that produces top-level audit-activity stats for a time window."""
    audit_model: AuditModel = _coerce_audit_model(audit)
    receipts_table: str = audit_model.table

    @tool
    async def summarize_audit_activity(
        since: Optional[str] = None,
        until: Optional[str] = None,
        thread_id: Optional[str] = None,
        tool_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Top-level summary of audit activity — the "how many, of what
        kinds, by whom" report an auditor asks for first.

        Answers questions like:
        - "How many tool invocations were audited last week?"
        - "What's the approve / reject / edit / respond ratio?"
        - "Which tools generate the most audit volume?"
        - "Which approvers were most active?"
        - "How many args-mismatch security incidents fired?"

        Args:
            since: ISO-8601 lower bound on recorded_at (inclusive).
            until: ISO-8601 upper bound on recorded_at (exclusive).
            thread_id: Restrict to a single conversation.
            tool_name: Restrict to invocations of a single tool.

        Returns:
            Dict with:
              - ``total_receipts``: overall count in the window.
              - ``by_receipt_kind``: {execution, rejection, approval}.
              - ``by_decision``: {approve, edit, reject, respond, null}
                (``null`` counts audit-only executions).
              - ``by_execution_status``: {ok, error, args_mismatch,
                not_executed_rejected}.
              - ``hitl_involved``: {true, false}.
              - ``unique_tools``: distinct ``tool_name`` count.
              - ``unique_approvers``: distinct non-null ``approver_sub`` count.
              - ``top_tools``: [{tool_name, receipts}] — top 10 by count.
              - ``top_approvers``: [{approver_sub, receipts}] — top 10 by count.
              - ``args_mismatch_count``: fail-closed rejection count.
              - ``error_count``: ``execution_status='error'`` count.
              - ``window``: {since, until} echoed as ISO strings.
        """
        clauses, params = _build_window_clauses(
            since=since, until=until, thread_id=thread_id, tool_name=tool_name
        )
        table_id: sql.Identifier = sql.Identifier(receipts_table)
        where_sql: sql.Composable = (
            sql.SQL("WHERE ") + sql.SQL(" AND ").join(sql.SQL(c) for c in clauses)
            if clauses
            else sql.SQL("")
        )
        where_with_approver_sql: sql.Composable = (
            where_sql + sql.SQL(" AND approver_sub IS NOT NULL")
            if clauses
            else sql.SQL("WHERE approver_sub IS NOT NULL")
        )

        sink: LakebaseAuditSink = _sink_for(audit_model)
        await sink.ensure_schema()
        pool: AsyncConnectionPool = await audit_model.database.aget_pool()
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    sql.SQL("SELECT COUNT(*) AS n FROM {table} {where}").format(
                        table=table_id, where=where_sql
                    ),
                    tuple(params),
                )
                total: int = int((await cur.fetchone() or {"n": 0})["n"])

                await cur.execute(
                    sql.SQL(
                        "SELECT receipt_kind, COUNT(*) AS n "
                        "FROM {table} {where} GROUP BY receipt_kind"
                    ).format(table=table_id, where=where_sql),
                    tuple(params),
                )
                by_receipt_kind: dict[str, int] = {
                    r["receipt_kind"]: int(r["n"]) for r in await cur.fetchall()
                }

                await cur.execute(
                    sql.SQL(
                        "SELECT COALESCE(decision, 'null') AS decision, COUNT(*) AS n "
                        "FROM {table} {where} GROUP BY decision"
                    ).format(table=table_id, where=where_sql),
                    tuple(params),
                )
                by_decision: dict[str, int] = {
                    r["decision"]: int(r["n"]) for r in await cur.fetchall()
                }

                await cur.execute(
                    sql.SQL(
                        "SELECT COALESCE(execution_status, 'null') AS execution_status, "
                        "COUNT(*) AS n FROM {table} {where} GROUP BY execution_status"
                    ).format(table=table_id, where=where_sql),
                    tuple(params),
                )
                by_execution_status: dict[str, int] = {
                    r["execution_status"]: int(r["n"]) for r in await cur.fetchall()
                }

                await cur.execute(
                    sql.SQL(
                        "SELECT hitl_involved, COUNT(*) AS n "
                        "FROM {table} {where} GROUP BY hitl_involved"
                    ).format(table=table_id, where=where_sql),
                    tuple(params),
                )
                hitl_involved_counts: dict[str, int] = {
                    ("true" if r["hitl_involved"] else "false"): int(r["n"])
                    for r in await cur.fetchall()
                }

                await cur.execute(
                    sql.SQL(
                        "SELECT tool_name, COUNT(*) AS n "
                        "FROM {table} {where} "
                        "GROUP BY tool_name ORDER BY n DESC LIMIT 10"
                    ).format(table=table_id, where=where_sql),
                    tuple(params),
                )
                top_tools: list[dict[str, Any]] = [
                    {"tool_name": r["tool_name"], "receipts": int(r["n"])}
                    for r in await cur.fetchall()
                ]

                await cur.execute(
                    sql.SQL(
                        "SELECT approver_sub, COUNT(*) AS n "
                        "FROM {table} {where} "
                        "GROUP BY approver_sub ORDER BY n DESC LIMIT 10"
                    ).format(table=table_id, where=where_with_approver_sql),
                    tuple(params),
                )
                top_approvers: list[dict[str, Any]] = [
                    {"approver_sub": r["approver_sub"], "receipts": int(r["n"])}
                    for r in await cur.fetchall()
                ]

                await cur.execute(
                    sql.SQL(
                        "SELECT COUNT(DISTINCT tool_name) AS n FROM {table} {where}"
                    ).format(table=table_id, where=where_sql),
                    tuple(params),
                )
                unique_tools: int = int((await cur.fetchone() or {"n": 0})["n"])

                await cur.execute(
                    sql.SQL(
                        "SELECT COUNT(DISTINCT approver_sub) AS n "
                        "FROM {table} {where}"
                    ).format(table=table_id, where=where_with_approver_sql),
                    tuple(params),
                )
                unique_approvers: int = int((await cur.fetchone() or {"n": 0})["n"])

        return {
            "total_receipts": total,
            "by_receipt_kind": by_receipt_kind,
            "by_decision": by_decision,
            "by_execution_status": by_execution_status,
            "hitl_involved": hitl_involved_counts,
            "unique_tools": unique_tools,
            "unique_approvers": unique_approvers,
            "top_tools": top_tools,
            "top_approvers": top_approvers,
            "args_mismatch_count": by_execution_status.get("args_mismatch", 0),
            "error_count": by_execution_status.get("error", 0),
            "window": {"since": since, "until": until},
        }

    return summarize_audit_activity


# ----------------------------------------------------------------------
# find_security_incidents
# ----------------------------------------------------------------------


def create_find_security_incidents_tool(audit: AuditConfigInput) -> BaseTool:
    """Return a tool that surfaces anything an auditor would flag as "smoke"."""
    audit_model: AuditModel = _coerce_audit_model(audit)
    receipts_table: str = audit_model.table

    @tool
    async def find_security_incidents(
        since: Optional[str] = None,
        until: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        List audit receipts that represent security-relevant incidents.

        Surfaces the three high-signal patterns an auditor triages first:

        - ``execution_status = 'args_mismatch'`` — fail-closed rejection
          because arguments changed between HITL approval and execution.
          Strong indicator of tampering or a middleware bug.
        - ``execution_status = 'error'`` — tool ran but raised. Not
          always malicious, but auditor-visible.
        - ``execution_status = 'not_executed_rejected'`` on receipts with
          ``decision != 'reject'`` — an unusual state where execution
          was blocked without an explicit user rejection.

        Args:
            since: ISO-8601 lower bound on recorded_at (inclusive).
            until: ISO-8601 upper bound on recorded_at (exclusive).
            limit: Maximum incidents to return, 1-200 (default 50).

        Returns:
            List of ``{receipt_id, thread_id, tool_name, approver_sub,
            incident_type, execution_status, execution_error, recorded_at,
            hitl_involved}`` ordered by recorded_at DESC. ``incident_type``
            is one of ``args_mismatch`` | ``execution_error`` |
            ``anomalous_non_execution``.
        """
        clauses, params = _build_window_clauses(since=since, until=until)
        clauses.append(
            "(execution_status IN ('args_mismatch','error') "
            "OR (execution_status = 'not_executed_rejected' "
            "AND (decision IS NULL OR decision NOT IN ('reject','respond'))))"
        )
        where_sql: sql.Composable = sql.SQL("WHERE ") + sql.SQL(" AND ").join(
            sql.SQL(c) for c in clauses
        )
        capped_limit: int = max(1, min(int(limit), 200))
        query: sql.Composed = sql.SQL(
            "SELECT receipt_id, thread_id, tool_name, approver_sub, "
            "execution_status, execution_error, hitl_involved, "
            "recorded_at, decision "
            "FROM {table} {where} "
            "ORDER BY recorded_at DESC LIMIT %s"
        ).format(table=sql.Identifier(receipts_table), where=where_sql)
        params.append(capped_limit)

        sink: LakebaseAuditSink = _sink_for(audit_model)
        await sink.ensure_schema()
        rows: list[dict[str, Any]] = await audit_model.database.aexecute_query(
            query, tuple(params)
        )

        def _incident_type(row: dict[str, Any]) -> str:
            status: Any = row.get("execution_status")
            if status == "args_mismatch":
                return "args_mismatch"
            if status == "error":
                return "execution_error"
            return "anomalous_non_execution"

        return [
            {
                "receipt_id": row["receipt_id"],
                "thread_id": row["thread_id"],
                "tool_name": row["tool_name"],
                "approver_sub": row["approver_sub"],
                "incident_type": _incident_type(row),
                "execution_status": row["execution_status"],
                "execution_error": row["execution_error"],
                "hitl_involved": bool(row["hitl_involved"]),
                "decision": row.get("decision"),
                "recorded_at": (
                    row["recorded_at"].isoformat()
                    if isinstance(row["recorded_at"], datetime)
                    else row["recorded_at"]
                ),
            }
            for row in rows
        ]

    return find_security_incidents


# ----------------------------------------------------------------------
# get_thread_timeline
# ----------------------------------------------------------------------


def create_get_thread_timeline_tool(audit: AuditConfigInput) -> BaseTool:
    """Return a tool that produces a full audit timeline for a single thread."""
    audit_model: AuditModel = _coerce_audit_model(audit)
    receipts_table: str = audit_model.table

    @tool
    async def get_thread_timeline(thread_id: str) -> dict[str, Any]:
        """
        Return the complete audit timeline for a conversation thread.

        Every audited tool invocation in this thread, in order — so an
        auditor can reconstruct "what did the agent do, in what
        sequence, and did the human approve each sensitive step?".

        Also inline-verifies the hash chain: any break shows up as
        ``chain_break: true`` on the offending receipt.

        Args:
            thread_id: LangGraph thread identifier.

        Returns:
            Dict with:
              - ``thread_id``: the thread queried
              - ``receipt_count``: number of receipts in the timeline
              - ``chain_valid``: True when every receipt's prev_hash
                matches the previous receipt's this_hash
              - ``timeline``: ordered list of receipt summaries, each
                including ``chain_break`` if the previous link doesn't
                match.
        """
        query: sql.Composed = sql.SQL(
            "SELECT receipt_id, tool_name, receipt_kind, decision, "
            "decision_detail, approver_sub, obo_token_sub, "
            "execution_status, hitl_involved, args_hash, "
            "args_hash_at_interrupt, prev_hash, this_hash, "
            "mlflow_trace_id, recorded_at "
            "FROM {table} "
            "WHERE thread_id = %s "
            "ORDER BY recorded_at ASC"
        ).format(table=sql.Identifier(receipts_table))

        sink: LakebaseAuditSink = _sink_for(audit_model)
        await sink.ensure_schema()
        rows: list[dict[str, Any]] = await audit_model.database.aexecute_query(
            query, (thread_id,)
        )

        timeline: list[dict[str, Any]] = []
        expected_prev: Optional[str] = None
        chain_valid: bool = True
        for row in rows:
            actual_prev: Optional[str] = row["prev_hash"]
            chain_break: bool = actual_prev != expected_prev
            if chain_break:
                chain_valid = False
            timeline.append(
                {
                    "receipt_id": row["receipt_id"],
                    "tool_name": row["tool_name"],
                    "receipt_kind": row["receipt_kind"],
                    "decision": row["decision"],
                    "decision_detail": row["decision_detail"],
                    "approver_sub": row["approver_sub"],
                    "obo_token_sub": row["obo_token_sub"],
                    "execution_status": row["execution_status"],
                    "hitl_involved": bool(row["hitl_involved"]),
                    "args_hash": row["args_hash"],
                    "args_hash_at_interrupt": row["args_hash_at_interrupt"],
                    "mlflow_trace_id": row["mlflow_trace_id"],
                    "recorded_at": (
                        row["recorded_at"].isoformat()
                        if isinstance(row["recorded_at"], datetime)
                        else row["recorded_at"]
                    ),
                    "chain_break": chain_break,
                }
            )
            expected_prev = row["this_hash"]

        return {
            "thread_id": thread_id,
            "receipt_count": len(rows),
            "chain_valid": chain_valid,
            "timeline": timeline,
        }

    return get_thread_timeline


# ----------------------------------------------------------------------
# get_approver_activity
# ----------------------------------------------------------------------


def create_get_approver_activity_tool(audit: AuditConfigInput) -> BaseTool:
    """Return a tool that summarises a single approver's audit trail."""
    audit_model: AuditModel = _coerce_audit_model(audit)
    receipts_table: str = audit_model.table

    @tool
    async def get_approver_activity(
        approver_sub: str,
        since: Optional[str] = None,
        until: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Summarise every audit receipt attributed to a single approver.

        Answers auditor questions like "show me everything user X
        approved / rejected in the last quarter" and "which tools has
        this approver ever authorized?".

        Args:
            approver_sub: The approver identity to look up (matches
                ``receipt.approver_sub``).
            since: ISO-8601 lower bound on recorded_at (inclusive).
            until: ISO-8601 upper bound on recorded_at (exclusive).

        Returns:
            Dict with:
              - ``approver_sub``: the identity queried
              - ``total_decisions``: overall receipt count for this
                approver in the window
              - ``by_decision``: {approve, edit, reject, respond}
              - ``unique_tools``: distinct tool_name count
              - ``tool_breakdown``: [{tool_name, receipts}] ordered by
                count desc
              - ``first_seen`` / ``last_seen``: ISO timestamps of the
                approver's oldest / newest receipt in the window
              - ``window``: {since, until} echoed
        """
        clauses, params = _build_window_clauses(since=since, until=until)
        clauses.append("approver_sub = %s")
        params.append(approver_sub)
        table_id: sql.Identifier = sql.Identifier(receipts_table)
        where_sql: sql.Composable = sql.SQL("WHERE ") + sql.SQL(" AND ").join(
            sql.SQL(c) for c in clauses
        )

        sink: LakebaseAuditSink = _sink_for(audit_model)
        await sink.ensure_schema()
        pool: AsyncConnectionPool = await audit_model.database.aget_pool()
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    sql.SQL(
                        "SELECT COUNT(*) AS n, MIN(recorded_at) AS first_seen, "
                        "MAX(recorded_at) AS last_seen "
                        "FROM {table} {where}"
                    ).format(table=table_id, where=where_sql),
                    tuple(params),
                )
                head: dict[str, Any] = (await cur.fetchone()) or {}

                await cur.execute(
                    sql.SQL(
                        "SELECT COALESCE(decision, 'null') AS decision, COUNT(*) AS n "
                        "FROM {table} {where} GROUP BY decision"
                    ).format(table=table_id, where=where_sql),
                    tuple(params),
                )
                by_decision: dict[str, int] = {
                    r["decision"]: int(r["n"]) for r in await cur.fetchall()
                }

                await cur.execute(
                    sql.SQL(
                        "SELECT tool_name, COUNT(*) AS n "
                        "FROM {table} {where} "
                        "GROUP BY tool_name ORDER BY n DESC"
                    ).format(table=table_id, where=where_sql),
                    tuple(params),
                )
                tool_breakdown: list[dict[str, Any]] = [
                    {"tool_name": r["tool_name"], "receipts": int(r["n"])}
                    for r in await cur.fetchall()
                ]

        first_seen: Any = head.get("first_seen")
        last_seen: Any = head.get("last_seen")
        return {
            "approver_sub": approver_sub,
            "total_decisions": int(head.get("n") or 0),
            "by_decision": by_decision,
            "unique_tools": len(tool_breakdown),
            "tool_breakdown": tool_breakdown,
            "first_seen": (
                first_seen.isoformat() if isinstance(first_seen, datetime) else None
            ),
            "last_seen": (
                last_seen.isoformat() if isinstance(last_seen, datetime) else None
            ),
            "window": {"since": since, "until": until},
        }

    return get_approver_activity


# ----------------------------------------------------------------------
# Shared window-clause helper
# ----------------------------------------------------------------------


def _build_window_clauses(
    *,
    since: Optional[str] = None,
    until: Optional[str] = None,
    thread_id: Optional[str] = None,
    tool_name: Optional[str] = None,
) -> tuple[list[str], list[Any]]:
    """Build shared ``WHERE`` clauses + parameter list for windowed queries."""
    clauses: list[str] = []
    params: list[Any] = []
    if since is not None:
        clauses.append("recorded_at >= %s")
        params.append(_parse_iso(since))
    if until is not None:
        clauses.append("recorded_at < %s")
        params.append(_parse_iso(until))
    if thread_id is not None:
        clauses.append("thread_id = %s")
        params.append(thread_id)
    if tool_name is not None:
        clauses.append("tool_name = %s")
        params.append(tool_name)
    return clauses, params


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


def create_audit_toolkit(
    audit: AuditConfigInput,
    extra_tools: Optional[ToolLike] = None,
) -> AuditToolkit:
    """Return an :class:`AuditToolkit` bound to the given ``AuditModel``.

    When ``extra_tools`` is provided (as a single ``BaseTool``, a
    ``Sequence[BaseTool]``, or another ``BaseToolkit``) those tools are
    bundled into the returned toolkit alongside the audit-query tools —
    dao-ai's factory-tool resolver expands the whole bundle via
    ``get_tools()``.

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
    return AuditToolkit(tools=create_audit_query_tools(audit, extra_tools))


def _parse_iso(value: str) -> datetime:
    """Parse an ISO-8601 timestamp; accept 'Z' as UTC."""
    normalised: str = value.replace("Z", "+00:00") if value.endswith("Z") else value
    return datetime.fromisoformat(normalised)
