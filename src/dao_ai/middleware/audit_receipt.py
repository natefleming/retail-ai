"""
Audit-receipt middleware.

Applies to tools whose ``function`` block carries an ``audit`` config.
On every invocation the middleware:

1. Canonicalises the tool call arguments (RFC 8785 JCS) and hashes them.
2. Captures OBO evidence (raw JWT, extracted ``sub``/``exp``) when the
   inbound request carried an ``X-Forwarded-Access-Token`` header. Never
   fabricated.
3. If the tool is also HITL-gated and an interrupt-time hash was stashed,
   verifies ``args_hash_at_execution == args_hash_at_interrupt``.
   Mismatch → **fail-closed**: a rejection receipt is written, an MLflow
   span event ``dao_ai.audit.args_mismatch`` is emitted, the exception
   propagates, and the tool never runs.
4. Invokes the tool.
5. Records a single ``execution`` (or ``rejection``) receipt to the sink,
   sealed into the per-thread hash chain.
6. Attaches non-sensitive audit metadata to the current MLflow span so
   traces link back to the receipt.
"""

from __future__ import annotations

import base64
import binascii
import json
import threading
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional, Sequence

import mlflow
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from loguru import logger

from dao_ai.audit import (
    AuditNonceError,
    AuditReceipt,
    AuditSinkManager,
    ExecutionStatus,
    LakebaseAuditSink,
    ReceiptKind,
    args_hash_of,
    canonical_jcs,
)
from dao_ai.middleware.base import AgentMiddleware

if TYPE_CHECKING:
    from langchain.agents.middleware.types import ToolCallRequest
    from langgraph.prebuilt import ToolRuntime

    from dao_ai.config import AuditModel, ToolModel
    from dao_ai.state import AgentState, Context


__all__ = [
    "AuditReceiptMiddleware",
    "AuditStash",
    "AuditStashEntry",
    "create_audit_middleware_from_tool_models",
    "put_stash",
    "take_stash",
]


# ----------------------------------------------------------------------
# Interrupt-time stash — filled by the HITL enrichment path in PR 4, read
# here on the execution side. Keyed by (thread_id, tool_call_id).
# ----------------------------------------------------------------------


class AuditStashEntry:
    """Data captured at interrupt time and consumed at execution time."""

    __slots__ = (
        "args_hash_at_interrupt",
        "nonce",
        "nonce_exp",
        "displayed_summary",
        "decision",
        "decision_detail",
        "approver_sub",
        "approver_email",
        "confirmed_via",
        "edited_args_jcs",
        "edited_args_hash",
    )

    def __init__(
        self,
        *,
        args_hash_at_interrupt: str,
        nonce: str,
        nonce_exp: datetime,
        displayed_summary: str,
        decision: Optional[str] = None,
        decision_detail: Optional[dict[str, Any]] = None,
        approver_sub: Optional[str] = None,
        approver_email: Optional[str] = None,
        confirmed_via: Optional[str] = None,
        edited_args_jcs: Optional[str] = None,
        edited_args_hash: Optional[str] = None,
    ) -> None:
        self.args_hash_at_interrupt: str = args_hash_at_interrupt
        self.nonce: str = nonce
        self.nonce_exp: datetime = nonce_exp
        self.displayed_summary: str = displayed_summary
        self.decision: Optional[str] = decision
        self.decision_detail: Optional[dict[str, Any]] = decision_detail
        self.approver_sub: Optional[str] = approver_sub
        self.approver_email: Optional[str] = approver_email
        self.confirmed_via: Optional[str] = confirmed_via
        self.edited_args_jcs: Optional[str] = edited_args_jcs
        self.edited_args_hash: Optional[str] = edited_args_hash


class AuditStash:
    """Process-scoped stash guarded by a threading lock."""

    _lock: threading.Lock = threading.Lock()
    _entries: dict[tuple[str, str], AuditStashEntry] = {}

    @classmethod
    def put(cls, thread_id: str, tool_call_id: str, entry: AuditStashEntry) -> None:
        with cls._lock:
            cls._entries[(thread_id, tool_call_id)] = entry

    @classmethod
    def take(
        cls, thread_id: str, tool_call_id: str
    ) -> Optional[AuditStashEntry]:
        """Retrieve and remove the stash entry — single-use, mirrors nonce lifecycle."""
        with cls._lock:
            return cls._entries.pop((thread_id, tool_call_id), None)

    @classmethod
    def reset(cls) -> None:
        """Test-only: clear the stash between tests."""
        with cls._lock:
            cls._entries.clear()


def put_stash(thread_id: str, tool_call_id: str, entry: AuditStashEntry) -> None:
    """Public helper for PR 4's HITL enrichment to inject an interrupt-time entry."""
    AuditStash.put(thread_id, tool_call_id, entry)


def take_stash(
    thread_id: str, tool_call_id: str
) -> Optional[AuditStashEntry]:
    """Public helper for tests to consume a stash entry directly."""
    return AuditStash.take(thread_id, tool_call_id)


# ----------------------------------------------------------------------
# OBO capture — never fabricate; only populate when a real header is present.
# ----------------------------------------------------------------------


def _decode_jwt_claims(token: str) -> Optional[dict[str, Any]]:
    """
    Decode the JWT claims payload without verification.

    Returns None if the token is malformed. This is *not* a security-relevant
    check — the raw token is stored verbatim on the receipt for later
    cryptographic verification against Databricks JWKS. Extracting ``sub`` and
    ``exp`` here just lets us cross-check the header identity claim and
    populate the columns used by the v1.5 purge job.
    """
    try:
        parts: list[str] = token.split(".")
        if len(parts) != 3:
            return None
        payload_b64: str = parts[1]
        padded: str = payload_b64 + "=" * (-len(payload_b64) % 4)
        raw: bytes = base64.urlsafe_b64decode(padded.encode("ascii"))
        claims: dict[str, Any] = json.loads(raw.decode("utf-8"))
        return claims
    except (ValueError, binascii.Error, json.JSONDecodeError):
        return None


def _extract_obo_evidence(
    headers: Optional[dict[str, Any]],
) -> tuple[Optional[str], Optional[datetime], Optional[str]]:
    """
    Return ``(raw_token, exp, sub)`` from ``X-Forwarded-Access-Token``.

    Never fabricates. Missing/invalid header returns ``(None, None, None)``.
    """
    if not headers:
        return None, None, None
    token: Optional[str] = None
    for key in ("X-Forwarded-Access-Token", "x-forwarded-access-token"):
        candidate: Any = headers.get(key)
        if isinstance(candidate, str) and candidate:
            token = candidate
            break
    if token is None:
        return None, None, None
    claims: Optional[dict[str, Any]] = _decode_jwt_claims(token)
    if claims is None:
        return token, None, None
    exp_epoch: Any = claims.get("exp")
    exp_dt: Optional[datetime] = (
        datetime.fromtimestamp(int(exp_epoch), tz=timezone.utc)
        if isinstance(exp_epoch, (int, float))
        else None
    )
    sub: Any = claims.get("sub")
    return token, exp_dt, sub if isinstance(sub, str) else None


def _extract_email(headers: Optional[dict[str, Any]]) -> Optional[str]:
    if not headers:
        return None
    for key in ("X-Forwarded-Email", "x-forwarded-email"):
        candidate: Any = headers.get(key)
        if isinstance(candidate, str) and candidate:
            return candidate
    return None


# ----------------------------------------------------------------------
# Middleware
# ----------------------------------------------------------------------


class AuditReceiptMiddleware(AgentMiddleware):
    """
    LangChain middleware that records tamper-evident audit receipts for a
    fixed set of audited tools.
    """

    def __init__(
        self,
        *,
        audited_tools: dict[str, "AuditModel"],
    ) -> None:
        super().__init__()
        # Map tool_name → resolved sink so we do not incur a manager lookup
        # per invocation.
        self._sinks_by_tool: dict[str, LakebaseAuditSink] = {
            tool_name: AuditSinkManager.for_config(audit_model)
            for tool_name, audit_model in audited_tools.items()
        }
        self._audited_tools: dict[str, "AuditModel"] = dict(audited_tools)

    @property
    def audited_tools(self) -> Sequence[str]:
        """Return the tool names covered by this middleware (defensive copy)."""
        return list(self._audited_tools.keys())

    async def awrap_tool_call(
        self,
        request: "ToolCallRequest",
        handler: Callable[
            ["ToolCallRequest"],
            Awaitable[ToolMessage | Command[Any]],
        ],
    ) -> ToolMessage | Command[Any]:
        tool_name: str = request.tool_call["name"]
        sink: Optional[LakebaseAuditSink] = self._sinks_by_tool.get(tool_name)
        if sink is None:
            # Not one of ours — delegate untouched.
            return await handler(request)

        runtime: "ToolRuntime" = request.runtime  # type: ignore[assignment]
        context: "Context" = runtime.context
        thread_id: str = context.thread_id or _thread_id_from_config(runtime.config)
        tool_call_id: str = (
            request.tool_call.get("id") or runtime.tool_call_id or uuid.uuid4().hex
        )
        args: dict[str, Any] = request.tool_call.get("args") or {}
        args_jcs: str = canonical_jcs(args)
        args_hash: str = args_hash_of(args)

        stash: Optional[AuditStashEntry] = AuditStash.take(thread_id, tool_call_id)
        mlflow_trace_id: Optional[str] = mlflow.get_active_trace_id()

        # 1. Fail-closed args-hash recheck (HITL tools only).
        #
        # For approve decisions, args must be byte-equal to interrupt time.
        # For edit decisions, the LangChain HITL middleware revises args to
        # the user's edited_action.args — those must byte-equal the hash we
        # stashed at _process_decision time.
        if stash is not None:
            expected_hash: str = (
                stash.edited_args_hash
                if stash.decision == "edit" and stash.edited_args_hash is not None
                else stash.args_hash_at_interrupt
            )
            if expected_hash != args_hash:
                await _record_args_mismatch_receipt(
                    sink=sink,
                    thread_id=thread_id,
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    args_jcs=args_jcs,
                    args_hash=args_hash,
                    stash=stash,
                    context=context,
                    mlflow_trace_id=mlflow_trace_id,
                )
                _emit_span_event(
                    "dao_ai.audit.args_mismatch", tool_call_id=tool_call_id
                )
                raise AuditNonceError(
                    f"Args hash mismatch for {tool_name} (decision="
                    f"{stash.decision or 'approve'}): expected "
                    f"{expected_hash[:8]}... vs execution {args_hash[:8]}... "
                    f"— refusing to execute."
                )

        # 2. Invoke the tool.
        execution_status: ExecutionStatus = ExecutionStatus.OK
        execution_error: Optional[str] = None
        try:
            result: ToolMessage | Command[Any] = await handler(request)
        except Exception as exc:  # noqa: BLE001
            execution_status = ExecutionStatus.ERROR
            execution_error = f"{type(exc).__name__}: {exc}"
            try:
                await _record_execution_receipt(
                    sink=sink,
                    thread_id=thread_id,
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    args_jcs=args_jcs,
                    args_hash=args_hash,
                    stash=stash,
                    context=context,
                    mlflow_trace_id=mlflow_trace_id,
                    execution_status=execution_status,
                    execution_error=execution_error,
                )
            except Exception as sink_exc:  # noqa: BLE001
                logger.warning(
                    "Audit sink write failed on tool exception path",
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    sink_error=repr(sink_exc),
                )
            raise

        # 3. Success — record the receipt best-effort.
        try:
            await _record_execution_receipt(
                sink=sink,
                thread_id=thread_id,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                args_jcs=args_jcs,
                args_hash=args_hash,
                stash=stash,
                context=context,
                mlflow_trace_id=mlflow_trace_id,
                execution_status=execution_status,
                execution_error=execution_error,
            )
        except Exception as sink_exc:  # noqa: BLE001
            # Fail-open on sink I/O errors — the caller's tool call already
            # succeeded. Integrity failures (args mismatch, nonce reuse) are
            # separately fail-closed above.
            logger.warning(
                "Audit sink write failed (fail-open on I/O)",
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                sink_error=repr(sink_exc),
            )
        return result


# ----------------------------------------------------------------------
# Helpers used by the middleware
# ----------------------------------------------------------------------


def _thread_id_from_config(config: Any) -> str:
    """Best-effort thread_id extraction from a RunnableConfig."""
    configurable: Any = None
    if isinstance(config, dict):
        configurable = config.get("configurable")
    if isinstance(configurable, dict):
        thread: Any = configurable.get("thread_id")
        if isinstance(thread, str) and thread:
            return thread
    return "unknown-thread"


async def _record_execution_receipt(
    *,
    sink: LakebaseAuditSink,
    thread_id: str,
    tool_call_id: str,
    tool_name: str,
    args_jcs: str,
    args_hash: str,
    stash: Optional[AuditStashEntry],
    context: "Context",
    mlflow_trace_id: Optional[str],
    execution_status: ExecutionStatus,
    execution_error: Optional[str],
) -> None:
    obo_token, obo_exp, obo_sub = _extract_obo_evidence(context.headers)
    approver_email: Optional[str] = _extract_email(context.headers)

    receipt: AuditReceipt = AuditReceipt(
        receipt_id=uuid.uuid4().hex,
        receipt_kind=ReceiptKind.EXECUTION,
        thread_id=thread_id,
        agent_id=None,
        mlflow_trace_id=mlflow_trace_id,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        args_jcs=args_jcs,
        args_hash=args_hash,
        args_hash_at_interrupt=(
            stash.args_hash_at_interrupt if stash is not None else None
        ),
        args_hash_at_resume=(args_hash if stash is not None else None),
        edited_args_jcs=stash.edited_args_jcs if stash is not None else None,
        edited_args_hash=stash.edited_args_hash if stash is not None else None,
        displayed_summary=stash.displayed_summary if stash is not None else None,
        decision=stash.decision if stash is not None else None,
        decision_detail=stash.decision_detail if stash is not None else None,
        approver_sub=(
            stash.approver_sub if stash is not None else None
        ) or (context.user_id if stash is not None else None),
        approver_email=(
            stash.approver_email if stash is not None else None
        ) or (approver_email if stash is not None else None),
        confirmed_via=stash.confirmed_via if stash is not None else None,
        obo_access_token=obo_token,
        obo_token_exp=obo_exp,
        obo_token_sub=obo_sub,
        nonce=stash.nonce if stash is not None else None,
        nonce_exp=stash.nonce_exp if stash is not None else None,
        execution_status=execution_status,
        execution_error=execution_error,
    )

    await sink.record(receipt)
    _attach_span_attributes(
        receipt_id=receipt.receipt_id,
        approver_sub=receipt.approver_sub,
        decision=receipt.decision,
        args_hash=args_hash,
        obo_present=obo_token is not None,
    )


async def _record_args_mismatch_receipt(
    *,
    sink: LakebaseAuditSink,
    thread_id: str,
    tool_call_id: str,
    tool_name: str,
    args_jcs: str,
    args_hash: str,
    stash: AuditStashEntry,
    context: "Context",
    mlflow_trace_id: Optional[str],
) -> None:
    obo_token, obo_exp, obo_sub = _extract_obo_evidence(context.headers)
    approver_email: Optional[str] = _extract_email(context.headers)

    receipt: AuditReceipt = AuditReceipt(
        receipt_id=uuid.uuid4().hex,
        receipt_kind=ReceiptKind.REJECTION,
        thread_id=thread_id,
        mlflow_trace_id=mlflow_trace_id,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        args_jcs=args_jcs,
        args_hash=args_hash,
        args_hash_at_interrupt=stash.args_hash_at_interrupt,
        args_hash_at_resume=args_hash,
        displayed_summary=stash.displayed_summary,
        decision=stash.decision,
        decision_detail=stash.decision_detail,
        approver_sub=stash.approver_sub or context.user_id,
        approver_email=stash.approver_email or approver_email,
        confirmed_via=stash.confirmed_via,
        obo_access_token=obo_token,
        obo_token_exp=obo_exp,
        obo_token_sub=obo_sub,
        nonce=stash.nonce,
        nonce_exp=stash.nonce_exp,
        execution_status=ExecutionStatus.ARGS_MISMATCH,
        execution_error="Args hash differed between interrupt and execution.",
    )
    try:
        await sink.record(receipt)
    except Exception as exc:  # noqa: BLE001
        # Even the mismatch receipt is best-effort I/O; the fail-closed contract
        # is enforced by the raise in the caller.
        logger.warning(
            "Failed to write args_mismatch receipt (still failing closed)",
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            sink_error=repr(exc),
        )
    _attach_span_attributes(
        receipt_id=receipt.receipt_id,
        approver_sub=receipt.approver_sub,
        decision=receipt.decision,
        args_hash=args_hash,
        obo_present=obo_token is not None,
    )


def _attach_span_attributes(
    *,
    receipt_id: str,
    approver_sub: Optional[str],
    decision: Optional[str],
    args_hash: str,
    obo_present: bool,
) -> None:
    """Set audit metadata as span attributes on the current MLflow span."""
    span = mlflow.get_current_active_span()
    if span is None:
        return
    span.set_attribute("dao_ai.audit.receipt_id", receipt_id)
    span.set_attribute("dao_ai.audit.args_hash", args_hash)
    span.set_attribute("dao_ai.audit.obo_token_present", obo_present)
    if approver_sub is not None:
        span.set_attribute("dao_ai.audit.approver_sub", approver_sub)
    if decision is not None:
        span.set_attribute("dao_ai.audit.decision", decision)


def _emit_span_event(name: str, **attributes: Any) -> None:
    """Attach a named event on the current MLflow span for security incidents."""
    span = mlflow.get_current_active_span()
    if span is None:
        return
    # add_event is the OTEL surface exposed by mlflow spans; fall back to a
    # namespaced attribute if not available in this mlflow version.
    add_event = getattr(span, "add_event", None)
    if callable(add_event):
        add_event(name, attributes=attributes)
    else:
        span.set_attribute(f"{name}.marker", True)
        for key, value in attributes.items():
            span.set_attribute(f"{name}.{key}", value)


# ----------------------------------------------------------------------
# Factory — scan tool_models and build one AuditReceiptMiddleware
# ----------------------------------------------------------------------


def create_audit_middleware_from_tool_models(
    tool_models: Sequence["ToolModel"],
) -> Optional[AuditReceiptMiddleware]:
    """
    Return a single ``AuditReceiptMiddleware`` covering every audited tool
    in ``tool_models``, or ``None`` if no tool has ``audit`` set.

    Mirrors ``create_hitl_middleware_from_tool_models`` at
    ``src/dao_ai/middleware/human_in_the_loop.py:170`` in structure and
    naming.
    """
    from dao_ai.config import BaseFunctionModel

    audited: dict[str, "AuditModel"] = {}

    for tool_model in tool_models:
        function = tool_model.function
        if not isinstance(function, BaseFunctionModel):
            continue
        audit_config: Optional["AuditModel"] = function.audit
        if audit_config is None:
            continue
        for func_tool in function.as_tools():
            tool_name: Optional[str] = getattr(func_tool, "name", None)
            if isinstance(tool_name, str) and tool_name:
                audited[tool_name] = audit_config
                logger.trace(
                    "Tool configured for audit",
                    tool_name=tool_name,
                    table=audit_config.table,
                )

    if not audited:
        logger.trace("No tools have audit configured — returning None")
        return None
    return AuditReceiptMiddleware(audited_tools=audited)
