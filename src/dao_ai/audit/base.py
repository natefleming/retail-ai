"""
Core primitives for the dao-ai audit subsystem.

Everything here is pure-Python and dependency-light so importing this module
never pulls Lakebase, Spark, or any other heavyweight backend. Backend
adapters live in sibling modules (``lakebase.py``).

Key primitives:

- ``canonical_jcs`` — RFC 8785 canonicalization of a JSON-compatible dict.
  Used for byte-equal argument hashing between interrupt-time and
  execution-time (WYSIWYS-style semantic binding).
- ``sha256_hex`` — hex SHA-256 helper used by hash-chain, nonce, and
  args-hash flows.
- ``AuditReceipt`` — the canonical receipt schema written to the sink and
  used to compute per-thread hash chains.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import rfc8785
from pydantic import BaseModel, ConfigDict, Field


class ReceiptKind(str, Enum):
    """Discriminates the three receipt scenarios documented in the plan."""

    EXECUTION = "execution"
    APPROVAL = "approval"
    REJECTION = "rejection"


class ExecutionStatus(str, Enum):
    """Terminal status of the tool execution attempt for this receipt."""

    OK = "ok"
    ERROR = "error"
    ARGS_MISMATCH = "args_mismatch"
    NOT_EXECUTED_REJECTED = "not_executed_rejected"


def canonical_jcs(value: Any) -> str:
    """
    Return the RFC 8785 (JCS) canonical JSON string for ``value``.

    RFC 8785 pins Unicode NFC, number canonicalization, and key ordering so
    the same logical dict produces byte-identical output across processes,
    languages, and versions. This matters for the args-hash binding: hand-
    rolled ``sort_keys=True`` variants drift on Unicode + number edge cases.

    ``value`` must be JSON-serialisable; non-serialisable inputs raise a
    ``TypeError`` (rfc8785 delegates to ``json.dumps`` internally).
    """
    return rfc8785.dumps(value).decode("utf-8")


def sha256_hex(data: str | bytes) -> str:
    """Hex SHA-256 digest of ``data``. Strings encoded as UTF-8."""
    payload = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.sha256(payload).hexdigest()


def args_hash_of(args: Any) -> str:
    """Convenience: canonical-JSON-then-hash of a tool call's arguments."""
    return sha256_hex(canonical_jcs(args))


class AuditReceipt(BaseModel):
    """
    Canonical audit receipt written to the sink for each audited event.

    HITL-specific fields are optional so the same schema serves both
    execution-only receipts (tool with ``audit`` but no HITL) and approval
    receipts (tool with both ``audit`` and ``human_in_the_loop``).
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    receipt_id: str = Field(
        description="Unique receipt identifier (uuid4 hex).",
    )
    schema_version: int = Field(
        default=1,
        description="Schema version — bumped when the receipt shape changes.",
    )
    receipt_kind: ReceiptKind = Field(
        description="Discriminator: execution / approval / rejection.",
    )

    thread_id: str = Field(
        description="LangGraph thread identifier for the enclosing conversation.",
    )
    agent_id: Optional[str] = Field(
        default=None,
        description="Agent name that owned the invocation. Optional for backward compat.",
    )
    mlflow_trace_id: Optional[str] = Field(
        default=None,
        description="MLflow trace_id of the enclosing turn, when tracing is active.",
    )
    tool_call_id: Optional[str] = Field(
        default=None,
        description="Tool-call identifier assigned by LangGraph.",
    )
    tool_name: str = Field(
        description="Fully qualified tool name as registered on the agent.",
    )

    args_jcs: str = Field(
        description="RFC 8785 canonical JSON of the executed arguments.",
    )
    args_hash: str = Field(
        description="Hex SHA-256 of args_jcs.",
    )
    args_hash_at_interrupt: Optional[str] = Field(
        default=None,
        description="args_hash captured at interrupt time. HITL only.",
    )
    args_hash_at_resume: Optional[str] = Field(
        default=None,
        description="args_hash recomputed immediately before execution. HITL only.",
    )
    edited_args_jcs: Optional[str] = Field(
        default=None,
        description="Canonical JSON of edited args when decision=edit. HITL only.",
    )
    edited_args_hash: Optional[str] = Field(
        default=None,
        description="Hex SHA-256 of edited_args_jcs. HITL only.",
    )

    displayed_summary: Optional[str] = Field(
        default=None,
        description="Harness-rendered summary string shown to the reviewer. HITL only.",
    )
    decision: Optional[str] = Field(
        default=None,
        description="HITL decision: approve|reject|edit|respond. HITL only.",
    )
    decision_detail: Optional[dict[str, Any]] = Field(
        default=None,
        description="Extra decision payload (edited args, response text, etc.). HITL only.",
    )

    approver_sub: Optional[str] = Field(
        default=None,
        description="Approver principal identifier (context.user_id in v1). HITL only.",
    )
    approver_email: Optional[str] = Field(
        default=None,
        description="Approver email extracted from X-Forwarded-Email. HITL only.",
    )
    confirmed_via: Optional[str] = Field(
        default=None,
        description="Approval channel: chat_ui / obo_jwt / webauthn. HITL only.",
    )

    obo_access_token: Optional[str] = Field(
        default=None,
        description=(
            "Raw OBO JWT captured at receipt time. Never fabricated — populated "
            "only when a real token is present on the inbound request. "
            "Sensitive; UC ACLs restrict SELECT to authorised reviewers."
        ),
    )
    obo_token_exp: Optional[datetime] = Field(
        default=None,
        description="JWT `exp` claim extracted without signature verification.",
    )
    obo_token_sub: Optional[str] = Field(
        default=None,
        description="JWT `sub` claim extracted without signature verification.",
    )

    nonce: Optional[str] = Field(
        default=None,
        description="Server-issued approval nonce. HITL only.",
    )
    nonce_exp: Optional[datetime] = Field(
        default=None,
        description="Nonce expiry timestamp. HITL only.",
    )

    execution_status: Optional[ExecutionStatus] = Field(
        default=None,
        description="Terminal execution status attached to this receipt.",
    )
    execution_error: Optional[str] = Field(
        default=None,
        description="Short error message when execution_status=error.",
    )

    recorded_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Server-side receipt timestamp (UTC).",
    )

    prev_hash: Optional[str] = Field(
        default=None,
        description="Hash of the previous receipt in this thread's chain (NULL for first).",
    )
    this_hash: str = Field(
        default="",
        description="Hash of this receipt's canonical body (excluding this_hash itself).",
    )

    def body_for_hash(self) -> dict[str, Any]:
        """
        Return the canonical dict used to compute ``this_hash``.

        All fields except ``this_hash`` are included, and ``recorded_at`` /
        ``obo_token_exp`` / ``nonce_exp`` / ``mlflow_trace_id`` are stringified
        so the JCS representation stays byte-stable across dialects.
        """
        return json.loads(
            self.model_dump_json(exclude={"this_hash"}),
        )

    def compute_this_hash(self) -> str:
        """Compute SHA-256(JCS(body_for_hash))."""
        return sha256_hex(canonical_jcs(self.body_for_hash()))
