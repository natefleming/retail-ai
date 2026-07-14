"""Integration tests for LakebaseAuditSink against a real PostgreSQL server.

These tests exercise the audit sink's SQL layer end-to-end — DDL
provisioning, receipt writes, hash-chain linkage, append-only trigger
enforcement, and the GENERATED ``hitl_involved`` column semantics.
They would have caught both bugs found on FEVM during v1 verification
(string.Template collapsing PL/pgSQL ``$$`` dollar-quoting, and SQL
three-valued-logic NULL-propagation through the OR chain of the
GENERATED column) *before* deploy.

Gated by ``@pytest.mark.integration`` + ``has_postgres_env()`` so they
skip locally when Postgres env vars are absent. Wire a Postgres service
in CI (or export ``PG_*``) to run them.

Environment variables required (any one set):
- ``PG_CONNECTION_STRING`` — a full libpq connection URL
- OR all of: ``PG_HOST``, ``PG_PORT``, ``PG_USER``, ``PG_PASSWORD``,
  ``PG_DATABASE``

Each test uses a unique table name (``audit_receipts_ci_<uuid>``) so
parallel runs don't collide, and a fixture cleans up (``DROP TABLE
IF EXISTS``) after each test.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Iterator

import pytest
from conftest import has_postgres_env
from psycopg import connect
from psycopg import sql as pgsql
from psycopg.errors import Error as PsycopgError

from dao_ai.audit import (
    AuditReceipt,
    AuditSinkManager,
    ExecutionStatus,
    LakebaseAuditSink,
    ReceiptKind,
    args_hash_of,
)
from dao_ai.audit.base import canonical_jcs
from dao_ai.config import AuditModel, DatabaseModel

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not has_postgres_env(),
        reason="PostgreSQL environment variables not available",
    ),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _pg_connection_kwargs() -> dict[str, object]:
    """Build psycopg connect kwargs from the env vars ``has_postgres_env`` gates on."""
    conn_string = os.environ.get("PG_CONNECTION_STRING")
    if conn_string:
        return {"conninfo": conn_string}
    return {
        "host": os.environ["PG_HOST"],
        "port": int(os.environ["PG_PORT"]),
        "user": os.environ["PG_USER"],
        "password": os.environ["PG_PASSWORD"],
        "dbname": os.environ["PG_DATABASE"],
    }


def _audit_model(table: str) -> AuditModel:
    """Build an AuditModel whose DatabaseModel connects via the PG_* env vars."""
    if "PG_CONNECTION_STRING" in os.environ:
        pytest.skip(
            "PG_CONNECTION_STRING not directly usable — supply discrete PG_* vars for these tests"
        )
    database = DatabaseModel(
        name="audit_integration_pg",
        host=os.environ["PG_HOST"],
        port=int(os.environ["PG_PORT"]),
        database=os.environ["PG_DATABASE"],
        user=os.environ["PG_USER"],
        password=os.environ["PG_PASSWORD"],
    )
    return AuditModel(database=database, table=table)


@pytest.fixture
def unique_table_name() -> str:
    """Fresh table name per test so parallel runs don't collide."""
    return f"audit_receipts_ci_{uuid.uuid4().hex[:12]}"


@pytest.fixture
def audit_sink(unique_table_name: str) -> Iterator[LakebaseAuditSink]:
    """Build a sink for the unique table, tear down all tables afterward."""
    AuditSinkManager.reset()  # ensure a fresh sink instance per test
    model = _audit_model(unique_table_name)
    sink = AuditSinkManager.for_config(model)
    try:
        yield sink
    finally:
        try:
            with connect(**_pg_connection_kwargs()) as conn:
                conn.autocommit = True
                with conn.cursor() as cur:
                    cur.execute(
                        pgsql.SQL("DROP TABLE IF EXISTS {tbl} CASCADE").format(
                            tbl=pgsql.Identifier(unique_table_name)
                        )
                    )
                    cur.execute(
                        pgsql.SQL("DROP TABLE IF EXISTS {tbl} CASCADE").format(
                            tbl=pgsql.Identifier(f"{unique_table_name}_nonces")
                        )
                    )
        except PsycopgError:
            pass  # teardown is best-effort
        AuditSinkManager.reset()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_receipt(
    *,
    thread_id: str,
    tool_name: str = "refund_customer",
    args: dict[str, object] | None = None,
    receipt_kind: ReceiptKind = ReceiptKind.EXECUTION,
    decision: str | None = None,
    hitl: bool = False,
) -> AuditReceipt:
    args_dict: dict[str, object] = args or {"amount": 42, "customer_id": "C-1"}
    return AuditReceipt(
        receipt_id=uuid.uuid4().hex,
        receipt_kind=receipt_kind,
        thread_id=thread_id,
        agent_id="ci-test-agent",
        mlflow_trace_id=None,
        tool_call_id=f"call-{uuid.uuid4().hex[:8]}",
        tool_name=tool_name,
        args_jcs=canonical_jcs(args_dict),
        args_hash=args_hash_of(args_dict),
        args_hash_at_interrupt=args_hash_of(args_dict) if hitl else None,
        args_hash_at_resume=args_hash_of(args_dict) if hitl else None,
        decision=decision,
        approver_sub="ci-user@example.com" if hitl else None,
        confirmed_via="chat_ui" if hitl else None,
        execution_status=ExecutionStatus.OK,
    )


def _run(coro):
    """Convenience: run an async coroutine in a fresh loop."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ensure_schema_is_idempotent(audit_sink: LakebaseAuditSink) -> None:
    """Two consecutive ensure_schema calls succeed and produce the same tables."""

    async def scenario() -> None:
        await audit_sink.ensure_schema()
        await audit_sink.ensure_schema()

    _run(scenario())

    # Verify the tables + generated column + trigger all exist.
    with connect(**_pg_connection_kwargs()) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = %s AND column_name = 'hitl_involved'",
                (audit_sink._receipts_table,),
            )
            row = cur.fetchone()
            assert row is not None, "hitl_involved GENERATED column must exist"

            cur.execute(
                "SELECT trigger_name FROM information_schema.triggers "
                "WHERE event_object_table = %s AND event_manipulation IN ('UPDATE','DELETE')",
                (audit_sink._receipts_table,),
            )
            triggers = [row[0] for row in cur.fetchall()]
            assert any(
                "no_update" in name or "no_delete" in name for name in triggers
            ), f"append-only trigger missing; found: {triggers}"


def test_record_and_head_hash_round_trip(audit_sink: LakebaseAuditSink) -> None:
    """A receipt written via record() is retrievable via head_hash()."""

    async def scenario() -> tuple[AuditReceipt, str | None]:
        receipt = _build_receipt(thread_id="thread-A")
        await audit_sink.record(receipt)
        head = await audit_sink.head_hash("thread-A")
        return receipt, head

    sealed, head = _run(scenario())
    assert head is not None
    assert head == sealed.this_hash


def test_hash_chain_links_across_receipts(audit_sink: LakebaseAuditSink) -> None:
    """Second receipt on the same thread carries prev_hash = first.this_hash."""

    async def scenario() -> tuple[AuditReceipt, AuditReceipt]:
        first = _build_receipt(thread_id="thread-B")
        await audit_sink.record(first)
        second = _build_receipt(thread_id="thread-B")
        await audit_sink.record(second)
        return first, second

    first, second = _run(scenario())
    assert first.prev_hash is None
    assert second.prev_hash == first.this_hash


def test_append_only_trigger_blocks_update_and_delete(
    audit_sink: LakebaseAuditSink,
) -> None:
    """UPDATE + DELETE both raise from the trigger with an audit-specific message."""

    async def scenario() -> AuditReceipt:
        receipt = _build_receipt(thread_id="thread-C")
        await audit_sink.record(receipt)
        return receipt

    receipt = _run(scenario())

    with connect(**_pg_connection_kwargs()) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            with pytest.raises(PsycopgError) as update_err:
                cur.execute(
                    pgsql.SQL(
                        "UPDATE {tbl} SET tool_name = 'tampered' WHERE receipt_id = %s"
                    ).format(tbl=pgsql.Identifier(audit_sink._receipts_table)),
                    (receipt.receipt_id,),
                )
            assert "append-only" in str(update_err.value).lower()

            with pytest.raises(PsycopgError) as delete_err:
                cur.execute(
                    pgsql.SQL("DELETE FROM {tbl} WHERE receipt_id = %s").format(
                        tbl=pgsql.Identifier(audit_sink._receipts_table)
                    ),
                    (receipt.receipt_id,),
                )
            assert "append-only" in str(delete_err.value).lower()


def test_hitl_involved_generated_column_semantics(
    audit_sink: LakebaseAuditSink,
) -> None:
    """The GENERATED column is TRUE for HITL rows and FALSE (not NULL) for audit-only."""

    async def scenario() -> tuple[AuditReceipt, AuditReceipt]:
        audit_only = _build_receipt(thread_id="thread-D", hitl=False)
        await audit_sink.record(audit_only)

        hitl_row = _build_receipt(
            thread_id="thread-D", hitl=True, decision="approve"
        )
        await audit_sink.record(hitl_row)
        return audit_only, hitl_row

    audit_only, hitl_row = _run(scenario())

    with connect(**_pg_connection_kwargs()) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                pgsql.SQL(
                    "SELECT hitl_involved FROM {tbl} WHERE receipt_id = %s"
                ).format(tbl=pgsql.Identifier(audit_sink._receipts_table)),
                (audit_only.receipt_id,),
            )
            row = cur.fetchone()
            assert row is not None
            assert row[0] is False, (
                f"audit-only rows must land hitl_involved=FALSE, "
                f"got {row[0]!r}"
            )

            cur.execute(
                pgsql.SQL(
                    "SELECT hitl_involved FROM {tbl} WHERE receipt_id = %s"
                ).format(tbl=pgsql.Identifier(audit_sink._receipts_table)),
                (hitl_row.receipt_id,),
            )
            row = cur.fetchone()
            assert row is not None
            assert row[0] is True


def test_nonce_lifecycle_atomic_single_use(audit_sink: LakebaseAuditSink) -> None:
    """record_nonce then consume_nonce succeeds; second consume returns False.

    Exercises the v1.5-dormant nonce path so the DDL and atomic
    ``UPDATE ... RETURNING`` semantics stay validated even while the
    HITL enrichment path doesn't use them in v1.
    """

    async def scenario() -> tuple[bool, bool]:
        expires = datetime.now(timezone.utc) + timedelta(minutes=5)
        await audit_sink.record_nonce(
            nonce="ci-nonce-1",
            thread_id="thread-E",
            tool_call_id="call-nonce",
            expires_at=expires,
        )
        first = await audit_sink.consume_nonce(
            nonce="ci-nonce-1",
            thread_id="thread-E",
            tool_call_id="call-nonce",
        )
        second = await audit_sink.consume_nonce(
            nonce="ci-nonce-1",
            thread_id="thread-E",
            tool_call_id="call-nonce",
        )
        return first, second

    first, second = _run(scenario())
    assert first is True, "first consume must succeed"
    assert second is False, "second consume must fail (single-use)"


def test_expired_nonce_rejected(audit_sink: LakebaseAuditSink) -> None:
    """A nonce past its expiry cannot be consumed."""

    async def scenario() -> bool:
        expired = datetime.now(timezone.utc) - timedelta(seconds=1)
        await audit_sink.record_nonce(
            nonce="ci-nonce-expired",
            thread_id="thread-F",
            tool_call_id="call-nonce",
            expires_at=expired,
        )
        return await audit_sink.consume_nonce(
            nonce="ci-nonce-expired",
            thread_id="thread-F",
            tool_call_id="call-nonce",
        )

    assert _run(scenario()) is False
