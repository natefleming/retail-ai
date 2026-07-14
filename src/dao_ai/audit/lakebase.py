"""
Lakebase-backed implementation of the audit receipt sink.

Reuses ``dao_ai.memory.postgres.AsyncPostgresPoolManager`` so audit writes
share the same connection pool infrastructure that the HITL checkpointer
already uses — no new provisioning, no new auth path.

The sink is created lazily by ``AuditSinkManager`` and dedup'd by
``(database_identity, table)`` so multiple audited tools referencing the
same YAML anchor share a single pool and a single hash-chain state.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from loguru import logger
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from dao_ai.audit.base import AuditReceipt
from dao_ai.audit.chain import HashChain
from dao_ai.audit.nonces import NonceIssuer

if TYPE_CHECKING:
    from dao_ai.config import AuditModel


_DDL_TEMPLATE: Optional[str] = None


def _load_ddl_template() -> str:
    """Load the DDL template once per process."""
    global _DDL_TEMPLATE
    if _DDL_TEMPLATE is None:
        from importlib.resources import files

        _DDL_TEMPLATE = (
            files("dao_ai.audit").joinpath("ddl.sql").read_text(encoding="utf-8")
        )
    return _DDL_TEMPLATE


# Column order shared between INSERT and _insert_params so they cannot drift.
_RECEIPT_COLUMNS: tuple[str, ...] = (
    "receipt_id",
    "schema_version",
    "receipt_kind",
    "thread_id",
    "agent_id",
    "mlflow_trace_id",
    "tool_call_id",
    "tool_name",
    "args_jcs",
    "args_hash",
    "args_hash_at_interrupt",
    "args_hash_at_resume",
    "edited_args_jcs",
    "edited_args_hash",
    "displayed_summary",
    "decision",
    "decision_detail",
    "approver_sub",
    "approver_email",
    "confirmed_via",
    "obo_access_token",
    "obo_token_exp",
    "obo_token_sub",
    "nonce",
    "nonce_exp",
    "execution_status",
    "execution_error",
    "recorded_at",
    "prev_hash",
    "this_hash",
)


class LakebaseAuditSink:
    """
    Records audit receipts + approval nonces to a Lakebase database.

    All I/O is async and safe to call from within the request-handling
    event loop. The sink lazily initialises its schema (idempotent DDL)
    on first write and reuses the shared psycopg async pool.
    """

    def __init__(self, config: "AuditModel") -> None:
        self._config: "AuditModel" = config
        self._receipts_table: str = config.table
        self._nonces_table: str = f"{config.table}_nonces"
        self._schema_ready: bool = False
        self._schema_lock: asyncio.Lock = asyncio.Lock()
        self.chain: HashChain = HashChain(self)
        self.nonces: NonceIssuer = NonceIssuer(
            self, ttl_seconds=config.nonce_ttl_seconds
        )

    # ---- Schema lifecycle -----------------------------------------------

    async def _pool(self) -> AsyncConnectionPool:
        from dao_ai.memory.postgres import AsyncPostgresPoolManager

        return await AsyncPostgresPoolManager.get_pool(self._config.database)

    async def ensure_schema(self) -> None:
        if self._schema_ready:
            return
        async with self._schema_lock:
            if self._schema_ready:
                return
            # Plain string.replace instead of string.Template so PL/pgSQL
            # dollar-quoting ($$ ... $$) is preserved verbatim — Template's
            # `$$` → `$` collapse would break the trigger body.
            ddl: str = (
                _load_ddl_template()
                .replace("${receipts_table}", self._receipts_table)
                .replace("${nonces_table}", self._nonces_table)
            )
            pool: AsyncConnectionPool = await self._pool()
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(ddl)
            logger.info(
                "Audit schema ensured",
                receipts_table=self._receipts_table,
                nonces_table=self._nonces_table,
            )
            self._schema_ready = True

    # ---- Receipt writes -------------------------------------------------

    async def record(self, receipt: AuditReceipt) -> None:
        """
        Persist ``receipt`` after sealing it into the hash chain.

        The caller is expected to populate every non-hash field; this method
        computes ``prev_hash`` / ``this_hash`` and issues the INSERT.
        """
        await self.ensure_schema()
        sealed: AuditReceipt = await self.chain.link_and_seal(receipt)
        pool: AsyncConnectionPool = await self._pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(self._insert_sql(), self._insert_params(sealed))
        logger.info(
            "Audit receipt recorded",
            receipt_id=sealed.receipt_id,
            thread_id=sealed.thread_id,
            tool_name=sealed.tool_name,
            receipt_kind=sealed.receipt_kind,
        )

    def _insert_sql(self) -> str:
        placeholders: str = ", ".join(["%s"] * len(_RECEIPT_COLUMNS))
        return (
            f"INSERT INTO {self._receipts_table} "
            f"({', '.join(_RECEIPT_COLUMNS)}) "
            f"VALUES ({placeholders})"
        )

    def _insert_params(self, r: AuditReceipt) -> tuple[object, ...]:
        decision_detail: Optional[str] = (
            json.dumps(r.decision_detail) if r.decision_detail is not None else None
        )
        return (
            r.receipt_id,
            r.schema_version,
            r.receipt_kind,
            r.thread_id,
            r.agent_id,
            r.mlflow_trace_id,
            r.tool_call_id,
            r.tool_name,
            r.args_jcs,
            r.args_hash,
            r.args_hash_at_interrupt,
            r.args_hash_at_resume,
            r.edited_args_jcs,
            r.edited_args_hash,
            r.displayed_summary,
            r.decision,
            decision_detail,
            r.approver_sub,
            r.approver_email,
            r.confirmed_via,
            r.obo_access_token,
            r.obo_token_exp,
            r.obo_token_sub,
            r.nonce,
            r.nonce_exp,
            r.execution_status,
            r.execution_error,
            r.recorded_at,
            r.prev_hash,
            r.this_hash,
        )

    # ---- Hash-chain support (called by HashChain) -----------------------

    async def head_hash(self, thread_id: str) -> Optional[str]:
        """Return the ``this_hash`` of the most recent receipt for ``thread_id``."""
        await self.ensure_schema()
        pool: AsyncConnectionPool = await self._pool()
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    f"SELECT this_hash FROM {self._receipts_table} "
                    f"WHERE thread_id = %s "
                    f"ORDER BY recorded_at DESC LIMIT 1",
                    (thread_id,),
                )
                row: Optional[dict[str, str]] = await cur.fetchone()
        if row is None:
            return None
        return row["this_hash"]

    # ---- Nonces (called by NonceIssuer) --------------------------------

    async def record_nonce(
        self,
        *,
        nonce: str,
        thread_id: str,
        tool_call_id: str,
        expires_at: datetime,
    ) -> None:
        await self.ensure_schema()
        pool: AsyncConnectionPool = await self._pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    f"INSERT INTO {self._nonces_table} "
                    f"(nonce, thread_id, tool_call_id, expires_at) "
                    f"VALUES (%s, %s, %s, %s)",
                    (nonce, thread_id, tool_call_id, expires_at),
                )

    async def consume_nonce(
        self,
        *,
        nonce: str,
        thread_id: str,
        tool_call_id: str,
    ) -> bool:
        """
        Atomically mark ``nonce`` as used. Returns True on success.

        Returns False when the nonce is missing, expired, already used, or
        bound to a different ``(thread_id, tool_call_id)`` pair — the caller
        (NonceIssuer.consume) raises AuditNonceError on False.
        """
        await self.ensure_schema()
        pool: AsyncConnectionPool = await self._pool()
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    f"UPDATE {self._nonces_table} "
                    f"SET used_at = NOW() "
                    f"WHERE nonce = %s "
                    f"  AND thread_id = %s "
                    f"  AND tool_call_id = %s "
                    f"  AND used_at IS NULL "
                    f"  AND expires_at > NOW() "
                    f"RETURNING nonce",
                    (nonce, thread_id, tool_call_id),
                )
                row: Optional[dict[str, str]] = await cur.fetchone()
        return row is not None
