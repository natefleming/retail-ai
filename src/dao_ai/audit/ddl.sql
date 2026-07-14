-- Idempotent DDL for the dao-ai audit subsystem.
-- Applied by LakebaseAuditSink.ensure_schema() on first use.
--
-- Placeholders (substituted server-side, not string-interpolated for values):
--   ${receipts_table}  — audit receipts table
--   ${nonces_table}    — approval nonce table

CREATE TABLE IF NOT EXISTS ${receipts_table} (
    receipt_id                TEXT PRIMARY KEY,
    schema_version            INTEGER NOT NULL DEFAULT 1,
    receipt_kind              TEXT NOT NULL,

    thread_id                 TEXT NOT NULL,
    agent_id                  TEXT,
    mlflow_trace_id           TEXT,
    tool_call_id              TEXT,
    tool_name                 TEXT NOT NULL,

    args_jcs                  TEXT NOT NULL,
    args_hash                 TEXT NOT NULL,
    args_hash_at_interrupt    TEXT,
    args_hash_at_resume       TEXT,
    edited_args_jcs           TEXT,
    edited_args_hash          TEXT,

    displayed_summary         TEXT,
    decision                  TEXT,
    decision_detail           JSONB,

    approver_sub              TEXT,
    approver_email            TEXT,
    confirmed_via             TEXT,

    obo_access_token          TEXT,
    obo_token_exp             TIMESTAMPTZ,
    obo_token_sub             TEXT,

    nonce                     TEXT,
    nonce_exp                 TIMESTAMPTZ,

    execution_status          TEXT,
    execution_error           TEXT,

    recorded_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    prev_hash                 TEXT,
    this_hash                 TEXT NOT NULL,

    -- Generated first-class HITL marker. Kept as a GENERATED ALWAYS ...
    -- STORED column so it's always in sync with its source fields, is
    -- index-friendly, and can be filtered on directly in SQL / BI tools
    -- without recomputing the predicate every query. The rule mirrors
    -- ``dao_ai.tools.audit_query._row_had_hitl``: any single HITL
    -- signal is sufficient.
    hitl_involved             BOOLEAN GENERATED ALWAYS AS (
        args_hash_at_interrupt IS NOT NULL
        OR receipt_kind = 'rejection'
        OR decision IN ('approve', 'edit', 'reject', 'respond')
    ) STORED
);

CREATE INDEX IF NOT EXISTS ${receipts_table}_hitl_involved_idx
    ON ${receipts_table} (hitl_involved, recorded_at);

CREATE INDEX IF NOT EXISTS ${receipts_table}_thread_recorded_idx
    ON ${receipts_table} (thread_id, recorded_at);

CREATE INDEX IF NOT EXISTS ${receipts_table}_tool_call_idx
    ON ${receipts_table} (tool_call_id)
    WHERE tool_call_id IS NOT NULL;

-- Append-only enforcement. Any UPDATE or DELETE against the receipts table
-- raises an exception. Truncation is still possible via superuser DDL; the
-- v1.5 WORM anchor closes that residual gap.
CREATE OR REPLACE FUNCTION ${receipts_table}_reject_mutation()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'audit receipts are append-only (attempted %)', TG_OP;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS ${receipts_table}_no_update ON ${receipts_table};
CREATE TRIGGER ${receipts_table}_no_update
    BEFORE UPDATE ON ${receipts_table}
    FOR EACH ROW EXECUTE FUNCTION ${receipts_table}_reject_mutation();

DROP TRIGGER IF EXISTS ${receipts_table}_no_delete ON ${receipts_table};
CREATE TRIGGER ${receipts_table}_no_delete
    BEFORE DELETE ON ${receipts_table}
    FOR EACH ROW EXECUTE FUNCTION ${receipts_table}_reject_mutation();

-- Approval nonces. Single-use enforced by UPDATE ... WHERE used_at IS NULL
-- RETURNING. Not append-only — the row is updated exactly once from
-- issued -> consumed.
CREATE TABLE IF NOT EXISTS ${nonces_table} (
    nonce           TEXT PRIMARY KEY,
    thread_id       TEXT NOT NULL,
    tool_call_id    TEXT NOT NULL,
    issued_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at      TIMESTAMPTZ NOT NULL,
    used_at         TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS ${nonces_table}_thread_call_idx
    ON ${nonces_table} (thread_id, tool_call_id);
