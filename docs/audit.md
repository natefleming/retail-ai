# Auditable Tool Invocations

dao-ai can record tamper-evident audit receipts for every invocation of a
sensitive tool — with or without human-in-the-loop approval. Presence of an
`audit:` block on a tool's `function` opts that tool in; absence leaves the
runtime bit-for-bit unchanged.

The feature was designed against the intent-verification research captured in
the vault (`00-inbox/dao-ai-auditable-hitl-approvals-research-2026-07-12.md`)
and satisfies the "who approved this exact thing" bar for SOC2 / SOX / HIPAA
non-repudiation work.

## When to reach for it

| Scenario                                     | Configuration                                |
|----------------------------------------------|----------------------------------------------|
| Tool executes autonomously, needs audit log  | `audit:` only                                |
| Tool needs approval, no audit trail          | `human_in_the_loop:` only                    |
| Tool needs approval AND non-repudiation      | Both `audit:` and `human_in_the_loop:`       |

Auditing is per-tool. Tools without `audit:` see no behavioural change even
when the agent hosts other audited tools.

## Config

```yaml
resources:
  databases:
    audit_db: &audit_db
      project: retail-consumer-goods
      autoscaling_min_cu: 0

tools:
  refund_customer:
    name: refund_customer
    function:
      type: python
      name: my_package.tools.refund_customer
      audit:                              # <-- opt this tool into auditing
        database: *audit_db               # anchor reuse (same Lakebase as checkpointer)
        table: audit_receipts             # default; override if you host multiple audits
        # nonce_ttl_seconds: 300          # default; 30-3600 accepted
      human_in_the_loop:                  # <-- optional: also gate on approval
        review_prompt: "Refund of ${amount} to customer ${customer_id}?"
        allowed_decisions: [approve, edit, reject]
```

For multiple audited tools, define the block once as a YAML anchor on the
first tool and reference from later tools:

```yaml
- function:
    name: refund_customer
    audit: &audit
      database: *audit_db
      table: audit_receipts
- function:
    name: cancel_subscription
    audit: *audit
```

## What lands in Lakebase

Every audited invocation writes exactly one row to the configured table with
these columns (nullable HITL fields land NULL for audit-only tools):

| Column                    | Notes                                                                 |
|---------------------------|-----------------------------------------------------------------------|
| `receipt_id`              | UUID.                                                                 |
| `receipt_kind`            | `execution` \| `approval` \| `rejection`.                             |
| `thread_id`, `agent_id`   | LangGraph thread + agent identifiers.                                 |
| `mlflow_trace_id`         | Links back to the MLflow trace for the same turn.                     |
| `tool_call_id`, `tool_name`| From the LangGraph tool call.                                        |
| `args_jcs`, `args_hash`   | RFC 8785 canonical JSON of args + hex SHA-256 hash.                   |
| `args_hash_at_interrupt` / `_at_resume` | Populated on HITL tools — must match; drift is fail-closed. |
| `displayed_summary`       | Harness-generated review text shown to the reviewer (never model-generated). |
| `decision`, `decision_detail` | HITL decision + payload.                                          |
| `approver_sub`, `approver_email` | Identity fields from `X-Forwarded-User` / `X-Forwarded-Email`. |
| `confirmed_via`           | `chat_ui` (v1) — reserved for `obo_jwt` (v2) and `webauthn` (v3).     |
| `obo_access_token`, `obo_token_exp`, `obo_token_sub` | Raw OBO JWT + extracted claims. Only populated when the header is present. |
| `nonce`, `nonce_exp`      | Server-issued single-use approval nonce.                              |
| `execution_status`, `execution_error` | `ok` / `error` / `args_mismatch` / `not_executed_rejected`. |
| `prev_hash`, `this_hash`  | Hash-chain link over the receipts for this `thread_id`.               |
| `recorded_at`             | Server timestamp (UTC).                                               |

Two side tables live alongside it:

- `<table>_nonces` — approval nonces, atomic single-use via `UPDATE ...
  WHERE used_at IS NULL RETURNING`.
- The receipts table has `BEFORE UPDATE`/`BEFORE DELETE` triggers that refuse
  mutation — append-only at the SQL layer.

## What lands on MLflow spans

For every audited invocation the outer agent span picks up these attributes
(via `set_attribute`, so they are immutable — receipts are the source of
truth for anything sensitive):

- `dao_ai.audit.receipt_id`
- `dao_ai.audit.args_hash`
- `dao_ai.audit.obo_token_present` (bool)
- `dao_ai.audit.approver_sub` (HITL only)
- `dao_ai.audit.decision` (HITL only)
- Span event `dao_ai.audit.args_mismatch` on fail-closed rejections.

The raw OBO JWT is **never** attached to spans — traces have broader read
access than the receipts table.

## Sensitive data handling

The OBO access token is a live, short-lived JWT (~1h). Because it is
cryptographically verifiable against Databricks JWKS for weeks after
issuance, storing it makes independent post-hoc re-verification possible.

The receipts table is designed to be governance-scoped:

- Create a Unity Catalog group named `hitl_audit_reviewers` (or your own
  equivalent) and grant `SELECT` on the receipts table only to that group.
- Never grant public / catalog-wide SELECT.
- Consider a v1.5 purge job that NULLs `obo_access_token` after 30 days
  while retaining `obo_token_sub` / `obo_token_exp` for attribution.

## Deployment: Databricks Apps

Databricks Apps forwards user identity on `X-Forwarded-User` /
`X-Forwarded-Email` / `X-Forwarded-Access-Token`. The audit middleware
picks these up automatically. Deployment flow:

```bash
uv run dao-ai generate-bundle --force --development -c my_config.yaml
databricks bundle deploy -p <profile>
# Critical: link the trace destination BEFORE bundle run.
uv run dao-ai link-trace-destination -c my_config.yaml -p <profile>
databricks bundle run <app_name> -p <profile>
```

The `link-trace-destination` step provisions the OTEL span tables in the
configured `trace_location` schema and links the experiment to them. Without
it, span attributes still emit but the OTEL tables are missing on cold-start
tables and traces do not surface in the Databricks Trace UI.

## Deployment: Model Serving

Model Serving handles trace linking + SP experiment `CAN_EDIT` inside
`agents.deploy()`. No separate CLI step is required. The audit table's SP
identity is the one configured on the `DatabaseModel` — do not mix with OBO
for long-running audit writes.

Model Serving requests do not carry `X-Forwarded-*` headers in the classical
sense; OBO on MS is captured via `mlflow.models.get_request_headers()`.
Callers that don't forward a token land NULL in `obo_access_token` — the
receipt still records `approver_sub` from context.

## Guarantees

- **Args-hash binding.** For HITL-audited tools, `args_hash` is recomputed
  at execution time and byte-compared against the value stashed at
  interrupt time. Drift → hard-reject, rejection receipt with
  `execution_status=args_mismatch`, no tool execution.
- **Fail-closed on integrity failures.** Args mismatch, nonce reuse, and
  expired nonces raise; no configuration knob turns them off.
- **Fail-open on sink I/O.** If a receipt write fails (Lakebase down,
  network partition), the tool call itself proceeds and a WARN is
  emitted. Integrity is preserved; observability is best-effort.
- **Append-only at the SQL layer.** `BEFORE UPDATE`/`BEFORE DELETE`
  triggers on the receipts table refuse mutation. Superuser DDL can still
  truncate — v1.5 WORM anchor closes that residual gap.

## Known limitations (v1)

- **Approval confirm renders in the same chat surface.** The
  `displayed_summary` is harness-generated (not model-generated), which
  is the achievable subset in v1. Full out-of-band confirmation via
  WebAuthn/passkey lands in v3.
- **Process-local approval stash.** The interrupt-time args_hash + nonce
  live in an in-process cache (`AuditStash`), which does not survive a
  process restart during pause. A restart-then-resume writes an
  execution receipt but leaves HITL fields NULL. Cross-process approval
  persistence is a v1.5 improvement.
- **OBO JWT is stored raw.** Signature verification (v2) and semantic-
  diff LLM-judge for `edit` decisions (also v2+) are follow-ons.

## References

- Vault: `00-inbox/dao-ai-auditable-hitl-approvals-research-2026-07-12.md`
- Config: `config/examples/07_human_in_the_loop/human_in_the_loop_audited.yaml`
- Source: `src/dao_ai/audit/`, `src/dao_ai/middleware/audit_receipt.py`,
  `src/dao_ai/middleware/audit_hitl.py`
- Tests: `tests/dao_ai/audit/`
