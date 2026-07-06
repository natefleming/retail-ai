# Parallel Fan-Out — Live Endpoint Verification Runbook

This runbook walks through the required live-endpoint verification for the
parallel fan-out feature. All commands assume you are at the repo root
(`~/development/databricks/dao-ai`) with a working Databricks CLI auth.

## 0. Prerequisites

```bash
# Re-authenticate if the token is stale.
databricks auth login --profile DEFAULT

# Confirm auth works.
databricks current-user me
```

## 1. Static validation (already-green sanity check)

```bash
# Config must load and pass all cross-field validators.
uv run dao-ai validate -c config/examples/13_orchestration/parallel_fan_out_pattern.yaml

# Regenerate + inspect the schema so `agents` / `join` are visible.
uv run --quiet python -c "from dao_ai.config import AppConfig; import json; \
  print(json.dumps(AppConfig.model_json_schema()['\$defs']['HandoffRouteModel'], indent=2))"
```

Expect: `agents` (array of strings/AgentModel) and `join` (string/AgentModel)
properties on `HandoffRouteModel`, alongside `agent` and `is_deterministic`.

## 2. Full unit + integration sweep

```bash
uv run pytest -v --timeout=120 \
  tests/dao_ai/test_deterministic_handoffs.py \
  tests/dao_ai/test_deterministic_handoff_bridge.py \
  tests/dao_ai/test_handoff_constraints.py \
  tests/dao_ai/test_parallel_handoffs.py \
  tests/dao_ai/test_parallel_fan_out_integration.py \
  tests/dao_ai/test_swarm_multi_turn_sticky.py \
  tests/dao_ai/test_swarm_middleware.py
```

Expect: **113 passed** (no red).

## 3. Deploy the example to a serving endpoint

```bash
# Generates the DAB and deploys the model + endpoint.
uv run dao-ai deploy -c config/examples/13_orchestration/parallel_fan_out_pattern.yaml
```

Note the endpoint name emitted by `deploy` — you will need it in step 4.

## 4. Live inference — full fan-out case

```bash
# Fan-out prompt: mentions all three worker domains → LLM should invoke all
# three parallel handoff tools in one turn.
databricks serving-endpoints query <endpoint-name> \
  --json '{
    "messages": [
      {
        "role": "user",
        "content": "Is the DeWalt 20V drill in stock, what does it cost, and what is the return policy?"
      }
    ]
  }'
```

**Assertions:**

- ✅ Response is a synthesized answer from `synthesizer_agent` covering all
  three domains (price + stock + policy).
- ✅ Response does NOT contain raw worker outputs verbatim (proves the join
  actually reduced).

## 5. MLflow trace — confirm parallel execution

Open the MLflow experiment linked to the endpoint. Find the trace for the
request above. In the trace tree:

**Assertions:**

- ✅ `triage_agent` span appears first.
- ✅ `pricing_agent`, `inventory_agent`, `policy_agent` spans **overlap in
  wall-clock time** (siblings ran concurrently, not sequentially).
- ✅ `synthesizer_agent` span starts **after** all three sibling spans
  end (fan-in barrier is respected).
- ✅ Total wall time ≈ max(worker latencies) + triage + synthesizer, NOT
  sum(worker latencies).

Record the trace ID in the PR description.

## 6. Follow-up turn — confirm `active_agent` resume at join

```bash
# Same session (thread_id / conversation_id). The router must resume at the
# join, NOT restart at triage.
databricks serving-endpoints query <endpoint-name> \
  --json '{
    "messages": [
      { "role": "user", "content": "Is the DeWalt 20V drill in stock, what does it cost, and what is the return policy?" },
      { "role": "assistant", "content": "<synthesizer response from step 4>" },
      { "role": "user", "content": "Actually, do you have it in the yellow color?" }
    ]
  }'
```

**Assertions:**

- ✅ Follow-up trace shows `synthesizer_agent` as the ONLY agent that ran
  (no re-fan-out).
- ✅ Endpoint `active_agent` metadata (in the response's `custom_outputs`
  or in the trace's state snapshot) equals `synthesizer_agent`.

This is the piece that proves the sibling handler wrapper persisted
`active_agent = join` correctly across turns.

## 7. Degenerate case — LLM invokes only one sibling

Craft a narrow prompt so the LLM only reaches for one parallel handoff:

```bash
databricks serving-endpoints query <endpoint-name> \
  --json '{
    "messages": [
      { "role": "user", "content": "What is the return policy for tools?" }
    ]
  }'
```

**Assertions:**

- ✅ Only `policy_agent` runs (a single sibling).
- ✅ `synthesizer_agent` still runs exactly once and produces the final
  response (degenerate cohort behaves like a normal deterministic pipeline).

## What to include in the PR description

- Trace IDs from steps 5 and 6.
- Screenshot (or trace URL) of the fan-out step showing overlapping sibling
  spans.
- Confirmation that step 6's follow-up trace has only the synthesizer node.
- Confirmation that step 7's degenerate trace runs one worker + synthesizer.
