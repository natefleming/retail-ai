#!/usr/bin/env bash
#
# validate-cli.sh — run a dao-ai config through the common CLI lifecycle a
# user hits during a manual validation session.
#
# Drives the real `dao-ai` CLI against a live Databricks workspace, so it needs
# a profile with the workspace already provisioned (VS / Lakebase / Genie, etc).
# The command sequence is hard-coded; the only inputs are the profile and an
# optional config path.
#
# Usage:
#   tests/scripts/validate-cli.sh -p fevm
#   tests/scripts/validate-cli.sh -p fevm ./examples/04_genie/genie.yaml
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Inputs: -p/--profile (required), optional config path (positional).
# ---------------------------------------------------------------------------
PROFILE=""
CONFIG="./examples/01_getting_started/ai_gateway.yaml"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--profile) PROFILE="$2"; shift 2 ;;
    -h|--help)    grep '^#' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)            CONFIG="$1"; shift ;;
  esac
done

if [[ -z "$PROFILE" ]]; then
  echo "ERROR: -p/--profile is required (e.g. -p fevm)." >&2
  exit 1
fi

# Echo each command before running it.
dai() {
  printf '\n\033[1;34m$ dao-ai -p %s %s\033[0m\n' "$PROFILE" "$*"
  uv run dao-ai -p "$PROFILE" "$@"
}

# Send ONE real inference request to the just-deployed resource and require a
# valid response. Run this immediately after `up --wait` so it doubly proves the
# readiness gate: `up --wait` only returns once the App/endpoint is servable, so
# this request must succeed straight away (no retry/sleep). `kind` is "app"
# (apps/mcp) or "endpoint" (model_serving).
#
# dao-ai agents ALWAYS speak the MLflow Responses contract (`{"input": [...]}`)
# on `/invocations` — identical request/response body for Apps and Model Serving
# (see docs/background_agents.md); only the URL differs:
#   Apps:          POST <app.url>/invocations              (behind the Apps proxy)
#   Model Serving: POST <host>/serving-endpoints/<ep>/invocations
infer() {
  local kind="$1"
  printf '\n\033[1;35m# immediate inference check (%s) against %s\033[0m\n' \
    "$kind" "$CONFIG"
  DAO_AI_INFER_KIND="$kind" DAO_AI_INFER_CONFIG="$CONFIG" \
    DATABRICKS_CONFIG_PROFILE="$PROFILE" uv run python - <<'PY'
import os
import sys

import httpx
from databricks.sdk import WorkspaceClient

from dao_ai.config import AppConfig

kind = os.environ["DAO_AI_INFER_KIND"]
config = AppConfig.from_file(os.environ["DAO_AI_INFER_CONFIG"])
w = WorkspaceClient()
host = (w.config.host or "").rstrip("/")

if kind == "app":
    url = w.apps.get(name=config.app.app_resource_name).url.rstrip("/") + "/invocations"
else:  # endpoint (model_serving)
    url = f"{host}/serving-endpoints/{config.app.endpoint_name}/invocations"

# Same Responses-contract body for both surfaces.
headers = w.config.authenticate() or {}
headers["Content-Type"] = "application/json"
resp = httpx.post(
    url,
    headers=headers,
    json={"input": [{"role": "user", "content": "Reply with a short greeting."}]},
    timeout=120.0,
)
resp.raise_for_status()
body = resp.text

if not body.strip():
    print("Inference check FAILED: empty response", file=sys.stderr)
    sys.exit(1)
print(f"  inference OK — response: {body[:200]}")
PY
}

# ---------------------------------------------------------------------------
# Sanity: version, resolved env, config validity, declared params, MCP tools.
# ---------------------------------------------------------------------------
dai version
dai doctor
dai validate -c "$CONFIG"
dai parameters -c "$CONFIG"
dai mcp tools -c "$CONFIG"

# ---------------------------------------------------------------------------
# Provisioning workflow lifecycle (granular build → sync → start, then the
# idempotent one-shot up), then tear it down.
# ---------------------------------------------------------------------------
dai workflow build -c "$CONFIG"
dai workflow sync  -c "$CONFIG"
dai workflow start -c "$CONFIG"
# --wait: block until the deployed App/endpoint is fully deleted so the next
# deploy (the `up` below, and the agent section) can't race the async teardown.
dai workflow down -c "$CONFIG" --wait
# `up --wait`: block until the workflow-deployed App is READY to serve, then send
# a real inference request immediately to prove it.
dai workflow up   -c "$CONFIG" --wait
infer app
# Last workflow down before the agent section redeploys the same app — wait it out.
dai workflow down  -c "$CONFIG" --wait

# ---------------------------------------------------------------------------
# Agent lifecycle on Apps (default mode): granular build → sync → start → down,
# then the idempotent one-shot up, then tear it down.
# ---------------------------------------------------------------------------
dai agent build -c "$CONFIG"
dai agent sync  -c "$CONFIG"
dai agent start -c "$CONFIG"
# --wait: block until the app is fully deleted so the following `up` can't race
# the async teardown (400 "compute is in DELETING state"). Omit for fire-and-forget.
dai agent down  -c "$CONFIG" --wait
# `up --wait`: block until the App is READY (compute ACTIVE + GET /health 200),
# then send a real inference request immediately to prove it is servable.
dai agent up    -c "$CONFIG" --wait
infer app
dai monitor logs -c "$CONFIG"
dai agent down  -c "$CONFIG"

# ---------------------------------------------------------------------------
# Agent on Model Serving (-m ms). `up --wait` blocks until the endpoint is READY
# + its served model is DEPLOYMENT_READY; then query it immediately. Model
# Serving cold-start (container build + model load) routinely exceeds the 600s
# default, so give it a generous 1800s — live-observed ~12min+ on fevm.
# ---------------------------------------------------------------------------
dai agent up   -c "$CONFIG" -m ms --wait 1800
infer endpoint
dai agent down -c "$CONFIG" -m ms

# ---------------------------------------------------------------------------
# Agent via the --direct SDK fast-path (no bundle on disk). `up --wait` here too,
# then an immediate inference request against the directly-deployed App.
# ---------------------------------------------------------------------------
dai agent up   -c "$CONFIG" --direct --wait
infer app
dai agent down -c "$CONFIG"

# Reaching here means every command above exited 0 (set -euo pipefail aborts on
# the first failure), so this banner is an unambiguous end-to-end success signal.
echo -e "\n\033[1;32m✅ validate-cli.sh: all commands completed successfully.\033[0m"
