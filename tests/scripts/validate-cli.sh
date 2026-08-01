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
dai workflow down -c "$CONFIG"
dai workflow up    -c "$CONFIG"
dai workflow down  -c "$CONFIG"

# ---------------------------------------------------------------------------
# Agent lifecycle on Apps (default mode): granular build → sync → start → down,
# then the idempotent one-shot up, then tear it down.
# ---------------------------------------------------------------------------
dai agent build -c "$CONFIG"
dai agent sync  -c "$CONFIG"
dai agent start -c "$CONFIG"
# `down` waits for the app to be fully deleted by default, so the following `up`
# never races the teardown (no `--wait` needed; pass `--no-wait` to skip).
dai agent down  -c "$CONFIG"
dai agent up    -c "$CONFIG"
dai monitor logs -c "$CONFIG"
dai agent down  -c "$CONFIG"

# ---------------------------------------------------------------------------
# Agent on Model Serving (-m ms).
# ---------------------------------------------------------------------------
dai agent up   -c "$CONFIG" -m ms
dai agent down -c "$CONFIG" -m ms

# ---------------------------------------------------------------------------
# Agent via the --direct SDK fast-path (no bundle on disk).
# ---------------------------------------------------------------------------
dai agent up   -c "$CONFIG" --direct
dai agent down -c "$CONFIG"

# Reaching here means every command above exited 0 (set -euo pipefail aborts on
# the first failure), so this banner is an unambiguous end-to-end success signal.
echo -e "\n\033[1;32m✅ validate-cli.sh: all commands completed successfully.\033[0m"
