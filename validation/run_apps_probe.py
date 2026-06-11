"""Probe the Databricks Apps platform with each scenario's expected scope set.

Sends ``PATCH /api/2.0/apps/<probe>`` for every scenario using its
``expected_scopes`` payload. A response containing ``user_api_scopes`` means
the platform accepts the full set; an error response means at least one
string is rejected.

This complements ``run_unit_check.py`` (which proves the generator output)
by proving the Apps API accepts what the generator emits.

Run: ``python validation/run_apps_probe.py --profile fevm --app scope-probe-nf``
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from scenarios import SCENARIOS  # noqa: E402


def update_app(profile: str, app: str, scopes: list[str]) -> tuple[bool, list[str], str]:
    body = json.dumps({"name": app, "user_api_scopes": scopes})
    result = subprocess.run(
        ["databricks", "apps", "update", app, "--profile", profile, "--json", body],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False, [], result.stderr.strip() or result.stdout.strip()
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return False, [], "non-json response: " + result.stdout[:300]
    actual = data.get("user_api_scopes") or []
    return True, actual, ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="fevm")
    ap.add_argument("--app", default="scope-probe-nf")
    args = ap.parse_args()

    failures: list[str] = []
    for sc in SCENARIOS:
        scopes = sorted(sc.expected_scopes)
        if not scopes:
            print(f"-   {sc.name:<32}  (empty scope set — skipping platform probe)")
            continue
        ok, actual, err = update_app(args.profile, args.app, scopes)
        if not ok:
            failures.append(f"{sc.name}: REJECTED — {err}")
            print(f"[!] {sc.name:<32}  REJECTED  {err[:150]}")
            continue
        # Platform should echo back exactly what we sent (set-equal).
        if set(actual) != set(scopes):
            failures.append(
                f"{sc.name}: SET DIVERGES — sent {scopes}, got {actual}"
            )
            print(
                f"[≠] {sc.name:<32}  DIVERGES  sent={scopes} got={actual}"
            )
            continue
        print(f"[✓] {sc.name:<32}  accepted ({len(scopes)} scope(s))")

    print()
    if failures:
        print("=== FAILURES ===")
        for f in failures:
            print(f)
        return 1
    print("=== ALL SCENARIOS' SCOPE SETS ACCEPTED BY APPS PLATFORM ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
