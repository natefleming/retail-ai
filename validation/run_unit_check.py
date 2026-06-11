"""Run scope-generation checks for every scenario.

For each scenario, exercise both surfaces:
- ``generate_user_api_scopes`` (Apps path)
- ``build_auth_policy`` (Model Serving path) — must produce the same scope set

Confirms the unified emission contract.

Run: ``python validation/run_unit_check.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dao_ai.apps.resources import generate_user_api_scopes
from dao_ai.providers.databricks import build_auth_policy

from scenarios import SCENARIOS  # noqa: E402


def main() -> int:
    failures: list[str] = []
    for sc in SCENARIOS:
        config = sc.build()

        apps_scopes = set(generate_user_api_scopes(config))
        ms_scopes = set(build_auth_policy(config).user_auth_policy.api_scopes)

        # Both surfaces must produce the same set.
        if apps_scopes != ms_scopes:
            failures.append(
                f"{sc.name}: Apps and MS scope sets diverge\n"
                f"  Apps: {sorted(apps_scopes)}\n"
                f"  MS:   {sorted(ms_scopes)}\n"
            )
            continue

        missing = sc.expected_scopes - apps_scopes
        if missing:
            failures.append(
                f"{sc.name}: missing expected scopes {sorted(missing)} "
                f"(got {sorted(apps_scopes)})\n"
            )
            continue

        leaked = sc.forbidden_scopes & apps_scopes
        if leaked:
            failures.append(
                f"{sc.name}: forbidden scopes present {sorted(leaked)} "
                f"(got {sorted(apps_scopes)})\n"
            )
            continue

        flag = "[+]" if sc.positive else "[-]"
        print(f"{flag} {sc.name:<32}  → {sorted(apps_scopes)}")

    print()
    if failures:
        print("=== FAILURES ===")
        for f in failures:
            print(f)
        return 1
    print(f"=== ALL {len(SCENARIOS)} SCENARIOS PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
