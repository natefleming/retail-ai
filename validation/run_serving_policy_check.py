"""Validate the Model Serving side of OBO scope emission.

For each scenario:
1. Build the :class:`AppConfig`.
2. Run ``build_auth_policy(config)`` — the same function MLflow calls when
   ``log_model(auth_policy=...)`` packages an agent for Model Serving.
3. Inspect the resulting ``UserAuthPolicy.api_scopes`` (the field that
   becomes the deployed endpoint's user-scope claim list).
4. Confirm the strings match what Apps just accepted (cross-surface parity).

Run: ``python validation/run_serving_policy_check.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from dao_ai.providers.databricks import build_auth_policy  # noqa: E402
from mlflow.models.auth_policy import AuthPolicy, UserAuthPolicy  # noqa: E402

from scenarios import SCENARIOS  # noqa: E402


def main() -> int:
    failures: list[str] = []
    for sc in SCENARIOS:
        config = sc.build()
        policy = build_auth_policy(config)

        # Structural checks: policy is the right MLflow type.
        if not isinstance(policy, AuthPolicy):
            failures.append(f"{sc.name}: build_auth_policy did not return AuthPolicy")
            continue
        if not isinstance(policy.user_auth_policy, UserAuthPolicy):
            failures.append(f"{sc.name}: user_auth_policy is not UserAuthPolicy")
            continue

        # Contract check: api_scopes is a list of strings, set-equal to
        # the scenario's expected_scopes (plus the dynamic catalog.*:read
        # auto-additions, already encoded in expected_scopes).
        emitted = set(policy.user_auth_policy.api_scopes)
        missing = sc.expected_scopes - emitted
        leaked = sc.forbidden_scopes & emitted
        if missing or leaked:
            failures.append(
                f"{sc.name}: missing={sorted(missing)} leaked={sorted(leaked)}"
            )
            continue

        flag = "[+]" if sc.positive else "[-]"
        print(
            f"{flag} {sc.name:<32}  UserAuthPolicy.api_scopes = "
            f"{sorted(emitted)}"
        )

    print()
    if failures:
        print("=== FAILURES ===")
        for f in failures:
            print(f)
        return 1
    print(
        f"=== ALL {len(SCENARIOS)} SCENARIOS PRODUCE THE EXPECTED "
        f"UserAuthPolicy (Model Serving path) ==="
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
