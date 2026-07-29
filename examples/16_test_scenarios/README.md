# 16 - Test Scenarios (internal)

> **Not a feature example.** These configs are internal test harnesses used to
> validate dao-ai deploy/provisioning behavior, not part of the numbered
> learning path. They're intentionally left out of the top-level examples
> Directory Guide.

## Contents

| File | Purpose |
|------|---------|
| [`genie_provisioning_only.yaml`](./genie_provisioning_only.yaml) | Minimal config that validates the Genie-space provisioning cascade (`from_space_id → from_name → room.create()`) under a real bundle deploy, including the "reuse on second deploy" path. See the header comment in the file for the exact run/cleanup steps. |

## Usage

Each file documents its own run procedure in a header comment. In general:

```bash
uv run dao-ai workflow up \
  --config examples/16_test_scenarios/genie_provisioning_only.yaml \
  --profile <profile> --mode model_serving
```

Run it a second time to exercise the reuse-on-redeploy path; clean up the
provisioned Genie space afterward as noted in the file.
