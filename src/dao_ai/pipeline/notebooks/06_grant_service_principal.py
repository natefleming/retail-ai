# Databricks notebook source
# Dependency bootstrap. Install dao-ai (which pulls its own transitive deps) via
# uv — the newest bundled ../dist wheel in development mode, else the published
# PyPI package. In a deployed job the serverless environment has already
# installed it; this reinstall is harmless. ``%restart_python`` makes the freshly
# installed package importable in the cells below.
# No extras suffix: this notebook only calls core APIs. Notebooks that build the
# agent graph (07_deploy_agent, 09_run_evaluation) install ``[all]``;
# 01_ingest_and_transform installs ``[excel]``. The install spec is single-quoted
# in the magic so a dev wheel's ``+local`` version tag and any ``[extras]``
# survive shell glob/bracket expansion.
import glob, os

from packaging.version import Version

# Newest by *version*, not by filename: a lexical sort puts 0.2.8 above
# 0.2.10. ``Version`` also parses a dev wheel's ``+local`` tag correctly.
def _wheel_version(wheel: str) -> Version:
    return Version(os.path.basename(wheel).split("-")[1])

_wheels: list[str] = sorted(
    glob.glob("../dist/dao_ai-*.whl"), key=_wheel_version, reverse=True
)
_dao_ai_dep: str = _wheels[0] if _wheels else "dao-ai"

# MAGIC %uv pip install --quiet '{_dao_ai_dep}'
# MAGIC %restart_python

# COMMAND ----------

# Record the installed dao-ai version plus the key libraries resolved under
# it, so each run's logs capture exactly what was installed. Alphabetical; the
# list is short and hand-curated.
from importlib.metadata import version

print(f"dao-ai=={version('dao-ai')}")
print(f"databricks-sdk=={version('databricks-sdk')}")
print(f"mlflow=={version('mlflow')}")

# COMMAND ----------

dbutils.widgets.text(name="config-path", defaultValue="")

# There is no `../config` discovery fallback. That directory exists only in the
# staged bundle layout, and the bundle stages exactly one config — the same one
# the job passes here — so discovery could only ever guess, and guessing is how
# the wrong config gets loaded. `config-path` is the single input.
widget_path: str | None = dbutils.widgets.get("config-path") or None
if not widget_path:
    raise ValueError(
        "No config to grant against: the `config-path` widget is empty. In a "
        "staged pipeline bundle the config sits beside this notebook under "
        "`../config/` and the job always passes it; running this notebook by "
        "hand, set `config-path` to an absolute workspace path, for example "
        "`/Workspace/Users/you@example.com/dao-ai/examples/04_genie/genie_basic.yaml`."
    )

config_path: str = widget_path

print(config_path)

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

from dotenv import find_dotenv, load_dotenv

_loaded: bool = load_dotenv(find_dotenv())

# COMMAND ----------

# Serverless v5 FIPS: select psycopg's pure-Python impl before importing
# dao_ai.config, which transitively imports psycopg via databricks-langchain;
# the binary wheel's vendored OpenSSL aborts (SIGABRT) on import. Job-scoped.
import os

os.environ["PSYCOPG_IMPL"] = "python"

# COMMAND ----------

from dao_ai.config import AppConfig

config: AppConfig = AppConfig.from_file(path=config_path)

# COMMAND ----------

# Nothing declared, nothing to grant. `grant_all` is already safe for an
# undeclared config (it would warn per synthetic target and skip), but exiting
# here avoids a pointless workspace round-trip and says so plainly in the run log.
if not config.service_principals:
    print("No service_principals declared in the config — nothing to grant.")
    dbutils.notebook.exit("no service_principals declared")  # noqa: F821

# COMMAND ----------

# Grant each declared service principal the resources it owns.
#
# This is a SEPARATE task from 00_provision_service_principal on purpose. That one
# creates the identity and stores its credentials; this one authorizes it against
# resources that must already exist. Running both in one call — as
# ``provision_all(..., do_grant=True)`` used to — meant granting at the FRONT of
# the DAG, before the tables (01/02), the Lakebase project (03), the UC functions
# (04) and the Genie space (05) had been created, so every grant failed "absent",
# was warned, and never re-applied.
#
# It runs BEFORE 07_deploy_agent because the deployed agent needs its permissions
# at startup: granting afterwards means a live agent failing on its first tool
# call. The one grant that cannot precede the deploy — the serving endpoint — is
# best-effort by design and no-ops when the endpoint is absent.
#
# The WorkspaceClient is ambient: the job's identity. Granting needs MANAGE on the
# target, so if a grant reports "denied", that identity is what to grant.
from databricks.sdk import WorkspaceClient

from dao_ai.service_principal import (
    GRANT_FAILURE_ABSENT,
    GRANT_FAILURE_DENIED,
    Grant,
    GrantPlan,
    ServicePrincipalTarget,
    grant_all,
    resolve_sp_targets,
)

w: WorkspaceClient = WorkspaceClient()

targets: list[ServicePrincipalTarget] = resolve_sp_targets(config)
plans: list[GrantPlan] = grant_all(w, config=config, targets=targets)

# COMMAND ----------

# Report every grant, and tally the failures by kind. Deliberately does NOT raise:
# a missing privilege on one resource must not tear down a provisioning run whose
# infrastructure work already succeeded. The counts make a partial grant visible
# in the run output instead.
applied_total: int = 0
absent_total: int = 0
denied_total: int = 0
error_total: int = 0
skipped_total: int = 0

plan: GrantPlan
for plan in plans:
    print(f"\nservice principal '{plan.principal}'")
    if not plan.grants:
        print("  (no resources to grant)")
        continue

    grant_item: Grant
    for grant_item in plan.grants:
        label: str = f"[{grant_item.kind}] {grant_item.target}"
        if grant_item.applied is True:
            applied_total += 1
            print(f"  OK      {label}")
        elif grant_item.applied is False:
            if grant_item.failure_kind == GRANT_FAILURE_ABSENT:
                absent_total += 1
                detail = "target does not exist in this workspace"
            elif grant_item.failure_kind == GRANT_FAILURE_DENIED:
                denied_total += 1
                detail = "denied — the job identity lacks GRANT/MANAGE rights"
            else:
                error_total += 1
                detail = grant_item.error or "failed"
            print(f"  FAILED  {label}: {detail}")
        else:
            skipped_total += 1
            print(f"  SKIP    {label}: {grant_item.note or 'not attempted'}")

print(
    f"\ngrants: {applied_total} applied, {absent_total} absent, "
    f"{denied_total} denied, {error_total} error, {skipped_total} skipped"
)

# COMMAND ----------

# Hand the tally to downstream tasks (and the run output) rather than raising, so
# a partially-granted service principal is visible without blocking the deploy.
import json

dbutils.notebook.exit(  # noqa: F821
    json.dumps(
        {
            "applied": applied_total,
            "absent": absent_total,
            "denied": denied_total,
            "error": error_total,
            "skipped": skipped_total,
        }
    )
)
