# Databricks notebook source
# Dependency bootstrap. Install dao-ai (which pulls its own transitive deps) via
# uv — the newest bundled ../dist wheel in development mode, else the published
# PyPI package. In a deployed job the serverless environment has already
# installed it; this reinstall is harmless. ``%restart_python`` makes the freshly
# installed package importable in the cells below.
# No extras suffix: this provisioning notebook only calls core APIs. Notebooks
# that build the agent graph (06_deploy_agent, 08_run_evaluation) install
# ``[all]``; 01_ingest_and_transform installs ``[excel]``. The install spec is
# single-quoted in the magic so a dev wheel's ``+local`` version tag and any
# ``[extras]`` survive shell glob/bracket expansion.
import glob

_dao_ai_dep = next(
    iter(sorted(glob.glob("../dist/dao_ai-*.whl"), reverse=True)), "dao-ai"
)

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

import os
from typing import Sequence


def find_yaml_files_os_walk(base_path: str) -> Sequence[str]:
    # Tolerate a missing/non-dir base path: when the pipeline runs from a
    # wheel-only bundle an explicit `config-path` is always supplied, so the
    # `../config` discovery dropdown is optional. Return [] instead of raising.
    if not os.path.isdir(base_path):
        return []

    yaml_files = []

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith((".yaml", ".yml")):
                yaml_files.append(os.path.join(root, file))

    return sorted(yaml_files)


# COMMAND ----------

dbutils.widgets.text(name="config-path", defaultValue="")

config_files: Sequence[str] = find_yaml_files_os_walk("../config")
dbutils.widgets.dropdown(
    name="config-paths", choices=config_files, defaultValue=next(iter(config_files), "")
)

# ``overwrite`` is deliberately a widget rather than always-on. This task runs on
# every pipeline execution, so rotating credentials by default would replace the
# secret a deployed agent is currently authenticating with, on every deploy.
dbutils.widgets.dropdown(
    name="overwrite", choices=["false", "true"], defaultValue="false"
)

config_path: str | None = dbutils.widgets.get("config-path") or None
project_path: str = dbutils.widgets.get("config-paths") or None

config_path: str = config_path or project_path
overwrite: bool = dbutils.widgets.get("overwrite") == "true"

print(config_path)

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

# COMMAND ----------

from dao_ai.config import AppConfig

config: AppConfig = AppConfig.from_file(path=config_path)

# COMMAND ----------

# Provision every service principal the config declares, granting each one the
# resources it owns. Same library entry point the ``dao-ai sp provision`` CLI
# calls — the CLI is one provisioning surface, this is another.
#
# The WorkspaceClient is ambient: the job's identity. Creating service principals
# and writing secret scopes need more privilege than reading a table, so if this
# task fails with a permissions error, that identity is what to grant.
from databricks.sdk import WorkspaceClient

from dao_ai.service_principal import provision_all, resolve_sp_targets

w: WorkspaceClient = WorkspaceClient()

targets = resolve_sp_targets(config)
result = provision_all(
    w,
    config=config,
    targets=targets,
    overwrite=overwrite,
)

for provisioned in result.results:
    verb = "reused" if provisioned.reused else "created"
    print(f"\nservice principal '{provisioned.name}' — {provisioned.display_name}")
    print(f"  {verb}: client_id={provisioned.client_id or '(none)'}")
    if provisioned.blocked_reason:
        print(f"  BLOCKED: {provisioned.blocked_reason}")
        continue
    if provisioned.secret_action:
        print(
            f"  secrets: {provisioned.secret_action} "
            f"(scope {provisioned.stored_scope})"
        )
    if provisioned.existing_keys:
        print(f"  already populated: {', '.join(provisioned.existing_keys)}")
    plan = provisioned.grant_plan
    if plan is not None:
        applied = sum(1 for g in plan.grants if g.applied is True)
        print(f"  grants: {applied}/{len(plan.grants)} applied")
        for g in plan.grants:
            if g.applied is False:
                print(f"    FAILED [{g.kind}] {g.target}: {g.error}")
            elif g.note:
                print(f"    SKIP [{g.kind}] {g.target}: {g.note}")

# COMMAND ----------

# Fail the task when a service principal could not be provisioned, so the
# downstream tasks that depend on it (provision-lakebase needs the client_id as
# its Postgres role subject; unity-catalog-tools creates functions the SP needs
# EXECUTE on) do not run against a half-provisioned identity.
if result.blocked:
    raise RuntimeError(
        "Service-principal provisioning was blocked for: "
        + "; ".join(f"{name}: {reason}" for name, reason in result.blocked)
    )
