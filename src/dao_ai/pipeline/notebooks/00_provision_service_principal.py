# Databricks notebook source
# Dependency bootstrap. Install dao-ai (which pulls its own transitive deps) via
# uv — the newest bundled ../dist wheel in development mode, else the published
# PyPI package. In a deployed job the serverless environment has already
# installed it; this reinstall is harmless. ``%restart_python`` makes the freshly
# installed package importable in the cells below.
# No extras suffix: this provisioning notebook only calls core APIs. Notebooks
# that build the agent graph (07_deploy_agent, 09_run_evaluation) install
# ``[all]``; 01_ingest_and_transform installs ``[excel]``. The install spec is
# single-quoted in the magic so a dev wheel's ``+local`` version tag and any
# ``[extras]`` survive shell glob/bracket expansion.
import glob

_dao_ai_dep: str = next(
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

from dao_ai.utils import find_config_files

# COMMAND ----------

dbutils.widgets.text(name="config-path", defaultValue="")

config_files: list[str] = find_config_files("../config")
dbutils.widgets.dropdown(
    name="config-paths", choices=config_files, defaultValue=next(iter(config_files), "")
)

# ``overwrite`` is deliberately a widget rather than always-on. This task runs on
# every pipeline execution, so rotating credentials by default would replace the
# secret a deployed agent is currently authenticating with, on every deploy.
dbutils.widgets.dropdown(
    name="overwrite", choices=["false", "true"], defaultValue="false"
)

explicit_path: str | None = dbutils.widgets.get("config-path") or None
discovered_path: str | None = dbutils.widgets.get("config-paths") or None

# An explicit `config-path` always wins; the `../config` dropdown is the fallback
# for an interactive run. Re-binding `config_path` to a second type (as the
# sibling notebooks do) makes the annotation a lie, so the inputs get their own
# names and the result is annotated once.
resolved_path: str | None = explicit_path or discovered_path
if not resolved_path:
    # Neither the explicit widget nor `../config` discovery yielded a path, so
    # there is nothing to load. Fail here naming both inputs, rather than at the
    # read with a bare TypeError/FileNotFoundError on an empty path.
    raise ValueError(
        "No config to provision from: the `config-path` widget is empty and no "
        "YAML was found under ../config. Set `config-path` to the staged config "
        "(the pipeline bundle always passes it)."
    )

config_path: str = resolved_path
overwrite: bool = dbutils.widgets.get("overwrite") == "true"

print(config_path)

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

from dotenv import find_dotenv, load_dotenv

_loaded: bool = load_dotenv(find_dotenv())

# COMMAND ----------

from dao_ai.config import AppConfig

config: AppConfig = AppConfig.from_file(path=config_path)

# COMMAND ----------

# Nothing declared, nothing to provision. This task runs unconditionally (it gates
# provision-lakebase, whose Postgres role takes the SP's client_id as its subject),
# but a config that never asked for a service principal has no secret scope or key
# names to store credentials under — ``provision_all`` would raise on the synthetic
# "default" target ``resolve_sp_targets`` returns for such a config, failing the
# task and skipping every downstream one.
#
# Skipping here matches every sibling provisioning task: each iterates its own
# declarations (``config.resources.databases``, ``.vector_stores``, ``genie_rooms``)
# and no-ops on an empty mapping. Service principals are opt-in the same way — a
# Lakebase config declares them because a Postgres role's subject IS the SP's
# client_id, and an agent that needs none should not have one invented for it.
if not config.service_principals:
    print("No service_principals declared in the config — nothing to provision.")
    # Exits successfully, so the tasks gated on this one still run. No payload:
    # unlike 05_provision_genie, nothing downstream reads this task's taskValues.
    dbutils.notebook.exit("no service_principals declared")  # noqa: F821

# COMMAND ----------

# Create every service principal the config declares and store its credentials.
# Same library entry point the ``dao-ai sp provision`` CLI calls — the CLI is one
# provisioning surface, this is another.
#
# ``do_grant=False`` on purpose: granting is 06_grant_service_principal's job.
# Every grant target is created by a LATER task — tables (01/02), the Lakebase
# project (03), UC functions (04), the Genie space (05) — so granting here would
# fail "absent" on all of them, and nothing re-grants afterwards. Splitting also
# separates the privileges: creating an identity is an account-level operation,
# authorizing it is a per-resource one.
#
# The WorkspaceClient is ambient: the job's identity. Creating service principals
# and writing secret scopes need more privilege than reading a table, so if this
# reports a permissions failure, that identity is what to grant. It no longer
# fails the task — see ``provision_all``'s warn-and-continue.
from databricks.sdk import WorkspaceClient

from dao_ai.service_principal import (
    MultiProvisionResult,
    ProvisionResult,
    ServicePrincipalTarget,
    provision_all,
    resolve_sp_targets,
)

w: WorkspaceClient = WorkspaceClient()

targets: list[ServicePrincipalTarget] = resolve_sp_targets(config)
result: MultiProvisionResult = provision_all(
    w,
    config=config,
    targets=targets,
    overwrite=overwrite,
    do_grant=False,
)

provisioned: ProvisionResult
for provisioned in result.results:
    verb: str = "reused" if provisioned.reused else "created"
    print(f"\nservice principal '{provisioned.name}' — {provisioned.display_name}")
    if provisioned.blocked_reason:
        print(f"  BLOCKED: {provisioned.blocked_reason}")
        continue
    if provisioned.provision_error:
        # Attempted and failed — most often a missing create-SP privilege. Reported
        # rather than raised so the rest of the workflow still provisions; the
        # grant task will report the resources it consequently could not authorize.
        print(f"  NOT PROVISIONED ({provisioned.provision_failure_kind}):")
        print(f"    {provisioned.provision_error}")
        continue
    print(f"  {verb}: client_id={provisioned.client_id or '(none)'}")
    if provisioned.secret_action:
        print(
            f"  secrets: {provisioned.secret_action} (scope {provisioned.stored_scope})"
        )
    if provisioned.existing_keys:
        print(f"  already populated: {', '.join(provisioned.existing_keys)}")

print("\nGrants are applied by the grant-service-principal task, after the")
print("resources they target exist.")

# COMMAND ----------

# Fail the task only when provisioning was *refused* because it could not be done
# safely — today, a secret key that already holds a value with no matching service
# principal. Rotating it would break whatever currently authenticates with it, so
# the operator has to choose. A missing PRIVILEGE is deliberately not fatal (see
# ``provision_all``): it degrades the run instead of blocking the infrastructure
# work that has nothing to do with the service principal.
if result.blocked:
    blocked: list[tuple[str, str]] = result.blocked
    raise RuntimeError(
        "Service-principal provisioning was blocked for: "
        + "; ".join(f"{name}: {reason}" for name, reason in blocked)
    )
