# Databricks notebook source
# Dependency bootstrap. Install dao-ai via uv, self-contained (stdlib only, no
# dao_ai import — a bootstrap must not import the package it installs). Prefer the
# deploy's pinned ``dao_ai_dep`` parameter (a ./dist wheel re-anchored to ../dist in
# development, else a version/PyPI spec); fall back to the newest local ../dist wheel,
# else PyPI, only for standalone runs. ``%restart_python`` makes the freshly installed
# package importable below. The spec is single-quoted in the magic so a dev wheel's
# ``+local`` tag and any ``[extras]`` survive shell expansion.
import glob, os

from packaging.version import Version


# Newest by *version*, not by filename: a lexical sort puts 0.2.8 above 0.2.10.
def _wheel_version(wheel: str) -> Version:
    return Version(os.path.basename(wheel).split("-")[1])


_wheels = sorted(glob.glob("../dist/dao_ai-*.whl"), key=_wheel_version, reverse=True)
dbutils.widgets.text(name="dao_ai_dep", defaultValue="")
_pin = dbutils.widgets.get("dao_ai_dep")
if _pin.endswith(".whl"):
    _dao_ai_dep = os.path.join("..", _pin.removeprefix("./"))
elif _pin:
    _dao_ai_dep = _pin
else:
    _dao_ai_dep = _wheels[0] if _wheels else "dao-ai"

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

# ``overwrite`` is deliberately a widget rather than always-on. This task runs on
# every pipeline execution, so rotating credentials by default would replace the
# secret a deployed agent is currently authenticating with, on every deploy.
dbutils.widgets.dropdown(
    name="overwrite", choices=["false", "true"], defaultValue="false"
)

# There is no `../config` discovery fallback. That directory exists only in the
# staged bundle layout, and the bundle stages exactly one config — the same one
# the job passes here — so discovery could only ever guess, and guessing is how
# the wrong config gets loaded. `config-path` is the single input.
widget_path: str | None = dbutils.widgets.get("config-path") or None
if not widget_path:
    raise ValueError(
        "No config to provision from: the `config-path` widget is empty. In a "
        "staged pipeline bundle the config sits beside this notebook under "
        "`../config/` and the job always passes it; running this notebook by "
        "hand, set `config-path` to an absolute workspace path, for example "
        "`/Workspace/Users/you@example.com/dao-ai/examples/04_genie/genie_basic.yaml`."
    )

config_path: str = widget_path
overwrite: bool = dbutils.widgets.get("overwrite") == "true"

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
