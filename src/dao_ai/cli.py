import argparse
import getpass
import json
import os
import signal
import subprocess
import sys
import traceback
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Optional, Sequence

from dotenv import find_dotenv, load_dotenv
from loguru import logger

from dao_ai.config import AppConfig
from dao_ai.config_vars import ConfigVariableError, resolve_parameters
from dao_ai.graph import create_dao_ai_graph
from dao_ai.logging import configure_logging
from dao_ai.models import save_image
from dao_ai.utils import normalize_name

configure_logging(level="ERROR")

DATABRICKS_AUTH_ENV_VARS: list[str] = [
    "DATABRICKS_TOKEN",
    "DATABRICKS_CLIENT_ID",
    "DATABRICKS_CLIENT_SECRET",
    "DATABRICKS_HOST",
    "DATABRICKS_AUTH_TYPE",
]


def _apply_profile_context(profile: Optional[str]) -> None:
    """When a --profile is specified, make it authoritative for this process.

    A ``.env`` file (loaded at module import via ``load_dotenv``) or the user's
    shell can inject ``DATABRICKS_TOKEN`` / ``DATABRICKS_HOST`` etc. that the
    Databricks SDK treats as higher priority than ``DATABRICKS_CONFIG_PROFILE``.
    When the operator explicitly picks a profile on the CLI, those injected
    credentials silently hijack every subsequent SDK call — pointing
    Genie/workspace API requests at the wrong host. Pop them so the profile
    wins. No-op when ``profile`` is not set.
    """
    if not profile:
        return
    for var in DATABRICKS_AUTH_ENV_VARS:
        os.environ.pop(var, None)
    os.environ["DATABRICKS_CONFIG_PROFILE"] = profile

    # Materialize a static bearer token from the profile up-front, while the
    # process is still single-threaded, and export it as DATABRICKS_HOST /
    # DATABRICKS_TOKEN so every subsequent SDK call uses token auth instead of
    # the profile's ``databricks-cli`` (OAuth U2M) auth.
    #
    # Why: OAuth U2M mints tokens by forking the ``databricks`` binary. During
    # ``dao-ai deploy``, MLflow's ``log_model`` validation runs the full agent
    # in-process (LLMs, tools, tracing) with live gRPC channels / threads, and
    # any SDK token refresh then forks unsafely — producing
    # ``forced token refresh: cache update: exit status 45`` and a ``401`` that
    # non-deterministically aborts the deploy. Resolving the token once here,
    # before any threads exist, removes every in-process fork. Best-effort:
    # if resolution fails we leave the profile in place and let the SDK's
    # normal auth chain handle it.
    try:
        from databricks.sdk import WorkspaceClient

        w = WorkspaceClient(profile=profile)
        headers: dict[str, str] = w.config.authenticate() or {}
        authorization: str = headers.get("Authorization", "")
        host: Optional[str] = w.config.host
        if authorization.startswith("Bearer ") and host:
            os.environ["DATABRICKS_HOST"] = host
            os.environ["DATABRICKS_TOKEN"] = authorization[len("Bearer ") :]
            # Token auth must win over the profile for the SDK's auth-type
            # detection, so drop the profile pointer now that we have a token.
            os.environ.pop("DATABRICKS_CONFIG_PROFILE", None)
            logger.debug(
                "Materialized static bearer token from profile for fork-safe auth",
                profile=profile,
                host=host,
            )
    except Exception as e:
        logger.debug(
            "Could not materialize token from profile; leaving profile-based "
            "auth in place",
            profile=profile,
            error=str(e),
        )


def get_default_user_id() -> str:
    """
    Get the default user ID for the CLI session.

    Tries to get the current user from Databricks, falls back to local user.

    Returns:
        User ID string (Databricks username or local username)
    """
    try:
        # Try to get current user from Databricks SDK
        from databricks.sdk import WorkspaceClient

        w = WorkspaceClient()
        current_user = w.current_user.me()
        user_id = current_user.user_name
        logger.debug(f"Using Databricks user: {user_id}")
        return user_id
    except Exception as e:
        # Fall back to local system user
        logger.debug(f"Could not get Databricks user, using local user: {e}")
        local_user = getpass.getuser()
        logger.debug(f"Using local user: {local_user}")
        return local_user


def detect_cloud_provider(profile: Optional[str] = None) -> Optional[str]:
    """
    Detect the cloud provider from the Databricks workspace URL.

    The cloud provider is determined by the workspace URL pattern:
    - Azure: *.azuredatabricks.net
    - AWS: *.cloud.databricks.com (without gcp subdomain)
    - GCP: *.gcp.databricks.com

    Args:
        profile: Optional Databricks CLI profile name

    Returns:
        Cloud provider string ('azure', 'aws', 'gcp') or None if detection fails
    """
    saved_vars: dict[str, str] = {}
    try:
        import os

        from databricks.sdk import WorkspaceClient

        if profile:
            for var in DATABRICKS_AUTH_ENV_VARS:
                if var in os.environ:
                    saved_vars[var] = os.environ.pop(var)

        # Create workspace client with optional profile
        if profile:
            logger.debug(f"Creating WorkspaceClient with profile: {profile}")
            w = WorkspaceClient(profile=profile)
        else:
            logger.debug("Creating WorkspaceClient with default/ambient credentials")
            w = WorkspaceClient()

        # Get the workspace URL from config
        host = w.config.host
        logger.debug(f"WorkspaceClient host: {host}, profile used: {profile}")
        if not host:
            logger.warning("Could not determine workspace URL for cloud detection")
            return None

        host_lower = host.lower()

        if "azuredatabricks.net" in host_lower:
            logger.debug(f"Detected Azure cloud from workspace URL: {host}")
            return "azure"
        elif ".gcp.databricks.com" in host_lower:
            logger.debug(f"Detected GCP cloud from workspace URL: {host}")
            return "gcp"
        elif ".cloud.databricks.com" in host_lower or "databricks.com" in host_lower:
            logger.debug(f"Detected AWS cloud from workspace URL: {host}")
            return "aws"
        else:
            logger.warning(f"Could not determine cloud provider from URL: {host}")
            return None

    except Exception as e:
        logger.warning(f"Could not detect cloud provider: {e}")
        return None
    finally:
        os.environ.update(saved_vars)


env_path: str = find_dotenv()
if env_path:
    logger.info(f"Loading environment variables from: {env_path}")
    _ = load_dotenv(env_path)


def _parse_var_args(raw: Optional[list[str]]) -> dict[str, str]:
    """Parse a list of ``KEY=VALUE`` strings into a dict.

    Raises ``SystemExit`` (mirrors argparse's behaviour for invalid args)
    when an item lacks ``=``.
    """
    out: dict[str, str] = {}
    for item in raw or []:
        if "=" not in item:
            raise SystemExit(f"--var expects KEY=VALUE, got: {item!r}")
        key, value = item.split("=", 1)
        out[key.strip()] = value
    return out


def _add_var_argument(parser: ArgumentParser) -> None:
    """Add a repeatable ``--param KEY=VALUE`` flag (alias: ``--var``) to a subparser."""
    parser.add_argument(
        "--param",
        "--var",
        dest="var",
        action="append",
        metavar="KEY=VALUE",
        help=(
            "Override a ${param.KEY} / ${var.KEY} substitution in the config "
            "file. Repeatable (e.g. --param catalog=main --param schema=dao_ai). "
            "Alias: --var."
        ),
    )


def _add_profile_argument(parser: ArgumentParser) -> None:
    """Add the ``-p/--profile`` flag to a subparser.

    Shared so every subcommand spells the Databricks profile flag identically.
    Handlers must call ``_apply_profile_context(options.profile)`` before
    constructing any WorkspaceClient so the profile is authoritative.
    """
    parser.add_argument(
        "-p",
        "--profile",
        type=str,
        help="The Databricks CLI profile to use for authentication.",
    )


def _print_config_variable_error(err: ConfigVariableError) -> None:
    """Render a ConfigVariableError to stderr in a user-friendly form."""
    print(f"\nConfig parameter error in {err.path}:", file=sys.stderr)
    if err.missing_required:
        print("  Missing required parameters:", file=sys.stderr)
        for name in err.missing_required:
            print(f"    - {name}", file=sys.stderr)
        print(
            "  Pass with --var name=value or set the equivalent env var "
            "(NAME upper-cased, dots/dashes -> underscores).",
            file=sys.stderr,
        )
    if err.undeclared:
        print(
            "  Undeclared ${var.NAME} / ${param.NAME} references:",
            file=sys.stderr,
        )
        for name in err.undeclared:
            print(f"    - {name}", file=sys.stderr)
        print(
            "  Add them to the top-level parameters: block in the config.",
            file=sys.stderr,
        )


def parse_args(args: Sequence[str]) -> Namespace:
    parser: ArgumentParser = ArgumentParser(
        prog="dao-ai",
        description="DAO AI Agent Command Line Interface - A comprehensive tool for managing, validating, and visualizing multi-agent DAO AI systems",
        epilog="""
Examples:
  dao-ai schema                                          # Generate JSON schema for configuration validation
  dao-ai validate -c config/model_config.yaml            # Validate a specific configuration file
  dao-ai graph -o architecture.png -c my_config.yaml -v  # Generate visual graph with verbose output
  dao-ai chat -c config/retail.yaml --custom-input store_num=87887  # Start interactive chat session
  dao-ai list-mcp-tools -c config/mcp_config.yaml --apply-filters  # List filtered MCP tools only
  dao-ai validate                                        # Validate with detailed logging
  dao-ai pipeline --deploy                               # Deploy the DAO AI pipeline
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity level (use -v, -vv, -vvv, or -vvvv for ERROR, WARNING, INFO, DEBUG, or TRACE levels)",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Available commands for managing the DAO AI system",
        metavar="COMMAND",
    )

    # Version command
    _version_parser: ArgumentParser = subparsers.add_parser(
        "version",
        help="Show dao-ai version and environment information",
        description="Display the dao-ai version along with Python, key dependency versions, and platform details.",
    )

    # Schema command
    _: ArgumentParser = subparsers.add_parser(
        "schema",
        help="Generate JSON schema for configuration validation",
        description="""
Generate the JSON schema definition for the DAO AI configuration format.
This schema can be used for IDE autocompletion, validation tools, and documentation.
The output is a complete JSON Schema that describes all valid configuration options,
including agents, tools, models, orchestration patterns, and guardrails.
        """,
        epilog="Example: dao-ai schema > config_schema.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Validation command
    validation_parser: ArgumentParser = subparsers.add_parser(
        "validate",
        help="Validate configuration file syntax and semantics",
        description="""
Validate a DAO AI configuration file for correctness and completeness.
This command checks:
- YAML syntax and structure
- Required fields and data types
- Agent configurations and dependencies
- Tool definitions and availability
- Model specifications and compatibility
- Orchestration patterns (supervisor/swarm)
- Guardrail configurations

Exit codes:
  0 - Configuration is valid
  1 - Configuration contains errors
        """,
        epilog="""
Examples:
  dao-ai validate                                  # Validate default ./config/model_config.yaml
  dao-ai validate -c config/production.yaml       # Validate specific config file
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    validation_parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        metavar="FILE",
        help="Path to the model configuration file to validate (default: ./config/model_config.yaml)",
    )

    # Create-experiment command
    create_experiment_parser: ArgumentParser = subparsers.add_parser(
        "create-experiment",
        help="Create (or look up) an MLflow experiment and print its id",
        description="""
Provision or resolve an MLflow experiment on Databricks and print the
resulting id + metadata. Delegates to
``DatabricksProvider.create_experiment`` — same code path used by
``dao-ai deploy`` when it resolves ``app.experiment`` from a config.

Pass ``--name`` for create-if-missing behavior (default) or ``--id``
to verify an existing experiment. Exactly one of the two is required.
        """,
        epilog="""
Examples:
  dao-ai create-experiment --name /Shared/rcg/hardware_store_traces
  dao-ai create-experiment --name /Shared/team/agent -p fevm
  dao-ai create-experiment --id 1952423719449237 --output json
  dao-ai create-experiment --name /Shared/only-if-exists --no-create
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _create_exp_ident_group = create_experiment_parser.add_mutually_exclusive_group(
        required=True
    )
    _create_exp_ident_group.add_argument(
        "--name",
        type=str,
        metavar="PATH",
        help="Workspace path (e.g. /Shared/team/traces). Created if missing unless --no-create.",
    )
    _create_exp_ident_group.add_argument(
        "--id",
        type=str,
        metavar="ID",
        help="Numeric experiment id. Fetched (not created).",
    )
    create_experiment_parser.add_argument(
        "--no-create",
        action="store_true",
        help="With --name: fail instead of creating when the experiment is missing.",
    )
    create_experiment_parser.add_argument(
        "-p",
        "--profile",
        type=str,
        help="Databricks profile to use for authentication.",
    )
    create_experiment_parser.add_argument(
        "-o",
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text).",
    )

    # Link-trace-destination command
    link_trace_parser: ArgumentParser = subparsers.add_parser(
        "link-trace-destination",
        help="Link an MLflow experiment to its UC trace destination",
        description="""
Link an MLflow experiment to the Unity Catalog trace destination
declared under ``app.trace_location`` in the config. Idempotent — safe
to run repeatedly.

Run this AFTER ``databricks bundle deploy`` and BEFORE
``databricks bundle run``. Fixes the case where MLflow rejects the
runtime link attempt with "already contains traces" on a re-deploy or
after a trace-location change, which otherwise causes silent trace loss.

Experiment resolution order:
  1. ``--experiment-id`` flag (explicit override)
  2. ``config.app.experiment.resolved_id`` if ``experiment:`` is set
  3. Bundle-declared experiment name (``/Users/<current-user>/<app-name>``)
     looked up via MlflowClient — matches what dao-ai generate-bundle
     writes for the auto-declared experiment path.

No-op when ``config.app.trace_location`` is not set.
        """,
        epilog="""
Examples:
  # Typical bundle flow — insert between deploy and run
  databricks bundle deploy --target dev -p fevm
  dao-ai link-trace-destination -c config.yaml -p fevm
  databricks bundle run my-app --target dev -p fevm

  # Explicit experiment id
  dao-ai link-trace-destination -c config.yaml --experiment-id 1234567890 -p fevm

Notes:
  * Restart the app after linking. MLflow's tracer provider is
    initialized once at process startup and caches the resolved UC
    destination — linking against a running app won't retroactively
    route in-flight traces to the new location. Trigger
    `databricks apps restart <name>` (or any bundle re-deploy).
  * Databricks does NOT allow un-linking a UC trace destination.
    The OSS `mlflow.tracing.unset_experiment_trace_location` API exists,
    but the Databricks control plane rejects it with:
      BAD_REQUEST: Unlinking an experiment from a Unity Catalog trace
      location is not allowed.
    Consequently, `catalog` / `schema` / `table_prefix` cannot be
    changed once the experiment is linked. To move traces to a different
    UC destination, create a fresh experiment (new name or id), point
    `MLFLOW_EXPERIMENT_ID` at it, link, then restart. The old experiment
    continues writing to its original UC destination forever.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    link_trace_parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        metavar="FILE",
        help="Path to the model configuration file (must set app.trace_location).",
    )
    link_trace_parser.add_argument(
        "-p",
        "--profile",
        type=str,
        help="Databricks profile to use for authentication.",
    )
    link_trace_parser.add_argument(
        "--experiment-id",
        type=str,
        metavar="ID",
        help="Explicit experiment id (skips resolution from config/bundle name).",
    )
    link_trace_parser.add_argument(
        "--app-sp",
        type=str,
        metavar="CLIENT_ID",
        help=(
            "Service principal client_id (UUID) of the Databricks App runtime "
            "identity to grant experiment CAN_EDIT and UC OTEL table SELECT+MODIFY. "
            "When omitted, auto-resolved via ``apps.get(config.app.name)`` — pass "
            "explicitly to override or to grant a non-default principal. "
            "Set ``app.manage_permissions: false`` in the config to skip grants "
            "entirely (admin-provisioned scenarios)."
        ),
    )
    _add_var_argument(link_trace_parser)

    # Grant trace permissions command
    grant_trace_parser: ArgumentParser = subparsers.add_parser(
        "grant-trace-permissions",
        help="Grant an App SP the experiment + UC OTEL table permissions MLflow tracing needs.",
        description="""
Grant the experiment ``CAN_EDIT`` ACL and the UC OTEL trace-table
``USE_CATALOG``/``USE_SCHEMA``/``SELECT``/``MODIFY`` privileges that
MLflow tracing needs at runtime to persist traces into a UC-backed
experiment's OTEL Delta tables.

Standalone counterpart to the grant step that ``dao-ai deploy`` / ``dao-ai
pipeline`` runs automatically inside ``deploy_app_agent`` /
``deploy_model_serving_agent``. Useful for the ``generate-bundle`` + ``bundle
deploy`` + ``link-trace-destination`` flow (where no full deploy fires),
or for retroactively fixing grants when an app was deployed by an
identity that lacked GRANT rights.

Idempotent — repeated calls with the same principal + privileges no-op
on the workspace side.

Experiment resolution order matches ``link-trace-destination``:
  1. ``--experiment-id`` flag
  2. ``config.app.experiment.resolved_id`` if ``experiment:`` is set
  3. Bundle-declared name lookup (``/Users/<current-user>/<app-name>``)

App SP resolution:
  1. ``--app-sp`` flag (explicit)
  2. ``apps.get(config.app.name).service_principal_client_id``

No-op when ``config.app.trace_location`` is not set.
        """,
        epilog="""
Examples:
  # Retroactively grant an already-deployed App its trace-write permissions
  dao-ai grant-trace-permissions -c config.yaml -p fevm

  # Grant a specific SP explicitly (e.g. shared workload identity)
  dao-ai grant-trace-permissions -c config.yaml --app-sp <uuid> -p fevm
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    grant_trace_parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        metavar="FILE",
        help="Path to the model configuration file (must set app.trace_location).",
    )
    grant_trace_parser.add_argument(
        "-p",
        "--profile",
        type=str,
        help="Databricks profile to use for authentication.",
    )
    grant_trace_parser.add_argument(
        "--experiment-id",
        type=str,
        metavar="ID",
        help="Explicit experiment id (skips resolution from config/bundle name).",
    )
    grant_trace_parser.add_argument(
        "--app-sp",
        type=str,
        metavar="CLIENT_ID",
        help=(
            "Service principal client_id (UUID) to grant. When omitted, "
            "auto-resolved via ``apps.get(config.app.name)``."
        ),
    )
    _add_var_argument(grant_trace_parser)

    # Graph command
    graph_parser: ArgumentParser = subparsers.add_parser(
        "graph",
        help="Generate visual representation of the agent workflow",
        description="""
Generate a visual graph representation of the configured DAO AI system.
This creates a diagram showing:
- Agent nodes and their relationships
- Orchestration flow (supervisor or swarm patterns)
- Tool dependencies and connections
- Message routing and state transitions
- Conditional logic and decision points
        """,
        epilog="""
Examples:
  dao-ai graph -o architecture.png                # Generate PNG diagram
  dao-ai graph -o workflow.png -c prod.yaml       # Generate PNG from specific config
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    graph_parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        metavar="FILE",
        help="Output file path for the generated graph.",
    )
    graph_parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        metavar="FILE",
        help="Path to the model configuration file to visualize",
    )

    pipeline_parser: ArgumentParser = subparsers.add_parser(
        "pipeline",
        help="Pipeline orchestration for deployment",
        description="""
Deploy and run the DAO AI pipeline. This command wraps the underlying
Databricks Asset Bundle workflow:
- Deploys DAO AI as a Databricks asset bundle
- Runs the DAO AI system with the current configuration
""",
        epilog="""
Examples:
    dao-ai pipeline --deploy
    dao-ai pipeline --run
""",
    )

    pipeline_parser.add_argument(
        "-p",
        "--profile",
        type=str,
        help="The Databricks profile to use for deployment",
    )
    pipeline_parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        metavar="FILE",
        help="Path to the model configuration file for the bundle",
    )
    pipeline_parser.add_argument(
        "-d",
        "--deploy",
        action="store_true",
        help="Deploy the DAO AI asset bundle",
    )
    pipeline_parser.add_argument(
        "--destroy",
        action="store_true",
        help="Destroy the DAO AI asset bundle",
    )
    pipeline_parser.add_argument(
        "-r",
        "--run",
        action="store_true",
        help="Run the DAO AI system with the current configuration",
    )
    pipeline_parser.add_argument(
        "-t",
        "--target",
        type=str,
        help="Bundle target name (default: auto-generated from app name and cloud)",
    )
    pipeline_parser.add_argument(
        "--cloud",
        type=str,
        choices=["azure", "aws", "gcp"],
        help="Cloud provider (auto-detected from workspace URL if not specified)",
    )
    pipeline_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without executing the deployment or run commands",
    )
    pipeline_parser.add_argument(
        "--deployment-target",
        type=str,
        choices=["model_serving", "apps", "both"],
        default=None,
        help="Agent deployment target: 'model_serving', 'apps', or 'both'. "
        "If not specified, uses app.deployment_target from config file, "
        "or defaults to 'model_serving'. Passed to the deploy notebook.",
    )
    pipeline_parser.add_argument(
        "--development",
        dest="development",
        default=None,
        action="store_true",
        help="Ship local dao-ai source/wheel instead of the published PyPI "
        "package. Rebuild the wheel first with 'uv build --wheel'. "
        "Defaults to auto-detect from the install type. Passed to the deploy "
        "notebook.",
    )
    pipeline_parser.add_argument(
        "--no-development",
        dest="development",
        action="store_false",
        help="Force the published PyPI dao-ai package even from a local/editable "
        "install. Passed to the deploy notebook.",
    )

    # Generate bundle command
    generate_bundle_parser: ArgumentParser = subparsers.add_parser(
        "generate-bundle",
        help="Generate Databricks App bundle files from a config",
        description="""
Generate a complete, deployable Databricks Apps bundle directory from a dao-ai config file.
Creates databricks.yaml, app.yaml, pyproject.toml, and scaffolding files.
        """,
        epilog="""
Examples:
  dao-ai generate-bundle -c config/retail.yaml -o ./my-bundle
  dao-ai generate-bundle -c config/retail.yaml -o ./my-bundle --overwrite
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    generate_bundle_parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        metavar="FILE",
        help="Path to the dao-ai configuration file",
    )
    generate_bundle_parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=".",
        metavar="DIR",
        help="Directory to write the generated bundle files to (default: current directory)",
    )
    generate_bundle_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the output directory",
    )
    generate_bundle_parser.add_argument(
        "--development",
        dest="development",
        default=None,
        action="store_true",
        help="Bundle local dao-ai source/wheel instead of pinning the published "
        "PyPI package. Rebuild the wheel first with 'uv build --wheel'. "
        "Defaults to auto-detect from the install type.",
    )
    generate_bundle_parser.add_argument(
        "--no-development",
        dest="development",
        action="store_false",
        help="Force the published PyPI dao-ai package even from a local/editable "
        "install.",
    )
    generate_bundle_parser.add_argument(
        "-p",
        "--profile",
        type=str,
        help="The Databricks profile to use for config loading",
    )

    # Generate MCP-server bundle command
    generate_mcp_parser: ArgumentParser = subparsers.add_parser(
        "generate-mcp",
        help="Generate a Databricks Apps bundle that runs the dao-ai MCP server",
        description="""
Generate a deploy-ready Databricks Apps bundle from a dao-ai config that exposes
the configured Genie cache + Vector Search retriever tools as MCP tools over
Streamable HTTP. Mirrors `generate-bundle` but emits the MCP-only artifact
(databricks.yml, app.yaml, pyproject.toml, requirements.txt, README.md).
        """,
        epilog="""
Examples:
  dao-ai generate-mcp -c config/retail.yaml -o ./retail-mcp
  dao-ai generate-mcp -c config/retail.yaml -o ./retail-mcp --overwrite
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    generate_mcp_parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        metavar="FILE",
        help="Path to the dao-ai configuration file",
    )
    generate_mcp_parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=".",
        metavar="DIR",
        help="Directory to write the generated MCP bundle files to (default: current directory)",
    )
    generate_mcp_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the output directory",
    )
    generate_mcp_parser.add_argument(
        "--development",
        dest="development",
        default=None,
        action="store_true",
        help="Bundle local dao-ai source/wheel instead of pinning the published "
        "PyPI package. Rebuild the wheel first with 'uv build --wheel'. "
        "Defaults to auto-detect from the install type.",
    )
    generate_mcp_parser.add_argument(
        "--no-development",
        dest="development",
        action="store_false",
        help="Force the published PyPI dao-ai package even from a local/editable "
        "install.",
    )
    generate_mcp_parser.add_argument(
        "-p",
        "--profile",
        type=str,
        help="The Databricks profile to use for config loading",
    )

    # Deploy command
    deploy_parser: ArgumentParser = subparsers.add_parser(
        "deploy",
        help="Deploy configuration file syntax and semantics",
        description="""
Deploy the DAO AI system using the specified configuration file.
This command validates the configuration and deploys the DAO AI agents, tools, and models to the
        """,
        epilog="""
Examples:
  dao-ai deploy                                  # Validate default ./config/model_config.yaml
  dao-ai deploy -c config/production.yaml       # Validate specific config file
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    deploy_parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        metavar="FILE",
        help="Path to the model configuration file to deploy.",
    )
    deploy_parser.add_argument(
        "-t",
        "--target",
        type=str,
        choices=["model_serving", "apps", "both"],
        default=None,
        help="Deployment target: 'model_serving', 'apps', or 'both'. "
        "If not specified, uses app.deployment_target from config file, "
        "or defaults to 'model_serving'.",
    )
    _add_profile_argument(deploy_parser)
    deploy_parser.add_argument(
        "--development",
        dest="development",
        default=None,
        action="store_true",
        help="Ship local dao-ai source/wheel instead of the published PyPI "
        "package (Apps target). Rebuild the wheel first with 'uv build --wheel'. "
        "Defaults to auto-detect from the install type.",
    )
    deploy_parser.add_argument(
        "--no-development",
        dest="development",
        action="store_false",
        help="Force the published PyPI dao-ai package even from a local/editable "
        "install (Apps target).",
    )

    # List MCP tools command
    list_mcp_parser: ArgumentParser = subparsers.add_parser(
        "list-mcp-tools",
        help="List available MCP tools from configuration",
        description="""
List all available MCP tools from the configured MCP servers.
This command shows:
- All MCP servers/functions in the configuration
- Available tools from each server
- Full descriptions for each tool (no truncation)
- Tool parameters in readable format (type, required/optional, descriptions)
- Which tools are included/excluded based on filters
- Filter patterns (include_tools, exclude_tools)

Use this command to:
- Discover available tools before configuring agents
- Review tool descriptions and parameter schemas
- Debug tool filtering configuration
- Verify MCP server connectivity

Options:
- Use --apply-filters to only show tools that will be loaded (hides excluded tools)
- Without --apply-filters, see all available tools with include/exclude status

Note: Schemas are displayed in a concise, readable format instead of verbose JSON
        """,
        epilog="""Examples:
  dao-ai list-mcp-tools -c config/model_config.yaml
  dao-ai list-mcp-tools -c config/model_config.yaml --apply-filters
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    list_mcp_parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./config/model_config.yaml",
        required=False,
        metavar="FILE",
        help="Path to the model configuration file (default: ./config/model_config.yaml)",
    )
    list_mcp_parser.add_argument(
        "--apply-filters",
        action="store_true",
        help="Only show tools that pass include/exclude filters (hide excluded tools)",
    )

    # Monitor command
    monitor_parser: ArgumentParser = subparsers.add_parser(
        "monitor",
        help="Manage production monitoring scorers",
        description="""
Manage production monitoring scorers for the deployed agent.
Scorers continuously evaluate production traces for quality,
safety, and guideline compliance.

Requires app.monitoring to be configured in the YAML config.
        """,
        epilog="""
Examples:
  dao-ai monitor enable -c config/model_config.yaml     # Register and start monitoring scorers
  dao-ai monitor status -c config/model_config.yaml     # Show active scorers and sample rates
  dao-ai monitor disable -c config/model_config.yaml    # Stop all monitoring scorers
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    monitor_parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        metavar="FILE",
        help="Path to the model configuration file",
    )
    monitor_parser.add_argument(
        "action",
        choices=["enable", "status", "disable"],
        help="Monitoring action: enable (register/start scorers), "
        "status (list active scorers), disable (stop all scorers)",
    )

    chat_parser: ArgumentParser = subparsers.add_parser(
        "chat",
        help="Interactive chat with the DAO AI system",
        description="""
Start an interactive chat session with the DAO AI system.
This command provides a REPL (Read-Eval-Print Loop) interface where you can
send messages to the configured agents and receive streaming responses in real-time.

The chat session maintains conversation history and supports the full agent
orchestration capabilities defined in your configuration file.

Use Ctrl-D (EOF) to exit the chat session gracefully.
Use Ctrl-C to interrupt and exit immediately.
        """,
        epilog="""
Examples:
  dao-ai chat -c config/model_config.yaml                              # Start chat (auto-detects user)
  dao-ai chat -c config/retail.yaml --custom-input store_num=87887     # Chat with custom store number
  dao-ai chat -c config/prod.yaml --user-id john.doe@company.com       # Chat with specific user ID
  dao-ai chat -c config/retail.yaml --custom-input store_num=123 --custom-input region=west  # Multiple custom inputs
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    chat_parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        metavar="FILE",
        help="Path to the model configuration file to validate",
    )
    chat_parser.add_argument(
        "--custom-input",
        action="append",
        metavar="KEY=VALUE",
        help="Custom configurable input as key=value pair (can be used multiple times)",
    )
    chat_parser.add_argument(
        "--user-id",
        type=str,
        default=None,  # Will be set to actual user in handle_chat_command
        metavar="ID",
        help="User ID for the chat session (default: current Databricks user or local username)",
    )
    chat_parser.add_argument(
        "--thread-id",
        type=str,
        default=None,
        metavar="ID",
        help="Thread ID for the chat session (default: auto-generated UUID)",
    )

    # Parameters command (alias: vars)
    vars_parser: ArgumentParser = subparsers.add_parser(
        "parameters",
        aliases=["vars"],
        help="Inspect declared parameters in a configuration",
        description="""
Inspect the declared parameters: block in a DAO AI config file.

Shows each declared parameter, whether it is required, its declared default,
and the value that would be substituted into the YAML for the current
combination of --param overrides and process environment variables.

Use this to discover what knobs a config exposes before deploying or running it.
        """,
        epilog="""
Examples:
  dao-ai parameters list -c config/model_config.yaml
  dao-ai parameters list -c config/retail.yaml --param catalog=nfleming
  dao-ai vars list -c config/model_config.yaml             # legacy alias
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    vars_parser.add_argument(
        "action",
        choices=["list"],
        help="Parameters action: 'list' prints declared parameters and resolved values.",
    )
    vars_parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        metavar="FILE",
        help="Path to the model configuration file to inspect.",
    )

    for sub in (
        validation_parser,
        graph_parser,
        pipeline_parser,
        generate_bundle_parser,
        generate_mcp_parser,
        deploy_parser,
        list_mcp_parser,
        monitor_parser,
        chat_parser,
        vars_parser,
    ):
        _add_var_argument(sub)

    # Add -p/--profile to the subcommands that touch Databricks but don't
    # already declare it inline (deploy/pipeline/generate-*/create-experiment/
    # link-trace/grant-trace define their own). Without it a shell/.env
    # DATABRICKS_* var silently wins — the hijack _apply_profile_context guards.
    for sub in (
        validation_parser,
        graph_parser,
        list_mcp_parser,
        monitor_parser,
        chat_parser,
        vars_parser,
    ):
        _add_profile_argument(sub)

    options = parser.parse_args(args)

    # Generate a new thread_id UUID if not provided (only for chat command)
    if hasattr(options, "thread_id") and options.thread_id is None:
        import uuid

        options.thread_id = str(uuid.uuid4())

    return options


def handle_chat_command(options: Namespace) -> None:
    """Interactive chat REPL with the DAO AI system with Human-in-the-Loop support."""
    _apply_profile_context(options.profile)
    logger.debug("Starting chat session with DAO AI system...")

    # Set up signal handler for clean Ctrl+C handling
    def signal_handler(sig: int, frame: Any) -> None:
        try:
            print("\n\n👋 Chat session interrupted. Goodbye!")
            sys.stdout.flush()
        except Exception:
            pass
        sys.exit(0)

    # Store original handler and set our handler
    original_handler = signal.signal(signal.SIGINT, signal_handler)

    try:
        # Set default user_id if not provided
        if options.user_id is None:
            options.user_id = get_default_user_id()

        try:
            config: AppConfig = AppConfig.from_file(
                options.config, params=_parse_var_args(options.var)
            )
        except ConfigVariableError as e:
            _print_config_variable_error(e)
            sys.exit(1)
        app = create_dao_ai_graph(config)

        print("🤖 DAO AI Chat Session Started")
        print("Type your message and press Enter. Use Ctrl-D to exit.")
        print("-" * 50)

        # Show current configuration
        print("📋 Session Configuration:")
        print(f"   Config file: {options.config}")
        print(f"   Thread ID: {options.thread_id}")
        print(f"   User ID: {options.user_id}")
        if options.custom_input:
            print("   Custom inputs:")
            for custom_input in options.custom_input:
                print(f"     {custom_input}")
        print("-" * 50)

        # Import streaming function and interrupt handling
        from langchain_core.messages import AIMessage, HumanMessage

        from dao_ai.models import _extract_text_content

        # Conversation history
        messages = []

        while True:
            try:
                # Read user input
                user_input = input("\n👤 You: ").strip()

                if not user_input:
                    continue

                # Add user message to history
                user_message = HumanMessage(content=user_input)
                messages.append(user_message)

                # Parse custom inputs from command line
                configurable = {
                    "thread_id": options.thread_id,
                    "user_id": options.user_id,
                }

                # Add custom key=value pairs if provided
                if options.custom_input:
                    for custom_input in options.custom_input:
                        try:
                            key, value = custom_input.split("=", 1)
                            # Try to convert to appropriate type
                            if value.isdigit():
                                configurable[key] = int(value)
                            elif value.lower() in ("true", "false"):
                                configurable[key] = value.lower() == "true"
                            elif value.replace(".", "", 1).isdigit():
                                configurable[key] = float(value)
                            else:
                                configurable[key] = value
                        except ValueError:
                            print(
                                f"⚠️  Warning: Invalid custom input format '{custom_input}'. Expected key=value format."
                            )
                            continue

                # Normalize user_id for memory namespace compatibility (replace . with _)
                # This matches the normalization in models.py _convert_to_context
                if configurable.get("user_id"):
                    configurable["user_id"] = configurable["user_id"].replace(".", "_")

                # Create Context object from configurable dict
                from dao_ai.state import Context

                context = Context(**configurable)

                # Prepare config with all context fields for checkpointer/memory
                # Note: langmem tools require user_id in config.configurable for namespace resolution
                config = {"configurable": context.model_dump()}

                # Invoke the graph and handle interrupts (HITL)
                # Wrap in async function to maintain connection pool throughout
                logger.debug(f"Invoking graph with {len(messages)} messages")
                logger.debug(f"Context: {context}")
                logger.debug(f"Config: {config}")

                import asyncio

                from langgraph.errors import GraphInterrupt
                from langgraph.types import Command

                async def _invoke_with_hitl():
                    """Invoke graph and handle HITL interrupts in single async context."""
                    try:
                        result = await app.ainvoke(
                            {"messages": messages},
                            config=config,
                            context=context,
                        )
                    except GraphInterrupt:
                        logger.info(
                            "HITL: GraphInterrupt raised, recovering state from checkpointer"
                        )
                        snapshot = await app.aget_state(config)
                        result = dict(snapshot.values)
                        if snapshot.interrupts:
                            result["__interrupt__"] = list(snapshot.interrupts)

                    # Check for interrupts (Human-in-the-Loop) using __interrupt__
                    # This is the modern LangChain pattern
                    while "__interrupt__" in result:
                        interrupts = result["__interrupt__"]
                        logger.info(f"HITL: {len(interrupts)} interrupt(s) detected")

                        # Collect decisions for all interrupts
                        decisions = []

                        for interrupt in interrupts:
                            interrupt_value = interrupt.value
                            action_requests = interrupt_value.get("action_requests", [])

                            for action_request in action_requests:
                                # Display interrupt information
                                print("\n⚠️  Human in the Loop - Tool Approval Required")
                                print(f"{'=' * 60}")

                                tool_name = action_request.get("name", "unknown")
                                tool_args = action_request.get("args", {})
                                description = action_request.get("description", "")

                                print(f"Tool: {tool_name}")
                                if description:
                                    print(f"\n{description}\n")

                                print("Arguments:")
                                for arg_name, arg_value in tool_args.items():
                                    # Truncate long values
                                    arg_str = str(arg_value)
                                    if len(arg_str) > 100:
                                        arg_str = arg_str[:97] + "..."
                                    print(f"  - {arg_name}: {arg_str}")

                                print(f"{'=' * 60}")

                                # Prompt user for decision
                                while True:
                                    decision_input = (
                                        input(
                                            "\nAction? (a)pprove / (r)eject / (e)dit / (h)elp: "
                                        )
                                        .strip()
                                        .lower()
                                    )

                                    if decision_input in ["a", "approve"]:
                                        logger.info("User approved tool call")
                                        print("✅ Approved - continuing execution...")
                                        decisions.append({"type": "approve"})
                                        break
                                    elif decision_input in ["r", "reject"]:
                                        logger.info("User rejected tool call")
                                        feedback = input(
                                            "   Feedback for agent (optional): "
                                        ).strip()
                                        if feedback:
                                            decisions.append(
                                                {"type": "reject", "message": feedback}
                                            )
                                        else:
                                            decisions.append(
                                                {
                                                    "type": "reject",
                                                    "message": "Tool call rejected by user",
                                                }
                                            )
                                        print(
                                            "❌ Rejected - agent will receive feedback..."
                                        )
                                        break
                                    elif decision_input in ["e", "edit"]:
                                        print(
                                            "ℹ️  Edit functionality not yet implemented in CLI"
                                        )
                                        print("   Please approve or reject.")
                                        continue
                                    elif decision_input in ["h", "help"]:
                                        print("\nAvailable actions:")
                                        print(
                                            "  (a)pprove - Execute the tool call as shown"
                                        )
                                        print(
                                            "  (r)eject  - Cancel the tool call with optional feedback"
                                        )
                                        print(
                                            "  (e)dit    - Modify arguments (not yet implemented)"
                                        )
                                        print("  (h)elp    - Show this help message")
                                        continue
                                    else:
                                        print("Invalid option. Type 'h' for help.")
                                        continue

                        # Resume execution with decisions using Command
                        # This is the modern LangChain pattern
                        logger.debug(f"Resuming with {len(decisions)} decision(s)")
                        result = await app.ainvoke(
                            Command(resume={"decisions": decisions}),
                            config=config,
                            context=context,
                        )

                    return result

                try:
                    # Use async invoke - keep connection pool alive throughout HITL
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                try:
                    result = loop.run_until_complete(_invoke_with_hitl())
                except KeyboardInterrupt:
                    # Re-raise to be caught by outer handler
                    raise
                except asyncio.CancelledError:
                    # Treat cancellation like KeyboardInterrupt
                    raise KeyboardInterrupt
                except Exception as e:
                    logger.error(f"Error invoking graph: {e}")
                    print(f"\n❌ Error: {e}")
                    continue

                # After all interrupts handled, display the final response
                print("\n🤖 Assistant: ", end="", flush=True)

                response_content = ""
                structured_response = None
                try:
                    # Debug: Log what's in the result
                    logger.debug(f"Result keys: {result.keys() if result else 'None'}")
                    if result:
                        for key in result.keys():
                            logger.debug(f"Result['{key}'] type: {type(result[key])}")

                    # Get the latest messages from the result
                    if result and "messages" in result:
                        latest_messages = result["messages"]
                        # Find the last AI message
                        for msg in reversed(latest_messages):
                            if isinstance(msg, AIMessage):
                                if msg.content:
                                    response_content = _extract_text_content(
                                        msg.content
                                    )
                                    print(response_content, end="", flush=True)
                                    break

                    # Check for structured output and display it separately
                    if result and "structured_response" in result:
                        structured_response = result["structured_response"]
                        import json

                        structured_json = json.dumps(
                            structured_response.model_dump()
                            if hasattr(structured_response, "model_dump")
                            else structured_response,
                            indent=2,
                        )

                        # If there was message content, add separator
                        if response_content.strip():
                            print("\n\n📊 Structured Output:")
                            print(structured_json)
                        else:
                            # No message content, just show structured output
                            print(structured_json, end="", flush=True)

                        response_content = response_content or structured_json

                    print()  # New line after response

                    # Add assistant response to history if we got content
                    if response_content.strip():
                        assistant_message = AIMessage(content=response_content)
                        messages.append(assistant_message)
                    else:
                        print("(No response content generated)")

                except Exception as e:
                    print(f"\n❌ Error processing response: {e}")
                    print(f"Stack trace:\n{traceback.format_exc()}")
                    logger.error(f"Response processing error: {e}")
                    logger.error(f"Stack trace: {traceback.format_exc()}")

            except (EOFError, KeyboardInterrupt):
                # Handle Ctrl-D (EOF) or Ctrl-C (interrupt)
                # Use try/except for print in case stdout is closed
                try:
                    print("\n\n👋 Goodbye! Chat session ended.")
                    sys.stdout.flush()
                except Exception:
                    pass
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                logger.error(f"Chat error: {e}")
                traceback.print_exc()

    except (EOFError, KeyboardInterrupt):
        # Handle interrupts during initialization
        try:
            print("\n\n👋 Chat session interrupted. Goodbye!")
            sys.stdout.flush()
        except Exception:
            pass
    except Exception as e:
        logger.error(f"Failed to initialize chat session: {e}")
        print(f"❌ Failed to start chat session: {e}")
        sys.exit(1)
    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, original_handler)


def handle_schema_command(options: Namespace) -> None:
    logger.debug("Generating JSON schema...")
    print(json.dumps(AppConfig.model_json_schema(), indent=2))


def handle_create_experiment_command(options: Namespace) -> None:
    """Create (or resolve) an MLflow experiment and print its metadata."""
    _apply_profile_context(options.profile)

    from dao_ai.config import ExperimentModel
    from dao_ai.providers.databricks import DatabricksProvider

    kwargs: dict[str, Any] = {"create_if_not_exists": not options.no_create}
    if options.name:
        kwargs["name"] = options.name
    if options.id:
        kwargs["id"] = options.id

    try:
        experiment_model = ExperimentModel(**kwargs)
        provider = DatabricksProvider()
        experiment = provider.create_experiment(experiment_model)
    except Exception as e:
        logger.error(f"Failed to create/resolve experiment: {e}")
        sys.exit(1)

    if options.output == "json":
        print(
            json.dumps(
                {
                    "experiment_id": experiment.experiment_id,
                    "name": experiment.name,
                    "artifact_location": experiment.artifact_location,
                    "lifecycle_stage": experiment.lifecycle_stage,
                    "creation_time": experiment.creation_time,
                    "last_update_time": experiment.last_update_time,
                },
                indent=2,
            )
        )
    else:
        print(f"experiment_id:     {experiment.experiment_id}")
        print(f"name:              {experiment.name}")
        print(f"artifact_location: {experiment.artifact_location}")
        print(f"lifecycle_stage:   {experiment.lifecycle_stage}")


def handle_link_trace_destination_command(options: Namespace) -> None:
    """Link an MLflow experiment to its UC trace destination (idempotent).

    Standalone verb intended to run between ``databricks bundle deploy``
    and ``databricks bundle run``. Fixes silent trace loss on re-deploys
    where the runtime's link attempt is rejected with "already contains
    traces" — this call runs from the operator's machine with their own
    credentials, so the tag-based idempotency check (or the API call
    itself, on a truly-first link) can succeed cleanly.

    After linking, unless ``--no-grants`` is passed, also grants the App's
    runtime SP the experiment CAN_EDIT ACL and the UC OTEL trace-table
    SELECT+MODIFY privileges MLflow tracing needs — matching the grant
    step that ``dao-ai deploy_app_agent`` / ``deploy_model_serving_agent``
    perform automatically on their own deploy paths. Without this the
    bundle-based flow (``generate-bundle`` + ``bundle deploy`` +
    ``link-trace-destination``) leaves the App SP without table-write
    permissions and traces are silently dropped at runtime.
    """
    _apply_profile_context(options.profile)

    try:
        config: AppConfig = AppConfig.from_file(
            options.config, params=_parse_var_args(options.var)
        )
    except ConfigVariableError as e:
        _print_config_variable_error(e)
        sys.exit(1)

    if not (config.app and config.app.trace_location):
        print(
            "No app.trace_location configured — nothing to link.",
            file=sys.stderr,
        )
        return

    experiment_id: Optional[str] = _resolve_experiment_id_for_link(
        config, options.experiment_id
    )
    if experiment_id is None:
        sys.exit(1)  # _resolve_* already printed a diagnostic

    from dao_ai.providers.databricks import _link_experiment_trace_location

    try:
        _link_experiment_trace_location(config, experiment_id)
    except Exception as e:  # noqa: BLE001
        print(
            f"Failed to link experiment {experiment_id} to UC trace destination: "
            f"{type(e).__name__}: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"Linked experiment {experiment_id} to "
        f"{config.app.trace_location.catalog_name}."
        f"{config.app.trace_location.schema_name}"
        + (
            f" (table_prefix={config.app.trace_location.resolved_table_prefix})"
            if config.app.trace_location.resolved_table_prefix
            else ""
        )
    )

    _grant_trace_writes_to_app_sp(config, experiment_id, sp_override=options.app_sp)


def handle_grant_trace_permissions_command(options: Namespace) -> None:
    """Grant an App SP the experiment + UC OTEL table permissions.

    Standalone verb equivalent to the grant step that ``deploy_app_agent`` /
    ``deploy_model_serving_agent`` runs. Useful for the bundle-based flow
    (which skips those deploy paths) or for retroactively fixing grants.
    """
    _apply_profile_context(options.profile)

    try:
        config: AppConfig = AppConfig.from_file(
            options.config, params=_parse_var_args(options.var)
        )
    except ConfigVariableError as e:
        _print_config_variable_error(e)
        sys.exit(1)

    if not (config.app and config.app.trace_location):
        print(
            "No app.trace_location configured — nothing to grant.",
            file=sys.stderr,
        )
        return

    experiment_id: Optional[str] = _resolve_experiment_id_for_link(
        config, options.experiment_id
    )
    if experiment_id is None:
        sys.exit(1)

    _grant_trace_writes_to_app_sp(
        config, experiment_id, sp_override=options.app_sp
    )


def _grant_trace_writes_to_app_sp(
    config: AppConfig,
    experiment_id: str,
    sp_override: Optional[str],
) -> None:
    """Resolve the App SP and grant it the trace-write privileges.

    Shared implementation for ``link-trace-destination`` (post-link grant)
    and ``grant-trace-permissions`` (standalone). Respects
    ``config.app.manage_permissions`` — when False, skips silently on the
    assumption that an admin has pre-provisioned grants.

    Resolution order for the App SP:
      1. ``sp_override`` (``--app-sp`` flag).
      2. ``apps.get(config.app.name).service_principal_client_id``.

    Both grant calls WARN-and-continue on failure inside the provider
    helpers, so a deployer without GRANT rights sees diagnostics but the
    command still exits 0 — the App will surface the missing grants
    itself at trace-write time.
    """
    if not config.app.manage_permissions:
        print(
            "app.manage_permissions=false — skipping SP grants. "
            "Ensure an admin has pre-provisioned experiment CAN_EDIT and "
            "UC OTEL SELECT+MODIFY on the trace tables.",
            file=sys.stderr,
        )
        return

    sp_id: Optional[str] = sp_override
    if not sp_id:
        try:
            from databricks.sdk import WorkspaceClient

            w = WorkspaceClient()
            app = w.apps.get(name=config.app.name)
            sp_id = app.service_principal_client_id or app.service_principal_id
        except Exception as e:  # noqa: BLE001
            print(
                f"Could not resolve App SP via apps.get({config.app.name!r}): "
                f"{type(e).__name__}: {e}. "
                "Deploy the app first (``databricks bundle deploy``) or pass "
                "``--app-sp <CLIENT_ID>`` explicitly.",
                file=sys.stderr,
            )
            return

    if not sp_id:
        print(
            f"App {config.app.name!r} has no service_principal_client_id — "
            "pass ``--app-sp <CLIENT_ID>`` explicitly.",
            file=sys.stderr,
        )
        return

    from dao_ai.providers.databricks import (
        _grant_experiment_permissions_to_principal,
        _grant_uc_trace_table_permissions_to_principal,
        _resolve_trace_table_prefix,
    )

    _grant_experiment_permissions_to_principal(
        principal=sp_id, experiment_id=experiment_id
    )

    table_prefix: str = _resolve_trace_table_prefix(
        config,
        None
        if config.app.trace_location.resolved_table_prefix
        else experiment_id,
    )
    _grant_uc_trace_table_permissions_to_principal(
        principal=sp_id,
        catalog_name=config.app.trace_location.catalog_name,
        schema_name=config.app.trace_location.schema_name,
        table_prefix=table_prefix,
    )
    print(
        f"Granted trace-write privileges to App SP {sp_id} on "
        f"{config.app.trace_location.catalog_name}."
        f"{config.app.trace_location.schema_name} "
        f"(table_prefix={table_prefix})"
    )


def _resolve_experiment_id_for_link(
    config: AppConfig,
    override: Optional[str],
) -> Optional[str]:
    """Resolve the MLflow experiment id for link-trace-destination.

    Resolution order:
      1. ``override`` from ``--experiment-id`` flag.
      2. ``config.app.experiment.resolved_id`` (path A — user set ``experiment:``).
      3. Bundle-declared name ``/Users/<current-user>/<app-name>`` looked up
         via ``MlflowClient.get_experiment_by_name`` (path B — the default
         auto-declared bundle experiment).

    Returns ``None`` on failure after printing a diagnostic to stderr. The
    caller exits(1); we don't raise so the CLI's user-facing output stays
    clean.
    """
    if override:
        return override

    if config.app.experiment is not None:
        resolved = config.app.experiment.resolved_id
        if resolved:
            return resolved
        print(
            "app.experiment is set but resolved_id is None — pass "
            "--experiment-id explicitly or run `dao-ai create-experiment` first.",
            file=sys.stderr,
        )
        return None

    # Path B: mirror bundle.py's deterministic name convention. DABs'
    # `--target dev` presets prefix bundle-resource names with
    # `[dev <sanitized-user>]` — try both the unprefixed name (prod-mode
    # deploys) and the dev-prefixed name (personal dev deploys) so the
    # CLI works in either mode without the operator having to specify.
    app_name = config.app.name.lower().replace("_", "-")
    try:
        from databricks.sdk import WorkspaceClient
        from mlflow.tracking import MlflowClient

        current_user = WorkspaceClient().current_user.me().user_name or "unknown"
        # DABs sanitizes the user for the dev prefix: lowercase, strip the
        # email domain, replace non-alnum with underscores.
        dev_tag = current_user.split("@", 1)[0].lower()
        dev_tag = "".join(c if c.isalnum() else "_" for c in dev_tag)
        candidates = [
            f"/Users/{current_user}/{app_name}",
            f"/Users/{current_user}/[dev {dev_tag}] {app_name}",
        ]
        mc = MlflowClient()
        exp = None
        for name in candidates:
            exp = mc.get_experiment_by_name(name)
            if exp is not None:
                break
    except Exception as e:  # noqa: BLE001
        print(
            f"Could not look up bundle-declared experiment: "
            f"{type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return None
    if exp is None:
        print(
            "Experiment not found under any of these names:",
            file=sys.stderr,
        )
        for name in candidates:
            print(f"  - {name}", file=sys.stderr)
        print(
            "Run `databricks bundle deploy` first (which materializes the "
            "experiment), or pass --experiment-id explicitly.",
            file=sys.stderr,
        )
        return None
    return exp.experiment_id


def handle_graph_command(options: Namespace) -> None:
    _apply_profile_context(options.profile)
    logger.debug("Generating graph representation...")
    try:
        config: AppConfig = AppConfig.from_file(
            options.config, params=_parse_var_args(options.var)
        )
    except ConfigVariableError as e:
        _print_config_variable_error(e)
        sys.exit(1)
    app = create_dao_ai_graph(config)
    save_image(app, options.output)


def handle_deploy_command(options: Namespace) -> None:
    from dao_ai.config import DeploymentTarget

    # Make --profile authoritative for this process before any WorkspaceClient
    # is constructed. Both the provider's ``self.w`` and the bare
    # ``WorkspaceClient()`` instances in the grant helpers read
    # ``DATABRICKS_CONFIG_PROFILE``, so this routes the whole deploy — log,
    # register, deploy, and the SP grants — at the selected profile.
    _apply_profile_context(options.profile)

    logger.debug(f"Validating configuration from {options.config}...")
    try:
        try:
            config: AppConfig = AppConfig.from_file(
                options.config, params=_parse_var_args(options.var)
            )
        except ConfigVariableError as e:
            _print_config_variable_error(e)
            sys.exit(1)

        # Hybrid target resolution:
        # 1. CLI --target takes precedence
        # 2. Fall back to config.app.deployment_target
        # 3. Default to MODEL_SERVING (handled in deploy_agent)
        target: DeploymentTarget | None = None
        if options.target is not None:
            target = DeploymentTarget(options.target)
            logger.info(f"Using CLI-specified deployment target: {target.value}")
        elif config.app is not None and config.app.deployment_target is not None:
            target = config.app.deployment_target
            logger.info(f"Using config file deployment target: {target.value}")
        else:
            logger.info("No deployment target specified, defaulting to model_serving")

        # Only log/register the MLflow model for Model Serving deployments.
        # Apps deploy directly from the config + wheel/PyPI package.
        development: bool | None = getattr(options, "development", None)
        if target != DeploymentTarget.APPS:
            config.create_agent(development=development)
        config.deploy_agent(target=target, development=development)
        sys.exit(0)
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        sys.exit(1)


def handle_monitor_command(options: Namespace) -> None:
    from dao_ai.providers.databricks import DatabricksProvider

    _apply_profile_context(options.profile)
    logger.debug(f"Loading configuration from {options.config}...")
    try:
        try:
            config: AppConfig = AppConfig.from_file(
                options.config, params=_parse_var_args(options.var)
            )
        except ConfigVariableError as e:
            _print_config_variable_error(e)
            sys.exit(1)

        if not config.app or not config.app.monitoring:
            logger.error("app.monitoring must be configured in the YAML for monitoring")
            sys.exit(1)

        provider = DatabricksProvider()
        experiment = provider.get_or_create_experiment(config)

        import mlflow

        mlflow.set_experiment(experiment_id=experiment.experiment_id)

        match options.action:
            case "enable":
                from dao_ai.evaluation import register_monitoring_scorers

                sql_warehouse_id: str | None = (
                    config.app.trace_location.warehouse_id
                    if config.app.trace_location
                    else None
                )
                registered = register_monitoring_scorers(
                    monitoring_config=config.app.monitoring,
                    experiment_id=experiment.experiment_id,
                    sql_warehouse_id=sql_warehouse_id,
                )
                print(
                    f"Enabled monitoring: {len(registered)} scorers registered and started"
                )
                for scorer in registered:
                    print(f"  - {scorer.name}")

            case "status":
                from dao_ai.evaluation import get_monitoring_scorers

                scorers = get_monitoring_scorers()
                if not scorers:
                    print("No monitoring scorers registered")
                else:
                    print(f"Monitoring scorers ({len(scorers)}):")
                    for scorer in scorers:
                        rate = getattr(scorer, "sample_rate", "N/A")
                        print(f"  - {scorer.name} (sample_rate={rate})")

            case "disable":
                from dao_ai.evaluation import stop_monitoring_scorers

                stopped = stop_monitoring_scorers()
                print(f"Disabled monitoring: {len(stopped)} scorers stopped")
                for scorer in stopped:
                    print(f"  - {scorer.name}")

        sys.exit(0)
    except SystemExit:
        raise
    except Exception as e:
        logger.error(f"Monitor command failed: {e}")
        sys.exit(1)


def handle_validate_command(options: Namespace) -> None:
    _apply_profile_context(options.profile)
    logger.debug(f"Validating configuration from {options.config}...")
    try:
        config: AppConfig = AppConfig.from_file(
            options.config, params=_parse_var_args(options.var)
        )
        _ = create_dao_ai_graph(config)
        config.model_dump(by_alias=True)
        sys.exit(0)
    except ConfigVariableError as e:
        _print_config_variable_error(e)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        sys.exit(1)


def _format_schema_pretty(schema: dict[str, Any], indent: int = 0) -> str:
    """
    Format a JSON schema in a more readable, concise format.

    Args:
        schema: The JSON schema to format
        indent: Current indentation level

    Returns:
        Pretty-formatted schema string
    """
    if not schema:
        return ""

    lines: list[str] = []
    indent_str = "  " * indent

    # Get required fields
    required_fields = set(schema.get("required", []))

    # Handle object type with properties
    if schema.get("type") == "object" and "properties" in schema:
        properties = schema["properties"]

        for prop_name, prop_schema in properties.items():
            is_required = prop_name in required_fields
            req_marker = " (required)" if is_required else " (optional)"

            prop_type = prop_schema.get("type", "any")
            prop_desc = prop_schema.get("description", "")

            # Handle different types
            if prop_type == "array":
                items = prop_schema.get("items", {})
                item_type = items.get("type", "any")
                type_str = f"array<{item_type}>"
            elif prop_type == "object":
                type_str = "object"
            else:
                type_str = prop_type

            # Format enum values if present
            if "enum" in prop_schema:
                enum_values = ", ".join(str(v) for v in prop_schema["enum"])
                type_str = f"{type_str} (one of: {enum_values})"

            # Build the line
            line = f"{indent_str}{prop_name}: {type_str}{req_marker}"
            if prop_desc:
                line += f"\n{indent_str}  └─ {prop_desc}"

            lines.append(line)

            # Recursively handle nested objects
            if prop_type == "object" and "properties" in prop_schema:
                nested = _format_schema_pretty(prop_schema, indent + 1)
                if nested:
                    lines.append(nested)

    # Handle simple types without properties
    elif "type" in schema:
        schema_type = schema["type"]
        if schema.get("description"):
            lines.append(f"{indent_str}Type: {schema_type}")
            lines.append(f"{indent_str}└─ {schema['description']}")
        else:
            lines.append(f"{indent_str}Type: {schema_type}")

    return "\n".join(lines)


def handle_list_mcp_tools_command(options: Namespace) -> None:
    """
    List available MCP tools from configuration.

    Shows all MCP servers and their available tools, indicating which
    are included/excluded based on filter configuration.
    """
    _apply_profile_context(options.profile)
    logger.debug(f"Listing MCP tools from configuration: {options.config}")

    try:
        from dao_ai.config import McpFunctionModel
        from dao_ai.tools.mcp import MCPToolInfo, _matches_pattern, list_mcp_tools

        try:
            config: AppConfig = AppConfig.from_file(
                options.config, params=_parse_var_args(options.var)
            )
        except ConfigVariableError as e:
            _print_config_variable_error(e)
            sys.exit(1)

        # Find all MCP tools in configuration
        mcp_tools_config: list[tuple[str, McpFunctionModel]] = []
        if config.tools:
            for tool_name, tool_model in config.tools.items():
                logger.debug(
                    f"Checking tool: {tool_name}, function type: {type(tool_model.function)}"
                )
                if tool_model.function and isinstance(
                    tool_model.function, McpFunctionModel
                ):
                    mcp_tools_config.append((tool_name, tool_model.function))

        if not mcp_tools_config:
            logger.warning("No MCP tools found in configuration")
            print("\n⚠️  No MCP tools configured in this file.")
            print(f"   Configuration: {options.config}")
            print(
                "\nTo add MCP tools, define them in the 'tools' section with 'type: mcp'"
            )
            sys.exit(0)

        # Collect all results first (aggregate before displaying)
        results: list[dict[str, Any]] = []
        for tool_name, mcp_function in mcp_tools_config:
            result = {
                "tool_name": tool_name,
                "mcp_function": mcp_function,
                "error": None,
                "all_tools": [],
                "included_tools": [],
                "excluded_tools": [],
            }

            try:
                logger.info(f"Connecting to MCP server: {mcp_function.mcp_url}")

                # Get all available tools (unfiltered)
                all_tools: list[MCPToolInfo] = list_mcp_tools(
                    mcp_function, apply_filters=False
                )

                # Get filtered tools (what will actually be loaded)
                filtered_tools: list[MCPToolInfo] = list_mcp_tools(
                    mcp_function, apply_filters=True
                )

                included_names = {t.name for t in filtered_tools}

                # Categorize tools
                for tool in sorted(all_tools, key=lambda t: t.name):
                    if tool.name in included_names:
                        result["included_tools"].append(tool)
                    else:
                        # Determine why it was excluded
                        reason = ""
                        if mcp_function.exclude_tools:
                            if _matches_pattern(tool.name, mcp_function.exclude_tools):
                                matching_patterns = [
                                    p
                                    for p in mcp_function.exclude_tools
                                    if _matches_pattern(tool.name, [p])
                                ]
                                reason = f" (matches exclude pattern: {', '.join(matching_patterns)})"
                        if not reason and mcp_function.include_tools:
                            reason = " (not in include list)"
                        result["excluded_tools"].append((tool, reason))

                result["all_tools"] = all_tools

            except KeyboardInterrupt:
                result["error"] = "Connection interrupted by user"
                results.append(result)
                break
            except Exception as e:
                logger.error(f"Failed to list tools from MCP server: {e}")
                result["error"] = str(e)

            results.append(result)

        # Now display all results at once (no logging interleaving)
        print(f"\n{'=' * 80}")
        print("MCP TOOLS DISCOVERY")
        print(f"Configuration: {options.config}")
        print(f"{'=' * 80}\n")

        for result in results:
            tool_name = result["tool_name"]
            mcp_function = result["mcp_function"]

            print(f"📦 Tool: {tool_name}")
            print(f"   Server: {mcp_function.mcp_url}")

            # Show connection type
            if mcp_function.connection:
                print(f"   Connection: UC Connection '{mcp_function.connection.name}'")
            else:
                print(f"   Transport: {mcp_function.transport.value}")

            # Show filters if configured
            if mcp_function.include_tools or mcp_function.exclude_tools:
                print("\n   Filters:")
                if mcp_function.include_tools:
                    print(f"     Include: {', '.join(mcp_function.include_tools)}")
                if mcp_function.exclude_tools:
                    print(f"     Exclude: {', '.join(mcp_function.exclude_tools)}")

            # Check for errors
            if result["error"]:
                print(f"\n   ❌ Error: {result['error']}")
                print("   Could not connect to MCP server")
                if result["error"] != "Connection interrupted by user":
                    print(
                        "   Tip: Verify server URL, authentication, and network connectivity"
                    )
            else:
                all_tools = result["all_tools"]
                included_tools = result["included_tools"]
                excluded_tools = result["excluded_tools"]

                # Show stats based on --apply-filters flag
                if options.apply_filters:
                    # Simplified view: only show filtered tools count
                    print(
                        f"\n   Available Tools: {len(included_tools)} (after filters)"
                    )
                else:
                    # Full view: show all, included, and excluded counts
                    print(f"\n   Available Tools: {len(all_tools)} total")
                    print(f"   ├─ ✓ Included: {len(included_tools)}")
                    print(f"   └─ ✗ Excluded: {len(excluded_tools)}")

                # Show included tools with FULL descriptions and schemas
                if included_tools:
                    if options.apply_filters:
                        print(f"\n   Tools ({len(included_tools)}):")
                    else:
                        print(f"\n   ✓ Included Tools ({len(included_tools)}):")

                    for tool in included_tools:
                        print(f"\n     • {tool.name}")
                        if tool.description:
                            # Show full description (no truncation)
                            print(f"       Description: {tool.description}")
                        if tool.input_schema:
                            # Pretty print schema in readable format
                            print("       Parameters:")
                            pretty_schema = _format_schema_pretty(
                                tool.input_schema, indent=0
                            )
                            if pretty_schema:
                                # Indent the schema for better readability
                                for line in pretty_schema.split("\n"):
                                    print(f"         {line}")
                            else:
                                print("         (none)")

                # Show excluded tools only if NOT applying filters
                if excluded_tools and not options.apply_filters:
                    print(f"\n   ✗ Excluded Tools ({len(excluded_tools)}):")
                    for tool, reason in excluded_tools:
                        print(f"     • {tool.name}{reason}")

            print(f"\n{'-' * 80}\n")

        # Summary
        print(f"{'=' * 80}")
        print(f"Summary: Found {len(mcp_tools_config)} MCP server(s)")
        print(f"{'=' * 80}\n")

        sys.exit(0)

    except FileNotFoundError:
        logger.error(f"Configuration file not found: {options.config}")
        print(f"\n❌ Error: Configuration file not found: {options.config}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to list MCP tools: {e}")
        logger.debug(traceback.format_exc())
        print(f"\n❌ Error: {e}")
        sys.exit(1)


def handle_version_command(options: Namespace) -> None:
    """Display dao-ai version and environment information."""
    import platform
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as pkg_version

    from dao_ai.utils import dao_ai_version, is_published

    print(f"dao-ai {dao_ai_version()}")
    print(f"  Published: {is_published()}")
    print(f"  Python:    {platform.python_version()}")
    print(f"  Platform:  {platform.platform()}")

    deps = [
        "mlflow",
        "langchain-core",
        "langgraph",
        "langchain",
        "databricks-sdk",
        "databricks-langchain",
        "databricks-ai-bridge",
        "pydantic",
    ]
    print("  Dependencies:")
    for dep in deps:
        try:
            v = pkg_version(dep)
            print(f"    {dep}: {v}")
        except PackageNotFoundError:
            print(f"    {dep}: not installed")

    # Databricks auth info
    host = os.environ.get("DATABRICKS_HOST")
    profile = os.environ.get("DATABRICKS_CONFIG_PROFILE")
    if host:
        print(f"  Databricks Host:    {host}")
    if profile:
        print(f"  Databricks Profile: {profile}")
    if not host:
        try:
            from databricks.sdk import WorkspaceClient

            w = WorkspaceClient()
            print(f"  Databricks Host:    {w.config.host}")
            if w.config.auth_type:
                print(f"  Auth Type:          {w.config.auth_type}")
        except Exception:
            pass


def setup_logging(verbosity: int) -> None:
    levels: dict[int, str] = {
        0: "ERROR",
        1: "WARNING",
        2: "INFO",
        3: "DEBUG",
        4: "TRACE",
    }
    level: str = levels.get(verbosity, "TRACE")
    configure_logging(level=level)


def generate_bundle_from_template(config_path: Path, app_name: str) -> Path:
    """
    Generate an app-specific databricks.yaml from databricks.yaml.template.

    This function:
    1. Reads databricks.yaml.template (permanent template file)
    2. Replaces __APP_NAME__ with the actual app name
    3. Writes to databricks.yaml (overwrites if exists)
    4. Returns the path to the generated file

    The generated databricks.yaml is overwritten on each deployment and is not tracked in git.
    The template contains cloud-specific targets (azure, aws, gcp) with appropriate node types.

    Args:
        config_path: Path to the app config file
        app_name: Normalized app name

    Returns:
        Path to the generated databricks.yaml file
    """
    cwd = Path.cwd()
    template_path = cwd / "databricks.yaml.template"
    output_path = cwd / "databricks.yaml"

    if not template_path.exists():
        logger.error(f"Template file {template_path} does not exist.")
        sys.exit(1)

    # Read template
    with open(template_path, "r") as f:
        template_content = f.read()

    # Replace template variables
    bundle_content = template_content.replace("__APP_NAME__", app_name)

    # Write generated databricks.yaml (overwrite if exists)
    with open(output_path, "w") as f:
        f.write(bundle_content)

    logger.info(f"Generated bundle configuration at {output_path} from template")
    return output_path


def run_databricks_command(
    command: list[str],
    profile: Optional[str] = None,
    config: Optional[str] = None,
    target: Optional[str] = None,
    cloud: Optional[str] = None,
    dry_run: bool = False,
    deployment_target: Optional[str] = None,
    development: bool | None = None,
    config_vars: Optional[dict[str, str]] = None,
) -> None:
    """Execute a databricks CLI command with optional profile, target, and cloud.

    Args:
        command: The databricks CLI command to execute (e.g., ["bundle", "deploy"])
        profile: Optional Databricks CLI profile name
        config: Optional path to the configuration file
        target: Optional bundle target name (if not provided, auto-generated from app name and cloud)
        cloud: Optional cloud provider ('azure', 'aws', 'gcp'). Auto-detected if not specified.
        dry_run: If True, print the command without executing
        deployment_target: Optional agent deployment target ('model_serving' or 'apps').
            Passed to the deploy notebook via bundle variable.
        development: Optional tri-state source selection passed to the deploy
            notebook via the ``development`` bundle variable — ``True`` ships
            local dao-ai source/wheel, ``False`` the published PyPI package,
            ``None`` (default) auto-detects from the install type. Emitted as
            ``"true"``/``"false"``/``"auto"``.
        config_vars: Optional ``${param.NAME}`` overrides for the dao-ai config.
            Forwarded to ``AppConfig.from_file`` and to the underlying
            ``databricks bundle`` command as ``--var name=value`` so the
            asset-bundle layer's own ``${var.NAME}`` substitution sees the same
            values when the names overlap.
    """
    config_path = Path(config) if config else None

    if config_path and not config_path.exists():
        logger.error(f"Configuration file {config_path} does not exist.")
        sys.exit(1)

    # Make --profile authoritative for this process: strip any DATABRICKS_*
    # auth env vars that a .env or the shell may have injected, so SDK calls
    # resolve against the profile instead of silently using the wrong host.
    _apply_profile_context(profile)

    try:
        app_config: AppConfig = (
            AppConfig.from_file(config_path, params=config_vars)
            if config_path
            else None
        )
    except ConfigVariableError as e:
        _print_config_variable_error(e)
        sys.exit(1)
    normalized_name: str = normalize_name(app_config.app.name) if app_config else None

    # Auto-detect cloud provider if not specified (used for target selection)
    if not cloud:
        cloud = detect_cloud_provider(profile)
        if cloud:
            logger.info(f"Auto-detected cloud provider: {cloud}")
        else:
            logger.warning("Could not detect cloud provider. Defaulting to 'azure'.")
            cloud = "azure"

    # Generate app-specific bundle from template (overwrites databricks.yaml temporarily)
    if config_path and app_config:
        generate_bundle_from_template(config_path, normalized_name)

    # Use app-specific cloud target: {app_name}-{cloud}
    # This ensures each app has unique deployment identity while supporting cloud-specific settings
    # Can be overridden with explicit --target
    if not target:
        target = f"{normalized_name}-{cloud}"
        logger.info(f"Using app-specific cloud target: {target}")

    # Build databricks command
    # --profile is a global flag, --target is a subcommand flag for `databricks bundle`
    cmd = ["databricks"]
    if profile:
        cmd.extend(["--profile", profile])

    cmd.extend(command)

    # --target must come after the bundle subcommand (it's a subcommand-specific flag)
    if target:
        cmd.extend(["--target", target])

    # Add config_path variable for notebooks
    if config_path and app_config:
        # Calculate relative path from notebooks directory to config file
        config_abs = config_path.resolve()
        cwd = Path.cwd()
        notebooks_dir = cwd / "notebooks"

        try:
            relative_config = config_abs.relative_to(notebooks_dir)
        except ValueError:
            relative_config = Path(os.path.relpath(config_abs, notebooks_dir))

        cmd.append(f'--var="config_path={relative_config}"')

    # Forward dao-ai parameter overrides to the asset-bundle layer too, so
    # ${var.NAME} references inside databricks.yaml resolve from the same
    # input values when the names overlap.
    for key, value in (config_vars or {}).items():
        cmd.append(f'--var="{key}={value}"')

    # Add deployment_target variable for notebooks (hybrid resolution)
    # Priority: CLI arg > config file > default (model_serving)
    resolved_deployment_target: str = "model_serving"
    if deployment_target is not None:
        resolved_deployment_target = deployment_target
        logger.debug(
            f"Using CLI-specified deployment target: {resolved_deployment_target}"
        )
    elif app_config and app_config.app and app_config.app.deployment_target:
        # deployment_target is DeploymentTarget enum (str subclass) or string
        # str() works for both since DeploymentTarget inherits from str
        resolved_deployment_target = str(app_config.app.deployment_target)
        logger.debug(
            f"Using config file deployment target: {resolved_deployment_target}"
        )
    else:
        logger.debug("Using default deployment target: model_serving")

    cmd.append(f'--var="deployment_target={resolved_deployment_target}"')

    # Forward the development tri-state to the deploy notebook via a bundle var.
    # Always emit (mirrors deployment_target) so the notebook widget default
    # never diverges from the CLI intent. None → "auto" (notebook resolves via
    # is_published()), True → "true" (local source/wheel), False → "false" (PyPI).
    resolved_development: str = (
        "auto" if development is None else ("true" if development else "false")
    )
    cmd.append(f'--var="development={resolved_development}"')

    logger.debug(f"Executing command: {' '.join(cmd)}")

    if dry_run:
        logger.info(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        return

    env = os.environ.copy()
    if profile:
        for var in DATABRICKS_AUTH_ENV_VARS:
            env.pop(var, None)

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env,
        )

        for line in iter(process.stdout.readline, ""):
            print(line.rstrip())

        process.wait()

        if process.returncode != 0:
            logger.error(f"Command failed with exit code {process.returncode}")
            sys.exit(1)
        else:
            logger.info("Command executed successfully")

    except FileNotFoundError:
        logger.error("databricks CLI not found. Please install the Databricks CLI.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Command execution failed: {e}")
        sys.exit(1)


def handle_pipeline_command(options: Namespace) -> None:
    logger.debug("Preparing pipeline configuration...")
    profile: Optional[str] = options.profile
    config: Optional[str] = options.config
    target: Optional[str] = options.target
    cloud: Optional[str] = options.cloud
    dry_run: bool = options.dry_run
    deployment_target: Optional[str] = options.deployment_target
    development: bool | None = getattr(options, "development", None)
    config_vars: dict[str, str] = _parse_var_args(options.var)

    if options.deploy:
        logger.info("Deploying DAO AI asset bundle...")
        run_databricks_command(
            ["bundle", "deploy"],
            profile=profile,
            config=config,
            target=target,
            cloud=cloud,
            dry_run=dry_run,
            deployment_target=deployment_target,
            development=development,
            config_vars=config_vars,
        )
    if options.run:
        logger.info("Running DAO AI system with current configuration...")
        run_databricks_command(
            ["bundle", "run", "deploy_job"],
            profile=profile,
            config=config,
            target=target,
            cloud=cloud,
            dry_run=dry_run,
            deployment_target=deployment_target,
            development=development,
            config_vars=config_vars,
        )
    if options.destroy:
        logger.info("Destroying DAO AI system with current configuration...")
        run_databricks_command(
            ["bundle", "destroy", "--auto-approve"],
            profile=profile,
            config=config,
            target=target,
            cloud=cloud,
            dry_run=dry_run,
            deployment_target=deployment_target,
            development=development,
            config_vars=config_vars,
        )
    if not any([options.deploy, options.run, options.destroy]):
        logger.warning("No action specified. Use --deploy, --run or --destroy flags.")


def handle_generate_bundle_command(options: Namespace) -> None:
    logger.debug("Generating bundle...")
    config_path: str = options.config
    output_dir: str = options.output_dir
    overwrite: bool = options.overwrite
    # Resolve the --development/--no-development tri-state to a concrete bool
    # (None -> auto-detect via is_published()) so write_bundle's bool contract
    # matches deploy's source-selection semantics.
    from dao_ai.utils import resolve_use_local_source

    development: bool = resolve_use_local_source(options.development)
    profile: str | None = options.profile

    _apply_profile_context(profile)

    try:
        config: AppConfig = AppConfig.from_file(
            config_path, params=_parse_var_args(options.var), initialize=False
        )
    except ConfigVariableError as e:
        _print_config_variable_error(e)
        sys.exit(1)
    if config.app is None:
        logger.error("Config must have an 'app' section to generate a bundle")
        sys.exit(1)

    # Resolve resources so Genie room tables and warehouses can be discovered
    config._resolve_all_resources()

    from dao_ai.apps.bundle import write_bundle

    write_bundle(
        config, Path(output_dir), overwrite=overwrite, development=development
    )


def handle_generate_mcp_command(options: Namespace) -> None:
    """Emit a Databricks Apps bundle that runs the dao-ai MCP server.

    The emitted server exposes the whole dao-ai agent graph as a single MCP
    tool. Requires ``config.app.name`` (used as both the deployed App name
    and the MCP tool name); ``config.app.description`` is strongly
    recommended (surfaced to MCP clients as the tool description).
    """
    logger.debug("Generating MCP bundle...")
    config_path: str = options.config
    output_dir: str = options.output_dir
    overwrite: bool = options.overwrite
    # Resolve the --development/--no-development tri-state to a concrete bool
    # (None -> auto-detect via is_published()) so write_mcp_bundle's bool
    # contract matches deploy's source-selection semantics.
    from dao_ai.utils import resolve_use_local_source

    development: bool = resolve_use_local_source(options.development)
    profile: str | None = options.profile

    _apply_profile_context(profile)

    try:
        config: AppConfig = AppConfig.from_file(
            config_path, params=_parse_var_args(options.var), initialize=False
        )
    except ConfigVariableError as e:
        _print_config_variable_error(e)
        sys.exit(1)

    # Resolve resources so any Genie / VS lookups can fully bind.
    config._resolve_all_resources()

    from dao_ai.mcp.generate import write_mcp_bundle

    write_mcp_bundle(
        config, Path(output_dir), overwrite=overwrite, development=development
    )


def handle_vars_command(options: Namespace) -> None:
    """Handle ``dao-ai vars list`` -- inspect declared parameters in a config.

    Reads only the top-level ``parameters:`` block so this works even when
    other sections of the YAML are incomplete or fail downstream validation.
    """
    import yaml

    from dao_ai.config_vars import (
        ParameterDeclarationModel,
        WorkspaceVariableError,
        substitute_workspace_refs,
    )

    _apply_profile_context(options.profile)
    cli_vars: dict[str, str] = _parse_var_args(options.var)

    try:
        raw_text: str = Path(options.config).read_text()
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {options.config}")
        sys.exit(1)

    try:
        rendered_text: str = substitute_workspace_refs(raw_text, source=options.config)
    except WorkspaceVariableError as e:
        logger.error(str(e))
        sys.exit(1)

    try:
        raw_dict: dict[str, Any] = yaml.safe_load(rendered_text) or {}
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML in {options.config}: {e}")
        sys.exit(1)

    decl_block: dict[str, Any] = raw_dict.get("parameters", {}) or {}
    try:
        declarations: dict[str, ParameterDeclarationModel] = {
            name: ParameterDeclarationModel(**(spec or {}))
            for name, spec in decl_block.items()
        }
    except Exception as e:
        logger.error(f"Invalid 'parameters:' declaration in {options.config}: {e}")
        sys.exit(1)

    if options.action == "list":
        resolved = resolve_parameters(declarations, cli_vars=cli_vars)

        if not resolved:
            print(
                f"\nNo parameters declared in {options.config}.\n"
                "Add a top-level 'parameters:' block to declare ${param.NAME} inputs.\n"
            )
            sys.exit(0)

        name_w = max(len("NAME"), max(len(p.name) for p in resolved))
        default_w = max(len("DEFAULT"), max(len(p.default or "-") for p in resolved))
        value_w = max(
            len("RESOLVED"),
            max(len(p.value if p.value is not None else "-") for p in resolved),
        )
        source_w = max(len("SOURCE"), max(len(p.source) for p in resolved))

        header = (
            f"{'NAME':<{name_w}}  "
            f"{'REQUIRED':<8}  "
            f"{'DEFAULT':<{default_w}}  "
            f"{'RESOLVED':<{value_w}}  "
            f"{'SOURCE':<{source_w}}  "
            f"DESCRIPTION"
        )
        print()
        print(header)
        print("-" * len(header))
        for p in resolved:
            print(
                f"{p.name:<{name_w}}  "
                f"{('yes' if p.required else 'no'):<8}  "
                f"{(p.default or '-'):<{default_w}}  "
                f"{(p.value if p.value is not None else '-'):<{value_w}}  "
                f"{p.source:<{source_w}}  "
                f"{p.description or ''}"
            )
        print()

        missing = [p.name for p in resolved if p.source == "MISSING"]
        sys.exit(1 if missing else 0)


def main() -> None:
    options: argparse.Namespace = parse_args(sys.argv[1:])
    setup_logging(options.verbose)
    match options.command:
        case "version":
            handle_version_command(options)
        case "schema":
            handle_schema_command(options)
        case "create-experiment":
            handle_create_experiment_command(options)
        case "link-trace-destination":
            handle_link_trace_destination_command(options)
        case "grant-trace-permissions":
            handle_grant_trace_permissions_command(options)
        case "validate":
            handle_validate_command(options)
        case "graph":
            handle_graph_command(options)
        case "pipeline":
            handle_pipeline_command(options)
        case "generate-bundle":
            handle_generate_bundle_command(options)
        case "generate-mcp":
            handle_generate_mcp_command(options)
        case "deploy":
            handle_deploy_command(options)
        case "monitor":
            handle_monitor_command(options)
        case "chat":
            handle_chat_command(options)
        case "list-mcp-tools":
            handle_list_mcp_tools_command(options)
        case "parameters" | "vars":
            handle_vars_command(options)
        case _:
            logger.error(f"Unknown command: {options.command}")
            sys.exit(1)


if __name__ == "__main__":
    main()
