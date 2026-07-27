# PYTHON_ARGCOMPLETE_OK
import argparse
import getpass
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import traceback
from argparse import ArgumentParser, Namespace
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence

from dotenv import find_dotenv, load_dotenv
from loguru import logger

from dao_ai.config import AppConfig

if TYPE_CHECKING:
    from dao_ai.config import McpFunctionModel
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
    # ``dao-ai agent deploy``, MLflow's ``log_model`` validation runs the full agent
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


def _global_parent_parser() -> ArgumentParser:
    """Shared parent parser carrying ``-p/--profile`` and ``-v/--verbose``.

    Passed via ``parents=[_global_parent_parser()]`` to BOTH the top-level
    ``ArgumentParser`` and every subparser / nested-verb parser so that either
    flag is accepted at any level. ``default=argparse.SUPPRESS`` prevents a
    level that *didn't* set the flag from clobbering a value that a higher
    (or lower) level *did* set — the last parse that actually saw the flag
    wins (subcommand beats top-level when both are present). Callers must do
    ``getattr(options, "profile", None)`` / ``getattr(options, "verbose", 0)``
    since SUPPRESS means the attribute may be absent when the flag is omitted
    everywhere.
    """
    p = ArgumentParser(add_help=False)
    p.add_argument(
        "-p",
        "--profile",
        type=str,
        default=argparse.SUPPRESS,
        help="The Databricks CLI profile to use (accepted at any level). When "
        "set, ambient DATABRICKS_* env vars (shell or .env) are cleared for this "
        "process so the profile is authoritative.",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=argparse.SUPPRESS,
        help="Increase verbosity (-v..-vvvv). Accepted at any level.",
    )
    return p


def _add_bundle_common_args(parser: ArgumentParser, *, kind: str) -> None:
    """Add the flags every bundle verb shares: -c/-o/-p/--dry-run/--param.

    Shared across ``generate``/``deploy``/``run``/``destroy`` for all three
    nouns (agent/mcp/workflow) so the config path, staging dir, profile, and
    dry-run switch are spelled identically everywhere. ``kind`` only customizes
    the ``--output-dir`` help text (``<base>/<kind>/<app>``).
    """
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        metavar="FILE",
        help="Path to the dao-ai configuration file",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        metavar="DIR",
        help=f"Directory for the bundle staging dir (default: <base>/{kind}/"
        "<app-name>; <base> is $DAO_AI_BUNDLE_DIR or ./.dao-ai/bundle). deploy/"
        "run/destroy act on this same dir without regenerating it.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the databricks bundle commands without executing them.",
    )
    _add_var_argument(parser)


def _add_bundle_source_args(parser: ArgumentParser) -> None:
    """Add source-selection flags (--overwrite, --development tri-state).

    Only meaningful on the ``generate`` verb — deploy/run/destroy never
    re-stage, so these would be inert there.
    """
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the staging directory (and discard "
        "local hand-edits in a default staging dir).",
    )
    parser.add_argument(
        "--development",
        dest="development",
        default=None,
        action="store_true",
        help="Bundle local dao-ai source/wheel instead of pinning the published "
        "PyPI package. Rebuild the wheel first with 'uv build --wheel'. "
        "Defaults to auto-detect from the install type.",
    )
    parser.add_argument(
        "--no-development",
        dest="development",
        action="store_false",
        help="Force the published PyPI dao-ai package even from a local/editable "
        "install.",
    )


def _add_workflow_target_args(parser: ArgumentParser) -> None:
    """Add workflow-only target/cloud flags used to resolve the bundle target.

    The workflow bundle's target is ``<app>-<cloud>`` (contrast the App
    bundle's fixed ``dev``), so deploy/run/destroy need cloud + optional target
    overrides to address the right target.
    """
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        help="Bundle target name (default: auto-generated from app name and cloud)",
    )
    parser.add_argument(
        "--cloud",
        type=str,
        choices=["azure", "aws", "gcp"],
        help="Cloud provider. Auto-detected from the workspace URL; required only "
        "if detection fails.",
    )


# Convenience input aliases for serving-mode values -> canonical ServingMode
# value. Accepted on the CLI and normalized post-parse so ``options.mode`` is
# always canonical downstream (matches the enum / bundle --var / notebook).
_MODE_ALIASES: dict[str, str] = {
    "ms": "model_serving",
    "model-serving": "model_serving",
}


def _canonical_modes(choices: list[str]) -> list[str]:
    """Expand a canonical mode-choice list with its accepted input aliases."""
    return choices + [a for a, canon in _MODE_ALIASES.items() if canon in choices]


def _normalize_mode(value: str) -> str:
    """Map a mode alias (``ms``, ``model-serving``) to its canonical value."""
    return _MODE_ALIASES.get(value, value)


def _add_mode_argument(parser: ArgumentParser, *, choices: list[str]) -> None:
    """Add the serving-mode flag. Choices are per-verb so an unusable value is
    rejected at parse time (never offered when it cannot succeed). Input aliases
    (``ms``/``model-serving`` -> ``model_serving``) are accepted where the
    canonical value is valid and normalized in :func:`parse_args`.
    """
    accepted: list[str] = _canonical_modes(choices)
    _mode_choices_str = " | ".join(choices)
    _mode_aliases_note = (
        " Aliases accepted for model_serving: ms, model-serving."
        if "model_serving" in choices
        else ""
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=accepted,
        default="apps",
        metavar="{" + ",".join(choices) + "}",
        help=(
            f"Serving target: {_mode_choices_str} (default: apps).{_mode_aliases_note}"
        ),
    )


def _add_noun_verb_parsers(
    subparsers: "argparse._SubParsersAction",
    *,
    noun: str,
    noun_description: str,
    noun_epilog: str,
    generate_description: str,
    generate_epilog: str,
    parents: list[ArgumentParser],
) -> None:
    """Register the up/generate/deploy/run/destroy verb parsers for one noun.

    ``noun`` is ``"agent"``/``"workflow"``. ``up`` is the orchestration verb
    (generate-if-needed → deploy → run). ``generate`` is the pure staging verb
    (writes a bundle, no deploy/run). ``deploy``/``run``/``destroy`` are the
    DAB-faithful granular primitives. The workflow noun additionally gets
    target/cloud flags on ``up``/``generate``/``deploy``/``run``/``destroy``.
    ``parents`` is forwarded to every ``add_parser`` call so the global
    ``-p/--profile`` and ``-v/--verbose`` flags are accepted at each level.
    """
    is_workflow: bool = noun == "workflow"

    parser: ArgumentParser = subparsers.add_parser(
        noun,
        help=f"Manage the {noun} bundle (up | generate | deploy | run | destroy)",
        description=noun_description,
        epilog=noun_epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=parents,
    )
    verbs = parser.add_subparsers(dest="subcommand", required=True)

    # --- up: orchestration verb (generate-if-needed → deploy → run) ----------
    up_parser: ArgumentParser = verbs.add_parser(
        "up",
        help=(
            f"Bring the {noun} up: generate (if needed), deploy, and start it — "
            "the one-command path to a live agent. Use `deploy`/`run` for granular "
            "control."
        ),
        description=(
            f"Bring the {noun} up: generate (if needed), deploy, and start it.\n\n"
            "For bundle modes (apps, mcp): stages the bundle when unstaged, runs\n"
            "`databricks bundle deploy`, links the trace destination, then runs\n"
            "`databricks bundle run <app>`. Use `--direct` to skip the bundle\n"
            "entirely and deploy via the SDK (apps/mcp only).\n\n"
            "For --mode model_serving: registers then deploys the endpoint\n"
            "(serving once READY; `run` is n/a for endpoints)."
        ),
        parents=parents,
    )
    _add_bundle_common_args(up_parser, kind=noun)
    _add_bundle_source_args(up_parser)
    if is_workflow:
        _add_workflow_target_args(up_parser)
    _add_mode_argument(up_parser, choices=["model_serving", "apps", "mcp"])
    up_parser.add_argument(
        "--direct",
        action="store_true",
        default=False,
        help=(
            "Deploy via the SDK directly (no DAB bundle on disk) — fast "
            "iteration path for --mode apps|mcp. Inherently starts the app."
        ),
    )

    # --- generate: pure staging verb (no deploy/run) -------------------------
    generate_parser: ArgumentParser = verbs.add_parser(
        "generate",
        help=f"Generate the {noun} bundle from a config (stage only, no deploy)",
        description=generate_description,
        epilog=generate_epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=parents,
    )
    _add_bundle_common_args(generate_parser, kind=noun)
    _add_bundle_source_args(generate_parser)
    if is_workflow:
        _add_workflow_target_args(generate_parser)
        _add_mode_argument(generate_parser, choices=["apps", "mcp"])
    else:
        _add_mode_argument(generate_parser, choices=["apps", "mcp"])

    for verb, verb_help in (
        (
            "deploy",
            f"Deploy the {noun} bundle — auto-generates if nothing is staged",
        ),
        ("run", f"Run the deployed {noun} bundle"),
        ("destroy", f"Destroy the deployed {noun} bundle"),
    ):
        _deploy_desc = (
            f"Deploy the {noun} bundle to Databricks. When a staged bundle exists "
            f"(written by `dao-ai {noun} generate`) it is deployed in place, "
            f"preserving any hand-edits. When nothing is staged, the bundle is "
            f"generated first automatically. Use `--mode` to select the serving "
            f"target (apps | mcp | model_serving; default: apps). "
            f"Use `dao-ai {noun} up` to deploy AND start in one command."
        )
        verb_parser: ArgumentParser = verbs.add_parser(
            verb,
            help=verb_help,
            description=_deploy_desc
            if verb == "deploy"
            else (
                f"{verb_help}. Acts on the staging dir written by "
                f"`dao-ai {noun} generate` (or -o); errors if nothing is staged."
            ),
            parents=parents,
        )
        _add_bundle_common_args(verb_parser, kind=noun)
        if is_workflow:
            _add_workflow_target_args(verb_parser)
        if verb == "deploy":
            _add_mode_argument(verb_parser, choices=["model_serving", "apps", "mcp"])
            # Source-selection flags: deploy can auto-generate (needs development
            # for the bundle writer) and handles --mode model_serving (needs
            # development for create_agent/deploy_agent). run/destroy act on
            # already-built artifacts so source flags are omitted there.
            _add_bundle_source_args(verb_parser)
        else:
            _add_mode_argument(verb_parser, choices=["apps", "mcp"])


# Env var that overrides the base directory for generated bundles. The
# per-app ``<kind>/<app>`` structure is always appended underneath, so
# per-config isolation is preserved regardless of the base.
_BUNDLE_DIR_ENV_VAR = "DAO_AI_BUNDLE_DIR"
_DEFAULT_BUNDLE_BASE = ".dao-ai/bundle"


def _default_bundle_base() -> Path:
    """Base dir for generated bundles: ``$DAO_AI_BUNDLE_DIR`` or ``.dao-ai/bundle``.

    The built-in default is gitignored. If you point the env var at a path that
    is NOT gitignored, deploys still work — each generated ``databricks.yaml``
    carries an explicit ``sync.include`` for its own source, so App source syncs
    regardless of whether the staging dir is git-ignored.
    """
    return Path(os.environ.get(_BUNDLE_DIR_ENV_VAR) or _DEFAULT_BUNDLE_BASE)


def _default_bundle_dir(kind: str, app_name: str) -> Path:
    """Default per-app bundle staging dir: ``<base>/<kind>/<app>``.

    ``<base>`` is :func:`_default_bundle_base` (``$DAO_AI_BUNDLE_DIR`` or the
    built-in ``.dao-ai/bundle``). Shared namespace for all three generators
    (``kind`` in ``{"workflow", "agent", "mcp"}``) so their output is grouped
    under one root and multiple configs never collide on a shared dir.
    """
    return _default_bundle_base() / kind / normalize_name(app_name)


# Marker file dao-ai drops in a default staging dir after each generate. Its
# mtime is the "last generated" timestamp used to detect later hand-edits.
_STAGING_MARKER = ".dao-ai-generated"


def _config_fingerprint(config: AppConfig, *, development: bool) -> str:
    """Stable hash of the config content that determines the generated bundle.

    Computed from the UNRESOLVED config (``model_dump`` with no network resource
    resolution) plus the resolved ``development`` source-selection flag — the two
    inputs the bundle writer consumes. MUST be taken before
    ``config._resolve_all_resources()`` mutates the config in place, so the
    generate-time stamp and the deploy-time check hash identical inputs.

    Stamped into the staging marker so ``deploy``/``up`` can tell when the source
    config changed since the bundle was last generated (see
    :func:`_staged_config_is_stale`) and re-stage instead of silently shipping a
    stale bundle.
    """
    payload: str = json.dumps(
        {"config": config.model_dump(mode="json"), "development": development},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _write_staging_marker(
    bundle_dir: Path, *, is_default: bool, fingerprint: str = ""
) -> None:
    """Stamp a default staging dir as freshly dao-ai-generated.

    Only marks a dao-ai-owned default dir (never a user ``-o`` dir). Written
    LAST, after the writer, so its mtime post-dates every generated file; a
    later hand-edit then shows an mtime newer than the marker, which
    :func:`_clean_default_staging_dir` uses to protect edits from a re-generate.

    ``fingerprint`` (from :func:`_config_fingerprint`) records the source config
    the bundle was generated from; :func:`_staged_config_is_stale` reads it back
    to detect config drift. App bundles pass it; the workflow bundle omits it
    (empty marker) since ``workflow up`` always re-stages.
    """
    if not is_default or not bundle_dir.exists():
        return
    (bundle_dir / _STAGING_MARKER).write_text(fingerprint)


def _staging_dir_has_local_edits(bundle_dir: Path) -> bool:
    """True if a marked staging dir looks hand-edited since its last generate.

    Missing marker → treat as edited (a legacy or user-created dir we didn't
    stamp). Otherwise any file with an mtime newer than the marker (an edit or a
    newly-added stray file) counts as a local edit.
    """
    marker: Path = bundle_dir / _STAGING_MARKER
    if not marker.exists():
        return True
    marker_mtime: float = marker.stat().st_mtime
    for path in bundle_dir.rglob("*"):
        if path == marker or path.is_dir():
            continue
        if path.stat().st_mtime > marker_mtime:
            return True
    return False


def _staged_config_is_stale(bundle_dir: Path, fingerprint: str) -> bool:
    """True if the staged bundle was generated from a different source config.

    Compares ``fingerprint`` (the current config, from
    :func:`_config_fingerprint`) against the value stamped into the
    ``.dao-ai-generated`` marker at generate time. A missing or empty marker (a
    legacy dir or a workflow bundle) returns False: there is no recorded
    fingerprint to contradict, and :func:`_staging_dir_has_local_edits` already
    guards those dirs from an unwanted regenerate.
    """
    marker: Path = bundle_dir / _STAGING_MARKER
    if not marker.exists():
        return False
    staged: str = marker.read_text().strip()
    if not staged:
        return False
    return staged != fingerprint


def _clean_default_staging_dir(
    bundle_dir: Path, *, is_default: bool, overwrite: bool, noun: str
) -> None:
    """Clear a dao-ai-owned default staging dir before regenerating into it.

    Only touches an ``is_default`` path dao-ai chose under
    :func:`_default_bundle_base` — never a user-supplied ``-o`` dir, which may
    hold hand-edits. Regenerating into a stale dir otherwise produces an
    internally-inconsistent bundle (e.g. a ``--development`` run leaves ``dist/``
    wheels + a dev ``pyproject.toml`` that a later ``--no-development`` run then
    mixes with a published layout). Guarded to only ever remove a path under the
    resolved default base.

    Edit safety: if the dir carries local modifications since the last generate
    (see :func:`_staging_dir_has_local_edits`), refuse to wipe unless
    ``overwrite`` — pointing the user at ``<noun> deploy`` to ship what's on
    disk. Untouched dao-ai output is wiped silently as before.
    """
    if not is_default or not bundle_dir.exists():
        return
    base = _default_bundle_base().resolve()
    resolved = bundle_dir.resolve()
    # safety: only wipe paths strictly under the owned base dir
    if base not in resolved.parents:
        return
    if not overwrite and _staging_dir_has_local_edits(bundle_dir):
        logger.error(
            f"Staging dir {bundle_dir} has local modifications. "
            f"Ship it with `dao-ai {noun} deploy -c <config>`, "
            f"or pass --overwrite to discard the edits and regenerate."
        )
        sys.exit(1)
    shutil.rmtree(bundle_dir)


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
    # One shared parent holds -p/--profile and -v/--verbose with SUPPRESS
    # defaults so that a level which doesn't see the flag never overwrites
    # a value that was written at another level.  See _global_parent_parser().
    _GLOBAL: ArgumentParser = _global_parent_parser()

    parser: ArgumentParser = ArgumentParser(
        prog="dao-ai",
        description="Build and operate multi-agent AI systems on Databricks.",
        epilog="""
Getting started:
  dao-ai agent up      -c config.yaml -p fevm                   # generate + deploy + start — one command to a live agent
  dao-ai agent deploy  -c config.yaml -p fevm                   # push the bundle (auto-generates if unstaged); start with `run`

Bring an agent up (generate → deploy → run):
  dao-ai agent up -c config.yaml -p fevm                       # live agent (Apps, default --mode apps)
  dao-ai agent up -c config.yaml --mode model_serving -p fevm  # deploy to a Model Serving endpoint
  dao-ai agent up -c config.yaml -m ms -p fevm                 # same, using the -m ms alias
  dao-ai agent up -c config.yaml --mode mcp -p fevm            # bring up an MCP server
  dao-ai agent up -c config.yaml --direct -p fevm              # SDK fast-path (no bundle on disk)
  dao-ai -p fevm agent up -c config.yaml                       # -p accepted at the top level too

Granular lifecycle (DAB-style primitives):
  dao-ai agent generate -c config.yaml -p fevm                 # stage the bundle only (inspect / hand-edit)
  dao-ai agent deploy   -c config.yaml -p fevm                 # push resources (bundle deploy)
  dao-ai agent run      -c config.yaml -p fevm                 # start the deployed app (bundle run)
  dao-ai agent destroy  -c config.yaml -p fevm                 # tear it down

Provision infrastructure:
  dao-ai workflow up  -c config.yaml -p fevm                    # provision infra (VS, Lakebase, Genie…) + deploy the agent
  dao-ai workflow run -c config.yaml -p fevm                    # run the staged provisioning job

Trace & experiments:
  dao-ai trace create --name /Shared/team/traces -p fevm        # create (or resolve) the MLflow experiment for traces
  dao-ai trace link   -c config.yaml -p fevm                    # link experiment to its UC trace destination (after deploy, before run)
  dao-ai trace grant  -c config.yaml -p fevm                    # grant a deployed App/MS SP its trace-write permissions

Production monitoring:
  dao-ai monitor scorers enable  -c config.yaml -p fevm         # register + start monitoring scorers
  dao-ai monitor scorers status  -c config.yaml -p fevm         # list active scorers
  dao-ai monitor scorers disable -c config.yaml -p fevm         # stop all monitoring scorers

Inspect & utilities:
  dao-ai mcp tools   -c config.yaml                # MCP tools the agent config sees
  dao-ai mcp inspect --app my-mcp-app              # live MCP server: health + tool list
  dao-ai mcp call    <tool> --app my-mcp-app --args '{...}'   # smoke-test one tool
  dao-ai validate   -c config.yaml                 # validate config syntax + semantics
  dao-ai parameters list -c config.yaml            # show declared parameters + resolved values
  dao-ai schema                                    # dump JSON schema for IDE validation
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[_GLOBAL],
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
        help="Show dao-ai version and build metadata",
        description="Display the dao-ai version along with Python, key dependency versions, and platform details. Side-effect free: no network calls or Databricks auth resolution.",
        parents=[_GLOBAL],
    )

    # Doctor command
    _doctor_parser: ArgumentParser = subparsers.add_parser(
        "doctor",
        help="Show the resolved Databricks environment and connection details",
        description="Resolve and display the Databricks host, profile, and auth type. May make network calls or fail when no credentials are configured.",
        parents=[_GLOBAL],
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
        parents=[_GLOBAL],
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
        parents=[_GLOBAL],
    )
    validation_parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        metavar="FILE",
        help="Path to the model configuration file to validate (default: ./config/model_config.yaml)",
    )

    # Trace noun: `dao-ai trace <create|link|grant>`
    trace_parser: ArgumentParser = subparsers.add_parser(
        "trace",
        help="Manage MLflow experiments and UC trace destinations (create | link | grant)",
        description="Manage MLflow experiments and Unity Catalog trace destinations for deployed agents.",
        epilog="""
Examples:
  # Provision a new experiment (create-if-missing)
  dao-ai trace create --name /Shared/team/traces -p fevm

  # Link experiment to its UC trace destination (run after bundle deploy, before bundle run)
  dao-ai trace link -c config.yaml -p fevm

  # Grant an already-deployed App SP its trace-write permissions retroactively
  dao-ai trace grant -c config.yaml -p fevm

  # Explicit experiment lookup by id
  dao-ai trace create --id 1952423719449237 --output json -p fevm
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[_GLOBAL],
    )
    trace_verbs = trace_parser.add_subparsers(dest="subcommand", required=True)

    # trace create (was: create-experiment)
    trace_create_parser: ArgumentParser = trace_verbs.add_parser(
        "create",
        help="Create (or look up) an MLflow experiment and print its id",
        description="""
Provision or resolve an MLflow experiment on Databricks and print the
resulting id + metadata. Delegates to
``DatabricksProvider.create_experiment`` — same code path used by
``dao-ai agent deploy`` when it resolves ``app.experiment`` from a config.

Pass ``--name`` for create-if-missing behavior (default) or ``--id``
to verify an existing experiment. Exactly one of the two is required.
        """,
        epilog="""
Examples:
  dao-ai trace create --name /Shared/rcg/hardware_store_traces
  dao-ai trace create --name /Shared/team/agent -p fevm
  dao-ai trace create --id 1952423719449237 --output json
  dao-ai trace create --name /Shared/only-if-exists --no-create
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[_GLOBAL],
    )
    _create_exp_ident_group = trace_create_parser.add_mutually_exclusive_group(
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
    trace_create_parser.add_argument(
        "--no-create",
        action="store_true",
        help="With --name: fail instead of creating when the experiment is missing.",
    )
    trace_create_parser.add_argument(
        "-o",
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text).",
    )

    # trace link (was: link-trace-destination)
    trace_link_parser: ArgumentParser = trace_verbs.add_parser(
        "link",
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
     looked up via MlflowClient — matches what dao-ai agent generate
     writes for the auto-declared experiment path.

No-op when ``config.app.trace_location`` is not set.
        """,
        epilog="""
Examples:
  # Typical bundle flow — insert between deploy and run
  databricks bundle deploy --target dev -p fevm
  dao-ai trace link -c config.yaml -p fevm
  databricks bundle run my-app --target dev -p fevm

  # Explicit experiment id
  dao-ai trace link -c config.yaml --experiment-id 1234567890 -p fevm

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
        parents=[_GLOBAL],
    )
    trace_link_parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        metavar="FILE",
        help="Path to the model configuration file (must set app.trace_location).",
    )
    trace_link_parser.add_argument(
        "--experiment-id",
        type=str,
        metavar="ID",
        help="Explicit experiment id (skips resolution from config/bundle name).",
    )
    trace_link_parser.add_argument(
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
    _add_var_argument(trace_link_parser)

    # trace grant (was: grant-trace-permissions)
    trace_grant_parser: ArgumentParser = trace_verbs.add_parser(
        "grant",
        help="Grant an App SP the experiment + UC OTEL table permissions MLflow tracing needs.",
        description="""
Grant the experiment ``CAN_EDIT`` ACL and the UC OTEL trace-table
``USE_CATALOG``/``USE_SCHEMA``/``SELECT``/``MODIFY`` privileges that
MLflow tracing needs at runtime to persist traces into a UC-backed
experiment's OTEL Delta tables.

Standalone counterpart to the grant step that ``dao-ai agent deploy`` /
``dao-ai workflow`` runs automatically inside ``deploy_app_agent`` /
``deploy_model_serving_agent``. Useful for the ``agent generate`` + ``bundle
deploy`` + ``dao-ai trace link`` flow (where no full deploy fires),
or for retroactively fixing grants when an app was deployed by an
identity that lacked GRANT rights.

Idempotent — repeated calls with the same principal + privileges no-op
on the workspace side.

Experiment resolution order matches ``dao-ai trace link``:
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
  dao-ai trace grant -c config.yaml -p fevm

  # Grant a specific SP explicitly (e.g. shared workload identity)
  dao-ai trace grant -c config.yaml --app-sp <uuid> -p fevm
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[_GLOBAL],
    )
    trace_grant_parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        metavar="FILE",
        help="Path to the model configuration file (must set app.trace_location).",
    )
    trace_grant_parser.add_argument(
        "--experiment-id",
        type=str,
        metavar="ID",
        help="Explicit experiment id (skips resolution from config/bundle name).",
    )
    trace_grant_parser.add_argument(
        "--app-sp",
        type=str,
        metavar="CLIENT_ID",
        help=(
            "Service principal client_id (UUID) to grant. When omitted, "
            "auto-resolved via ``apps.get(config.app.name)``."
        ),
    )
    _add_var_argument(trace_grant_parser)

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
        parents=[_GLOBAL],
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

    # --- Bundle nouns: `dao-ai <noun> <up|generate|deploy|run|destroy>` -------
    # Each noun owns the full up/generate/deploy/run/destroy lifecycle as
    # discrete verbs. `up` is the orchestration verb (generate-if-needed →
    # deploy → run). `generate` stages only. `deploy`/`run`/`destroy` are the
    # DAB-faithful granular primitives acting on the already-staged dir.
    _WORKFLOW_DESC = """
Manage the provisioning Workflow — a Databricks Job (Lakeflow Workflow) that
provisions backing infrastructure (schemas, Vector Search, Lakebase, Genie, UC
functions), deploys the agent, and runs evaluation. Emits a self-contained
Databricks Asset Bundle.
"""
    _WORKFLOW_EPILOG = """
Examples:
  # One command: generate → deploy → run the provisioning job
  dao-ai workflow up -c config.yaml -p fevm

  # Generate bundle only, inspect, then deploy and run separately
  dao-ai workflow generate -c config.yaml
  dao-ai workflow deploy   -c config.yaml -p fevm
  dao-ai workflow run      -c config.yaml -p fevm
"""
    _AGENT_DESC = """
Manage the agent bundle — a Databricks App (or MCP server / Model Serving
endpoint). `up` is the one-command path: generate (if needed) → deploy →
start. `generate` writes a bundle only. `deploy`/`run` are the DAB-faithful
granular primitives.
"""
    _AGENT_EPILOG = """
Examples:
  # One command: generate (if needed) → deploy → start the App
  dao-ai agent up -c config.yaml -p fevm

  # One command for an MCP server
  dao-ai agent up -c config.yaml --mode mcp -p fevm

  # Deploy staged bundle only (auto-generates if nothing is staged yet)
  dao-ai agent deploy -c config.yaml -p fevm

  # Deploy to Model Serving endpoint; -m ms is an accepted alias
  dao-ai agent deploy -c config.yaml --mode model_serving -p fevm
  dao-ai agent deploy -c config.yaml -m ms -p fevm

  # SDK fast-path: up via SDK without writing a bundle to disk (apps/mcp only)
  dao-ai agent up -c config.yaml --direct -p fevm

  # Pass -p at the top level (equivalent)
  dao-ai -p fevm agent up -c config.yaml
"""

    _add_noun_verb_parsers(
        subparsers,
        noun="agent",
        noun_description=_AGENT_DESC,
        noun_epilog=_AGENT_EPILOG,
        generate_description=_AGENT_DESC,
        generate_epilog=_AGENT_EPILOG,
        parents=[_GLOBAL],
    )
    _add_noun_verb_parsers(
        subparsers,
        noun="workflow",
        noun_description=_WORKFLOW_DESC,
        noun_epilog=_WORKFLOW_EPILOG,
        generate_description=_WORKFLOW_DESC,
        generate_epilog=_WORKFLOW_EPILOG,
        parents=[_GLOBAL],
    )

    # MCP noun: `dao-ai mcp <tools|inspect|call>` — MCP inspection/test utilities.
    # NOTE: MCP *deployment* is NOT here — it lives on `agent --mode mcp`. This
    # noun groups read-only/test utilities only.
    mcp_parser: ArgumentParser = subparsers.add_parser(
        "mcp",
        help="Inspect and test MCP servers and tools (tools | inspect | call)",
        description="""
Inspect and test Model Context Protocol (MCP) servers and tools.

Two distinct surfaces, told apart by their flags:
  -c/--config  → the MCP tools an agent CONFIG declares (what your agent sees)
  --url/--app  → a LIVE MCP server (what a running server exposes)

To DEPLOY a dao-ai agent as an MCP server, use `agent --mode mcp` — deployment
is intentionally not part of this noun.
        """,
        epilog="""
Examples:
  dao-ai mcp tools   -c config.yaml                # MCP tools the agent config sees
  dao-ai mcp tools   -c config.yaml --apply-filters
  dao-ai mcp inspect --app my-mcp-app -p fevm      # live server: health + tool list
  dao-ai mcp inspect --url https://host/.../mcp
  dao-ai mcp call    ask --app my-mcp-app --args '{"input":"hi"}' -p fevm
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[_GLOBAL],
    )
    mcp_verbs = mcp_parser.add_subparsers(dest="subcommand", required=True)

    # mcp tools (was: top-level `tools`)
    mcp_tools_parser: ArgumentParser = mcp_verbs.add_parser(
        "tools",
        help="List the MCP tools an agent config declares (with filter status)",
        description="""
List the MCP tools declared in a dao-ai config. Shows each tool's name,
description, parameter schema, and include/exclude filter status. Useful for
verifying connectivity and discovering tool names before writing agent configs.

Use --apply-filters to see only the tools that will actually be loaded (hides
excluded tools). Without it, all available tools are shown with filter status.
        """,
        epilog="""
Examples:
  dao-ai mcp tools -c config.yaml                 # list all tools + filter status
  dao-ai mcp tools -c config.yaml --apply-filters # only show tools that pass filters
  dao-ai mcp tools -c config.yaml -p fevm         # use a specific Databricks profile
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[_GLOBAL],
    )
    mcp_tools_parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./config/model_config.yaml",
        required=False,
        metavar="FILE",
        help="Path to the model configuration file (default: ./config/model_config.yaml)",
    )
    mcp_tools_parser.add_argument(
        "--apply-filters",
        action="store_true",
        help="Only show tools that pass include/exclude filters (hide excluded tools)",
    )

    # mcp inspect — connect to a live MCP server and list its health + tools
    mcp_inspect_parser: ArgumentParser = mcp_verbs.add_parser(
        "inspect",
        help="Connect to a live MCP server and show its health + available tools",
        description="""
Connect to a live MCP server and show its health (best-effort /healthz) plus
the tools it exposes. Point at any MCP server with --url, or at a Databricks App
with --app (e.g. a dao-ai agent deployed via `agent --mode mcp`).
        """,
        epilog="""
Examples:
  dao-ai mcp inspect --app my-mcp-app -p fevm
  dao-ai mcp inspect --url https://host/api/2.0/mcp/sql -p fevm
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[_GLOBAL],
    )
    _mcp_target_group = mcp_inspect_parser.add_mutually_exclusive_group(required=True)
    _mcp_target_group.add_argument(
        "--url",
        type=str,
        metavar="URL",
        help="Direct MCP server URL (e.g. https://host/.../mcp).",
    )
    _mcp_target_group.add_argument(
        "--app",
        type=str,
        metavar="NAME",
        help="Databricks App name; its /mcp endpoint is resolved via the SDK.",
    )

    # mcp call — invoke a single tool on a live MCP server
    mcp_call_parser: ArgumentParser = mcp_verbs.add_parser(
        "call",
        help="Invoke a single tool on a live MCP server and print the result",
        description="""
Invoke a single tool on a live MCP server and print its result. Smoke-tests a
deployed MCP server end to end. Target the server with --url or --app.
        """,
        epilog="""
Examples:
  dao-ai mcp call ask --app my-mcp-app --args '{"input":"hello"}' -p fevm
  dao-ai mcp call execute_sql --url https://host/api/2.0/mcp/sql \\
      --args '{"query":"SELECT 1"}' -p fevm
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[_GLOBAL],
    )
    mcp_call_parser.add_argument(
        "tool",
        type=str,
        metavar="TOOL",
        help="Name of the tool to invoke.",
    )
    _mcp_call_target_group = mcp_call_parser.add_mutually_exclusive_group(required=True)
    _mcp_call_target_group.add_argument(
        "--url",
        type=str,
        metavar="URL",
        help="Direct MCP server URL (e.g. https://host/.../mcp).",
    )
    _mcp_call_target_group.add_argument(
        "--app",
        type=str,
        metavar="NAME",
        help="Databricks App name; its /mcp endpoint is resolved via the SDK.",
    )
    mcp_call_parser.add_argument(
        "--args",
        type=str,
        default="{}",
        metavar="JSON",
        help="JSON object of tool arguments (default: '{}').",
    )

    # Monitor command
    monitor_parser: ArgumentParser = subparsers.add_parser(
        "monitor",
        help="Manage production monitoring for the deployed agent (scorers, logs)",
        description="""
Manage production monitoring for the deployed agent.

Sub-groups:
  scorers  Register, inspect, and stop MLflow monitoring scorers that
           continuously evaluate production traces for quality, safety, and
           guideline compliance. Requires app.monitoring in the YAML config.
  logs     Fetch or stream runtime logs for the deployed agent (Databricks
           Apps or Model Serving).
        """,
        epilog="""
Examples:
  dao-ai monitor scorers enable -c config/model_config.yaml    # Register and start monitoring scorers
  dao-ai monitor scorers status -c config/model_config.yaml    # Show active scorers and sample rates
  dao-ai monitor scorers disable -c config/model_config.yaml   # Stop all monitoring scorers
  dao-ai monitor logs -c config/model_config.yaml              # Last 200 lines of app logs
  dao-ai monitor logs -c config/model_config.yaml --follow     # Stream app logs (apps only)
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[_GLOBAL],
    )
    monitor_verbs = monitor_parser.add_subparsers(dest="subcommand", required=True)

    monitor_scorers_parser: ArgumentParser = monitor_verbs.add_parser(
        "scorers",
        help="Register, inspect, or stop MLflow monitoring scorers",
        description="""
Manage MLflow production monitoring scorers for the deployed agent.
Scorers continuously evaluate production traces for quality, safety, and
guideline compliance.

Requires app.monitoring to be configured in the YAML config.
        """,
        epilog="""
Examples:
  dao-ai monitor scorers enable -c config/model_config.yaml    # Register and start monitoring scorers
  dao-ai monitor scorers status -c config/model_config.yaml    # Show active scorers and sample rates
  dao-ai monitor scorers disable -c config/model_config.yaml   # Stop all monitoring scorers
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[_GLOBAL],
    )
    monitor_scorers_parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        metavar="FILE",
        help="Path to the model configuration file",
    )
    monitor_scorers_parser.add_argument(
        "action",
        choices=["enable", "status", "disable"],
        help="Scorers action: enable (register/start scorers), "
        "status (list active scorers), disable (stop all scorers)",
    )

    monitor_logs_parser: ArgumentParser = monitor_verbs.add_parser(
        "logs",
        help="Fetch or stream logs for the deployed agent (apps or model_serving)",
        description="""
Fetch or stream runtime logs for the deployed agent.

Capability matrix:
  apps           snapshot (--lines) AND streaming (--follow)   [via databricks CLI]
  model_serving  snapshot only (--lines); --follow NOT supported [via Databricks SDK]

Provide either -c/--config (to derive the app/endpoint name from the YAML) or
--name (an explicit app/endpoint name), but not both.
        """,
        epilog="""
Examples:
  dao-ai monitor logs -c config/model_config.yaml                    # last 200 lines (apps, default)
  dao-ai monitor logs -c config/model_config.yaml --lines 500        # last 500 lines
  dao-ai monitor logs -c config/model_config.yaml --follow           # stream (apps only)
  dao-ai monitor logs -c config/model_config.yaml -m model_serving   # model serving snapshot
  dao-ai monitor logs --name my-app -p fevm                          # explicit app name
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[_GLOBAL],
    )
    monitor_logs_source = monitor_logs_parser.add_mutually_exclusive_group(
        required=True
    )
    monitor_logs_source.add_argument(
        "-c",
        "--config",
        type=str,
        metavar="FILE",
        help="Path to the model configuration file (derives the app/endpoint name)",
    )
    monitor_logs_source.add_argument(
        "--name",
        type=str,
        metavar="NAME",
        help="Explicit App name (apps) or endpoint name (model_serving), used "
        "literally. Mutually exclusive with --config.",
    )
    _add_mode_argument(monitor_logs_parser, choices=["apps", "model_serving"])
    monitor_logs_parser.add_argument(
        "--lines",
        type=int,
        default=200,
        metavar="N",
        help="Number of trailing log lines to fetch (default: 200; 0 = all, apps only)",
    )
    monitor_logs_parser.add_argument(
        "--follow",
        action="store_true",
        help="Stream logs continuously (apps only; not supported for model_serving)",
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
        parents=[_GLOBAL],
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
        parents=[_GLOBAL],
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

    # Add --param/--var to the non-bundle subcommands (bundle verbs get it from
    # _add_bundle_common_args; trace verbs add it inline above).
    # -p/--profile and -v/--verbose are now global (from _GLOBAL parent) so
    # they no longer need per-subcommand attachment here.
    for sub in (
        validation_parser,
        graph_parser,
        mcp_tools_parser,
        monitor_scorers_parser,
        monitor_logs_parser,
        chat_parser,
        vars_parser,
    ):
        _add_var_argument(sub)

    # Shell completion via argcomplete (a core dep; enable in your shell — see
    # docs/cli-reference.md). Still guarded so a stripped env without argcomplete
    # is harmless. The ``# PYTHON_ARGCOMPLETE_OK`` marker at the top of this file
    # lets ``register-python-argcomplete dao-ai`` discover the entry point.
    try:
        import argcomplete

        argcomplete.autocomplete(parser)
    except ImportError:
        pass

    options = parser.parse_args(args)

    # Normalize the SUPPRESS-defaulted globals so handlers can use plain
    # attribute access.  When neither level saw the flag, the attribute is
    # absent from the namespace; set safe defaults here so downstream code
    # (handlers, setup_logging, _apply_profile_context) never needs getattr.
    if not hasattr(options, "profile"):
        options.profile = None
    if not hasattr(options, "verbose"):
        options.verbose = 0

    # Normalize serving-mode input aliases (ms/model-serving -> model_serving)
    # so options.mode is always the canonical ServingMode value downstream.
    if getattr(options, "mode", None) is not None:
        options.mode = _normalize_mode(options.mode)

    # --direct is only meaningful for the bundle-less SDK path (apps/mcp).
    # model_serving always deploys via the SDK, so pairing the two is a
    # contradiction — reject it here rather than silently ignoring --direct.
    if (
        getattr(options, "direct", False)
        and getattr(options, "mode", None) == "model_serving"
    ):
        parser.error(
            "--direct is not valid with --mode model_serving; "
            "model_serving always deploys via the SDK (no bundle)."
        )

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


def handle_trace_command(options: Namespace) -> None:
    """Dispatch ``dao-ai trace <create|link|grant>``."""
    match options.subcommand:
        case "create":
            handle_create_experiment_command(options)
        case "link":
            handle_link_trace_destination_command(options)
        case "grant":
            handle_grant_trace_permissions_command(options)


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
    bundle-based flow (``agent generate`` + ``bundle deploy`` +
    ``dao-ai trace link``) leaves the App SP without table-write
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

    _grant_trace_writes_to_app_sp(config, experiment_id, sp_override=options.app_sp)


def _grant_trace_writes_to_app_sp(
    config: AppConfig,
    experiment_id: str,
    sp_override: Optional[str],
) -> None:
    """Resolve the App SP and grant it the trace-write privileges.

    Shared implementation for ``dao-ai trace link`` (post-link grant)
    and ``dao-ai trace grant`` (standalone). Respects
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
        None if config.app.trace_location.resolved_table_prefix else experiment_id,
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
    """Resolve the MLflow experiment id for dao-ai trace link.

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
            "--experiment-id explicitly or run `dao-ai trace create` first.",
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
            f"Could not look up bundle-declared experiment: {type(e).__name__}: {e}",
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


def handle_monitor_command(options: Namespace) -> None:
    """Dispatch `dao-ai monitor <scorers|logs>`.

    ``scorers enable|status|disable`` manages MLflow production monitoring
    scorers. ``logs`` fetches or streams runtime logs for the deployed agent
    (Databricks Apps via the ``databricks`` CLI, Model Serving via the SDK).
    """
    if options.subcommand == "logs":
        _handle_monitor_logs(options)
        return

    if options.subcommand != "scorers":
        logger.error(f"Unknown monitor sub-command: {options.subcommand}")
        sys.exit(1)

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


def _handle_monitor_logs(options: Namespace) -> None:
    """Fetch or stream logs for the deployed agent.

    Resolves the app/endpoint name from ``--name`` (used literally) or
    ``--config`` (derived from the YAML), then dispatches on ``--mode``:
    ``apps`` streams via the ``databricks`` CLI; ``model_serving`` returns a
    point-in-time snapshot via the Databricks SDK (``--follow`` unsupported).
    """
    if options.lines < 0:
        logger.error(f"--lines must be >= 0 (got {options.lines}); 0 means all")
        sys.exit(1)

    # Make the profile authoritative before any SDK call (model_serving) and
    # before the CLI subprocess (apps) inherits this process's environment:
    # this strips ambient DATABRICKS_* vars a `.env`/shell may have injected so
    # they can't hijack the child CLI even though we also pass it ``-p``.
    _apply_profile_context(options.profile)

    config: AppConfig | None = None
    if options.config:
        logger.debug(f"Loading configuration from {options.config}...")
        try:
            config = AppConfig.from_file(
                options.config, params=_parse_var_args(options.var)
            )
        except ConfigVariableError as e:
            _print_config_variable_error(e)
            sys.exit(1)
        if not config.app:
            logger.error("app must be configured in the YAML to fetch logs")
            sys.exit(1)

    try:
        from dao_ai.monitoring import fetch_model_serving_logs, stream_app_logs

        if options.mode == "model_serving":
            if options.follow:
                logger.error(
                    "--follow is not supported for mode=model_serving "
                    "(snapshot only); omit --follow for the current snapshot"
                )
                sys.exit(1)
            # endpoint_name defaults to app.name via set_default_endpoint_name;
            # fall back explicitly so a partially-built config still resolves.
            endpoint_name: str = options.name or (
                config.app.endpoint_name or config.app.name
            )
            print(
                fetch_model_serving_logs(
                    endpoint_name=endpoint_name, lines=options.lines
                )
            )
            sys.exit(0)

        # apps mode
        app_name: str = options.name or config.app.app_resource_name
        sys.exit(
            stream_app_logs(
                app_name=app_name,
                lines=options.lines,
                follow=options.follow,
                profile=options.profile,
            )
        )
    except KeyboardInterrupt:
        # Expected way to stop `--follow`; exit cleanly instead of dumping a
        # traceback (KeyboardInterrupt is a BaseException, so it would escape
        # the `except Exception` below and reach the un-guarded main()).
        sys.exit(130)
    except SystemExit:
        raise
    except Exception as e:
        logger.error(f"Monitor logs command failed: {e}")
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


def _handle_mcp_tools(options: Namespace) -> None:
    """
    List available MCP tools from configuration (``dao-ai mcp tools``).

    Shows all MCP servers declared in the config and their available tools,
    indicating which are included/excluded based on filter configuration.
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


def _mcp_function_from_args(options: Namespace) -> "McpFunctionModel":
    """Build an in-memory McpFunctionModel from ``--url`` or ``--app``.

    Used by the live-server verbs (``mcp inspect`` / ``mcp call``). The model's
    ``mcp_url`` property resolves an ``--app`` name to its ``/mcp`` endpoint via
    the SDK, and its inherited auth chain (OBO → SP → PAT → ambient) is reused
    for the connection.
    """
    from dao_ai.config import DatabricksAppModel, McpFunctionModel

    if getattr(options, "app", None):
        return McpFunctionModel(app=DatabricksAppModel(name=options.app))
    return McpFunctionModel(url=options.url)


def _handle_mcp_inspect(options: Namespace) -> None:
    """Connect to a live MCP server and show its health + available tools.

    Accepts either ``--url`` (any MCP server) or ``--app`` (a Databricks App,
    e.g. a dao-ai agent deployed with ``agent --mode mcp``). Health is
    best-effort — arbitrary MCP servers need not expose ``/healthz``.
    """
    _apply_profile_context(options.profile)

    try:
        from dao_ai.tools.mcp import list_mcp_tools

        function = _mcp_function_from_args(options)
        mcp_url: str = function.mcp_url

        print(f"\n{'=' * 80}")
        print("MCP SERVER INSPECTION")
        print(f"Server: {mcp_url}")
        print(f"{'=' * 80}\n")

        # Health is best-effort: only dao-ai MCP servers expose /healthz.
        health_url = f"{mcp_url.rstrip('/').removesuffix('/mcp')}/healthz"
        try:
            import httpx

            resp = httpx.get(health_url, timeout=10.0)
            if resp.status_code == 200:
                print(f"   Health: ✓ {resp.json()}")
            else:
                print(f"   Health: n/a (HTTP {resp.status_code})")
        except Exception:
            print("   Health: n/a (no /healthz endpoint)")

        tools = list_mcp_tools(function, apply_filters=False)
        print(f"\n   Available Tools: {len(tools)}\n")
        for tool in sorted(tools, key=lambda t: t.name):
            print(f"     • {tool.name}")
            if tool.description:
                print(f"       Description: {tool.description}")
            if tool.input_schema:
                print("       Parameters:")
                pretty_schema = _format_schema_pretty(tool.input_schema, indent=0)
                if pretty_schema:
                    for line in pretty_schema.split("\n"):
                        print(f"         {line}")
                else:
                    print("         (none)")

        print(f"\n{'=' * 80}\n")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Failed to inspect MCP server: {e}")
        logger.debug(traceback.format_exc())
        print(f"\n❌ Error: {e}")
        sys.exit(1)


def _handle_mcp_call(options: Namespace) -> None:
    """Invoke a single tool on a live MCP server and print the result.

    Accepts either ``--url`` or ``--app`` plus a ``TOOL`` name and a JSON
    ``--args`` object. Smoke-tests a deployed MCP server end to end.
    """
    _apply_profile_context(options.profile)

    try:
        args: dict[str, Any] = json.loads(options.args)
    except json.JSONDecodeError as e:
        print(f"\n❌ Error: --args must be a JSON object: {e}", file=sys.stderr)
        sys.exit(1)
    if not isinstance(args, dict):
        print("\n❌ Error: --args must be a JSON object (e.g. '{\"q\": \"hi\"}')",
              file=sys.stderr)
        sys.exit(1)

    try:
        from dao_ai.tools.mcp import call_mcp_tool

        function = _mcp_function_from_args(options)
        result: str = call_mcp_tool(function, options.tool, args)
        print(result)
        sys.exit(0)

    except Exception as e:
        logger.error(f"Failed to call MCP tool '{options.tool}': {e}")
        logger.debug(traceback.format_exc())
        print(f"\n❌ Error: {e}")
        sys.exit(1)


def handle_mcp_command(options: Namespace) -> None:
    """Dispatch ``dao-ai mcp <tools|inspect|call>`` to its handler."""
    match options.subcommand:
        case "tools":
            _handle_mcp_tools(options)
        case "inspect":
            _handle_mcp_inspect(options)
        case "call":
            _handle_mcp_call(options)
        case _:
            logger.error(f"Unknown mcp sub-command: {options.subcommand}")
            sys.exit(1)


def handle_version_command(options: Namespace) -> None:
    """Display the dao-ai version and build metadata.

    Intentionally side-effect free: no network calls and no Databricks auth
    resolution, so this command is fast and never fails. Environment and
    connection details live in the ``doctor`` command.
    """
    import platform
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as pkg_version

    from dao_ai.utils import dao_ai_version, is_published

    print(f"dao-ai {dao_ai_version()}")
    print(f"  Published: {is_published()}")
    print(f"  Python:    {platform.python_version()}")
    print(f"  Platform:  {platform.platform()}")

    deps = [
        "databricks-langchain",
        "databricks-sdk",
        "langchain",
        "langgraph",
        "mlflow",
    ]
    print("  Dependencies:")
    for dep in deps:
        try:
            v = pkg_version(dep)
            print(f"    {dep}: {v}")
        except PackageNotFoundError:
            print(f"    {dep}: not installed")


def _reauth_command(error: Exception, profile: Optional[str]) -> str:
    """Return the copy-pasteable ``databricks auth login`` command to fix auth.

    The Databricks SDK auth error embeds a reauth hint like
    ``$ databricks auth login --profile DEFAULT`` inside a wall of config dump.
    Extract just that command so ``doctor`` can print a single actionable line.
    Falls back to a command for the requested/DEFAULT profile if the SDK message
    doesn't include one.
    """
    import re

    match = re.search(r"databricks auth login[^\n.]*", str(error))
    if match:
        return match.group(0).strip()
    return f"databricks auth login --profile {profile or 'DEFAULT'}"


def handle_doctor_command(options: Namespace) -> None:
    """Display the resolved Databricks environment and connection details.

    Unlike ``version``, this command resolves Databricks auth and may make
    network calls, prompt, or fail when no credentials are configured.
    """
    _apply_profile_context(options.profile)

    from databricks.sdk import WorkspaceClient

    print("Databricks environment:")
    if options.profile:
        print(f"  Requested Profile:  {options.profile}")

    # Resolve the config (host/profile/auth type) first, then actively
    # authenticate so we report whether the credentials actually work — not
    # just what was configured. A resolved host with an expired token is the
    # most common failure and looks "fine" until an SDK call is made.
    try:
        w = WorkspaceClient()
        print(f"  Databricks Host:    {w.config.host}")
        if w.config.profile:
            print(f"  Databricks Profile: {w.config.profile}")
        if w.config.auth_type:
            print(f"  Auth Type:          {w.config.auth_type}")
    except Exception as e:
        print("  Authenticated:      no")
        print(f"  To authenticate:    {_reauth_command(e, options.profile)}")
        return

    try:
        w.config.authenticate()
        print("  Authenticated:      yes")
    except Exception as e:
        print("  Authenticated:      no")
        print(f"  To authenticate:    {_reauth_command(e, options.profile)}")


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


def _exec_bundle_command(
    command: list[str],
    *,
    profile: Optional[str],
    target: Optional[str],
    cwd: Path,
    extra_vars: Optional[list[str]] = None,
    dry_run: bool = False,
) -> None:
    """Run ``databricks bundle <command>`` from ``cwd`` and stream its output.

    Shared executor for every dao-ai DAB generator (``agent generate``,
    ``agent generate --mode mcp``, ``workflow generate``). Assembles ``databricks [--profile P]
    <command...> [--target T] [--var ...]``, runs it with ``cwd`` set to the
    staged/generated bundle dir, streams stdout, and ``sys.exit(1)`` on failure.

    Args:
        command: bundle verb, e.g. ``["bundle", "deploy"]``.
        profile: Databricks CLI profile (also strips ambient DATABRICKS_* env).
        target: bundle target name appended as ``--target``.
        cwd: directory to run the command from (the bundle root).
        extra_vars: pre-formatted ``--var="k=v"`` args to append.
        dry_run: print the command instead of executing.
    """
    cmd: list[str] = ["databricks"]
    if profile:
        cmd.extend(["--profile", profile])
    cmd.extend(command)
    if target:
        cmd.extend(["--target", target])
    cmd.extend(extra_vars or [])

    logger.debug(f"Executing command (cwd={cwd}): {' '.join(cmd)}")
    if dry_run:
        logger.info(f"[DRY RUN] Would execute (cwd={cwd}): {' '.join(cmd)}")
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
            cwd=str(cwd),
        )
        for line in iter(process.stdout.readline, ""):
            print(line.rstrip())
        process.wait()
        if process.returncode != 0:
            logger.error(f"Command failed with exit code {process.returncode}")
            sys.exit(1)
        logger.info("Command executed successfully")
    except FileNotFoundError:
        logger.error("databricks CLI not found. Please install the Databricks CLI.")
        sys.exit(1)
    except Exception as e:  # noqa: BLE001
        logger.error(f"Command execution failed: {e}")
        sys.exit(1)


def run_databricks_command(
    command: Optional[list[str]] = None,
    profile: Optional[str] = None,
    config: Optional[str] = None,
    target: Optional[str] = None,
    cloud: Optional[str] = None,
    dry_run: bool = False,
    mode: Optional[str] = None,
    development: bool | None = None,
    config_vars: Optional[dict[str, str]] = None,
    output_dir: Optional[str] = None,
    overwrite: bool = False,
    stage: bool = True,
) -> None:
    """Execute a databricks CLI command with optional profile, target, and cloud.

    Args:
        command: The databricks CLI command to execute (e.g., ["bundle", "deploy"])
        profile: Optional Databricks CLI profile name
        config: Optional path to the configuration file
        target: Optional bundle target name (if not provided, auto-generated from app name and cloud)
        cloud: Optional cloud provider ('azure', 'aws', 'gcp'). Auto-detected
            from the workspace URL; required only if detection fails.
        dry_run: If True, print the command without executing
        mode: Optional agent serving mode ('model_serving', 'apps', or 'mcp').
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
        output_dir: Directory to stage the bundle into. When ``None`` (default),
            a per-app dir ``.dao-ai/bundle/workflow/<app>`` is used so
            deploying multiple configs never collides on a shared staging dir.
        overwrite: Overwrite copied-in *content* (config, data/functions assets)
            in the staging dir. Derived artifacts (databricks.yaml,
            requirements.txt, notebooks) are always regenerated regardless.
        stage: When True (default) write/refresh the bundle in the staging dir
            before running ``command`` — the behavior of ``workflow generate``.
            When False, skip staging entirely and run ``command`` against the
            EXISTING staged dir (the standalone ``workflow deploy/run/destroy``
            verbs); the bundle ``--var`` values are still re-derived from config
            + the already-staged ``dist/`` so the bundle verb has what it needs.
            Errors if nothing is staged there.
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

    # Auto-detect cloud provider if not specified (used for target selection).
    # On a real run, if detection fails, stop and ask the operator to pass
    # --cloud rather than silently guessing a cloud (a wrong guess targets the
    # wrong node types / bundle target and produces a confusing downstream
    # failure). Under --dry-run we still want to render the preview even without
    # workspace auth, so fall back to a visible placeholder instead of exiting.
    if not cloud:
        cloud = detect_cloud_provider(profile)
        if cloud:
            logger.info(f"Auto-detected cloud provider: {cloud}")
        elif dry_run:
            logger.warning(
                "Could not detect the cloud provider from the workspace URL. "
                "A real run requires --cloud {aws|azure|gcp}; using a placeholder "
                "for this dry-run preview."
            )
            cloud = "<cloud>"
        else:
            logger.error(
                "Could not detect the cloud provider from the workspace URL. "
                "Re-run with --cloud {aws|azure|gcp} to specify it explicitly."
            )
            sys.exit(1)

    # Stage a self-contained pipeline bundle from the installed dao-ai wheel's
    # packaged assets (databricks.yaml, notebooks, requirements.txt) plus the
    # resolved config — no source checkout required. `bundle deploy/run/destroy`
    # then run with cwd == the staging dir.
    #
    # When --output-dir is omitted, derive a PER-APP default under the shared
    # `.dao-ai/bundle/workflow/<app>` namespace rather than a single shared dir.
    # The staging dir's contents (bundle name, targets, config) are app-specific,
    # so a shared default would let a second config's deploy fail against the
    # first config's stale databricks.yaml. A per-app default isolates each
    # bundle's DABs state and makes deploying many configs "just work".
    is_default_dir: bool = output_dir is None
    if output_dir is not None:
        staging_dir = Path(output_dir).resolve()
    elif normalized_name:
        staging_dir = _default_bundle_dir("workflow", app_config.app.name).resolve()
    else:
        staging_dir = (_default_bundle_base() / "workflow").resolve()

    config_rel_to_notebooks: str | None = None
    dao_ai_dep: str = "dao-ai"
    if config_path and app_config:
        from dao_ai._extras import (
            format_extras_suffix,
            resolve_required_extras_or_all,
        )
        from dao_ai.pipeline.bundle import write_pipeline_bundle
        from dao_ai.utils import dao_ai_version, resolve_use_local_source

        use_local_source: bool = resolve_use_local_source(development)

        # The provisioning job's serverless env installs dao-ai with exactly
        # the optional-feature extras this config exercises, so provisioning
        # notebooks (ingest, vector-search, deploy) can import feature paths.
        # Published mode pins the exact version for reproducible re-runs
        # (parity with Model Serving + the Apps requirements.txt); the local
        # wheel path (set below when use_local_source) overrides this.
        extras_suffix: str = format_extras_suffix(
            resolve_required_extras_or_all(app_config, target="pipeline")
        )
        dao_ai_dep = f"dao-ai{extras_suffix}=={dao_ai_version()}"

        if stage:
            # Regenerate the owned default dir cleanly so stale dev artifacts
            # (e.g. dist/ wheels) don't linger across dev/published switches.
            _clean_default_staging_dir(
                staging_dir,
                is_default=is_default_dir,
                overwrite=overwrite,
                noun="workflow",
            )
            write_pipeline_bundle(
                app_config,
                staging_dir,
                overwrite=overwrite,
                development=use_local_source,
            )
            _write_staging_marker(staging_dir, is_default=is_default_dir)
        elif not (staging_dir / "databricks.yaml").exists():
            # Standalone deploy/run/destroy on an unstaged dir: nothing to run.
            logger.error(
                f"No staged workflow bundle at {staging_dir}. "
                f"Run `dao-ai workflow generate -c {config}"
                f"{f' -o {output_dir}' if output_dir else ''}` first."
            )
            sys.exit(1)

        # The config is staged at <staging>/config/<name>; notebooks run with
        # CWD == <staging>/notebooks, so the config-path var is deterministic.
        config_filename: str = Path(config_path).name
        config_rel_to_notebooks = f"../config/{config_filename}"

        # The serverless environment installs dao-ai via the ``dao_ai_dep``
        # bundle variable: the staged local wheel in development mode (the
        # notebooks run with CWD == <staging>/notebooks, so ``./dist`` resolves),
        # else the unbounded ``dao-ai`` PyPI spec. write_pipeline_bundle staged
        # exactly one wheel under <staging>/dist in development mode — the
        # no-stage path reads that same already-staged wheel.
        if use_local_source:
            staged_wheels = sorted((staging_dir / "dist").glob("dao_ai-*.whl"))
            if not staged_wheels:
                raise RuntimeError(
                    "Development pipeline bundle has no staged dao-ai wheel "
                    f"under {staging_dir / 'dist'}. "
                    "Run `dao-ai workflow generate --development` first."
                )
            # Bare wheel path — NO ``[extras]`` suffix. ``databricks bundle``
            # treats a serverless-env dependency that looks like a local path as
            # an artifact and *globs* it, so any ``[...]`` (extras) is parsed as
            # a glob character class → "no files match pattern". The extras'
            # backing packages are instead pinned as separate PyPI deps in the
            # env spec (glob-safe) at bundle-generation time — same approach as
            # the Model Serving dev path (``get_installed_packages``). See
            # ``generate_pipeline_databricks_yaml``.
            dao_ai_dep = f"./dist/{staged_wheels[-1].name}"

    # Use app-specific cloud target: {app_name}-{cloud}
    # This ensures each app has unique deployment identity while supporting cloud-specific settings
    # Can be overridden with explicit --target
    if not target:
        target = f"{normalized_name}-{cloud}"
        logger.info(f"Using app-specific cloud target: {target}")

    # Assemble the bundle --var overrides specific to the workflow job.
    extra_vars: list[str] = []
    if config_rel_to_notebooks:
        extra_vars.append(f'--var="config_path={config_rel_to_notebooks}"')

    # Forward dao-ai parameter overrides to the asset-bundle layer too, so
    # ${var.NAME} references inside databricks.yaml resolve from the same
    # input values when the names overlap.
    for key, value in (config_vars or {}).items():
        extra_vars.append(f'--var="{key}={value}"')

    # Add mode variable for notebooks.
    # Priority: CLI arg > default (model_serving)
    resolved_mode: str = mode or "model_serving"
    logger.debug(f"Using serving mode: {resolved_mode}")
    extra_vars.append(f'--var="mode={resolved_mode}"')

    # Forward the development tri-state to the deploy notebook via a bundle var.
    # Always emit (mirrors mode) so the notebook widget default
    # never diverges from the CLI intent. None → "auto" (notebook resolves via
    # is_published()), True → "true" (local source/wheel), False → "false" (PyPI).
    resolved_development: str = (
        "auto" if development is None else ("true" if development else "false")
    )
    extra_vars.append(f'--var="development={resolved_development}"')

    # The serverless environment's dao-ai dependency: staged local wheel
    # (development) or the ``dao-ai`` PyPI spec (published). Installed before
    # each task's notebook runs, so the notebooks carry no install of their own.
    extra_vars.append(f'--var="dao_ai_dep={dao_ai_dep}"')

    # command=None -> stage-only (generate the bundle, don't run a bundle verb).
    # This is what `workflow generate` does (staging only; deploy/run/up drive
    # the bundle verbs).
    if command is None:
        logger.info(f"Workflow bundle staged in {staging_dir}")
        return

    _exec_bundle_command(
        command,
        profile=profile,
        target=target,
        cwd=staging_dir,
        extra_vars=extra_vars,
        dry_run=dry_run,
    )

    # `bundle run` already prints a Run URL; after a deploy (no run) surface the
    # job's workspace URL so the user has a link either way.
    if not dry_run and command[:2] == ["bundle", "deploy"]:
        _print_job_link(
            staging_dir, profile=profile, target=target, extra_vars=extra_vars
        )


def _print_job_link(
    cwd: Path,
    *,
    profile: Optional[str],
    target: Optional[str],
    extra_vars: Optional[list[str]] = None,
) -> None:
    """Print the deployed workflow job's workspace URL via ``bundle summary``.

    Best-effort: `bundle summary --output json` exposes each resource's `url`.
    Passes the same ``--var`` overrides deploy used — summary re-evaluates the
    bundle and errors on unset required vars (e.g. ``config_path``) otherwise.
    Never raises — a failure just skips the link.
    """
    import json as _json

    cmd = ["databricks"]
    if profile:
        cmd.extend(["--profile", profile])
    cmd.extend(["bundle", "summary", "--output", "json"])
    if target:
        cmd.extend(["--target", target])
    cmd.extend(extra_vars or [])
    try:
        out = subprocess.run(
            cmd, cwd=str(cwd), capture_output=True, text=True, timeout=60
        )
        if out.returncode != 0:
            return
        jobs = (_json.loads(out.stdout).get("resources") or {}).get("jobs") or {}
        for _key, job in jobs.items():
            if job.get("url"):
                print(f"\n  Job URL: {job['url']}\n")
                return
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Could not resolve job URL: {e}")


def _link_and_grant_trace(
    config: AppConfig,
    *,
    dry_run: bool,
) -> None:
    """Link the experiment trace destination and grant the App SP, if configured.

    No-op unless ``config.app.trace_location`` is set. Runs the same logic as
    ``dao-ai trace link`` (which also grants) — it MUST run after
    ``bundle deploy`` (the experiment + app exist) and before ``bundle run``
    (the app's own runtime link is rejected on re-deploys with "already contains
    traces", causing silent trace loss). Reuses the existing helpers so there is
    a single implementation of the link + grant steps.
    """
    if not (config.app and config.app.trace_location):
        return

    if dry_run:
        logger.info(
            "[DRY RUN] Would link experiment trace destination and grant the "
            "App SP trace-write privileges (app.trace_location is set)."
        )
        return

    experiment_id: Optional[str] = _resolve_experiment_id_for_link(config, None)
    if experiment_id is None:
        # Diagnostic already printed; don't abort the deploy — the operator can
        # run `dao-ai trace link` manually.
        logger.warning(
            "Could not resolve experiment id for trace linking; skipping. "
            "Run `dao-ai trace link -c <config>` manually."
        )
        return

    from dao_ai.providers.databricks import _link_experiment_trace_location

    try:
        _link_experiment_trace_location(config, experiment_id)
        logger.info(
            f"Linked experiment {experiment_id} to "
            f"{config.app.trace_location.catalog_name}."
            f"{config.app.trace_location.schema_name}"
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            f"Trace-destination link failed ({type(e).__name__}: {e}); "
            "run `dao-ai trace link` manually."
        )
        return

    _grant_trace_writes_to_app_sp(config, experiment_id, sp_override=None)


def deploy_app_bundle(
    config: AppConfig,
    *,
    output_dir: Path,
    deploy: bool,
    run: bool,
    destroy: bool,
    profile: Optional[str],
    dry_run: bool = False,
) -> None:
    """Deploy/run/destroy an already-staged App bundle (agent or MCP).

    Shared driver for the ``up``, ``deploy``, ``run``, and ``destroy`` verbs.
    The App bundle uses a single ``dev`` target and is run by its app-name
    resource key (contrast the workflow job, run by ``deploy_job``). When both
    ``deploy`` and ``run`` are True (as in ``up``), the order is: deploy →
    auto trace-link+grant → run, matching the sequence the docs prescribe.

    Args:
        config: loaded AppConfig (for trace_location + app name).
        output_dir: the staged bundle dir (``bundle`` commands run here).
        deploy / run / destroy: which action(s) to perform.
        profile: Databricks CLI profile.
        dry_run: print instead of executing.
    """
    if config.app is None:
        raise ValueError("Config must have an 'app' section to deploy a bundle.")
    app_name = config.app.name.lower().replace("_", "-")
    target = "dev"

    if destroy:
        _exec_bundle_command(
            ["bundle", "destroy", "--auto-approve"],
            profile=profile,
            target=target,
            cwd=output_dir,
            dry_run=dry_run,
        )
        return

    if deploy:
        _exec_bundle_command(
            ["bundle", "deploy"],
            profile=profile,
            target=target,
            cwd=output_dir,
            dry_run=dry_run,
        )
        # Between deploy and run: link the trace destination + grant the App SP
        # so spans actually persist (otherwise silently dropped on Apps).
        _link_and_grant_trace(config, dry_run=dry_run)

    if run:
        _exec_bundle_command(
            ["bundle", "run", app_name],
            profile=profile,
            target=target,
            cwd=output_dir,
            dry_run=dry_run,
        )

    if (deploy or run) and not dry_run:
        _print_app_link(app_name)


def _print_app_link(app_name: str) -> None:
    """Print the deployed Databricks App's URL for quick access.

    Best-effort: resolves the URL via the Apps API (``bundle summary`` does not
    expose an app URL). Never raises — a lookup failure just skips the link.
    """
    try:
        from databricks.sdk import WorkspaceClient

        url = WorkspaceClient().apps.get(name=app_name).url
        if url:
            print(f"\n  App URL: {url}\n")
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Could not resolve app URL for {app_name}: {e}")


def _print_endpoint_link(endpoint_name: str) -> None:
    """Print the deployed Model Serving endpoint's workspace URL.

    Best-effort: builds ``<host>/ml/endpoints/<name>`` from the resolved
    workspace host. Never raises — a lookup failure just skips the link.
    """
    try:
        from databricks.sdk import WorkspaceClient

        host = (WorkspaceClient().config.host or "").rstrip("/")
        if host:
            print(f"\n  Serving endpoint URL: {host}/ml/endpoints/{endpoint_name}\n")
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Could not resolve endpoint URL for {endpoint_name}: {e}")


def handle_workflow_command(options: Namespace) -> None:
    """Dispatch `dao-ai workflow <up|generate|deploy|run|destroy>`.

    ``up`` orchestrates generate → deploy → run in one command. ``generate``
    stages the bundle only. The standalone ``deploy``/``run``/``destroy`` verbs
    act on the ALREADY-STAGED dir without re-staging (``stage=False``).
    """
    match options.subcommand:
        case "up":
            _handle_up_workflow_command(options)
        case "generate":
            handle_generate_workflow_command(options)
        case "deploy":
            _exec_workflow_verb(options, ["bundle", "deploy"])
        case "run":
            _exec_workflow_verb(options, ["bundle", "run", "deploy_job"])
        case "destroy":
            _exec_workflow_verb(options, ["bundle", "destroy", "--auto-approve"])


def _exec_workflow_verb(options: Namespace, command: list[str]) -> None:
    """Run a bundle verb against an already-staged workflow bundle (no restage)."""
    run_databricks_command(
        command,
        profile=options.profile,
        config=options.config,
        target=options.target,
        cloud=options.cloud,
        dry_run=options.dry_run,
        mode=getattr(options, "mode", None),
        config_vars=_parse_var_args(options.var),
        output_dir=options.output_dir,
        stage=False,
    )


def handle_generate_workflow_command(options: Namespace) -> None:
    """Stage the workflow bundle only; no deploy/run (use `up` or standalone verbs)."""
    logger.debug("Preparing provisioning-workflow configuration...")
    profile: Optional[str] = options.profile
    config: Optional[str] = options.config
    target: Optional[str] = options.target
    cloud: Optional[str] = options.cloud
    dry_run: bool = options.dry_run
    mode: Optional[str] = getattr(options, "mode", None)
    development: bool | None = getattr(options, "development", None)
    config_vars: dict[str, str] = _parse_var_args(options.var)
    output_dir: str | None = getattr(options, "output_dir", None)
    overwrite: bool = getattr(options, "overwrite", False)

    # Generate-only: stage the bundle so the user can inspect it or deploy
    # manually (`cd <output-dir> && databricks bundle deploy ...`). Use
    # `dao-ai workflow up` to generate → deploy → run in one command.
    logger.info("Staging DAO AI workflow bundle...")
    run_databricks_command(
        None,
        profile=profile,
        config=config,
        target=target,
        cloud=cloud,
        dry_run=dry_run,
        mode=mode,
        development=development,
        config_vars=config_vars,
        output_dir=output_dir,
        overwrite=overwrite,
    )


def _handle_up_workflow_command(options: Namespace) -> None:
    """Orchestrate workflow up: generate → deploy → run deploy_job."""
    logger.debug("Bringing workflow up (generate → deploy → run)...")
    profile: Optional[str] = options.profile
    config: Optional[str] = options.config
    target: Optional[str] = options.target
    cloud: Optional[str] = options.cloud
    dry_run: bool = options.dry_run
    mode: Optional[str] = getattr(options, "mode", None)
    development: bool | None = getattr(options, "development", None)
    config_vars: dict[str, str] = _parse_var_args(options.var)
    output_dir: str | None = getattr(options, "output_dir", None)
    overwrite: bool = getattr(options, "overwrite", False)

    logger.info("Deploying DAO AI asset bundle...")
    run_databricks_command(
        ["bundle", "deploy"],
        profile=profile,
        config=config,
        target=target,
        cloud=cloud,
        dry_run=dry_run,
        mode=mode,
        development=development,
        config_vars=config_vars,
        output_dir=output_dir,
        overwrite=overwrite,
    )
    logger.info("Running DAO AI provisioning workflow...")
    run_databricks_command(
        ["bundle", "run", "deploy_job"],
        profile=profile,
        config=config,
        target=target,
        cloud=cloud,
        dry_run=dry_run,
        mode=mode,
        development=development,
        config_vars=config_vars,
        output_dir=output_dir,
        overwrite=overwrite,
    )


def _resolve_bundle_dir(
    kind: str, config: AppConfig, output_dir: str | None
) -> tuple[Path, bool]:
    """Resolve the staging dir for an App bundle, shared by generate + verbs.

    Returns ``(bundle_dir, is_default)``. ``is_default`` is True when the dir
    was chosen by dao-ai (``<base>/<kind>/<app>``) rather than supplied via
    ``-o``. Both ``<noun> generate`` and the standalone ``deploy``/``run``/
    ``destroy`` verbs call this so they always agree on the same directory.
    """
    is_default_dir: bool = output_dir is None
    bundle_dir: Path = (
        Path(output_dir)
        if output_dir is not None
        else _default_bundle_dir(kind, config.app.name)
    ).resolve()
    return bundle_dir, is_default_dir


def _load_app_config(options: Namespace, *, what: str) -> AppConfig:
    """Load the config for a bundle command; exit cleanly if it lacks ``app``.

    ``what`` names the artifact for the error message (e.g. "a bundle",
    "an MCP bundle"). Applies the profile context first so config loading and
    later SDK calls resolve against ``-p``.
    """
    _apply_profile_context(options.profile)
    try:
        config: AppConfig = AppConfig.from_file(
            options.config, params=_parse_var_args(options.var), initialize=False
        )
    except ConfigVariableError as e:
        _print_config_variable_error(e)
        sys.exit(1)
    if config.app is None:
        logger.error(f"Config must have an 'app' section to generate {what}")
        sys.exit(1)
    return config


def _stage_app_bundle(
    config: AppConfig,
    bundle_dir: Path,
    *,
    is_default_dir: bool,
    writer: Callable[..., None],
    development: bool,
    overwrite: bool,
    noun: str,
    fingerprint: str,
) -> None:
    """Write bundle files to *bundle_dir* (clean → write → marker).

    This is the staging-only half of :func:`_generate_app_bundle`.  It does NOT
    run any bundle actions (deploy/run/destroy); callers that want actions invoke
    :func:`_run_bundle_actions` themselves.  Extracted so the ``deploy``
    auto-generate path can stage without risking recursion through
    ``_run_bundle_actions``.

    ``fingerprint`` is the current config's hash (:func:`_config_fingerprint`),
    computed by the caller BEFORE ``config._resolve_all_resources()`` and stamped
    into the marker so a later ``deploy`` can detect source-config drift.
    """
    logger.debug(f"Staging {noun} bundle to {bundle_dir}...")
    # Regenerate the owned default dir from scratch so a prior --development run
    # can't leave stale dist/ + dev pyproject that a published rebuild mixes in.
    # Refuses to wipe a default dir that carries hand-edits unless --overwrite.
    _clean_default_staging_dir(
        bundle_dir, is_default=is_default_dir, overwrite=overwrite, noun=noun
    )
    writer(config, bundle_dir, overwrite=overwrite, development=development)
    _write_staging_marker(
        bundle_dir, is_default=is_default_dir, fingerprint=fingerprint
    )


def _generate_app_bundle(options: Namespace, *, kind: str, writer, what: str) -> None:
    """Stage an App bundle (agent or mcp) — pure staging, no deploy or run.

    ``writer`` is ``write_bundle`` / ``write_mcp_bundle``. ``kind`` selects the
    default staging dir and ``what`` the guard message. Drops the
    ``.dao-ai-generated`` marker after writing. Use ``dao-ai <noun> up`` to
    generate, deploy, and start in one command.
    """
    logger.debug(f"Generating {kind} bundle...")
    # Resolve the --development/--no-development tri-state to a concrete bool
    # (None -> auto-detect via is_published()) so the writer's bool contract
    # matches deploy's source-selection semantics.
    from dao_ai.utils import resolve_use_local_source

    development: bool = resolve_use_local_source(options.development)
    config: AppConfig = _load_app_config(options, what=what)

    # Fingerprint the UNRESOLVED config before _resolve_all_resources() mutates
    # it, so the stamp matches the deploy-time staleness check (which also hashes
    # the unresolved config).
    fingerprint: str = _config_fingerprint(config, development=development)

    # Resolve resources so Genie room tables and warehouses can be discovered.
    config._resolve_all_resources()

    bundle_dir, is_default_dir = _resolve_bundle_dir(kind, config, options.output_dir)
    _stage_app_bundle(
        config,
        bundle_dir,
        is_default_dir=is_default_dir,
        writer=writer,
        development=development,
        overwrite=options.overwrite,
        noun=kind,
        fingerprint=fingerprint,
    )


def _deploy_run_destroy_app_bundle(
    options: Namespace, *, kind: str, deploy: bool, run: bool, destroy: bool
) -> None:
    """Shared driver for deploy/run/destroy/up on an App bundle.

    Called by the ``up`` verb (deploy=True, run=True) and the standalone
    ``deploy``/``run``/``destroy`` verbs. Auto-generates when unstaged and
    ``deploy=True``.

    Routing:

    1. ``--mode model_serving``: call ``config.create_agent`` then
       ``config.deploy_agent(target=MODEL_SERVING)`` — no DAB bundle. ``run``
       is irrelevant for an endpoint (it serves once READY).
    2. ``up --direct`` (or any direct=True): call
       ``config.deploy_agent(target=ServingMode(mode))`` via SDK (no bundle on
       disk). ``mode`` here is apps|mcp — ``--direct`` + ``--mode model_serving``
       is rejected at parse time (see :func:`parse_args`).
    3. Bundle path: if ``databricks.yaml`` is absent AND ``deploy=True``,
       auto-stage via :func:`_stage_app_bundle` (CDK/SAM norm), then deploy.
       If already staged, deploy in-place (preserves hand-edits). ``run`` /
       ``destroy`` still error when unstaged.
    """
    from dao_ai.utils import resolve_use_local_source

    what: str = "a bundle" if kind == "agent" else "an MCP bundle"
    mode: str = getattr(options, "mode", "apps") or "apps"
    direct: bool = getattr(options, "direct", False)

    # --- Route 1: model_serving (always SDK, never a bundle) ---
    if deploy and mode == "model_serving":
        from dao_ai.config import ServingMode

        config: AppConfig = _load_app_config(options, what=what)
        development: bool = resolve_use_local_source(
            getattr(options, "development", None)
        )
        # Model Serving deploys a REGISTERED MLflow model, so log/register the
        # current config first (mirrors the workflow deploy notebook + the former
        # `dao-ai deploy` handler). Without this, deploy_model_serving_agent
        # deploys a stale-or-nonexistent model version.
        config.create_agent(development=development)
        config.deploy_agent(target=ServingMode.MODEL_SERVING, development=development)
        return

    # --- Route 2: --direct (SDK path for apps/mcp; model_serving rejected at parse) ---
    if deploy and direct:
        from dao_ai.config import ServingMode

        config = _load_app_config(options, what=what)
        development = resolve_use_local_source(getattr(options, "development", None))
        # Apps/MCP deploy directly from config + wheel (no MLflow model to
        # register — matches the former `if target != APPS: create_agent()`
        # gate). model_serving is handled by Route 1 above, so mode is apps|mcp here.
        config.deploy_agent(
            target=ServingMode(mode),
            development=development,
        )
        return

    # --- Route 3: bundle path ---
    config = _load_app_config(options, what=what)
    bundle_dir, is_default_dir = _resolve_bundle_dir(kind, config, options.output_dir)
    is_staged: bool = (bundle_dir / "databricks.yaml").exists()

    if not is_staged and not deploy:
        # run/destroy on an unstaged dir: nothing to act on.
        logger.error(
            f"No staged {kind} bundle at {bundle_dir}. "
            f"Run `dao-ai {kind} generate -c {options.config}"
            f"{f' -o {options.output_dir}' if options.output_dir else ''}` first."
        )
        sys.exit(1)

    if deploy:
        development = resolve_use_local_source(options.development)
        overwrite: bool = options.overwrite
        # Fingerprint the UNRESOLVED config (matches the generate-time stamp).
        fingerprint: str = _config_fingerprint(config, development=development)

        # Decide whether to (re)stage. Fresh dir → stage (CDK/SAM auto-generate
        # norm). Already staged → stage only when the source config drifted from
        # the staged fingerprint, and only if the dir carries no hand-edits (or
        # --overwrite). A stale bundle with hand-edits deploys in place with a
        # warning so the user's edits win but they're told source changes aren't
        # reflected. run/destroy never reach here unstaged (guarded above).
        should_stage: bool = not is_staged
        if is_staged and _staged_config_is_stale(bundle_dir, fingerprint):
            # Only a dao-ai-owned default dir is safe to regenerate; a user `-o`
            # dir is never touched. --overwrite forces regen even over hand-edits;
            # otherwise a clean default dir regenerates and an edited one is left
            # in place with a warning so the user's edits win.
            can_regen: bool = is_default_dir and (
                overwrite or not _staging_dir_has_local_edits(bundle_dir)
            )
            if can_regen:
                logger.info(
                    f"Source config changed since {bundle_dir} was generated; "
                    f"regenerating the {kind} bundle before deploy."
                )
                should_stage = True
            else:
                logger.warning(
                    f"Source config changed since {bundle_dir} was generated, but "
                    f"the staging dir has local edits — deploying it as-is. Pass "
                    f"--overwrite (or re-run `dao-ai {kind} generate --overwrite`) "
                    f"to discard the edits and regenerate from the current config."
                )

        if should_stage:
            if not is_staged:
                logger.info(
                    f"No staged {kind} bundle at {bundle_dir}; "
                    f"auto-generating before deploy."
                )
            # Resolve resources so Genie room tables and warehouses can be discovered.
            config._resolve_all_resources()
            writer: Callable[..., None] = _mode_writer(mode)
            _stage_app_bundle(
                config,
                bundle_dir,
                is_default_dir=is_default_dir,
                writer=writer,
                development=development,
                overwrite=overwrite,
                noun=kind,
                fingerprint=fingerprint,
            )

    dry_run: bool = options.dry_run
    if run and not deploy and not destroy and not dry_run:
        _verify_app_deployed_or_exit(config.app.name, kind=kind, config=options.config)

    deploy_app_bundle(
        config,
        output_dir=bundle_dir,
        deploy=deploy,
        run=run,
        destroy=destroy,
        profile=options.profile,
        dry_run=dry_run,
    )


def _verify_app_deployed_or_exit(app_name: str, *, kind: str, config: str) -> None:
    """Best-effort: exit with guidance if the app has never been deployed.

    A ``run`` against a staged-but-undeployed bundle otherwise fails deep inside
    ``databricks bundle run`` with an opaque error. Resolves the app via the
    Apps API; a not-found result means deploy first. Any other lookup failure is
    swallowed — this is a friendliness check, not a gate.
    """
    normalized: str = app_name.lower().replace("_", "-")
    try:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.errors import NotFound

        try:
            WorkspaceClient().apps.get(name=normalized)
        except NotFound:
            logger.error(
                f"App '{normalized}' is not deployed yet. "
                f"Run `dao-ai {kind} up -c {config}` to deploy and run."
            )
            sys.exit(1)
    except ImportError:
        return


def _mode_bundle_kind(mode: str) -> str:
    """Staging-dir kind derived from serving mode.

    ``mcp`` keeps its own dir so apps and mcp bundles never clobber each
    other; every other mode (apps, model_serving) maps to ``"agent"``.
    """
    return "mcp" if mode == "mcp" else "agent"


def _mode_writer(mode: str) -> Callable[..., None]:
    """Bundle writer function for the given serving mode."""
    if mode == "mcp":
        from dao_ai.mcp.generate import write_mcp_bundle

        return write_mcp_bundle
    from dao_ai.apps.bundle import write_bundle

    return write_bundle


def handle_agent_command(options: Namespace) -> None:
    """Dispatch `dao-ai agent <up|generate|deploy|run|destroy>`.

    ``up``           → generate (if needed) → deploy → run (one-command path).
    ``--mode mcp``   → writes the MCP-server bundle (staging dir ``mcp/<app>``)
    ``--mode apps``  → writes the chat-agent bundle  (staging dir ``agent/<app>``)
    ``--mode model_serving`` → only valid for `deploy`/`up`; SDK path only.
    ``--direct``     → only valid for `up`; SDK deploy without a bundle on disk.
    """
    mode: str = getattr(options, "mode", "apps") or "apps"
    kind: str = _mode_bundle_kind(mode)
    writer = _mode_writer(mode)
    what: str = "an MCP bundle" if mode == "mcp" else "a bundle"

    match options.subcommand:
        case "up":
            _deploy_run_destroy_app_bundle(
                options, kind=kind, deploy=True, run=True, destroy=False
            )
        case "generate":
            _generate_app_bundle(options, kind=kind, writer=writer, what=what)
        case "deploy":
            _deploy_run_destroy_app_bundle(
                options, kind=kind, deploy=True, run=False, destroy=False
            )
        case "run":
            _deploy_run_destroy_app_bundle(
                options, kind=kind, deploy=False, run=True, destroy=False
            )
        case "destroy":
            _deploy_run_destroy_app_bundle(
                options, kind=kind, deploy=False, run=False, destroy=True
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
    # parse_args already normalizes profile/verbose (SUPPRESS → None/0).
    setup_logging(options.verbose)

    command: str = options.command
    match command:
        case "version":
            handle_version_command(options)
        case "doctor":
            handle_doctor_command(options)
        case "schema":
            handle_schema_command(options)
        case "trace":
            handle_trace_command(options)
        case "validate":
            handle_validate_command(options)
        case "graph":
            handle_graph_command(options)
        case "agent":
            handle_agent_command(options)
        case "workflow":
            handle_workflow_command(options)
        case "monitor":
            handle_monitor_command(options)
        case "chat":
            handle_chat_command(options)
        case "mcp":
            handle_mcp_command(options)
        case "parameters" | "vars":
            handle_vars_command(options)
        case _:
            logger.error(f"Unknown command: {options.command}")
            sys.exit(1)


if __name__ == "__main__":
    main()
