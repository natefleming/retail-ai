"""Chat UI constants and helpers for Databricks Apps deployments.

Ships the in-repo **dao-ai Console** (a Next.js app under ``dao_ai/apps/chat``)
with the wheel, stages it into a writable working dir on the Databricks Apps
container, and builds it there (the Apps runtime has Node.js + npm pre-installed).
Also defines the env vars used by both the ``generate-agent`` and
``deploy-agent`` paths.

The Console replaces the previously cloned ``e2e-chatbot-app-next`` template:
the source lives in the repo (and the wheel), so no runtime ``git clone`` is
needed and the UI evolves in lockstep with the agent's streaming contract.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

from loguru import logger

# Working-dir name for the staged/built Console (under the app's cwd on the
# Apps container). Kept distinct from the package source dir (``chat``).
CHAT_APP_DIR: str = "dao-ai-console"

_DEFAULT_BACKEND_PORT: int = 8000
_DEFAULT_FRONTEND_PORT: int = 3000
_DEFAULT_PROXY_TIMEOUT: int = 300

# Env-var prefixes stripped from the npm subprocess environment. The Console
# talks to the agent only over HTTP (``API_PROXY`` / ``/v1/*``) and never opens
# a database connection, so DB-binding vars are irrelevant to its build/run;
# stripping them keeps the Node toolchain from picking up ambient credentials.
_DB_ENV_VAR_PREFIXES: tuple[str, ...] = (
    "PG",
    "POSTGRES",
    "DATABASE_URL",
    "DATABRICKS_LAKEBASE",
    "DATABRICKS_DATABASE",
)


def sanitized_npm_env() -> dict[str, str]:
    """Return a copy of ``os.environ`` with DB-binding env vars stripped.

    Used when invoking the Console's npm subprocesses so the Node toolchain
    never inherits database credentials it has no use for.
    """
    stripped = sorted(k for k in os.environ if k.startswith(_DB_ENV_VAR_PREFIXES))
    if stripped:
        logger.debug(
            "Stripping DB-binding env vars from npm subprocess", keys=stripped
        )
    return {
        k: v for k, v in os.environ.items() if not k.startswith(_DB_ENV_VAR_PREFIXES)
    }


class ChatUIBuildError(Exception):
    """Raised when the Console cannot be staged or built (missing source/npm/Node)."""


def _packaged_chat_dir() -> Path:
    """Path to the Console source shipped inside the wheel (``dao_ai/apps/chat``)."""
    return Path(__file__).resolve().parent / "chat"


def _is_built(chat_dir: Path) -> bool:
    """Return True if the Next.js production build output (``.next``) exists."""
    next_dir = chat_dir / ".next"
    return next_dir.is_dir() and any(next_dir.iterdir())


# Never copy build output or dependencies when staging source.
_STAGE_IGNORE = shutil.ignore_patterns(
    "node_modules", ".next", "out", "*.log", ".env", ".env.*"
)


def stage_chat_app(target_parent: Path) -> Path:
    """Copy the packaged Console source into ``target_parent / CHAT_APP_DIR``.

    Idempotent: returns the existing directory if already staged. Only source
    is copied (build output and ``node_modules`` are excluded), so the build
    happens fresh in this writable location rather than in site-packages.
    """
    target_dir = target_parent / CHAT_APP_DIR
    if target_dir.exists():
        return target_dir

    src = _packaged_chat_dir()
    if not src.is_dir():
        raise ChatUIBuildError(
            f"dao-ai Console source not found at {src}. The wheel may have been "
            "built without the UI. Set `app.enable_chat_proxy: false` in your "
            "config to run the agent backend without a UI."
        )

    shutil.copytree(src, target_dir, ignore=_STAGE_IGNORE)
    logger.info("Staged dao-ai Console source", path=str(target_dir))
    return target_dir


def _npm_run(chat_dir: Path, args: list[str], description: str) -> None:
    """Run an npm command inside *chat_dir*. Raises ChatUIBuildError on failure."""
    npm_bin = shutil.which("npm")
    if not npm_bin:
        raise ChatUIBuildError(
            "npm is not installed. Install Node.js >= 18 (which includes npm) "
            "or set `app.enable_chat_proxy: false` in your config."
        )

    logger.info(f"Running npm {description}...", cwd=str(chat_dir))
    result = subprocess.run(
        [npm_bin, *args],
        cwd=chat_dir,
        capture_output=True,
        text=True,
        env=sanitized_npm_env(),
    )
    if result.returncode != 0:
        stderr = result.stderr or result.stdout
        raise ChatUIBuildError(
            f"npm {description} failed (exit code {result.returncode}):\n{stderr}"
        )


def ensure_chat_ui_built(
    target_parent: Path,
    *,
    force_rebuild: bool = False,
) -> Path:
    """Stage (if needed) and build the Console in *target_parent / CHAT_APP_DIR*.

    Returns the path to the built Console directory. Raises ``ChatUIBuildError``
    if the source is missing or npm/Node is unavailable.
    """
    chat_dir = stage_chat_app(target_parent)

    if _is_built(chat_dir) and not force_rebuild:
        logger.info("Console already built, skipping rebuild", path=str(chat_dir))
        return chat_dir

    _npm_run(chat_dir, ["install"], "install")
    _npm_run(chat_dir, ["run", "build"], "build")

    if not _is_built(chat_dir):
        raise ChatUIBuildError(
            f"`next build` completed but no output found in {chat_dir / '.next'}."
        )

    logger.info("Console built successfully", path=str(chat_dir))
    return chat_dir


def chat_ui_env_vars(
    backend_port: int = _DEFAULT_BACKEND_PORT,
    frontend_port: int = _DEFAULT_FRONTEND_PORT,
    timeout_seconds: int = _DEFAULT_PROXY_TIMEOUT,
    ui_config: Optional[dict[str, Any]] = None,
) -> list[dict[str, str]]:
    """Return env-var dicts needed by the AgentServer chat proxy + Console.

    Used by both ``generate_databricks_yaml`` (generate-agent path) and
    ``generate_app_yaml`` (deploy-agent path) to keep them in sync.

    ``CHAT_PROXY_ALLOWED_PATH_PREFIXES`` adds ``/_next/`` to the MLflow
    AgentServer proxy allowlist so Next.js static assets are forwarded to the
    Console frontend (the default allowlist only covers Vite's ``/assets/``).
    ``DAO_AI_UI_CONFIG`` carries the resolved ``AppUIModel`` to the Console.
    """
    env: list[dict[str, str]] = [
        {"name": "API_PROXY", "value": f"http://localhost:{backend_port}/invocations"},
        {"name": "CHAT_APP_PORT", "value": str(frontend_port)},
        {"name": "CHAT_PROXY_TIMEOUT_SECONDS", "value": str(timeout_seconds)},
        {"name": "CHAT_PROXY_ALLOWED_PATH_PREFIXES", "value": "/_next/"},
    ]
    if ui_config:
        env.append({"name": "DAO_AI_UI_CONFIG", "value": json.dumps(ui_config)})
    return env


def resolve_ui_config(
    *,
    app_name: str,
    app_description: Optional[str],
    ui: Optional[Any],
) -> Optional[dict[str, Any]]:
    """Resolve the AppUIModel dict for the Console, defaulting the title and
    subtitle to the deployed app's ``name`` / ``description`` when not set.

    So a Console with no explicit ``ui.title``/``ui.subtitle`` shows the actual
    agent's name and description (in the header and the new-session screen)
    rather than a generic placeholder. ``ui`` is an ``AppUIModel`` (or None).
    """
    cfg: dict[str, Any] = dict(ui.model_dump(mode="json")) if ui is not None else {}
    if not cfg.get("title"):
        cfg["title"] = app_name
    if not cfg.get("subtitle") and app_description:
        cfg["subtitle"] = app_description
    return cfg or None
