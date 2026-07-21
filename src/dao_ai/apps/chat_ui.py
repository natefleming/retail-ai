"""Chat UI constants and helpers for Databricks Apps deployments.

Provides shared constants, the clone+build logic for the e2e-chatbot-app-next
template, and the env-var definitions used by both the ``generate-agent``
and ``deploy-agent`` paths.

The chat UI is cloned from GitHub and built at runtime on the Databricks Apps
container (which has Node.js 20 + npm + git pre-installed), following the same
pattern as the official Databricks agent templates.
"""

import os
import shutil
import subprocess
from pathlib import Path

from loguru import logger

CHAT_APP_DIR: str = "e2e-chatbot-app-next"
REPO_URL_HTTPS: str = "https://github.com/databricks/app-templates.git"
REPO_URL_SSH: str = "git@github.com:databricks/app-templates.git"

_DEFAULT_BACKEND_PORT: int = 8000
_DEFAULT_FRONTEND_PORT: int = 3000
_DEFAULT_PROXY_TIMEOUT: int = 300

# Env-var prefixes whose presence triggers the chat UI's drizzle migration at
# npm-build time (and its runtime persistence code paths). The bundled
# e2e-chatbot-app-next template's migrator hits a "drizzle" schema that isn't
# auto-created, so the build fails when these are set. Stripping them forces
# the chat UI's isDatabaseAvailable() check to return false, routing into
# ephemeral mode for both build and runtime — the SPA renders without
# persisting chat history.
_DB_ENV_VAR_PREFIXES: tuple[str, ...] = (
    "PG",
    "POSTGRES",
    "DATABASE_URL",
    "DATABRICKS_LAKEBASE",
    "DATABRICKS_DATABASE",
)


def sanitized_npm_env() -> dict[str, str]:
    """Return a copy of os.environ with DB-binding env vars stripped.

    Used when invoking the chat UI's npm subprocesses so the bundled
    e2e-chatbot-app-next template skips its drizzle migration and runs in
    ephemeral mode regardless of whether a postgres resource is bound to the
    Databricks App.
    """
    stripped = sorted(k for k in os.environ if k.startswith(_DB_ENV_VAR_PREFIXES))
    if stripped:
        logger.debug(
            "Stripping DB-binding env vars from npm subprocess",
            keys=stripped,
        )
    return {
        k: v for k, v in os.environ.items() if not k.startswith(_DB_ENV_VAR_PREFIXES)
    }


class ChatUIBuildError(Exception):
    """Raised when the chat UI cannot be built (missing Node.js, npm, or git)."""


def _is_built(chat_dir: Path) -> bool:
    """Return True if the chat UI has already been built."""
    server_dist = chat_dir / "server" / "dist"
    return server_dist.is_dir() and any(server_dist.iterdir())


def _clone_chat_app(target_parent: Path) -> Path:
    """Clone the e2e-chatbot-app-next template into *target_parent*.

    Tries HTTPS first, then SSH. Raises ChatUIBuildError on failure.
    """
    target_dir = target_parent / CHAT_APP_DIR
    if target_dir.exists():
        return target_dir

    git_bin = shutil.which("git")
    if not git_bin:
        raise ChatUIBuildError(
            "git is not installed. Install git or set "
            "`app.enable_chat_proxy: false` in your config."
        )

    temp_dir = target_parent / "temp-app-templates"
    for url in [REPO_URL_HTTPS, REPO_URL_SSH]:
        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "--sparse",
                    url,
                    str(temp_dir),
                ],
                check=True,
                capture_output=True,
            )
            break
        except subprocess.CalledProcessError:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            continue
    else:
        raise ChatUIBuildError(
            "Failed to clone the app-templates repository from GitHub.\n"
            "Ensure you have network access and git is configured.\n"
            "Alternatively, set `app.enable_chat_proxy: false` in your config."
        )

    subprocess.run(
        ["git", "sparse-checkout", "set", CHAT_APP_DIR],
        cwd=temp_dir,
        check=True,
        capture_output=True,
    )
    (temp_dir / CHAT_APP_DIR).rename(target_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)

    logger.info("Cloned chat UI template", path=str(target_dir))
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
    """Clone (if needed) and build the chat UI in *target_parent / CHAT_APP_DIR*.

    Returns the path to the built chat UI directory.  Raises
    ``ChatUIBuildError`` if git, npm, or Node.js is missing.
    """
    chat_dir = _clone_chat_app(target_parent)

    if _is_built(chat_dir) and not force_rebuild:
        logger.info("Chat UI already built, skipping rebuild", path=str(chat_dir))
        return chat_dir

    _npm_run(chat_dir, ["install"], "install")
    _npm_run(chat_dir, ["run", "build"], "build")

    if not _is_built(chat_dir):
        raise ChatUIBuildError(
            f"npm run build completed but no output found in {chat_dir / 'server' / 'dist'}. "
            "Check the e2e-chatbot-app-next template for build issues."
        )

    logger.info("Chat UI built successfully", path=str(chat_dir))
    return chat_dir


def chat_ui_env_vars(
    backend_port: int = _DEFAULT_BACKEND_PORT,
    frontend_port: int = _DEFAULT_FRONTEND_PORT,
    timeout_seconds: int = _DEFAULT_PROXY_TIMEOUT,
) -> list[dict[str, str]]:
    """Return env-var dicts needed by the AgentServer chat proxy.

    Used by both ``generate_databricks_yaml`` (generate-agent path) and
    ``generate_app_yaml`` (deploy-agent path) to keep them in sync.
    """
    return [
        {"name": "API_PROXY", "value": f"http://localhost:{backend_port}/invocations"},
        {"name": "CHAT_APP_PORT", "value": str(frontend_port)},
        {"name": "CHAT_PROXY_TIMEOUT_SECONDS", "value": str(timeout_seconds)},
    ]
