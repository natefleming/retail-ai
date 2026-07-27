"""Runtime log retrieval for the deployed agent.

Two deployment targets, two mechanisms:

- ``apps`` mode shells out to ``databricks apps logs`` (streaming + tail). The
  Databricks Python SDK exposes no Apps logs API, so the CLI is the only way to
  reach the app's ``logz/stream`` websocket.
- ``model_serving`` mode uses the Databricks SDK ``serving_endpoints.logs``,
  which returns a point-in-time snapshot (no streaming).
"""

from __future__ import annotations

import shutil
import subprocess

from loguru import logger


def stream_app_logs(
    *,
    app_name: str,
    lines: int = 200,
    follow: bool = False,
    profile: str | None = None,
) -> int:
    """Fetch or stream Databricks App logs to stdout via the ``databricks`` CLI.

    Args:
        app_name: The workspace App name.
        lines: Number of trailing log lines to fetch (``0`` = all).
        follow: When True, stream continuously until interrupted.
        profile: Optional Databricks CLI profile.

    Returns:
        The CLI process return code.

    Raises:
        RuntimeError: If the ``databricks`` CLI is not on ``PATH``.
    """
    if shutil.which("databricks") is None:
        raise RuntimeError(
            "The `databricks` CLI (>= 1.3.0) is required to fetch Apps logs but "
            "was not found on PATH. Install it or use -m model_serving."
        )
    cmd: list[str] = [
        "databricks",
        "apps",
        "logs",
        app_name,
        "--tail-lines",
        str(lines),
    ]
    if follow:
        cmd.append("--follow")
    if profile:
        cmd.extend(["-p", profile])
    logger.debug(f"Running: {' '.join(cmd)}")
    # Inherit stdout/stderr so --follow streams live and Ctrl-C reaches the child.
    return subprocess.run(cmd).returncode


def fetch_model_serving_logs(*, endpoint_name: str, lines: int = 200) -> str:
    """Return a snapshot of Model Serving endpoint logs (most-recent lines).

    Args:
        endpoint_name: The serving endpoint name.
        lines: Number of trailing lines to return (``<= 0`` = full snapshot).

    Returns:
        The (possibly trimmed) log text.

    Raises:
        RuntimeError: If the endpoint has no served entities yet.
    """
    from dao_ai.providers.databricks import DatabricksProvider

    w = DatabricksProvider().w
    endpoint = w.serving_endpoints.get(endpoint_name)
    entities = endpoint.config.served_entities if endpoint.config else None
    if not entities:
        raise RuntimeError(f"Endpoint {endpoint_name} has no served entities yet")
    served_model_name: str = entities[0].name
    if len(entities) > 1:
        logger.warning(
            "Endpoint {} has {} served entities; Model Serving logs are "
            "per-served-model, so showing logs for {!r} only.",
            endpoint_name,
            len(entities),
            served_model_name,
        )
    text: str = w.serving_endpoints.logs(endpoint_name, served_model_name).logs or ""
    if lines and lines > 0:
        text = "\n".join(text.splitlines()[-lines:])
    return text
