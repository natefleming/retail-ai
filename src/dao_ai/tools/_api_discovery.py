"""Wire-shape discovery + resolution for `type: app` and `type: serving_endpoint`.

Two discovery sources, one precedence helper, defensive against every
failure mode so dao-ai bundles can be built and validated even when
target apps/endpoints are not yet running.

- :func:`discover_app_agent_api` — HTTP GET ``<app_url>/agent/info``,
  reads the ``agent_api`` field. Used by ``type: app`` tools.
- :func:`discover_serving_endpoint_api` — SDK
  ``WorkspaceClient.serving_endpoints.get(name).task``, maps to the
  matching OpenAI API contract. Used by ``type: serving_endpoint``
  tools (Model Serving endpoints do NOT expose ``/agent/info``).
- :func:`resolve_api` — precedence helper: explicit user setting >
  lazy-probed value > per-type default. The probe is invoked ONLY when
  the user did not set ``api:`` explicitly.
"""

from __future__ import annotations

from typing import Callable, Literal, NamedTuple, Optional

import httpx
from databricks.sdk import WorkspaceClient
from loguru import logger

ApiContract = Literal["responses", "completions"]
ApiOrigin = Literal["explicit", "discovery", "default"]


class ResolvedApi(NamedTuple):
    """Outcome of :func:`resolve_api`. ``origin`` records which branch
    actually decided the value, so callers can log accurately even when
    discovery happened to return the same value as the per-type default."""

    value: ApiContract
    origin: ApiOrigin


# Map from ``ServingEndpointDetailed.task`` (Databricks SDK field) to the
# OpenAI API contract that matches its wire shape. Only the two values
# dao-ai cares about — anything else (embeddings, unknown future tasks,
# None) returns None and falls back to the per-type default.
_TASK_TO_API: dict[str, ApiContract] = {
    "agent/v1/responses": "responses",  # UC-registered ResponsesAgent (mlflow.agents)
    "llm/v1/chat": "completions",  # FMAPI chat completions / external models
}


def discover_app_agent_api(
    app_url: str,
    workspace_client: WorkspaceClient,
    *,
    timeout_seconds: float = 5.0,
) -> Optional[ApiContract]:
    """GET ``<app_url>/agent/info`` and read the ``agent_api`` field.

    Per MLflow Agent Server (``mlflow/genai/agent_server/server.py``),
    ``agent_api`` is set to ``"responses"`` iff the server's ``agent_type``
    is ``ResponsesAgent``; absent otherwise. There is no ``"completions"``
    value — Chat Completions is implied by the field's absence and is
    selected via the caller's per-type default, not by discovery.

    Args:
        app_url: Base URL of the Databricks App (e.g. ``https://<app>.aws.databricksapps.com``).
        workspace_client: SDK client used to mint the ``Authorization`` header
            (`/agent/info` requires the same OAuth/PAT/OBO auth as inference).
        timeout_seconds: httpx GET timeout. Defaults to 5s.

    Returns:
        ``"responses"`` if the field is present with that value.
        ``None`` for every other state (field absent, unknown value,
        4xx/5xx, network error, non-JSON body, auth failure) — caller
        falls back to per-type default.
    """
    info_url: str = f"{app_url.rstrip('/')}/agent/info"
    try:
        headers = dict(workspace_client.config.authenticate())
    except Exception as exc:
        logger.debug(
            f"discover_app_agent_api: auth failed ({type(exc).__name__}: {exc})"
        )
        return None
    try:
        resp = httpx.get(info_url, headers=headers, timeout=timeout_seconds)
    except httpx.HTTPError as exc:
        logger.debug(
            f"discover_app_agent_api: GET {info_url} failed "
            f"({type(exc).__name__}: {exc})"
        )
        return None
    if resp.status_code != 200:
        logger.debug(f"discover_app_agent_api: GET {info_url} → {resp.status_code}")
        return None
    try:
        body = resp.json()
    except ValueError:
        logger.debug(f"discover_app_agent_api: {info_url} returned non-JSON body")
        return None
    value = (body or {}).get("agent_api")
    if value == "responses":
        return "responses"
    if value is not None:
        logger.debug(
            f"discover_app_agent_api: agent_api={value!r} (unknown); "
            "falling back to default"
        )
    return None


def discover_serving_endpoint_api(
    endpoint_name: str,
    workspace_client: WorkspaceClient,
) -> Optional[ApiContract]:
    """Probe ``WorkspaceClient.serving_endpoints.get(name).task`` and map
    the result to an OpenAI API contract.

    Model Serving endpoints do NOT expose ``/agent/info`` — discovery has
    to use the SDK's ``ServingEndpointDetailed.task`` field instead.

    Args:
        endpoint_name: Name of the Model Serving endpoint.
        workspace_client: SDK client used to fetch the endpoint metadata.

    Returns:
        ``"responses"`` for ``task="agent/v1/responses"`` (UC-registered
        ResponsesAgent). ``"completions"`` for ``task="llm/v1/chat"``
        (FMAPI / external chat models). ``None`` for unknown / future
        tasks, ``None`` task, or SDK errors — caller falls back to
        per-type default.
    """
    try:
        info = workspace_client.serving_endpoints.get(endpoint_name)
    except Exception as exc:
        logger.debug(
            "discover_serving_endpoint_api: "
            f"serving_endpoints.get({endpoint_name!r}) failed "
            f"({type(exc).__name__}: {exc})"
        )
        return None
    task: Optional[str] = getattr(info, "task", None)
    if task is None:
        logger.debug(
            f"discover_serving_endpoint_api: endpoint {endpoint_name!r} "
            "has no task field"
        )
        return None
    mapped: Optional[ApiContract] = _TASK_TO_API.get(task)
    if mapped is None:
        logger.debug(
            f"discover_serving_endpoint_api: task={task!r} (unmapped); "
            "falling back to default"
        )
    return mapped


def resolve_api(
    explicit: Optional[ApiContract],
    discover: Callable[[], Optional[ApiContract]],
    default: ApiContract,
) -> ResolvedApi:
    """Resolve the OpenAI API contract using a fixed precedence order.

    Precedence: explicit user setting > lazy-probed value > per-type default.

    The discovery callable is invoked **only** when ``explicit`` is None.
    This is the critical invariant — when the user has set ``api:``
    explicitly on their tool config, the network probe must never run.

    Returns a :class:`ResolvedApi` carrying the resolved value plus the
    origin that decided it (``"explicit"``, ``"discovery"``, or
    ``"default"``). ``"discovery"`` is reported whenever the probe ran
    and produced a non-None value, even if that value happens to match
    the per-type default.
    """
    if explicit is not None:
        return ResolvedApi(explicit, "explicit")
    discovered = discover()
    if discovered is not None:
        return ResolvedApi(discovered, "discovery")
    return ResolvedApi(default, "default")
