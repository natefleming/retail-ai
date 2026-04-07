"""Generic REST API tool for making HTTP calls via UC Connection or traditional HTTP."""

import json as json_module
from typing import Any, Callable, Optional
from urllib.parse import urlencode

import requests
from databricks.sdk.service.serving import (
    ExternalFunctionRequestHttpMethod,
    HttpRequestResponse,
)
from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from loguru import logger

from dao_ai.config import AnyVariable, ConnectionModel, value_of
from dao_ai.state import Context
from dao_ai.tools.tracing import ResourceInfo, set_resource_attributes

_METHOD_MAP: dict[str, ExternalFunctionRequestHttpMethod] = {
    "GET": ExternalFunctionRequestHttpMethod.GET,
    "POST": ExternalFunctionRequestHttpMethod.POST,
    "PUT": ExternalFunctionRequestHttpMethod.PUT,
    "PATCH": ExternalFunctionRequestHttpMethod.PATCH,
    "DELETE": ExternalFunctionRequestHttpMethod.DELETE,
}


def _read_response_body(response: HttpRequestResponse) -> str:
    """Read the body text from an HttpRequestResponse."""
    if response.contents:
        return response.contents.read().decode("utf-8")
    return ""


def create_rest_api_tool(
    connection: Optional[ConnectionModel | dict[str, Any]] = None,
    base_url: Optional[AnyVariable] = None,
    auth_token: Optional[AnyVariable] = None,
    default_headers: Optional[dict[str, AnyVariable]] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Callable[..., str]:
    """
    Create a tool that makes REST API calls via a UC Connection or traditional HTTP.

    Exactly one of ``connection`` or ``base_url`` must be provided.

    **UC Connection mode** (``connection``):
        Uses ``serving_endpoints.http_request()`` through a Unity Catalog connection.
        Supports on-behalf-of-user (OBO) authentication via ``workspace_client_from``.

    **Traditional HTTP mode** (``base_url``):
        Uses the ``requests`` library directly. Optional ``auth_token`` is sent as a
        ``Bearer`` token in the ``Authorization`` header. Additional fixed headers
        can be supplied via ``default_headers``.

    Args:
        connection: Unity Catalog connection (ConnectionModel or dict).
            Mutually exclusive with ``base_url``.
        base_url: Base URL for traditional HTTP calls (e.g. ``https://api.example.com/v1``).
            Supports AnyVariable (env vars, secrets). Mutually exclusive with ``connection``.
        auth_token: Bearer token for traditional HTTP mode. Supports AnyVariable.
            Ignored when ``connection`` is provided.
        default_headers: Fixed headers for traditional HTTP mode, keyed by header name.
            Values support AnyVariable. Ignored when ``connection`` is provided.
        name: Custom tool name (default: ``rest_api_call``).
        description: Custom tool description shown to the LLM.

    Returns:
        A tool function that makes REST API calls.

    Raises:
        ValueError: If both or neither of ``connection`` and ``base_url`` are provided.
    """
    logger.debug("Creating REST API tool", name=name)

    if connection is not None and base_url is not None:
        raise ValueError(
            "Provide either 'connection' (UC Connection) or 'base_url' "
            "(traditional HTTP), not both."
        )
    if connection is None and base_url is None:
        raise ValueError(
            "Provide either 'connection' (UC Connection) or 'base_url' "
            "(traditional HTTP)."
        )

    if isinstance(connection, dict):
        connection = ConnectionModel(**connection)

    resolved_base_url: str | None = None
    resolved_headers: dict[str, str] = {}

    if base_url is not None:
        resolved_base_url = str(value_of(base_url))
        if auth_token is not None:
            resolved_headers["Authorization"] = f"Bearer {value_of(auth_token)}"
        if default_headers:
            for header_name, header_value in default_headers.items():
                resolved_headers[header_name] = str(value_of(header_value))

    if name is None:
        name = "rest_api_call"

    if description is None:
        description = (
            "Make an HTTP request to a REST API endpoint. "
            "Specify the HTTP method, path, and optionally a JSON body or query parameters."
        )

    @tool(
        name_or_callable=name,
        description=description,
    )
    def rest_api_call(
        method: str,
        path: str,
        json_body: Optional[dict[str, Any]] = None,
        query_params: Optional[dict[str, str]] = None,
        runtime: ToolRuntime[Context] = None,
    ) -> str:
        """
        Make an HTTP request to a REST API endpoint.

        Args:
            method: HTTP method (GET, POST, PUT, PATCH, DELETE).
            path: Relative path for the API endpoint (e.g. '/users/123').
            json_body: Optional JSON request body as a dictionary.
            query_params: Optional query parameters as a dictionary.
            runtime: Runtime context (automatically injected).

        Returns:
            The response body as a string, or an error message on failure.
        """
        normalized_method: str = method.strip().upper()

        if connection is not None:
            return _call_via_uc_connection(
                connection=connection,
                method=normalized_method,
                path=path,
                json_body=json_body,
                query_params=query_params,
                runtime=runtime,
            )
        else:
            return _call_via_http(
                base_url=resolved_base_url,
                headers=resolved_headers,
                method=normalized_method,
                path=path,
                json_body=json_body,
                query_params=query_params,
            )

    return rest_api_call


def _call_via_uc_connection(
    connection: ConnectionModel,
    method: str,
    path: str,
    json_body: Optional[dict[str, Any]],
    query_params: Optional[dict[str, str]],
    runtime: Optional[ToolRuntime[Context]],
) -> str:
    """Execute an HTTP request through a UC Connection."""
    from databricks.sdk import WorkspaceClient

    set_resource_attributes(
        ResourceInfo("rest_api", connection.on_behalf_of_user, connection.name)
    )

    http_method: ExternalFunctionRequestHttpMethod | None = _METHOD_MAP.get(method)
    if http_method is None:
        return (
            f"Unsupported HTTP method: {method}. Use GET, POST, PUT, PATCH, or DELETE."
        )

    context: Context | None = runtime.context if runtime else None
    workspace_client: WorkspaceClient = connection.workspace_client_from(context)

    json_str: str | None = json_module.dumps(json_body) if json_body else None
    params_str: str | None = urlencode(query_params) if query_params else None

    try:
        response: HttpRequestResponse = workspace_client.serving_endpoints.http_request(
            connection_name=connection.name,
            method=http_method,
            path=path,
            json=json_str,
            params=params_str,
        )
        return _read_response_body(response)
    except Exception as e:
        logger.error(
            "REST API call failed via UC Connection",
            connection=connection.name,
            method=method,
            path=path,
            error=str(e),
        )
        return f"REST API call failed: {e}"


def _call_via_http(
    base_url: str,
    headers: dict[str, str],
    method: str,
    path: str,
    json_body: Optional[dict[str, Any]],
    query_params: Optional[dict[str, str]],
) -> str:
    """Execute an HTTP request using the requests library."""
    set_resource_attributes(ResourceInfo("rest_api", False, base_url))

    url: str = base_url.rstrip("/") + "/" + path.lstrip("/") if path else base_url

    try:
        response: requests.Response = requests.request(
            method=method,
            url=url,
            json=json_body,
            params=query_params,
            headers=headers if headers else None,
            timeout=30,
        )
        return response.text
    except Exception as e:
        logger.error(
            "REST API call failed via HTTP",
            url=url,
            method=method,
            error=str(e),
        )
        return f"REST API call failed: {e}"
