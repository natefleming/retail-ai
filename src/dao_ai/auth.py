"""Shared HTTP auth primitives for invoking Databricks resources from dao-ai.

Two pieces ship here:

- :class:`WorkspaceBearerAuth` — an :class:`httpx.Auth` subclass that
  delegates to ``WorkspaceClient.config.authenticate``. Same pattern as
  ``databricks_openai.utils.clients.BearerAuth``, but without the OAuth-
  strategy gate that breaks PAT-auth workspace clients (the forwarded
  user-token path). Use this any time dao-ai needs to POST to a
  Databricks-hosted HTTP endpoint (Databricks App, Model Serving) and
  wants the workspace client's auth strategy honored as-is.

- :class:`OBONotAvailableError` — raised by ``IsDatabricksResource.workspace_client_from``
  when ``strict=True`` and the resource asked for OBO but no forwarded
  user token is present in the call context. Surfaces misconfigured OBO
  at first-call instead of silently falling back to the service-principal
  identity.
"""

from __future__ import annotations

from typing import Generator, Optional

from databricks.sdk import WorkspaceClient
from httpx import Auth, Request, Response


class WorkspaceBearerAuth(Auth):
    """httpx auth that reads the Authorization header from a WorkspaceClient.

    Equivalent to ``BearerAuth(workspace_client.config.authenticate)`` from
    ``databricks_openai``, but ships in dao-ai so callers can opt out of
    that package's OAuth-strategy gate (``_validate_oauth_for_apps``).
    """

    def __init__(self, workspace_client: WorkspaceClient) -> None:
        self._workspace_client = workspace_client

    def auth_flow(self, request: Request) -> Generator[Request, Response, None]:
        auth_headers = self._workspace_client.config.authenticate()
        request.headers["Authorization"] = auth_headers["Authorization"]
        yield request


class OBONotAvailableError(ValueError):
    """Raised when a resource asked for OBO but no forwarded user token exists.

    Carries ``resource_name`` and ``field`` so the message can pinpoint the
    misconfiguration (e.g. ``apps.translator_app.on_behalf_of_user=true``
    in a config whose calling agent does not have OBO enabled).
    """

    def __init__(
        self,
        *,
        resource_name: Optional[str] = None,
        field: Optional[str] = None,
    ) -> None:
        self.resource_name = resource_name
        self.field = field
        message = (
            "Resource declared on_behalf_of_user=true but no forwarded user "
            "token is available in the call context. The calling agent must "
            "also run with OBO so its forwarded user identity can propagate."
        )
        if resource_name:
            message += f" Offending resource: {resource_name!r}"
        if field:
            message += f" (field: {field})"
        super().__init__(message)
