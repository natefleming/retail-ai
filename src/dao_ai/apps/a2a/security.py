"""Convenience constants and factories for A2A ``security_schemes``.

A2A Agent Cards advertise authentication requirements through a
``securitySchemes`` dict shaped after OpenAPI 3. dao-ai types
``A2AModel.security_schemes`` against a2a-sdk's
:class:`a2a.types.SecurityScheme` discriminated union so malformed schemes
fail at config load. This module ships a validated, copy-paste-friendly
library of named schemes for the most common Databricks-flavored cases.

Three workspace-agnostic bearer constants are usable as-is:

* :data:`BEARER_DATABRICKS_PAT` — advertises a Databricks PAT.
* :data:`BEARER_DATABRICKS_M2M` — advertises a Databricks OAuth M2M
  (service-principal) token.
* :data:`BEARER_DATABRICKS_OBO` — bearer scheme whose description tells
  callers that the Databricks Apps proxy forwards the user token via
  ``x-forwarded-access-token``.

Workspace-specific factories take an optional ``host`` argument; when
omitted, the host is resolved from ``$DATABRICKS_HOST`` (auto-set in the
Databricks Apps runtime) or the ambient ``WorkspaceClient`` config:

* :func:`api_key_header` — generic header-bound API key.
* :func:`oauth2_databricks_authorization_code` — three-legged OAuth2 with
  ``oidc/v1/authorize`` + ``oidc/v1/token``.
* :func:`oauth2_databricks_client_credentials` — OAuth2 client-credentials
  flow against ``oidc/v1/token``.
* :func:`oauth2_databricks_obo` — three-legged OAuth2 with a
  ``user_impersonation`` scope hint, suitable for advertising OBO.
* :func:`openid_connect_databricks` — OIDC discovery pointing at
  ``{host}/oidc/.well-known/openid-configuration``.

In YAML, the same workspace-agnostic schemes can be authored inline with
dao-ai's ``${workspace.host}`` substitution (resolved at config load):

.. code-block:: yaml

    a2a:
      security_schemes:
        oauth2:
          type: oauth2
          flows:
            authorizationCode:
              authorizationUrl: ${workspace.host}/oidc/v1/authorize
              tokenUrl: ${workspace.host}/oidc/v1/token
              scopes:
                user_impersonation: Act on behalf of the calling user.

Module-level construction of every constant doubles as an import-time
smoke test against a2a-sdk's typed validators.
"""

from __future__ import annotations

import os
from typing import Optional

from a2a.types import (
    APIKeySecurityScheme,
    AuthorizationCodeOAuthFlow,
    ClientCredentialsOAuthFlow,
    HTTPAuthSecurityScheme,
    In,
    OAuth2SecurityScheme,
    OAuthFlows,
    OpenIdConnectSecurityScheme,
    SecurityScheme,
)

from dao_ai.utils import normalize_host

__all__ = [
    # Re-exports for convenience.
    "APIKeySecurityScheme",
    "HTTPAuthSecurityScheme",
    "OAuth2SecurityScheme",
    "OAuthFlows",
    "OpenIdConnectSecurityScheme",
    "SecurityScheme",
    # Constants.
    "BEARER_DATABRICKS_PAT",
    "BEARER_DATABRICKS_M2M",
    "BEARER_DATABRICKS_OBO",
    # Factories.
    "api_key_header",
    "oauth2_databricks_authorization_code",
    "oauth2_databricks_client_credentials",
    "oauth2_databricks_obo",
    "openid_connect_databricks",
]


_OBO_DESCRIPTION = (
    "Databricks Apps forwards the calling user's bearer token to dao-ai "
    "via the ``x-forwarded-access-token`` header. dao-ai routes OBO-tagged "
    "resources through that token; M2M resources stay on the app SP."
)


# ---------------------------------------------------------------------------
# Workspace-agnostic bearer schemes
# ---------------------------------------------------------------------------

BEARER_DATABRICKS_PAT: HTTPAuthSecurityScheme = HTTPAuthSecurityScheme(
    scheme="bearer",
    bearer_format="Databricks PAT",
    description=(
        "Bearer token issued by a Databricks workspace user "
        "(``databricks tokens create`` or workspace UI)."
    ),
)
"""Bearer scheme for Databricks Personal Access Tokens."""

BEARER_DATABRICKS_M2M: HTTPAuthSecurityScheme = HTTPAuthSecurityScheme(
    scheme="bearer",
    bearer_format="Databricks OAuth M2M",
    description=(
        "Bearer token minted via OAuth M2M client-credentials for a Databricks "
        "service principal (``databricks auth token --profile <m2m-profile>``)."
    ),
)
"""Bearer scheme for Databricks OAuth M2M (service-principal) tokens."""

BEARER_DATABRICKS_OBO: HTTPAuthSecurityScheme = HTTPAuthSecurityScheme(
    scheme="bearer",
    bearer_format="Databricks OAuth (forwarded by Apps proxy via x-forwarded-access-token; OBO supported)",
    description=_OBO_DESCRIPTION,
)
"""Bearer scheme advertising On-Behalf-Of-User support via the Apps proxy."""


# ---------------------------------------------------------------------------
# Workspace-specific factories
# ---------------------------------------------------------------------------


def _resolve_host(host: Optional[str]) -> str:
    """Resolve a Databricks workspace host.

    Order — prefer the authoritative source over the raw env var:

      1. ``host`` argument (whitespace-trimmed, normalized to https://).
      2. ``WorkspaceClient().config.host`` — the ambient Databricks SDK
         client. Honors ``DATABRICKS_CONFIG_PROFILE``, ``~/.databrickscfg``,
         and the Apps runtime's injected credentials. This is the
         authoritative source whenever an ambient client is constructible.
      3. ``$DATABRICKS_HOST`` env var as a last-ditch fallback for
         environments where the SDK cannot be initialized.

    YAML configs should NOT call this — they should use dao-ai's
    ``${workspace.host}`` template, which is resolved at
    :meth:`dao_ai.config.AppConfig.from_file` load time.
    """
    if host:
        normalized = normalize_host(host)
        if normalized:
            return normalized.rstrip("/")

    try:
        from databricks.sdk import WorkspaceClient

        ws_host = WorkspaceClient().config.host
        if ws_host:
            normalized = normalize_host(ws_host)
            if normalized:
                return normalized.rstrip("/")
    except Exception as exc:  # pragma: no cover — covered via mocking in tests
        from loguru import logger

        logger.trace(
            "Ambient WorkspaceClient unavailable in _resolve_host; "
            "falling back to $DATABRICKS_HOST. {err}",
            err=exc,
        )

    env_host = os.environ.get("DATABRICKS_HOST")
    if env_host:
        normalized = normalize_host(env_host)
        if normalized:
            return normalized.rstrip("/")

    raise ValueError(
        "Workspace host could not be resolved. Pass `host=` explicitly, "
        "run with an ambient Databricks profile, or set $DATABRICKS_HOST."
    )


def api_key_header(
    name: str, *, description: Optional[str] = None
) -> APIKeySecurityScheme:
    """Generic API-key-in-header scheme.

    YAML equivalent::

        a2a:
          security_schemes:
            x_api_key:
              type: apiKey
              in: header
              name: X-API-Key
    """
    return APIKeySecurityScheme(
        type="apiKey",
        name=name,
        in_=In.header,
        description=description or f"API key in the ``{name}`` header.",
    )


def oauth2_databricks_authorization_code(
    host: Optional[str] = None,
    *,
    scopes: Optional[dict[str, str]] = None,
) -> OAuth2SecurityScheme:
    """Three-legged OAuth2 (authorization_code) against the workspace OIDC endpoints.

    YAML equivalent::

        a2a:
          security_schemes:
            oauth2:
              type: oauth2
              flows:
                authorizationCode:
                  authorizationUrl: ${workspace.host}/oidc/v1/authorize
                  tokenUrl: ${workspace.host}/oidc/v1/token
                  scopes:
                    all-apis: Full Databricks REST API surface.
    """
    base = _resolve_host(host)
    return OAuth2SecurityScheme(
        type="oauth2",
        description="Databricks OAuth (authorization_code) — three-legged user flow.",
        flows=OAuthFlows(
            authorization_code=AuthorizationCodeOAuthFlow(
                authorization_url=f"{base}/oidc/v1/authorize",
                token_url=f"{base}/oidc/v1/token",
                scopes=scopes or {"all-apis": "Full Databricks REST API surface."},
            )
        ),
    )


def oauth2_databricks_client_credentials(
    host: Optional[str] = None,
    *,
    scopes: Optional[dict[str, str]] = None,
) -> OAuth2SecurityScheme:
    """OAuth2 client_credentials flow against ``oidc/v1/token``.

    YAML equivalent::

        a2a:
          security_schemes:
            oauth2_m2m:
              type: oauth2
              flows:
                clientCredentials:
                  tokenUrl: ${workspace.host}/oidc/v1/token
                  scopes:
                    all-apis: Full Databricks REST API surface.
    """
    base = _resolve_host(host)
    return OAuth2SecurityScheme(
        type="oauth2",
        description="Databricks OAuth (client_credentials) — service-principal M2M flow.",
        flows=OAuthFlows(
            client_credentials=ClientCredentialsOAuthFlow(
                token_url=f"{base}/oidc/v1/token",
                scopes=scopes or {"all-apis": "Full Databricks REST API surface."},
            )
        ),
    )


def oauth2_databricks_obo(
    host: Optional[str] = None,
) -> OAuth2SecurityScheme:
    """Three-legged OAuth2 advertising On-Behalf-Of-User via ``user_impersonation``.

    Use when an A2A caller needs to know that resources marked
    ``on_behalf_of_user: true`` will act as the calling user. The dao-ai
    Apps proxy still does the actual token forwarding via
    ``x-forwarded-access-token``; this declaration is documentation for the
    caller.

    YAML equivalent::

        a2a:
          security_schemes:
            oauth2:
              type: oauth2
              flows:
                authorizationCode:
                  authorizationUrl: ${workspace.host}/oidc/v1/authorize
                  tokenUrl: ${workspace.host}/oidc/v1/token
                  scopes:
                    user_impersonation: Act on behalf of the calling user.
    """
    base = _resolve_host(host)
    return OAuth2SecurityScheme(
        type="oauth2",
        description=_OBO_DESCRIPTION,
        flows=OAuthFlows(
            authorization_code=AuthorizationCodeOAuthFlow(
                authorization_url=f"{base}/oidc/v1/authorize",
                token_url=f"{base}/oidc/v1/token",
                scopes={
                    "user_impersonation": "Act on behalf of the calling user.",
                },
            )
        ),
    )


def openid_connect_databricks(
    host: Optional[str] = None,
) -> OpenIdConnectSecurityScheme:
    """OIDC discovery scheme pointing at the workspace's well-known config.

    YAML equivalent::

        a2a:
          security_schemes:
            oidc:
              type: openIdConnect
              openIdConnectUrl: ${workspace.host}/oidc/.well-known/openid-configuration
    """
    base = _resolve_host(host)
    return OpenIdConnectSecurityScheme(
        type="openIdConnect",
        open_id_connect_url=f"{base}/oidc/.well-known/openid-configuration",
        description="OpenID Connect discovery for the Databricks workspace.",
    )
