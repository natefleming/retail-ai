"""GCP service-account credential resolution for DAO AI tools.

Resolves service-account credentials from one of three sources, auto-detected
by inspecting the resolved string:

1. **Inline JSON body** — value begins with ``{``. Parsed with
   :func:`google.oauth2.service_account.Credentials.from_service_account_info`.
   Suitable for storing the key in a Databricks secret scope.

2. **Databricks Volume path** — value begins with ``/Volumes/``. Loaded from
   the local FUSE mount when available (in-runtime) or downloaded to a
   temporary file via the Databricks Files API (outside the runtime).

3. **Local filesystem path** — any other non-empty string. Loaded with
   :func:`google.oauth2.service_account.Credentials.from_service_account_file`.

All three sources return a :class:`google.oauth2.service_account.Credentials`
instance that ``google-auth`` can refresh on demand.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

import google.auth
from google.auth.credentials import Credentials as BaseCredentials
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from google.oauth2.service_account import Credentials
from loguru import logger

from dao_ai.config import (
    AnyVariable,
    CompositeVariableModel,
    EnvironmentVariableModel,
    PrimitiveVariableModel,
    SecretVariableModel,
    value_of,
)


def coerce_any_variable(value: Any) -> Any:
    """Coerce a raw dict (as delivered from factory args) into an ``AnyVariable`` model.

    Factory function arguments are stored as ``dict[str, Any]`` in
    :class:`FactoryFunctionModel` with no schema coercion, so a YAML value
    like ``{env: FOO}`` arrives as a plain dict — which ``value_of`` passes
    through unchanged. This helper inspects the dict shape and wraps it in
    the appropriate ``AnyVariable`` subclass so that ``value_of`` can
    resolve it.

    Non-dict values are returned unchanged.
    """
    if not isinstance(value, dict):
        return value
    if "options" in value:
        return CompositeVariableModel(**value)
    if "env" in value:
        return EnvironmentVariableModel(**value)
    if "secret" in value and "scope" in value:
        return SecretVariableModel(**value)
    if "value" in value:
        return PrimitiveVariableModel(**value)
    return value


DEFAULT_SCOPES: list[str] = ["https://www.googleapis.com/auth/cloud-platform"]


def load_gcp_credentials(
    credentials: AnyVariable,
    scopes: Optional[list[str]] = None,
) -> Credentials:
    """Resolve GCP service-account credentials from any supported source.

    Not instrumented with ``@mlflow.trace`` — the input may contain an
    inline JSON service-account key and MLflow's auto input-capture would
    persist the private key inside the trace.

    Args:
        credentials: Raw string or ``AnyVariable`` (env, secret, composite)
            that resolves to a filesystem path, Databricks volume path, or
            inline JSON body.
        scopes: OAuth scopes requested on the returned credentials. Defaults
            to ``https://www.googleapis.com/auth/cloud-platform``.

    Returns:
        Loaded ``Credentials``. The caller must refresh before use (or call
        :func:`mint_gcp_access_token`).

    Raises:
        ValueError: If ``credentials`` resolves to an empty or non-string value.
    """
    resolved: object = value_of(coerce_any_variable(credentials))
    if not isinstance(resolved, str) or not resolved:
        raise ValueError(
            "GCP credentials must resolve to a non-empty string "
            "(file path, /Volumes/... path, or JSON body)."
        )

    scope_list: list[str] = scopes or DEFAULT_SCOPES
    stripped: str = resolved.lstrip()

    if stripped.startswith("{"):
        logger.debug("Loading GCP credentials from inline JSON body")
        info: dict = json.loads(stripped)
        return service_account.Credentials.from_service_account_info(
            info, scopes=scope_list
        )
    if resolved.startswith("/Volumes/"):
        logger.debug(f"Loading GCP credentials from Databricks Volume: {resolved}")
        return _load_from_volume(resolved, scope_list)

    logger.debug(f"Loading GCP credentials from local file: {resolved}")
    return service_account.Credentials.from_service_account_file(
        resolved, scopes=scope_list
    )


def mint_gcp_access_token(credentials: BaseCredentials) -> str:
    """Return a valid access token, refreshing if expired.

    Accepts any ``google.auth.credentials.Credentials`` subclass — service
    account creds, ADC creds (user, metadata-server, WIF), or impersonated
    creds — so the same helper works across all supported auth modes.

    Not instrumented with ``@mlflow.trace`` — the return value is the
    access token itself, and MLflow's auto output-capture would write the
    token into trace storage.
    """
    if not credentials.valid:
        credentials.refresh(Request())
    return credentials.token


def load_gcp_adc_credentials(
    scopes: Optional[list[str]] = None,
) -> BaseCredentials:
    """Discover credentials via Google's Application Default Credentials chain.

    Delegates to :func:`google.auth.default`, which searches (in order):
    ``GOOGLE_APPLICATION_CREDENTIALS`` env var, gcloud user credentials,
    the GCE/Cloud Run/GKE metadata server, and Workload Identity
    Federation configs. Useful for local development (``gcloud auth
    application-default login``) and for workloads running on GCP where
    a service-account file isn't mounted.

    Not instrumented with ``@mlflow.trace`` — the returned credentials
    may carry refresh tokens that MLflow's auto-capture would persist.

    Args:
        scopes: OAuth scopes to request. Defaults to ``cloud-platform``.

    Returns:
        Discovered ``Credentials``. Caller must refresh before use (or
        call :func:`mint_gcp_access_token`).

    Raises:
        google.auth.exceptions.DefaultCredentialsError: If no credentials
            can be discovered on this host.
    """
    scope_list: list[str] = scopes or DEFAULT_SCOPES
    credentials, project_id = google.auth.default(scopes=scope_list)
    logger.debug(
        "Loaded GCP credentials via Application Default Credentials",
        project_id=project_id,
        credential_type=type(credentials).__name__,
    )
    return credentials


def _load_from_volume(path: str, scopes: list[str]) -> Credentials:
    """Load credentials from a Unity Catalog Volume path.

    Tries the local FUSE mount first (which is how ``/Volumes/...`` appears
    on Databricks runtimes). Falls back to downloading via the Files API so
    the same config works from a local dev machine.
    """
    if os.path.isfile(path):
        return service_account.Credentials.from_service_account_file(
            path, scopes=scopes
        )

    from databricks.sdk import WorkspaceClient

    workspace_client: WorkspaceClient = WorkspaceClient()
    with tempfile.NamedTemporaryFile(
        prefix="gcp_sa_", suffix=".json", delete=False
    ) as tmp:
        response = workspace_client.files.download(path)
        tmp.write(response.contents.read())
        tmp_path: str = tmp.name

    try:
        return service_account.Credentials.from_service_account_file(
            tmp_path, scopes=scopes
        )
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except OSError:
            logger.warning(f"Failed to remove temp credentials file: {tmp_path}")
