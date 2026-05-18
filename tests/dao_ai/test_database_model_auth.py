"""Tests for :meth:`IsDatabricksResource.workspace_client` diagnostics.

The property already implemented the right four-branch precedence
(OBO -> SP -> PAT -> Ambient), but silent fall-through when SP secrets
resolved to None left users staring at unrelated postgres errors
(``password authentication failed for user '<my-email>'``) downstream.

These tests pin the diagnostic surface:

1. Configuring ``client_id`` + ``client_secret`` that resolve cleanly
   produces a service-principal ``WorkspaceClient`` (no ambient
   fall-through), with ``workspace_host`` defaulted from
   ``get_default_databricks_host`` when not provided.
2. Configuring ``client_id`` but having ``value_of`` resolve to None
   emits a warning naming the unresolved field, then falls through to
   ambient cleanly (no exception). The warning is the contract: when
   the user expressed SP intent, dao-ai must not silently swallow it.
"""

from __future__ import annotations

import logging

import pytest
from databricks.sdk import WorkspaceClient

from dao_ai.config import (
    DatabaseModel,
    SecretVariableModel,
)


@pytest.mark.unit
def test_workspace_client_sp_branch_fires_when_credentials_resolve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """client_id + client_secret + workspace_host -> SP WorkspaceClient."""
    db = DatabaseModel(
        name="testdb",
        project="testproj",
        client_id="11111111-2222-3333-4444-555555555555",
        client_secret="dummy-secret",
        workspace_host="https://test.cloud.databricks.com",
    )

    # The constructed client should carry the SP creds, not ambient.
    constructed: dict = {}

    def _fake_init(
        self: WorkspaceClient,
        host: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        auth_type: str | None = None,
        token: str | None = None,
        **_: object,
    ) -> None:
        constructed["host"] = host
        constructed["client_id"] = client_id
        constructed["client_secret"] = client_secret
        constructed["auth_type"] = auth_type
        constructed["token"] = token

    monkeypatch.setattr(WorkspaceClient, "__init__", _fake_init)

    _ = db.workspace_client

    assert constructed["client_id"] == "11111111-2222-3333-4444-555555555555"
    assert constructed["client_secret"] == "dummy-secret"
    assert constructed["host"] == "https://test.cloud.databricks.com"
    assert constructed["auth_type"] == "oauth-m2m"
    assert constructed["token"] is None


@pytest.mark.unit
def test_workspace_client_sp_defaults_host_via_get_default_databricks_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SP credentials without workspace_host pick the host from the resolver."""
    monkeypatch.setattr(
        "dao_ai.config.get_default_databricks_host",
        lambda: "https://from-resolver.cloud.databricks.com",
        raising=False,
    )
    # The resolver lives on dao_ai.utils -- patch both import sites the
    # function is referenced from.
    monkeypatch.setattr(
        "dao_ai.utils.get_default_databricks_host",
        lambda: "https://from-resolver.cloud.databricks.com",
    )

    db = DatabaseModel(
        name="testdb",
        project="testproj",
        client_id="11111111-2222-3333-4444-555555555555",
        client_secret="dummy-secret",
        # no workspace_host
    )

    constructed_host: dict = {}

    def _fake_init(
        self: WorkspaceClient,
        host: str | None = None,
        **_: object,
    ) -> None:
        constructed_host["host"] = host

    monkeypatch.setattr(WorkspaceClient, "__init__", _fake_init)

    _ = db.workspace_client
    assert constructed_host["host"] == "https://from-resolver.cloud.databricks.com"


@pytest.mark.unit
def test_workspace_client_warns_when_sp_creds_unresolvable(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A SecretVariableModel that resolves to None must produce a warning.

    Matches the lab-07-memory failure mode: the YAML sets
    ``client_id: *client_id`` pointing at a secret scope the running
    identity can't read; dao-ai used to fall through to ambient auth
    silently. The warning is the diagnostic contract.
    """
    # Patch SecretVariableModel.as_value to simulate an unreadable scope.
    monkeypatch.setattr(
        SecretVariableModel,
        "as_value",
        lambda self: None,
    )

    db = DatabaseModel(
        name="testdb",
        project="testproj",
        client_id=SecretVariableModel(scope="dao_ai_workshop", secret="client_id"),
        client_secret=SecretVariableModel(
            scope="dao_ai_workshop", secret="client_secret"
        ),
        workspace_host="https://test.cloud.databricks.com",
    )

    # Don't actually construct a WorkspaceClient.
    monkeypatch.setattr(WorkspaceClient, "__init__", lambda self, **_: None)

    # loguru -> stderr by default; reroute to caplog so pytest sees it.
    from loguru import logger as _logger

    handler_id = _logger.add(caplog.handler, level="WARNING")
    try:
        _ = db.workspace_client
    finally:
        _logger.remove(handler_id)

    warnings = [
        rec.message
        for rec in caplog.records
        if rec.levelno >= logging.WARNING and "resolves to None" in rec.message
    ]
    assert any("client_id" in w for w in warnings), (
        f"expected warning naming client_id, saw: {warnings}"
    )
    assert any("client_secret" in w for w in warnings), (
        f"expected warning naming client_secret, saw: {warnings}"
    )


@pytest.mark.unit
def test_workspace_client_ambient_no_creds_no_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """When neither client_id nor pat is set, the property silently uses ambient.

    This is the canonical "interactive notebook = ambient" path. No
    warning should fire because no SP intent was expressed.
    """
    db = DatabaseModel(
        name="testdb",
        project="testproj",
        # no client_id, client_secret, pat, or on_behalf_of_user
    )

    monkeypatch.setattr(WorkspaceClient, "__init__", lambda self, **_: None)

    from loguru import logger as _logger

    handler_id = _logger.add(caplog.handler, level="WARNING")
    try:
        _ = db.workspace_client
    finally:
        _logger.remove(handler_id)

    sp_warnings = [
        rec.message for rec in caplog.records if "resolves to None" in rec.message
    ]
    assert sp_warnings == [], (
        f"unexpected SP-resolution warnings on ambient path: {sp_warnings}"
    )


@pytest.mark.unit
def test_workspace_client_pat_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pat is honored when client_id/client_secret are unset."""
    db = DatabaseModel(
        name="testdb",
        project="testproj",
        pat="dapi-fake-token-value",
        workspace_host="https://test.cloud.databricks.com",
    )

    constructed: dict = {}

    def _fake_init(
        self: WorkspaceClient,
        host: str | None = None,
        token: str | None = None,
        auth_type: str | None = None,
        **_: object,
    ) -> None:
        constructed["host"] = host
        constructed["token"] = token
        constructed["auth_type"] = auth_type

    monkeypatch.setattr(WorkspaceClient, "__init__", _fake_init)

    _ = db.workspace_client
    assert constructed["token"] == "dapi-fake-token-value"
    assert constructed["auth_type"] == "pat"
    assert constructed["host"] == "https://test.cloud.databricks.com"
