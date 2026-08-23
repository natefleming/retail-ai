"""Tests for the ``--with-connection`` feature: deploy-time creation of a UC
HTTP/MCP connection and its Unity AI Gateway MCP-service registration.

Covers the naming helpers, the ``ConnectionRegistrationModel`` config block, the
schema-resolution fallback, the CLI flag + validation, the provider guard, and
the full ``register_mcp_connection`` API sequence (with idempotency).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from dao_ai import cli
from dao_ai.cli import parse_args
from dao_ai.config import (
    AppModel,
    ConnectionRegistrationModel,
    RegisteredModelModel,
    SchemaModel,
    connection_name_for,
    mcp_service_name_for,
    resolve_connection_registration,
)
from dao_ai.providers.databricks import DatabricksProvider, ServingMode


# --------------------------------------------------------------------------- #
# Naming helpers
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_connection_name_for() -> None:
    # hyphens -> underscores, mcp_ prefix + _conn suffix
    assert connection_name_for("My-Agent") == "mcp_my_agent_conn"
    assert connection_name_for("my_agent") == "mcp_my_agent_conn"
    # idempotent on an already mcp-prefixed app name (matches app_name_for)
    assert connection_name_for("mcp-my-agent") == "mcp_my_agent_conn"
    # idempotent on an already-suffixed name
    assert connection_name_for("mcp_my_agent_conn") == "mcp_my_agent_conn"


@pytest.mark.unit
def test_mcp_service_name_for() -> None:
    assert mcp_service_name_for("My-Agent") == "mcp_my_agent"
    assert mcp_service_name_for("mcp-my-agent") == "mcp_my_agent"
    assert mcp_service_name_for("mcp_my_agent") == "mcp_my_agent"


# --------------------------------------------------------------------------- #
# ConnectionRegistrationModel
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_connection_registration_model_defaults() -> None:
    sm = SchemaModel(catalog_name="main", schema_name="genie")
    reg = ConnectionRegistrationModel(schema=sm)
    assert reg.grant_principals == ["account users"]
    assert reg.name is None
    assert reg.service_name is None
    assert reg.schema_model.full_name == "main.genie"
    # Auth defaults to M2M.
    assert reg.on_behalf_of_user is False
    assert reg.oauth_client_id is None
    assert reg.oauth_client_secret is None
    assert reg.oauth_scope == "all-apis"


@pytest.mark.unit
def test_connection_registration_u2m_requires_client_id() -> None:
    """U2M (on_behalf_of_user) requires a DEDICATED custom OAuth app client_id.
    The app's own oauth2_app_client_id cannot be used (redirect allowlist pinned
    to the app URL — GAIA-435), so there is no auto-derivation fallback."""
    sm = SchemaModel(catalog_name="c", schema_name="s")
    with pytest.raises(Exception, match="oauth_client_id"):
        ConnectionRegistrationModel(schema=sm, on_behalf_of_user=True)
    # With a dedicated client_id (+ optional secret) it is valid.
    reg = ConnectionRegistrationModel(
        schema=sm,
        on_behalf_of_user=True,
        oauth_client_id="dedicated-cid",
        oauth_client_secret="dedicated-secret",
    )
    assert reg.on_behalf_of_user is True
    assert reg.oauth_client_id == "dedicated-cid"
    assert reg.oauth_client_secret == "dedicated-secret"


@pytest.mark.unit
def test_connection_registration_model_populate_by_name() -> None:
    """Constructible by both the ``schema`` alias (YAML) and ``schema_model``
    field name (code / the resolver fallback)."""
    sm = SchemaModel(catalog_name="c", schema_name="s")
    assert ConnectionRegistrationModel(schema=sm).schema_model is sm
    assert ConnectionRegistrationModel(schema_model=sm).schema_model is sm


@pytest.mark.unit
def test_connection_registration_model_forbids_extra() -> None:
    sm = SchemaModel(catalog_name="c", schema_name="s")
    with pytest.raises(Exception):
        ConnectionRegistrationModel(schema=sm, bogus="x")


# --------------------------------------------------------------------------- #
# resolve_connection_registration
# --------------------------------------------------------------------------- #
def _app(**kw: Any) -> AppModel:
    """AppModel with validation bypassed (the resolver only reads attributes)."""
    kw.setdefault("connection", None)
    kw.setdefault("registered_model", None)
    return AppModel.model_construct(name="my-agent", **kw)


@pytest.mark.unit
def test_resolve_explicit_block_wins() -> None:
    sm = SchemaModel(catalog_name="main", schema_name="genie")
    reg = ConnectionRegistrationModel(schema=sm)
    app = _app(connection=reg)
    assert resolve_connection_registration(app) is reg


@pytest.mark.unit
def test_resolve_fallback_registered_model_schema() -> None:
    sm = SchemaModel(catalog_name="main", schema_name="genie")
    app = _app(registered_model=RegisteredModelModel(schema=sm, name="m"))
    reg = resolve_connection_registration(app)
    assert reg.schema_model.full_name == "main.genie"
    assert reg.grant_principals == ["account users"]


@pytest.mark.unit
def test_resolve_fallback_registered_model_fqn_name() -> None:
    app = _app(registered_model=RegisteredModelModel(name="cat.sch.model"))
    reg = resolve_connection_registration(app)
    assert reg.schema_model.full_name == "cat.sch"


@pytest.mark.unit
@pytest.mark.parametrize(
    "registered_model",
    [None, RegisteredModelModel(name="short_name_only")],
)
def test_resolve_no_schema_raises(registered_model: Optional[Any]) -> None:
    app = _app(registered_model=registered_model)
    with pytest.raises(ValueError, match="target schema"):
        resolve_connection_registration(app)


# --------------------------------------------------------------------------- #
# CLI flag + validation
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_cli_with_connection_requires_as_mcp() -> None:
    with pytest.raises(SystemExit) as exc:
        parse_args(
            ["agent", "up", "-c", "x.yaml", "--mode", "apps", "--with-connection"]
        )
    assert exc.value.code == 1


@pytest.mark.unit
def test_cli_with_connection_and_as_mcp_ok() -> None:
    opts = parse_args(
        [
            "agent",
            "up",
            "-c",
            "x.yaml",
            "--mode",
            "apps",
            "--as-mcp",
            "--with-connection",
        ]
    )
    assert opts.with_connection is True
    assert opts.as_mcp is True


@pytest.mark.unit
def test_cli_with_connection_defaults_false() -> None:
    opts = parse_args(["agent", "up", "-c", "x.yaml", "--mode", "apps", "--as-mcp"])
    assert opts.with_connection is False


# --------------------------------------------------------------------------- #
# Provider guard
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_provider_deploy_agent_guard_requires_as_mcp() -> None:
    provider = DatabricksProvider(w=MagicMock())
    with pytest.raises(ValueError, match="with_connection requires as_mcp"):
        provider.deploy_agent(
            config=MagicMock(),
            mode=ServingMode.APPS,
            as_mcp=False,
            with_connection=True,
        )


# --------------------------------------------------------------------------- #
# register_mcp_connection — the full API sequence
# --------------------------------------------------------------------------- #
class _FakeWorkspaceClient:
    """Minimal WorkspaceClient capturing the calls register_mcp_connection makes."""

    def __init__(self, *, existing_connections: Optional[list[str]] = None) -> None:
        self.updated_permissions: list[dict[str, Any]] = []
        self.created_connections: list[dict[str, Any]] = []
        self.api_calls: list[dict[str, Any]] = []
        self._existing = existing_connections or []
        self.secrets_created_for: list[int] = []

        outer = self

        class _Apps:
            def get(self, name: str) -> Any:
                return MagicMock(
                    url="https://mcp-my-agent-123.databricksapps.com/",
                    service_principal_client_id="client-abc",
                    service_principal_id=42,
                )

            def update_permissions(
                self, app_name: str, access_control_list: list[Any]
            ) -> None:
                outer.updated_permissions.append(
                    {"app": app_name, "acl": access_control_list}
                )

        class _Connections:
            def get(self, name: str) -> Any:
                from databricks.sdk.errors import NotFound

                if name in outer._existing:
                    # SimpleNamespace (not MagicMock): MagicMock reserves the
                    # ``name`` kwarg for its repr, so ``.name`` would never equal
                    # the value.
                    return SimpleNamespace(
                        name=name,
                        options=outer.connection_options.get(name, {}),
                        credential_type=outer.connection_credential_types.get(
                            name, "OAUTH_M2M"
                        ),
                    )
                raise NotFound(f"connection {name} does not exist")

            def create(self, **kwargs: Any) -> None:
                outer.created_connections.append(kwargs)
                outer._existing.append(kwargs["name"])

            def update(self, **kwargs: Any) -> None:
                outer.updated_connections.append(kwargs)

            def delete(self, name: str) -> None:
                outer.deleted_connections.append(name)

        class _SecretsProxy:
            def create(self, service_principal_id: int) -> Any:
                outer.secrets_created_for.append(service_principal_id)
                return MagicMock(secret="s3cr3t", id="secret-id-1")

            def delete(self, service_principal_id: int, secret_id: str) -> None:
                outer.deleted_secrets.append((service_principal_id, secret_id))

        class _ApiClient:
            def do(
                self,
                method: str,
                path: str,
                query: Optional[dict[str, Any]] = None,
                body: Optional[dict[str, Any]] = None,
            ) -> Any:
                outer.api_calls.append(
                    {"method": method, "path": path, "query": query, "body": body}
                )
                # GET mcp-services -> report the configured existing services
                if method == "GET":
                    return {"mcp_services": outer._existing_services}
                return {}

        self.apps = _Apps()
        self.connections = _Connections()
        self.service_principal_secrets_proxy = _SecretsProxy()
        self.api_client = _ApiClient()
        self.config = MagicMock(host="https://host.databricks.com")
        self._existing_services: list[dict[str, Any]] = []
        self.connection_options: dict[str, dict[str, str]] = {}
        self.connection_credential_types: dict[str, str] = {}
        self.updated_connections: list[dict[str, Any]] = []
        self.deleted_connections: list[str] = []
        self.deleted_secrets: list[tuple[int, str]] = []


def _config_with_connection(
    grant_principals: Optional[list[str]] = None,
    *,
    on_behalf_of_user: bool = False,
    oauth_client_id: Optional[str] = None,
    oauth_client_secret: Optional[str] = None,
) -> Any:
    """AppConfig-shaped mock whose ``app.connection`` is a real registration."""
    sm = SchemaModel(catalog_name="main", schema_name="genie")
    reg = ConnectionRegistrationModel(
        schema=sm,
        on_behalf_of_user=on_behalf_of_user,
        oauth_client_id=oauth_client_id,
        oauth_client_secret=oauth_client_secret,
    )
    if grant_principals is not None:
        reg.grant_principals = grant_principals
    config = MagicMock()
    config.app.name = "my-agent"
    config.app.connection = reg
    config.app.registered_model = None
    return config


@pytest.mark.unit
def test_register_mcp_connection_full_sequence() -> None:
    w = _FakeWorkspaceClient()
    provider = DatabricksProvider(w=w)
    provider.register_mcp_connection(_config_with_connection())

    # 1. app SP granted CAN_USE on its own app
    assert len(w.updated_permissions) == 1
    assert w.updated_permissions[0]["app"] == "mcp-my-agent"

    # 2. a fresh secret was minted and the HTTP/MCP connection created
    assert w.secrets_created_for == [42]
    assert len(w.created_connections) == 1
    created = w.created_connections[0]
    assert created["name"] == "mcp_my_agent_conn"
    opts = created["options"]
    assert opts["is_mcp_connection"] == "true"
    assert opts["base_path"] == "/mcp"
    assert opts["host"] == "https://mcp-my-agent-123.databricksapps.com"
    assert opts["token_endpoint"] == "https://host.databricks.com/oidc/v1/token"
    assert opts["client_id"] == "client-abc"
    assert opts["client_secret"] == "s3cr3t"

    # 3. MCP service registered (POST) under the target schema
    posts = [c for c in w.api_calls if c["method"] == "POST"]
    assert len(posts) == 1
    post = posts[0]
    assert post["path"] == "/api/2.1/unity-catalog/mcp-services"
    assert post["query"] == {
        "parent": "schemas/main.genie",
        "mcp_service_id": "mcp_my_agent",
    }
    assert (
        post["body"]["config"]["source_connection"]["name"]
        == "connections/mcp_my_agent_conn"
    )

    # 4. grants: USE_CONNECTION on the connection, EXECUTE on the service
    patches = [c for c in w.api_calls if c["method"] == "PATCH"]
    assert len(patches) == 2  # one principal x two securables
    conn_grant = next(p for p in patches if "/connection/" in p["path"])
    svc_grant = next(p for p in patches if "/mcp_service/" in p["path"])
    assert conn_grant["path"].endswith("/connection/mcp_my_agent_conn")
    assert conn_grant["body"]["changes"][0] == {
        "principal": "account users",
        "add": ["USE_CONNECTION"],
    }
    assert svc_grant["path"].endswith("/mcp_service/main.genie.mcp_my_agent")
    assert svc_grant["body"]["changes"][0] == {
        "principal": "account users",
        "add": ["EXECUTE"],
    }


@pytest.mark.unit
def test_register_mcp_connection_u2m() -> None:
    """U2M (on_behalf_of_user) creates an OAUTH_U2M_MAPPING connection with the
    DEDICATED custom OAuth app's client_id + secret + authorization_endpoint +
    credential-exchange method and NO app-SP minted secret, and grants the
    forwarding users (not the app SP) CAN_USE on the app."""
    w = _FakeWorkspaceClient()
    provider = DatabricksProvider(w=w)
    provider.register_mcp_connection(
        _config_with_connection(
            on_behalf_of_user=True,
            oauth_client_id="dedicated-cid",
            oauth_client_secret="dedicated-secret",
        )
    )

    # No app-SP secret is minted for U2M.
    assert w.secrets_created_for == []

    # Connection created with U2M options: authorization_endpoint + the dedicated
    # OAuth app client_id/secret + credential exchange method.
    created = w.created_connections[0]["options"]
    assert created["authorization_endpoint"] == (
        "https://host.databricks.com/oidc/v1/authorize"
    )
    assert created["client_id"] == "dedicated-cid"
    assert created["client_secret"] == "dedicated-secret"
    assert created["is_mcp_connection"] == "true"
    assert created["base_path"] == "/mcp"
    # U2M requests offline_access so a refresh token is issued (else the
    # connection stops working ~1h after each user's consent).
    assert "offline_access" in created["oauth_scope"].split()

    # CAN_USE granted to the forwarding users (group), NOT the app service principal.
    assert len(w.updated_permissions) == 1
    acl = w.updated_permissions[0]["acl"]
    assert [e.group_name for e in acl] == ["account users"]
    assert all(e.service_principal_name is None for e in acl)


@pytest.mark.unit
def test_register_mcp_connection_u2m_secret_optional() -> None:
    """A public dedicated OAuth client (no secret) still registers — client_secret
    is only added to the options when provided."""
    w = _FakeWorkspaceClient()
    provider = DatabricksProvider(w=w)
    provider.register_mcp_connection(
        _config_with_connection(on_behalf_of_user=True, oauth_client_id="dedicated-cid")
    )
    created = w.created_connections[0]["options"]
    assert created["client_id"] == "dedicated-cid"
    assert "client_secret" not in created
    assert w.secrets_created_for == []


@pytest.mark.unit
def test_register_mcp_connection_multiple_principals() -> None:
    w = _FakeWorkspaceClient()
    provider = DatabricksProvider(w=w)
    provider.register_mcp_connection(
        _config_with_connection(grant_principals=["account users", "analysts"])
    )
    patches = [c for c in w.api_calls if c["method"] == "PATCH"]
    assert len(patches) == 4  # two principals x two securables
    principals = {p["body"]["changes"][0]["principal"] for p in patches}
    assert principals == {"account users", "analysts"}


@pytest.mark.unit
def test_register_mcp_connection_no_drift_left_as_is() -> None:
    """An existing connection whose managed options already match the config is
    left as-is: no create, no update, no secret minted."""
    w = _FakeWorkspaceClient(existing_connections=["mcp_my_agent_conn"])
    w.connection_options["mcp_my_agent_conn"] = {
        "host": "https://mcp-my-agent-123.databricksapps.com",
        "base_path": "/mcp",
        "is_mcp_connection": "true",
        "oauth_scope": "all-apis",
    }
    provider = DatabricksProvider(w=w)
    provider.register_mcp_connection(_config_with_connection())
    assert w.created_connections == []
    assert w.updated_connections == []
    assert w.secrets_created_for == []
    # service registration + grants still run
    assert any(c["method"] == "POST" for c in w.api_calls)


@pytest.mark.unit
def test_register_mcp_connection_reconciles_scope_drift() -> None:
    """An existing M2M connection whose oauth_scope drifted from the config is
    UPDATED in place (not left stale, not recreated): full options re-sent with a
    freshly minted secret so the connection stays usable."""
    w = _FakeWorkspaceClient(existing_connections=["mcp_my_agent_conn"])
    w.connection_options["mcp_my_agent_conn"] = {
        "host": "https://mcp-my-agent-123.databricksapps.com",
        "base_path": "/mcp",
        "is_mcp_connection": "true",
        "oauth_scope": "stale-scope",
    }
    provider = DatabricksProvider(w=w)
    provider.register_mcp_connection(_config_with_connection())
    assert w.created_connections == []  # reconciled, not recreated
    assert len(w.updated_connections) == 1
    updated = w.updated_connections[0]
    assert updated["name"] == "mcp_my_agent_conn"
    assert updated["options"]["oauth_scope"] == "all-apis"
    assert updated["options"]["client_secret"] == "s3cr3t"  # re-minted M2M secret
    assert w.secrets_created_for == [42]


@pytest.mark.unit
def test_register_mcp_connection_reconciles_u2m_adds_offline_access() -> None:
    """An existing U2M connection missing offline_access is reconciled to add it —
    a U2M connection needs the refresh-token scope or it stops working ~1h after
    each consent. Reuses the config's dedicated OAuth client id; mints no app-SP
    secret."""
    w = _FakeWorkspaceClient(existing_connections=["mcp_my_agent_conn"])
    w.connection_credential_types["mcp_my_agent_conn"] = "OAUTH_U2M_MAPPING"
    w.connection_options["mcp_my_agent_conn"] = {
        "host": "https://mcp-my-agent-123.databricksapps.com",
        "base_path": "/mcp",
        "is_mcp_connection": "true",
        "oauth_scope": "all-apis",  # missing offline_access
    }
    provider = DatabricksProvider(w=w)
    provider.register_mcp_connection(
        _config_with_connection(on_behalf_of_user=True, oauth_client_id="dedicated-cid")
    )
    assert w.created_connections == []
    assert len(w.updated_connections) == 1
    scope = w.updated_connections[0]["options"]["oauth_scope"].split()
    assert "offline_access" in scope and "all-apis" in scope
    assert w.updated_connections[0]["options"]["client_id"] == "dedicated-cid"
    assert w.secrets_created_for == []  # U2M mints no app-SP secret


@pytest.mark.unit
def test_register_mcp_connection_mode_mismatch_raises() -> None:
    """An existing connection whose credential_type differs from the deploy mode
    cannot be flipped in place — fail loud with drop+recreate guidance, before any
    write or secret mint."""
    w = _FakeWorkspaceClient(existing_connections=["mcp_my_agent_conn"])
    w.connection_credential_types["mcp_my_agent_conn"] = "OAUTH_U2M_MAPPING"
    provider = DatabricksProvider(w=w)
    with pytest.raises(RuntimeError, match="credential type cannot be changed"):
        provider.register_mcp_connection(_config_with_connection())  # M2M deploy
    assert w.updated_connections == []
    assert w.created_connections == []
    assert w.secrets_created_for == []


@pytest.mark.unit
def test_register_mcp_connection_idempotent_service() -> None:
    """An existing MCP service is not re-created (no POST)."""
    w = _FakeWorkspaceClient()
    w._existing_services = [{"name": "mcp-services/main.genie.mcp_my_agent"}]
    provider = DatabricksProvider(w=w)
    provider.register_mcp_connection(_config_with_connection())
    assert [c for c in w.api_calls if c["method"] == "POST"] == []
    # grants still applied
    assert len([c for c in w.api_calls if c["method"] == "PATCH"]) == 2


@pytest.mark.unit
@pytest.mark.parametrize(
    "app_kwargs",
    [
        # missing SP identity
        {
            "url": "https://x/",
            "service_principal_client_id": None,
            "service_principal_id": None,
        },
        # SP present but URL not yet assigned -> must NOT create a host="" connection
        {
            "url": "",
            "service_principal_client_id": "client-abc",
            "service_principal_id": 42,
        },
    ],
)
def test_register_mcp_connection_not_ready_fails_loud(
    app_kwargs: dict[str, Any],
) -> None:
    """When the app URL/SP isn't resolvable, fail loudly and create nothing —
    never a connection with an empty host that idempotency would then skip."""
    w = _FakeWorkspaceClient()

    class _AppsNotReady:
        def get(self, name: str) -> Any:
            return MagicMock(**app_kwargs)

    w.apps = _AppsNotReady()
    provider = DatabricksProvider(w=w)
    with pytest.raises(RuntimeError, match="not ready"):
        provider.register_mcp_connection(_config_with_connection())
    assert w.created_connections == []
    assert w.api_calls == []


@pytest.mark.unit
def test_register_mcp_connection_create_failure_scrubs_secret() -> None:
    """A connection.create failure re-raises a scrubbed error (no secret, and no
    chained exception whose message might echo the request payload)."""
    w = _FakeWorkspaceClient()

    def _boom(**kwargs: Any) -> None:
        raise ValueError(
            f"bad request client_secret={kwargs['options']['client_secret']}"
        )

    w.connections.create = _boom  # type: ignore[method-assign]
    provider = DatabricksProvider(w=w)
    with pytest.raises(RuntimeError) as exc:
        provider.register_mcp_connection(_config_with_connection())
    assert "s3cr3t" not in str(exc.value)  # secret value redacted
    assert "***" in str(exc.value)  # ...to a placeholder
    assert "bad request" in str(exc.value)  # but the real cause is preserved
    assert exc.value.__cause__ is None  # `raise ... from None` suppressed the chain
    # The minted secret is cleaned up rather than orphaned on the app SP.
    assert w.deleted_secrets == [(42, "secret-id-1")]


@pytest.mark.unit
def test_register_post_real_failure_raises() -> None:
    """A genuine "does not exist" POST failure must RAISE (not be swallowed by an
    over-broad substring match) — the fail-loud design must hold."""
    w = _FakeWorkspaceClient()
    orig_do = w.api_client.do

    def _do(method: str, path: str, **kw: Any) -> Any:
        if method == "POST":
            raise RuntimeError("parent schema does not exist")
        return orig_do(method, path, **kw)

    w.api_client.do = _do  # type: ignore[method-assign]
    provider = DatabricksProvider(w=w)
    with pytest.raises(RuntimeError, match="does not exist"):
        provider.register_mcp_connection(_config_with_connection())


@pytest.mark.unit
def test_register_mcp_connection_grant_failure_is_best_effort() -> None:
    """The connection + service are created even if the final grants fail (the
    grant step is best-effort; a deployer may lack GRANT rights)."""
    w = _FakeWorkspaceClient()
    orig_do = w.api_client.do

    def _do(method: str, path: str, **kw: Any) -> Any:
        if method == "PATCH":
            raise PermissionError("no GRANT rights")
        return orig_do(method, path, **kw)

    w.api_client.do = _do  # type: ignore[method-assign]
    provider = DatabricksProvider(w=w)
    # Does not raise despite the PATCH grants failing.
    provider.register_mcp_connection(_config_with_connection())
    assert len(w.created_connections) == 1
    assert any(c["method"] == "POST" for c in w.api_calls)


@pytest.mark.unit
def test_register_mcp_connection_partial_grant_failure_attempts_all() -> None:
    """One principal's grant failure must not skip the rest — every principal ×
    securable PATCH is attempted, and the deploy does not raise."""
    w = _FakeWorkspaceClient()
    orig_do = w.api_client.do
    patch_targets: list[str] = []

    def _do(method: str, path: str, **kw: Any) -> Any:
        if method == "PATCH":
            patch_targets.append(path)
            # Fail only the first principal's connection grant.
            if (
                path.endswith("/connection/mcp_my_agent_conn")
                and kw["body"]["changes"][0]["principal"] == "account users"
            ):
                raise PermissionError("nope")
        return orig_do(method, path, **kw)

    w.api_client.do = _do  # type: ignore[method-assign]
    provider = DatabricksProvider(w=w)
    provider.register_mcp_connection(
        _config_with_connection(grant_principals=["account users", "analysts"])
    )
    # All 4 grants (2 principals x 2 securables) were attempted despite the fail.
    assert len(patch_targets) == 4


@pytest.mark.unit
def test_register_idempotency_boundary_no_false_positive() -> None:
    """A service in a catalog whose name is a SUFFIX of ours (xmain vs main)
    must NOT count as already-registered — the POST must still fire."""
    w = _FakeWorkspaceClient()
    w._existing_services = [{"name": "mcp-services/xmain.genie.mcp_my_agent"}]
    provider = DatabricksProvider(w=w)
    provider.register_mcp_connection(_config_with_connection())
    assert any(c["method"] == "POST" for c in w.api_calls)


@pytest.mark.unit
def test_register_post_already_exists_tolerated() -> None:
    """If the GET-based idempotency check misses (API drift) and the POST reports
    an already-exists conflict, treat it as success rather than failing."""
    w = _FakeWorkspaceClient()
    orig_do = w.api_client.do

    def _do(method: str, path: str, **kw: Any) -> Any:
        if method == "POST":
            raise RuntimeError("MCP service already exists")
        return orig_do(method, path, **kw)

    w.api_client.do = _do  # type: ignore[method-assign]
    provider = DatabricksProvider(w=w)
    provider.register_mcp_connection(_config_with_connection())  # no raise


# --------------------------------------------------------------------------- #
# unregister_mcp_connection — teardown
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_unregister_deletes_service_and_connection() -> None:
    """Deletes the MCP service and the connection when the connection is the one
    dao-ai created for this app (is_mcp_connection + host bound to the app)."""
    w = _FakeWorkspaceClient(existing_connections=["mcp_my_agent_conn"])
    w.connection_options["mcp_my_agent_conn"] = {
        "is_mcp_connection": "true",
        "host": "https://mcp-my-agent-123.databricksapps.com",
    }
    provider = DatabricksProvider(w=w)
    provider.unregister_mcp_connection(_config_with_connection())

    deletes = [c for c in w.api_calls if c["method"] == "DELETE"]
    assert len(deletes) == 1
    assert deletes[0]["path"].endswith("/mcp-services/main.genie.mcp_my_agent")
    assert w.deleted_connections == ["mcp_my_agent_conn"]


@pytest.mark.unit
def test_unregister_leaves_foreign_connection() -> None:
    """A same-named connection that isn't this app's MCP connection is left
    alone (guard: is_mcp_connection false / host mismatch)."""
    w = _FakeWorkspaceClient(existing_connections=["mcp_my_agent_conn"])
    w.connection_options["mcp_my_agent_conn"] = {
        # not an MCP connection, and host points somewhere unrelated
        "host": "https://someone-elses-service.example.com",
    }
    provider = DatabricksProvider(w=w)
    provider.unregister_mcp_connection(_config_with_connection())
    # service delete still attempted, but the connection is NOT deleted
    assert any(c["method"] == "DELETE" for c in w.api_calls)
    assert w.deleted_connections == []


@pytest.mark.unit
def test_unregister_app_gone_falls_back_to_mcp_marker() -> None:
    """When the app is already deleted (can't resolve its URL authoritatively),
    fall back to the is_mcp_connection marker to identify our connection."""
    from databricks.sdk.errors import NotFound

    w = _FakeWorkspaceClient(existing_connections=["mcp_my_agent_conn"])
    w.connection_options["mcp_my_agent_conn"] = {
        "is_mcp_connection": "true",
        "host": "https://mcp-my-agent-123.databricksapps.com",
    }

    class _AppsGone:
        def get(self, name: str) -> Any:
            raise NotFound("app deleted")

    w.apps = _AppsGone()
    provider = DatabricksProvider(w=w)
    provider.unregister_mcp_connection(_config_with_connection())
    assert w.deleted_connections == ["mcp_my_agent_conn"]


@pytest.mark.unit
def test_unregister_noop_when_connection_absent() -> None:
    """No connection present -> nothing deleted (still tries the service)."""
    w = _FakeWorkspaceClient()  # no existing connections
    provider = DatabricksProvider(w=w)
    provider.unregister_mcp_connection(_config_with_connection())
    assert w.deleted_connections == []


@pytest.mark.unit
def test_unregister_noop_when_no_schema() -> None:
    """No derivable schema -> nothing this feature registered; clean no-op."""
    w = _FakeWorkspaceClient()
    config = MagicMock()
    config.app.name = "my-agent"
    config.app.connection = None
    config.app.registered_model = None
    provider = DatabricksProvider(w=w)
    provider.unregister_mcp_connection(config)  # no raise
    assert w.api_calls == []
    assert w.deleted_connections == []


@pytest.mark.unit
def test_deploy_apps_agent_registers_when_with_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """deploy_apps_agent(as_mcp=True, with_connection=True) calls the registrar
    exactly once, after deploying the MCP app; skips it when with_connection is
    False."""
    provider = DatabricksProvider(w=MagicMock())
    monkeypatch.setattr(provider, "_deploy_app", lambda *a, **k: None)
    registered: list[bool] = []
    monkeypatch.setattr(
        provider, "register_mcp_connection", lambda config: registered.append(True)
    )
    # avoid importing the real extras resolver machinery
    monkeypatch.setattr(
        "dao_ai._extras.resolve_required_extras", lambda config, target: set()
    )
    monkeypatch.setattr("dao_ai._extras.expand_all", lambda x: set())

    provider.deploy_apps_agent(
        MagicMock(), as_mcp=True, with_connection=True, development=None
    )
    assert registered == [True]

    registered.clear()
    provider.deploy_apps_agent(
        MagicMock(), as_mcp=True, with_connection=False, development=None
    )
    assert registered == []


# --------------------------------------------------------------------------- #
# CLI bundle path (route 2, the default agent path) registers after deploy
# --------------------------------------------------------------------------- #
_MCP_CONFIG = (
    "resources:\n  models:\n    m: &m\n      name: databricks-gpt-5-4-mini\n"
    "agents:\n  g: &g\n    name: g\n    description: d\n    model: *m\n"
    "    prompt: p\n"
    "app:\n  name: my_app\n  agents:\n    - *g\n"
    "  connection:\n    schema:\n      catalog_name: main\n      schema_name: genie\n"
)


def _run_agent_up(
    tmp_path: Any, extra_args: list[str], monkeypatch: pytest.MonkeyPatch
) -> list[Any]:
    """Run `agent up` on the bundle path with the writer/deploy stubbed out;
    return the configs passed to a patched register_mcp_connection."""
    import pathlib

    from dao_ai.providers.databricks import DatabricksProvider

    cfg = tmp_path / "c.yaml"
    cfg.write_text(_MCP_CONFIG)
    out = tmp_path / "out"

    def fake_writer(config: object, bundle_dir: object, **kw: object) -> None:
        pathlib.Path(str(bundle_dir)).mkdir(parents=True, exist_ok=True)
        (pathlib.Path(str(bundle_dir)) / "databricks.yaml").write_text("bundle: {}\n")

    registered: list[Any] = []
    monkeypatch.setattr(cli, "_apply_profile_context", lambda p: None)
    monkeypatch.setattr("dao_ai.apps.bundle.write_bundle", fake_writer)
    monkeypatch.setattr(
        DatabricksProvider,
        "register_mcp_connection",
        lambda self, config: registered.append(config),
    )
    from unittest.mock import patch

    with patch.object(cli, "deploy_app_bundle"):
        opts = parse_args(
            ["agent", "up", "-c", str(cfg), "-s", str(out), "--mode", "apps"]
            + extra_args
        )
        cli.handle_agent_command(opts)
    return registered


@pytest.mark.unit
def test_cli_bundle_path_registers_connection(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    registered = _run_agent_up(tmp_path, ["--as-mcp", "--with-connection"], monkeypatch)
    assert len(registered) == 1


@pytest.mark.unit
def test_cli_bundle_path_no_register_without_flag(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    registered = _run_agent_up(tmp_path, ["--as-mcp"], monkeypatch)
    assert registered == []


# A config with no app.connection block AND no registered_model -> no derivable
# connection schema.
_MCP_CONFIG_NO_SCHEMA = (
    "resources:\n  models:\n    m: &m\n      name: databricks-gpt-5-4-mini\n"
    "agents:\n  g: &g\n    name: g\n    description: d\n    model: *m\n"
    "    prompt: p\n"
    "app:\n  name: my_app\n  agents:\n    - *g\n"
)


@pytest.mark.unit
def test_cli_bundle_path_validates_schema_before_deploy(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`up --as-mcp --with-connection` with no derivable schema fails BEFORE the
    app is deployed (fail fast), not after — deploy_app_bundle is never called."""
    import pathlib
    from unittest.mock import patch

    cfg = tmp_path / "c.yaml"
    cfg.write_text(_MCP_CONFIG_NO_SCHEMA)
    out = tmp_path / "out"

    def fake_writer(config: object, bundle_dir: object, **kw: object) -> None:
        pathlib.Path(str(bundle_dir)).mkdir(parents=True, exist_ok=True)
        (pathlib.Path(str(bundle_dir)) / "databricks.yaml").write_text("bundle: {}\n")

    monkeypatch.setattr(cli, "_apply_profile_context", lambda p: None)
    monkeypatch.setattr("dao_ai.apps.bundle.write_bundle", fake_writer)
    with patch.object(cli, "deploy_app_bundle") as dep:
        opts = parse_args(
            [
                "agent",
                "up",
                "-c",
                str(cfg),
                "-s",
                str(out),
                "--mode",
                "apps",
                "--as-mcp",
                "--with-connection",
            ]
        )
        with pytest.raises(ValueError, match="target schema"):
            cli.handle_agent_command(opts)
    dep.assert_not_called()


@pytest.mark.unit
def test_cli_down_unregisters_connection(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`agent down --as-mcp` tears down the UC connection + MCP service."""
    from unittest.mock import patch

    from dao_ai.providers.databricks import DatabricksProvider

    cfg = tmp_path / "c.yaml"
    cfg.write_text(_MCP_CONFIG)
    out = tmp_path / "out"
    out.mkdir()
    (out / "databricks.yaml").write_text("bundle: {}\n")  # already staged

    unregistered: list[Any] = []
    monkeypatch.setattr(cli, "_apply_profile_context", lambda p: None)
    monkeypatch.setattr(
        DatabricksProvider,
        "unregister_mcp_connection",
        lambda self, config: unregistered.append(config),
    )
    with patch.object(cli, "deploy_app_bundle"):
        opts = parse_args(
            [
                "agent",
                "down",
                "-c",
                str(cfg),
                "-s",
                str(out),
                "--mode",
                "apps",
                "--as-mcp",
            ]
        )
        cli.handle_agent_command(opts)
    assert len(unregistered) == 1
