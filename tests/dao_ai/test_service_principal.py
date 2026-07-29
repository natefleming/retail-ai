"""Unit tests for the ``dao-ai service-principal`` helpers.

Cover the pure grant-plan resource walk + privilege mapping, the create/store
SDK call shapes (mocked WorkspaceClient), and principal/secret resolution.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from dao_ai.config import (
    AppConfig,
    AppModel,
    FunctionModel,
    SchemaModel,
    SecretVariableModel,
    ServicePrincipalModel,
    TableModel,
    WarehouseModel,
)
from dao_ai.service_principal import (
    build_grant_plan,
    create,
    default_scope_from_config,
    grant,
    provision,
    resolve_principal_from_config,
    resolve_secret_target,
    store,
)

PRINCIPAL = "11111111-1111-1111-1111-111111111111"


def _schema() -> SchemaModel:
    return SchemaModel(catalog_name="cat", schema_name="sch")


# =============================================================================
# build_grant_plan — resource walk + privilege mapping
# =============================================================================


def test_grant_plan_schema_catalog_privileges() -> None:
    config = AppConfig(schemas={"s": _schema()})
    plan = build_grant_plan(config, PRINCIPAL)

    by_target = {(g.securable_type, g.target): g for g in plan.grants}
    assert ("catalog", "cat") in by_target
    assert by_target[("catalog", "cat")].privileges == ["USE_CATALOG"]
    assert ("schema", "cat.sch") in by_target
    assert by_target[("schema", "cat.sch")].privileges == ["USE_SCHEMA", "SELECT", "EXECUTE"]


def test_grant_plan_table_and_function_privileges() -> None:
    from dao_ai.config import ResourcesModel

    schema = _schema()
    config = AppConfig(
        schemas={"s": schema},
        resources=ResourcesModel(
            tables={"t": TableModel(schema=schema, name="products")},
            functions={"f": FunctionModel(schema=schema, name="find_x")},
        ),
    )
    plan = build_grant_plan(config, PRINCIPAL)
    mapping = {(g.securable_type, g.target): list(g.privileges) for g in plan.grants}

    assert mapping[("table", "cat.sch.products")] == ["SELECT"]
    assert mapping[("function", "cat.sch.find_x")] == ["EXECUTE"]


def test_grant_plan_dedupes_catalog_and_schema() -> None:
    from dao_ai.config import ResourcesModel

    schema = _schema()
    config = AppConfig(
        schemas={"s": schema},
        resources=ResourcesModel(
            tables={
                "a": TableModel(schema=schema, name="one"),
                "b": TableModel(schema=schema, name="two"),
            },
        ),
    )
    plan = build_grant_plan(config, PRINCIPAL)
    catalogs = [g for g in plan.grants if g.securable_type == "catalog"]
    schemas = [g for g in plan.grants if g.securable_type == "schema"]
    assert len(catalogs) == 1  # deduped even though two tables share it
    assert len(schemas) == 1


def test_grant_plan_warehouse_and_serving_endpoint() -> None:
    from dao_ai.config import ResourcesModel

    from dao_ai.config import AgentModel, InferenceEndpointModel

    config = AppConfig(
        resources=ResourcesModel(
            warehouses={"w": WarehouseModel(warehouse_id="wh-123")},
        ),
        app=AppModel(
            name="my-agent",
            endpoint_name="my_agent_ep",
            agents=[
                AgentModel(
                    name="a",
                    description="d",
                    model=InferenceEndpointModel(name="databricks-gpt-5-4-mini"),
                )
            ],
        ),
    )
    plan = build_grant_plan(config, PRINCIPAL)
    kinds = {g.kind: g for g in plan.grants}

    assert kinds["warehouse"].target == "wh-123"
    assert kinds["warehouse"].privileges == ["CAN_USE"]
    assert kinds["serving_endpoint"].target == "my_agent_ep"
    assert kinds["serving_endpoint"].privileges == ["CAN_QUERY"]


def test_grant_plan_empty_config_yields_nothing() -> None:
    plan = build_grant_plan(AppConfig(), PRINCIPAL)
    assert plan.grants == []
    assert plan.principal == PRINCIPAL


# =============================================================================
# grant — dry-run applies nothing; real run issues UC PATCH
# =============================================================================


def test_grant_dry_run_makes_no_sdk_calls() -> None:
    w = MagicMock()
    config = AppConfig(schemas={"s": _schema()})
    plan = grant(w, principal=PRINCIPAL, config=config, dry_run=True)

    assert plan.grants  # a plan was produced
    w.api_client.do.assert_not_called()


def test_grant_applies_uc_patch_for_schema() -> None:
    w = MagicMock()
    config = AppConfig(schemas={"s": _schema()})
    grant(w, principal=PRINCIPAL, config=config, dry_run=False)

    # catalog USE_CATALOG + schema USE_SCHEMA/SELECT/EXECUTE via PATCH
    calls = w.api_client.do.call_args_list
    patched = {
        c.args[1]: c.kwargs["body"]["changes"][0]
        for c in calls
        if c.args[0] == "PATCH"
    }
    assert "/api/2.1/unity-catalog/permissions/catalog/cat" in patched
    schema_body = patched["/api/2.1/unity-catalog/permissions/schema/cat.sch"]
    assert schema_body["principal"] == PRINCIPAL
    assert schema_body["add"] == ["USE_SCHEMA", "SELECT", "EXECUTE"]


def test_grant_continues_after_a_failure() -> None:
    w = MagicMock()
    # First PATCH raises; the walk should warn-and-continue to the rest.
    w.api_client.do.side_effect = [RuntimeError("no GRANT rights"), None, None, None]
    config = AppConfig(schemas={"s": _schema()})
    grant(w, principal=PRINCIPAL, config=config, dry_run=False)
    assert w.api_client.do.call_count >= 2  # did not abort on the first failure


# =============================================================================
# create / store — SDK call shapes
# =============================================================================


def test_create_new_service_principal_mints_secret() -> None:
    w = MagicMock()
    w.service_principals.list.return_value = []  # none existing
    w.service_principals.create.return_value = MagicMock(
        id="42", application_id=PRINCIPAL
    )
    w.service_principal_secrets_proxy.create.return_value = MagicMock(secret="sekret")

    result = create(w, display_name="my-sp")

    w.service_principals.create.assert_called_once()
    w.service_principal_secrets_proxy.create.assert_called_once_with(
        service_principal_id="42"
    )
    assert result.client_id == PRINCIPAL
    assert result.client_secret == "sekret"
    assert result.reused is False


def test_create_reuses_existing_service_principal() -> None:
    w = MagicMock()
    w.service_principals.list.return_value = [
        MagicMock(id="7", application_id=PRINCIPAL, display_name="my-sp")
    ]
    w.service_principal_secrets_proxy.create.return_value = MagicMock(secret="fresh")

    result = create(w, display_name="my-sp")

    w.service_principals.create.assert_not_called()
    assert result.reused is True
    assert result.client_secret == "fresh"  # a fresh secret is still minted


def test_store_creates_scope_and_puts_secrets() -> None:
    w = MagicMock()
    store(
        w,
        scope="myscope",
        client_id_key="CID",
        client_secret_key="CSEC",
        client_id="the-id",
        client_secret="the-secret",
    )
    w.secrets.create_scope.assert_called_once_with(scope="myscope")
    put_calls = {c.kwargs["key"]: c.kwargs["string_value"] for c in w.secrets.put_secret.call_args_list}
    assert put_calls == {"CID": "the-id", "CSEC": "the-secret"}


def test_store_tolerates_existing_scope() -> None:
    w = MagicMock()
    w.secrets.create_scope.side_effect = RuntimeError("RESOURCE_ALREADY_EXISTS: scope")
    store(
        w,
        scope="s",
        client_id_key="a",
        client_secret_key="b",
        client_id="x",
        client_secret="y",
    )
    assert w.secrets.put_secret.call_count == 2  # proceeded despite the scope error


# =============================================================================
# principal / secret-target resolution
# =============================================================================


def test_resolve_principal_prefers_override() -> None:
    config = AppConfig(
        service_principals={
            "sp": ServicePrincipalModel(client_id="from-config", client_secret="x")
        }
    )
    assert resolve_principal_from_config(config, override="from-flag") == "from-flag"
    assert resolve_principal_from_config(config) == "from-config"


def test_resolve_secret_target_reads_scope_from_config() -> None:
    config = AppConfig(
        service_principals={
            "sp": ServicePrincipalModel(
                client_id=SecretVariableModel(scope="myscope", secret="CID"),
                client_secret=SecretVariableModel(scope="myscope", secret="CSEC"),
            )
        }
    )
    scope, cid_key, csec_key = resolve_secret_target(config)
    assert scope == "myscope"
    assert cid_key == "CID"
    assert csec_key == "CSEC"


def test_default_scope_prefers_app_name() -> None:
    from dao_ai.config import AgentModel, InferenceEndpointModel

    config = AppConfig(
        app=AppModel(
            name="my-agent",
            agents=[
                AgentModel(
                    name="a",
                    description="d",
                    model=InferenceEndpointModel(name="databricks-gpt-5-4-mini"),
                )
            ],
        )
    )
    assert default_scope_from_config(config) == "my-agent"


def test_default_scope_falls_back_to_catalog() -> None:
    config = AppConfig(schemas={"s": _schema()})
    assert default_scope_from_config(config) == "cat"


# =============================================================================
# provision — one-shot create + store + grant
# =============================================================================


def test_provision_end_to_end_stores_and_grants() -> None:
    w = MagicMock()
    w.service_principals.list.return_value = []
    w.service_principals.create.return_value = MagicMock(id="9", application_id=PRINCIPAL)
    w.service_principal_secrets_proxy.create.return_value = MagicMock(secret="s3cr3t")

    config = AppConfig(
        schemas={"s": _schema()},
        service_principals={
            "sp": ServicePrincipalModel(
                client_id=SecretVariableModel(scope="myscope", secret="CID"),
                client_secret=SecretVariableModel(scope="myscope", secret="CSEC"),
            )
        },
    )
    result = provision(w, config=config, display_name="my-sp")

    # secret written to the config's scope, never surfaced on the result
    assert result.stored is True
    assert result.stored_scope == "myscope"
    assert not hasattr(result, "client_secret")
    put_keys = {c.kwargs["key"] for c in w.secrets.put_secret.call_args_list}
    assert put_keys == {"CID", "CSEC"}
    # granted the schema (UC PATCH issued)
    assert result.grant_plan is not None and result.grant_plan.grants
    assert w.api_client.do.called


def test_provision_derives_scope_when_no_sp_block() -> None:
    from dao_ai.config import AgentModel, InferenceEndpointModel

    w = MagicMock()
    w.service_principals.list.return_value = []
    w.service_principals.create.return_value = MagicMock(id="9", application_id=PRINCIPAL)
    w.service_principal_secrets_proxy.create.return_value = MagicMock(secret="x")

    config = AppConfig(
        app=AppModel(
            name="my-agent",
            agents=[
                AgentModel(
                    name="a",
                    description="d",
                    model=InferenceEndpointModel(name="databricks-gpt-5-4-mini"),
                )
            ],
        )
    )
    result = provision(w, config=config, display_name="my-sp")
    assert result.stored_scope == "my-agent"  # derived from app name


def test_provision_no_store_no_grant_flags() -> None:
    w = MagicMock()
    w.service_principals.list.return_value = []
    w.service_principals.create.return_value = MagicMock(id="9", application_id=PRINCIPAL)
    w.service_principal_secrets_proxy.create.return_value = MagicMock(secret="x")

    result = provision(
        w,
        config=AppConfig(schemas={"s": _schema()}),
        display_name="my-sp",
        do_store=False,
        do_grant=False,
    )
    assert result.stored is False
    assert result.grant_plan is None
    w.secrets.put_secret.assert_not_called()
    w.api_client.do.assert_not_called()


def test_resolve_secret_target_override_wins() -> None:
    config = AppConfig(
        service_principals={
            "sp": ServicePrincipalModel(
                client_id=SecretVariableModel(scope="cfg", secret="CID"),
                client_secret=SecretVariableModel(scope="cfg", secret="CSEC"),
            )
        }
    )
    scope, cid_key, csec_key = resolve_secret_target(
        config, scope_override="flag-scope"
    )
    assert scope == "flag-scope"  # override wins over config
