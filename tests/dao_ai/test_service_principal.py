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
    GRANT_FAILURE_ABSENT,
    GRANT_FAILURE_DENIED,
    GRANT_FAILURE_ERROR,
    SecretRef,
    ServicePrincipalTarget,
    build_grant_plan,
    build_ownership_map,
    classify_grant_error,
    create,
    find_service_principal,
    secret_keys_present,
    default_scope_from_config,
    grant,
    provision,
    resolve_principal_from_config,
    resolve_secret_target,
    resource_owner,
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
    assert by_target[("schema", "cat.sch")].privileges == [
        "USE_SCHEMA",
        "SELECT",
        "EXECUTE",
    ]


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
    from dao_ai.config import AgentModel, InferenceEndpointModel, ResourcesModel

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


def test_grant_plan_volume_and_connection_privileges() -> None:
    from dao_ai.config import ConnectionModel, ResourcesModel, VolumeModel

    schema = _schema()
    config = AppConfig(
        schemas={"s": schema},
        resources=ResourcesModel(
            volumes={"v": VolumeModel(schema=schema, name="landing")},
            connections={"c": ConnectionModel(name="my_conn")},
        ),
    )
    plan = build_grant_plan(config, PRINCIPAL)
    mapping = {(g.securable_type, g.target): list(g.privileges) for g in plan.grants}

    assert mapping[("volume", "cat.sch.landing")] == ["READ_VOLUME"]
    assert mapping[("connection", "my_conn")] == ["USE_CONNECTION"]


def test_grant_plan_volume_reuses_schema_grant() -> None:
    from dao_ai.config import ResourcesModel, VolumeModel

    schema = _schema()
    config = AppConfig(
        schemas={"s": schema},
        resources=ResourcesModel(
            volumes={"v": VolumeModel(schema=schema, name="landing")},
        ),
    )
    plan = build_grant_plan(config, PRINCIPAL)
    # The volume's schema is deduped against the top-level schema grant.
    schemas = [g for g in plan.grants if g.securable_type == "schema"]
    assert len(schemas) == 1


def _lakebase_db(client_id: str, project: str = "my-proj", name: str = "lb"):
    from dao_ai.config import DatabaseModel

    return DatabaseModel(
        name=name,
        project=project,
        client_id=client_id,
        client_secret="secret",
    )


def test_grant_plan_lakebase_role_matching_sp_has_no_skip_note() -> None:
    from dao_ai.config import ResourcesModel

    config = AppConfig(
        resources=ResourcesModel(databases={"d": _lakebase_db(PRINCIPAL)}),
    )
    plan = build_grant_plan(config, PRINCIPAL)
    lb = [g for g in plan.grants if g.kind == "lakebase_role"]
    assert len(lb) == 1
    assert lb[0].target == "my-proj"
    assert lb[0].privileges == ["DATABRICKS_SUPERUSER"]
    assert lb[0].note is None  # matching SP → will be created at apply time


def test_grant_plan_lakebase_role_mismatched_sp_is_skipped() -> None:
    from dao_ai.config import ResourcesModel

    config = AppConfig(
        resources=ResourcesModel(
            databases={"d": _lakebase_db("22222222-2222-2222-2222-222222222222")}
        ),
    )
    plan = build_grant_plan(config, PRINCIPAL)
    lb = next(g for g in plan.grants if g.kind == "lakebase_role")
    assert lb.note is not None
    assert "SKIP" in lb.note
    assert "22222222-2222-2222-2222-222222222222" in lb.note


def test_grant_plan_lakebase_role_unset_client_id_is_skipped() -> None:
    from dao_ai.config import DatabaseModel, ResourcesModel

    config = AppConfig(
        resources=ResourcesModel(
            databases={"d": DatabaseModel(name="lb", project="my-proj")}
        ),
    )
    plan = build_grant_plan(config, PRINCIPAL)
    lb = next(g for g in plan.grants if g.kind == "lakebase_role")
    assert lb.note is not None
    assert "client_id is unset" in lb.note


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
        c.args[1]: c.kwargs["body"]["changes"][0] for c in calls if c.args[0] == "PATCH"
    }
    assert "/api/2.1/unity-catalog/permissions/catalog/cat" in patched
    schema_body = patched["/api/2.1/unity-catalog/permissions/schema/cat.sch"]
    assert schema_body["principal"] == PRINCIPAL
    assert schema_body["add"] == ["USE_SCHEMA", "SELECT", "EXECUTE"]


def test_grant_continues_after_a_failure_and_tracks_status() -> None:
    w = MagicMock()
    # First PATCH raises; the walk should warn-and-continue to the rest.
    w.api_client.do.side_effect = [RuntimeError("no GRANT rights"), None, None, None]
    config = AppConfig(schemas={"s": _schema()})
    plan = grant(w, principal=PRINCIPAL, config=config, dry_run=False)
    assert w.api_client.do.call_count >= 2  # did not abort on the first failure
    # per-grant status is tracked: the first is failed, later ones applied
    assert plan.grants[0].applied is False
    assert plan.grants[0].error
    assert any(g.applied is True for g in plan.grants[1:])


def test_grant_applies_uc_patch_for_volume_and_connection() -> None:
    from dao_ai.config import ConnectionModel, ResourcesModel, VolumeModel

    w = MagicMock()
    schema = _schema()
    config = AppConfig(
        schemas={"s": schema},
        resources=ResourcesModel(
            volumes={"v": VolumeModel(schema=schema, name="landing")},
            connections={"c": ConnectionModel(name="my_conn")},
        ),
    )
    grant(w, principal=PRINCIPAL, config=config, dry_run=False)

    calls = w.api_client.do.call_args_list
    patched = {
        c.args[1]: c.kwargs["body"]["changes"][0] for c in calls if c.args[0] == "PATCH"
    }
    assert patched["/api/2.1/unity-catalog/permissions/volume/cat.sch.landing"][
        "add"
    ] == ["READ_VOLUME"]
    assert patched["/api/2.1/unity-catalog/permissions/connection/my_conn"]["add"] == [
        "USE_CONNECTION"
    ]


def test_grant_lakebase_role_created_when_sp_matches(monkeypatch) -> None:
    import dao_ai.providers.databricks as dbx
    from dao_ai.config import ResourcesModel

    provider = MagicMock()
    monkeypatch.setattr(dbx, "DatabricksProvider", lambda w: provider)

    w = MagicMock()
    config = AppConfig(
        resources=ResourcesModel(databases={"d": _lakebase_db(PRINCIPAL)}),
    )
    plan = grant(w, principal=PRINCIPAL, config=config, dry_run=False)

    provider.create_lakebase_autoscaling_role.assert_called_once()
    lb = next(g for g in plan.grants if g.kind == "lakebase_role")
    assert lb.applied is True


def test_grant_lakebase_role_skipped_when_sp_mismatched(monkeypatch) -> None:
    import dao_ai.providers.databricks as dbx
    from dao_ai.config import ResourcesModel

    provider = MagicMock()
    monkeypatch.setattr(dbx, "DatabricksProvider", lambda w: provider)

    w = MagicMock()
    config = AppConfig(
        resources=ResourcesModel(
            databases={"d": _lakebase_db("22222222-2222-2222-2222-222222222222")}
        ),
    )
    plan = grant(w, principal=PRINCIPAL, config=config, dry_run=False)

    provider.create_lakebase_autoscaling_role.assert_not_called()
    lb = next(g for g in plan.grants if g.kind == "lakebase_role")
    # skipped (not attempted) → applied stays None, never reads as success
    assert lb.applied is None


def test_grant_lakebase_role_resolves_by_key_not_project(monkeypatch) -> None:
    """Two DBs share a project but pin different client_ids; only the matching
    one must be acted on — apply must re-resolve by config key, not project, so
    it can't pick the mismatched model and create a role for the wrong SP."""
    import dao_ai.providers.databricks as dbx
    from dao_ai.config import ResourcesModel

    provider = MagicMock()
    monkeypatch.setattr(dbx, "DatabricksProvider", lambda w: provider)

    other = "22222222-2222-2222-2222-222222222222"
    match_db = _lakebase_db(PRINCIPAL, project="shared-proj", name="match")
    mismatch_db = _lakebase_db(other, project="shared-proj", name="mismatch")
    w = MagicMock()
    config = AppConfig(
        resources=ResourcesModel(
            databases={"match": match_db, "mismatch": mismatch_db}
        ),
    )
    plan = grant(w, principal=PRINCIPAL, config=config, dry_run=False)

    # Exactly one role created, and it's the model whose client_id == principal.
    # ``client_id=None`` on the single-SP path: no ownership map, so the provider
    # falls back to reading DatabaseModel.client_id itself (legacy behaviour).
    provider.create_lakebase_autoscaling_role.assert_called_once_with(
        match_db, client_id=None
    )
    lb = {g.resource_key: g for g in plan.grants if g.kind == "lakebase_role"}
    assert lb["match"].applied is True and lb["match"].note is None
    assert lb["mismatch"].applied is None and lb["mismatch"].note


def test_grant_lakebase_role_not_created_on_dry_run(monkeypatch) -> None:
    import dao_ai.providers.databricks as dbx
    from dao_ai.config import ResourcesModel

    provider = MagicMock()
    monkeypatch.setattr(dbx, "DatabricksProvider", lambda w: provider)

    w = MagicMock()
    config = AppConfig(
        resources=ResourcesModel(databases={"d": _lakebase_db(PRINCIPAL)}),
    )
    grant(w, principal=PRINCIPAL, config=config, dry_run=True)
    provider.create_lakebase_autoscaling_role.assert_not_called()


def test_grant_warehouse_uses_additive_update_not_set() -> None:
    """set_permissions REPLACES the whole ACL — grant must use update_permissions."""
    from dao_ai.config import ResourcesModel, WarehouseModel

    w = MagicMock()
    config = AppConfig(
        resources=ResourcesModel(warehouses={"w": WarehouseModel(warehouse_id="wh-1")})
    )
    grant(w, principal=PRINCIPAL, config=config, dry_run=False)
    w.warehouses.update_permissions.assert_called_once()
    w.warehouses.set_permissions.assert_not_called()  # never the destructive variant


def test_grant_genie_uses_additive_update_not_set() -> None:
    from dao_ai.config import GenieRoomModel, ResourcesModel

    w = MagicMock()
    config = AppConfig(
        resources=ResourcesModel(
            genie_rooms={"g": GenieRoomModel(space_id="01f0space")}
        )
    )
    grant(w, principal=PRINCIPAL, config=config, dry_run=False)
    w.permissions.update.assert_called_once()
    w.permissions.set.assert_not_called()


def test_grant_serving_endpoint_uses_resolved_id_and_additive_update() -> None:
    from dao_ai.config import AgentModel, AppModel, InferenceEndpointModel

    w = MagicMock()
    # get(name=...) returns an endpoint whose id differs from its name
    w.serving_endpoints.get.return_value = MagicMock(id="ep-internal-id")
    config = AppConfig(
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
        )
    )
    grant(w, principal=PRINCIPAL, config=config, dry_run=False)
    w.serving_endpoints.set_permissions.assert_not_called()
    w.serving_endpoints.update_permissions.assert_called_once()
    # keyed on the resolved id, not the endpoint name
    assert (
        w.serving_endpoints.update_permissions.call_args.kwargs["serving_endpoint_id"]
        == "ep-internal-id"
    )


def test_grant_serving_endpoint_skips_when_endpoint_absent() -> None:
    """The serving grant is best-effort: if the endpoint isn't deployed, skip it
    (no update_permissions) rather than erroring — AppModel always populates
    endpoint_name (defaulting from name), so existence is checked at apply time."""
    from dao_ai.config import AgentModel, AppModel, InferenceEndpointModel

    w = MagicMock()
    w.serving_endpoints.get.side_effect = RuntimeError("RESOURCE_DOES_NOT_EXIST")
    config = AppConfig(
        app=AppModel(
            name="apps-only",
            agents=[
                AgentModel(
                    name="a",
                    description="d",
                    model=InferenceEndpointModel(name="databricks-gpt-5-4-mini"),
                )
            ],
        )
    )
    plan = grant(w, principal=PRINCIPAL, config=config, dry_run=False)
    # planned, but not applied (endpoint absent) — and never the destructive set
    assert any(g.kind == "serving_endpoint" for g in plan.grants)
    w.serving_endpoints.update_permissions.assert_not_called()
    w.serving_endpoints.set_permissions.assert_not_called()


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
    put_calls = {
        c.kwargs["key"]: c.kwargs["string_value"]
        for c in w.secrets.put_secret.call_args_list
    }
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
    w.service_principals.create.return_value = MagicMock(
        id="9", application_id=PRINCIPAL
    )
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


def _appmodel_no_sp(name: str = "my-agent"):
    from dao_ai.config import AgentModel, AppModel, InferenceEndpointModel

    return AppModel(
        name=name,
        agents=[
            AgentModel(
                name="a",
                description="d",
                model=InferenceEndpointModel(name="databricks-gpt-5-4-mini"),
            )
        ],
    )


def test_provision_without_sp_block_requires_explicit_keys() -> None:
    """No service_principals block + no key overrides → error, never guess keys."""
    import pytest

    w = MagicMock()
    w.service_principals.list.return_value = []
    w.service_principals.create.return_value = MagicMock(
        id="9", application_id=PRINCIPAL
    )
    w.service_principal_secrets_proxy.create.return_value = MagicMock(secret="x")

    config = AppConfig(app=_appmodel_no_sp())
    with pytest.raises(ValueError, match="which secret keys"):
        provision(w, config=config, display_name="my-sp")
    # fail-fast: validated before creating anything, so no orphaned SP
    w.service_principals.create.assert_not_called()


def test_provision_without_sp_block_succeeds_with_explicit_keys() -> None:
    """Explicit --scope/--*-key overrides make provision work on any config."""
    w = MagicMock()
    w.service_principals.list.return_value = []
    w.service_principals.create.return_value = MagicMock(
        id="9", application_id=PRINCIPAL
    )
    w.service_principal_secrets_proxy.create.return_value = MagicMock(secret="x")

    config = AppConfig(app=_appmodel_no_sp())
    result = provision(
        w,
        config=config,
        display_name="my-sp",
        scope="my-scope",
        client_id_key="CID",
        client_secret_key="CSEC",
    )
    assert result.stored is True
    assert result.stored_scope == "my-scope"
    put = {c.kwargs["key"] for c in w.secrets.put_secret.call_args_list}
    assert put == {"CID", "CSEC"}


def test_provision_no_store_no_grant_flags() -> None:
    w = MagicMock()
    w.service_principals.list.return_value = []
    w.service_principals.create.return_value = MagicMock(
        id="9", application_id=PRINCIPAL
    )
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


def test_resolve_secret_target_from_variables_block() -> None:
    """Configs that wire creds via top-level `variables:` (no service_principals)."""
    from dao_ai.config import CompositeVariableModel, SecretVariableModel

    config = AppConfig(
        variables={
            "client_id": CompositeVariableModel(
                options=[SecretVariableModel(scope="rcg", secret="RETAIL_AI_CLIENT_ID")]
            ),
            "client_secret": CompositeVariableModel(
                options=[
                    SecretVariableModel(scope="rcg", secret="RETAIL_AI_CLIENT_SECRET")
                ]
            ),
        }
    )
    scope, cid_key, csec_key = resolve_secret_target(config)
    assert scope == "rcg"
    assert cid_key == "RETAIL_AI_CLIENT_ID"
    assert csec_key == "RETAIL_AI_CLIENT_SECRET"


def test_resolve_secret_target_does_not_guess_from_environment_vars() -> None:
    """environment_vars is NOT string-matched to infer keys — that would be a guess."""
    from dao_ai.config import AgentModel, AppModel, InferenceEndpointModel

    config = AppConfig(
        app=AppModel(
            name="hw",
            agents=[
                AgentModel(
                    name="a",
                    description="d",
                    model=InferenceEndpointModel(name="databricks-gpt-5-4-mini"),
                )
            ],
            environment_vars={
                "RETAIL_AI_DATABRICKS_CLIENT_ID": "{{secrets/rcg/RETAIL_AI_DATABRICKS_CLIENT_ID}}",
                "RETAIL_AI_DATABRICKS_CLIENT_SECRET": "{{secrets/rcg/RETAIL_AI_DATABRICKS_CLIENT_SECRET}}",
            },
        )
    )
    # No service_principals block and no client_id/client_secret variables → unresolved.
    scope, cid_key, csec_key = resolve_secret_target(config)
    assert cid_key is None
    assert csec_key is None


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


# =============================================================================
# classify_grant_error — absent vs denied vs error
# =============================================================================


def test_classify_grant_error_absent_from_sdk_not_found() -> None:
    from databricks.sdk.errors import NotFound

    assert classify_grant_error(NotFound("nope")) == GRANT_FAILURE_ABSENT


def test_classify_grant_error_absent_from_resource_does_not_exist() -> None:
    """``ResourceDoesNotExist`` subclasses ``NotFound`` — same bucket."""
    from databricks.sdk.errors import ResourceDoesNotExist

    assert classify_grant_error(ResourceDoesNotExist("gone")) == GRANT_FAILURE_ABSENT


def test_classify_grant_error_absent_from_catalog_does_not_exist_message() -> None:
    """The real-world shape: a UC PATCH against a catalog that isn't there."""
    err = RuntimeError("Catalog 'hardware_store' does not exist.")
    assert classify_grant_error(err) == GRANT_FAILURE_ABSENT


def test_classify_grant_error_denied_from_sdk_permission_denied() -> None:
    from databricks.sdk.errors import PermissionDenied

    assert classify_grant_error(PermissionDenied("no")) == GRANT_FAILURE_DENIED


def test_classify_grant_error_denied_from_not_authorized_message() -> None:
    """The Lakebase 'Can Manage' shape, which is not an SDK-typed error here."""
    err = RuntimeError(
        "The user is not authorized to make the request, please contact the "
        "workspace admin to assign 'Can Manage' for Database project"
    )
    assert classify_grant_error(err) == GRANT_FAILURE_DENIED


def test_classify_grant_error_falls_back_to_generic() -> None:
    assert classify_grant_error(RuntimeError("kaboom")) == GRANT_FAILURE_ERROR


def test_grant_records_failure_kind_absent_for_missing_catalog() -> None:
    """A failed grant carries both the error text and its classification."""
    from databricks.sdk.errors import NotFound

    w = MagicMock()
    w.api_client.do.side_effect = NotFound("Catalog 'nope' does not exist.")
    config = AppConfig(schemas={"s": _schema()})
    plan = grant(w, principal=PRINCIPAL, config=config, dry_run=False)

    catalog_grant = next(g for g in plan.grants if g.securable_type == "catalog")
    assert catalog_grant.applied is False
    assert catalog_grant.failure_kind == GRANT_FAILURE_ABSENT
    assert "does not exist" in catalog_grant.error


def test_grant_records_failure_kind_denied() -> None:
    from databricks.sdk.errors import PermissionDenied

    w = MagicMock()
    w.api_client.do.side_effect = PermissionDenied("PERMISSION_DENIED on catalog")
    config = AppConfig(schemas={"s": _schema()})
    plan = grant(w, principal=PRINCIPAL, config=config, dry_run=False)

    catalog_grant = next(g for g in plan.grants if g.securable_type == "catalog")
    assert catalog_grant.failure_kind == GRANT_FAILURE_DENIED


# =============================================================================
# ownership — which declared SP owns which resource
# =============================================================================


def _target(
    name: str,
    scope: str = "sc",
    cid_key: str = "CID",
    csec_key: str = "CSEC",
    model: object = None,
    configured: str | None = None,
    resolved: str | None = None,
) -> ServicePrincipalTarget:
    return ServicePrincipalTarget(
        name=name,
        display_name=f"app-{name}",
        model=model,
        scope=scope,
        client_id_key=cid_key,
        client_secret_key=csec_key,
        client_id_ref=SecretRef(scope=scope, key=cid_key),
        client_secret_ref=SecretRef(scope=scope, key=csec_key),
        configured_client_id=configured,
        resolved_client_id=resolved,
    )


def test_ownership_matches_resource_by_secret_ref() -> None:
    """The load-bearing case: same (scope, key) → same owner. No API calls."""
    schema = _schema()
    table = TableModel(
        schema=schema,
        name="t",
        client_id=SecretVariableModel(scope="sc", secret="CID"),
        client_secret=SecretVariableModel(scope="sc", secret="CSEC"),
    )
    targets = [_target("memory_sp"), _target("tools_sp", cid_key="T_CID")]
    assert resource_owner(table, targets) == "memory_sp"


def test_ownership_matches_resource_by_shared_service_principal_model() -> None:
    """A shared YAML anchor re-validates into distinct objects but compares equal."""
    sp = ServicePrincipalModel(
        client_id=SecretVariableModel(scope="sc", secret="CID"),
        client_secret=SecretVariableModel(scope="sc", secret="CSEC"),
    )
    schema = _schema()
    table = TableModel(schema=schema, name="t", service_principal=sp)
    targets = [_target("tools_sp", cid_key="T_CID"), _target("memory_sp", model=sp)]
    assert resource_owner(table, targets) == "memory_sp"


def test_ownership_matches_resource_by_literal_client_id() -> None:
    """A literal client_id falls through to value comparison."""
    schema = _schema()
    table = TableModel(schema=schema, name="t", client_id=PRINCIPAL, client_secret="s")
    targets = [_target("memory_sp", resolved=PRINCIPAL)]
    assert resource_owner(table, targets) == "memory_sp"


def test_ownership_treats_resource_without_client_id_as_shared() -> None:
    table = TableModel(schema=_schema(), name="t")
    assert resource_owner(table, [_target("memory_sp")]) is None


def test_ownership_unmatched_secret_ref_is_shared_without_reading_it(
    monkeypatch,
) -> None:
    """A secret-backed client_id matching no target must NOT be resolved.

    This is the case that actually leaks API calls: the ref lookup misses, and a
    naive implementation falls through to ``value_of`` — a live, uncached
    ``get_secret`` — for every unowned resource on every target.
    """

    def _boom(*args, **kwargs):
        raise AssertionError(
            "an unmatched secret-backed client_id must not be resolved"
        )

    monkeypatch.setattr("dao_ai.config.SecretVariableModel.as_value", _boom)
    table = TableModel(
        schema=_schema(),
        name="t",
        client_id=SecretVariableModel(scope="other", secret="NOPE"),
        client_secret=SecretVariableModel(scope="other", secret="NOPE2"),
    )
    assert resource_owner(table, [_target("memory_sp")]) is None


def test_ownership_secret_ref_match_never_reads_the_secret(monkeypatch) -> None:
    """Guardrail: value_of() on a secret is a live uncached get_secret call.

    Matching by (scope, key) must not trigger it — otherwise one API call
    becomes N_resources x N_sps per plan.
    """

    def _boom(*args, **kwargs):
        raise AssertionError("ownership matching must not read secret values")

    monkeypatch.setattr("dao_ai.config.SecretVariableModel.as_value", _boom)
    table = TableModel(
        schema=_schema(),
        name="t",
        client_id=SecretVariableModel(scope="sc", secret="CID"),
        client_secret=SecretVariableModel(scope="sc", secret="CSEC"),
    )
    assert resource_owner(table, [_target("memory_sp")]) == "memory_sp"


def test_ownership_matches_fresh_sp_before_client_id_is_resolvable() -> None:
    """A brand-new SP has no id yet; the secret ref still identifies its resources."""
    table = TableModel(
        schema=_schema(),
        name="t",
        client_id=SecretVariableModel(scope="sc", secret="CID"),
        client_secret=SecretVariableModel(scope="sc", secret="CSEC"),
    )
    fresh = _target("memory_sp", configured=None, resolved=None)
    assert resource_owner(table, [fresh]) == "memory_sp"


def test_ownership_map_records_only_owned_resources() -> None:
    from dao_ai.config import ResourcesModel

    schema = _schema()
    owned = TableModel(
        schema=schema,
        name="owned",
        client_id=SecretVariableModel(scope="sc", secret="CID"),
        client_secret=SecretVariableModel(scope="sc", secret="CSEC"),
    )
    shared = TableModel(schema=schema, name="shared")
    config = AppConfig(
        schemas={"s": schema},
        resources=ResourcesModel(tables={"owned": owned, "shared": shared}),
    )
    ownership = build_ownership_map(config, [_target("memory_sp")])

    assert ownership.owner_of("tables", "owned") == "memory_sp"
    assert ownership.owner_of("tables", "shared") is None
    # Shared resources are granted to every SP; owned ones only to their owner.
    assert ownership.owns("tables", "shared", "memory_sp")
    assert ownership.owns("tables", "shared", "tools_sp")
    assert ownership.owns("tables", "owned", "memory_sp")
    assert not ownership.owns("tables", "owned", "tools_sp")


def test_ownership_map_covers_databases() -> None:
    from dao_ai.config import ResourcesModel

    db = _lakebase_db(PRINCIPAL)
    config = AppConfig(resources=ResourcesModel(databases={"d": db}))
    ownership = build_ownership_map(config, [_target("memory_sp", resolved=PRINCIPAL)])
    assert ownership.owner_of("databases", "d") == "memory_sp"


def test_ownership_rejects_two_sps_sharing_one_client_id_key() -> None:
    import pytest

    config = AppConfig()
    targets = [_target("a"), _target("b")]  # same scope + CID key
    with pytest.raises(ValueError, match="both read"):
        build_ownership_map(config, targets)


def test_ownership_map_empty_when_no_resources() -> None:
    ownership = build_ownership_map(AppConfig(), [_target("memory_sp")])
    assert ownership.owners == {}
    # Everything is shared when nothing is owned.
    assert ownership.owns("tables", "anything", "memory_sp")


# =============================================================================
# build_grant_plan — ownership filtering (multi-SP)
# =============================================================================


def _owned_table(name: str, cid_key: str = "CID"):
    schema = _schema()
    return TableModel(
        schema=schema,
        name=name,
        client_id=SecretVariableModel(scope="sc", secret=cid_key),
        client_secret=SecretVariableModel(scope="sc", secret="CSEC"),
    )


def test_grant_plan_without_ownership_is_unchanged() -> None:
    """Backward compat: ownership=None grants the whole config, as before."""
    from dao_ai.config import ResourcesModel

    schema = _schema()
    config = AppConfig(
        schemas={"s": schema},
        resources=ResourcesModel(
            tables={
                "a": _owned_table("a"),
                "b": TableModel(schema=schema, name="b"),
            }
        ),
    )
    plan = build_grant_plan(config, PRINCIPAL)
    targets_granted = {g.target for g in plan.grants}
    assert "cat.sch.a" in targets_granted
    assert "cat.sch.b" in targets_granted


def test_grant_plan_owned_table_granted_only_to_its_owner() -> None:
    from dao_ai.config import ResourcesModel

    config = AppConfig(
        schemas={"s": _schema()},
        resources=ResourcesModel(tables={"a": _owned_table("a")}),
    )
    targets = [_target("memory_sp"), _target("tools_sp", cid_key="T_CID")]
    ownership = build_ownership_map(config, targets)

    owner_plan = build_grant_plan(
        config, PRINCIPAL, ownership=ownership, sp_name="memory_sp", targets=targets
    )
    other_plan = build_grant_plan(
        config, PRINCIPAL, ownership=ownership, sp_name="tools_sp", targets=targets
    )
    assert "cat.sch.a" in {g.target for g in owner_plan.grants}
    assert "cat.sch.a" not in {g.target for g in other_plan.grants}


def test_grant_plan_shared_resource_granted_to_every_sp() -> None:
    from dao_ai.config import ResourcesModel

    config = AppConfig(
        schemas={"s": _schema()},
        resources=ResourcesModel(
            tables={"shared": TableModel(schema=_schema(), name="shared")}
        ),
    )
    targets = [_target("memory_sp"), _target("tools_sp", cid_key="T_CID")]
    ownership = build_ownership_map(config, targets)

    for sp_name in ("memory_sp", "tools_sp"):
        plan = build_grant_plan(
            config, PRINCIPAL, ownership=ownership, sp_name=sp_name, targets=targets
        )
        assert "cat.sch.shared" in {g.target for g in plan.grants}


def test_grant_plan_owned_resource_still_grants_its_catalog_and_schema() -> None:
    """An owner needs USE_CATALOG/USE_SCHEMA to reach the table it owns."""
    from dao_ai.config import ResourcesModel

    config = AppConfig(resources=ResourcesModel(tables={"a": _owned_table("a")}))
    targets = [_target("memory_sp")]
    ownership = build_ownership_map(config, targets)
    plan = build_grant_plan(
        config, PRINCIPAL, ownership=ownership, sp_name="memory_sp", targets=targets
    )
    by_type = {g.securable_type for g in plan.grants}
    assert "catalog" in by_type and "schema" in by_type


def test_grant_plan_top_level_schemas_are_shared_across_sps() -> None:
    """Catalogs/schemas are shared infrastructure — never owned by one SP."""
    config = AppConfig(schemas={"s": _schema()})
    targets = [_target("memory_sp"), _target("tools_sp", cid_key="T_CID")]
    ownership = build_ownership_map(config, targets)

    for sp_name in ("memory_sp", "tools_sp"):
        plan = build_grant_plan(
            config, PRINCIPAL, ownership=ownership, sp_name=sp_name, targets=targets
        )
        assert ("catalog", "cat") in {(g.securable_type, g.target) for g in plan.grants}


def test_grant_plan_lakebase_role_planned_only_for_owning_sp() -> None:
    """The non-owner gets NO lakebase entry at all — not even a SKIP note.

    With N service principals, a per-SP mismatch note would print N-1 alarming
    SKIPs per project per run; the owner's grant already tells the story.
    """
    from dao_ai.config import ResourcesModel

    db = _lakebase_db(PRINCIPAL)
    config = AppConfig(resources=ResourcesModel(databases={"d": db}))
    targets = [
        _target("memory_sp", resolved=PRINCIPAL),
        _target(
            "tools_sp", cid_key="T_CID", resolved="99999999-9999-9999-9999-999999999999"
        ),
    ]
    ownership = build_ownership_map(config, targets)

    owner_plan = build_grant_plan(
        config, PRINCIPAL, ownership=ownership, sp_name="memory_sp", targets=targets
    )
    other_plan = build_grant_plan(
        config, PRINCIPAL, ownership=ownership, sp_name="tools_sp", targets=targets
    )
    owner_lb = [g for g in owner_plan.grants if g.kind == "lakebase_role"]
    other_lb = [g for g in other_plan.grants if g.kind == "lakebase_role"]

    assert len(owner_lb) == 1 and owner_lb[0].note is None
    assert other_lb == []


def test_grant_plan_lakebase_role_carries_owning_sp_client_id() -> None:
    """The role subject comes from the target, not from re-reading the config."""
    from dao_ai.config import ResourcesModel

    db = _lakebase_db(PRINCIPAL)
    config = AppConfig(resources=ResourcesModel(databases={"d": db}))
    # The target's resolved id deliberately differs from the ``principal``
    # argument, so this can only pass if the target — not the argument — is the
    # source of the role subject.
    from_target = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    targets = [_target("memory_sp", resolved=from_target)]
    ownership = build_ownership_map(config, targets)
    plan = build_grant_plan(
        config, PRINCIPAL, ownership=ownership, sp_name="memory_sp", targets=targets
    )
    lb = next(g for g in plan.grants if g.kind == "lakebase_role")
    assert lb.principal_override == from_target


def test_grant_plan_lakebase_role_planned_when_scope_unpopulated(monkeypatch) -> None:
    """THE ONE-PASS PROOF.

    A brand-new SP's client_id is not in the secret scope yet, so
    ``value_of(database.client_id)`` yields nothing. Under ownership the role is
    still planned, with the freshly minted id as its subject — no second run.
    """
    from dao_ai.config import DatabaseModel, ResourcesModel

    fresh_id = "de6db65b-59f0-4368-87ed-9b06f6054da0"
    db = DatabaseModel(
        name="lb",
        project="my-proj",
        client_id=SecretVariableModel(scope="sc", secret="CID"),
        client_secret=SecretVariableModel(scope="sc", secret="CSEC"),
    )
    config = AppConfig(resources=ResourcesModel(databases={"d": db}))
    targets = [_target("memory_sp", configured=None, resolved=fresh_id)]
    ownership = build_ownership_map(config, targets)

    # Reading the secret would fail here (scope not populated) — the plan must
    # not need it.
    def _boom(*args, **kwargs):
        raise AssertionError("the one-pass path must not read the secret scope")

    monkeypatch.setattr("dao_ai.config.SecretVariableModel.as_value", _boom)
    plan = build_grant_plan(
        config, fresh_id, ownership=ownership, sp_name="memory_sp", targets=targets
    )
    lb = next(g for g in plan.grants if g.kind == "lakebase_role")
    assert lb.note is None
    assert lb.principal_override == fresh_id


def test_grant_plan_dry_run_placeholder_principal_emits_no_skip_note() -> None:
    """A dry run of a not-yet-created SP must not print a bogus mismatch SKIP."""
    from dao_ai.config import ResourcesModel
    from dao_ai.service_principal import placeholder_principal

    db = _lakebase_db(PRINCIPAL)
    config = AppConfig(resources=ResourcesModel(databases={"d": db}))
    targets = [_target("memory_sp", resolved=None, configured=None)]
    ownership = build_ownership_map(config, targets)
    plan = build_grant_plan(
        config,
        placeholder_principal("memory_sp"),
        ownership=ownership,
        sp_name="memory_sp",
        targets=targets,
    )
    lb = next(g for g in plan.grants if g.kind == "lakebase_role")
    # It reports "no id yet", which is honest, and never claims an id mismatch.
    assert lb.note is not None
    assert "does not exist yet" in lb.note
    assert "configured for client_id" not in lb.note


# =============================================================================
# secret existence probe + --overwrite + dry-run plumbing
# =============================================================================


def _ws_with_secrets(scope: str = "sc", keys: tuple[str, ...] = ()) -> MagicMock:
    """A mock WorkspaceClient whose ``scope`` holds ``keys``."""
    w = MagicMock()
    w.secrets.list_secrets.return_value = [MagicMock(**{"key": k}) for k in keys]
    # ``name`` is a reserved MagicMock constructor kwarg (it names the mock), so
    # the attribute has to be assigned after construction.
    scope_obj = MagicMock()
    scope_obj.name = scope
    w.secrets.list_scopes.return_value = [scope_obj]
    return w


def test_secret_keys_present_lists_existing_keys() -> None:
    w = _ws_with_secrets(keys=("CID", "CSEC"))
    assert secret_keys_present(w, "sc") == frozenset({"CID", "CSEC"})
    # Never reads a value — list_secrets returns metadata only.
    w.secrets.get_secret.assert_not_called()


def test_secret_keys_present_returns_empty_set_for_absent_scope() -> None:
    from databricks.sdk.errors import ResourceDoesNotExist

    w = MagicMock()
    w.secrets.list_secrets.side_effect = ResourceDoesNotExist(
        "Scope nope does not exist!"
    )
    assert secret_keys_present(w, "nope") == frozenset()


def test_store_overwrites_by_default() -> None:
    """Default overwrite=True preserves the original unconditional behaviour."""
    w = _ws_with_secrets(keys=("CID", "CSEC"))
    result = store(
        w,
        scope="sc",
        client_id_key="CID",
        client_secret_key="CSEC",
        client_id="new-id",
        client_secret="new-secret",
    )
    assert result.written == ["CID", "CSEC"]
    assert result.skipped == []
    assert w.secrets.put_secret.call_count == 2


def test_store_refuses_existing_keys_without_overwrite() -> None:
    w = _ws_with_secrets(keys=("CID", "CSEC"))
    result = store(
        w,
        scope="sc",
        client_id_key="CID",
        client_secret_key="CSEC",
        client_id="new-id",
        client_secret="new-secret",
        overwrite=False,
    )
    assert result.written == []
    assert result.skipped == ["CID", "CSEC"]
    w.secrets.put_secret.assert_not_called()
    # A live credential must not be clobbered, and the scope is left alone.
    w.secrets.create_scope.assert_not_called()


def test_store_writes_absent_keys_without_overwrite() -> None:
    """Only the keys that already hold a value are protected."""
    w = _ws_with_secrets(keys=("CID",))
    result = store(
        w,
        scope="sc",
        client_id_key="CID",
        client_secret_key="CSEC",
        client_id="i",
        client_secret="s",
        overwrite=False,
    )
    assert result.written == ["CSEC"]
    assert result.skipped == ["CID"]


def test_store_dry_run_writes_nothing() -> None:
    w = _ws_with_secrets(keys=())
    result = store(
        w,
        scope="sc",
        client_id_key="CID",
        client_secret_key="CSEC",
        client_id="i",
        client_secret="s",
        dry_run=True,
    )
    assert result.written == ["CID", "CSEC"]
    w.secrets.put_secret.assert_not_called()
    w.secrets.create_scope.assert_not_called()


def test_store_dry_run_reports_whether_scope_exists() -> None:
    w = _ws_with_secrets(scope="sc", keys=())
    assert store(
        w,
        scope="sc",
        client_id_key="CID",
        client_secret_key="CSEC",
        client_id="i",
        client_secret="s",
        dry_run=True,
    ).scope_existed
    assert not store(
        w,
        scope="other",
        client_id_key="CID",
        client_secret_key="CSEC",
        client_id="i",
        client_secret="s",
        dry_run=True,
    ).scope_existed


def test_find_service_principal_mints_nothing() -> None:
    w = MagicMock()
    w.service_principals.list.return_value = [
        MagicMock(**{"display_name": "app-sp", "application_id": PRINCIPAL, "id": "42"})
    ]
    found = find_service_principal(w, display_name="app-sp")
    assert found is not None and found.application_id == PRINCIPAL
    w.service_principals.create.assert_not_called()
    w.service_principal_secrets_proxy.create.assert_not_called()


def test_find_service_principal_returns_none_when_absent() -> None:
    w = MagicMock()
    w.service_principals.list.return_value = []
    assert find_service_principal(w, display_name="nope") is None


def test_create_dry_run_mints_no_secret_for_existing_sp() -> None:
    """Minting registers a new OAuth secret — itself a mutation. Dry run must not."""
    w = MagicMock()
    w.service_principals.list.return_value = [
        MagicMock(**{"display_name": "app-sp", "application_id": PRINCIPAL, "id": "42"})
    ]
    result = create(w, display_name="app-sp", dry_run=True)

    assert result.reused is True
    assert result.client_id == PRINCIPAL
    assert result.client_secret is None
    w.service_principals.create.assert_not_called()
    w.service_principal_secrets_proxy.create.assert_not_called()


def test_create_dry_run_reports_would_create_for_absent_sp() -> None:
    w = MagicMock()
    w.service_principals.list.return_value = []
    result = create(w, display_name="brand-new", dry_run=True)

    assert result.reused is False
    assert result.client_id == ""  # unknown until actually created
    assert result.client_secret is None
    w.service_principals.create.assert_not_called()
    w.service_principal_secrets_proxy.create.assert_not_called()
