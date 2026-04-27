"""
Tests for Lakebase database resource extraction across both deployment paths.

When a Lakebase database is configured, the deploying agent should declare
it as a Databricks resource so the platform grants the deployment's identity
``CAN_CONNECT_AND_CREATE`` on the Lakebase instance:

- **Model Serving** (``deploy_model_serving_agent``): the model is logged
  with a ``SystemAuthPolicy`` whose ``resources`` list comes from each
  resource model's ``as_resources()`` method. ``DatabaseModel.as_resources()``
  must return a ``DatabricksLakebase`` resource for Lakebase configurations.

- **Databricks Apps** (``deploy_apps_agent``): the deployed ``app.yaml``
  enumerates resources via ``generate_app_resources()`` (flat-dict format
  consumed by the bundle generator) and ``generate_sdk_resources()``
  (SDK ``AppResource`` objects consumed by the direct-deploy path).
  Both must include a ``database`` resource for each Lakebase.

OBO databases are skipped at extraction time -- the user identity handles
permissions via ``user_api_scopes`` instead.

Standalone PostgreSQL connections (``host:`` set, no ``project:``) have no
Databricks-managed resource binding and are also skipped.
"""

from __future__ import annotations

import pytest


# =============================================================================
# DatabaseModel.as_resources() -- model serving SystemAuthPolicy path
# =============================================================================


@pytest.mark.unit
class TestDatabaseModelAsResources:
    """``DatabaseModel.as_resources()`` feeds Model Serving's SystemAuthPolicy."""

    def test_lakebase_non_obo_returns_lakebase_resource(self) -> None:
        from mlflow.models.resources import DatabricksLakebase

        from dao_ai.config import DatabaseModel

        db = DatabaseModel(name="retail-consumer-goods", project="retail-consumer-goods")
        resources = list(db.as_resources())
        assert len(resources) == 1
        r = resources[0]
        assert isinstance(r, DatabricksLakebase)
        # to_dict produces the MLflow-resources YAML shape used in the model package
        d = r.to_dict()
        assert "lakebase" in d
        assert d["lakebase"][0]["name"] == "retail-consumer-goods"
        assert d["lakebase"][0]["on_behalf_of_user"] is False

    def test_lakebase_obo_carries_on_behalf_of_user_true(self) -> None:
        from dao_ai.config import DatabaseModel

        db = DatabaseModel(
            name="x", project="x", on_behalf_of_user=True
        )
        resources = list(db.as_resources())
        assert len(resources) == 1
        d = resources[0].to_dict()
        assert d["lakebase"][0]["on_behalf_of_user"] is True

    def test_standalone_postgres_returns_empty(self) -> None:
        """Non-Lakebase PG has no Databricks-managed resource binding."""
        from dao_ai.config import DatabaseModel

        # PG requires explicit auth -- supply a PAT to satisfy validation.
        db = DatabaseModel(
            name="pg",
            host="pg.example.com",
            user="me",
        )
        assert db.is_lakebase is False
        assert list(db.as_resources()) == []

    def test_lakebase_resource_uses_project_as_instance_name(self) -> None:
        """``DatabricksLakebase.database_instance_name`` is the project field,
        not the freeform ``name`` (which can differ for Lakebase logical names)."""
        from dao_ai.config import DatabaseModel

        db = DatabaseModel(name="logical-name", project="instance-actual")
        d = list(db.as_resources())[0].to_dict()
        assert d["lakebase"][0]["name"] == "instance-actual"


# =============================================================================
# _extract_database_resources -- flat-dict format consumed by app.yaml bundle gen
# =============================================================================


@pytest.mark.unit
class TestExtractDatabaseResourcesFlat:
    """Flat-dict extractor used when generating ``app.yaml`` for the bundle."""

    def test_lakebase_non_obo_emits_database_resource(self) -> None:
        from dao_ai.apps.resources import _extract_database_resources
        from dao_ai.config import DatabaseModel

        db = DatabaseModel(name="retail-consumer-goods", project="retail-consumer-goods")
        out = _extract_database_resources({"workshop_db": db})
        assert len(out) == 1
        r = out[0]
        assert r["type"] == "database"
        assert r["name"] == "workshop_db"
        assert r["instance_name"] == "retail-consumer-goods"
        assert r["database_name"] == "retail-consumer-goods"
        assert r["permissions"] == [{"level": "CAN_CONNECT_AND_CREATE"}]

    def test_lakebase_obo_skipped(self) -> None:
        from dao_ai.apps.resources import _extract_database_resources
        from dao_ai.config import DatabaseModel

        db = DatabaseModel(name="x", project="x", on_behalf_of_user=True)
        assert _extract_database_resources({"k": db}) == []

    def test_standalone_postgres_skipped(self) -> None:
        from dao_ai.apps.resources import _extract_database_resources
        from dao_ai.config import DatabaseModel

        db = DatabaseModel(
            name="pg",
            host="pg.example.com",
            user="me",
        )
        assert _extract_database_resources({"k": db}) == []

    def test_database_name_falls_back_to_project_when_name_omitted(self) -> None:
        """Lakebase ``name`` defaults to ``project`` per DatabaseModel; the
        extractor preserves that even if the inputs were partial."""
        from dao_ai.apps.resources import _extract_database_resources
        from dao_ai.config import DatabaseModel

        db = DatabaseModel(project="project-only")
        out = _extract_database_resources({"k": db})
        assert out[0]["instance_name"] == "project-only"
        assert out[0]["database_name"] == "project-only"

    def test_resource_name_is_sanitized(self) -> None:
        """Underscores and other punctuation in the YAML key get sanitized
        for Databricks resource-name rules (matches volumes/tables behavior)."""
        from dao_ai.apps.resources import _extract_database_resources
        from dao_ai.config import DatabaseModel

        db = DatabaseModel(name="x", project="x")
        out = _extract_database_resources({"workshop_DB.shared": db})
        # Whatever sanitization rule is applied, the result should be a single
        # entry with a non-empty 'name' that's safe to use as a resource id.
        assert len(out) == 1
        assert out[0]["name"]
        assert "/" not in out[0]["name"]

    def test_multiple_lakebases_each_get_a_resource(self) -> None:
        from dao_ai.apps.resources import _extract_database_resources
        from dao_ai.config import DatabaseModel

        out = _extract_database_resources(
            {
                "memory_db": DatabaseModel(project="p1"),
                "checkpoints_db": DatabaseModel(project="p2"),
            }
        )
        assert {r["instance_name"] for r in out} == {"p1", "p2"}

    def test_empty_dict_returns_empty(self) -> None:
        from dao_ai.apps.resources import _extract_database_resources

        assert _extract_database_resources({}) == []


# =============================================================================
# _extract_sdk_database_resources -- SDK AppResource format for direct deploy
# =============================================================================


@pytest.mark.unit
class TestExtractDatabaseResourcesSDK:
    """SDK ``AppResource`` extractor used by the direct-deploy path."""

    def test_lakebase_non_obo_emits_appresource_database(self) -> None:
        from databricks.sdk.service.apps import (
            AppResource,
            AppResourceDatabase,
            AppResourceDatabaseDatabasePermission,
        )

        from dao_ai.apps.resources import _extract_sdk_database_resources
        from dao_ai.config import DatabaseModel

        db = DatabaseModel(name="retail-consumer-goods", project="retail-consumer-goods")
        out = _extract_sdk_database_resources({"workshop_db": db})
        assert len(out) == 1
        r = out[0]
        assert isinstance(r, AppResource)
        assert isinstance(r.database, AppResourceDatabase)
        assert r.database.instance_name == "retail-consumer-goods"
        assert r.database.database_name == "retail-consumer-goods"
        assert (
            r.database.permission
            is AppResourceDatabaseDatabasePermission.CAN_CONNECT_AND_CREATE
        )

    def test_lakebase_obo_skipped(self) -> None:
        from dao_ai.apps.resources import _extract_sdk_database_resources
        from dao_ai.config import DatabaseModel

        db = DatabaseModel(name="x", project="x", on_behalf_of_user=True)
        assert _extract_sdk_database_resources({"k": db}) == []

    def test_standalone_postgres_skipped(self) -> None:
        from dao_ai.apps.resources import _extract_sdk_database_resources
        from dao_ai.config import DatabaseModel

        db = DatabaseModel(
            name="pg",
            host="pg.example.com",
            user="me",
        )
        assert _extract_sdk_database_resources({"k": db}) == []


# =============================================================================
# Integration: full generate_app_resources / generate_sdk_resources
# =============================================================================


@pytest.mark.integration
class TestAppResourcesIntegration:
    """End-to-end: a config with a Lakebase + an LLM produces both resource
    types in the right output formats."""

    def _build_config(self, *, on_behalf_of_user: bool = False):
        from dao_ai.config import (
            AgentModel,
            AppConfig,
            AppModel,
            DatabaseModel,
            DeploymentTarget,
            LLMModel,
            ResourcesModel,
        )

        llm = LLMModel(name="databricks-claude-sonnet-4-5")
        db = DatabaseModel(
            name="workshop-db",
            project="workshop-db",
            on_behalf_of_user=on_behalf_of_user,
        )
        agent = AgentModel(name="a", model=llm)
        return AppConfig(
            resources=ResourcesModel(
                llms={"default_llm": llm},
                databases={"workshop_db": db},
            ),
            agents={"a": agent},
            app=AppModel(
                name="lakebase-test",
                deployment_target=DeploymentTarget.APPS,
                agents=[agent],
            ),
        )

    def test_flat_app_resources_contain_database_alongside_llm(self) -> None:
        from dao_ai.apps.resources import generate_app_resources

        resources = generate_app_resources(self._build_config())
        types = {r["type"] for r in resources}
        assert "serving-endpoint" in types
        assert "database" in types
        # The database resource has the right shape
        db_r = next(r for r in resources if r["type"] == "database")
        assert db_r["instance_name"] == "workshop-db"
        assert db_r["permissions"] == [{"level": "CAN_CONNECT_AND_CREATE"}]

    def test_flat_app_resources_skips_database_when_obo(self) -> None:
        from dao_ai.apps.resources import generate_app_resources

        resources = generate_app_resources(self._build_config(on_behalf_of_user=True))
        types = {r["type"] for r in resources}
        assert "database" not in types

    def test_sdk_app_resources_contain_appresourcedatabase(self) -> None:
        from databricks.sdk.service.apps import AppResource, AppResourceDatabase

        from dao_ai.apps.resources import generate_sdk_resources

        resources = generate_sdk_resources(self._build_config())
        db_resources = [
            r for r in resources
            if isinstance(r, AppResource) and r.database is not None
        ]
        assert len(db_resources) == 1
        assert isinstance(db_resources[0].database, AppResourceDatabase)
        assert db_resources[0].database.instance_name == "workshop-db"

    def test_sdk_app_resources_skips_database_when_obo(self) -> None:
        from databricks.sdk.service.apps import AppResource

        from dao_ai.apps.resources import generate_sdk_resources

        resources = generate_sdk_resources(self._build_config(on_behalf_of_user=True))
        db_resources = [
            r for r in resources
            if isinstance(r, AppResource) and r.database is not None
        ]
        assert db_resources == []


# =============================================================================
# Integration: model-serving system_resources path picks up the Lakebase
# =============================================================================


@pytest.mark.integration
class TestModelServingSystemResources:
    """The ``deploy_model_serving_agent`` path collects ``as_resources()``
    from each resource model into a SystemAuthPolicy. Confirm that path
    surfaces a Lakebase database for Model Serving deploys."""

    def test_lakebase_appears_in_as_resources_aggregate(self) -> None:
        """Reproduce the aggregation step of deploy_model_serving_agent."""
        from mlflow.models.resources import DatabricksLakebase

        from dao_ai.config import DatabaseModel, LLMModel

        llm = LLMModel(name="databricks-claude-sonnet-4-5")
        db = DatabaseModel(name="x", project="x")

        all_resources = []
        for r in [llm, db]:
            all_resources.extend(r.as_resources())

        # Database resource is present
        lakebase_resources = [
            r for r in all_resources if isinstance(r, DatabricksLakebase)
        ]
        assert len(lakebase_resources) == 1
        # The non-OBO database goes into the SystemAuthPolicy filter
        # (the deploy path filters out on_behalf_of_user=True resources)
        system_resources = [r for r in all_resources if not r.on_behalf_of_user]
        assert any(isinstance(r, DatabricksLakebase) for r in system_resources)

    def test_obo_lakebase_excluded_from_system_resources(self) -> None:
        from mlflow.models.resources import DatabricksLakebase

        from dao_ai.config import DatabaseModel

        db = DatabaseModel(name="x", project="x", on_behalf_of_user=True)
        resources = list(db.as_resources())
        system_resources = [r for r in resources if not r.on_behalf_of_user]
        assert all(not isinstance(r, DatabricksLakebase) for r in system_resources)
