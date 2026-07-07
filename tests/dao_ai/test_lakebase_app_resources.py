"""
Tests for Lakebase database resource extraction across both deployment paths.

The two deploy paths emit Lakebase resources very differently:

- **Databricks Apps** (``deploy_apps_agent``): the deployed app declares a
  ``postgres`` app resource via ``generate_app_resources()`` (flat-dict for
  the bundle) and ``generate_sdk_resources()`` (SDK ``AppResource``). The
  platform binds the auto-SP to the Lakebase project with
  ``CAN_CONNECT_AND_CREATE``. The Apps platform's ``postgres`` resource type
  fully supports autoscaling Lakebase projects and requires both ``branch``
  and ``database`` as full resource paths
  (``projects/<p>/branches/<b>[/databases/<id>]``).

- **Model Serving** (``deploy_model_serving_agent``): the model is logged
  with a ``SystemAuthPolicy`` whose ``resources`` list comes from each
  resource model's ``as_resources()`` method. **MLflow's
  ``DatabricksLakebase`` resource does not support autoscaling Lakebase
  projects** -- only the deprecated provisioned-instance shape -- and the
  MLflow team confirmed (2026-04-10) that autoscaling support isn't planned
  for the time being. Emitting the resource for an autoscaling project
  causes the endpoint to fail to start with
  ``NOT_FOUND: Database instance is not found``. dao-ai therefore returns
  ``[]`` from ``DatabaseModel.as_resources()`` for Lakebase. Users who need
  Model Serving + Lakebase must manage auth in agent code via OAuth M2M
  (``client_id`` / ``client_secret`` on the DatabaseModel).

  Reference: https://github.com/mlflow/mlflow/issues/22452

OBO databases are skipped at extraction time on the Apps path -- the user
identity handles permissions via ``user_api_scopes`` instead.

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
    """``DatabaseModel.as_resources()`` feeds Model Serving's SystemAuthPolicy.

    MLflow's ``DatabricksLakebase`` resource only supports the deprecated
    provisioned-instance shape; logging it for an autoscaling Lakebase project
    breaks Model Serving deploys (NOT_FOUND on the database resource at endpoint
    start). Per the MLflow team, autoscaling support isn't planned for the time
    being -- see https://github.com/mlflow/mlflow/issues/22452 (2026-04-10).

    dao-ai therefore returns ``[]`` for autoscaling Lakebase databases. Users
    who need Model Serving + Lakebase must manage auth in agent code via OAuth
    M2M (``client_id`` / ``client_secret`` on the DatabaseModel)."""

    def test_lakebase_does_not_emit_databrickslakebase_resource(self) -> None:
        """Autoscaling Lakebase projects intentionally emit no MLflow
        Lakebase resource -- the existing ``DatabricksLakebase`` MLflow class
        only supports provisioned instances, not autoscaling projects."""
        from dao_ai.config import DatabaseModel

        db = DatabaseModel(
            name="retail-consumer-goods", project="retail-consumer-goods"
        )
        assert list(db.as_resources()) == []

    def test_lakebase_obo_also_returns_empty(self) -> None:
        """OBO databases also return [] -- the on_behalf_of_user flag does not
        change the autoscaling-incompatibility issue with MLflow."""
        from dao_ai.config import DatabaseModel

        db = DatabaseModel(name="x", project="x", on_behalf_of_user=True)
        assert list(db.as_resources()) == []

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


# =============================================================================
# _extract_database_resources -- flat-dict format consumed by app.yaml bundle gen
# =============================================================================


@pytest.mark.unit
class TestExtractDatabaseResourcesFlat:
    """Flat-dict extractor used when generating ``app.yaml`` for the bundle."""

    def test_lakebase_non_obo_emits_postgres_resource(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Autoscaling Lakebase projects use the ``postgres`` resource type
        (not the deprecated ``database``/instance shape). When the user
        doesn't pin a branch, the extractor resolves the project's default
        branch and emits the full resource path -- the Apps platform
        validates the branch by calling ``get_branch(name=...)`` server-side
        and rejects bare branch IDs. Same applies to the ``database`` field,
        which must be the full Database resource path."""
        from dao_ai.apps import resources as resources_mod
        from dao_ai.apps.resources import _extract_database_resources
        from dao_ai.config import DatabaseModel

        monkeypatch.setattr(
            DatabaseModel, "resolve_default_branch", lambda self: "main"
        )
        monkeypatch.setattr(
            resources_mod,
            "_resolve_lakebase_database_path",
            lambda db, branch_path: f"{branch_path}/databases/db-test",
        )

        db = DatabaseModel(
            name="retail-consumer-goods", project="retail-consumer-goods"
        )
        out = _extract_database_resources({"workshop_db": db})
        assert len(out) == 1
        r = out[0]
        assert r["type"] == "postgres"
        assert r["name"] == "workshop_db"
        # Both `branch` and `database` are full resource paths -- the Apps
        # platform rejects bare IDs with INVALID_PARAMETER_VALUE.
        assert r["branch"] == "projects/retail-consumer-goods/branches/main"
        assert (
            r["database"]
            == "projects/retail-consumer-goods/branches/main/databases/db-test"
        )
        assert r["permissions"] == [{"level": "CAN_CONNECT_AND_CREATE"}]

    def test_lakebase_with_branch_propagates_to_resource(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A user-pinned branch (a bare branch ID) is wrapped into the full
        resource path; the branch resolver is not consulted."""
        from dao_ai.apps import resources as resources_mod
        from dao_ai.apps.resources import _extract_database_resources
        from dao_ai.config import DatabaseModel

        def _boom(self: object) -> str:
            raise AssertionError("resolver should not be called when branch is pinned")

        monkeypatch.setattr(DatabaseModel, "resolve_default_branch", _boom)
        monkeypatch.setattr(
            resources_mod,
            "_resolve_lakebase_database_path",
            lambda db, branch_path: f"{branch_path}/databases/db-test",
        )

        db = DatabaseModel(project="retail-consumer-goods", branch="dev")
        out = _extract_database_resources({"workshop_db": db})
        assert out[0]["branch"] == "projects/retail-consumer-goods/branches/dev"
        assert (
            out[0]["database"]
            == "projects/retail-consumer-goods/branches/dev/databases/db-test"
        )

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

    def test_project_only_emits_postgres_with_project_in_database_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The ``database`` field on the AppResource is the full Database
        resource path -- which embeds the project at the root."""
        from dao_ai.apps import resources as resources_mod
        from dao_ai.apps.resources import _extract_database_resources
        from dao_ai.config import DatabaseModel

        monkeypatch.setattr(
            DatabaseModel, "resolve_default_branch", lambda self: "main"
        )
        monkeypatch.setattr(
            resources_mod,
            "_resolve_lakebase_database_path",
            lambda db, branch_path: f"{branch_path}/databases/db-test",
        )

        db = DatabaseModel(project="project-only")
        out = _extract_database_resources({"k": db})
        assert out[0]["database"].startswith("projects/project-only/")

    def test_resource_name_is_sanitized(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Underscores and other punctuation in the YAML key get sanitized
        for Databricks resource-name rules (matches volumes/tables behavior)."""
        from dao_ai.apps import resources as resources_mod
        from dao_ai.apps.resources import _extract_database_resources
        from dao_ai.config import DatabaseModel

        monkeypatch.setattr(
            DatabaseModel, "resolve_default_branch", lambda self: "main"
        )
        monkeypatch.setattr(
            resources_mod,
            "_resolve_lakebase_database_path",
            lambda db, branch_path: f"{branch_path}/databases/db-test",
        )

        db = DatabaseModel(name="x", project="x")
        out = _extract_database_resources({"workshop_DB.shared": db})
        # Whatever sanitization rule is applied, the result should be a single
        # entry with a non-empty 'name' that's safe to use as a resource id.
        assert len(out) == 1
        assert out[0]["name"]
        assert "/" not in out[0]["name"]

    def test_multiple_lakebases_each_get_a_resource(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dao_ai.apps import resources as resources_mod
        from dao_ai.apps.resources import _extract_database_resources
        from dao_ai.config import DatabaseModel

        monkeypatch.setattr(
            DatabaseModel, "resolve_default_branch", lambda self: "main"
        )
        monkeypatch.setattr(
            resources_mod,
            "_resolve_lakebase_database_path",
            lambda db, branch_path: f"{branch_path}/databases/db-test",
        )

        out = _extract_database_resources(
            {
                "memory_db": DatabaseModel(project="p1"),
                "checkpoints_db": DatabaseModel(project="p2"),
            }
        )
        # Each Lakebase becomes its own postgres resource with a project-rooted path.
        roots = {r["database"].split("/")[1] for r in out}
        assert roots == {"p1", "p2"}

    def test_empty_dict_returns_empty(self) -> None:
        from dao_ai.apps.resources import _extract_database_resources

        assert _extract_database_resources({}) == []


# =============================================================================
# _extract_sdk_database_resources -- SDK AppResource format for direct deploy
# =============================================================================


@pytest.mark.unit
class TestExtractDatabaseResourcesSDK:
    """SDK ``AppResource`` extractor used by the direct-deploy path."""

    def test_lakebase_non_obo_emits_appresource_postgres(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from databricks.sdk.service.apps import (
            AppResource,
            AppResourcePostgres,
            AppResourcePostgresPostgresPermission,
        )

        from dao_ai.apps import resources as resources_mod
        from dao_ai.apps.resources import _extract_sdk_database_resources
        from dao_ai.config import DatabaseModel

        monkeypatch.setattr(
            DatabaseModel, "resolve_default_branch", lambda self: "main"
        )
        monkeypatch.setattr(
            resources_mod,
            "_resolve_lakebase_database_path",
            lambda db, branch_path: f"{branch_path}/databases/db-test",
        )

        db = DatabaseModel(
            name="retail-consumer-goods", project="retail-consumer-goods"
        )
        out = _extract_sdk_database_resources({"workshop_db": db})
        assert len(out) == 1
        r = out[0]
        assert isinstance(r, AppResource)
        # Autoscaling Lakebase -> AppResourcePostgres (the platform shape
        # for projects); AppResourceDatabase is the deprecated provisioned
        # instance shape and must NOT be used.
        assert r.database is None
        assert isinstance(r.postgres, AppResourcePostgres)
        # Both branch and database are full resource paths -- the platform
        # validates each with get_branch / get_database and rejects bare IDs.
        assert r.postgres.branch == "projects/retail-consumer-goods/branches/main"
        assert (
            r.postgres.database
            == "projects/retail-consumer-goods/branches/main/databases/db-test"
        )
        assert (
            r.postgres.permission
            is AppResourcePostgresPostgresPermission.CAN_CONNECT_AND_CREATE
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

    def test_flat_app_resources_contain_postgres_alongside_llm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dao_ai.apps import resources as resources_mod
        from dao_ai.apps.resources import generate_app_resources
        from dao_ai.config import DatabaseModel

        monkeypatch.setattr(
            DatabaseModel, "resolve_default_branch", lambda self: "main"
        )
        monkeypatch.setattr(
            resources_mod,
            "_resolve_lakebase_database_path",
            lambda db, branch_path: f"{branch_path}/databases/db-test",
        )

        resources = generate_app_resources(self._build_config())
        types = {r["type"] for r in resources}
        assert "serving-endpoint" in types
        assert "postgres" in types
        pg_r = next(r for r in resources if r["type"] == "postgres")
        assert pg_r["database"].startswith("projects/workshop-db/branches/")
        assert pg_r["database"].endswith("/databases/db-test")
        assert pg_r["permissions"] == [{"level": "CAN_CONNECT_AND_CREATE"}]

    def test_flat_app_resources_skips_postgres_when_obo(self) -> None:
        from dao_ai.apps.resources import generate_app_resources

        resources = generate_app_resources(self._build_config(on_behalf_of_user=True))
        types = {r["type"] for r in resources}
        assert "postgres" not in types

    def test_sdk_app_resources_contain_appresourcepostgres(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from databricks.sdk.service.apps import AppResource, AppResourcePostgres

        from dao_ai.apps import resources as resources_mod
        from dao_ai.apps.resources import generate_sdk_resources
        from dao_ai.config import DatabaseModel

        monkeypatch.setattr(
            DatabaseModel, "resolve_default_branch", lambda self: "main"
        )
        monkeypatch.setattr(
            resources_mod,
            "_resolve_lakebase_database_path",
            lambda db, branch_path: f"{branch_path}/databases/db-test",
        )

        resources = generate_sdk_resources(self._build_config())
        pg_resources = [
            r
            for r in resources
            if isinstance(r, AppResource) and r.postgres is not None
        ]
        assert len(pg_resources) == 1
        assert isinstance(pg_resources[0].postgres, AppResourcePostgres)
        assert pg_resources[0].postgres.database.startswith(
            "projects/workshop-db/branches/"
        )
        assert pg_resources[0].postgres.database.endswith("/databases/db-test")
        # No legacy AppResourceDatabase mixed in
        assert pg_resources[0].database is None

    def test_sdk_app_resources_skips_postgres_when_obo(self) -> None:
        from databricks.sdk.service.apps import AppResource

        from dao_ai.apps.resources import generate_sdk_resources

        resources = generate_sdk_resources(self._build_config(on_behalf_of_user=True))
        pg_resources = [
            r
            for r in resources
            if isinstance(r, AppResource) and r.postgres is not None
        ]
        assert pg_resources == []


# =============================================================================
# Integration: model-serving system_resources path picks up the Lakebase
# =============================================================================


@pytest.mark.integration
class TestModelServingSystemResources:
    """The ``deploy_model_serving_agent`` path collects ``as_resources()``
    from each resource model into a SystemAuthPolicy.

    For autoscaling Lakebase, dao-ai intentionally returns ``[]`` (see
    ``DatabaseModel.as_resources``) because MLflow's ``DatabricksLakebase``
    resource doesn't support autoscaling projects -- emitting it would
    cause the Model Serving endpoint to fail to start with
    ``NOT_FOUND: Database instance is not found``.

    Reference: https://github.com/mlflow/mlflow/issues/22452 (2026-04-10)."""

    def test_lakebase_does_not_appear_in_as_resources_aggregate(self) -> None:
        """Reproduce the aggregation step of deploy_model_serving_agent;
        the Lakebase database should NOT contribute resources."""
        from mlflow.models.resources import DatabricksLakebase

        from dao_ai.config import DatabaseModel, LLMModel

        llm = LLMModel(name="databricks-claude-sonnet-4-5")
        db = DatabaseModel(name="x", project="x")

        all_resources = []
        for r in [llm, db]:
            all_resources.extend(r.as_resources())

        # The LLM contributes a serving-endpoint resource; the Lakebase
        # contributes nothing (no DatabricksLakebase resource is emitted).
        lakebase_resources = [
            r for r in all_resources if isinstance(r, DatabricksLakebase)
        ]
        assert lakebase_resources == []

    def test_obo_lakebase_also_excluded(self) -> None:
        from mlflow.models.resources import DatabricksLakebase

        from dao_ai.config import DatabaseModel

        db = DatabaseModel(name="x", project="x", on_behalf_of_user=True)
        resources = list(db.as_resources())
        assert all(not isinstance(r, DatabricksLakebase) for r in resources)


# =============================================================================
# _resolve_lakebase_database_path — SDK auto-detect with graceful fallback.
#
# Precedence:
#   1. Full-path in db.database → verbatim (escape hatch).
#   2. Explicit db.database_id → verbatim (skip SDK, user knows best).
#   3. SDK auto-detect → match by pg_name → return d.name.
#   4. SDK success without match → first database's d.name (single-DB projects).
#   5. SDK failure → construct with db.database_id default + WARNING log.
# =============================================================================


def _mock_sdk_client(
    monkeypatch: pytest.MonkeyPatch,
    databases: list | Exception,
) -> object:
    """Install a fake workspace_client on DatabaseModel whose
    ``postgres.list_databases`` returns ``databases`` (a list) or raises
    (if given an Exception instance). Returns the fake client so callers
    can inspect ``call_count`` etc."""
    from unittest.mock import MagicMock

    from dao_ai.config import DatabaseModel

    class _Postgres:
        call_count = 0
        last_branch: str | None = None

        def list_databases(self_inner, branch_path: str):
            _Postgres.call_count += 1
            _Postgres.last_branch = branch_path
            if isinstance(databases, Exception):
                raise databases
            return iter(databases)

    class _WC:
        postgres = _Postgres()

    fake = MagicMock(spec=_WC)
    fake.postgres = _WC.postgres
    monkeypatch.setattr(
        DatabaseModel, "workspace_client", property(lambda self: fake)
    )
    return _WC.postgres


def _forbid_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if workspace_client is touched."""
    from dao_ai.config import DatabaseModel

    def _boom(_self: object) -> object:
        raise AssertionError(
            "_resolve_lakebase_database_path must not access workspace_client "
            "in this precedence branch"
        )

    monkeypatch.setattr(DatabaseModel, "workspace_client", property(_boom))


@pytest.mark.unit
class TestResolveLakebaseDatabasePathPrecedence:
    """Cover each of the 5 precedence levels documented on the resolver."""

    def test_1_full_path_in_database_field_passthrough(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """User wrote a full ``projects/.../databases/<id>`` string in
        ``database:``. Returned verbatim, no SDK call."""
        from dao_ai.apps.resources import _resolve_lakebase_database_path
        from dao_ai.config import DatabaseModel

        custom_path = (
            "projects/retail-consumer-goods/branches/production/"
            "databases/db-vllm-t1lbxazynr"
        )
        db = DatabaseModel(
            project="retail-consumer-goods",
            branch="production",
            database=custom_path,
        )
        _forbid_sdk(monkeypatch)

        result = _resolve_lakebase_database_path(
            db, "projects/retail-consumer-goods/branches/production"
        )
        assert result == custom_path

    def test_2_explicit_database_id_override_skips_sdk(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """User set ``database_id`` explicitly in YAML → construct with
        that value, no SDK call. Recommended shape for custom-provisioned
        Lakebases with a non-default resource id."""
        from dao_ai.apps.resources import _resolve_lakebase_database_path
        from dao_ai.config import DatabaseModel

        db = DatabaseModel(
            project="retail-consumer-goods",
            branch="production",
            database_id="db-vllm-t1lbxazynr",
        )
        _forbid_sdk(monkeypatch)

        result = _resolve_lakebase_database_path(
            db, "projects/retail-consumer-goods/branches/production"
        )
        assert result == (
            "projects/retail-consumer-goods/branches/production/databases/db-vllm-t1lbxazynr"
        )

    def test_3_sdk_autodetect_matches_by_pg_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No database_id set → SDK auto-detects. Returns d.name of the
        database whose status.postgres_database matches db.database. This
        is the transparent auto-detection path for custom-provisioned
        setups (matches v0.1.101 behavior)."""
        from types import SimpleNamespace

        from dao_ai.apps.resources import _resolve_lakebase_database_path
        from dao_ai.config import DatabaseModel

        db = DatabaseModel(project="commerce-swarm", branch="production")
        pg = _mock_sdk_client(
            monkeypatch,
            [
                SimpleNamespace(
                    name="projects/commerce-swarm/branches/production/databases/databricks-postgres",
                    status=SimpleNamespace(postgres_database="databricks_postgres"),
                ),
            ],
        )

        result = _resolve_lakebase_database_path(
            db, "projects/commerce-swarm/branches/production"
        )
        assert result == (
            "projects/commerce-swarm/branches/production/databases/databricks-postgres"
        )
        assert pg.call_count == 1

    def test_4_sdk_returns_databases_no_match_falls_back_to_first(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SDK returned databases but none match db.database by pg_name
        → return the first database's name (typical single-DB project
        case; matches v0.1.101's ``databases[0].name`` fallback)."""
        from types import SimpleNamespace

        from dao_ai.apps.resources import _resolve_lakebase_database_path
        from dao_ai.config import DatabaseModel

        db = DatabaseModel(project="p", branch="main", database="something_else")
        _mock_sdk_client(
            monkeypatch,
            [
                SimpleNamespace(
                    name="projects/p/branches/main/databases/actual-db",
                    status=SimpleNamespace(postgres_database="not_the_pg_name"),
                ),
            ],
        )

        result = _resolve_lakebase_database_path(
            db, "projects/p/branches/main"
        )
        assert result == "projects/p/branches/main/databases/actual-db"

    def test_5_sdk_failure_falls_back_to_database_id_default(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """SDK raises (auth broken / wrong profile) → resolver logs a
        WARNING and falls back to ``{branch}/databases/{database_id}``
        with the ``databricks-postgres`` default. Fixes v0.1.101's
        silent-fallback-to-wrong-path bug (which used ``db.database``,
        the pg-level name)."""
        from dao_ai.apps.resources import _resolve_lakebase_database_path
        from dao_ai.config import DatabaseModel

        db = DatabaseModel(project="commerce-swarm", branch="production")
        _mock_sdk_client(
            monkeypatch, RuntimeError("401 Unauthorized")
        )

        # loguru: capture WARNING via a dedicated sink.
        from loguru import logger as loguru_logger

        captured: list[str] = []
        sink_id = loguru_logger.add(
            lambda msg: captured.append(str(msg)),
            level="WARNING",
            format="{message}",
        )
        try:
            result = _resolve_lakebase_database_path(
                db, "projects/commerce-swarm/branches/production"
            )
        finally:
            loguru_logger.remove(sink_id)

        # Falls back to the databricks-postgres default (hyphenated),
        # NOT db.database (which is databricks_postgres, underscored).
        assert result == (
            "projects/commerce-swarm/branches/production/databases/databricks-postgres"
        )
        # A WARNING was logged mentioning the fallback + how to fix.
        joined = " ".join(captured)
        assert "auto-detection failed" in joined
        assert "database_id" in joined

    def test_5b_sdk_returns_empty_falls_back_to_database_id_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SDK returned an empty database list → still fall back to the
        database_id default. Same as the failure case, no WARNING (we
        got a successful response, just with no rows)."""
        from dao_ai.apps.resources import _resolve_lakebase_database_path
        from dao_ai.config import DatabaseModel

        db = DatabaseModel(project="p", branch="main")
        _mock_sdk_client(monkeypatch, [])

        result = _resolve_lakebase_database_path(
            db, "projects/p/branches/main"
        )
        assert result == "projects/p/branches/main/databases/databricks-postgres"

    def test_2_wins_over_3_explicit_database_id_beats_sdk(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Precedence guard: even when SDK would return something valid,
        an explicit database_id override wins. This lets users bypass
        SDK entirely (fast, deterministic, no profile requirement)."""
        from dao_ai.apps.resources import _resolve_lakebase_database_path
        from dao_ai.config import DatabaseModel

        db = DatabaseModel(
            project="p",
            branch="main",
            database_id="user-override-id",
        )
        _forbid_sdk(monkeypatch)  # Would blow up if the resolver called SDK.

        result = _resolve_lakebase_database_path(
            db, "projects/p/branches/main"
        )
        assert result == "projects/p/branches/main/databases/user-override-id"
