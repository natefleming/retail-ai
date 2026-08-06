"""Tests for adapting OBO user API scopes to Model Serving's allowlist.

The Apps platform and Model Serving accept *different* OBO scope sets. Apps
grants UC reads per securable (``catalog.catalogs:read`` / ``.schemas:read`` /
``.tables:read``); Model Serving takes the single coarse ``unity-catalog``.

Passing the Apps set to Model Serving fails the deploy outright:

    InvalidParameterValue: Invalid user API scope(s) specified for model: ...
    Invalid scopes: catalog.catalogs:read, catalog.schemas:read,
    catalog.tables:read

Observed on FEVM deploying hardware_store_lakebase (`--mode ms`), whose UC
functions are all ``on_behalf_of_user: true``.
"""

from __future__ import annotations

import pytest

from dao_ai.apps.resources import (
    API_SCOPE_TO_USER_SCOPES,
    MODEL_SERVING_USER_API_SCOPES,
    VALID_USER_API_SCOPES,
    adapt_user_api_scopes_for_model_serving,
    generate_user_api_scopes,
)
from dao_ai.config import (
    AgentModel,
    AppConfig,
    AppModel,
    FunctionModel,
    InferenceEndpointModel,
    ResourcesModel,
    SchemaModel,
    TableModel,
)


def _schema() -> SchemaModel:
    return SchemaModel(catalog_name="cat", schema_name="sch")


def _app() -> AppModel:
    return AppModel(
        name="myapp",
        agents=[
            AgentModel(
                name="a",
                description="d",
                model=InferenceEndpointModel(name="databricks-gpt-5-4-mini"),
            )
        ],
    )


@pytest.mark.unit
class TestModelServingScopeAdaptation:
    def test_catalog_read_scopes_become_unity_catalog(self) -> None:
        """The exact scopes Model Serving rejected, and its accepted equivalent."""
        adapted = adapt_user_api_scopes_for_model_serving(
            ["catalog.catalogs:read", "catalog.schemas:read", "catalog.tables:read"]
        )
        assert adapted == ["unity-catalog"]

    def test_scopes_valid_on_both_planes_pass_through(self) -> None:
        scopes = ["sql", "vector-search", "serving.serving-endpoints", "mcp.functions"]
        assert adapt_user_api_scopes_for_model_serving(scopes) == sorted(scopes)

    def test_unknown_scopes_are_dropped_not_forwarded(self) -> None:
        """Anything Model Serving would reject must not reach the deploy call."""
        adapted = adapt_user_api_scopes_for_model_serving(
            ["sql", "definitely-not-a-real-scope"]
        )
        assert adapted == ["sql"]

    def test_output_is_always_within_the_model_serving_allowlist(self) -> None:
        """Every scope dao-ai can emit for Apps must adapt to something valid."""
        every_apps_scope = sorted(VALID_USER_API_SCOPES)
        adapted = adapt_user_api_scopes_for_model_serving(every_apps_scope)
        assert adapted, "adaptation dropped everything"
        assert not set(adapted) - MODEL_SERVING_USER_API_SCOPES

    def test_result_is_sorted_and_deduped(self) -> None:
        adapted = adapt_user_api_scopes_for_model_serving(
            ["catalog.tables:read", "sql", "catalog.schemas:read", "sql"]
        )
        assert adapted == ["sql", "unity-catalog"]

    def test_empty_input_yields_empty_output(self) -> None:
        assert adapt_user_api_scopes_for_model_serving([]) == []


@pytest.mark.unit
class TestCatalogScopesComeFromTheResourceMap:
    def test_sql_statement_execution_carries_the_catalog_read_scopes(self) -> None:
        """They belong to the declared api_scope, not to an isinstance check.

        TableModel and FunctionModel both declare ``sql.statement-execution``, so
        routing the UC read scopes through this map is what lets them be
        translated per deployment target.
        """
        mapped = API_SCOPE_TO_USER_SCOPES["sql.statement-execution"]
        assert {
            "catalog.catalogs:read",
            "catalog.schemas:read",
            "catalog.tables:read",
        } <= mapped

    def test_obo_function_still_emits_catalog_scopes_for_apps(self) -> None:
        """Backward compatibility: the Apps path is unchanged."""
        schema = _schema()
        config = AppConfig(
            schemas={"s": schema},
            resources=ResourcesModel(
                functions={
                    "f": FunctionModel(
                        schema=schema, name="find_x", on_behalf_of_user=True
                    )
                }
            ),
            app=_app(),
        )
        scopes = set(generate_user_api_scopes(config))
        assert "catalog.catalogs:read" in scopes
        assert "catalog.tables:read" in scopes
        assert "sql" in scopes

    def test_obo_table_still_emits_catalog_scopes_for_apps(self) -> None:
        schema = _schema()
        config = AppConfig(
            schemas={"s": schema},
            resources=ResourcesModel(
                tables={
                    "t": TableModel(
                        schema=schema, name="products", on_behalf_of_user=True
                    )
                }
            ),
            app=_app(),
        )
        assert "catalog.tables:read" in set(generate_user_api_scopes(config))

    def test_config_without_obo_emits_no_catalog_scopes(self) -> None:
        """A non-OBO table needs no user scopes at all — it is SP-backed."""
        schema = _schema()
        config = AppConfig(
            schemas={"s": schema},
            resources=ResourcesModel(
                tables={
                    "t": TableModel(
                        schema=schema, name="products", on_behalf_of_user=False
                    )
                }
            ),
            app=_app(),
        )
        scopes = set(generate_user_api_scopes(config))
        assert not {s for s in scopes if s.startswith("catalog.")}

    def test_obo_function_adapts_to_unity_catalog_for_model_serving(self) -> None:
        """End to end: the failing config's shape now deploys to Model Serving."""
        schema = _schema()
        config = AppConfig(
            schemas={"s": schema},
            resources=ResourcesModel(
                functions={
                    "f": FunctionModel(
                        schema=schema, name="find_x", on_behalf_of_user=True
                    )
                }
            ),
            app=_app(),
        )
        ms_scopes = adapt_user_api_scopes_for_model_serving(
            generate_user_api_scopes(config)
        )
        assert "unity-catalog" in ms_scopes
        assert not [s for s in ms_scopes if s.startswith("catalog.")]
        assert not set(ms_scopes) - MODEL_SERVING_USER_API_SCOPES
