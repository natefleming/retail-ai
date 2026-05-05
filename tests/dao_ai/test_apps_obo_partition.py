"""Tests for the OBO/non-OBO partition in Apps deploy resource generation.

The contract:

- A resource with ``on_behalf_of_user=True`` carries its OAuth scope into
  the app's ``user_api_scopes`` (so the user's forwarded token can call
  the underlying API).
- A resource with ``on_behalf_of_user=False`` is declared in the app's
  ``resources:`` block (so the platform grants the app SP the requested
  permission level).

A resource must not appear in *both* outputs: an OBO resource is never
hit by the app SP, so listing it as an app resource has the platform
prompt the operator to authorize a permission the app SP will never use.

These tests cover every extractor in ``src/dao_ai/apps/resources.py``
plus an integration test against the canonical ``sporting_goods_store``
config so the partition is exercised on a realistic graph.
"""

from __future__ import annotations

import pytest

from dao_ai.apps.resources import (
    _extract_connection_resources,
    _extract_function_resources,
    _extract_genie_resources,
    _extract_llm_resources,
    _extract_table_resources,
    _extract_vector_search_resources,
    _extract_volume_resources,
    _extract_warehouse_resources,
    generate_app_resources,
    generate_user_api_scopes,
)
from dao_ai.config import (
    AgentModel,
    AppConfig,
    AppModel,
    ConnectionModel,
    DeploymentTarget,
    FunctionModel,
    GenieRoomModel,
    IndexModel,
    LLMModel,
    ResourcesModel,
    SchemaModel,
    TableModel,
    VolumeModel,
    WarehouseModel,
    VectorStoreModel,
)


@pytest.mark.unit
class TestPerExtractorObOFilter:
    """Each ``_extract_*_resources`` helper must skip resources where
    ``on_behalf_of_user=True`` so OBO resources don't leak into the
    app's resources block."""

    def test_llm_extractor_skips_obo(self) -> None:
        llms = {
            "obo_llm": LLMModel(
                name="databricks-claude-sonnet-4-5",
                on_behalf_of_user=True,
            ),
            "system_llm": LLMModel(
                name="databricks-gte-large-en",
                on_behalf_of_user=False,
            ),
        }
        names = {r["name"] for r in _extract_llm_resources(llms)}
        assert names == {"system_llm"}

    def test_vector_search_extractor_skips_obo(self) -> None:
        schema = SchemaModel(catalog_name="cat", schema_name="sch")
        index = IndexModel(schema=schema, name="my_index")
        vector_stores = {
            "obo_vs": VectorStoreModel(index=index, on_behalf_of_user=True),
            "system_vs": VectorStoreModel(
                index=IndexModel(schema=schema, name="my_index_2"),
                on_behalf_of_user=False,
            ),
        }
        names = {r["name"] for r in _extract_vector_search_resources(vector_stores)}
        assert names == {"system_vs"}

    def test_warehouse_extractor_skips_obo(self) -> None:
        warehouses = {
            "obo_wh": WarehouseModel(
                name="obo", warehouse_id="abc", on_behalf_of_user=True
            ),
            "system_wh": WarehouseModel(
                name="sys", warehouse_id="def", on_behalf_of_user=False
            ),
        }
        names = {r["name"] for r in _extract_warehouse_resources(warehouses)}
        assert names == {"system_wh"}

    def test_genie_extractor_skips_obo(self) -> None:
        genie_rooms = {
            "obo_room": GenieRoomModel(
                name="obo", space_id="01f0obo", on_behalf_of_user=True
            ),
            "system_room": GenieRoomModel(
                name="sys", space_id="01f0sys", on_behalf_of_user=False
            ),
        }
        names = {r["name"] for r in _extract_genie_resources(genie_rooms)}
        assert names == {"system_room"}

    def test_function_extractor_skips_obo(self) -> None:
        schema = SchemaModel(catalog_name="cat", schema_name="sch")
        functions = {
            "obo_fn": FunctionModel(
                schema=schema, name="obo_fn", on_behalf_of_user=True
            ),
            "system_fn": FunctionModel(
                schema=schema, name="system_fn", on_behalf_of_user=False
            ),
        }
        names = {r["name"] for r in _extract_function_resources(functions)}
        assert names == {"system_fn"}

    def test_table_extractor_skips_obo(self) -> None:
        # _extract_table_resources already filters today; verify it stays
        # consistent with the rest of the family.
        schema = SchemaModel(catalog_name="cat", schema_name="sch")
        tables = {
            "obo_t": TableModel(schema=schema, name="obo_t", on_behalf_of_user=True),
            "system_t": TableModel(
                schema=schema, name="system_t", on_behalf_of_user=False
            ),
        }
        names = {r["name"] for r in _extract_table_resources(tables)}
        assert names == {"system_t"}

    def test_volume_extractor_skips_obo(self) -> None:
        schema = SchemaModel(catalog_name="cat", schema_name="sch")
        volumes = {
            "obo_v": VolumeModel(schema=schema, name="obo_v", on_behalf_of_user=True),
            "system_v": VolumeModel(
                schema=schema, name="system_v", on_behalf_of_user=False
            ),
        }
        names = {r["name"] for r in _extract_volume_resources(volumes)}
        assert names == {"system_v"}

    def test_connection_extractor_skips_obo(self) -> None:
        connections = {
            "obo_c": ConnectionModel(name="obo_c", on_behalf_of_user=True),
            "system_c": ConnectionModel(name="system_c", on_behalf_of_user=False),
        }
        names = {r["name"] for r in _extract_connection_resources(connections)}
        assert names == {"system_c"}


@pytest.mark.unit
class TestGenerateUserApiScopesPartition:
    """``generate_user_api_scopes`` already filters by OBO; lock that in."""

    def test_user_api_scopes_only_from_obo_resources(self) -> None:
        # OBO LLM contributes serving.serving-endpoints; non-OBO Genie
        # would otherwise also contribute dashboards.genie. Only the OBO
        # one should land in user_api_scopes.
        config = AppConfig(
            resources=ResourcesModel(
                llms={
                    "obo_llm": LLMModel(
                        name="x", on_behalf_of_user=True
                    ),
                    "system_llm": LLMModel(
                        name="y", on_behalf_of_user=False
                    ),
                },
                genie_rooms={
                    "system_room": GenieRoomModel(
                        name="r", space_id="abc", on_behalf_of_user=False
                    ),
                },
            ),
        )
        scopes = generate_user_api_scopes(config)
        assert "serving.serving-endpoints" in scopes
        # dashboards.genie is the user_api_scope mapping for the Genie room.
        # Since the room is non-OBO, it must NOT appear in user_api_scopes.
        assert "dashboards.genie" not in scopes


@pytest.mark.unit
class TestGenerateAppResourcesPartition:
    """End-to-end: ``generate_app_resources`` must not surface OBO resources."""

    def _config_with_obo_and_system_llms(self) -> AppConfig:
        obo = LLMModel(name="databricks-gpt-5-4-mini", on_behalf_of_user=True)
        system = LLMModel(name="databricks-gte-large-en", on_behalf_of_user=False)
        agent = AgentModel(name="a", model=obo)
        return AppConfig(
            resources=ResourcesModel(
                llms={"obo_llm": obo, "system_llm": system},
            ),
            agents={"a": agent},
            app=AppModel(
                name="obo-test",
                deployment_target=DeploymentTarget.APPS,
                agents=[agent],
            ),
        )

    def test_app_resources_excludes_obo_llms(self) -> None:
        config = self._config_with_obo_and_system_llms()
        resources = generate_app_resources(config)
        names = {r["name"] for r in resources}
        assert "system_llm" in names
        assert "obo_llm" not in names, (
            "OBO LLMs must not appear in app.yaml resources block — they're "
            "served via user_api_scopes, the app SP never calls them"
        )

    def test_app_resources_obo_llms_appear_in_user_api_scopes(self) -> None:
        config = self._config_with_obo_and_system_llms()
        scopes = generate_user_api_scopes(config)
        # OBO LLM contributes serving.serving-endpoints; non-OBO LLM does not.
        assert scopes == ["serving.serving-endpoints"]


@pytest.mark.unit
class TestSdkExtractorObOFilter:
    """The SDK-format extractors are used by ``AppConfig.deploy_agent(target=APPS)``
    via ``apps.create/update`` (not the bundle path). They must apply the same
    OBO filter as the flat-dict family so both deploy paths agree."""

    def test_sdk_llm_extractor_skips_obo(self) -> None:
        from dao_ai.apps.resources import _extract_sdk_llm_resources

        llms = {
            "obo_llm": LLMModel(
                name="databricks-claude-sonnet-4-5", on_behalf_of_user=True
            ),
            "system_llm": LLMModel(
                name="databricks-gte-large-en", on_behalf_of_user=False
            ),
        }
        resources = _extract_sdk_llm_resources(llms)
        names = {r.name for r in resources}
        assert "system_llm" in names
        assert "obo_llm" not in names

    def test_sdk_warehouse_extractor_skips_obo(self) -> None:
        from dao_ai.apps.resources import _extract_sdk_warehouse_resources

        warehouses = {
            "obo_wh": WarehouseModel(
                name="obo", warehouse_id="abc", on_behalf_of_user=True
            ),
            "system_wh": WarehouseModel(
                name="sys", warehouse_id="def", on_behalf_of_user=False
            ),
        }
        names = {r.name for r in _extract_sdk_warehouse_resources(warehouses)}
        assert names == {"system_wh"}

    def test_sdk_genie_extractor_skips_obo(self) -> None:
        from dao_ai.apps.resources import _extract_sdk_genie_resources

        genie_rooms = {
            "obo_room": GenieRoomModel(
                name="obo", space_id="01f0obo", on_behalf_of_user=True
            ),
            "system_room": GenieRoomModel(
                name="sys", space_id="01f0sys", on_behalf_of_user=False
            ),
        }
        names = {r.name for r in _extract_sdk_genie_resources(genie_rooms)}
        assert names == {"system_room"}

    def test_sdk_volume_extractor_skips_obo(self) -> None:
        from dao_ai.apps.resources import _extract_sdk_volume_resources

        schema = SchemaModel(catalog_name="cat", schema_name="sch")
        volumes = {
            "obo_v": VolumeModel(
                schema=schema, name="obo_v", on_behalf_of_user=True
            ),
            "system_v": VolumeModel(
                schema=schema, name="system_v", on_behalf_of_user=False
            ),
        }
        names = {r.name for r in _extract_sdk_volume_resources(volumes)}
        assert names == {"system_v"}


@pytest.mark.integration
class TestSportingGoodsSdkResourcesPartition:
    """Integration: ``generate_sdk_resources`` on canonical sporting_goods config.

    The SDK-format output is what ``AppConfig.deploy_agent(target=APPS)`` sends
    to ``apps.create/update`` (not the bundle path). Verifies OBO LLMs don't
    leak into SDK resources either.
    """

    CONFIG_PATH = (
        "config/examples/15_complete_applications/sporting_goods_store.yaml"
    )
    OBO_RESOURCE_NAMES: frozenset[str] = frozenset(
        {"fast_llm", "supervisor_llm", "tool_calling_llm", "decomposition_llm"}
    )

    @pytest.fixture(scope="class")
    def config(self) -> AppConfig:
        return AppConfig.from_file(self.CONFIG_PATH, initialize=False)

    def test_sdk_resources_excludes_obo_llms(self, config: AppConfig) -> None:
        from dao_ai.apps.resources import generate_sdk_resources

        sdk_resources = generate_sdk_resources(config)
        # SDK AppResource.name is sanitized; compare against sanitized OBO names
        from dao_ai.apps.resources import _sanitize_resource_name

        names = {r.name for r in sdk_resources}
        sanitized_obo = {_sanitize_resource_name(n) for n in self.OBO_RESOURCE_NAMES}
        leaked = sanitized_obo & names
        assert not leaked, (
            f"OBO LLMs leaked into SDK app resources: {sorted(leaked)}. "
            "These should only contribute to user_api_scopes, not be granted "
            "to the app SP."
        )

    def test_sdk_resources_includes_system_llms(self, config: AppConfig) -> None:
        from dao_ai.apps.resources import generate_sdk_resources, _sanitize_resource_name

        sdk_resources = generate_sdk_resources(config)
        names = {r.name for r in sdk_resources}
        # Non-OBO LLMs should be present
        for name in ("judge_llm", "embedding_model"):
            assert _sanitize_resource_name(name) in names


@pytest.mark.integration
class TestSportingGoodsConfigPartition:
    """End-to-end on the canonical sporting_goods_store.yaml.

    The config has 4 OBO LLMs (fast/supervisor/tool_calling/decomposition)
    and 13 non-OBO resources. Verifies the partition holds against the
    real config we actually deploy.
    """

    CONFIG_PATH = (
        "config/examples/15_complete_applications/sporting_goods_store.yaml"
    )

    OBO_RESOURCE_NAMES: frozenset[str] = frozenset(
        {"fast_llm", "supervisor_llm", "tool_calling_llm", "decomposition_llm"}
    )

    @pytest.fixture(scope="class")
    def config(self) -> AppConfig:
        return AppConfig.from_file(self.CONFIG_PATH, initialize=False)

    def test_obo_llms_excluded_from_app_resources(self, config: AppConfig) -> None:
        resources = generate_app_resources(config)
        names_in_resources = {r["name"] for r in resources}
        leaked = self.OBO_RESOURCE_NAMES & names_in_resources
        assert not leaked, (
            f"OBO LLMs leaked into app.yaml resources block: {sorted(leaked)}. "
            "These are served via user_api_scopes; listing them as app "
            "resources is wrong because the app SP never calls them."
        )

    def test_user_api_scopes_covers_obo_resources(self, config: AppConfig) -> None:
        scopes = generate_user_api_scopes(config)
        # OBO LLMs need serving.serving-endpoints
        assert "serving.serving-endpoints" in scopes

    def test_non_obo_resources_present_in_app_resources(
        self, config: AppConfig
    ) -> None:
        resources = generate_app_resources(config)
        names_in_resources = {r["name"] for r in resources}
        # Pick representative non-OBO resources from the canonical config
        for name in (
            "judge_llm",
            "embedding_model",
            "products_vector_store",
            "shared_warehouse",
            "merchandising_analytics_room",
            "sales_pricing_room",
            "find_product_by_sku",
        ):
            assert name in names_in_resources, (
                f"non-OBO resource '{name}' missing from app.yaml resources"
            )
