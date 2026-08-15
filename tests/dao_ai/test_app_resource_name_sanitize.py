"""Regression tests (F5) for Databricks Apps resource-name sanitization.

Dict keys for genie-derived tables/functions can exceed the Apps 2–30 char
resource-name rule (e.g. ``tbl_3978880469080159_trace_unified`` = 34 chars, or
``<room>_<catalog>_<schema>_<table>`` from the primary genie validator). Those
keys are emitted verbatim as the resource ``name`` and made ``databricks bundle
deploy`` fail with 400 INVALID_PARAMETER_VALUE. Both emit points must sanitize +
uniquify the name.
"""

from __future__ import annotations

import pytest

from dao_ai.apps.resources import (
    _extract_function_resources,
    _extract_table_resources,
    _extract_warehouse_resources,
    _unique_resource_name,
    generate_app_resources,
)
from dao_ai.config import (
    AgentModel,
    AppConfig,
    AppModel,
    ConnectionModel,
    FunctionModel,
    GenieRoomModel,
    LLMModel,
    ResourcesModel,
    SecretVariableModel,
    ServicePrincipalModel,
    TableModel,
    VolumeModel,
    WarehouseModel,
)


@pytest.mark.unit
class TestUniqueResourceName:
    def test_long_name_truncated_to_30(self) -> None:
        # The exact matrix failure key (34 chars).
        name = _unique_resource_name("tbl_3978880469080159_trace_unified", set())
        assert 2 <= len(name) <= 30

    def test_short_name_unchanged(self) -> None:
        used: set[str] = set()
        assert _unique_resource_name("tbl_products", used) == "tbl_products"

    def test_collision_gets_distinct_suffix_within_budget(self) -> None:
        used: set[str] = set()
        base = "tbl_" + ("segment_" * 6)  # > 30 chars, shared prefix
        a = _unique_resource_name(base, used)
        b = _unique_resource_name(base + "_other", used)
        assert a != b
        assert len(a) <= 30 and len(b) <= 30

    def test_dots_and_hyphens_normalized(self) -> None:
        name = _unique_resource_name("cat.sch-tbl", set())
        assert "." not in name and "-" not in name


@pytest.mark.unit
class TestExtractResourcesSanitizeNames:
    def test_table_resource_name_within_limit(self) -> None:
        tables = {
            # 34-char key — the matrix repro.
            "tbl_3978880469080159_trace_unified": TableModel(
                name="cat.sch.3978880469080159_trace_unified"
            ),
            "tbl_products": TableModel(name="cat.sch.products"),
        }
        resources = _extract_table_resources(tables)
        names = [r["name"] for r in resources]
        assert all(2 <= len(n) <= 30 for n in names)
        # table_name (the actual UC target) is preserved untouched.
        assert any(
            r["table_name"] == "cat.sch.3978880469080159_trace_unified"
            for r in resources
        )

    def test_function_resource_name_within_limit(self) -> None:
        functions = {
            "fn_a_very_long_function_name_exceeding_limit": FunctionModel(
                name="cat.sch.a_very_long_function_name_exceeding_limit"
            ),
        }
        resources = _extract_function_resources(functions)
        assert all(2 <= len(r["name"]) <= 30 for r in resources)

    def test_two_long_table_keys_stay_distinct(self) -> None:
        tables = {
            "tbl_" + ("x_prefix_" * 5) + "alpha": TableModel(name="cat.sch.alpha"),
            "tbl_" + ("x_prefix_" * 5) + "beta": TableModel(name="cat.sch.beta"),
        }
        names = [r["name"] for r in _extract_table_resources(tables)]
        assert len(names) == len(set(names))  # no collision
        assert all(len(n) <= 30 for n in names)


@pytest.mark.unit
class TestWarehouseResourceNamesSanitized:
    """A warehouse derived from a Genie room is keyed ``<room-key>_warehouse``.

    That suffix spends 10 of the 30 characters before the room key is counted, so
    any room key past 20 chars overflowed the limit and 400'd the deploy. Before
    Genie warehouse discovery worked, the derived key was never produced on the
    common bare-``space_id`` path, so this was unreachable; now it is the default.
    """

    def test_long_derived_key_is_truncated_to_the_limit(self) -> None:
        warehouses = {
            "retail_inventory_genie_warehouse": WarehouseModel(
                name="wh", warehouse_id="abc123"
            )
        }
        resources = _extract_warehouse_resources(warehouses)

        assert len(resources) == 1
        name = resources[0]["name"]
        assert 2 <= len(name) <= 30
        assert resources[0]["sql_warehouse_id"] == "abc123"

    def test_short_key_is_left_alone(self) -> None:
        """Existing short keys must keep their name — renaming them would break
        any binding that refers to the resource.
        """
        warehouses = {"orders_warehouse": WarehouseModel(name="wh", warehouse_id="w1")}
        resources = _extract_warehouse_resources(warehouses)

        assert resources[0]["name"] == "orders_warehouse"

    def test_two_long_keys_stay_distinct(self) -> None:
        """Truncation must not fuse two keys that share a 30-char prefix."""
        warehouses = {
            "retail_inventory_genie_room_one_warehouse": WarehouseModel(
                name="a", warehouse_id="w1"
            ),
            "retail_inventory_genie_room_two_warehouse": WarehouseModel(
                name="b", warehouse_id="w2"
            ),
        }
        resources = _extract_warehouse_resources(warehouses)
        names = [r["name"] for r in resources]

        assert len(set(names)) == 2
        assert all(2 <= len(n) <= 30 for n in names)

    def test_obo_warehouse_is_still_skipped(self) -> None:
        """Sanitizing the name must not disturb the OBO filter."""
        warehouses = {
            "obo_warehouse": WarehouseModel(
                name="wh", warehouse_id="w1", on_behalf_of_user=True
            )
        }

        assert _extract_warehouse_resources(warehouses) == []


@pytest.mark.unit
class TestGenerateAppResourcesNormalizesEveryName:
    """``generate_app_resources`` normalizes names centrally.

    Sanitizing inside individual extractors could only ever cover the types that
    remembered to do it, and each extractor's ``used`` set is blind to the other
    eleven — two types emitting the same 30-char name would ship two identically
    named resources, leaving any ``value_from`` binding ambiguous. The single
    pass over the assembled list is what makes the guarantee global.
    """

    @staticmethod
    def _config(**resource_kwargs: object) -> AppConfig:
        llm = LLMModel(name="databricks-claude-sonnet-4-5")
        agent = AgentModel(name="a", model=llm)
        return AppConfig(
            resources=ResourcesModel(llms={"default_llm": llm}, **resource_kwargs),
            agents={"a": agent},
            app=AppModel(name="normalize-test", agents=[agent]),
        )

    def test_long_genie_room_key_is_brought_within_limit(self) -> None:
        """``_extract_genie_resources`` emits the mapping key verbatim; a 42-char
        room key produced a 43-char resource name that the Apps API rejects.
        """
        key = "retail_inventory_genie_room_for_the_stores"
        config = self._config(
            genie_rooms={key: GenieRoomModel(space_id="01f1539923891")}
        )

        resources = generate_app_resources(config)
        genie = [r for r in resources if r.get("type") == "genie-space"]

        assert len(genie) == 1
        assert 2 <= len(genie[0]["name"]) <= 30

    def test_names_are_unique_across_resource_types(self) -> None:
        """A warehouse and a volume keyed to the same 30-char prefix must not
        both claim that name.
        """
        shared = "shared_prefix_that_is_quite_long_indeed"
        config = self._config(
            warehouses={shared: WarehouseModel(name="wh", warehouse_id="w1")},
            volumes={shared: VolumeModel(name="cat.sch.vol")},
        )

        names = [r["name"] for r in generate_app_resources(config)]

        assert len(names) == len(set(names))
        assert all(2 <= len(n) <= 30 for n in names)

    def test_every_name_obeys_the_limit(self) -> None:
        long_key = "an_extremely_long_resource_mapping_key_beyond_the_limit"
        config = self._config(
            warehouses={f"{long_key}_wh": WarehouseModel(name="w", warehouse_id="w1")},
            genie_rooms={f"{long_key}_genie": GenieRoomModel(space_id="sp1")},
            volumes={f"{long_key}_vol": VolumeModel(name="cat.sch.vol")},
            connections={f"{long_key}_conn": ConnectionModel(name="my_conn")},
        )

        names = [r["name"] for r in generate_app_resources(config)]

        assert names, "expected resources to be generated"
        assert all(2 <= len(n) <= 30 for n in names)
        assert len(names) == len(set(names))

    def test_secret_and_warehouse_wanting_the_same_name_stay_distinct(self) -> None:
        """The realistic cross-type collision, and the one no per-extractor set
        could catch: a secret resource is named ``<scope>_<key>``, so a warehouse
        keyed after that same pair wants the identical name. Both used to ship as
        ``sc_wh``, leaving two same-named app resources.
        """
        llm = LLMModel(name="databricks-claude-sonnet-4-5")
        agent = AgentModel(name="a", model=llm)
        service_principal = ServicePrincipalModel(
            name="dao-ai-sp",
            client_id=SecretVariableModel(scope="sc", secret="wh"),
            client_secret=SecretVariableModel(scope="sc", secret="cs"),
        )
        config = AppConfig(
            resources=ResourcesModel(
                llms={"default_llm": llm},
                warehouses={"sc_wh": WarehouseModel(name="wh", warehouse_id="w1")},
            ),
            agents={"a": agent},
            app=AppModel(
                name="normalize-test",
                agents=[agent],
                service_principal=service_principal,
            ),
        )

        resources = generate_app_resources(config)
        names = [r["name"] for r in resources]
        warehouse = next(r for r in resources if r.get("type") == "sql-warehouse")

        assert len(names) == len(set(names))
        # Extraction order decides the winner: warehouses are extracted before
        # secrets, so the warehouse keeps the clean name.
        assert warehouse["name"] == "sc_wh"
        assert {r["name"] for r in resources if r.get("type") == "secret"} == {
            "sc_wh_1",
            "sc_cs",
        }
