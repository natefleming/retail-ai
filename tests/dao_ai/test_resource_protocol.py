"""Tests for the shared ``Provisionable`` / ``Refreshable`` / ``ManagedResource`` protocol.

Verifies that the dual-mode resource models declare the right interfaces so
callers can type-check polymorphic collections (e.g. iterate every
``Provisionable`` in a config and call ``.create()``).
"""

from __future__ import annotations

import pytest

from dao_ai.config import (
    GenieRoomModel,
    IndexModel,
    SchemaModel,
    TableModel,
    VectorStoreModel,
    VolumeModel,
    VolumePathModel,
    WarehouseModel,
)
from dao_ai.resource_protocol import (
    ManagedResource,
    Provisionable,
    Refreshable,
)


@pytest.mark.unit
class TestProtocolImplementations:
    """isinstance checks for every model that opts into the lifecycle contracts."""

    def test_genie_room_is_managed_resource(self):
        room = GenieRoomModel(name="example")
        assert isinstance(room, ManagedResource)
        assert isinstance(room, Provisionable)
        assert isinstance(room, Refreshable)

    def test_vector_store_is_managed_resource(self):
        vs = VectorStoreModel(index=IndexModel(name="cat.sch.idx"))
        assert isinstance(vs, ManagedResource)
        assert isinstance(vs, Provisionable)
        assert isinstance(vs, Refreshable)

    def test_schema_model_is_provisionable_only(self):
        schema = SchemaModel(catalog_name="cat", schema_name="sch")
        assert isinstance(schema, Provisionable)
        assert not isinstance(schema, Refreshable)
        assert not isinstance(schema, ManagedResource)

    def test_volume_model_is_provisionable_only(self):
        schema = SchemaModel(catalog_name="cat", schema_name="sch")
        volume = VolumeModel(schema=schema, name="my_volume")
        assert isinstance(volume, Provisionable)
        assert not isinstance(volume, Refreshable)

    def test_volume_path_model_is_provisionable_only(self):
        path = VolumePathModel(path="/Volumes/cat/sch/vol/sub")
        assert isinstance(path, Provisionable)
        assert not isinstance(path, Refreshable)

    def test_warehouse_model_is_neither(self):
        # Warehouses don't implement either contract today (see plan: deferred).
        wh = WarehouseModel(name="wh", warehouse_id="abc")
        assert not isinstance(wh, Provisionable)
        assert not isinstance(wh, Refreshable)


@pytest.mark.unit
class TestPolymorphicIteration:
    """The contracts let callers operate on heterogeneous collections."""

    def test_filter_provisionable_from_mixed_list(self):
        schema = SchemaModel(catalog_name="cat", schema_name="sch")
        room = GenieRoomModel(name="genie")
        vs = VectorStoreModel(index=IndexModel(name="cat.sch.idx"))
        warehouse = WarehouseModel(name="wh", warehouse_id="abc")
        table = TableModel(schema=schema, name="t1")

        all_models = [schema, room, vs, warehouse, table]
        provisionable = [m for m in all_models if isinstance(m, Provisionable)]
        assert {type(m).__name__ for m in provisionable} == {
            "SchemaModel",
            "GenieRoomModel",
            "VectorStoreModel",
        }

    def test_filter_refreshable_from_mixed_list(self):
        schema = SchemaModel(catalog_name="cat", schema_name="sch")
        room = GenieRoomModel(name="genie")
        vs = VectorStoreModel(index=IndexModel(name="cat.sch.idx"))

        refreshable = [
            m for m in (schema, room, vs) if isinstance(m, Refreshable)
        ]
        assert {type(m).__name__ for m in refreshable} == {
            "GenieRoomModel",
            "VectorStoreModel",
        }


@pytest.mark.unit
class TestProtocolContract:
    """The contract guarantees ``refresh()`` returns self for chaining."""

    def test_genie_refresh_returns_self(self):
        room = GenieRoomModel(name="genie")
        assert room.refresh(payload={}) is room

    def test_vector_store_refresh_returns_self(self):
        vs = VectorStoreModel(index=IndexModel(name="cat.sch.idx"))
        assert vs.refresh(details={}) is vs
