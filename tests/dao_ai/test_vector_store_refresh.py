"""Tests for ``VectorStoreModel.refresh()`` and ``VectorStoreModel.from_index()``.

The refresh path hydrates a model that has only an ``index`` reference into a
fully-populated model with ``source_table``, ``embedding_source_column``,
``embedding_model``, ``endpoint``, ``primary_key``, and ``columns`` pulled from
the live ``index.describe()`` response.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from dao_ai.config import (
    IndexModel,
    LLMModel,
    VectorSearchEndpoint,
    VectorStoreModel,
)


def _describe_payload(**overrides: Any) -> dict[str, Any]:
    """Default ``index.describe()`` shape, with optional overrides."""
    base: dict[str, Any] = {
        "name": "cat.sch.products_index",
        "endpoint_name": "retail_endpoint",
        "primary_key": "product_id",
        "delta_sync_index_spec": {
            "source_table": "cat.sch.products",
            "embedding_source_columns": [
                {
                    "name": "description",
                    "embedding_model_endpoint_name": "databricks-gte-large-en",
                }
            ],
            "columns_to_sync": ["product_id", "description", "price"],
        },
    }
    base.update(overrides)
    return base


@pytest.mark.unit
class TestVectorStoreRefresh:
    def test_refresh_populates_fields_from_describe(self):
        vs = VectorStoreModel(index=IndexModel(name="cat.sch.products_index"))
        result = vs.refresh(details=_describe_payload())

        assert result is vs
        assert vs.source_table is not None
        assert vs.source_table.full_name == "cat.sch.products"
        assert vs.embedding_source_column == "description"
        assert isinstance(vs.embedding_model, LLMModel)
        assert vs.embedding_model.name == "databricks-gte-large-en"
        assert isinstance(vs.endpoint, VectorSearchEndpoint)
        assert vs.endpoint.name == "retail_endpoint"
        assert vs.primary_key == "product_id"
        assert vs.columns == ["product_id", "description", "price"]

    def test_refresh_is_idempotent(self):
        vs = VectorStoreModel(index=IndexModel(name="cat.sch.products_index"))
        payload = _describe_payload()
        vs.refresh(details=payload)
        first_columns = list(vs.columns)
        first_endpoint = vs.endpoint.name
        vs.refresh(details=payload)
        assert vs.columns == first_columns
        assert vs.endpoint.name == first_endpoint

    def test_refresh_handles_missing_optional_fields(self):
        vs = VectorStoreModel(index=IndexModel(name="cat.sch.products_index"))
        # No embedding_source_columns, no columns_to_sync
        vs.refresh(
            details={
                "endpoint_name": "ep",
                "primary_key": "pk",
                "delta_sync_index_spec": {"source_table": "cat.sch.t"},
            }
        )
        assert vs.endpoint.name == "ep"
        assert vs.primary_key == "pk"
        assert vs.source_table.full_name == "cat.sch.t"
        # Untouched fields remain at their defaults / None
        assert vs.embedding_source_column is None
        assert vs.embedding_model is None

    def test_refresh_raises_without_index(self):
        vs = VectorStoreModel.__new__(VectorStoreModel)
        # bypass __init__ which would require index or source_table
        object.__setattr__(vs, "__dict__", {"index": None})
        with pytest.raises(ValueError, match="index"):
            vs.refresh(details={})

    def test_refresh_force_invalidates_cache(self):
        vs = VectorStoreModel(index=IndexModel(name="cat.sch.products_index"))
        # First, prime the cache via the details path
        vs.refresh(details=_describe_payload())
        # Then directly populate cache and verify force triggers a re-fetch
        vs._index_details = _describe_payload(endpoint_name="cached_endpoint")

        with patch("dao_ai.providers.databricks.DatabricksProvider") as MockProvider:
            mock_provider = MagicMock()
            mock_index = MagicMock()
            mock_index.describe.return_value = _describe_payload(
                endpoint_name="fresh_endpoint"
            )
            mock_provider.get_vector_index.return_value = mock_index
            MockProvider.return_value = mock_provider

            vs.refresh(force=True)
            assert vs.endpoint.name == "fresh_endpoint"
            mock_provider.get_vector_index.assert_called_once()


@pytest.mark.unit
class TestFromIndex:
    def test_from_index_returns_hydrated_model(self):
        with patch("dao_ai.providers.databricks.DatabricksProvider") as MockProvider:
            mock_provider = MagicMock()
            mock_index = MagicMock()
            mock_index.describe.return_value = _describe_payload()
            mock_provider.get_vector_index.return_value = mock_index
            mock_provider.find_primary_key.return_value = ["product_id"]
            MockProvider.return_value = mock_provider

            vs = VectorStoreModel.from_index("cat.sch.products_index")

        assert vs.index.full_name == "cat.sch.products_index"
        assert vs.source_table.full_name == "cat.sch.products"
        assert vs.embedding_source_column == "description"
        assert vs.endpoint.name == "retail_endpoint"
        assert vs._resolved is True
