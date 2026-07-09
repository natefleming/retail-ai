"""Unit tests for the ``AnyVectorStore`` discriminated union.

Covers:
  - Both concrete types register under ``resources.vector_stores``.
  - Legacy YAML entries (no ``type`` field) default to
    :class:`AiSearchVectorStoreModel`.
  - :class:`LakebaseVectorStoreModel` delegates the IsDatabricksResource
    trio (``as_resources``, ``api_scopes``, ``on_behalf_of_user``) to its
    nested :class:`DatabaseModel`, so deploy-path iteration works
    polymorphically.
  - The app-bundle vector-search-index extractor filters to
    :class:`AiSearchVectorStoreModel` entries only.
  - Backwards-compat aliases still resolve.
"""

from __future__ import annotations

import pytest

from dao_ai.config import (
    AiSearchIndexModel,
    AiSearchVectorStoreModel,
    AppConfig,
    DatabaseModel,
    IndexModel,
    LakebaseVectorStoreModel,
    ResourcesModel,
    SchemaModel,
    VectorSearchEndpoint,
    VectorStoreModel,
)


def _ai_search() -> AiSearchVectorStoreModel:
    return AiSearchVectorStoreModel(
        index=IndexModel(
            name="c.s.i", schema=SchemaModel(catalog_name="c", schema_name="s")
        ),
        endpoint=VectorSearchEndpoint(name="e"),
    )


def _lakebase() -> LakebaseVectorStoreModel:
    return LakebaseVectorStoreModel(
        database=DatabaseModel(project="p"),
        table="kb_articles",
        content_column="passage",
        embedding_column="embedding",
        embedding_model="databricks-gte-large-en",
    )


@pytest.mark.unit
class TestBackwardsCompatibleAliases:
    def test_legacy_names_resolve_to_new_class(self) -> None:
        # Python code importing either legacy name keeps working
        assert AiSearchIndexModel is AiSearchVectorStoreModel
        assert VectorStoreModel is AiSearchVectorStoreModel


@pytest.mark.unit
class TestTypeDiscriminator:
    def test_ai_search_default_type(self) -> None:
        assert _ai_search().type == "ai_search"

    def test_lakebase_default_type(self) -> None:
        assert _lakebase().type == "lakebase_search"


@pytest.mark.unit
class TestUnionDispatchFromDict:
    """YAML/dict input goes through the ``_vector_store_discriminator``
    callable — legacy entries without ``type`` default to ai_search."""

    def test_registers_both_types_side_by_side(self) -> None:
        resources = ResourcesModel(
            vector_stores={"ai": _ai_search(), "lb": _lakebase()},
        )
        assert isinstance(resources.vector_stores["ai"], AiSearchVectorStoreModel)
        assert isinstance(resources.vector_stores["lb"], LakebaseVectorStoreModel)

    def test_dict_with_lakebase_type_dispatches_correctly(self) -> None:
        resources = ResourcesModel(
            vector_stores={
                "lb": {
                    "type": "lakebase_search",
                    "database": {"project": "p"},
                    "table": "t",
                    "content_column": "c",
                    "embedding_column": "e",
                    "embedding_model": "m",
                }
            }
        )
        assert isinstance(resources.vector_stores["lb"], LakebaseVectorStoreModel)

    def test_dict_without_type_defaults_to_ai_search(self) -> None:
        """Legacy YAML back-compat — the discriminator returns 'ai_search'
        when ``type`` is absent, so pre-refactor configs continue to load."""
        resources = ResourcesModel(
            vector_stores={
                "legacy_ai": {
                    # NO ``type`` field
                    "index": {
                        "name": "c.s.i",
                        "schema": {"catalog_name": "c", "schema_name": "s"},
                    },
                    "endpoint": {"name": "e"},
                }
            }
        )
        assert isinstance(
            resources.vector_stores["legacy_ai"], AiSearchVectorStoreModel
        )


@pytest.mark.unit
class TestLakebaseDelegation:
    """LakebaseVectorStoreModel is NOT IsDatabricksResource, but it
    delegates the three members deploy paths iterate to its database."""

    def test_on_behalf_of_user_delegates_to_database(self) -> None:
        lb = _lakebase()
        assert lb.on_behalf_of_user == lb.database.on_behalf_of_user

    def test_api_scopes_delegates_to_database(self) -> None:
        lb = _lakebase()
        assert lb.api_scopes == lb.database.api_scopes

    def test_as_resources_delegates_to_database(self) -> None:
        lb = _lakebase()
        assert lb.as_resources() == lb.database.as_resources()


@pytest.mark.unit
class TestAppBundleVectorSearchExtractor:
    """``_extract_vector_search_resources`` emits vector-search-index bundle
    entries only for AI Search vector stores; Lakebase entries produce
    nothing (they authenticate via their DatabaseModel at runtime)."""

    def test_skips_lakebase_entries(self) -> None:
        from dao_ai.apps.resources import _extract_vector_search_resources

        vector_stores = {"ai": _ai_search(), "lb": _lakebase()}
        resources = _extract_vector_search_resources(vector_stores)
        assert len(resources) == 1
        assert resources[0]["name"] == "ai"
        assert resources[0]["type"] == "vector-search-index"

    def test_only_lakebase_produces_empty(self) -> None:
        from dao_ai.apps.resources import _extract_vector_search_resources

        vector_stores = {"lb": _lakebase()}
        assert _extract_vector_search_resources(vector_stores) == []
