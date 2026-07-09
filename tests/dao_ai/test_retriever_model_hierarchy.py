"""Tests for the BaseRetrieverModel hierarchy + AnyRetriever discriminated union.

Covers:
- The class hierarchy (`AiSearchRetrieverModel` and `LakebaseRetrieverModel` both
  subclass `BaseRetrieverModel`).
- `AppConfig.retrievers` dispatches heterogeneous entries via `AnyRetriever`.
- The callable discriminator defaults missing ``type`` to ``"ai_search"`` so
  existing YAMLs that never wrote ``type:`` continue to parse.
- Both concrete models implement `as_tools()` and return the correct tool.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from dao_ai.config import (
    AiSearchIndexModel,
    AiSearchRetrieverModel,
    AnyRetriever,
    AppConfig,
    BaseRetrieverModel,
    IndexModel,
    LakebaseRetrieverModel,
    LakebaseVectorStoreModel,
    RetrieverType,
    SchemaModel,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _ai_search_dict() -> dict[str, Any]:
    return {
        "vector_store": {
            "index": {
                "schema": {"catalog_name": "cat", "schema_name": "sch"},
                "name": "products_ix",
            }
        }
    }


def _lakebase_dict() -> dict[str, Any]:
    return {
        "type": "lakebase_search",
        "vector_store": {
            "database": {"project": "my-lakebase"},
            "table": "kb_articles",
            "content_column": "passage",
            "embedding_column": "embedding",
            "embedding_model": "databricks-gte-large-en",
            "metadata_columns": ["category"],
        },
    }


# ---------------------------------------------------------------------------
# Class hierarchy
# ---------------------------------------------------------------------------


class TestHierarchy:
    def test_ai_search_retriever_is_base(self) -> None:
        assert issubclass(AiSearchRetrieverModel, BaseRetrieverModel)

    def test_lakebase_retriever_is_base(self) -> None:
        assert issubclass(LakebaseRetrieverModel, BaseRetrieverModel)

    def test_base_is_abstract(self) -> None:
        """BaseRetrieverModel cannot be instantiated directly — as_tools is abstract."""
        with pytest.raises(TypeError):
            BaseRetrieverModel()  # type: ignore[abstract]

    def test_shared_fields_on_base(self) -> None:
        """columns + search_parameters live on the base."""
        assert "columns" in BaseRetrieverModel.model_fields
        assert "search_parameters" in BaseRetrieverModel.model_fields


# ---------------------------------------------------------------------------
# Discriminator dispatch
# ---------------------------------------------------------------------------


class TestDiscriminator:
    def _adapter(self) -> TypeAdapter[AnyRetriever]:
        return TypeAdapter(AnyRetriever)

    def test_missing_type_defaults_to_ai_search(self) -> None:
        adapter = self._adapter()
        r = adapter.validate_python(_ai_search_dict())
        assert isinstance(r, AiSearchRetrieverModel)
        assert r.type == RetrieverType.AI_SEARCH.value

    def test_explicit_ai_search_type(self) -> None:
        adapter = self._adapter()
        r = adapter.validate_python({"type": "ai_search", **_ai_search_dict()})
        assert isinstance(r, AiSearchRetrieverModel)

    def test_explicit_lakebase_search_type(self) -> None:
        adapter = self._adapter()
        r = adapter.validate_python(_lakebase_dict())
        assert isinstance(r, LakebaseRetrieverModel)
        assert r.type == RetrieverType.LAKEBASE_SEARCH.value

    def test_unknown_type_rejected(self) -> None:
        adapter = self._adapter()
        with pytest.raises(ValidationError):
            adapter.validate_python({"type": "not_a_real_retriever", **_ai_search_dict()})

    def test_lakebase_shape_without_type_field_dispatches_wrong(self) -> None:
        """If someone writes lakebase shape but forgets `type: lakebase_search`,
        the default AI Search dispatch will reject the shape at validation.
        This is expected behavior — the discriminator can't infer intent from
        vector_store shape (that's why we default to ai_search)."""
        adapter = self._adapter()
        # Same as _lakebase_dict but with `type` removed. Falls through to
        # AiSearchRetrieverModel which will reject the Lakebase-shaped
        # vector_store.
        raw = _lakebase_dict()
        del raw["type"]
        with pytest.raises(ValidationError):
            adapter.validate_python(raw)


# ---------------------------------------------------------------------------
# AppConfig.retrievers heterogeneous dispatch
# ---------------------------------------------------------------------------


class TestAppConfigRetrievers:
    def test_heterogeneous_retrievers(self) -> None:
        cfg = AppConfig.model_validate(
            {
                "resources": {},
                "retrievers": {
                    "ai": _ai_search_dict(),
                    "lb": _lakebase_dict(),
                },
            }
        )
        assert isinstance(cfg.retrievers["ai"], AiSearchRetrieverModel)
        assert isinstance(cfg.retrievers["lb"], LakebaseRetrieverModel)

    def test_legacy_yaml_without_type_still_parses(self) -> None:
        """Existing YAMLs that never wrote `type:` on retriever entries
        continue to dispatch to AiSearchRetrieverModel (backward compat)."""
        cfg = AppConfig.model_validate(
            {
                "resources": {},
                "retrievers": {
                    "products": {
                        "vector_store": {
                            "index": {
                                "schema": {
                                    "catalog_name": "c",
                                    "schema_name": "s",
                                },
                                "name": "products_ix",
                            }
                        },
                        "columns": ["product_id", "name"],
                    }
                },
            }
        )
        assert isinstance(cfg.retrievers["products"], AiSearchRetrieverModel)


# ---------------------------------------------------------------------------
# as_tools() implementation
# ---------------------------------------------------------------------------


class TestAsTools:
    def test_ai_search_retriever_as_tools_delegates_to_factory(self) -> None:
        """`AiSearchRetrieverModel.as_tools()` calls `create_ai_search_tool`.
        We don't fully build the tool here (that needs UC lookups) — just
        verify the method exists and returns a list."""
        from unittest.mock import patch

        vs = AiSearchIndexModel(
            index=IndexModel(
                schema=SchemaModel(catalog_name="c", schema_name="s"),
                name="ix",
            )
        )
        retriever = AiSearchRetrieverModel(vector_store=vs)

        with patch(
            "dao_ai.tools.create_ai_search_tool",
            return_value="STUB_TOOL",
        ) as mock_factory:
            tools = retriever.as_tools()

        assert tools == ["STUB_TOOL"]
        mock_factory.assert_called_once()
        # The retriever is passed by keyword.
        assert mock_factory.call_args.kwargs["retriever"] is retriever

    def test_lakebase_retriever_as_tools_delegates_to_factory(self) -> None:
        from unittest.mock import patch

        vs = LakebaseVectorStoreModel(**_lakebase_dict()["vector_store"])
        retriever = LakebaseRetrieverModel(vector_store=vs)

        with patch(
            "dao_ai.tools.create_lakebase_search_tool",
            return_value="STUB_TOOL",
        ) as mock_factory:
            tools = retriever.as_tools()

        assert tools == ["STUB_TOOL"]
        mock_factory.assert_called_once()
        assert mock_factory.call_args.kwargs["retriever"] is retriever


# ---------------------------------------------------------------------------
# Rename regression — the old symbol name must be gone from the public API.
# ---------------------------------------------------------------------------


class TestRenameRegression:
    def test_old_symbol_removed(self) -> None:
        """`RetrieverModel` was renamed to `AiSearchRetrieverModel`. Ensure
        no back-compat alias slipped in — the codebase should import the
        new name explicitly."""
        import dao_ai.config as cfg

        assert not hasattr(cfg, "RetrieverModel"), (
            "RetrieverModel should be renamed to AiSearchRetrieverModel; "
            "no back-compat alias per the plan."
        )
        assert hasattr(cfg, "AiSearchRetrieverModel")
