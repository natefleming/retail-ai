"""Tests for embedding dimension auto-discovery in memory store managers."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from dao_ai.config import LLMModel, StoreModel
from dao_ai.memory.core import _resolve_embedding_dims


class TestResolveEmbeddingDims:
    """Tests for the _resolve_embedding_dims helper."""

    @pytest.mark.unit
    def test_returns_configured_dims_when_provided(self) -> None:
        embeddings = Mock()
        result = _resolve_embedding_dims(embeddings, configured_dims=1024)
        assert result == 1024
        embeddings.embed_documents.assert_not_called()

    @pytest.mark.unit
    def test_auto_detects_dims_when_none(self) -> None:
        embeddings = Mock()
        embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3, 0.4]]

        result = _resolve_embedding_dims(embeddings, configured_dims=None)

        assert result == 4
        embeddings.embed_documents.assert_called_once_with(["test"])

    @pytest.mark.unit
    def test_auto_detects_large_dims(self) -> None:
        embeddings = Mock()
        embeddings.embed_documents.return_value = [[0.0] * 1536]

        result = _resolve_embedding_dims(embeddings, configured_dims=None)

        assert result == 1536


class TestStoreModelDimsDefault:
    """Tests for StoreModel.dims default behavior."""

    @pytest.mark.unit
    def test_dims_defaults_to_none(self) -> None:
        store = StoreModel(name="test_store")
        assert store.dims is None

    @pytest.mark.unit
    def test_dims_accepts_explicit_value(self) -> None:
        store = StoreModel(name="test_store", dims=1024)
        assert store.dims == 1024

    @pytest.mark.unit
    def test_dims_accepts_none_explicitly(self) -> None:
        store = StoreModel(name="test_store", dims=None)
        assert store.dims is None


class TestInMemoryStoreManagerDimsDiscovery:
    """Tests for InMemoryStoreManager embedding dims auto-discovery."""

    @pytest.mark.unit
    @patch("dao_ai.memory.core.DatabricksEmbeddings")
    def test_store_with_embedding_model_auto_discovers_dims(
        self, mock_embeddings_cls: MagicMock
    ) -> None:
        mock_embeddings = Mock()
        mock_embeddings.embed_documents.return_value = [[0.0] * 768]
        mock_embeddings_cls.return_value = mock_embeddings

        embedding_model = LLMModel(name="databricks-gte-large-en")
        store_model = StoreModel(
            name="test_store",
            embedding_model=embedding_model,
        )

        from dao_ai.memory.core import InMemoryStoreManager

        manager = InMemoryStoreManager(store_model)
        store = manager.store()

        assert store is not None
        mock_embeddings.embed_documents.assert_called_once_with(["test"])

    @pytest.mark.unit
    @patch("dao_ai.memory.core.DatabricksEmbeddings")
    def test_store_with_explicit_dims_skips_discovery(
        self, mock_embeddings_cls: MagicMock
    ) -> None:
        mock_embeddings = Mock()
        mock_embeddings_cls.return_value = mock_embeddings

        embedding_model = LLMModel(name="databricks-gte-large-en")
        store_model = StoreModel(
            name="test_store",
            embedding_model=embedding_model,
            dims=1024,
        )

        from dao_ai.memory.core import InMemoryStoreManager

        manager = InMemoryStoreManager(store_model)
        store = manager.store()

        assert store is not None
        mock_embeddings.embed_documents.assert_not_called()

    @pytest.mark.unit
    def test_store_without_embedding_model_skips_discovery(self) -> None:
        store_model = StoreModel(name="test_store")

        from dao_ai.memory.core import InMemoryStoreManager

        manager = InMemoryStoreManager(store_model)
        store = manager.store()

        assert store is not None
