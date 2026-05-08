"""Tests for VectorSearchEndpoint.target_qps support."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from dao_ai.config import (
    IndexModel,
    LLMModel,
    SchemaModel,
    TableModel,
    VectorSearchEndpoint,
    VectorSearchEndpointType,
    VectorStoreModel,
)
from dao_ai.providers.databricks import DatabricksProvider


@pytest.mark.unit
def test_target_qps_happy_path_on_standard_endpoint() -> None:
    """target_qps=500 on a STANDARD endpoint validates and is preserved."""
    endpoint = VectorSearchEndpoint(
        name="my-endpoint",
        type=VectorSearchEndpointType.STANDARD,
        target_qps=500,
    )
    assert endpoint.target_qps == 500
    assert endpoint.name == "my-endpoint"


@pytest.mark.unit
def test_target_qps_default_is_none() -> None:
    """target_qps is optional and defaults to None."""
    endpoint = VectorSearchEndpoint(name="my-endpoint")
    assert endpoint.target_qps is None


@pytest.mark.unit
def test_target_qps_rejected_on_optimized_storage_endpoint() -> None:
    """target_qps on OPTIMIZED_STORAGE raises with a clear message."""
    with pytest.raises(ValidationError) as exc_info:
        VectorSearchEndpoint(
            name="my-endpoint",
            type=VectorSearchEndpointType.OPTIMIZED_STORAGE,
            target_qps=500,
        )
    assert "only supported on STANDARD" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.parametrize("invalid_value", [0, -1, -500])
def test_target_qps_rejects_non_positive_values(invalid_value: int) -> None:
    """target_qps must be > 0; zero and negatives are rejected."""
    with pytest.raises(ValidationError):
        VectorSearchEndpoint(
            name="my-endpoint",
            type=VectorSearchEndpointType.STANDARD,
            target_qps=invalid_value,
        )


@pytest.mark.unit
def test_target_qps_omitted_from_serialized_output_when_none() -> None:
    """When target_qps is unset, model_dump(exclude_none=True) omits it."""
    endpoint = VectorSearchEndpoint(name="my-endpoint")
    dumped = endpoint.model_dump(exclude_none=True)
    assert "target_qps" not in dumped
    # Sanity: name and type still present.
    assert dumped["name"] == "my-endpoint"


@pytest.mark.unit
def test_target_qps_present_in_serialized_output_when_set() -> None:
    """When target_qps is set, model_dump includes it."""
    endpoint = VectorSearchEndpoint(name="my-endpoint", target_qps=500)
    dumped = endpoint.model_dump(exclude_none=True)
    assert dumped["target_qps"] == 500


# ==================== Schema tests above ====================
# ==================== Provider call-site tests ====================
#
# These tests verify that DatabricksProvider.create_vector_store passes the
# user-configured target_qps through to VectorSearchClient.create_endpoint_and_wait
# under the SDK's current kwarg name `min_qps`. We don't construct a real
# VectorSearchClient — we set the provider's `vsc` setter directly to a
# MagicMock so we can assert call kwargs.


def _build_provisioning_vector_store(
    target_qps: int | None = None,
) -> VectorStoreModel:
    """Build a VectorStoreModel in provisioning mode (the only mode that
    triggers create_vector_store's endpoint-creation path)."""
    schema = SchemaModel(catalog_name="test_cat", schema_name="test_sch")
    endpoint = VectorSearchEndpoint(
        name="test_endpoint",
        type=VectorSearchEndpointType.STANDARD,
        target_qps=target_qps,
    )
    return VectorStoreModel(
        index=IndexModel(schema=schema, name="test_index"),
        source_table=TableModel(schema=schema, name="test_source"),
        embedding_source_column="description",
        embedding_model=LLMModel(name="databricks-gte-large-en"),
        endpoint=endpoint,
        primary_key="id",
    )


def _build_provider_with_mocked_vsc() -> tuple[DatabricksProvider, MagicMock]:
    """Return a DatabricksProvider with a MagicMock VectorSearchClient
    installed via the public setter."""
    provider = DatabricksProvider()
    mock_vsc = MagicMock()
    provider.vsc = mock_vsc
    return provider, mock_vsc


@pytest.mark.unit
def test_create_vector_store_passes_min_qps_when_target_qps_set() -> None:
    """When target_qps is set and the endpoint does not exist, the SDK
    is called with min_qps equal to the configured target_qps."""
    provider, mock_vsc = _build_provider_with_mocked_vsc()
    vector_store = _build_provisioning_vector_store(target_qps=500)

    # Endpoint does not exist; index does not exist (skip wait-loop branch).
    with patch(
        "dao_ai.providers.databricks.endpoint_exists", return_value=False
    ), patch("dao_ai.providers.databricks.index_exists", return_value=False):
        provider.create_vector_store(vector_store)

    mock_vsc.create_endpoint_and_wait.assert_called_once()
    call_kwargs = mock_vsc.create_endpoint_and_wait.call_args.kwargs
    assert call_kwargs["name"] == "test_endpoint"
    assert call_kwargs["endpoint_type"] == "STANDARD"
    assert call_kwargs["verbose"] is True
    assert call_kwargs["min_qps"] == 500


@pytest.mark.unit
def test_create_vector_store_omits_min_qps_when_target_qps_unset() -> None:
    """When target_qps is unset, the SDK is called WITHOUT a min_qps key
    in kwargs (not min_qps=None — the key must be absent)."""
    provider, mock_vsc = _build_provider_with_mocked_vsc()
    vector_store = _build_provisioning_vector_store(target_qps=None)

    with patch(
        "dao_ai.providers.databricks.endpoint_exists", return_value=False
    ), patch("dao_ai.providers.databricks.index_exists", return_value=False):
        provider.create_vector_store(vector_store)

    mock_vsc.create_endpoint_and_wait.assert_called_once()
    call_kwargs = mock_vsc.create_endpoint_and_wait.call_args.kwargs
    assert "min_qps" not in call_kwargs


@pytest.mark.unit
def test_create_vector_store_skips_endpoint_creation_when_endpoint_exists() -> None:
    """When the endpoint already exists, create_endpoint_and_wait is NOT
    called even if target_qps is set (deliberate scope: no auto-reconcile).
    A debug log records the configured value as a breadcrumb."""
    provider, mock_vsc = _build_provider_with_mocked_vsc()
    vector_store = _build_provisioning_vector_store(target_qps=500)

    with patch(
        "dao_ai.providers.databricks.endpoint_exists", return_value=True
    ), patch(
        "dao_ai.providers.databricks.index_exists", return_value=False
    ), patch("dao_ai.providers.databricks.logger") as mock_logger:
        provider.create_vector_store(vector_store)

    mock_vsc.create_endpoint_and_wait.assert_not_called()
    mock_logger.debug.assert_any_call(
        "endpoint already exists; target_qps not reconciled",
        endpoint_name="test_endpoint",
        configured_target_qps=500,
    )


@pytest.mark.unit
def test_create_vector_store_no_breadcrumb_log_when_target_qps_unset() -> None:
    """The 'target_qps not reconciled' debug log is gated on target_qps
    being set; when target_qps is None and the endpoint exists, it
    should not fire."""
    provider, mock_vsc = _build_provider_with_mocked_vsc()
    vector_store = _build_provisioning_vector_store(target_qps=None)

    with patch(
        "dao_ai.providers.databricks.endpoint_exists", return_value=True
    ), patch(
        "dao_ai.providers.databricks.index_exists", return_value=False
    ), patch("dao_ai.providers.databricks.logger") as mock_logger:
        provider.create_vector_store(vector_store)

    mock_vsc.create_endpoint_and_wait.assert_not_called()
    # Verify the breadcrumb log was not emitted.
    breadcrumb_calls = [
        c for c in mock_logger.debug.call_args_list
        if c.args and c.args[0] == "endpoint already exists; target_qps not reconciled"
    ]
    assert breadcrumb_calls == []
