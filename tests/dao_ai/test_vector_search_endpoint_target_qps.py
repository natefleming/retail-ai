"""Tests for VectorSearchEndpoint.target_qps support."""

import pytest
from pydantic import ValidationError

from dao_ai.config import VectorSearchEndpoint, VectorSearchEndpointType


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
