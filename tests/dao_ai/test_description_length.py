"""Verify the 200-char cap on IsDatabricksResource description fields.

The cap matches the Databricks Apps platform limit on ``AppResource.description``;
overlong values would otherwise fail only at deploy time.
"""

import pytest
from pydantic import ValidationError

from dao_ai.config import (
    APP_RESOURCE_DESCRIPTION_MAX_LENGTH,
    DatabaseModel,
    GenieRoomModel,
    InferenceEndpointModel,
    WarehouseModel,
)

_MAX = APP_RESOURCE_DESCRIPTION_MAX_LENGTH


@pytest.mark.unit
def test_max_length_constant_is_200() -> None:
    assert _MAX == 200


@pytest.mark.unit
@pytest.mark.parametrize(
    "factory",
    [
        lambda desc: InferenceEndpointModel(name="ep", description=desc),
        lambda desc: WarehouseModel(name="wh", description=desc),
        lambda desc: GenieRoomModel(name="g", space_id="s", description=desc),
        lambda desc: DatabaseModel(name="db", project="p", description=desc),
    ],
    ids=["inference_endpoint", "warehouse", "genie_room", "database"],
)
def test_description_length_limit(factory) -> None:
    assert factory(None).description is None
    assert factory("").description == ""
    assert factory("a" * _MAX).description == "a" * _MAX

    with pytest.raises(ValidationError) as exc_info:
        factory("a" * (_MAX + 1))
    assert "description" in str(exc_info.value)
