"""Tests for discover_custom_input_fields — surfaces the configurable fields a
CustomFieldValidationMiddleware declares so the Console can prepopulate them."""

import pytest

from dao_ai.apps.custom_inputs import discover_custom_input_fields
from dao_ai.config import MiddlewareModel


def _store_validation() -> MiddlewareModel:
    return MiddlewareModel(
        name="dao_ai.middleware.create_custom_field_validation_middleware",
        args={
            "fields": [
                {
                    "name": "store_num",
                    "description": "Your store number for inventory lookups",
                    "example_value": "12345",
                },
                {
                    "name": "user_id",  # runtime-managed → excluded
                    "description": "Your unique user identifier",
                    "required": False,
                    "example_value": "my_user_id",
                },
            ]
        },
    )


class TestDiscoverCustomInputFields:
    @pytest.mark.unit
    def test_extracts_fields_and_excludes_runtime_managed(self) -> None:
        # A nested structure (dicts + lists + models) like the real config graph.
        config = {"app": {"orchestration": {"middleware": [_store_validation()]}}}
        fields = discover_custom_input_fields(config)
        assert len(fields) == 1
        field = fields[0]
        assert field["name"] == "store_num"
        assert field["required"] is True  # defaults to required
        assert field["example_value"] == "12345"
        assert "inventory" in field["description"]

    @pytest.mark.unit
    def test_no_such_middleware_yields_empty(self) -> None:
        other = MiddlewareModel(name="dao_ai.middleware.create_summarization_middleware")
        assert discover_custom_input_fields([other]) == []

    @pytest.mark.unit
    def test_dedupes_by_name_first_wins(self) -> None:
        dup = MiddlewareModel(
            name="dao_ai.middleware.create_custom_field_validation_middleware",
            args={"fields": [{"name": "store_num", "example_value": "999"}]},
        )
        fields = discover_custom_input_fields([_store_validation(), dup])
        assert [f["name"] for f in fields] == ["store_num"]
        assert fields[0]["example_value"] == "12345"

    @pytest.mark.unit
    def test_handles_no_config(self) -> None:
        assert discover_custom_input_fields(None) == []
