"""Tests for AppUIModel — the config gate for the dao-ai Console UI."""

import pytest
from pydantic import ValidationError

from dao_ai.config import AppUIModel


class TestAppUIModel:
    @pytest.mark.unit
    def test_defaults_yield_full_console(self) -> None:
        ui = AppUIModel()
        assert ui.enabled is True
        assert ui.mode == "end_user"
        assert ui.inspector is True
        assert ui.session_history is True
        assert ui.title is None
        assert ui.subtitle is None

    @pytest.mark.unit
    def test_developer_mode_accepted(self) -> None:
        ui = AppUIModel(mode="developer", inspector=False, subtitle="x")
        assert ui.mode == "developer"
        assert ui.inspector is False
        assert ui.subtitle == "x"

    @pytest.mark.unit
    def test_invalid_mode_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AppUIModel(mode="admin")

    @pytest.mark.unit
    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            AppUIModel(unknown_field=True)

    @pytest.mark.unit
    def test_round_trips_through_json_dump(self) -> None:
        ui = AppUIModel(mode="developer")
        dumped = ui.model_dump(mode="json")
        assert dumped["mode"] == "developer"
        assert AppUIModel(**dumped) == ui
