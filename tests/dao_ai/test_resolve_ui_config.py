"""Tests for resolve_ui_config — defaults Console title/subtitle to the
deployed agent's name/description when AppUIModel doesn't set them."""

import pytest

from dao_ai.apps.chat_ui import resolve_ui_config
from dao_ai.config import AppUIModel


class TestResolveUIConfig:
    @pytest.mark.unit
    def test_defaults_to_app_name_and_description_when_ui_absent(self) -> None:
        cfg = resolve_ui_config(app_name="my_agent", app_description="Does things", ui=None)
        assert cfg["title"] == "my_agent"
        assert cfg["subtitle"] == "Does things"

    @pytest.mark.unit
    def test_no_description_yields_title_only(self) -> None:
        cfg = resolve_ui_config(app_name="my_agent", app_description=None, ui=None)
        assert cfg["title"] == "my_agent"
        assert "subtitle" not in cfg

    @pytest.mark.unit
    def test_explicit_ui_title_subtitle_win(self) -> None:
        ui = AppUIModel(title="Custom Title", subtitle="Custom Sub")
        cfg = resolve_ui_config(app_name="my_agent", app_description="Does things", ui=ui)
        assert cfg["title"] == "Custom Title"
        assert cfg["subtitle"] == "Custom Sub"

    @pytest.mark.unit
    def test_partial_ui_fills_only_missing(self) -> None:
        ui = AppUIModel(mode="developer")  # no title/subtitle
        cfg = resolve_ui_config(app_name="my_agent", app_description="Does things", ui=ui)
        assert cfg["title"] == "my_agent"
        assert cfg["subtitle"] == "Does things"
        assert cfg["mode"] == "developer"
