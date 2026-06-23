"""Tests that validate the BackgroundModel schema and its wiring into AppModel."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dao_ai.config import AppModel, BackgroundModel, DatabaseModel


def _db() -> DatabaseModel:
    # Autoscaling Lakebase shape — the lightest valid DatabaseModel.
    return DatabaseModel(project="my-lakebase-project")


def test_background_model_defaults():
    m = BackgroundModel(database=_db())
    assert m.default_enabled is False
    assert m.max_duration_seconds == 1800
    assert m.poll_interval_seconds == 1.0
    assert m.responses_table_name == "dao_ai_responses"
    assert m.messages_table_name == "dao_ai_response_messages"


def test_background_model_rejects_bad_values():
    with pytest.raises(ValidationError):
        BackgroundModel(database=_db(), max_duration_seconds=0)
    with pytest.raises(ValidationError):
        BackgroundModel(database=_db(), poll_interval_seconds=0)


def test_app_model_background_optional():
    from dao_ai.config import RegisteredModelModel

    app = AppModel(
        name="demo",
        registered_model=RegisteredModelModel(name="demo"),
        agents=[
            # Minimal agent — AppModel requires at least one.
            {
                "name": "agent_a",
                "model": {"name": "databricks-claude-sonnet-4"},
            }
        ],
    )
    assert app.background is None


def test_app_model_accepts_background_block():
    from dao_ai.config import RegisteredModelModel

    app = AppModel(
        name="demo",
        registered_model=RegisteredModelModel(name="demo"),
        agents=[
            {
                "name": "agent_a",
                "model": {"name": "databricks-claude-sonnet-4"},
            }
        ],
        background=BackgroundModel(database=_db()),
    )
    assert app.background is not None
    assert app.background.database.project == "my-lakebase-project"
