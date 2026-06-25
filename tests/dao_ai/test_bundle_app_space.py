"""Bundle emission tests for ``AppModel.space`` (Databricks App Spaces).

Locks in the contract that ``_build_app_block`` emits
``resources.apps.<name>.space`` only when ``app.space`` is set, matching the
DABs schema (CLI v1.3.0) where ``space`` is a private-preview string field on
the App resource.
"""

from __future__ import annotations

import pytest

from dao_ai.apps.bundle import _build_app_block
from dao_ai.config import (
    AgentModel,
    AppConfig,
    AppModel,
    DeploymentTarget,
    InferenceEndpointModel,
)


def _config(*, space: str | None = None) -> AppConfig:
    extra: dict = {}
    if space is not None:
        extra["space"] = space
    return AppConfig(
        app=AppModel(
            name="dao-ai-space-test",
            description="test agent",
            deployment_target=DeploymentTarget.APPS,
            enable_chat_proxy=False,
            agents=[
                AgentModel(
                    name="greeter",
                    description="test agent",
                    model=InferenceEndpointModel(name="databricks-gpt-5-4-mini"),
                )
            ],
            **extra,
        ),
    )


@pytest.mark.unit
class TestAppSpaceEmission:
    def test_omits_space_when_unset(self) -> None:
        _, _, apps_block = _build_app_block(_config(), "dao_ai.yaml")
        (app_def,) = apps_block.values()
        assert "space" not in app_def

    def test_emits_space_when_set(self) -> None:
        _, _, apps_block = _build_app_block(
            _config(space="retail-builders"), "dao_ai.yaml"
        )
        (app_def,) = apps_block.values()
        assert app_def["space"] == "retail-builders"
