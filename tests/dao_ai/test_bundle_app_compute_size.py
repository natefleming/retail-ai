"""Compute-size coercion + bundle emission tests for ``AppModel.workload_size``.

``workload_size`` is shared by both deployment targets. Two helpers coerce it:

* ``apps_compute_size()`` maps to the Databricks Apps ``compute_size`` domain.
  Apps has no Small tier and its platform default is MEDIUM, so Small/Medium
  return ``None`` (leave the default, never resize existing apps); Large/XLarge
  map to ``LARGE``/``XLARGE``.
* ``serving_workload_size()`` clamps to the Model Serving domain, which has no
  XLarge tier: XLarge -> Large, everything else passes through.

``_build_app_block`` emits ``compute_size`` in the app_def ONLY when
``apps_compute_size()`` is truthy (i.e. Large/XLarge).
"""

from __future__ import annotations

import pytest

from dao_ai.apps.bundle import _build_app_block
from dao_ai.config import (
    AgentModel,
    AppConfig,
    AppModel,
    InferenceEndpointModel,
)


def _config(*, workload_size: str = "Small") -> AppConfig:
    return AppConfig(
        app=AppModel(
            name="dao-ai-compute-size-test",
            description="test agent",
            enable_chat_proxy=False,
            workload_size=workload_size,
            agents=[
                AgentModel(
                    name="greeter",
                    description="test agent",
                    model=InferenceEndpointModel(name="databricks-gpt-5-4-mini"),
                )
            ],
        ),
    )


def _compute_size(apps_block: dict) -> str | None:
    (app_def,) = apps_block.values()
    return app_def.get("compute_size")


@pytest.mark.unit
class TestAppsComputeSizeCoercion:
    @pytest.mark.parametrize(
        "workload_size, expected",
        [
            ("Small", None),
            ("Medium", None),
            ("Large", "LARGE"),
            ("XLarge", "XLARGE"),
        ],
    )
    def test_apps_compute_size(self, workload_size: str, expected: str | None) -> None:
        assert _config(workload_size=workload_size).app.apps_compute_size() == expected

    @pytest.mark.parametrize(
        "workload_size, expected",
        [
            ("Small", "Small"),
            ("Medium", "Medium"),
            ("Large", "Large"),
            ("XLarge", "Large"),  # clamped: Model Serving has no XLarge tier
        ],
    )
    def test_serving_workload_size(
        self, workload_size: str, expected: str
    ) -> None:
        assert (
            _config(workload_size=workload_size).app.serving_workload_size()
            == expected
        )


@pytest.mark.unit
class TestComputeSizeEmission:
    def test_omits_compute_size_for_small(self) -> None:
        # Small -> platform default (MEDIUM); no key emitted, existing apps
        # are never resized on redeploy.
        _, _, apps_block = _build_app_block(_config(workload_size="Small"), "dao_ai.yaml")
        assert _compute_size(apps_block) is None

    def test_omits_compute_size_for_medium(self) -> None:
        _, _, apps_block = _build_app_block(
            _config(workload_size="Medium"), "dao_ai.yaml"
        )
        assert _compute_size(apps_block) is None

    def test_emits_large(self) -> None:
        _, _, apps_block = _build_app_block(_config(workload_size="Large"), "dao_ai.yaml")
        assert _compute_size(apps_block) == "LARGE"

    def test_emits_xlarge(self) -> None:
        _, _, apps_block = _build_app_block(
            _config(workload_size="XLarge"), "dao_ai.yaml"
        )
        assert _compute_size(apps_block) == "XLARGE"
