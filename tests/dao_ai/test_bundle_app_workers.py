"""Bundle emission tests for ``AppModel.workers`` (backend worker count).

Locks in the contract that ``_build_app_block`` emits the ``DAO_AI_APP_WORKERS``
env var (which ``start_app.py`` forwards to the backend as ``--workers``) ONLY
when ``app.workers`` is set. The field defaults to ``None``: unset means the
backend auto-sizes to the container's available CPUs at runtime, so no env var
is emitted and the count is decided in-process (gunicorn preload_app for
workers>1). An explicit integer overrides the auto-sizing and IS emitted.
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


def _config(*, workers: int | None = None) -> AppConfig:
    extra: dict = {}
    if workers is not None:
        extra["workers"] = workers
    return AppConfig(
        app=AppModel(
            name="dao-ai-workers-test",
            description="test agent",
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


def _workers_env(apps_block: dict) -> str | None:
    (app_def,) = apps_block.values()
    for e in app_def.get("config", {}).get("env", []):
        if e.get("name") == "DAO_AI_APP_WORKERS":
            return e.get("value")
    return None


@pytest.mark.unit
class TestAppWorkersEmission:
    def test_default_workers_is_none(self) -> None:
        # Unset -> auto-size at runtime (no fixed default baked into config).
        assert _config().app.workers is None

    def test_omits_workers_env_when_unset(self) -> None:
        # No explicit count -> no DAO_AI_APP_WORKERS env; the backend auto-sizes.
        _, _, apps_block = _build_app_block(_config(), "dao_ai.yaml")
        assert _workers_env(apps_block) is None

    def test_emits_explicit_workers_env(self) -> None:
        _, _, apps_block = _build_app_block(_config(workers=4), "dao_ai.yaml")
        assert _workers_env(apps_block) == "4"

    def test_workers_must_be_positive(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _config(workers=0)
