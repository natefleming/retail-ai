"""F6b: CompositeVariableModel must WARN when it falls back to default_value.

An env/scope-backed value silently resolving to its hardcoded ``default_value``
(e.g. in a ``--direct`` deploy where the container's env/scope isn't populated)
is a silent-misconfig trap — it's how a stale genie ``space_id`` sailed through.
The warning makes that fallback visible; a resolved source or an absent default
must stay quiet.
"""

from __future__ import annotations

import pytest
from loguru import logger

from dao_ai.config import (
    CompositeVariableModel,
    EnvironmentVariableModel,
)

_MISSING_ENV = "DAO_AI_TEST_DEFINITELY_UNSET_VAR"


def _capture(fn) -> list[str]:
    """Run ``fn`` with a temporary WARNING sink; return captured messages."""
    msgs: list[str] = []
    sink_id = logger.add(lambda m: msgs.append(m), level="WARNING")
    try:
        fn()
    finally:
        logger.remove(sink_id)
    return msgs


@pytest.mark.unit
class TestCompositeDefaultWarning:
    def test_warns_when_falling_back_to_default(self, monkeypatch) -> None:
        monkeypatch.delenv(_MISSING_ENV, raising=False)
        comp = CompositeVariableModel(
            options=[EnvironmentVariableModel(env=_MISSING_ENV)],
            default_value="01fSTALE",
        )
        msgs = _capture(lambda: comp.as_value())
        assert comp.as_value() == "01fSTALE"
        assert any("default_value" in m for m in msgs), msgs

    def test_no_warn_when_source_resolves(self, monkeypatch) -> None:
        monkeypatch.setenv("DAO_AI_TEST_SET_VAR", "live-value")
        comp = CompositeVariableModel(
            options=[EnvironmentVariableModel(env="DAO_AI_TEST_SET_VAR")],
            default_value="fallback",
        )
        msgs = _capture(lambda: comp.as_value())
        assert comp.as_value() == "live-value"
        assert not msgs, f"unexpected warnings: {msgs}"

    def test_no_warn_when_no_default(self, monkeypatch) -> None:
        monkeypatch.delenv(_MISSING_ENV, raising=False)
        comp = CompositeVariableModel(
            options=[EnvironmentVariableModel(env=_MISSING_ENV)],
        )
        msgs = _capture(lambda: comp.as_value())
        assert comp.as_value() is None
        assert not msgs, f"unexpected warnings: {msgs}"
