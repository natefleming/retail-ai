"""Regression tests for the MLflow-autolog warning suppression path.

Background: ``mlflow.langchain.autolog`` installs its tracer via a
monkey-patch of ``BaseCallbackManager.__init__``. LangChain's memory-
extraction chain (``langmem.knowledge.extraction.MemoryStoreManager``)
uses ``get_executor_for_config(config)`` to spawn a nested thread pool
for parallel ``store.search`` calls, and each new callback manager on
those sub-threads gets a fresh ``MlflowLangchainTracer`` instance whose
per-instance ``_run_span_mapping`` cannot see the parent tracer's
run_ids. That triggers two families of warnings on every request:

- ``mlflow.utils.autologging_utils``: ``Encountered unexpected error
  during autologging: Span for run_id X not found.``
- ``mlflow.entities.span``: ``Failed to end span X: .``

The warnings are harmless (memory writes still succeed, main-pipeline
traces still land). They cannot be silenced at the RunnableConfig level
because autolog injects at the callback-manager-init level. And they
cannot be silenced by ``mlflow.utils.autologging_utils.disable_autologging``
because that flips a process-global flag that races with concurrent
foreground requests. So we filter them at the log-record level via
``suppress_autolog_context_warnings`` in ``dao_ai/logging.py``.

These tests pin that filter behavior.
"""

from __future__ import annotations

import logging

import pytest

from dao_ai.logging import suppress_autolog_context_warnings


@pytest.fixture
def _fresh_loggers() -> None:
    """Reset filters on the two loggers before/after each test."""

    def _clear() -> None:
        for name in ("mlflow.utils.autologging_utils", "mlflow.entities.span"):
            log = logging.getLogger(name)
            for f in list(log.filters):
                log.removeFilter(f)

    _clear()
    yield
    _clear()


def _emit(logger_name: str, message: str) -> logging.LogRecord:
    """Build a synthetic warning record — same shape MLflow emits."""
    return logging.LogRecord(
        name=logger_name,
        level=logging.WARNING,
        pathname=__file__,
        lineno=0,
        msg=message,
        args=None,
        exc_info=None,
    )


def _passes_all_filters(logger_name: str, record: logging.LogRecord) -> bool:
    log = logging.getLogger(logger_name)
    return all(f.filter(record) for f in log.filters)


@pytest.mark.unit
def test_filter_drops_span_for_run_id_not_found(_fresh_loggers: None) -> None:
    """Memory-extraction cross-thread callback warning is silenced."""
    suppress_autolog_context_warnings()
    record = _emit(
        "mlflow.utils.autologging_utils",
        "Encountered unexpected error during autologging: "
        "Span for run_id 019f2d1c-9b9f-7720-a880-6036114b135e not found.",
    )
    assert _passes_all_filters("mlflow.utils.autologging_utils", record) is False


@pytest.mark.unit
def test_filter_drops_context_var_warning(_fresh_loggers: None) -> None:
    """Pre-existing nest_asyncio ContextVar warning is silenced (backcompat)."""
    suppress_autolog_context_warnings()
    record = _emit(
        "mlflow.utils.autologging_utils",
        "Token <ContextVarToken ...> was created in a different Context "
        "and cannot be used to reset it.",
    )
    assert _passes_all_filters("mlflow.utils.autologging_utils", record) is False


@pytest.mark.unit
def test_filter_drops_failed_to_end_span(_fresh_loggers: None) -> None:
    """Sibling ``Failed to end span`` warning is silenced."""
    suppress_autolog_context_warnings()
    record = _emit(
        "mlflow.entities.span",
        "Failed to end span d3a1418cf6624aa8: . "
        "For full traceback, set logging level to debug.",
    )
    assert _passes_all_filters("mlflow.entities.span", record) is False


@pytest.mark.unit
def test_filter_lets_unrelated_autolog_warnings_through(_fresh_loggers: None) -> None:
    """Real, actionable autolog warnings still surface — filter is targeted."""
    suppress_autolog_context_warnings()
    record = _emit(
        "mlflow.utils.autologging_utils",
        "MLflow autologging encountered a warning: model registry unavailable.",
    )
    assert _passes_all_filters("mlflow.utils.autologging_utils", record) is True


@pytest.mark.unit
def test_filter_lets_unrelated_span_warnings_through(_fresh_loggers: None) -> None:
    """Non-'Failed to end span' entries.span warnings still surface."""
    suppress_autolog_context_warnings()
    record = _emit("mlflow.entities.span", "Span exceeded max attribute count.")
    assert _passes_all_filters("mlflow.entities.span", record) is True


@pytest.mark.unit
def test_filter_matches_partial_span_for_run_id_variants(_fresh_loggers: None) -> None:
    """Guard against wording drift — match any 'Span for run_id ... not found'."""
    suppress_autolog_context_warnings()
    for msg in [
        "Span for run_id abc not found.",
        "Encountered unexpected error during autologging: Span for run_id X not found.",
        "Span for run_id 019f... not found (retrying).",
    ]:
        record = _emit("mlflow.utils.autologging_utils", msg)
        assert _passes_all_filters("mlflow.utils.autologging_utils", record) is False, msg
