"""Tests for search_session_traces — discovers a thread's MLflow traces via the
mlflow.trace.session link so a reloaded conversation can rebuild its Timeline."""

from types import SimpleNamespace

import pytest

from dao_ai.apps import traces


def _info(trace_id: str, request: str, ts: str) -> SimpleNamespace:
    return SimpleNamespace(
        trace_id=trace_id,
        request_preview=request,
        request_time=ts,
        trace_metadata={"mlflow.trace.session": "sess-1"},
    )


class _FakeClient:
    """search_traces raises for all but the matching filter (models the
    candidate-filter fallthrough)."""

    def __init__(self, working_filter: str, infos: list) -> None:
        self._working = working_filter
        self._infos = infos
        self.tried: list[str] = []

    def search_traces(
        self, *, experiment_ids, filter_string, max_results, order_by, include_spans=True
    ):
        self.tried.append(filter_string)
        if self._working in filter_string:
            return self._infos
        raise ValueError(f"unsupported filter: {filter_string}")


class TestSearchSessionTraces:
    @pytest.mark.unit
    def test_returns_refs_ordered(self, monkeypatch) -> None:
        infos = [_info("tr-1", "hello", "2026-01-01"), _info("tr-2", "bye", "2026-01-02")]
        client = _FakeClient("metadata.`mlflow.trace.session`", infos)
        monkeypatch.setattr(traces, "_active_experiment_id", lambda: "999")
        monkeypatch.setattr("mlflow.MlflowClient", lambda: client)
        refs = traces.search_session_traces("sess-1")
        assert refs == [
            {"trace_id": "tr-1", "request_preview": "hello", "request_time": "2026-01-01"},
            {"trace_id": "tr-2", "request_preview": "bye", "request_time": "2026-01-02"},
        ]

    @pytest.mark.unit
    def test_falls_through_to_working_filter(self, monkeypatch) -> None:
        # First candidate filter is rejected; a later one works.
        client = _FakeClient("tags.`mlflow.trace.session`", [_info("tr-9", "q", "2026")])
        monkeypatch.setattr(traces, "_active_experiment_id", lambda: "999")
        monkeypatch.setattr("mlflow.MlflowClient", lambda: client)
        refs = traces.search_session_traces("sess-1")
        assert [r["trace_id"] for r in refs] == ["tr-9"]
        assert len(client.tried) >= 2  # tried earlier candidates first

    @pytest.mark.unit
    def test_empty_when_no_experiment(self, monkeypatch) -> None:
        monkeypatch.setattr(traces, "_active_experiment_id", lambda: None)
        assert traces.search_session_traces("sess-1") == []

    @pytest.mark.unit
    def test_empty_when_all_filters_fail(self, monkeypatch) -> None:
        client = _FakeClient("NONEXISTENT", [])
        monkeypatch.setattr(traces, "_active_experiment_id", lambda: "999")
        monkeypatch.setattr("mlflow.MlflowClient", lambda: client)
        assert traces.search_session_traces("sess-1") == []

    @pytest.mark.unit
    def test_empty_session_id(self) -> None:
        assert traces.search_session_traces("") == []
