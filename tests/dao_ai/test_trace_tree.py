"""Tests for ``build_trace_tree`` — converts MLflow's flat span list into the
nested waterfall shape the Console Timeline view consumes.

The builder is duck-typed over span objects (it reads ``span_id``,
``parent_id``, ``name``, ``start_time_ns``, ``end_time_ns``, ``span_type``,
``status``, ``inputs``, ``outputs``, ``attributes``, ``events``) so it can be
unit-tested with lightweight fakes and does not require a live MLflow trace.
"""

from types import SimpleNamespace

import pytest

from dao_ai.apps.traces import build_trace_tree

MS = 1_000_000  # nanoseconds per millisecond


def _span(**kw) -> SimpleNamespace:
    defaults = dict(
        span_id="s",
        parent_id=None,
        name="span",
        start_time_ns=0,
        end_time_ns=MS,
        span_type="CHAIN",
        status="OK",
        inputs={},
        outputs={},
        attributes={},
        events=[],
    )
    defaults.update(kw)
    return SimpleNamespace(**defaults)


class TestBuildTraceTree:
    @pytest.mark.unit
    def test_nests_children_and_computes_offsets_and_durations(self) -> None:
        root = _span(
            span_id="s0",
            parent_id=None,
            name="agent",
            start_time_ns=1 * MS,
            end_time_ns=6 * MS,  # 5ms
        )
        child = _span(
            span_id="s1",
            parent_id="s0",
            name="search_docs",
            span_type="TOOL",
            start_time_ns=2 * MS,  # +1ms from root
            end_time_ns=4 * MS,  # 2ms
            inputs={"query": "hi"},
            outputs={"docs": 2},
        )
        tree = build_trace_tree([child, root], trace_id="tr-1")

        assert tree["trace_id"] == "tr-1"
        assert tree["root_span_id"] == "s0"
        assert tree["duration_ms"] == 5.0

        assert len(tree["spans"]) == 1
        node = tree["spans"][0]
        assert node["span_id"] == "s0"
        assert node["start_offset_ms"] == 0.0
        assert node["duration_ms"] == 5.0
        assert len(node["children"]) == 1

        c = node["children"][0]
        assert c["span_id"] == "s1"
        assert c["name"] == "search_docs"
        assert c["span_type"] == "TOOL"
        assert c["start_offset_ms"] == 1.0
        assert c["duration_ms"] == 2.0
        assert c["inputs"] == {"query": "hi"}
        assert c["outputs"] == {"docs": 2}

    @pytest.mark.unit
    def test_span_with_missing_parent_is_treated_as_top_level(self) -> None:
        orphan = _span(
            span_id="s2", parent_id="ghost", start_time_ns=1 * MS, end_time_ns=2 * MS
        )
        tree = build_trace_tree([orphan], trace_id="tr-2")
        assert len(tree["spans"]) == 1
        assert tree["spans"][0]["span_id"] == "s2"

    @pytest.mark.unit
    def test_events_mapped_with_offset(self) -> None:
        event = SimpleNamespace(
            name="mcp.progress",
            timestamp=3 * MS,
            attributes={"progress": 0.5},
        )
        root = _span(
            span_id="s0", start_time_ns=1 * MS, end_time_ns=6 * MS, events=[event]
        )
        tree = build_trace_tree([root], trace_id="tr-3")
        events = tree["spans"][0]["events"]
        assert len(events) == 1
        assert events[0]["name"] == "mcp.progress"
        assert events[0]["timestamp_ms"] == 2.0  # 3ms - 1ms root start
        assert events[0]["attributes"] == {"progress": 0.5}

    @pytest.mark.unit
    def test_bearer_and_headers_redacted_from_inputs(self) -> None:
        root = _span(
            span_id="s0",
            start_time_ns=0,
            end_time_ns=MS,
            inputs={
                "query": "hi",
                "authorization": "Bearer secret",
                "headers": {"x": 1},
                "x-forwarded-access-token": "tok",
            },
        )
        tree = build_trace_tree([root], trace_id="tr-4")
        inputs = tree["spans"][0]["inputs"]
        assert inputs == {"query": "hi"}
        assert "authorization" not in inputs
        assert "headers" not in inputs
        assert "x-forwarded-access-token" not in inputs

    @pytest.mark.unit
    def test_empty_spans_returns_empty_tree(self) -> None:
        tree = build_trace_tree([], trace_id="tr-5")
        assert tree["trace_id"] == "tr-5"
        assert tree["spans"] == []
        assert tree["root_span_id"] is None


class TestTraceRetrievalResilience:
    """The request path must never raise — a permission-denied / corrupted /
    unreadable trace degrades to None so the route 404s (not 500s)."""

    @pytest.mark.unit
    def test_get_trace_tree_returns_none_on_permission_error(
        self, monkeypatch
    ) -> None:
        import mlflow

        from dao_ai.apps import traces

        def _boom(_trace_id):
            raise RuntimeError(
                "PermissionDenied: SP is not authorized to use this SQL Endpoint"
            )

        monkeypatch.setattr(mlflow, "get_trace", _boom)
        # Must not raise, and must give up immediately (non-NOT_FOUND is terminal).
        assert traces.get_trace_tree("trace:/c.s.123/abc", timeout_seconds=0.1) is None

    @pytest.mark.unit
    def test_wait_for_trace_false_on_terminal_error(self, monkeypatch) -> None:
        import mlflow

        from dao_ai.apps import traces

        monkeypatch.setattr(
            mlflow, "get_trace", lambda _t: (_ for _ in ()).throw(ValueError("boom"))
        )
        assert traces.wait_for_trace("tr-x", timeout_seconds=0.1) is False

    @pytest.mark.unit
    def test_get_trace_tree_success(self, monkeypatch) -> None:
        import mlflow

        from dao_ai.apps import traces

        span = _span(span_id="root", parent_id=None, name="agent")
        trace = SimpleNamespace(data=SimpleNamespace(spans=[span]))
        monkeypatch.setattr(mlflow, "get_trace", lambda _t: trace)
        tree = traces.get_trace_tree("tr-ok", timeout_seconds=1.0)
        assert tree is not None
        assert tree["spans"][0]["name"] == "agent"
