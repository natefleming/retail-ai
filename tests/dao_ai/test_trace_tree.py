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

    @pytest.mark.unit
    def test_wait_polls_when_get_trace_returns_none(self, monkeypatch) -> None:
        # mlflow.get_trace returns None (not raises) for a not-yet-propagated
        # trace, so the poll must key off the return value, not an exception.
        import mlflow

        from dao_ai.apps import traces

        calls = {"n": 0}

        def _none(_trace_id):
            calls["n"] += 1
            return None

        monkeypatch.setattr(mlflow, "get_trace", _none)
        assert traces.wait_for_trace("tr-x", timeout_seconds=0.1) is False
        assert calls["n"] >= 2  # actually polled, not a single immediate True


class TestBuildTreeRobustness:
    @pytest.mark.unit
    def test_missing_start_does_not_skew_baseline(self) -> None:
        base = 1_700_000_000_000_000_000  # epoch-scale ns
        good = _span(span_id="a", parent_id=None, start_time_ns=base, end_time_ns=base + 10 * MS)
        missing = _span(span_id="b", parent_id="a", start_time_ns=0, end_time_ns=base + 8 * MS)
        tree = build_trace_tree([good, missing], trace_id="tr")
        assert tree["spans"][0]["start_offset_ms"] == 0.0
        assert tree["duration_ms"] < 1000  # not an astronomical epoch offset

    @pytest.mark.unit
    def test_self_parenting_span_is_root_not_infinite(self) -> None:
        loopy = _span(span_id="x", parent_id="x", name="loopy")
        tree = build_trace_tree([loopy], trace_id="tr")  # must not recurse
        assert tree["spans"][0]["span_id"] == "x"
        assert tree["spans"][0]["children"] == []


class TestTraceUiUrl:
    @pytest.mark.unit
    def test_prefers_active_experiment(self, monkeypatch) -> None:
        # The app's active experiment is authoritative regardless of the UC
        # location's table_prefix (which may be a custom, non-id string).
        monkeypatch.setenv("DATABRICKS_HOST", "https://ws.cloud.databricks.com")
        from dao_ai.apps import traces

        monkeypatch.setattr(traces, "_active_experiment_id", lambda: "777")
        url = traces.build_trace_ui_url("trace:/cat.sch.sales_genie/abc123")
        assert url == (
            "https://ws.cloud.databricks.com/ml/experiments/777/traces/tr-abc123"
        )

    @pytest.mark.unit
    def test_uc_falls_back_to_numeric_embedded_id(self, monkeypatch) -> None:
        # When the active experiment can't be resolved, use the location's
        # trailing segment only if it's numeric (the default = experiment id).
        monkeypatch.setenv("DATABRICKS_HOST", "https://ws.cloud.databricks.com")
        from dao_ai.apps import traces

        monkeypatch.setattr(traces, "_active_experiment_id", lambda: None)
        url = traces.build_trace_ui_url("trace:/cat.sch.540443496685391/abc123")
        assert url == (
            "https://ws.cloud.databricks.com/ml/experiments/540443496685391/traces/tr-abc123"
        )

    @pytest.mark.unit
    def test_uc_custom_prefix_no_active_experiment_yields_none(self, monkeypatch) -> None:
        # Custom (non-numeric) table_prefix + no active experiment → no bad link.
        monkeypatch.setenv("DATABRICKS_HOST", "https://ws.cloud.databricks.com")
        from dao_ai.apps import traces

        monkeypatch.setattr(traces, "_active_experiment_id", lambda: None)
        assert traces.build_trace_ui_url("trace:/cat.sch.sales_genie/abc123") is None

    @pytest.mark.unit
    def test_control_plane_id_uses_active_experiment(self, monkeypatch) -> None:
        monkeypatch.setenv("DATABRICKS_HOST", "ws.cloud.databricks.com")
        from dao_ai.apps import traces

        monkeypatch.setattr(traces, "_active_experiment_id", lambda: "999")
        url = traces.build_trace_ui_url("tr-deadbeef")
        assert url == "https://ws.cloud.databricks.com/ml/experiments/999/traces/tr-deadbeef"

    @pytest.mark.unit
    def test_no_host_yields_none(self, monkeypatch) -> None:
        monkeypatch.delenv("DATABRICKS_HOST", raising=False)
        from dao_ai.apps import traces

        monkeypatch.setattr(traces, "_workspace_host", lambda: None)
        assert traces.build_trace_ui_url("tr-x") is None


class TestBuildTraceTreeRobustness:
    """Regression tests for /code-review findings on malformed span data."""

    @pytest.mark.unit
    def test_falsy_end_time_ns_does_not_yield_negative_duration(self) -> None:
        # Every span reports a falsy end_time_ns but real starts — duration must
        # clamp to >= 0 instead of going negative (which blows bar widths off-scale).
        spans = [
            _span(span_id="a", parent_id=None, start_time_ns=5 * MS, end_time_ns=0),
            _span(span_id="b", parent_id="a", start_time_ns=6 * MS, end_time_ns=None),
        ]
        tree = build_trace_tree(spans, trace_id="tr-x")
        assert tree["duration_ms"] >= 0

    @pytest.mark.unit
    def test_two_node_parent_cycle_keeps_both_spans(self) -> None:
        # A <-> B mutual-parent cycle: neither may be dropped from the tree, and
        # the builder must not recurse forever.
        spans = [
            _span(span_id="a", parent_id="b", start_time_ns=1 * MS, end_time_ns=2 * MS),
            _span(span_id="b", parent_id="a", start_time_ns=1 * MS, end_time_ns=2 * MS),
        ]
        tree = build_trace_tree(spans, trace_id="tr-cycle")

        seen: set[str] = set()

        def _walk(nodes: list) -> None:
            for n in nodes:
                seen.add(n["span_id"])
                _walk(n["children"])

        _walk(tree["spans"])
        assert seen == {"a", "b"}  # no span lost to the cycle
