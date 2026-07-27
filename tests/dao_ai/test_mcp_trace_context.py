"""Unit tests for W3C trace-context propagation over MCP ``_meta`` (SEP-414).

Covers:
1. ``build_trace_context_meta`` mints a well-formed ``traceparent`` from the
   active MLflow span and returns ``{}`` when no span is active.
2. ``merge_trace_context_meta`` is non-destructive — caller-supplied keys
   (conversation_id, progressToken) always win, and the ``meta=None`` call
   shape is preserved when there is nothing to add.
3. ``extract_trace_context_meta`` stamps inbound context onto the active span
   and degrades to a no-op for absent / malformed meta.

No Databricks credentials required — MLflow span is mocked.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from dao_ai.tools.mcp_trace_context import (
    build_trace_context_meta,
    merge_trace_context_meta,
    stamp_trace_context,
    trace_context_tags,
)

_TRACE_HEX = "41a20ee334efb76f29492d9f4d4a03a5"  # 32 hex
_SPAN_HEX = "9286aeaf7fe8b38a"  # 16 hex


def _span(trace_id: str | None, span_id: str | None) -> MagicMock:
    span = MagicMock()
    span.trace_id = trace_id
    span.span_id = span_id
    return span


class TestBuildTraceContextMeta:
    def test_empty_when_no_active_span(self) -> None:
        with patch(
            "dao_ai.tools.mcp_trace_context.mlflow.get_current_active_span",
            return_value=None,
        ):
            assert build_trace_context_meta() == {}

    def test_wellformed_traceparent_from_tr_prefixed_id(self) -> None:
        span = _span(f"tr-{_TRACE_HEX}", _SPAN_HEX)
        with patch(
            "dao_ai.tools.mcp_trace_context.mlflow.get_current_active_span",
            return_value=span,
        ):
            meta = build_trace_context_meta()
        assert meta["traceparent"] == f"00-{_TRACE_HEX}-{_SPAN_HEX}-01"
        assert meta["baggage"] == f"mlflow.trace_id=tr-{_TRACE_HEX}"

    def test_traceparent_from_uc_uri_trace_id(self) -> None:
        # UC-backed trace ids arrive as a trace:/cat.schema.prefix/<hex> URI;
        # the OTel-native hex is the final hex run.
        span = _span(f"trace:/cat.schema.my_prefix/{_TRACE_HEX}", _SPAN_HEX)
        with patch(
            "dao_ai.tools.mcp_trace_context.mlflow.get_current_active_span",
            return_value=span,
        ):
            meta = build_trace_context_meta()
        assert meta["traceparent"] == f"00-{_TRACE_HEX}-{_SPAN_HEX}-01"

    def test_baggage_only_when_hex_wrong_width(self) -> None:
        # A short / non-standard id can't form a valid traceparent; carry the
        # mlflow trace id in baggage rather than emit a malformed header.
        span = _span("tr-deadbeef", "cafe")
        with patch(
            "dao_ai.tools.mcp_trace_context.mlflow.get_current_active_span",
            return_value=span,
        ):
            meta = build_trace_context_meta()
        assert "traceparent" not in meta
        assert meta["baggage"] == "mlflow.trace_id=tr-deadbeef"

    def test_never_raises_on_span_access_error(self) -> None:
        with patch(
            "dao_ai.tools.mcp_trace_context.mlflow.get_current_active_span",
            side_effect=RuntimeError("boom"),
        ):
            assert build_trace_context_meta() == {}


class TestMergeTraceContextMeta:
    def test_none_preserved_when_no_span(self) -> None:
        with patch(
            "dao_ai.tools.mcp_trace_context.mlflow.get_current_active_span",
            return_value=None,
        ):
            assert merge_trace_context_meta(None) is None

    def test_adds_trace_keys_to_existing_meta(self) -> None:
        span = _span(f"tr-{_TRACE_HEX}", _SPAN_HEX)
        with patch(
            "dao_ai.tools.mcp_trace_context.mlflow.get_current_active_span",
            return_value=span,
        ):
            merged = merge_trace_context_meta({"conversation_id": "abc"})
        assert merged["conversation_id"] == "abc"
        assert merged["traceparent"] == f"00-{_TRACE_HEX}-{_SPAN_HEX}-01"

    def test_caller_keys_win(self) -> None:
        span = _span(f"tr-{_TRACE_HEX}", _SPAN_HEX)
        with patch(
            "dao_ai.tools.mcp_trace_context.mlflow.get_current_active_span",
            return_value=span,
        ):
            merged = merge_trace_context_meta(
                {"traceparent": "caller-wins", "progressToken": "tok"}
            )
        # Caller-supplied keys are never overwritten.
        assert merged["traceparent"] == "caller-wins"
        assert merged["progressToken"] == "tok"
        assert merged["baggage"] == f"mlflow.trace_id=tr-{_TRACE_HEX}"


class TestTraceContextTags:
    def test_extracts_present_keys(self) -> None:
        meta = SimpleNamespace(
            traceparent=f"00-{_TRACE_HEX}-{_SPAN_HEX}-01",
            tracestate=None,
            baggage="mlflow.trace_id=tr-x",
        )
        tags = trace_context_tags(meta)
        assert tags == {
            "mcp.trace_context.traceparent": f"00-{_TRACE_HEX}-{_SPAN_HEX}-01",
            "mcp.trace_context.baggage": "mlflow.trace_id=tr-x",
        }

    def test_dict_meta_supported(self) -> None:
        assert trace_context_tags({"traceparent": "00-x-y-01"}) == {
            "mcp.trace_context.traceparent": "00-x-y-01"
        }

    def test_empty_when_meta_none(self) -> None:
        assert trace_context_tags(None) == {}

    def test_empty_when_no_trace_keys(self) -> None:
        assert trace_context_tags({"unrelated": "x"}) == {}


class TestStampTraceContext:
    def test_stamps_attributes_on_span(self) -> None:
        span = MagicMock()
        stamp_trace_context(span, {"traceparent": "00-x-y-01", "baggage": "b"})
        keys = {c.args[0] for c in span.set_attribute.call_args_list}
        assert "mcp.trace_context.traceparent" in keys
        assert "mcp.trace_context.baggage" in keys

    def test_noop_when_no_context(self) -> None:
        span = MagicMock()
        stamp_trace_context(span, None)
        span.set_attribute.assert_not_called()

    def test_never_raises_on_span_error(self) -> None:
        span = MagicMock()
        span.set_attribute.side_effect = RuntimeError("boom")
        # Must not raise.
        stamp_trace_context(span, {"traceparent": "00-x-y-01"})
