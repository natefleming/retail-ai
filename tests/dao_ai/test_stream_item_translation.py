"""Tests for ``_stream_item_for_envelope`` — translation of collector envelopes
into OpenAI Responses ``item`` payloads for ``response.output_item.added`` SSE
events.

Tool lifecycle reuses the MLflow ResponsesAgent item taxonomy
(``function_call`` / ``function_call_output``); MCP notifications keep their
existing ``custom_tool_call`` shape for backward compatibility.
"""

import json

import pytest

from dao_ai.models import _stream_item_for_envelope


class TestToolLifecycleItems:
    @pytest.mark.unit
    def test_tool_start_becomes_function_call(self) -> None:
        env = {
            "channel": "dao_ai.tool.start",
            "call_id": "run-123",
            "name": "search_docs",
            "arguments": {"query": "hello"},
            "started_at": "2026-09-02T10:00:00+00:00",
        }
        item = _stream_item_for_envelope(env, item_id="msg_1", seq=1)
        assert item["type"] == "function_call"
        assert item["call_id"] == "run-123"
        assert item["name"] == "search_docs"
        assert json.loads(item["arguments"]) == {"query": "hello"}
        assert item["status"] == "in_progress"
        assert item["started_at"] == "2026-09-02T10:00:00+00:00"

    @pytest.mark.unit
    def test_tool_end_becomes_function_call_output_with_duration(self) -> None:
        env = {
            "channel": "dao_ai.tool.end",
            "call_id": "run-123",
            "duration_ms": 42.5,
            "result_summary": "found 3 documents",
        }
        item = _stream_item_for_envelope(env, item_id="msg_1", seq=2)
        assert item["type"] == "function_call_output"
        assert item["call_id"] == "run-123"
        assert item["output"] == "found 3 documents"
        assert item["status"] == "completed"
        assert item["duration_ms"] == 42.5

    @pytest.mark.unit
    def test_tool_error_becomes_function_call_output_error(self) -> None:
        env = {
            "channel": "dao_ai.tool.error",
            "call_id": "run-9",
            "duration_ms": 5.0,
            "error": "boom",
        }
        item = _stream_item_for_envelope(env, item_id="msg_1", seq=3)
        assert item["type"] == "function_call_output"
        assert item["status"] == "error"
        assert "boom" in item["output"]
        assert item["duration_ms"] == 5.0


class TestMcpNotificationItem:
    @pytest.mark.unit
    def test_mcp_envelope_keeps_custom_tool_call_shape(self) -> None:
        env = {
            "channel": "mcp.progress",
            "server_name": "filesystem",
            "progress": 0.5,
        }
        item = _stream_item_for_envelope(env, item_id="msg_1", seq=7)
        assert item["type"] == "custom_tool_call"
        assert item["name"] == "mcp.progress"
        assert item["input"] == env
        assert item["id"] == "mcp_filesystem_msg_1_7"
        assert item["status"] == "in_progress"
