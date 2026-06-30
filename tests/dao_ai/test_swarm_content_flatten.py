"""Tests pinning the inter-agent content-flatten contract.

These tests verify the assumptions behind two fixes that landed in
``feature/surface-to-user-v3``:

1. ``_flatten_message_content`` handles all three content shapes
   Databricks Foundation Model APIs return:
   - plain ``str`` (Llama, basic mode)
   - JSON-encoded ``str`` of a content-block array (gpt-oss-120b via
     ChatDatabricks chat-completions)
   - Python ``list[dict]`` (Claude / Responses-API native)

2. The handoff tool's ``Command(update={"messages": ...})`` writes the
   *flattened* trigger AIMessage to parent state — not the raw
   structured-content version. Without this, downstream agents'
   chat-completions LLM calls fail with:

     Bad request: field 'messages.content' expects 'string' but got 'array'

   The flatten must happen inside ``create_handoff_tool`` because the
   handoff Command bypasses ``extract_agent_response`` (it emits with
   ``graph=Command.PARENT`` which writes directly to parent state).
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command

from dao_ai.orchestration.core import (
    _flatten_message_content,
    create_handoff_tool,
)


# ---------------------------------------------------------------------------
# _flatten_message_content
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_flatten_plain_string_passthrough() -> None:
    """A plain text content string must pass through unchanged."""
    assert _flatten_message_content("hello world") == "hello world"
    assert _flatten_message_content("") == ""
    # Non-array strings that happen to start with [ but don't parse as JSON
    # also pass through unchanged.
    assert _flatten_message_content("[ this is not json ]") == "[ this is not json ]"


@pytest.mark.unit
def test_flatten_python_list_of_content_blocks() -> None:
    """Python list of content blocks (Claude / Responses-API native) joins
    text-type blocks and drops everything else."""
    content: list[dict[str, Any]] = [
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "thinking"}]},
        {"type": "text", "text": "INTENT: discovery"},
    ]
    assert _flatten_message_content(content) == "INTENT: discovery"


@pytest.mark.unit
def test_flatten_python_list_multiple_text_blocks() -> None:
    """Multiple text blocks join in order."""
    content: list[dict[str, Any]] = [
        {"type": "text", "text": "Hello "},
        {"type": "reasoning", "summary": "ignored"},
        {"type": "text", "text": "world"},
    ]
    assert _flatten_message_content(content) == "Hello world"


@pytest.mark.unit
def test_flatten_json_encoded_string_of_content_blocks() -> None:
    """ChatDatabricks's gpt-oss-120b returns ``str`` whose value is the
    JSON-encoded array of content blocks. Confirmed via:

        m = ChatDatabricks(endpoint='databricks-gpt-oss-120b')
        out = m.invoke(...)
        # type(out.content) is str
        # out.content == '[{"type":"reasoning",...},{"type":"text",...}]'

    The flatten must detect the ``[`` prefix, json.loads, and recurse."""
    json_str: str = (
        '[{"type": "reasoning", "summary": [{"type":"summary_text","text":"x"}]}, '
        '{"type": "text", "text": "INTENT: discovery"}]'
    )
    assert _flatten_message_content(json_str) == "INTENT: discovery"


@pytest.mark.unit
def test_flatten_list_of_string_blocks() -> None:
    """A list containing raw string elements joins them."""
    content: list[Any] = ["chunk1", "chunk2", "chunk3"]
    assert _flatten_message_content(content) == "chunk1chunk2chunk3"


@pytest.mark.unit
def test_flatten_unknown_block_types_dropped() -> None:
    """Unknown block types (image, tool_use, audio) are dropped — only
    ``type=='text'`` blocks contribute. Keeps the function FMAPI-agnostic."""
    content: list[dict[str, Any]] = [
        {"type": "image", "url": "..."},
        {"type": "tool_use", "id": "...", "name": "..."},
        {"type": "text", "text": "visible"},
        {"type": "audio", "data": "..."},
    ]
    assert _flatten_message_content(content) == "visible"


@pytest.mark.unit
def test_flatten_non_list_non_string_passthrough() -> None:
    """Unknown content types (dict, int, None) pass through unchanged."""
    obj: dict[str, str] = {"weird": "shape"}
    assert _flatten_message_content(obj) is obj
    assert _flatten_message_content(42) == 42
    assert _flatten_message_content(None) is None


# Handoff tool's runtime path is tested via integration on live deploys
# (the unit-test setup for ToolRuntime requires a full langgraph runtime
# context which is outside the scope of these flatten contracts). The
# tests above cover the flatten function in isolation, which is the
# single load-bearing piece of the handoff's content-normalization fix.
