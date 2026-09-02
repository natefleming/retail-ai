"""Tests for ``_split_content`` — the pure helper that separates answer text
from reasoning/thinking text without any markdown formatting.

This is the parsing primitive behind the streaming path's separated-reasoning
channel. Unlike ``_extract_text_content`` (which folds reasoning into a markdown
blockquote for legacy non-streaming callers), ``_split_content`` returns the two
channels as a plain ``(text, reasoning)`` tuple so the streaming agent can emit
them as distinct Responses items.
"""

import json

import pytest

from dao_ai.models import _split_content


class TestSplitContent:
    """Unit tests for the ``_split_content`` helper."""

    @pytest.mark.unit
    def test_text_only_list_has_empty_reasoning(self) -> None:
        content = [{"type": "text", "text": "Just text, nothing fancy."}]
        text, reasoning = _split_content(content)
        assert text == "Just text, nothing fancy."
        assert reasoning == ""

    @pytest.mark.unit
    def test_reasoning_plus_text_split_cleanly(self) -> None:
        content = [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "thinking..."}],
            },
            {"type": "text", "text": "The answer is 42."},
        ]
        text, reasoning = _split_content(content)
        assert text == "The answer is 42."
        assert reasoning == "thinking..."

    @pytest.mark.unit
    def test_reasoning_only_yields_empty_text(self) -> None:
        content = [{"type": "reasoning", "reasoning": "thinking hard"}]
        text, reasoning = _split_content(content)
        assert text == ""
        assert reasoning == "thinking hard"

    @pytest.mark.unit
    def test_no_markdown_artifacts_in_either_channel(self) -> None:
        content = [
            {"type": "reasoning", "reasoning": "my reasoning"},
            {"type": "text", "text": "my response"},
        ]
        text, reasoning = _split_content(content)
        # Neither channel carries blockquote / italics formatting.
        assert ">" not in text and ">" not in reasoning
        assert "*" not in text and "*" not in reasoning
        assert text == "my response"
        assert reasoning == "my reasoning"

    @pytest.mark.unit
    def test_multiple_reasoning_blocks_joined(self) -> None:
        content = [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "first thought"}],
            },
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "second thought"}],
            },
            {"type": "text", "text": "Final answer."},
        ]
        text, reasoning = _split_content(content)
        assert text == "Final answer."
        assert "first thought" in reasoning
        assert "second thought" in reasoning

    @pytest.mark.unit
    def test_multiple_text_blocks_concatenated(self) -> None:
        content = [
            {"type": "text", "text": "Part one. "},
            {"type": "text", "text": "Part two."},
        ]
        text, reasoning = _split_content(content)
        assert text == "Part one. Part two."
        assert reasoning == ""

    @pytest.mark.unit
    def test_plain_string_passthrough(self) -> None:
        text, reasoning = _split_content("hello world")
        assert text == "hello world"
        assert reasoning == ""

    @pytest.mark.unit
    def test_json_stringified_list_is_parsed(self) -> None:
        content = json.dumps(
            [
                {"type": "reasoning", "reasoning": "json thought"},
                {"type": "text", "text": "json answer"},
            ]
        )
        text, reasoning = _split_content(content)
        assert text == "json answer"
        assert reasoning == "json thought"
