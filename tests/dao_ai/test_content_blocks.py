"""Tests for content block extraction reproducing hardware_store_lakebase scenarios.

These tests reproduce the exact content block formats observed when
``ChatDatabricks`` (Chat Completions API) JSON-stringifies Claude's
reasoning/text content blocks during streaming inference.
"""

import json

import pytest

from dao_ai.models import _extract_reasoning_text, _extract_text_content

# ---------------------------------------------------------------------------
# Fixtures: exact payloads from hardware_store_lakebase deployment logs
# ---------------------------------------------------------------------------

FAVORITE_COLOR_JSON = json.dumps(
    [
        {
            "type": "reasoning",
            "summary": [
                {
                    "type": "summary_text",
                    "text": "The user asks favorite color. Memory says favorite color is green. So answer: green.",
                }
            ],
        },
        {"type": "text", "text": "Your favorite color is green."},
    ]
)

DOG_NAME_JSON = json.dumps(
    [
        {
            "type": "reasoning",
            "summary": [
                {
                    "type": "summary_text",
                    "text": "We have stored dog's name. Now user likely will ask again about dog's name. "
                    "We should respond acknowledging and confirming we remember.",
                }
            ],
        },
        {
            "type": "text",
            "text": "Got it! I\u2019ve saved that your dog\u2019s name is Ripley\u202fLongshanks. "
            "I\u2019ll remember that for future conversations. \U0001f43e If you need anything "
            "else\u2014whether it\u2019s more info about the Big\u202fGreen\u202fEgg grills, "
            "accessories, or anything else\u2014just let me know!",
        },
    ]
)


# ---------------------------------------------------------------------------
# TestExtractReasoningText
# ---------------------------------------------------------------------------


class TestExtractReasoningText:
    """Unit tests for the ``_extract_reasoning_text`` helper."""

    @pytest.mark.unit
    def test_databricks_openai_summary_format(self) -> None:
        block = {
            "type": "reasoning",
            "summary": [
                {"type": "summary_text", "text": "step one"},
                {"type": "summary_text", "text": "step two"},
            ],
        }
        assert _extract_reasoning_text(block) == "step one step two"

    @pytest.mark.unit
    def test_langchain_standard_reasoning(self) -> None:
        block = {"type": "reasoning", "reasoning": "I need to think about this."}
        assert _extract_reasoning_text(block) == "I need to think about this."

    @pytest.mark.unit
    def test_anthropic_thinking_format(self) -> None:
        block = {"type": "thinking", "thinking": "Let me reason through this..."}
        assert _extract_reasoning_text(block) == "Let me reason through this..."

    @pytest.mark.unit
    def test_text_block_returns_none(self) -> None:
        block = {"type": "text", "text": "hello"}
        assert _extract_reasoning_text(block) is None

    @pytest.mark.unit
    def test_unknown_type_returns_none(self) -> None:
        block = {"type": "image", "url": "http://example.com"}
        assert _extract_reasoning_text(block) is None

    @pytest.mark.unit
    def test_reasoning_with_extras(self) -> None:
        block = {
            "type": "reasoning",
            "reasoning": "deep thought",
            "extras": {"signature": "abc123"},
        }
        assert _extract_reasoning_text(block) == "deep thought"

    @pytest.mark.unit
    def test_empty_summary_list(self) -> None:
        block = {"type": "reasoning", "summary": []}
        assert _extract_reasoning_text(block) is None

    @pytest.mark.unit
    def test_summary_with_non_summary_text_items(self) -> None:
        block = {
            "type": "reasoning",
            "summary": [
                {"type": "summary_text", "text": "valid"},
                {"type": "other", "text": "ignored"},
            ],
        }
        assert _extract_reasoning_text(block) == "valid"


# ---------------------------------------------------------------------------
# TestExtractTextContentJsonStringified
# ---------------------------------------------------------------------------


class TestExtractTextContentJsonStringified:
    """Reproduce the actual bug: JSON-stringified content blocks from ChatDatabricks."""

    @pytest.mark.unit
    def test_favorite_color_response(self) -> None:
        """Exact payload from 'what is my favorite color?' query."""
        result = _extract_text_content(FAVORITE_COLOR_JSON)
        assert "Your favorite color is green." in result
        # Must NOT contain raw JSON structure
        assert '"type"' not in result
        assert '"reasoning"' not in result

    @pytest.mark.unit
    def test_dog_name_response(self) -> None:
        """Exact payload from 'what is my dog's name?' query."""
        result = _extract_text_content(DOG_NAME_JSON)
        assert "Ripley" in result
        assert "Longshanks" in result
        # Must NOT contain raw JSON structure
        assert '"type"' not in result
        assert '"summary"' not in result

    @pytest.mark.unit
    def test_json_text_only(self) -> None:
        """JSON with only text blocks should produce clean text."""
        content = json.dumps([{"type": "text", "text": "Hello world!"}])
        result = _extract_text_content(content)
        assert result == "Hello world!"

    @pytest.mark.unit
    def test_json_reasoning_only(self) -> None:
        """JSON with only reasoning should produce formatted reasoning."""
        content = json.dumps(
            [
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "thinking hard"}],
                }
            ]
        )
        result = _extract_text_content(content)
        assert "thinking hard" in result
        assert "*" in result  # markdown italics

    @pytest.mark.unit
    def test_malformed_json_passthrough(self) -> None:
        """Strings that look like JSON arrays but aren't should pass through."""
        content = "[this is not json]"
        assert _extract_text_content(content) == "[this is not json]"

    @pytest.mark.unit
    def test_regular_bracket_string(self) -> None:
        """Regular strings starting with [ that aren't JSON pass through."""
        content = "[INFO] Server started successfully"
        assert _extract_text_content(content) == "[INFO] Server started successfully"

    @pytest.mark.unit
    def test_json_array_of_non_dicts(self) -> None:
        """JSON array of strings (not content blocks) passes through."""
        content = json.dumps(["a", "b", "c"])
        assert _extract_text_content(content) == content

    @pytest.mark.unit
    def test_json_with_whitespace(self) -> None:
        """JSON with leading/trailing whitespace should still be parsed."""
        content = '  [{"type": "text", "text": "padded"}]  '
        assert _extract_text_content(content) == "padded"


# ---------------------------------------------------------------------------
# TestExtractTextContentPythonList
# ---------------------------------------------------------------------------


class TestExtractTextContentPythonList:
    """Test the Responses API path where content arrives as a Python list."""

    @pytest.mark.unit
    def test_databricks_reasoning_plus_text(self) -> None:
        content = [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "thinking..."}],
            },
            {"type": "text", "text": "The answer is 42."},
        ]
        result = _extract_text_content(content)
        assert "The answer is 42." in result
        assert "thinking..." in result

    @pytest.mark.unit
    def test_anthropic_thinking_plus_text(self) -> None:
        content = [
            {"type": "thinking", "thinking": "Let me consider this carefully."},
            {"type": "text", "text": "Here is my response."},
        ]
        result = _extract_text_content(content)
        assert "Here is my response." in result
        assert "Let me consider this carefully." in result

    @pytest.mark.unit
    def test_langchain_standard_reasoning_plus_text(self) -> None:
        content = [
            {
                "type": "reasoning",
                "reasoning": "Analyzing the question...",
                "extras": {"signature": "xyz"},
            },
            {"type": "text", "text": "My conclusion."},
        ]
        result = _extract_text_content(content)
        assert "My conclusion." in result
        assert "Analyzing the question..." in result

    @pytest.mark.unit
    def test_multiple_text_blocks(self) -> None:
        content = [
            {"type": "text", "text": "Part one. "},
            {"type": "text", "text": "Part two."},
        ]
        result = _extract_text_content(content)
        assert result == "Part one. Part two."

    @pytest.mark.unit
    def test_multiple_reasoning_blocks(self) -> None:
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
        result = _extract_text_content(content)
        assert "Final answer." in result
        assert "first thought" in result
        assert "second thought" in result

    @pytest.mark.unit
    def test_text_only_no_reasoning_artifacts(self) -> None:
        """Text-only content should have no markdown formatting artifacts."""
        content = [{"type": "text", "text": "Just text, nothing fancy."}]
        result = _extract_text_content(content)
        assert result == "Just text, nothing fancy."
        assert ">" not in result
        assert "*" not in result


# ---------------------------------------------------------------------------
# TestExtractTextContentReasoningFormatting
# ---------------------------------------------------------------------------


class TestExtractTextContentReasoningFormatting:
    """Verify that reasoning text is formatted as markdown italics in a blockquote."""

    @pytest.mark.unit
    def test_reasoning_in_blockquote_italics(self) -> None:
        content = [
            {"type": "reasoning", "reasoning": "my reasoning"},
            {"type": "text", "text": "my response"},
        ]
        result = _extract_text_content(content)
        assert "> *my reasoning*" in result

    @pytest.mark.unit
    def test_text_follows_reasoning(self) -> None:
        content = [
            {"type": "reasoning", "reasoning": "thought"},
            {"type": "text", "text": "answer"},
        ]
        result = _extract_text_content(content)
        reasoning_pos = result.index("> *thought*")
        text_pos = result.index("answer")
        assert text_pos > reasoning_pos

    @pytest.mark.unit
    def test_no_reasoning_clean_output(self) -> None:
        content = [{"type": "text", "text": "clean"}]
        result = _extract_text_content(content)
        assert result == "clean"

    @pytest.mark.unit
    def test_json_stringified_reasoning_formatted(self) -> None:
        """JSON-stringified content should produce the same formatting."""
        content_list = [
            {"type": "reasoning", "reasoning": "json thought"},
            {"type": "text", "text": "json answer"},
        ]
        content_json = json.dumps(content_list)
        result = _extract_text_content(content_json)
        assert "> *json thought*" in result
        assert "json answer" in result
