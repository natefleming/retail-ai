"""Tests for ``session_items_from_messages`` — converts a LangGraph
checkpointer's ``state.values["messages"]`` (LangChain messages) into the
conversation items the Console renders when a past session is reloaded.

Reasoning is separated from the answer text (mirroring the streaming contract),
so a reloaded transcript matches what the live turn showed.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from dao_ai.apps.sessions import session_items_from_messages


class TestSessionItemsFromMessages:
    @pytest.mark.unit
    def test_human_message_becomes_user_item(self) -> None:
        items = session_items_from_messages([HumanMessage(content="hi there")])
        assert items == [{"role": "user", "content": "hi there"}]

    @pytest.mark.unit
    def test_ai_message_separates_reasoning_from_answer(self) -> None:
        msg = AIMessage(
            content=[
                {"type": "reasoning", "reasoning": "let me think"},
                {"type": "text", "text": "The answer is 42."},
            ]
        )
        items = session_items_from_messages([msg])
        assert items == [
            {
                "role": "assistant",
                "content": "The answer is 42.",
                "reasoning": "let me think",
            }
        ]

    @pytest.mark.unit
    def test_plain_ai_message_has_no_reasoning_key(self) -> None:
        items = session_items_from_messages([AIMessage(content="just an answer")])
        assert items == [{"role": "assistant", "content": "just an answer"}]

    @pytest.mark.unit
    def test_tool_message_becomes_tool_item_with_name(self) -> None:
        msg = ToolMessage(content="found 3 docs", name="search_docs", tool_call_id="c1")
        items = session_items_from_messages([msg])
        assert items == [
            {"role": "tool", "name": "search_docs", "content": "found 3 docs"}
        ]

    @pytest.mark.unit
    def test_empty_content_messages_skipped(self) -> None:
        items = session_items_from_messages(
            [AIMessage(content=""), HumanMessage(content="q")]
        )
        assert items == [{"role": "user", "content": "q"}]

    @pytest.mark.unit
    def test_full_turn_order_preserved(self) -> None:
        items = session_items_from_messages(
            [
                HumanMessage(content="find docs"),
                ToolMessage(content="2 hits", name="search", tool_call_id="c1"),
                AIMessage(content="Here are 2 results."),
            ]
        )
        assert [i["role"] for i in items] == ["user", "tool", "assistant"]
