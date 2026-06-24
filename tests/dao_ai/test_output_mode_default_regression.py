"""Regression tests for the orphan-tool_result bug class fixed by flipping
``OrchestrationModel.output_mode`` default to ``last_message``.

Background: trace ``tr-7f30e2da4c02accfc11bc08cae54eef2`` (fevm experiment
``1661766236280959``) captured a Claude supervisor 400 with::

    'messages.16.content.0: unexpected `tool_use_id` found in `tool_result`
    blocks: toolu_bdrk_01AJt5CB3JUz6wVBKeYCRQgh. Each `tool_result` block
    must have a corresponding `tool_use` block in the previous message.'

Root cause: worker subgraph's local history accumulated malformed messages
(``MemoryContextMiddleware`` inserting ``## Memories`` ``SystemMessage`` rows
between assistant tool_use and tool_result blocks; parallel tool_calls
emitted by Claude-family workers). With ``output_mode='full_history'`` the
malformed worker history propagated into the supervisor's state via
``extract_agent_response``. The supervisor's next LLM call then 400'd.

The fix: default ``output_mode`` is ``last_message``. ``extract_agent_response``
returns only the worker's final ``AIMessage`` with ``tool_calls`` stripped, so
no worker-side malformed messages can reach the supervisor. These tests pin
that behavior down so it doesn't regress.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from dao_ai.config import (
    LLMModel,
    OrchestrationModel,
    SupervisorModel,
    SwarmModel,
)
from dao_ai.orchestration.core import extract_agent_response

# =============================================================================
# Default value
# =============================================================================


@pytest.mark.unit
class TestOutputModeDefault:
    def test_default_supervisor(self) -> None:
        """A supervisor-only orchestration gets ``last_message`` by default."""
        m = OrchestrationModel(
            supervisor=SupervisorModel(model=LLMModel(name="test-model"))
        )
        assert m.output_mode == "last_message"

    def test_default_swarm(self) -> None:
        """A swarm-only orchestration gets ``last_message`` by default. Swarm
        configs that genuinely need cross-agent tool context opt in via an
        explicit ``output_mode: full_history`` in YAML."""
        m = OrchestrationModel(swarm=SwarmModel())
        assert m.output_mode == "last_message"

    def test_default_with_no_active_mode(self) -> None:
        """Even with no orchestration mode set, default is ``last_message``."""
        m = OrchestrationModel()
        assert m.output_mode == "last_message"

    def test_explicit_full_history_override(self) -> None:
        """Apps that actually want cross-agent tool context can opt in."""
        m = OrchestrationModel(
            supervisor=SupervisorModel(model=LLMModel(name="test-model")),
            output_mode="full_history",
        )
        assert m.output_mode == "full_history"


# =============================================================================
# extract_agent_response — verifies the worker-side malformed-history patterns
# from the bug never reach the supervisor's state under the new default.
# =============================================================================


@pytest.mark.unit
class TestExtractAgentResponseDefaultBlocksMalformedHistory:
    def test_strips_tool_calls_from_final_ai_message(self) -> None:
        """Worker's final AIMessage might still carry tool_calls (it's the
        last assistant turn that issued a tool call). last_message must strip
        them so the supervisor doesn't try to satisfy an orphan tool_use."""
        worker_history = [
            AIMessage(
                content="Searching...",
                tool_calls=[
                    {
                        "name": "find_sku",
                        "args": {"sku": "ABC"},
                        "id": "tu_1",
                        "type": "tool_call",
                    }
                ],
                id="ai-1",
            ),
            ToolMessage(content="found", tool_call_id="tu_1", id="tm-1"),
            AIMessage(
                content="Here you go",
                tool_calls=[
                    {
                        "name": "find_sku",
                        "args": {"sku": "XYZ"},
                        "id": "tu_2",
                        "type": "tool_call",
                    }
                ],
                id="ai-2",
            ),
        ]
        out = extract_agent_response(worker_history, output_mode="last_message")
        assert len(out) == 1
        assert isinstance(out[0], AIMessage)
        assert out[0].content == "Here you go"
        assert out[0].tool_calls == [] or out[0].tool_calls is None

    def test_drops_interleaved_system_message_from_memory_middleware(self) -> None:
        """The MemoryContextMiddleware injects ``## Memories`` SystemMessages
        between assistant tool_use and tool_result blocks (the actual bug
        cause). last_message extracts only the final AIMessage and never
        propagates the interleaved system message into the supervisor's view."""
        worker_history = [
            HumanMessage(content="any memory?", id="h-1"),
            SystemMessage(content="## Memories\n- nickname Nate", id="sm-1"),
            AIMessage(
                content="Looking it up.",
                tool_calls=[
                    {
                        "name": "search_memory",
                        "args": {"query": "nickname"},
                        "id": "tu_mem",
                        "type": "tool_call",
                    }
                ],
                id="ai-1",
            ),
            ToolMessage(content="found", tool_call_id="tu_mem", id="tm-1"),
            SystemMessage(content="## Memories\n- nickname Nate", id="sm-2"),
            AIMessage(content="Your nickname is Nate", id="ai-final"),
        ]
        out = extract_agent_response(worker_history, output_mode="last_message")
        assert len(out) == 1
        assert isinstance(out[0], AIMessage)
        assert out[0].content == "Your nickname is Nate"
        # No SystemMessage / no ToolMessage / no orphan tool_call propagated.
        assert not any(isinstance(m, (SystemMessage, ToolMessage)) for m in out)

    def test_drops_parallel_tool_calls_assistant_message(self) -> None:
        """Worker may emit text + parallel tool_calls in one assistant message
        (Claude's pattern). Under last_message that message becomes a clean
        text-only AIMessage when surfaced to the supervisor — no parallel
        tool_calls in supervisor's history for strict-validation LLMs to
        choke on."""
        worker_history = [
            AIMessage(
                content="Looking up products + memory",
                tool_calls=[
                    {
                        "name": "product_vector_search",
                        "args": {"query": "drills"},
                        "id": "tu_par_1",
                        "type": "tool_call",
                    },
                    {
                        "name": "search_memory",
                        "args": {"query": "drill prefs"},
                        "id": "tu_par_2",
                        "type": "tool_call",
                    },
                ],
                id="ai-parallel",
            ),
            ToolMessage(content="DEWALT etc.", tool_call_id="tu_par_1", id="tm-1"),
            ToolMessage(content="[]", tool_call_id="tu_par_2", id="tm-2"),
            AIMessage(content="Here are the DEWALT drills.", id="ai-final"),
        ]
        out = extract_agent_response(worker_history, output_mode="last_message")
        assert len(out) == 1
        assert isinstance(out[0], AIMessage)
        assert out[0].content == "Here are the DEWALT drills."

    def test_orphan_tool_result_does_not_propagate(self) -> None:
        """The exact bug shape from trace tr-7f30e2da... — a ToolMessage that
        has no matching tool_use in the immediately preceding assistant
        message — must not leak into the supervisor's state."""
        worker_history = [
            AIMessage(content="Let me search...", id="ai-text-only"),
            ToolMessage(content="[orphan]", tool_call_id="tu_orphan", id="tm-orphan"),
            AIMessage(content="Result is X.", id="ai-final"),
        ]
        out = extract_agent_response(worker_history, output_mode="last_message")
        assert len(out) == 1
        assert isinstance(out[0], AIMessage)
        assert out[0].content == "Result is X."
        assert not any(isinstance(m, ToolMessage) for m in out)

    def test_empty_worker_returns_empty(self) -> None:
        """Pathological case: worker returned no messages. Don't crash."""
        out = extract_agent_response([], output_mode="last_message")
        assert out == []

    def test_full_history_still_passes_through_when_explicitly_set(self) -> None:
        """Apps that explicitly opt in to full_history get the old behavior."""
        worker_history = [
            HumanMessage(content="hi"),
            SystemMessage(content="## Memories"),
            AIMessage(content="Done."),
        ]
        out = extract_agent_response(worker_history, output_mode="full_history")
        assert out == worker_history
