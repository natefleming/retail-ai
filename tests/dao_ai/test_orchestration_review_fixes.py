"""
Regression tests for the orchestration code-review findings fixed on
``fix/orchestration-review-findings``.

Each test class corresponds to one finding from the original review. Test
names match the verification section of the plan so it's obvious what each
guards against.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command

from dao_ai.config import (
    AgentModel,
    AppConfig,
    AppModel,
    LLMModel,
    OrchestrationModel,
    SupervisorModel,
    SwarmModel,
)
from dao_ai.messages import (
    last_ai_message,
    last_ai_message_with_tool_calls,
)
from dao_ai.orchestration.core import (
    SUPERVISOR_NODE,
    filter_messages_for_agent,
)
from dao_ai.orchestration.swarm import _create_deterministic_handler
from dao_ai.state import (
    GenieSpaceState,
    GenieState,
    SessionState,
    merge_session,
)


def _basic_app_config(agent_names: list[str]) -> AppConfig:
    """Build a minimal AppConfig with the given agent names.

    ``deployment_target='apps'`` lets us skip the ``registered_model`` field
    that's otherwise required.
    """
    agents = [
        AgentModel(name=name, model=LLMModel(name="test-model"))
        for name in agent_names
    ]
    return AppConfig(
        app=AppModel(
            name="dao_ai_test",
            deployment_target="apps",
            agents=agents,
            orchestration=OrchestrationModel(
                supervisor=SupervisorModel(model=LLMModel(name="test-model"))
            ),
        )
    )


def _make_tool_runtime(
    *, tool_call_id: str, state: dict[str, Any]
) -> ToolRuntime:
    """Build a ToolRuntime suitable for invoking @tool-decorated handoffs in tests."""
    return ToolRuntime(
        state=state,
        context=None,
        config=None,
        stream_writer=lambda *_: None,
        tool_call_id=tool_call_id,
        store=None,
        execution_info=None,
        server_info=None,
    )


# =============================================================================
# Finding #1: deterministic handler must pass through Command unchanged
# =============================================================================


@pytest.mark.unit
class TestDeterministicHandlerCommandPassthrough:
    """The deterministic-handoff wrapper must not crash when the inner
    handler returns a Command from the agentic-handoff (ParentCommand) path."""

    def test_returns_command_unchanged_when_inner_returns_command(self) -> None:
        agentic_command = Command(
            goto="agent_x",
            graph=Command.PARENT,
            update={"active_agent": "agent_x", "messages": []},
        )

        async def inner_handler(state: dict, runtime: Any) -> Command:
            return agentic_command

        wrapped = _create_deterministic_handler(inner_handler, "agent_y")
        result = asyncio.run(wrapped({"messages": []}, runtime=MagicMock()))

        assert isinstance(result, Command)
        # Agentic target wins; deterministic target must NOT have overwritten it.
        assert result.update["active_agent"] == "agent_x"
        assert result.goto == "agent_x"

    def test_sets_active_agent_when_inner_returns_dict(self) -> None:
        async def inner_handler(state: dict, runtime: Any) -> dict:
            return {"messages": [], "active_agent": "stale"}

        wrapped = _create_deterministic_handler(inner_handler, "agent_y")
        result = asyncio.run(wrapped({"messages": []}, runtime=MagicMock()))

        assert isinstance(result, dict)
        assert result["active_agent"] == "agent_y"


# =============================================================================
# Finding #2: filter_messages_for_agent preserves agent's own ToolMessages
# =============================================================================


@pytest.mark.unit
class TestFilterMessagesForAgentTagging:
    """Agents must see their own prior tool exchanges; peers' must be hidden.

    Ownership: AIMessage by ``msg.name``, ToolMessage by ``tool_call_id``
    pairing against an own AIMessage(tool_calls=…).
    """

    def test_keeps_own_tool_exchange_drops_peer_tool_exchange(self) -> None:
        # ToolMessage.name carries the *tool*'s name (set by langchain), not
        # the agent's — pairing works via tool_call_id.
        own_call = AIMessage(
            content="checking inventory",
            name="agent_a",
            tool_calls=[
                {"name": "lookup", "args": {"sku": "1"}, "id": "call_a1"}
            ],
        )
        own_result = ToolMessage(
            content="42 in stock", tool_call_id="call_a1", name="lookup"
        )
        peer_call = AIMessage(
            content="forecasting demand",
            name="agent_b",
            tool_calls=[{"name": "forecast", "args": {}, "id": "call_b1"}],
        )
        peer_result = ToolMessage(
            content="trending up", tool_call_id="call_b1", name="forecast"
        )

        history = [
            HumanMessage(content="how many widgets?"),
            own_call,
            own_result,
            peer_call,
            peer_result,
        ]

        filtered = filter_messages_for_agent(history, current_agent_name="agent_a")

        # agent_a's tool exchange present (paired by tool_call_id)
        assert own_call in filtered
        assert own_result in filtered
        # agent_b's ToolMessage dropped (no matching kept AIMessage)
        assert peer_result not in filtered
        # peer AIMessage with content survives but with tool_calls stripped
        peer_in_filtered = next(
            (m for m in filtered if isinstance(m, AIMessage) and m.name == "agent_b"),
            None,
        )
        assert peer_in_filtered is not None
        assert not peer_in_filtered.tool_calls

    def test_pairs_multiple_tool_calls_in_one_aimessage(self) -> None:
        own_call = AIMessage(
            content="parallel fetches",
            name="agent_a",
            tool_calls=[
                {"name": "lookup", "args": {"sku": "1"}, "id": "tc1"},
                {"name": "lookup", "args": {"sku": "2"}, "id": "tc2"},
            ],
        )
        result1 = ToolMessage(content="r1", tool_call_id="tc1", name="lookup")
        result2 = ToolMessage(content="r2", tool_call_id="tc2", name="lookup")
        history = [HumanMessage(content="q"), own_call, result1, result2]

        filtered = filter_messages_for_agent(history, current_agent_name="agent_a")
        assert result1 in filtered
        assert result2 in filtered

    def test_legacy_no_agent_name_strips_all_tool_messages(self) -> None:
        """When called with no agent name, behaviour matches the pre-refactor
        contract: all ToolMessages dropped, all tool_calls stripped."""
        history = [
            AIMessage(
                content="hi",
                tool_calls=[{"name": "x", "args": {}, "id": "c1"}],
            ),
            ToolMessage(content="result", tool_call_id="c1", name="x"),
        ]
        filtered = filter_messages_for_agent(history)
        assert all(not isinstance(m, ToolMessage) for m in filtered)
        for m in filtered:
            if isinstance(m, AIMessage):
                assert not m.tool_calls


# =============================================================================
# Finding #4: handoff_to_supervisor sets active_agent
# =============================================================================


@pytest.mark.unit
class TestHandoffToSupervisorSetsActiveAgent:
    def test_command_update_includes_supervisor_active_agent(self) -> None:
        # Build the supervisor handoff tool and invoke it with a synthetic
        # runtime that supplies the required tool_call_id and message state.
        from dao_ai.orchestration.supervisor import (
            _create_handoff_back_to_supervisor_tool,
        )

        tool: BaseTool = _create_handoff_back_to_supervisor_tool()

        runtime = _make_tool_runtime(
            tool_call_id="tc_1",
            state={
                "messages": [
                    AIMessage(
                        content="done",
                        tool_calls=[
                            {
                                "name": "handoff_to_supervisor",
                                "args": {"summary": "ok"},
                                "id": "tc_1",
                            }
                        ],
                    )
                ]
            },
        )

        result = tool.invoke({"summary": "ok", "runtime": runtime})

        assert isinstance(result, Command)
        assert result.update["active_agent"] == SUPERVISOR_NODE
        assert result.goto == SUPERVISOR_NODE


# =============================================================================
# Finding #5: last_ai_message_with_tool_calls helper
# =============================================================================


@pytest.mark.unit
class TestLastAIMessageWithToolCalls:
    def test_returns_most_recent_with_tool_calls(self) -> None:
        msgs = [
            HumanMessage(content="hi"),
            AIMessage(content="ok"),
            AIMessage(
                content="picking a tool",
                tool_calls=[{"name": "t1", "args": {}, "id": "c1"}],
            ),
            AIMessage(content="follow up"),
            AIMessage(
                content="picking another tool",
                tool_calls=[{"name": "t2", "args": {}, "id": "c2"}],
            ),
            AIMessage(content="final reply"),
        ]
        result = last_ai_message_with_tool_calls(msgs)
        assert result is not None
        assert result.tool_calls[0]["id"] == "c2"

    def test_returns_none_when_no_tool_calls(self) -> None:
        msgs = [
            HumanMessage(content="hi"),
            AIMessage(content="hello"),
        ]
        assert last_ai_message_with_tool_calls(msgs) is None

    def test_does_not_match_plain_ai_message(self) -> None:
        # Plain AIMessage (no tool_calls) is NOT what this helper wants.
        msgs = [AIMessage(content="hello")]
        assert last_ai_message_with_tool_calls(msgs) is None
        # Sanity: last_ai_message would still find it.
        assert last_ai_message(msgs) is not None


# =============================================================================
# Finding #7: SUPERVISOR_NODE name collision rejected
# =============================================================================


@pytest.mark.unit
class TestSupervisorNodeNameCollision:
    def test_create_supervisor_graph_rejects_supervisor_named_worker(
        self,
    ) -> None:
        from dao_ai.orchestration.supervisor import create_supervisor_graph

        config = _basic_app_config(["billing", SUPERVISOR_NODE])

        with pytest.raises(ValueError) as exc:
            create_supervisor_graph(config)

        msg = str(exc.value)
        assert SUPERVISOR_NODE in msg
        assert "collide" in msg.lower()


# =============================================================================
# Finding #9: SwarmModel.max_hops surfaces and applies
# =============================================================================


@pytest.mark.unit
class TestSwarmMaxHops:
    def test_default_is_25(self) -> None:
        assert SwarmModel().max_hops == 25

    def test_custom_value_round_trips(self) -> None:
        assert SwarmModel(max_hops=4).max_hops == 4

    def test_rejects_non_positive(self) -> None:
        with pytest.raises(Exception):
            SwarmModel(max_hops=0)


# =============================================================================
# Finding #10: merge_session generalizes across SessionState fields
# =============================================================================


@pytest.mark.unit
class TestMergeSessionGeneric:
    def test_genie_spaces_still_merge(self) -> None:
        a = SessionState(
            genie=GenieState(
                spaces={"sp_a": GenieSpaceState(conversation_id="conv_a")}
            )
        )
        b = SessionState(
            genie=GenieState(
                spaces={"sp_b": GenieSpaceState(conversation_id="conv_b")}
            )
        )
        merged = merge_session(a, b)
        assert set(merged.genie.spaces.keys()) == {"sp_a", "sp_b"}

    def test_overlapping_keys_take_new(self) -> None:
        a = SessionState(
            genie=GenieState(
                spaces={"sp": GenieSpaceState(conversation_id="conv_old")}
            )
        )
        b = SessionState(
            genie=GenieState(
                spaces={"sp": GenieSpaceState(conversation_id="conv_new")}
            )
        )
        merged = merge_session(a, b)
        assert merged.genie.spaces["sp"].conversation_id == "conv_new"

    def test_basemodel_walker_handles_arbitrary_dict_field(self) -> None:
        """The reducer's recursive walker should handle a hypothetical new
        SessionState field that's a BaseModel containing a dict, without
        special-casing."""
        from dao_ai.state import _merge_basemodel
        from pydantic import BaseModel, Field

        class FakeToolState(BaseModel):
            entries: dict[str, str] = Field(default_factory=dict)

        a = FakeToolState(entries={"k1": "v1_old"})
        b = FakeToolState(entries={"k1": "v1_new", "k2": "v2"})
        merged = _merge_basemodel(a, b)
        assert isinstance(merged, FakeToolState)
        assert merged.entries == {"k1": "v1_new", "k2": "v2"}


# =============================================================================
# Finding #3: OrchestrationModel.output_mode plumbed and configurable
# =============================================================================


@pytest.mark.unit
class TestOrchestrationOutputMode:
    def test_default_is_full_history(self) -> None:
        m = OrchestrationModel(
            supervisor=SupervisorModel(model=LLMModel(name="test-model"))
        )
        assert m.output_mode == "full_history"

    def test_accepts_last_message(self) -> None:
        m = OrchestrationModel(
            supervisor=SupervisorModel(model=LLMModel(name="test-model")),
            output_mode="last_message",
        )
        assert m.output_mode == "last_message"

    def test_rejects_unknown_value(self) -> None:
        with pytest.raises(Exception):
            OrchestrationModel(
                supervisor=SupervisorModel(model=LLMModel(name="test-model")),
                output_mode="garbage",
            )
