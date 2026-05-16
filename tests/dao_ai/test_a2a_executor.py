"""Unit tests for :class:`dao_ai.apps.a2a.executor.A2AAgentExecutor`.

Following dao-ai convention (this repo does not use pytest-asyncio), all
async exercise paths are driven via :func:`asyncio.run` inside sync test
bodies.

Covered:

* Part extraction (TextPart, DataPart, mixed).
* OBO header propagation from ``call_context.state['headers']``.
* Happy-path ``message/send`` → completed task with TextPart artifact.
* Structured-response surfacing as a ``DataPart`` alongside the text artifact.
* HITL: structured DataPart resume → :class:`Command` passed to graph.
* HITL: graph yields ``__interrupt__`` → input-required terminal state.
* :class:`langgraph.errors.GraphInterrupt` mid-invoke → input-required.
* ``cancel`` → canceled state.
* Unexpected exception → failed terminal state.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from a2a.auth.user import UnauthenticatedUser
from a2a.server.agent_execution import RequestContext
from a2a.server.context import ServerCallContext
from a2a.server.events import EventQueue
from a2a.types import (
    DataPart,
    Message,
    MessageSendParams,
    Part,
    Role,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
    TextPart,
)
from langchain_core.messages import AIMessage
from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from dao_ai.apps.a2a.executor import A2AAgentExecutor
from dao_ai.config import (
    AgentModel,
    AppConfig,
    AppModel,
    DeploymentTarget,
    InferenceEndpointModel,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _minimal_config() -> AppConfig:
    return AppConfig(
        app=AppModel(
            name="dao-ai-exec-test",
            description="executor test",
            deployment_target=DeploymentTarget.APPS,
            agents=[
                AgentModel(
                    name="greeter",
                    description="says hi",
                    model=InferenceEndpointModel(name="databricks-gpt-5-4-mini"),
                ),
            ],
        )
    )


def _make_context(
    *,
    text: str | None = None,
    data: dict | None = None,
    task_id: str = "task-1",
    context_id: str = "ctx-1",
    headers: dict[str, str] | None = None,
    current_task=None,
) -> RequestContext:
    parts: list[Part] = []
    if text is not None:
        parts.append(Part(root=TextPart(text=text)))
    if data is not None:
        parts.append(Part(root=DataPart(data=data)))

    message = Message(
        message_id="msg-1",
        role=Role.user,
        parts=parts,
        context_id=context_id,
        task_id=task_id,
    )
    params = MessageSendParams(message=message)

    call_context = ServerCallContext(
        user=UnauthenticatedUser(),
        state={"headers": dict(headers)} if headers else {},
    )
    return RequestContext(
        request=params,
        task_id=task_id,
        context_id=context_id,
        task=current_task,
        call_context=call_context,
    )


def _make_graph(
    *,
    ainvoke_return=None,
    ainvoke_side_effect=None,
    aget_state_return=None,
    checkpointer: bool = False,
) -> MagicMock:
    graph = MagicMock()
    graph.ainvoke = AsyncMock(
        return_value=ainvoke_return, side_effect=ainvoke_side_effect
    )
    graph.aget_state = AsyncMock(return_value=aget_state_return)
    graph.checkpointer = MagicMock() if checkpointer else None
    return graph


async def _drain(queue: EventQueue) -> list:
    """Drain all currently-enqueued events without blocking."""
    import asyncio as _asyncio

    out: list = []
    while True:
        try:
            out.append(await queue.dequeue_event(no_wait=True))
        except _asyncio.QueueEmpty:
            break
    return out


def _run_execute(executor: A2AAgentExecutor, context: RequestContext) -> list:
    """Drive execute() to completion and return all emitted events."""

    async def _run() -> list:
        queue = EventQueue()
        await executor.execute(context, queue)
        return await _drain(queue)

    return asyncio.run(_run())


def _run_cancel(executor: A2AAgentExecutor, context: RequestContext) -> list:
    async def _run() -> list:
        queue = EventQueue()
        await executor.cancel(context, queue)
        return await _drain(queue)

    return asyncio.run(_run())


# ---------------------------------------------------------------------------
# Part extraction
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_extract_text_only_message():
    cfg = _minimal_config()
    executor = A2AAgentExecutor(cfg, graph=_make_graph())
    messages, custom_inputs = executor._extract_request(_make_context(text="hello world"))
    assert messages == [{"role": "user", "content": "hello world"}]
    assert custom_inputs == {}


@pytest.mark.unit
def test_extract_data_only_message():
    cfg = _minimal_config()
    executor = A2AAgentExecutor(cfg, graph=_make_graph())
    messages, custom_inputs = executor._extract_request(
        _make_context(data={"decisions": [{"type": "approve"}]})
    )
    assert messages == []
    assert custom_inputs == {"decisions": [{"type": "approve"}]}


@pytest.mark.unit
def test_extract_mixed_message_merges_both():
    cfg = _minimal_config()
    executor = A2AAgentExecutor(cfg, graph=_make_graph())
    messages, custom_inputs = executor._extract_request(
        _make_context(text="please proceed", data={"flag": True})
    )
    assert messages == [{"role": "user", "content": "please proceed"}]
    assert custom_inputs == {"flag": True}


# ---------------------------------------------------------------------------
# OBO header propagation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_dao_context_picks_up_call_context_headers():
    cfg = _minimal_config()
    executor = A2AAgentExecutor(cfg, graph=_make_graph())
    context = _make_context(
        text="hi",
        headers={"x-forwarded-access-token": "tok-abc", "user-agent": "pytest"},
    )
    dao_ctx = executor._build_dao_context(context, custom_inputs={})
    assert dao_ctx.thread_id == "ctx-1"
    assert dao_ctx.headers == {
        "x-forwarded-access-token": "tok-abc",
        "user-agent": "pytest",
    }


@pytest.mark.unit
def test_build_dao_context_custom_inputs_override_call_context():
    cfg = _minimal_config()
    executor = A2AAgentExecutor(cfg, graph=_make_graph())
    context = _make_context(text="hi", headers={"x-fwd": "from-call-ctx"})
    dao_ctx = executor._build_dao_context(
        context,
        custom_inputs={"configurable": {"headers": {"x-fwd": "from-input"}}},
    )
    assert dao_ctx.headers["x-fwd"] == "from-input"


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_execute_happy_path_emits_completed():
    cfg = _minimal_config()
    graph = _make_graph(
        ainvoke_return={"messages": [AIMessage(content="hello back")]}
    )
    executor = A2AAgentExecutor(cfg, graph=graph)

    events = _run_execute(executor, _make_context(text="hi"))

    states = [
        e.status.state for e in events if isinstance(e, TaskStatusUpdateEvent)
    ]
    assert TaskState.submitted in states
    assert TaskState.working in states
    assert TaskState.completed in states
    assert states[-1] == TaskState.completed

    artifacts = [e for e in events if isinstance(e, TaskArtifactUpdateEvent)]
    assert len(artifacts) == 1
    assert any(
        getattr(p.root, "text", None) == "hello back"
        for p in artifacts[0].artifact.parts
    )


@pytest.mark.unit
def test_execute_emits_structured_response_as_data_part():
    cfg = _minimal_config()
    graph = _make_graph(
        ainvoke_return={
            "messages": [AIMessage(content="summary text")],
            "structured_response": {"total": 42, "currency": "USD"},
        }
    )
    executor = A2AAgentExecutor(cfg, graph=graph)
    events = _run_execute(executor, _make_context(text="hi"))

    artifacts = [e for e in events if isinstance(e, TaskArtifactUpdateEvent)]
    assert len(artifacts) == 1
    parts = artifacts[0].artifact.parts
    data_parts = [p for p in parts if isinstance(p.root, DataPart)]
    assert data_parts
    assert data_parts[0].root.data == {"total": 42, "currency": "USD"}


# ---------------------------------------------------------------------------
# HITL paths
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_execute_hitl_resume_with_datapart_decisions():
    cfg = _minimal_config()
    graph = _make_graph(
        ainvoke_return={"messages": [AIMessage(content="resumed and done")]}
    )
    executor = A2AAgentExecutor(cfg, graph=graph)

    decisions = [{"type": "approve"}]
    events = _run_execute(executor, _make_context(data={"decisions": decisions}))

    # Graph was invoked with a Command(resume=...) payload.
    call_args = graph.ainvoke.call_args
    assert call_args is not None
    first_arg = call_args.args[0]
    assert isinstance(first_arg, Command)
    assert first_arg.resume == {"decisions": decisions}

    states = [
        e.status.state for e in events if isinstance(e, TaskStatusUpdateEvent)
    ]
    assert states[-1] == TaskState.completed


@pytest.mark.unit
def test_execute_emits_input_required_when_response_has_interrupt():
    cfg = _minimal_config()

    class FakeInterrupt:
        value = {"question": "Approve action?"}

    graph = _make_graph(
        ainvoke_return={
            "messages": [AIMessage(content="...")],
            "__interrupt__": [FakeInterrupt()],
        }
    )
    executor = A2AAgentExecutor(cfg, graph=graph)
    events = _run_execute(executor, _make_context(text="run"))

    input_req_events = [
        e
        for e in events
        if isinstance(e, TaskStatusUpdateEvent)
        and e.status.state == TaskState.input_required
    ]
    assert input_req_events
    input_req = input_req_events[0]
    assert input_req.final is True

    msg = input_req.status.message
    assert msg is not None
    data_parts = [p for p in msg.parts if isinstance(p.root, DataPart)]
    assert data_parts
    assert "interrupts" in data_parts[0].root.data


@pytest.mark.unit
def test_execute_graph_interrupt_mid_invoke_emits_input_required():
    cfg = _minimal_config()

    class NoInterruptSnapshot:
        interrupts = ()

    class InterruptedSnapshot:
        interrupts = ({"value": {"question": "stop?"}},)

    graph = _make_graph(
        ainvoke_side_effect=GraphInterrupt("interrupted"),
        checkpointer=True,
    )
    # First aget_state (in decide_graph_turn pre-invoke) — non-interrupted, so
    # decide_graph_turn takes the fresh-invocation branch. Second aget_state
    # (inside the GraphInterrupt handler) — interrupted, so we emit
    # input-required with the interrupt payload.
    graph.aget_state = AsyncMock(
        side_effect=[NoInterruptSnapshot(), InterruptedSnapshot()]
    )
    executor = A2AAgentExecutor(cfg, graph=graph)
    events = _run_execute(executor, _make_context(text="run"))

    states = [
        e.status.state for e in events if isinstance(e, TaskStatusUpdateEvent)
    ]
    assert TaskState.input_required in states


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cancel_emits_canceled_state():
    cfg = _minimal_config()
    executor = A2AAgentExecutor(cfg, graph=_make_graph())

    events = _run_cancel(executor, _make_context(text=""))
    states = [
        e.status.state for e in events if isinstance(e, TaskStatusUpdateEvent)
    ]
    assert states == [TaskState.canceled]


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_execute_unexpected_exception_emits_failed():
    cfg = _minimal_config()
    graph = _make_graph(ainvoke_side_effect=RuntimeError("boom"))
    executor = A2AAgentExecutor(cfg, graph=graph)

    events = _run_execute(executor, _make_context(text="hi"))
    states = [
        e.status.state for e in events if isinstance(e, TaskStatusUpdateEvent)
    ]
    assert TaskState.failed in states
    assert states[-1] == TaskState.failed
