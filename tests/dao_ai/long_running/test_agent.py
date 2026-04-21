"""Unit tests for LongRunningResponsesAgent routing and lifecycle.

The store is mocked so these tests don't require a live Lakebase/Postgres
instance. End-to-end coverage against real endpoints lives in
``notebooks/14_long_running_agents_demo.py``.
"""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from dao_ai.long_running.agent import LongRunningResponsesAgent
from dao_ai.long_running.store import ResponseRecord, ResponseStatus


def _run(coro):
    return asyncio.run(coro)


def _make_record(
    response_id: str = "resp_x",
    status: ResponseStatus = ResponseStatus.IN_PROGRESS,
    error: dict | None = None,
) -> ResponseRecord:
    import datetime as _dt

    now = _dt.datetime.now(tz=_dt.timezone.utc)
    return ResponseRecord(
        response_id=response_id,
        thread_id="thread_1",
        agent_task_id=None,
        status=status,
        request_json=None,
        error_json=error,
        created_at=now,
        updated_at=now,
        completed_at=None,
    )


class _FakeInner:
    """Inner ResponsesAgent double with an apredict/apredict_stream contract."""

    def __init__(self, events=None, final_response=None):
        self._events = events or []
        self._final_response = final_response or ResponsesAgentResponse(output=[])
        self.apredict = AsyncMock(return_value=self._final_response)

    async def apredict_stream(
        self, request
    ) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
        for ev in self._events:
            yield ev


@pytest.fixture
def store():
    m = MagicMock()
    m.ensure_schema = AsyncMock()
    m.create = AsyncMock()
    m.get = AsyncMock(return_value=_make_record())
    m.set_status = AsyncMock()
    m.set_agent_task_id = AsyncMock()
    m.mark_cancelled = AsyncMock()
    m.append_event = AsyncMock(return_value=0)
    m.append_output = AsyncMock()
    m.get_output = AsyncMock(return_value=[])

    async def _iter(*_args, **_kwargs):
        if False:
            yield None

    m.iter_events = _iter
    return m


# ---------------------------------------------------------------------- passthrough


def test_passthrough_non_background_delegates_to_inner(store):
    inner = _FakeInner()
    agent = LongRunningResponsesAgent(inner=inner, store=store)

    request = ResponsesAgentRequest(input=[])
    response = _run(agent.apredict(request))

    assert response is inner.apredict.return_value
    store.create.assert_not_called()
    store.ensure_schema.assert_awaited_once()


# ---------------------------------------------------------------------- kickoff


def test_kickoff_returns_in_progress_and_creates_row(store):
    # threading.Event (not asyncio.Event) because the background task runs
    # on a different event loop than this test's.
    import threading as _threading

    started = _threading.Event()

    class SlowInner:
        async def apredict_stream(self, request):
            started.set()
            await asyncio.sleep(5)  # noqa: ASYNC110 — deliberately blocks
            if False:
                yield None

    inner = SlowInner()
    agent = LongRunningResponsesAgent(inner=inner, store=store)

    request = ResponsesAgentRequest(
        input=[],
        background=True,
        custom_inputs={"configurable": {"thread_id": "thread_1"}},
    )

    response = _run(agent.apredict(request))

    # Wait for the background task to have started (on its dedicated loop).
    assert started.wait(timeout=2), "background task never started"
    registered = list(agent._tasks.values())
    # Cancel dangling tasks across the thread boundary.
    for t in registered:
        t.get_loop().call_soon_threadsafe(t.cancel)

    # Top-level fields (OpenAI Responses API compatibility).
    assert response.id is not None and response.id.startswith("resp_")
    assert response.status == "in_progress"
    # Extended fields still in custom_outputs.
    info = response.custom_outputs["long_running"]
    assert info["status"] == "in_progress"
    assert info["response_id"] == response.id
    store.create.assert_awaited()
    assert len(registered) == 1


def test_clone_for_background_strips_long_running_markers():
    """_clone_for_background strips markers and forces background=False."""
    from dao_ai.long_running.agent import _clone_for_background

    request = ResponsesAgentRequest(
        input=[],
        background=True,
        custom_inputs={
            "configurable": {"thread_id": "t"},
            "operation": "retrieve",
            "response_id": "resp_leaked",
            "cursor": 5,
        },
    )
    cloned = _clone_for_background(request)

    assert cloned.background is False
    ci = cloned.custom_inputs or {}
    assert "operation" not in ci
    assert "response_id" not in ci
    assert "cursor" not in ci
    assert ci.get("configurable") == {"thread_id": "t"}


def test_kickoff_passes_cleaned_request_to_inner(store):
    """Inner agent sees a request with long-running markers removed."""
    seen_requests: list = []

    class RecordingInner:
        async def apredict_stream(self, request):
            seen_requests.append(request)
            yield ResponsesAgentStreamEvent(type="response.in_progress")

    agent = LongRunningResponsesAgent(inner=RecordingInner(), store=store)
    request = ResponsesAgentRequest(
        input=[],
        background=True,
        custom_inputs={"configurable": {"thread_id": "t"}},
    )

    _run(agent.apredict(request))

    # Background task runs on _BackgroundLoop (dedicated thread). Poll for up
    # to 2s waiting for it to call apredict_stream.
    import time as _time

    deadline = _time.monotonic() + 2.0
    while not seen_requests and _time.monotonic() < deadline:
        _time.sleep(0.02)

    assert seen_requests, "inner apredict_stream was not invoked"
    inner = seen_requests[0]
    assert inner.background is False
    # configurable should be preserved so the inner agent still gets thread_id etc.
    assert (inner.custom_inputs or {}).get("configurable") == {"thread_id": "t"}


# ---------------------------------------------------------------------- retrieve


def test_retrieve_returns_in_progress_when_not_terminal(store):
    store.get.return_value = _make_record(status=ResponseStatus.IN_PROGRESS)
    agent = LongRunningResponsesAgent(inner=_FakeInner(), store=store)

    request = ResponsesAgentRequest(
        input=[],
        custom_inputs={"operation": "retrieve", "response_id": "resp_x"},
    )

    response = _run(agent.apredict(request))

    assert response.custom_outputs["long_running"]["status"] == "in_progress"
    store.get_output.assert_not_called()


def test_retrieve_returns_output_when_completed(store):
    store.get.return_value = _make_record(status=ResponseStatus.COMPLETED)
    store.get_output.return_value = [
        {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "done"}],
            "status": "completed",
        }
    ]
    agent = LongRunningResponsesAgent(inner=_FakeInner(), store=store)

    request = ResponsesAgentRequest(
        input=[],
        custom_inputs={"operation": "retrieve", "response_id": "resp_x"},
    )

    response = _run(agent.apredict(request))

    assert response.id == "resp_x"
    assert response.status == "completed"
    assert len(response.output) == 1
    assert response.output[0].id == "msg_1"
    assert response.custom_outputs["long_running"]["status"] == "completed"


def test_retrieve_unknown_id_raises(store):
    store.get.return_value = None
    agent = LongRunningResponsesAgent(inner=_FakeInner(), store=store)

    request = ResponsesAgentRequest(
        input=[],
        custom_inputs={"operation": "retrieve", "response_id": "resp_missing"},
    )

    with pytest.raises(KeyError):
        _run(agent.apredict(request))


# ---------------------------------------------------------------------- cancel


def test_cancel_marks_cancelled_in_store(store):
    store.get.return_value = _make_record(status=ResponseStatus.CANCELLED)
    agent = LongRunningResponsesAgent(inner=_FakeInner(), store=store)

    async def _dummy():
        await asyncio.sleep(10)

    # Register a real running task that should be cancelled.
    async def _body():
        task = asyncio.create_task(_dummy())
        agent._tasks["resp_x"] = task
        response = await agent.apredict(
            ResponsesAgentRequest(
                input=[],
                custom_inputs={"operation": "cancel", "response_id": "resp_x"},
            )
        )
        # Let the cancellation propagate.
        try:
            await task
        except asyncio.CancelledError:
            pass
        return response, task

    response, task = _run(_body())

    assert response.custom_outputs["long_running"]["status"] == "cancelled"
    store.mark_cancelled.assert_awaited_once_with("resp_x")
    assert task.cancelled() or task.done()


# ---------------------------------------------------------------------- stream


def test_stream_kickoff_yields_response_created(store):
    agent = LongRunningResponsesAgent(inner=_FakeInner(), store=store)

    request = ResponsesAgentRequest(
        input=[], background=True, custom_inputs={"configurable": {"thread_id": "t"}}
    )

    async def _collect():
        events = []
        async for ev in agent.apredict_stream(request):
            events.append(ev)
        return events

    events = _run(_collect())
    assert len(events) == 1
    assert events[0].type == "response.created"
    # Top-level id must be present for strict OpenAI clients.
    assert getattr(events[0], "id", None) is not None


def test_stream_retrieve_polls_until_terminal(store):
    # First get => in_progress, second => completed
    store.get.side_effect = [
        _make_record(status=ResponseStatus.IN_PROGRESS),
        _make_record(status=ResponseStatus.IN_PROGRESS),
        _make_record(status=ResponseStatus.COMPLETED),
    ]

    async def _first_iter(*_a, **_k):
        yield 0, {"type": "response.in_progress"}

    async def _second_iter(*_a, **_k):
        yield 1, {"type": "response.in_progress"}

    async def _empty_iter(*_a, **_k):
        if False:
            yield None

    iters = iter([_first_iter, _empty_iter, _second_iter])

    def _dispatcher(*args, **kwargs):
        return next(iters)(*args, **kwargs)

    store.iter_events = _dispatcher

    agent = LongRunningResponsesAgent(
        inner=_FakeInner(), store=store, poll_interval_seconds=0.01
    )

    request = ResponsesAgentRequest(
        input=[],
        custom_inputs={"operation": "retrieve", "response_id": "resp_x", "cursor": 0},
    )

    async def _collect():
        events = []
        async for ev in agent.apredict_stream(request):
            events.append(ev)
            if len(events) > 10:
                break
        return events

    events = _run(_collect())
    types = [ev.type for ev in events]
    assert "response.in_progress" in types
    # Terminal marker: authoritative status lives in custom_outputs.long_running.status
    terminal_status = (
        events[-1].custom_outputs.get("long_running", {}).get("status")
        if events[-1].custom_outputs
        else None
    )
    assert terminal_status == "completed"


# ---------------------------------------------------------------------- aggregation


def test_run_background_persists_aggregated_output_items(store):
    """After the inner stream ends, output_item.done events become stored items."""
    item = {
        "type": "message",
        "id": "msg_final",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "done"}],
        "status": "completed",
    }

    class OneItemInner:
        async def apredict_stream(self, request):
            yield ResponsesAgentStreamEvent(type="response.output_item.done", item=item)

    agent = LongRunningResponsesAgent(inner=OneItemInner(), store=store)

    request = ResponsesAgentRequest(
        input=[],
        background=True,
        custom_inputs={"configurable": {"thread_id": "t"}},
    )

    _run(agent.apredict(request))

    # Background task runs on _BackgroundLoop (dedicated thread). Poll for
    # the aggregation + persistence to complete.
    import time as _time

    deadline = _time.monotonic() + 3.0
    while store.append_output.await_count == 0 and _time.monotonic() < deadline:
        _time.sleep(0.02)

    # append_output should have been called once with the aggregated items.
    assert store.append_output.await_count == 1
    call_args = store.append_output.await_args
    stored_items = (
        call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs["items"]
    )
    assert stored_items == [item]
    # And status should have been set to COMPLETED after that.
    # (We check the last set_status call — in happy path, only one is made.)
    statuses = [c.args[1] for c in store.set_status.await_args_list]
    assert ResponseStatus.COMPLETED in statuses


def test_retrieve_stream_bounded_iterations(store):
    """If the writer never reaches terminal, retrieve stream must fail the response."""
    # Always in_progress — no terminal state will ever be observed.
    store.get.return_value = _make_record(status=ResponseStatus.IN_PROGRESS)

    async def _empty_iter(*_a, **_k):
        if False:
            yield None

    store.iter_events = _empty_iter

    agent = LongRunningResponsesAgent(
        inner=_FakeInner(),
        store=store,
        # Force max_iterations = 2 so the bounded loop exits quickly.
        max_duration_seconds=1,
        poll_interval_seconds=0.5,
    )

    request = ResponsesAgentRequest(
        input=[],
        custom_inputs={"operation": "retrieve", "response_id": "resp_x"},
    )

    async def _collect():
        events = []
        async for ev in agent.apredict_stream(request):
            events.append(ev)
            if len(events) > 5:
                break
        return events

    events = _run(_collect())

    # Loop should have emitted a terminal failed event.
    assert events[-1].type == "response.failed"
    # And persisted the failure with the expected reason.
    assert store.set_status.await_args_list, "set_status was not called"
    last_call = store.set_status.await_args_list[-1]
    assert last_call.args[1] == ResponseStatus.FAILED
    assert last_call.kwargs["error"]["reason"] == "retrieve_poll_exhausted"
