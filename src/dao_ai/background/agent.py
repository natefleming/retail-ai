"""Background wrapper around a ``ResponsesAgent``.

This class composes a normal ``ResponsesAgent`` (typically the dao-ai
``LanggraphResponsesAgent``) and adds three operations, all delivered through
the same ``/invocations`` contract:

* **Kickoff** — ``request.background is True``. Returns immediately with a
  ``resp_...`` id and status ``in_progress``; the actual work runs as an
  ``asyncio.Task`` on the server event loop.
* **Retrieve** — ``custom_inputs.operation == "retrieve"``. Streams or returns
  the events written so far (optionally from a cursor).
* **Cancel** — ``custom_inputs.operation == "cancel"``. Best-effort same-pod
  task cancellation + sets ``status="cancelled"`` in the store.

If neither ``background`` nor ``operation`` is set, the call is delegated
unchanged to the wrapped agent — the synchronous path is a passthrough.

See ``examples/19_background_agents/deep_research.yaml`` for an
end-to-end example.
"""

from __future__ import annotations

import asyncio
import contextvars
import math
import threading
import traceback
import uuid
import warnings
from concurrent.futures import Future
from enum import Enum
from typing import Any, AsyncGenerator, Coroutine, Generator, Literal, Optional

import mlflow
from loguru import logger
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from dao_ai._tracing import to_thread_in_context
from dao_ai.background.store import (
    BackgroundStore,
    ResponseRecord,
    ResponseStatus,
)

# MLflow's ``Response`` validator only accepts
# {completed, failed, in_progress, incomplete} at the top level and warns on
# anything else. We intentionally surface ``cancelled`` in the top-level
# ``status`` field so existing clients (including the reference notebook)
# can treat it as a terminal state without having to read
# ``custom_outputs.background.status``. Mute the helper's warning so App /
# Model Serving logs stay clean; the authoritative status still lives in
# ``custom_outputs.background.status``.
warnings.filterwarnings(
    "ignore",
    message=r"Invalid status: cancelled\..*",
    category=UserWarning,
)

CUSTOM_INPUT_OPERATION = "operation"
CUSTOM_INPUT_RESPONSE_ID = "response_id"
CUSTOM_INPUT_CURSOR = "cursor"

OPERATION_RETRIEVE: Literal["retrieve"] = "retrieve"
OPERATION_CANCEL: Literal["cancel"] = "cancel"

# Clients (e.g. the Apps /v1/responses* routes) match on this to translate
# a "not found" response into an HTTP 404. Model Serving, which has no route
# layer, surfaces the same shape via the /invocations response body so both
# targets behave identically at the contract level.
ERROR_TYPE_NOT_FOUND: Literal["not_found"] = "not_found"

_RESPONSE_ID_PREFIX = "resp_"


class _Operation(str, Enum):
    """Internal enum used to dispatch a request to the right handler."""

    KICKOFF = "kickoff"
    RETRIEVE = "retrieve"
    CANCEL = "cancel"
    PASSTHROUGH = "passthrough"


class _BackgroundLoop:
    """Process-singleton background event loop for background tasks.

    Model Serving uses ``asyncio.run()`` per-request — when the request
    returns, the loop is torn down and any in-flight ``asyncio.create_task``
    gets cancelled before it can make progress. Databricks Apps doesn't have
    this problem (one persistent Uvicorn loop), but we want a uniform model.

    This class runs a dedicated daemon thread with its own event loop that
    persists for the lifetime of the Python process. Callers submit a
    coroutine via :meth:`submit` and get a ``concurrent.futures.Future`` plus
    the underlying ``asyncio.Task`` (so we can still cancel it).
    """

    _instance: Optional["_BackgroundLoop"] = None
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def instance(cls) -> "_BackgroundLoop":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="dao_ai-background",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait(timeout=5.0)

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()

    def submit(self, coro: Coroutine[Any, Any, Any]) -> tuple[Future, asyncio.Task]:
        """Schedule *coro* on the background loop.

        Returns a ``(concurrent.futures.Future, asyncio.Task)`` pair so the
        caller can wait on or cancel the work.
        """
        fut: Future = Future()
        task_holder: dict[str, asyncio.Task] = {}
        task_ready = threading.Event()
        # Capture the caller's contextvars so the asyncio.Task created in the
        # background loop inherits them (asyncio.Task copies the *current*
        # context at create_task() time). Without this the bg-loop task would
        # start with an empty context and MLflow's active-span ContextVar
        # would not propagate, producing orphan trace roots.
        caller_ctx = contextvars.copy_context()

        def _schedule() -> None:
            def _create_and_register() -> None:
                task = self._loop.create_task(coro)
                task_holder["task"] = task
                task.add_done_callback(
                    lambda t: (
                        fut.set_exception(t.exception())
                        if t.exception()
                        else fut.set_result(t.result() if not t.cancelled() else None)
                    )
                )
                task_ready.set()

            caller_ctx.run(_create_and_register)

        self._loop.call_soon_threadsafe(_schedule)
        task_ready.wait(timeout=5.0)
        return fut, task_holder["task"]


def _new_response_id() -> str:
    return f"{_RESPONSE_ID_PREFIX}{uuid.uuid4().hex}"


def _custom_inputs(request: ResponsesAgentRequest) -> dict[str, Any]:
    return request.custom_inputs or {}


def _thread_id(request: ResponsesAgentRequest) -> str:
    ci = _custom_inputs(request)
    configurable = ci.get("configurable") or {}
    tid = configurable.get("thread_id") or configurable.get("conversation_id")
    if tid:
        return str(tid)
    session = ci.get("session") or {}
    tid = session.get("conversation_id") or session.get("thread_id")
    if tid:
        return str(tid)
    return uuid.uuid4().hex


def _background_custom_outputs(
    record: ResponseRecord, *, extra: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """Build the ``custom_outputs.background`` payload for responses/events."""
    payload: dict[str, Any] = {
        "response_id": record.response_id,
        "status": record.status.value,
        "thread_id": record.thread_id,
        "error": record.error_json,
    }
    if extra:
        payload.update(extra)
    return {"background": payload}


def _status_response(
    record: ResponseRecord, output: Optional[list[dict[str, Any]]] = None
) -> ResponsesAgentResponse:
    # Pydantic will coerce dicts into OutputItem at validation time. The base
    # annotation on `output` is list[OutputItem]; the ignore quiets mypy.
    return ResponsesAgentResponse(
        id=record.response_id,
        status=record.status.value,
        output=output or [],  # type: ignore[arg-type]
        custom_outputs=_background_custom_outputs(record),
    )


def _status_stream_event(
    record: ResponseRecord,
    *,
    event_type: str = "response.in_progress",
) -> ResponsesAgentStreamEvent:
    # ResponsesAgentStreamEvent has ``extra="allow"`` so ``id`` passes through
    # to strict OpenAI clients that read ``event.id``. ``model_validate``
    # side-steps the fact that ``id`` isn't declared on the schema.
    return ResponsesAgentStreamEvent.model_validate(
        {
            "type": event_type,
            "id": record.response_id,
            "custom_outputs": _background_custom_outputs(record),
        }
    )


def _not_found_payload(response_id: str) -> dict[str, Any]:
    """``custom_outputs`` body signalling an unknown ``response_id``.

    Both deployment targets share this shape: Model Serving surfaces it in
    the ``/invocations`` response body (HTTP 200), while the Apps routes
    inspect it and return HTTP 404.
    """
    return {
        "background": {
            "response_id": response_id,
            "status": ResponseStatus.FAILED.value,
            "thread_id": None,
            "error": {
                "type": ERROR_TYPE_NOT_FOUND,
                "message": f"Unknown response_id: {response_id}",
            },
        }
    }


def _not_found_response(response_id: str) -> ResponsesAgentResponse:
    return ResponsesAgentResponse(
        id=response_id,
        status=ResponseStatus.FAILED.value,
        output=[],  # type: ignore[arg-type]
        custom_outputs=_not_found_payload(response_id),
    )


def _not_found_stream_event(response_id: str) -> ResponsesAgentStreamEvent:
    return ResponsesAgentStreamEvent.model_validate(
        {
            "type": "response.failed",
            "id": response_id,
            "custom_outputs": _not_found_payload(response_id),
        }
    )


def is_not_found_response(body: dict[str, Any]) -> bool:
    """Return True iff ``body`` is a structured "unknown response_id" response.

    Callers (Apps routes, test harnesses) use this to translate the body
    into an HTTP 404 or equivalent. Model Serving clients can use it to
    distinguish not-found from other failures.
    """
    lr = (body.get("custom_outputs") or {}).get("background") or {}
    err = lr.get("error") or {}
    return err.get("type") == ERROR_TYPE_NOT_FOUND


class BackgroundResponsesAgent(ResponsesAgent):
    """Compose a :class:`BackgroundStore` over an inner ``ResponsesAgent``.

    Both sync and async entry points are implemented (``predict`` /
    ``predict_stream`` delegate to the async versions via the same pattern
    as ``LanggraphResponsesAgent``).
    """

    def __init__(
        self,
        inner: ResponsesAgent,
        store: BackgroundStore,
        *,
        max_duration_seconds: int = 1800,
        poll_interval_seconds: float = 1.0,
        default_enabled: bool = False,
    ) -> None:
        self.inner = inner
        self.store = store
        self.max_duration_seconds = max_duration_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.default_enabled = default_enabled
        # Same-pod task registry for best-effort cancellation.
        self._tasks: dict[str, asyncio.Task] = {}

    # ------------------------------------------------------------------ sync

    def predict(  # type: ignore[override]
        self, request: ResponsesAgentRequest
    ) -> ResponsesAgentResponse:
        return asyncio.run(self.apredict(request))

    def predict_stream(  # type: ignore[override]
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        agen = self.apredict_stream(request)
        try:
            while True:
                try:
                    yield loop.run_until_complete(agen.__anext__())
                except StopAsyncIteration:
                    break
        finally:
            try:
                loop.run_until_complete(agen.aclose())
            except Exception as exc:  # noqa: BLE001 — best effort teardown
                logger.warning(
                    "Error closing background async generator", error=str(exc)
                )

    # ----------------------------------------------------------------- async

    @mlflow.trace(name="background.apredict", span_type="AGENT")
    async def apredict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        await self.store.ensure_schema()
        match self._classify(request):
            case _Operation.RETRIEVE:
                return await self._retrieve_non_stream(request)
            case _Operation.CANCEL:
                return await self._cancel(request)
            case _Operation.KICKOFF:
                return await self._kickoff_non_stream(request)
            case _Operation.PASSTHROUGH:
                return await self._delegate_apredict(request)

    @mlflow.trace(name="background.apredict_stream", span_type="AGENT")
    async def apredict_stream(
        self, request: ResponsesAgentRequest
    ) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
        await self.store.ensure_schema()
        match self._classify(request):
            case _Operation.RETRIEVE:
                async for ev in self._retrieve_stream(request):
                    yield ev
            case _Operation.CANCEL:
                response_id = self._required_response_id(request)
                record = await self._cancel_record(response_id)
                if record is None:
                    yield _not_found_stream_event(response_id)
                else:
                    yield _status_stream_event(
                        record, event_type="response.in_progress"
                    )
            case _Operation.KICKOFF:
                async for ev in self._kickoff_stream(request):
                    yield ev
            case _Operation.PASSTHROUGH:
                async for ev in self._delegate_apredict_stream(request):
                    yield ev

    # ------------------------------------------------------------ delegation

    async def _delegate_apredict(
        self, request: ResponsesAgentRequest
    ) -> ResponsesAgentResponse:
        if hasattr(self.inner, "apredict"):
            return await self.inner.apredict(request)  # type: ignore[attr-defined]
        # to_thread_in_context preserves contextvars so MLflow active-span
        # propagates into the sync predict() call.
        return await to_thread_in_context(self.inner.predict, request)

    async def _delegate_apredict_stream(
        self, request: ResponsesAgentRequest
    ) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
        if hasattr(self.inner, "apredict_stream"):
            async for ev in self.inner.apredict_stream(request):  # type: ignore[attr-defined]
                yield ev
            return

        def _sync_collect() -> list[ResponsesAgentStreamEvent]:
            return list(self.inner.predict_stream(request))

        # to_thread_in_context preserves contextvars so MLflow active-span
        # propagates into the sync predict_stream() call.
        for ev in await to_thread_in_context(_sync_collect):
            yield ev

    # --------------------------------------------------------------- helpers

    def _classify(self, request: ResponsesAgentRequest) -> _Operation:
        op: Literal["retrieve", "cancel"] | None = self._operation(request)
        if op == OPERATION_RETRIEVE:
            return _Operation.RETRIEVE
        if op == OPERATION_CANCEL:
            return _Operation.CANCEL
        if self._is_background(request):
            return _Operation.KICKOFF
        return _Operation.PASSTHROUGH

    def _operation(
        self, request: ResponsesAgentRequest
    ) -> Literal["retrieve", "cancel"] | None:
        op = _custom_inputs(request).get(CUSTOM_INPUT_OPERATION)
        if op == OPERATION_RETRIEVE:
            return OPERATION_RETRIEVE
        if op == OPERATION_CANCEL:
            return OPERATION_CANCEL
        return None

    def _is_background(self, request: ResponsesAgentRequest) -> bool:
        if request.background is True:
            return True
        if request.background is False:
            return False
        return self.default_enabled

    def _required_response_id(self, request: ResponsesAgentRequest) -> str:
        response_id = _custom_inputs(request).get(CUSTOM_INPUT_RESPONSE_ID)
        if not response_id:
            raise ValueError(
                "custom_inputs.response_id is required for retrieve/cancel operations"
            )
        return str(response_id)

    # -------------------------------------------------------------- kickoff

    async def _kickoff_non_stream(
        self, request: ResponsesAgentRequest
    ) -> ResponsesAgentResponse:
        response_id = _new_response_id()
        thread_id = _thread_id(request)
        await self.store.create(
            response_id=response_id,
            thread_id=thread_id,
            request=_serialize_request(request),
        )
        await self._spawn_background_task(response_id, request)
        record = await self.store.get(response_id)
        if record is None:
            raise RuntimeError(
                f"Response {response_id} vanished from the store immediately "
                "after create — check for concurrent admin deletion."
            )
        return _status_response(record)

    async def _kickoff_stream(
        self, request: ResponsesAgentRequest
    ) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
        response_id = _new_response_id()
        thread_id = _thread_id(request)
        await self.store.create(
            response_id=response_id,
            thread_id=thread_id,
            request=_serialize_request(request),
        )
        await self._spawn_background_task(response_id, request)
        record = await self.store.get(response_id)
        if record is None:
            raise RuntimeError(
                f"Response {response_id} vanished from the store immediately "
                "after create — check for concurrent admin deletion."
            )
        yield _status_stream_event(record, event_type="response.created")

    async def _spawn_background_task(
        self, response_id: str, request: ResponsesAgentRequest
    ) -> None:
        # Decouple the kicked-off request from the incoming one — we must not
        # leak our own custom_inputs (operation/response_id) into the inner run.
        inner_request = _clone_for_background(request)

        # Run on a persistent background event loop (in a dedicated daemon
        # thread) so the task survives the per-request asyncio.run() teardown
        # that Model Serving performs. Databricks Apps has a long-lived loop
        # and doesn't strictly need this, but using one mechanism keeps the
        # runtime identical across both deployment targets.
        _, task = _BackgroundLoop.instance().submit(
            self._run_background(response_id, inner_request)
        )
        # task is tied to the background loop, not the request loop. We keep
        # a reference here only to support same-pod cancellation.
        self._tasks[response_id] = task

        def _forget(_t: asyncio.Task, rid: str = response_id) -> None:
            self._tasks.pop(rid, None)

        task.add_done_callback(_forget)
        try:
            await self.store.set_agent_task_id(response_id, task.get_name())
        except Exception as exc:  # noqa: BLE001 — best effort, don't fail kickoff
            logger.warning(
                "Could not persist agent_task_id",
                response_id=response_id,
                error=str(exc),
            )

    async def _run_background(
        self, response_id: str, request: ResponsesAgentRequest
    ) -> None:
        logger.info("Background agent run starting", response_id=response_id)
        collected: list[dict[str, Any]] = []
        try:
            async with asyncio.timeout(self.max_duration_seconds):
                async for event in self._delegate_apredict_stream(request):
                    dumped = event.model_dump()
                    collected.append(dumped)
                    await self.store.append_event(response_id, dumped)

            # Aggregate stream chunks into final output items so non-streaming
            # retrieve callers get a populated ``output`` field. The reducer
            # accepts a list of dicts or events — we give it dicts.
            aggregated = ResponsesAgent.responses_agent_output_reducer(
                list(collected)  # invariant list; cast to satisfy type checker
            )
            items = aggregated.get("output") or []
            if items:
                await self.store.append_output(response_id, items)

            await self.store.set_status(response_id, ResponseStatus.COMPLETED)
            logger.info("Background agent run completed", response_id=response_id)
        except asyncio.CancelledError:
            await self.store.set_status(response_id, ResponseStatus.CANCELLED)
            logger.info("Background agent run cancelled", response_id=response_id)
            raise
        except asyncio.TimeoutError:
            await self.store.set_status(
                response_id,
                ResponseStatus.FAILED,
                error={
                    "reason": "timeout",
                    "max_duration_seconds": self.max_duration_seconds,
                },
            )
            logger.warning(
                "Background agent run exceeded max duration",
                response_id=response_id,
                max_duration_seconds=self.max_duration_seconds,
            )
        except Exception as exc:  # noqa: BLE001 — record and continue
            await self.store.set_status(
                response_id,
                ResponseStatus.FAILED,
                error={
                    "reason": "exception",
                    "type": exc.__class__.__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            logger.exception("Background agent run failed", response_id=response_id)
        finally:
            self._tasks.pop(response_id, None)

    # -------------------------------------------------------------- retrieve

    async def _retrieve_non_stream(
        self, request: ResponsesAgentRequest
    ) -> ResponsesAgentResponse:
        response_id = self._required_response_id(request)
        record = await self.store.get(response_id)
        if record is None:
            return _not_found_response(response_id)
        output: list[dict[str, Any]] = []
        if record.status == ResponseStatus.COMPLETED:
            output = await self.store.get_output(response_id)
        return _status_response(record, output=output)

    async def _retrieve_stream(
        self, request: ResponsesAgentRequest
    ) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
        response_id = self._required_response_id(request)
        cursor = int(_custom_inputs(request).get(CUSTOM_INPUT_CURSOR) or 0)

        # Bound the poll loop so a crashed writer can't hold the connection
        # open forever. Mirrors max_duration_seconds.
        max_iterations = max(
            1, math.ceil(self.max_duration_seconds / self.poll_interval_seconds)
        )

        for _ in range(max_iterations):
            record = await self.store.get(response_id)
            if record is None:
                yield _not_found_stream_event(response_id)
                return

            async for seq, event_payload in self.store.iter_events(
                response_id, cursor=cursor
            ):
                event_payload = dict(event_payload)
                event_payload.setdefault("type", "response.in_progress")
                event_payload.setdefault("id", response_id)
                event_payload["custom_outputs"] = _background_custom_outputs(
                    record, extra={"cursor": seq}
                )
                yield ResponsesAgentStreamEvent(**event_payload)
                cursor = seq

            if record.status.is_terminal:
                # ``response.completed`` in the Responses API requires a full
                # Response payload; we stream events sourced from a DB, so
                # emit a lightweight terminal marker whose authoritative
                # status lives in custom_outputs.background.status.
                yield _status_stream_event(
                    record,
                    event_type=(
                        "response.failed"
                        if record.status == ResponseStatus.FAILED
                        else "response.in_progress"
                    ),
                )
                return

            await asyncio.sleep(self.poll_interval_seconds)

        # Loop exhausted — the writer never reached a terminal state.
        await self.store.set_status(
            response_id,
            ResponseStatus.FAILED,
            error={
                "reason": "retrieve_poll_exhausted",
                "max_iterations": max_iterations,
                "poll_interval_seconds": self.poll_interval_seconds,
            },
        )
        final = await self.store.get(response_id)
        if final is None:
            raise RuntimeError(
                f"Response {response_id} vanished after set_status(FAILED) — "
                "check for concurrent admin deletion."
            )
        yield _status_stream_event(final, event_type="response.failed")

    # ----------------------------------------------------------------- cancel

    async def _cancel(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        response_id = self._required_response_id(request)
        record = await self._cancel_record(response_id)
        if record is None:
            return _not_found_response(response_id)
        return _status_response(record)

    async def _cancel_record(self, response_id: str) -> Optional[ResponseRecord]:
        # Short-circuit unknown ids so we don't UPDATE zero rows silently.
        existing = await self.store.get(response_id)
        if existing is None:
            return None
        task = self._tasks.get(response_id)
        if task is not None and not task.done():
            task.cancel()
        # Always reflect intent in the store — even if this pod doesn't own the task.
        await self.store.mark_cancelled(response_id)
        record = await self.store.get(response_id)
        if record is None:
            # Raced with a delete between mark_cancelled and the re-read; treat
            # as not-found so callers return the consistent structured error.
            return None
        return record


def _serialize_request(request: ResponsesAgentRequest) -> dict[str, Any]:
    """Best-effort JSON-safe snapshot of the original request for audit/debug."""
    try:
        return request.model_dump(mode="json")
    except Exception:  # noqa: BLE001 — the request may contain non-JSON types
        return {"error": "failed_to_serialize_request"}


def _clone_for_background(request: ResponsesAgentRequest) -> ResponsesAgentRequest:
    """Return a shallow copy with our background markers removed."""
    clone = request.model_copy(deep=True)
    if clone.custom_inputs:
        ci = dict(clone.custom_inputs)
        ci.pop(CUSTOM_INPUT_OPERATION, None)
        ci.pop(CUSTOM_INPUT_RESPONSE_ID, None)
        ci.pop(CUSTOM_INPUT_CURSOR, None)
        clone.custom_inputs = ci
    clone.background = False
    return clone
