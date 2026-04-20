"""Long-running wrapper around a ``ResponsesAgent``.

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

See ``config/examples/19_long_running_agents/deep_research.yaml`` for an
end-to-end example.
"""

from __future__ import annotations

import asyncio
import traceback
import uuid
from typing import Any, AsyncGenerator, Generator, Optional

import mlflow
from loguru import logger
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from dao_ai.long_running.store import (
    LongRunningStore,
    ResponseRecord,
    ResponseStatus,
)

CUSTOM_INPUT_OPERATION = "operation"
CUSTOM_INPUT_RESPONSE_ID = "response_id"
CUSTOM_INPUT_CURSOR = "cursor"

OPERATION_RETRIEVE = "retrieve"
OPERATION_CANCEL = "cancel"

_RESPONSE_ID_PREFIX = "resp_"


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


def _status_response(
    record: ResponseRecord, output: Optional[list[dict[str, Any]]] = None
) -> ResponsesAgentResponse:
    return ResponsesAgentResponse(
        output=output or [],  # type: ignore[arg-type]
        custom_outputs={
            "long_running": {
                "response_id": record.response_id,
                "status": record.status.value,
                "thread_id": record.thread_id,
                "error": record.error_json,
            }
        },
    )


def _status_stream_event(
    record: ResponseRecord,
    *,
    event_type: str = "response.in_progress",
) -> ResponsesAgentStreamEvent:
    return ResponsesAgentStreamEvent(
        type=event_type,
        custom_outputs={
            "long_running": {
                "response_id": record.response_id,
                "status": record.status.value,
                "thread_id": record.thread_id,
                "error": record.error_json,
            }
        },
    )


class LongRunningResponsesAgent(ResponsesAgent):
    """Compose a :class:`LongRunningStore` over an inner ``ResponsesAgent``.

    Both sync and async entry points are implemented (``predict`` /
    ``predict_stream`` delegate to the async versions via the same pattern
    as ``LanggraphResponsesAgent``).
    """

    def __init__(
        self,
        inner: ResponsesAgent,
        store: LongRunningStore,
        *,
        max_duration_seconds: int = 1800,
        poll_interval_seconds: float = 1.0,
        default_background: bool = False,
    ) -> None:
        self.inner = inner
        self.store = store
        self.max_duration_seconds = max_duration_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.default_background = default_background
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
                    "Error closing long-running async generator", error=str(exc)
                )

    # ----------------------------------------------------------------- async

    @mlflow.trace(name="long_running.apredict", span_type="AGENT")
    async def apredict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        await self.store.ensure_schema()
        op = self._operation(request)

        if op == OPERATION_RETRIEVE:
            return await self._retrieve_non_stream(request)
        if op == OPERATION_CANCEL:
            return await self._cancel(request)
        if self._is_background(request):
            return await self._kickoff_non_stream(request)

        return await self._delegate_apredict(request)

    @mlflow.trace(name="long_running.apredict_stream", span_type="AGENT")
    async def apredict_stream(
        self, request: ResponsesAgentRequest
    ) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
        await self.store.ensure_schema()
        op = self._operation(request)

        if op == OPERATION_RETRIEVE:
            async for ev in self._retrieve_stream(request):
                yield ev
            return
        if op == OPERATION_CANCEL:
            record = await self._cancel_record(request)
            yield _status_stream_event(record, event_type="response.in_progress")
            return
        if self._is_background(request):
            async for ev in self._kickoff_stream(request):
                yield ev
            return

        async for ev in self._delegate_apredict_stream(request):
            yield ev

    # ------------------------------------------------------------ delegation

    async def _delegate_apredict(
        self, request: ResponsesAgentRequest
    ) -> ResponsesAgentResponse:
        if hasattr(self.inner, "apredict"):
            return await self.inner.apredict(request)  # type: ignore[attr-defined]
        return await asyncio.to_thread(self.inner.predict, request)

    async def _delegate_apredict_stream(
        self, request: ResponsesAgentRequest
    ) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
        if hasattr(self.inner, "apredict_stream"):
            async for ev in self.inner.apredict_stream(request):  # type: ignore[attr-defined]
                yield ev
            return

        def _sync_collect() -> list[ResponsesAgentStreamEvent]:
            return list(self.inner.predict_stream(request))

        for ev in await asyncio.to_thread(_sync_collect):
            yield ev

    # --------------------------------------------------------------- helpers

    def _operation(self, request: ResponsesAgentRequest) -> Optional[str]:
        op = _custom_inputs(request).get(CUSTOM_INPUT_OPERATION)
        return str(op) if op else None

    def _is_background(self, request: ResponsesAgentRequest) -> bool:
        if request.background is True:
            return True
        if request.background is False:
            return False
        return self.default_background

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
        assert record is not None
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
        assert record is not None
        yield _status_stream_event(record, event_type="response.created")

    async def _spawn_background_task(
        self, response_id: str, request: ResponsesAgentRequest
    ) -> None:
        # Decouple the kicked-off request from the incoming one — we must not
        # leak our own custom_inputs (operation/response_id) into the inner run.
        inner_request = _clone_for_background(request)
        task = asyncio.create_task(
            self._run_background(response_id, inner_request),
            name=f"long_running:{response_id}",
        )
        self._tasks[response_id] = task
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
        try:

            async def _drive() -> None:
                async for event in self._delegate_apredict_stream(request):
                    await self.store.append_event(response_id, event.model_dump())

            await asyncio.wait_for(_drive(), timeout=self.max_duration_seconds)
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
            raise KeyError(f"Unknown response_id: {response_id}")
        output: list[dict[str, Any]] = []
        if record.status == ResponseStatus.COMPLETED:
            # Prefer stored final items; if only stream events exist, return those.
            output = await self.store.get_output(response_id)
        return _status_response(record, output=output)

    async def _retrieve_stream(
        self, request: ResponsesAgentRequest
    ) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
        response_id = self._required_response_id(request)
        cursor = int(_custom_inputs(request).get(CUSTOM_INPUT_CURSOR) or 0)

        while True:
            record = await self.store.get(response_id)
            if record is None:
                raise KeyError(f"Unknown response_id: {response_id}")

            async for seq, event_payload in self.store.iter_events(
                response_id, cursor=cursor
            ):
                event_payload = dict(event_payload)
                event_payload.setdefault("type", "response.in_progress")
                custom_outputs = dict(event_payload.get("custom_outputs") or {})
                custom_outputs.setdefault(
                    "long_running",
                    {
                        "response_id": response_id,
                        "status": record.status.value,
                        "cursor": seq,
                    },
                )
                event_payload["custom_outputs"] = custom_outputs
                yield ResponsesAgentStreamEvent(**event_payload)
                cursor = seq

            if record.status.is_terminal:
                # ``response.completed`` in the Responses API requires a full
                # Response payload. We're streaming events sourced from a DB,
                # not a live model call, so emit a lightweight terminal marker
                # whose authoritative status lives in custom_outputs.
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

    # ----------------------------------------------------------------- cancel

    async def _cancel(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        record = await self._cancel_record(request)
        return _status_response(record)

    async def _cancel_record(self, request: ResponsesAgentRequest) -> ResponseRecord:
        response_id = self._required_response_id(request)
        task = self._tasks.get(response_id)
        if task is not None and not task.done():
            task.cancel()
        # Always reflect intent in the store — even if this pod doesn't own the task.
        await self.store.mark_cancelled(response_id)
        record = await self.store.get(response_id)
        if record is None:
            raise KeyError(f"Unknown response_id: {response_id}")
        return record


def _serialize_request(request: ResponsesAgentRequest) -> dict[str, Any]:
    """Best-effort JSON-safe snapshot of the original request for audit/debug."""
    try:
        return request.model_dump(mode="json")
    except Exception:  # noqa: BLE001 — the request may contain non-JSON types
        return {"error": "failed_to_serialize_request"}


def _clone_for_background(request: ResponsesAgentRequest) -> ResponsesAgentRequest:
    """Return a shallow copy with our long-running markers removed."""
    clone = request.model_copy(deep=True)
    if clone.custom_inputs:
        ci = dict(clone.custom_inputs)
        ci.pop(CUSTOM_INPUT_OPERATION, None)
        ci.pop(CUSTOM_INPUT_RESPONSE_ID, None)
        ci.pop(CUSTOM_INPUT_CURSOR, None)
        clone.custom_inputs = ci
    clone.background = False
    return clone
