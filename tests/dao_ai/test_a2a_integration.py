"""End-to-end A2A integration tests over a FastAPI ``TestClient``.

These tests bring up the *full* A2A stack (executor + DefaultRequestHandler
+ A2AFastAPIApplication + routes) inside a single in-process FastAPI app
and exercise it via HTTP. Like the unit tests, the LangGraph is mocked,
so these tests run hermetically without a live workspace.

What we cover here that the per-module unit tests don't:

* GET ``/.well-known/agent-card.json`` returns a wire-format card.
* JSON-RPC ``message/send`` returns a Task envelope with the agent's
  output artifact attached.
* JSON-RPC ``tasks/get`` round-trips a previously-saved task.
* JSON-RPC ``message/stream`` returns SSE events including the terminal
  ``completed`` status.
* Both protocols register cleanly on the same FastAPI app: the agent-card
  route doesn't shadow any path the Responses handler would mount.
"""

import json
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from dao_ai.apps.a2a.agent_card import (
    DEFAULT_A2A_RPC_PATH,
    DEFAULT_AGENT_CARD_PATH,
    build_agent_card,
)
from dao_ai.apps.a2a.executor import A2AAgentExecutor
from dao_ai.apps.a2a.task_store import build_task_store
from dao_ai.config import (
    AgentModel,
    AppConfig,
    AppModel,
    DeploymentTarget,
    InferenceEndpointModel,
)


def _config() -> AppConfig:
    return AppConfig(
        app=AppModel(
            name="dao-ai-int-test",
            description="integration test agent",
            deployment_target=DeploymentTarget.APPS,
            agents=[
                AgentModel(
                    name="greeter",
                    description="says hi back",
                    model=InferenceEndpointModel(name="databricks-gpt-5-4-mini"),
                ),
            ],
        )
    )


def _build_app_with_stub_graph(graph_response: dict) -> FastAPI:
    """Build a FastAPI app that wires the full A2A stack with a stub graph."""
    from a2a.server.apps import A2AFastAPIApplication
    from a2a.server.request_handlers import DefaultRequestHandler

    cfg = _config()

    graph = MagicMock()
    graph.checkpointer = None
    graph.ainvoke = AsyncMock(return_value=graph_response)
    graph.aget_state = AsyncMock(return_value=None)

    executor = A2AAgentExecutor(cfg, graph=graph)
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=build_task_store(cfg),
    )
    application = A2AFastAPIApplication(
        agent_card=build_agent_card(cfg),
        http_handler=handler,
    )

    app = FastAPI()
    application.add_routes_to_app(
        app,
        agent_card_url=DEFAULT_AGENT_CARD_PATH,
        rpc_url=DEFAULT_A2A_RPC_PATH,
    )
    return app


def _jsonrpc(method: str, params: dict, request_id: str | None = None) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": request_id or uuid.uuid4().hex[:8],
        "method": method,
        "params": params,
    }


def _user_message(
    text: str,
    *,
    context_id: str | None = None,
    task_id: str | None = None,
) -> dict:
    """Build a JSON-RPC ``message/send`` params dict.

    For a NEW task, omit ``task_id`` — a2a-sdk's
    :class:`DefaultRequestHandler` rejects requests that reference a
    non-existent task. Only set ``task_id`` for HITL resume / continuation.
    """
    message: dict = {
        "messageId": uuid.uuid4().hex,
        "role": "user",
        "parts": [{"kind": "text", "text": text}],
    }
    if context_id is not None:
        message["contextId"] = context_id
    if task_id is not None:
        message["taskId"] = task_id
    return {"message": message}


@pytest.mark.integration
def test_get_agent_card_well_known_returns_card():
    app = _build_app_with_stub_graph(
        {"messages": [AIMessage(content="hello")]}
    )
    client = TestClient(app)

    resp = client.get(DEFAULT_AGENT_CARD_PATH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "dao-ai-int-test"
    assert body["url"].endswith(DEFAULT_A2A_RPC_PATH)
    assert isinstance(body["skills"], list) and body["skills"]
    assert body["capabilities"]["streaming"] is True


@pytest.mark.integration
def test_jsonrpc_message_send_returns_completed_task():
    app = _build_app_with_stub_graph(
        {"messages": [AIMessage(content="hello back from a2a")]}
    )
    client = TestClient(app)

    context_id = uuid.uuid4().hex
    payload = _jsonrpc(
        "message/send",
        _user_message("hi there", context_id=context_id),
    )
    resp = client.post(DEFAULT_A2A_RPC_PATH, json=payload)
    assert resp.status_code == 200
    body = resp.json()

    # JSON-RPC envelope
    assert body.get("jsonrpc") == "2.0"
    assert "error" not in body, body
    result = body["result"]

    # Result is a Task; check terminal state + artifact text
    assert result["kind"] == "task"
    assert result["status"]["state"] == "completed"
    assert result["contextId"] == context_id
    server_task_id = result["id"]
    assert server_task_id  # server generated

    artifacts = result.get("artifacts") or []
    assert artifacts, "expected at least one artifact"
    parts = artifacts[0]["parts"]
    text_parts = [p for p in parts if p.get("kind") == "text"]
    assert any(p["text"] == "hello back from a2a" for p in text_parts)


@pytest.mark.integration
def test_jsonrpc_tasks_get_roundtrips_saved_task():
    app = _build_app_with_stub_graph(
        {"messages": [AIMessage(content="persisted output")]}
    )
    client = TestClient(app)

    context_id = uuid.uuid4().hex
    send_resp = client.post(
        DEFAULT_A2A_RPC_PATH,
        json=_jsonrpc(
            "message/send",
            _user_message("hello", context_id=context_id),
        ),
    )
    assert send_resp.status_code == 200
    send_body = send_resp.json()
    assert "error" not in send_body, send_body
    server_task_id = send_body["result"]["id"]

    get_resp = client.post(
        DEFAULT_A2A_RPC_PATH,
        json=_jsonrpc("tasks/get", {"id": server_task_id}),
    )
    assert get_resp.status_code == 200
    body = get_resp.json()
    assert "error" not in body, body
    result = body["result"]
    assert result["id"] == server_task_id
    assert result["status"]["state"] == "completed"


@pytest.mark.integration
def test_jsonrpc_message_stream_emits_sse_completed():
    app = _build_app_with_stub_graph(
        {"messages": [AIMessage(content="streamed back")]}
    )
    client = TestClient(app)

    context_id = uuid.uuid4().hex
    payload = _jsonrpc(
        "message/stream",
        _user_message("hi", context_id=context_id),
    )

    with client.stream(
        "POST",
        DEFAULT_A2A_RPC_PATH,
        json=payload,
        headers={"Accept": "text/event-stream"},
    ) as resp:
        assert resp.status_code == 200
        ctype = resp.headers.get("content-type", "")
        assert "text/event-stream" in ctype

        data_events: list[dict] = []
        for raw in resp.iter_lines():
            if not raw or not raw.startswith("data:"):
                continue
            chunk = raw[len("data:") :].strip()
            if not chunk:
                continue
            data_events.append(json.loads(chunk))

    # Should see lifecycle: working → artifact-update → completed (terminal).
    assert data_events
    final_states = [
        ev.get("result", {}).get("status", {}).get("state")
        for ev in data_events
        if ev.get("result", {}).get("kind") == "status-update"
    ]
    assert "completed" in final_states


@pytest.mark.integration
def test_jsonrpc_hitl_resume_with_data_part_succeeds():
    """End-to-end: send a DataPart {'decisions': [...]} → graph receives Command(resume=...)."""
    app = _build_app_with_stub_graph(
        {"messages": [AIMessage(content="resumed and done")]}
    )
    client = TestClient(app)

    context_id = uuid.uuid4().hex
    payload = _jsonrpc(
        "message/send",
        {
            "message": {
                "messageId": uuid.uuid4().hex,
                "role": "user",
                "parts": [
                    {
                        "kind": "data",
                        "data": {"decisions": [{"type": "approve"}]},
                    }
                ],
                "contextId": context_id,
            }
        },
    )
    resp = client.post(DEFAULT_A2A_RPC_PATH, json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert "error" not in body, body
    assert body["result"]["status"]["state"] == "completed"
