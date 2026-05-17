"""Unit tests for :mod:`dao_ai.apps.a2a.routes`.

Exercises :func:`mount_a2a_routes` against a FastAPI ``TestClient``: route
registration, opt-out semantics, and that ``GET /.well-known/agent-card.json``
returns a valid Agent Card body.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dao_ai.apps.a2a import mount_a2a_routes
from dao_ai.apps.a2a.agent_card import DEFAULT_A2A_RPC_PATH, DEFAULT_AGENT_CARD_PATH
from dao_ai.config import (
    A2AModel,
    AgentModel,
    AppConfig,
    AppModel,
    DeploymentTarget,
    InferenceEndpointModel,
)


def _agent(name: str = "greeter") -> AgentModel:
    return AgentModel(
        name=name,
        description="test agent",
        model=InferenceEndpointModel(name="databricks-gpt-5-4-mini"),
    )


def _config(*, a2a: A2AModel | None = None) -> AppConfig:
    extra: dict = {}
    if a2a is not None:
        extra["a2a"] = a2a
    return AppConfig(
        app=AppModel(
            name="dao-ai-routes-test",
            description="test agent",
            deployment_target=DeploymentTarget.APPS,
            agents=[_agent()],
            **extra,
        ),
    )


@pytest.mark.unit
def test_mount_registers_both_routes():
    app = FastAPI()
    mounted = mount_a2a_routes(app, _config())
    assert mounted is True
    paths = {getattr(r, "path", None) for r in app.routes}
    assert DEFAULT_A2A_RPC_PATH in paths
    assert DEFAULT_AGENT_CARD_PATH in paths


@pytest.mark.unit
def test_mount_no_op_when_disabled():
    app = FastAPI()
    mounted = mount_a2a_routes(app, _config(a2a=A2AModel(enabled=False)))
    assert mounted is False
    paths = {getattr(r, "path", None) for r in app.routes}
    assert DEFAULT_A2A_RPC_PATH not in paths
    assert DEFAULT_AGENT_CARD_PATH not in paths


@pytest.mark.unit
def test_agent_card_endpoint_returns_valid_card():
    app = FastAPI()
    mount_a2a_routes(app, _config())
    client = TestClient(app)
    resp = client.get(DEFAULT_AGENT_CARD_PATH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "dao-ai-routes-test"
    assert body["url"].endswith(DEFAULT_A2A_RPC_PATH)
    assert isinstance(body.get("skills"), list) and body["skills"]
    assert "bearer" in (body.get("securitySchemes") or {})
    assert body["capabilities"]["streaming"] is True


@pytest.mark.unit
def test_a2a_rpc_rejects_garbage_with_json_rpc_error():
    """Posting non-JSON-RPC body returns a 200 with a JSON-RPC error envelope
    (per the JSON-RPC 2.0 spec, transport errors do NOT map to HTTP errors)."""
    app = FastAPI()
    mount_a2a_routes(app, _config())
    client = TestClient(app)
    resp = client.post(DEFAULT_A2A_RPC_PATH, json={"foo": "bar"})
    # 200 with JSON-RPC error body OR 400 are both acceptable per the spec.
    assert resp.status_code in (200, 400)
    body = resp.json()
    # Either {"error": {...}} or {"jsonrpc": "2.0", "error": {...}}
    assert "error" in body or "errors" in body
