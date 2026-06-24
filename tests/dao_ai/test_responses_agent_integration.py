"""End-to-end integration tests for create_responses_agent_tool over httpx.

These tests bring up a minimal FastAPI app exposing the OpenAI
``POST /v1/responses`` route, route DatabricksOpenAI's HTTP traffic at the
test server via an ``httpx.MockTransport``, and exercise the full factory →
``DatabricksOpenAI.responses.create`` → wire → assistant-text extraction
loop. The LangGraph integration is not exercised here (covered separately).
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from dao_ai.config import DatabricksAppModel
from dao_ai.tools.responses_agent import create_responses_agent_tool

# ---------------------------------------------------------------------------
# Test fixture — a FastAPI app stubbing the OpenAI Responses route
# ---------------------------------------------------------------------------


def _build_responses_stub(
    echo_prefix: str = "echo:",
) -> tuple[TestClient, list[dict[str, Any]]]:
    """Spin up a FastAPI app that mimics an OpenAI Responses endpoint.

    Returns the TestClient and a list that captures every incoming request
    body (so tests can assert on the wire payload).
    """
    captured_requests: list[dict[str, Any]] = []
    app: FastAPI = FastAPI()

    @app.post("/responses")
    async def responses_endpoint(request: Request) -> dict[str, Any]:
        body: dict[str, Any] = await request.json()
        captured_requests.append(body)

        # Echo the user's last message as the assistant's reply.
        user_text: str = ""
        for item in body.get("input", []):
            if isinstance(item, dict) and item.get("role") == "user":
                user_text = item.get("content", "")
                break

        return {
            "id": "resp_test_123",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "output": [
                {
                    "id": "msg_test_1",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": f"{echo_prefix}{user_text}",
                            "annotations": [],
                        }
                    ],
                }
            ],
            "model": body.get("model"),
            "parallel_tool_calls": True,
            "usage": {
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
            },
        }

    return TestClient(app), captured_requests


@pytest.fixture
def responses_stub() -> tuple[TestClient, list[dict[str, Any]]]:
    return _build_responses_stub()


# ---------------------------------------------------------------------------
# Stub DatabricksOpenAI to route its HTTP client at the FastAPI TestClient
# ---------------------------------------------------------------------------


class _PatchedDatabricksOpenAI:
    """Drop-in for DatabricksOpenAI that wraps a real OpenAI client whose
    HTTP transport is bound to the FastAPI TestClient via httpx.MockTransport."""

    def __init__(self, test_client: TestClient) -> None:
        from openai import OpenAI

        # httpx.MockTransport intercepts every request the OpenAI client
        # makes and forwards it to the FastAPI app via TestClient.
        def _handle(request: httpx.Request) -> httpx.Response:
            method: str = request.method
            url_path: str = request.url.path
            body: bytes = request.read()
            resp = test_client.request(
                method=method,
                url=url_path,
                content=body,
                headers=dict(request.headers),
            )
            return httpx.Response(
                status_code=resp.status_code,
                headers=dict(resp.headers),
                content=resp.content,
            )

        transport: httpx.MockTransport = httpx.MockTransport(_handle)
        http_client: httpx.Client = httpx.Client(transport=transport)
        self._openai = OpenAI(
            api_key="test-token",
            base_url="https://stub.databricks.test",
            http_client=http_client,
        )

    @property
    def responses(self) -> Any:
        return self._openai.responses


@pytest.fixture
def patched_databricks_openai(
    monkeypatch: pytest.MonkeyPatch,
    responses_stub: tuple[TestClient, list[dict[str, Any]]],
) -> tuple[TestClient, list[dict[str, Any]]]:
    """Replace DatabricksOpenAI in responses_agent.py with the stub."""
    test_client, captured = responses_stub

    def _factory(**_kwargs: Any) -> _PatchedDatabricksOpenAI:
        return _PatchedDatabricksOpenAI(test_client)

    monkeypatch.setattr(
        "dao_ai.tools.responses_agent.DatabricksOpenAI",
        _factory,
    )
    # Bypass real workspace-client construction.
    monkeypatch.setattr(
        DatabricksAppModel,
        "workspace_client_from",
        lambda self, context: MagicMock(name="WorkspaceClient"),
    )
    return test_client, captured


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


def test_responses_agent_end_to_end_through_httpx_transport(
    patched_databricks_openai: tuple[TestClient, list[dict[str, Any]]],
) -> None:
    """Full path: factory → DatabricksOpenAI → httpx → FastAPI stub → reply."""
    _test_client, captured = patched_databricks_openai

    tool = create_responses_agent_tool(DatabricksAppModel(name="my-app"))
    result: str = asyncio.run(tool.ainvoke({"prompt": "Hello"}))

    assert result == "echo:Hello"
    assert len(captured) == 1
    assert captured[0]["model"] == "apps/my-app"
    assert captured[0]["input"] == [{"role": "user", "content": "Hello"}]


def test_responses_agent_extracts_output_text_from_responses_envelope(
    patched_databricks_openai: tuple[TestClient, list[dict[str, Any]]],
) -> None:
    """OpenAI's response.output_text walks output[*].content[*].text — this
    test confirms the dao-ai factory consumes the envelope correctly."""
    _test_client, _captured = patched_databricks_openai

    tool = create_responses_agent_tool(DatabricksAppModel(name="app-x"))
    result: str = asyncio.run(tool.ainvoke({"prompt": "World"}))

    # The fixture prefixes user input with "echo:".
    assert result.startswith("echo:")
    assert "World" in result


def test_responses_agent_passes_model_with_apps_prefix(
    patched_databricks_openai: tuple[TestClient, list[dict[str, Any]]],
) -> None:
    _test_client, captured = patched_databricks_openai

    tool = create_responses_agent_tool(DatabricksAppModel(name="dao-ai-supplier-app"))
    asyncio.run(tool.ainvoke({"prompt": "ping"}))

    assert captured[0]["model"] == "apps/dao-ai-supplier-app"


def test_responses_agent_serializes_input_role_correctly(
    patched_databricks_openai: tuple[TestClient, list[dict[str, Any]]],
) -> None:
    """The request body MUST have a list of {role, content} items in `input`."""
    _test_client, captured = patched_databricks_openai

    tool = create_responses_agent_tool(DatabricksAppModel(name="app"))
    asyncio.run(tool.ainvoke({"prompt": "structured query"}))

    assert isinstance(captured[0]["input"], list)
    assert captured[0]["input"][0]["role"] == "user"
    assert captured[0]["input"][0]["content"] == "structured query"
