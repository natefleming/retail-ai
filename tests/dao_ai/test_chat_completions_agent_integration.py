"""End-to-end integration tests for create_chat_completions_agent_tool.

Same TestClient pattern as :mod:`test_responses_agent_integration`, but
stubs the OpenAI ``POST /chat/completions`` route and exercises the
chat-completions factory.
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
from dao_ai.tools.chat_completions_agent import (
    create_chat_completions_agent_tool,
)


def _build_chat_completions_stub(
    echo_prefix: str = "echo:",
) -> tuple[TestClient, list[dict[str, Any]]]:
    captured_requests: list[dict[str, Any]] = []
    app: FastAPI = FastAPI()

    @app.post("/chat/completions")
    async def chat_completions(request: Request) -> dict[str, Any]:
        body: dict[str, Any] = await request.json()
        captured_requests.append(body)

        user_text: str = ""
        for msg in body.get("messages", []):
            if isinstance(msg, dict) and msg.get("role") == "user":
                user_text = msg.get("content", "")
                break

        return {
            "id": "chatcmpl_test_123",
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"{echo_prefix}{user_text}",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }

    return TestClient(app), captured_requests


@pytest.fixture
def chat_completions_stub() -> tuple[TestClient, list[dict[str, Any]]]:
    return _build_chat_completions_stub()


class _PatchedDatabricksOpenAI:
    """Drop-in for DatabricksOpenAI routing requests through the FastAPI
    TestClient via httpx.MockTransport."""

    def __init__(self, test_client: TestClient) -> None:
        from openai import OpenAI

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
    def chat(self) -> Any:
        return self._openai.chat


@pytest.fixture
def patched_databricks_openai(
    monkeypatch: pytest.MonkeyPatch,
    chat_completions_stub: tuple[TestClient, list[dict[str, Any]]],
) -> tuple[TestClient, list[dict[str, Any]]]:
    test_client, captured = chat_completions_stub

    def _factory(**_kwargs: Any) -> _PatchedDatabricksOpenAI:
        return _PatchedDatabricksOpenAI(test_client)

    monkeypatch.setattr(
        "dao_ai.tools.chat_completions_agent.DatabricksOpenAI",
        _factory,
    )
    monkeypatch.setattr(
        DatabricksAppModel,
        "workspace_client_from",
        lambda self, context: MagicMock(name="WorkspaceClient"),
    )
    return test_client, captured


def test_chat_completions_agent_end_to_end_through_httpx_transport(
    patched_databricks_openai: tuple[TestClient, list[dict[str, Any]]],
) -> None:
    _test_client, captured = patched_databricks_openai
    tool = create_chat_completions_agent_tool(DatabricksAppModel(name="my-app"))
    result: str = asyncio.run(tool.ainvoke({"prompt": "Hello"}))

    assert result == "echo:Hello"
    assert len(captured) == 1
    assert captured[0]["model"] == "apps/my-app"
    assert captured[0]["messages"] == [{"role": "user", "content": "Hello"}]


def test_chat_completions_agent_extracts_message_content(
    patched_databricks_openai: tuple[TestClient, list[dict[str, Any]]],
) -> None:
    _test_client, _captured = patched_databricks_openai
    tool = create_chat_completions_agent_tool(DatabricksAppModel(name="app-x"))
    result: str = asyncio.run(tool.ainvoke({"prompt": "World"}))
    assert result.startswith("echo:")
    assert "World" in result


def test_chat_completions_agent_passes_model_with_apps_prefix(
    patched_databricks_openai: tuple[TestClient, list[dict[str, Any]]],
) -> None:
    _test_client, captured = patched_databricks_openai
    tool = create_chat_completions_agent_tool(
        DatabricksAppModel(name="dao-ai-supplier-app")
    )
    asyncio.run(tool.ainvoke({"prompt": "ping"}))
    assert captured[0]["model"] == "apps/dao-ai-supplier-app"


def test_chat_completions_agent_serializes_messages_correctly(
    patched_databricks_openai: tuple[TestClient, list[dict[str, Any]]],
) -> None:
    _test_client, captured = patched_databricks_openai
    tool = create_chat_completions_agent_tool(DatabricksAppModel(name="app"))
    asyncio.run(tool.ainvoke({"prompt": "structured query"}))

    assert isinstance(captured[0]["messages"], list)
    assert captured[0]["messages"][0]["role"] == "user"
    assert captured[0]["messages"][0]["content"] == "structured query"
