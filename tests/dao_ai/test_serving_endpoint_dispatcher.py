"""Unit tests for :func:`dao_ai.tools.create_serving_endpoint_dispatcher`.

The dispatcher does three things:
1. Lazily resolves the OpenAI API contract on first invocation via
   ``discover_serving_endpoint_api`` (SDK ``task`` field probe).
2. Caches the resolution for subsequent invocations.
3. Builds a ``ChatDatabricks`` with ``use_responses_api`` set per the
   resolved contract and invokes it.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from dao_ai.config import InferenceEndpointModel, LLMModel
from dao_ai.tools._api_discovery import ApiContract
from dao_ai.tools.serving_endpoint_dispatcher import (
    create_serving_endpoint_dispatcher,
)


@pytest.fixture(autouse=True)
def stub_workspace_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace the ambient WorkspaceClient() construction (used when the
    LLM isn't OBO) with a MagicMock so the dispatcher doesn't try to
    authenticate against a real workspace in unit tests."""
    fake_ws: MagicMock = MagicMock(name="WorkspaceClient")
    monkeypatch.setattr(
        "dao_ai.tools.serving_endpoint_dispatcher.WorkspaceClient",
        lambda *a, **kw: fake_ws,
    )
    return fake_ws


def _set_probe(
    monkeypatch: pytest.MonkeyPatch, returns: Optional[ApiContract]
) -> list[int]:
    """Force discover_serving_endpoint_api to return `returns`; return
    a list that's appended to on each probe call (so tests can assert
    probe count)."""
    counter: list[int] = []

    def _probe(name: str, ws: Any) -> Optional[ApiContract]:
        counter.append(1)
        return returns

    monkeypatch.setattr(
        "dao_ai.tools.serving_endpoint_dispatcher.discover_serving_endpoint_api",
        _probe,
    )
    return counter


def _invoke(tool, prompt: str) -> AIMessage:
    return asyncio.run(tool.ainvoke({"prompt": prompt}))


# ---------------------------------------------------------------------------
# Factory output
# ---------------------------------------------------------------------------


class TestFactoryOutput:
    def test_factory_returns_structured_tool(self) -> None:
        llm = InferenceEndpointModel(name="my-ep")
        tool = create_serving_endpoint_dispatcher(llm)
        assert tool.coroutine is not None
        assert tool.name == "my-ep"

    def test_factory_uses_explicit_name(self) -> None:
        tool = create_serving_endpoint_dispatcher(
            InferenceEndpointModel(name="my-ep"),
            name="custom_name",
        )
        assert tool.name == "custom_name"


# ---------------------------------------------------------------------------
# Explicit api: precedence (no probe runs)
# ---------------------------------------------------------------------------


class TestExplicitApiSkipsProbe:
    def test_explicit_responses_sets_use_responses_api_true_and_skips_probe(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        probe_counter = _set_probe(monkeypatch, "completions")
        mock_model = AsyncMock()
        mock_model.ainvoke.return_value = AIMessage(content="ok")

        with patch.object(LLMModel, "as_chat_model", return_value=mock_model):
            llm = InferenceEndpointModel(name="hardware-store-dao")
            tool = create_serving_endpoint_dispatcher(llm, api="responses")
            result = _invoke(tool, "hi")

        assert result.content == "ok"
        # Probe MUST NOT have run.
        assert probe_counter == []

    def test_explicit_completions_sets_use_responses_api_false_and_skips_probe(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        probe_counter = _set_probe(monkeypatch, "responses")
        mock_model = AsyncMock()
        mock_model.ainvoke.return_value = AIMessage(content="ok")

        with patch.object(LLMModel, "as_chat_model", return_value=mock_model):
            llm = InferenceEndpointModel(name="sonnet-4")
            tool = create_serving_endpoint_dispatcher(llm, api="completions")
            _invoke(tool, "hi")

        assert probe_counter == []


# ---------------------------------------------------------------------------
# Discovery resolves api on first invoke
# ---------------------------------------------------------------------------


class TestDiscoveryResolvesApi:
    def test_probe_returns_responses_sets_use_responses_api_true(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_probe(monkeypatch, "responses")
        captured_llm: dict[str, Any] = {}

        def _as_chat_model_spy(self):
            captured_llm["use_responses_api"] = self.use_responses_api
            mock = AsyncMock()
            mock.ainvoke.return_value = AIMessage(content="ok")
            return mock

        with patch.object(LLMModel, "as_chat_model", _as_chat_model_spy):
            llm = InferenceEndpointModel(name="hardware-store-dao")
            tool = create_serving_endpoint_dispatcher(llm)
            _invoke(tool, "hi")

        assert captured_llm["use_responses_api"] is True

    def test_probe_returns_completions_sets_use_responses_api_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_probe(monkeypatch, "completions")
        captured_llm: dict[str, Any] = {}

        def _as_chat_model_spy(self):
            captured_llm["use_responses_api"] = self.use_responses_api
            mock = AsyncMock()
            mock.ainvoke.return_value = AIMessage(content="ok")
            return mock

        with patch.object(LLMModel, "as_chat_model", _as_chat_model_spy):
            llm = InferenceEndpointModel(name="sonnet-4")
            tool = create_serving_endpoint_dispatcher(llm)
            _invoke(tool, "hi")

        assert captured_llm["use_responses_api"] is False

    def test_probe_returns_none_default_completions_uses_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_probe(monkeypatch, None)
        captured_llm: dict[str, Any] = {}

        def _as_chat_model_spy(self):
            captured_llm["use_responses_api"] = self.use_responses_api
            mock = AsyncMock()
            mock.ainvoke.return_value = AIMessage(content="ok")
            return mock

        with patch.object(LLMModel, "as_chat_model", _as_chat_model_spy):
            llm = InferenceEndpointModel(name="unknown-endpoint")
            tool = create_serving_endpoint_dispatcher(llm, default_api="completions")
            _invoke(tool, "hi")

        assert captured_llm["use_responses_api"] is False


# ---------------------------------------------------------------------------
# Caching — probe runs once per tool
# ---------------------------------------------------------------------------


class TestProbeCaching:
    def test_probe_runs_only_once_across_invocations(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        probe_counter = _set_probe(monkeypatch, "responses")
        mock_model = AsyncMock()
        mock_model.ainvoke.return_value = AIMessage(content="ok")

        with patch.object(LLMModel, "as_chat_model", return_value=mock_model):
            llm = InferenceEndpointModel(name="cached")
            tool = create_serving_endpoint_dispatcher(llm)
            _invoke(tool, "first")
            _invoke(tool, "second")
            _invoke(tool, "third")

        assert len(probe_counter) == 1
