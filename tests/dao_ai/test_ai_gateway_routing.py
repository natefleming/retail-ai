"""Unit tests for AI Gateway routing on InferenceEndpointModel.

Covers the `ai_gateway: bool` flag for routing through the Databricks AI
Gateway via ``ChatUnityAIGateway`` (a ``ChatDatabricks`` subclass with
``use_ai_gateway=True`` defaulted). The subclass exists so that MLflow
trace spans surface a distinct class name; functionally it inherits the
full ``ChatDatabricks`` behavior including the upstream name-field strip
in ``_convert_message_to_dict``.

Verifies:

- Pydantic validator rejects ``ai_gateway`` + ``use_responses_api``
- Default flag value leaves the legacy ``ChatDatabricks`` path intact
- ``ai_gateway=True`` routes ``as_chat_model()`` and
  ``chat_model_for_workspace_client()`` through ``ChatUnityAIGateway``
- ``ChatUnityAIGateway`` is a ``ChatDatabricks`` subclass with
  ``use_ai_gateway=True`` defaulted (verifies the trace-observability
  contract)
- Heterogeneous fallback lists (AI Gateway primary + legacy fallback)
  compose without raising
- Upstream ``ChatDatabricks`` strips the ``name`` field at the request
  boundary (the behavior dao-ai now relies on; smoke check)
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from databricks_langchain import ChatDatabricks
from pydantic import ValidationError

from dao_ai.config import ChatUnityAIGateway, InferenceEndpointModel

# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def test_default_ai_gateway_is_false() -> None:
    model = InferenceEndpointModel(name="databricks-gpt-5-4-mini")
    assert model.ai_gateway is False


def test_ai_gateway_true_with_responses_api_rejected() -> None:
    with pytest.raises(ValidationError) as exc_info:
        InferenceEndpointModel(
            name="databricks-claude-opus-4-6",
            ai_gateway=True,
            use_responses_api=True,
        )
    assert "use_responses_api" in str(exc_info.value)


def test_ai_gateway_true_with_obo_currently_allowed() -> None:
    """OBO + ai_gateway is permitted pending live verification."""
    model = InferenceEndpointModel(
        name="databricks-claude-opus-4-6",
        ai_gateway=True,
        on_behalf_of_user=True,
    )
    assert model.ai_gateway is True
    assert model.on_behalf_of_user is True


# ---------------------------------------------------------------------------
# ChatUnityAIGateway subclass contract
# ---------------------------------------------------------------------------


def test_chat_unity_ai_gateway_is_chat_databricks_subclass() -> None:
    assert issubclass(ChatUnityAIGateway, ChatDatabricks)


def test_chat_unity_ai_gateway_defaults_use_ai_gateway_true() -> None:
    instance = ChatUnityAIGateway.model_construct(model="databricks-claude-opus-4-6")
    assert instance.use_ai_gateway is True


def test_chat_databricks_parent_default_is_false() -> None:
    """Parent class default stays False; only the subclass flips it."""
    instance = ChatDatabricks.model_construct(model="databricks-claude-opus-4-6")
    assert instance.use_ai_gateway is False


def test_chat_unity_ai_gateway_llm_type_is_distinct() -> None:
    """_llm_type surfaces in MLflow traces and LangSmith — must be distinct
    from the parent ``chat-databricks`` so AI-Gateway-routed calls are
    visually distinguishable in observability tooling."""
    instance = ChatUnityAIGateway.model_construct(model="databricks-claude-opus-4-6")
    assert instance._llm_type == "chat-unity-ai-gateway"
    parent = ChatDatabricks.model_construct(model="databricks-claude-opus-4-6")
    assert parent._llm_type == "chat-databricks"
    assert instance._llm_type != parent._llm_type


# ---------------------------------------------------------------------------
# as_chat_model routing
# ---------------------------------------------------------------------------


def test_as_chat_model_default_uses_chat_databricks() -> None:
    model = InferenceEndpointModel(
        name="databricks-gpt-5-4-mini",
        temperature=0.2,
        max_tokens=512,
    )
    with (
        patch("dao_ai.config.ChatDatabricks") as mock_chat_databricks,
        patch("dao_ai.config.ChatUnityAIGateway") as mock_unity,
    ):
        mock_chat_databricks.return_value = MagicMock(name="chat_databricks_instance")
        result = model.as_chat_model()
    mock_chat_databricks.assert_called_once()
    mock_unity.assert_not_called()
    kwargs = mock_chat_databricks.call_args.kwargs
    assert kwargs["model"] == "databricks-gpt-5-4-mini"
    assert kwargs["temperature"] == 0.2
    assert kwargs["max_tokens"] == 512
    assert result is mock_chat_databricks.return_value


def test_as_chat_model_ai_gateway_uses_chat_unity_ai_gateway() -> None:
    model = InferenceEndpointModel(
        name="databricks-claude-opus-4-6",
        ai_gateway=True,
        temperature=0.1,
        max_tokens=1024,
    )
    with (
        patch("dao_ai.config.ChatUnityAIGateway") as mock_unity,
        patch("dao_ai.config.ChatDatabricks") as mock_chat_databricks,
    ):
        mock_unity.return_value = MagicMock(name="chat_unity_instance")
        result = model.as_chat_model()
    mock_chat_databricks.assert_not_called()
    mock_unity.assert_called_once()
    kwargs = mock_unity.call_args.kwargs
    assert kwargs["model"] == "databricks-claude-opus-4-6"
    assert kwargs["temperature"] == 0.1
    assert kwargs["max_tokens"] == 1024
    assert kwargs["disable_streaming"] is False
    assert result is mock_unity.return_value


def test_as_chat_model_ai_gateway_disables_streaming_when_requested() -> None:
    model = InferenceEndpointModel(
        name="databricks-claude-opus-4-6",
        ai_gateway=True,
        disable_streaming=True,
    )
    with patch("dao_ai.config.ChatUnityAIGateway") as mock_unity:
        model.as_chat_model()
    assert mock_unity.call_args.kwargs["disable_streaming"] is True


# ---------------------------------------------------------------------------
# OBO factory routing
# ---------------------------------------------------------------------------


def test_chat_model_for_workspace_client_uses_ai_gateway_when_flag_set() -> None:
    """OBO-style construction (per-request workspace_client) must route through
    ChatUnityAIGateway when ai_gateway=True on the config — not silently fall
    back to ChatDatabricks. This is the bug that motivated the shared factory."""
    model = InferenceEndpointModel(
        name="databricks-claude-opus-4-6",
        ai_gateway=True,
        on_behalf_of_user=True,
        temperature=0.0,
        max_tokens=128,
    )
    obo_wc = MagicMock()

    with (
        patch("dao_ai.config.ChatUnityAIGateway") as mock_unity,
        patch("dao_ai.config.ChatDatabricks") as mock_chat_databricks,
    ):
        mock_unity.return_value = MagicMock()
        model.chat_model_for_workspace_client(obo_wc)

    mock_chat_databricks.assert_not_called()
    mock_unity.assert_called_once()
    kwargs = mock_unity.call_args.kwargs
    assert kwargs["workspace_client"] is obo_wc
    assert kwargs["model"] == "databricks-claude-opus-4-6"


def test_chat_model_for_workspace_client_legacy_path_passes_workspace_client() -> None:
    """Without ai_gateway, OBO factory falls through to ChatDatabricks bound
    to the OBO workspace_client (unchanged legacy behavior)."""
    model = InferenceEndpointModel(
        name="databricks-gpt-5-4-mini",
        on_behalf_of_user=True,
        temperature=0.0,
        max_tokens=128,
    )
    obo_wc = MagicMock()

    with (
        patch("dao_ai.config.ChatDatabricks") as mock_chat_databricks,
        patch("dao_ai.config.ChatUnityAIGateway") as mock_unity,
    ):
        mock_chat_databricks.return_value = MagicMock()
        model.chat_model_for_workspace_client(obo_wc)

    mock_unity.assert_not_called()
    mock_chat_databricks.assert_called_once()
    kwargs = mock_chat_databricks.call_args.kwargs
    assert kwargs["workspace_client"] is obo_wc
    assert kwargs["model"] == "databricks-gpt-5-4-mini"


# ---------------------------------------------------------------------------
# Tool calling — unit-level wiring
# ---------------------------------------------------------------------------


def test_as_chat_model_returns_runnable_with_bind_tools() -> None:
    """The returned client must expose bind_tools() so dao-ai agent loops
    (which always call .bind_tools(tools)) work transparently against the
    AI Gateway path."""
    model = InferenceEndpointModel(name="databricks-claude-opus-4-6", ai_gateway=True)
    client = model.as_chat_model()
    assert hasattr(client, "bind_tools"), (
        "as_chat_model() must return a client with bind_tools() — agent loops "
        "depend on this. Got: " + type(client).__name__
    )


def test_heterogeneous_fallbacks_compose() -> None:
    """AI Gateway primary + legacy fallback should chain via with_fallbacks."""
    primary = InferenceEndpointModel(
        name="databricks-claude-opus-4-6",
        ai_gateway=True,
        fallbacks=["databricks-gpt-5-4-mini"],
    )

    unity_instance = MagicMock(name="ChatUnityAIGateway-instance")
    chat_databricks_instance = MagicMock(name="ChatDatabricks-instance")
    chained = MagicMock(name="with_fallbacks-result")
    unity_instance.with_fallbacks.return_value = chained

    with (
        patch("dao_ai.config.ChatUnityAIGateway", return_value=unity_instance),
        patch("dao_ai.config.ChatDatabricks", return_value=chat_databricks_instance),
    ):
        result = primary.as_chat_model()

    unity_instance.with_fallbacks.assert_called_once()
    (fallback_list,) = unity_instance.with_fallbacks.call_args.args
    assert fallback_list == [chat_databricks_instance]
    assert result is chained


# ---------------------------------------------------------------------------
# Subclass identity — the observability contract
# ---------------------------------------------------------------------------


def test_as_chat_model_ai_gateway_returns_chat_unity_ai_gateway() -> None:
    """When ai_gateway=True, as_chat_model() must return ChatUnityAIGateway —
    the named subclass that surfaces in MLflow trace spans."""
    model = InferenceEndpointModel(name="databricks-claude-opus-4-6", ai_gateway=True)
    client = model.as_chat_model()
    assert isinstance(client, ChatUnityAIGateway)
    assert isinstance(client, ChatDatabricks)  # subclass relationship


def test_as_chat_model_legacy_path_is_not_unity_subclass() -> None:
    """Regression: ai_gateway=False returns plain ChatDatabricks."""
    model = InferenceEndpointModel(name="databricks-gpt-5-4-mini")
    client = model.as_chat_model()
    assert isinstance(client, ChatDatabricks)
    assert not isinstance(client, ChatUnityAIGateway)


def test_chat_model_for_workspace_client_ai_gateway_returns_unity_subclass() -> None:
    """OBO factory must also return ChatUnityAIGateway when ai_gateway=True."""
    from databricks.sdk import WorkspaceClient

    model = InferenceEndpointModel(
        name="databricks-claude-opus-4-6",
        ai_gateway=True,
        on_behalf_of_user=True,
    )
    obo_wc = MagicMock(spec=WorkspaceClient)

    client = model.chat_model_for_workspace_client(obo_wc)
    assert isinstance(client, ChatUnityAIGateway)


# ---------------------------------------------------------------------------
# Upstream name-strip smoke check
#
# dao-ai used to ship a custom subclass that stripped the `name` field at the
# request-payload boundary because the AI Gateway 400s on `messages.N.name`.
# That behavior now lives upstream in
# ChatDatabricks._convert_message_to_dict; we verify it survives so that the
# subclass collapse is safe.
# ---------------------------------------------------------------------------


def test_chat_databricks_drops_name_field_on_message_conversion() -> None:
    """Upstream ChatDatabricks already strips the ``name`` field at the
    request-payload boundary — this is the behavior dao-ai relies on now
    that the custom AIGatewayChatOpenAI workaround is gone."""
    from databricks_langchain.chat_models import _convert_message_to_dict
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    for msg in [
        HumanMessage(content="hi", name="alice"),
        AIMessage(content="hello there", name="agent_a"),
        SystemMessage(content="you are helpful", name="sys"),
    ]:
        as_dict = _convert_message_to_dict(msg)
        assert "name" not in as_dict, (
            f"upstream regression: ChatDatabricks now propagates 'name' on "
            f"{type(msg).__name__}; revisit the AIGatewayChatOpenAI workaround"
        )


# ---------------------------------------------------------------------------
# Live AI Gateway smoke (integration)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("DAO_AI_INTEGRATION"),
    reason="Set DAO_AI_INTEGRATION=1 (with DATABRICKS_HOST/DATABRICKS_TOKEN or a configured profile) to run live AI Gateway smoke.",
)
def test_ai_gateway_chat_handles_supervisor_named_messages_live() -> None:
    """Live AI Gateway round-trip with supervisor-tagged AIMessage(name=...).

    Pre-refactor the gateway 400'd on ``messages.N.name``; the dao-ai custom
    subclass stripped it. Post-refactor upstream ``ChatDatabricks`` strips it
    natively, so this call must still succeed.
    """
    from langchain_core.messages import AIMessage, HumanMessage

    endpoint = InferenceEndpointModel(
        name="databricks-claude-opus-4-6",
        ai_gateway=True,
        max_tokens=64,
    )
    chat = endpoint.as_chat_model()
    assert isinstance(chat, ChatUnityAIGateway)
    out = chat.invoke(
        [
            HumanMessage(content="Say a one-word greeting.", name="user_alice"),
            AIMessage(content="hello there", name="loyalty_lead"),
            HumanMessage(content="Now say a one-word farewell."),
        ]
    )
    assert isinstance(out, AIMessage)
    assert out.content
