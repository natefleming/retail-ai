"""Unit tests for AI Gateway routing on InferenceEndpointModel.

Covers the `use_ai_gateway: bool` flag for routing through the Databricks AI
Gateway via ``ChatUnityAIGateway`` (a ``ChatDatabricks`` subclass with
``use_ai_gateway=True`` defaulted). The subclass exists so that MLflow
trace spans surface a distinct class name; functionally it inherits the
full ``ChatDatabricks`` behavior including the upstream name-field strip
in ``_convert_message_to_dict``.

Verifies:

- ``use_ai_gateway`` is the canonical key, with the legacy ``ai_gateway``
  spelling still accepted via a validation alias
- ``use_ai_gateway`` composes with ``use_responses_api`` — the gateway serves
  both /chat/completions and /responses, and the pairing works for models
  whose /responses usage block is complete (see the live matrix below)
- Default flag value leaves the legacy ``ChatDatabricks`` path intact
- ``use_ai_gateway=True`` routes ``as_chat_model()`` and
  ``chat_model_for_workspace_client()`` through ``ChatUnityAIGateway``
- ``ChatUnityAIGateway`` is a ``ChatDatabricks`` subclass with
  ``use_ai_gateway=True`` defaulted (verifies the trace-observability
  contract)
- A bare fallback/judge string inherits the primary's routing, so a
  UC-securable fallback resolves through the gateway rather than crashing
- Upstream ``ChatDatabricks`` strips the ``name`` field at the request
  boundary (the behavior dao-ai now relies on; smoke check)
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from databricks_langchain import ChatDatabricks

from dao_ai.config import ChatUnityAIGateway, InferenceEndpointModel

# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def test_default_use_ai_gateway_is_false() -> None:
    model = InferenceEndpointModel(name="databricks-gpt-5-4-mini")
    assert model.use_ai_gateway is False


def test_uc_securable_name_auto_enables_and_routes_through_the_gateway() -> None:
    """A UC-securable model with the flag omitted infers use_ai_gateway=True,
    and that inferred flag must actually reach the client: as_chat_model()
    routes through ChatUnityAIGateway with the qualified name as the model id."""
    model = InferenceEndpointModel.model_validate({"name": "system.ai.gpt-5-mini"})
    assert model.use_ai_gateway is True
    with (
        patch("dao_ai.config.ChatUnityAIGateway") as mock_unity,
        patch("dao_ai.config.ChatDatabricks") as mock_chat_databricks,
    ):
        model.as_chat_model()
    mock_chat_databricks.assert_not_called()
    mock_unity.assert_called_once()
    assert mock_unity.call_args.kwargs["model"] == "system.ai.gpt-5-mini"


# ---------------------------------------------------------------------------
# Canonical name and the legacy alias
# ---------------------------------------------------------------------------


def test_canonical_use_ai_gateway_key_parses() -> None:
    """The config key is spelled the same as the databricks-langchain kwarg
    it feeds (``use_ai_gateway``), which dao-ai was the only layer to spell
    differently."""
    model = InferenceEndpointModel.model_validate(
        {"name": "databricks-claude-opus-4-6", "use_ai_gateway": True}
    )
    assert model.use_ai_gateway is True


def test_legacy_ai_gateway_key_still_parses() -> None:
    """Existing YAML using ``ai_gateway:`` must keep working — this is what
    justifies a validation alias over a hard rename."""
    model = InferenceEndpointModel.model_validate(
        {"name": "databricks-claude-opus-4-6", "ai_gateway": True}
    )
    assert model.use_ai_gateway is True


def test_json_schema_property_is_use_ai_gateway() -> None:
    """The canonical (schema-facing) name must be the new one, so editors and
    the checked-in schema advertise ``use_ai_gateway`` rather than the legacy
    key. ``Field(alias=...)`` would invert this; ``AliasChoices`` does not."""
    schema = InferenceEndpointModel.model_json_schema()
    # The top level is a $ref into $defs, so resolve it before reading props.
    defn = schema["$defs"][schema["$ref"].rsplit("/", 1)[-1]]
    props = defn["properties"]
    assert "use_ai_gateway" in props
    assert "ai_gateway" not in props
    assert props["use_ai_gateway"]["default"] is False


# ---------------------------------------------------------------------------
# AI Gateway + Responses API — allowed
#
# dao-ai routes every gateway call through /ai-gateway/mlflow/v1, which serves
# both /chat/completions and /responses. Raw HTTP against ``fevm``: every model
# answers 200 on both, so the endpoints are not the constraint.
# ``usage_details`` is whether the /responses reply carried
# ``input_tokens_details``/``output_tokens_details``:
#
# | model                        | /chat/completions | /responses | details |
# | ---------------------------- | ----------------- | ---------- | ------- |
# | databricks-gpt-5-4           | 200               | 200        | yes     |
# | databricks-gpt-5-4-mini      | 200               | 200        | yes     |
# | databricks-gpt-5-mini        | 200               | 200        | yes     |
# | system.ai.gpt-5-4            | 200               | 200        | yes     |
# | databricks-gpt-oss-120b      | 200               | 200        | no      |
# | databricks-claude-sonnet-4-5 | 200               | 200        | no      |
#
# A single-turn probe is not the whole story. On a deployed app,
# system.ai.gpt-5-4-mini — full ``usage`` block, 200 on both paths above —
# failed as soon as the agent called a tool:
#
#   400 INVALID_PARAMETER_VALUE: Failed to parse ContentItem. Could not resolve
#   type id 'function_call' into a subtype of
#   [simple type, class com.databricks.fmapiproxy.translation.ContentItem]
#
# The gateway's /responses translator cannot deserialize a ``function_call``
# content item, on any model. Every supervisor/swarm handoff is a tool call, so
# the pairing is only useful for a single agent that calls nothing. Server-side,
# and orthogonal to the ``usage`` caveat below.
#
# The ``usage`` block is what decides whether a reply parses. OpenAI-family
# models return the ``*_details`` sub-objects and work end to end.
# gpt-oss-120b and claude-sonnet-4-5 omit them; langchain-openai maps the
# absent fields to ``None``, ``AIMessage`` rejects ``None`` there, and the
# reply raises ValidationError on ``usage_metadata``. That is a client-side
# limitation which can disappear on any langchain-openai release — not a
# gateway one, and not something a static validator can decide per model.
#
# So dao-ai does not police the pairing. (An earlier validator rejected it, on
# the separate and false premise that the gateway was chat-completions only.)
# The caveats above are documented rather than enforced, along with one more: the
# gateway addresses only Foundation Model and UC-securable models, so a custom
# ResponsesAgent endpoint 404s and needs the legacy path. The pairing does work
# for a tool-free agent on a model with a complete usage block, so enforcement
# would have to reject working configurations to catch the rest.
# ---------------------------------------------------------------------------


def test_gateway_with_responses_api_is_allowed() -> None:
    """Verified live: ``databricks-gpt-5-mini`` + ``use_ai_gateway`` +
    ``use_responses_api`` returns 200 with full ``usage_metadata``. Rejecting
    the pairing at config load would block a working configuration."""
    model = InferenceEndpointModel(
        name="databricks-gpt-5-mini",
        use_ai_gateway=True,
        use_responses_api=True,
    )
    assert model.use_ai_gateway is True
    assert model.use_responses_api is True


def test_gateway_with_responses_api_allowed_via_the_legacy_alias() -> None:
    """A config still spelling the flag ``ai_gateway:`` must be accepted on the
    same terms — the alias resolves before after-validators run."""
    model = InferenceEndpointModel.model_validate(
        {
            "name": "databricks-gpt-5-mini",
            "ai_gateway": True,
            "use_responses_api": True,
        }
    )
    assert model.use_ai_gateway is True
    assert model.use_responses_api is True


def test_gateway_responses_api_flows_through_to_the_client() -> None:
    """Accepting the combination is only useful if the flag actually reaches
    the client — that is what selects ``responses.create`` downstream."""
    model = InferenceEndpointModel(
        name="databricks-gpt-5-mini",
        use_ai_gateway=True,
        use_responses_api=True,
    )
    with patch("dao_ai.config.ChatUnityAIGateway") as mock_unity:
        model.as_chat_model()
    assert mock_unity.call_args.kwargs["use_responses_api"] is True


def test_uc_securable_name_composes_with_responses_api() -> None:
    """A schema-qualified UC-securable model is addressed the same way on the
    /responses path — ``full_name``, not ``name``, is the model id."""
    model = InferenceEndpointModel.model_validate(
        {
            "schema": {"catalog_name": "system", "schema_name": "ai"},
            "name": "gpt-5-mini",
            "use_ai_gateway": True,
            "use_responses_api": True,
        }
    )
    with patch("dao_ai.config.ChatUnityAIGateway") as mock_unity:
        model.as_chat_model()
    kwargs = mock_unity.call_args.kwargs
    assert kwargs["model"] == "system.ai.gpt-5-mini"
    assert kwargs["use_responses_api"] is True


def test_responses_api_without_the_gateway_is_untouched() -> None:
    """The feature's original use, verified live: a custom ResponsesAgent
    endpoint on the legacy ``/serving-endpoints/<name>/invocations`` path. The
    gateway cannot address custom endpoints at all (404), so this spelling
    stays the only way to reach one."""
    model = InferenceEndpointModel(
        name="genie-brain-ms-traces",
        use_responses_api=True,
    )
    assert model.use_ai_gateway is False
    with patch("dao_ai.config.ChatDatabricks") as mock_chat:
        model.as_chat_model()
    assert mock_chat.call_args.kwargs["use_responses_api"] is True


def test_gateway_without_responses_api_is_untouched() -> None:
    """The other half of the guard: the plain gateway path is unaffected."""
    model = InferenceEndpointModel(
        name="databricks-claude-opus-4-6", use_ai_gateway=True
    )
    with patch("dao_ai.config.ChatUnityAIGateway") as mock_unity:
        model.as_chat_model()
    assert mock_unity.call_args.kwargs["use_responses_api"] is False


def test_both_routing_branches_honor_the_renamed_field() -> None:
    """Two independent call sites read the flag (``as_chat_model`` and
    ``chat_model_for_workspace_client``); renaming one and not the other
    would silently drop OBO traffic off the gateway path."""
    model = InferenceEndpointModel(
        name="databricks-claude-opus-4-6", use_ai_gateway=True
    )
    with patch("dao_ai.config.ChatUnityAIGateway") as mock_unity:
        model.as_chat_model()
        model.chat_model_for_workspace_client(MagicMock())
    assert mock_unity.call_count == 2


def test_use_ai_gateway_true_with_obo_currently_allowed() -> None:
    """OBO + use_ai_gateway is permitted pending live verification."""
    model = InferenceEndpointModel(
        name="databricks-claude-opus-4-6",
        use_ai_gateway=True,
        on_behalf_of_user=True,
    )
    assert model.use_ai_gateway is True
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
# Sampling-param defaults (temperature, max_tokens)
# ---------------------------------------------------------------------------


def test_as_chat_model_omits_temperature_and_max_tokens_when_unset() -> None:
    """Reasoning-mode endpoints (e.g. Sonnet 5) reject any `temperature`
    value, and callers often want the endpoint's own `max_tokens` default.
    When the user does not set these in YAML, dao-ai must construct
    ChatDatabricks with `temperature=None` and `max_tokens=None` so the
    databricks-langchain client drops both fields from the payload."""
    model = InferenceEndpointModel(name="databricks-claude-sonnet-5")
    assert model.temperature is None
    assert model.max_tokens is None
    with patch("dao_ai.config.ChatDatabricks") as mock_chat:
        model.as_chat_model()
    kwargs = mock_chat.call_args.kwargs
    assert kwargs["temperature"] is None
    assert kwargs["max_tokens"] is None


def test_as_chat_model_forwards_explicit_temperature_and_max_tokens() -> None:
    """Explicit values must still flow through — the None default is only
    the *absence-of-config* behavior, not a suppression of user intent."""
    model = InferenceEndpointModel(
        name="databricks-claude-sonnet-4-5",
        temperature=0.3,
        max_tokens=1024,
    )
    with patch("dao_ai.config.ChatDatabricks") as mock_chat:
        model.as_chat_model()
    kwargs = mock_chat.call_args.kwargs
    assert kwargs["temperature"] == 0.3
    assert kwargs["max_tokens"] == 1024


def test_as_chat_model_omits_extra_params_when_unset() -> None:
    """When extra_params is unset, dao-ai must not pass the kwarg at all so
    the chat client keeps its own default (empty dict)."""
    model = InferenceEndpointModel(name="databricks-gpt-oss-120b")
    assert model.extra_params is None
    with patch("dao_ai.config.ChatDatabricks") as mock_chat:
        model.as_chat_model()
    assert "extra_params" not in mock_chat.call_args.kwargs


def test_as_chat_model_forwards_extra_params() -> None:
    """extra_params (e.g. reasoning_effort on gpt-oss) must flow through to
    the chat client verbatim."""
    model = InferenceEndpointModel(
        name="databricks-gpt-oss-120b",
        extra_params={"reasoning_effort": "low"},
    )
    with patch("dao_ai.config.ChatDatabricks") as mock_chat:
        model.as_chat_model()
    assert mock_chat.call_args.kwargs["extra_params"] == {"reasoning_effort": "low"}


def test_extra_params_forwarded_on_ai_gateway_path() -> None:
    """The AI-Gateway path must also forward extra_params."""
    model = InferenceEndpointModel(
        name="databricks-gpt-oss-120b",
        use_ai_gateway=True,
        extra_params={"reasoning_effort": "low"},
    )
    with patch("dao_ai.config.ChatUnityAIGateway") as mock_chat:
        model.as_chat_model()
    assert mock_chat.call_args.kwargs["extra_params"] == {"reasoning_effort": "low"}


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
        use_ai_gateway=True,
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
        use_ai_gateway=True,
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
    ChatUnityAIGateway when use_ai_gateway=True on the config — not silently fall
    back to ChatDatabricks. This is the bug that motivated the shared factory."""
    model = InferenceEndpointModel(
        name="databricks-claude-opus-4-6",
        use_ai_gateway=True,
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
    """Without use_ai_gateway, OBO factory falls through to ChatDatabricks bound
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
    model = InferenceEndpointModel(
        name="databricks-claude-opus-4-6", use_ai_gateway=True
    )
    client = model.as_chat_model()
    assert hasattr(client, "bind_tools"), (
        "as_chat_model() must return a client with bind_tools() — agent loops "
        "depend on this. Got: " + type(client).__name__
    )


def test_fallbacks_compose_via_with_fallbacks() -> None:
    """A primary and its string fallback chain via with_fallbacks."""
    primary = InferenceEndpointModel(
        name="databricks-claude-opus-4-6",
        use_ai_gateway=True,
        fallbacks=["databricks-gpt-5-4-mini"],
    )

    unity_instance = MagicMock(name="ChatUnityAIGateway-instance")
    chained = MagicMock(name="with_fallbacks-result")
    unity_instance.with_fallbacks.return_value = chained

    with patch("dao_ai.config.ChatUnityAIGateway", return_value=unity_instance):
        result = primary.as_chat_model()

    unity_instance.with_fallbacks.assert_called_once()
    assert result is chained


def test_string_fallback_inherits_gateway_from_primary() -> None:
    """A bare fallback string carries no routing of its own, so it inherits the
    primary's: a gateway primary's string fallback is promoted with
    ``use_ai_gateway=True`` and builds a ``ChatUnityAIGateway``, not a legacy
    ``ChatDatabricks``. (Was mixed-mode before; PR #294 review made routing
    consistent so a UC-securable fallback string can work at all.)"""
    primary = InferenceEndpointModel(
        name="databricks-claude-opus-4-6",
        use_ai_gateway=True,
        fallbacks=["databricks-gpt-5-4-mini"],
    )

    with (
        patch("dao_ai.config.ChatUnityAIGateway") as mock_unity,
        patch("dao_ai.config.ChatDatabricks") as mock_chat,
    ):
        primary.as_chat_model()

    # Primary + fallback, both via the gateway; no legacy client built.
    assert mock_unity.call_count == 2
    mock_chat.assert_not_called()
    assert mock_unity.call_args_list[1].kwargs["model"] == "databricks-gpt-5-4-mini"


def test_uc_securable_fallback_string_does_not_crash() -> None:
    """Regression (PR #294 review): a dotted UC-securable fallback string loaded
    fine, then crashed in ``as_chat_model`` because promotion dropped
    ``use_ai_gateway`` and ``validate_schema_qualification`` then rejected the
    three-level name. Inheriting the primary's flag lets it build and reach the
    gateway as the qualified id."""
    primary = InferenceEndpointModel(
        name="databricks-claude-opus-4-6",
        use_ai_gateway=True,
        fallbacks=["system.ai.claude-sonnet-4-5"],
    )

    with patch("dao_ai.config.ChatUnityAIGateway") as mock_unity:
        primary.as_chat_model()

    assert mock_unity.call_count == 2
    assert mock_unity.call_args_list[1].kwargs["model"] == "system.ai.claude-sonnet-4-5"


def test_string_fallback_stays_legacy_when_primary_is_legacy() -> None:
    """The inverse: a legacy primary's string fallback inherits ``False`` and
    stays on the ``ChatDatabricks`` path — existing configs are unaffected."""
    primary = InferenceEndpointModel(
        name="databricks-claude-opus-4-6",
        fallbacks=["databricks-gpt-5-4-mini"],
    )

    with (
        patch("dao_ai.config.ChatUnityAIGateway") as mock_unity,
        patch("dao_ai.config.ChatDatabricks") as mock_chat,
    ):
        primary.as_chat_model()

    mock_unity.assert_not_called()
    assert mock_chat.call_count == 2


def test_uc_securable_fallback_of_a_legacy_primary_still_reaches_the_gateway() -> None:
    """A non-gateway primary can still name a UC-securable fallback string. The
    primary rides the legacy path; the fallback is only addressable on the
    gateway, so it must infer use_ai_gateway on its own rather than inheriting
    the primary's False and crashing in ``as_chat_model`` at build time."""
    primary = InferenceEndpointModel(
        name="databricks-gpt-oss-120b",
        fallbacks=["system.ai.claude-sonnet-4-5"],
    )
    assert primary.use_ai_gateway is False

    with (
        patch("dao_ai.config.ChatUnityAIGateway") as mock_unity,
        patch("dao_ai.config.ChatDatabricks") as mock_chat,
    ):
        primary.as_chat_model()

    # Legacy primary on ChatDatabricks; UC-securable fallback on the gateway.
    mock_chat.assert_called_once()
    assert mock_chat.call_args.kwargs["model"] == "databricks-gpt-oss-120b"
    mock_unity.assert_called_once()
    assert mock_unity.call_args.kwargs["model"] == "system.ai.claude-sonnet-4-5"


def test_best_of_n_judge_string_inherits_gateway_from_primary() -> None:
    """The judge is promoted from a string the same way a fallback is, and hits
    the same crash on a UC-securable name. It inherits the primary's routing."""
    primary = InferenceEndpointModel.model_validate(
        {
            "name": "databricks-claude-opus-4-6",
            "use_ai_gateway": True,
            "best_of_n": {"n": 2, "judge": "system.ai.claude-sonnet-4-5"},
        }
    )

    with (
        patch("dao_ai.config.ChatUnityAIGateway") as mock_unity,
        patch("dao_ai.best_of_n.BestOfNChatModel"),
    ):
        primary.as_chat_model()

    # Primary + judge, both via the gateway.
    assert mock_unity.call_count == 2
    assert mock_unity.call_args_list[1].kwargs["model"] == "system.ai.claude-sonnet-4-5"


def test_fallback_equal_to_the_primary_is_deduped_by_full_name() -> None:
    """The self-dedup guard must key on ``full_name``, not ``name``. A
    schema-anchored primary (``system.ai.claude-sonnet-4-5``) with a fallback
    spelled as the equivalent full id is the *same* endpoint — keying on the
    short segment (``claude-sonnet-4-5`` != ``system.ai.claude-sonnet-4-5``)
    would miss it and retry the primary against itself."""
    primary = InferenceEndpointModel.model_validate(
        {
            "schema": {"catalog_name": "system", "schema_name": "ai"},
            "name": "claude-sonnet-4-5",
            "use_ai_gateway": True,
            "fallbacks": ["system.ai.claude-sonnet-4-5"],
        }
    )

    with patch("dao_ai.config.ChatUnityAIGateway") as mock_unity:
        primary.as_chat_model()

    # Only the primary is built; the equivalent fallback is deduped away.
    assert mock_unity.call_count == 1
    mock_unity.return_value.with_fallbacks.assert_not_called()


def test_distinct_models_sharing_a_short_name_are_not_deduped() -> None:
    """The inverse: two different securables that share a short segment must
    both survive. Keying on ``name`` would collapse ``system.ai.claude-...``
    and ``other.ai.claude-...`` into one and silently drop the fallback."""
    primary = InferenceEndpointModel.model_validate(
        {
            "schema": {"catalog_name": "system", "schema_name": "ai"},
            "name": "claude-sonnet-4-5",
            "use_ai_gateway": True,
            "fallbacks": [
                {
                    "schema": {"catalog_name": "other", "schema_name": "ai"},
                    "name": "claude-sonnet-4-5",
                    "use_ai_gateway": True,
                }
            ],
        }
    )

    with patch("dao_ai.config.ChatUnityAIGateway") as mock_unity:
        primary.as_chat_model()

    # Primary + the genuinely different fallback both built and composed.
    assert mock_unity.call_count == 2
    mock_unity.return_value.with_fallbacks.assert_called_once()


# ---------------------------------------------------------------------------
# Subclass identity — the observability contract
# ---------------------------------------------------------------------------


def test_as_chat_model_ai_gateway_returns_chat_unity_ai_gateway() -> None:
    """When use_ai_gateway=True, as_chat_model() must return ChatUnityAIGateway —
    the named subclass that surfaces in MLflow trace spans."""
    model = InferenceEndpointModel(
        name="databricks-claude-opus-4-6", use_ai_gateway=True
    )
    client = model.as_chat_model()
    assert isinstance(client, ChatUnityAIGateway)
    assert isinstance(client, ChatDatabricks)  # subclass relationship


def test_as_chat_model_legacy_path_is_not_unity_subclass() -> None:
    """Regression: use_ai_gateway=False returns plain ChatDatabricks."""
    model = InferenceEndpointModel(name="databricks-gpt-5-4-mini")
    client = model.as_chat_model()
    assert isinstance(client, ChatDatabricks)
    assert not isinstance(client, ChatUnityAIGateway)


def test_chat_model_for_workspace_client_ai_gateway_returns_unity_subclass() -> None:
    """OBO factory must also return ChatUnityAIGateway when use_ai_gateway=True."""
    from databricks.sdk import WorkspaceClient

    model = InferenceEndpointModel(
        name="databricks-claude-opus-4-6",
        use_ai_gateway=True,
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
        use_ai_gateway=True,
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
