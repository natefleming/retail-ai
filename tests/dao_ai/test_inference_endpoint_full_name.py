"""UC-securable model names on ``InferenceEndpointModel``.

The Unity AI Gateway addresses models as UC securables, so a model id can be
a three-level name (``system.ai.claude-sonnet-4-5``) rather than a serving
endpoint name (``databricks-claude-sonnet-4-5``). This adds the same
``schema`` + ``full_name`` interface every other UC-backed config class
already implements (``HasFullName``; see ``VolumeModel``, ``TableModel``,
``EvaluationDatasetModel``).

Three spellings must all work:

1. plain endpoint name — ``name: databricks-claude-sonnet-4-5`` (today's
   behavior, must be untouched)
2. fully qualified name alone — ``name: system.ai.claude-sonnet-4-5``
3. schema + short name — ``schema: *system_ai`` + ``name: claude-sonnet-4-5``

Verified live on the ``fevm`` workspace, and the reason the gateway flag is
required for the qualified spellings:

| model id                       | gateway | /serving-endpoints/<name>/invocations |
| ------------------------------ | ------- | ------------------------------------- |
| ``system.ai.claude-sonnet-4-5``| 200     | 404 ENDPOINT_NOT_FOUND                |
| ``databricks-claude-sonnet-4-5``| 200    | 200                                   |
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from dao_ai.config import AppConfig, HasFullName, InferenceEndpointModel, SchemaModel

SYSTEM_AI: dict[str, str] = {"catalog_name": "system", "schema_name": "ai"}


# ---------------------------------------------------------------------------
# The three accepted spellings
# ---------------------------------------------------------------------------


def test_plain_endpoint_name_is_unchanged() -> None:
    """Form 1. The overwhelmingly common case: no schema, no dots, and
    ``full_name`` must be the endpoint name verbatim."""
    model = InferenceEndpointModel(name="databricks-claude-sonnet-4-5")
    assert model.schema_model is None
    assert model.full_name == "databricks-claude-sonnet-4-5"


def test_fully_qualified_name_alone_is_accepted() -> None:
    """Form 2. A three-level name needs no schema to be usable."""
    model = InferenceEndpointModel(
        name="system.ai.claude-sonnet-4-5", use_ai_gateway=True
    )
    assert model.full_name == "system.ai.claude-sonnet-4-5"


def test_schema_plus_short_name_resolves_to_the_qualified_name() -> None:
    """Form 3. The point of the feature: reuse a SchemaModel anchor instead of
    repeating ``system.ai.`` on every model."""
    model = InferenceEndpointModel(
        schema=SchemaModel.model_validate(SYSTEM_AI),
        name="claude-sonnet-4-5",
        use_ai_gateway=True,
    )
    assert model.full_name == "system.ai.claude-sonnet-4-5"


def test_schema_accepts_a_plain_dict_from_yaml() -> None:
    """A YAML anchor arrives as a mapping, not a constructed SchemaModel."""
    model = InferenceEndpointModel.model_validate(
        {"schema": SYSTEM_AI, "name": "claude-sonnet-4-5", "use_ai_gateway": True}
    )
    assert isinstance(model.schema_model, SchemaModel)
    assert model.full_name == "system.ai.claude-sonnet-4-5"


def test_the_config_facing_key_is_schema() -> None:
    """``schema_model`` exists only because ``schema`` shadows a pydantic
    method. The documented key — what serialization emits and what the JSON
    schema advertises — must be ``schema``, matching TableModel and
    VolumeModel."""
    model = InferenceEndpointModel.model_validate(
        {"schema": SYSTEM_AI, "name": "claude-sonnet-4-5", "use_ai_gateway": True}
    )
    dumped = model.model_dump(by_alias=True)
    assert "schema" in dumped and "schema_model" not in dumped

    # ``model_json_schema()`` on this class returns a $defs-only wrapper (the
    # self-reference in ``fallbacks``), so read the definition the checked-in
    # schema file — the one IDEs load — actually exposes.
    schema = AppConfig.model_json_schema()
    properties = schema["$defs"]["InferenceEndpointModel"]["properties"]
    assert "schema" in properties and "schema_model" not in properties


def test_a_plain_dump_still_round_trips() -> None:
    """``model_dump()`` without ``by_alias`` emits the attribute name, and with
    ``extra="forbid"`` that would not re-validate unless the field also accepts
    its own name. Real configs hit this: ``LakebaseRetrieverModel`` is built
    from ``vector_store.model_dump()``, whose embedding model is an
    ``InferenceEndpointModel``."""
    model = InferenceEndpointModel.model_validate(
        {"schema": SYSTEM_AI, "name": "claude-sonnet-4-5", "use_ai_gateway": True}
    )
    assert "schema_model" in model.model_dump()
    assert (
        InferenceEndpointModel.model_validate(model.model_dump()).full_name
        == "system.ai.claude-sonnet-4-5"
    )


# ---------------------------------------------------------------------------
# The HasFullName interface
# ---------------------------------------------------------------------------


def test_implements_has_full_name() -> None:
    """Same interface as every other UC-backed config class, so callers can
    depend on the protocol rather than on the concrete type."""
    assert issubclass(InferenceEndpointModel, HasFullName)
    assert isinstance(InferenceEndpointModel(name="databricks-x"), HasFullName)


# ---------------------------------------------------------------------------
# Rejected combinations (caught at config load, where the message can name it)
# ---------------------------------------------------------------------------


def test_schema_plus_already_qualified_name_is_rejected() -> None:
    """Naive concatenation would yield
    ``system.ai.system.ai.claude-sonnet-4-5`` and fail as a confusing 404 at
    request time. VolumeModel and EvaluationDatasetModel both have this latent
    bug; this class should not inherit it."""
    with pytest.raises(ValidationError, match="already fully qualified"):
        InferenceEndpointModel.model_validate(
            {
                "schema": SYSTEM_AI,
                "name": "system.ai.claude-sonnet-4-5",
                "use_ai_gateway": True,
            }
        )


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"schema": SYSTEM_AI, "name": "claude-sonnet-4-5"}, id="schema"),
        pytest.param({"name": "system.ai.claude-sonnet-4-5"}, id="qualified-name"),
    ],
)
def test_uc_securable_name_without_the_gateway_is_rejected(
    payload: dict[str, object],
) -> None:
    """A three-level name is only addressable through the AI Gateway — the
    legacy serving path answers 404 ENDPOINT_NOT_FOUND (verified live). Failing
    at load with a message naming the flag beats a 404 on first invocation.

    Both spellings resolve the same ``full_name``, so both must be rejected:
    keying the check on ``schema`` alone let the equivalent dotted ``name``
    through to that 404."""
    with pytest.raises(ValidationError, match="use_ai_gateway"):
        InferenceEndpointModel.model_validate(payload)


# ---------------------------------------------------------------------------
# What actually reaches the wire
# ---------------------------------------------------------------------------


def test_qualified_name_is_what_reaches_the_chat_client() -> None:
    """``full_name`` is inert unless the value the client sends is the
    qualified one — that string becomes the OpenAI-style model id in the
    gateway request body."""
    model = InferenceEndpointModel.model_validate(
        {"schema": SYSTEM_AI, "name": "claude-sonnet-4-5", "use_ai_gateway": True}
    )
    with patch("dao_ai.config.ChatUnityAIGateway") as mock_unity:
        model.as_chat_model()
    assert mock_unity.call_args.kwargs["model"] == "system.ai.claude-sonnet-4-5"


def test_obo_branch_also_sends_the_qualified_name() -> None:
    """The OBO entry point is a second, independent construction site."""
    from unittest.mock import MagicMock

    model = InferenceEndpointModel.model_validate(
        {"schema": SYSTEM_AI, "name": "claude-sonnet-4-5", "use_ai_gateway": True}
    )
    with patch("dao_ai.config.ChatUnityAIGateway") as mock_unity:
        model.chat_model_for_workspace_client(MagicMock())
    assert mock_unity.call_args.kwargs["model"] == "system.ai.claude-sonnet-4-5"


def test_unqualified_models_still_send_the_bare_endpoint_name() -> None:
    """The regression guard for every existing config: switching the wire site
    to ``full_name`` must be a no-op when no schema is set, on the legacy
    ``ChatDatabricks`` path in particular — it addresses endpoints by name and
    404s on a three-level name."""
    model = InferenceEndpointModel(name="databricks-claude-sonnet-4-5")
    with patch("dao_ai.config.ChatDatabricks") as mock_chat:
        model.as_chat_model()
    assert mock_chat.call_args.kwargs["model"] == "databricks-claude-sonnet-4-5"


# ---------------------------------------------------------------------------
# The deploy manifest / auth policy
# ---------------------------------------------------------------------------


def test_endpoint_resource_is_emitted_for_a_real_endpoint() -> None:
    """The regression guard: every existing config keeps its serving-endpoint
    resource, which is what grants the deployed model's service principal
    access to the endpoint."""
    resources = InferenceEndpointModel(
        name="databricks-claude-sonnet-4-5"
    ).as_resources()
    assert [r.to_dict() for r in resources] == [
        {
            "serving_endpoint": [
                {"name": "databricks-claude-sonnet-4-5", "on_behalf_of_user": False}
            ]
        }
    ]


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(
            {"name": "system.ai.claude-sonnet-4-5", "use_ai_gateway": True},
            id="qualified-name-alone",
        ),
        pytest.param(
            {"schema": SYSTEM_AI, "name": "claude-sonnet-4-5", "use_ai_gateway": True},
            id="schema-plus-short-name",
        ),
    ],
)
def test_no_endpoint_resource_for_a_uc_securable_model(payload: dict) -> None:
    """A UC-securable model is not a serving endpoint, and MLflow has no
    resource type for one (there is no ``DatabricksModel``). Verified live on
    ``fevm``: ``serving_endpoints.get('system.ai.claude-sonnet-4-5')`` raises
    ResourceDoesNotExist, and the short name does not resolve either — so
    whichever spelling went into ``DatabricksServingEndpoint(endpoint_name=...)``
    would name something that does not exist. Emit nothing rather than a
    resource the platform cannot resolve; access is governed by UC grants on
    the model."""
    assert InferenceEndpointModel.model_validate(payload).as_resources() == []


def test_apps_emits_a_resource_for_a_real_endpoint() -> None:
    """The regression guard for the Apps target: a plain endpoint name still
    produces the ``serving-endpoint`` app resource that grants the app SP
    CAN_QUERY."""
    from dao_ai.apps.resources import _extract_llm_resources

    resources = _extract_llm_resources(
        {"default_llm": InferenceEndpointModel(name="databricks-claude-sonnet-4-5")}
    )
    assert [(r["type"], r["serving_endpoint_name"]) for r in resources] == [
        ("serving-endpoint", "databricks-claude-sonnet-4-5")
    ]


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(
            {"name": "system.ai.claude-sonnet-4-5", "use_ai_gateway": True},
            id="qualified-name-alone",
        ),
        pytest.param(
            {"schema": SYSTEM_AI, "name": "claude-sonnet-4-5", "use_ai_gateway": True},
            id="schema-plus-short-name",
        ),
    ],
)
def test_apps_emits_no_resource_for_a_uc_securable_model(payload: dict) -> None:
    """Apps has its own resource list, built independently of
    ``as_resources()`` — and it is validated eagerly, so a name the platform
    cannot resolve fails the whole deploy rather than degrading. Observed live
    on ``fevm``::

        POST /api/2.0/apps -> ResourceDoesNotExist:
          Endpoint with name 'system.ai.claude-sonnet-4-5' does not exist.

    The short spelling is worse than a hard failure: ``claude-sonnet-4-5``
    might resolve for someone as an unrelated custom endpoint, silently
    granting CAN_QUERY on the wrong thing. Skip, exactly as the Model Serving
    path does."""
    from dao_ai.apps.resources import _extract_llm_resources

    assert (
        _extract_llm_resources(
            {"uc_llm": InferenceEndpointModel.model_validate(payload)}
        )
        == []
    )


@pytest.mark.parametrize(
    "extractor",
    ["_extract_llm_resources", "_extract_sdk_llm_resources"],
)
def test_every_apps_extractor_skips_a_uc_securable_model(extractor: str) -> None:
    """Apps builds its resource list through *two* independent extractors —
    ``generate_app_resources`` uses the raw-dict one, ``generate_sdk_resources``
    (the path ``_deploy_app`` actually takes) uses the SDK one. Fixing only the
    first left the deploy failing identically, so both are pinned here."""
    from dao_ai.apps import resources as apps_resources

    fn = getattr(apps_resources, extractor)
    model = InferenceEndpointModel.model_validate(
        {"schema": SYSTEM_AI, "name": "claude-sonnet-4-5", "use_ai_gateway": True}
    )
    assert fn({"uc_llm": model}) == []


# ---------------------------------------------------------------------------
# The Model Serving auto-auth gap (verified live on ``fevm``)
# ---------------------------------------------------------------------------


def _auth_policy_logs(*models: InferenceEndpointModel) -> str:
    """Build the Model Serving auth policy, returning the captured log text."""
    from loguru import logger

    from dao_ai.config import AppConfig, ResourcesModel
    from dao_ai.providers.databricks import build_auth_policy

    lines: list[str] = []
    sink_id = logger.add(
        lambda m: lines.append(f"{m.record['message']} {m.record['extra']}"),
        level="WARNING",
    )
    try:
        build_auth_policy(
            AppConfig(
                resources=ResourcesModel(
                    models={f"m{i}": m for i, m in enumerate(models)}
                )
            )
        )
    finally:
        logger.remove(sink_id)
    return "\n".join(lines)


def test_uc_securable_model_warns_at_model_serving_deploy() -> None:
    """A UC-securable model is unreachable from a Model Serving container, and
    the user finds out as an opaque runtime 404. Verified live on ``fevm``: the
    logged policy declared only the two ``databricks-*`` endpoints, and every
    UC-securable worker failed with::

        404 NOT_FOUND: 'system.ai.claude-sonnet-4-5' does not exist.

    while the same name answers 200 under a full-scope token. Cause: agent
    deploys authenticate with a token downscoped to the declared resources, and
    no MLflow resource type can declare a UC-securable model — so ``[]`` from
    ``as_resources()`` is correct *and* leaves the model unreachable. Warn at
    deploy, where the operator can still act, rather than at request time."""
    text = _auth_policy_logs(
        InferenceEndpointModel.model_validate(
            {"schema": SYSTEM_AI, "name": "claude-sonnet-4-5", "use_ai_gateway": True}
        )
    )
    assert "system.ai.claude-sonnet-4-5" in text
    assert "on_behalf_of_user" in text


def test_each_unreachable_model_is_named_once() -> None:
    """Two config keys can address one model — the qualified spelling and a
    schema anchor resolve to the same ``full_name``, as they do in the live
    verification config. The warning is a list of things to go fix, so it names
    each model once rather than once per key that happens to reference it."""
    text = _auth_policy_logs(
        InferenceEndpointModel(name="system.ai.claude-sonnet-4-5", use_ai_gateway=True),
        InferenceEndpointModel.model_validate(
            {"schema": SYSTEM_AI, "name": "claude-sonnet-4-5", "use_ai_gateway": True}
        ),
    )
    assert text.count("system.ai.claude-sonnet-4-5") == 1


def test_no_warning_for_an_ordinary_endpoint() -> None:
    """The regression guard: every existing config deploys without new noise."""
    assert _auth_policy_logs(InferenceEndpointModel(name="databricks-x")) == ""


def test_uri_uses_the_qualified_name() -> None:
    """``uri`` feeds MLflow judges/scorers (``model=self.model.uri``). The short
    name would silently address a different model; the qualified one is the
    honest identifier."""
    model = InferenceEndpointModel.model_validate(
        {"schema": SYSTEM_AI, "name": "claude-sonnet-4-5", "use_ai_gateway": True}
    )
    assert model.uri == "databricks:/system.ai.claude-sonnet-4-5"
    assert InferenceEndpointModel(name="databricks-x").uri == "databricks:/databricks-x"
