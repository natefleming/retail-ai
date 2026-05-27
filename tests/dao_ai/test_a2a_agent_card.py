"""Unit tests for :mod:`dao_ai.apps.a2a.agent_card`.

Covers default skill derivation, security scheme conditional on
``a2a.on_behalf_of_user``, explicit override paths, and the
``DATABRICKS_APP_URL`` env / ``a2a.server_url`` precedence rules.
"""

import pytest
from a2a.types import (
    APIKeySecurityScheme,
    HTTPAuthSecurityScheme,
    SecurityScheme,
)
from pydantic import ValidationError

from dao_ai.apps.a2a.agent_card import (
    DEFAULT_A2A_RPC_PATH,
    build_agent_card,
    effective_a2a,
)
from dao_ai.config import (
    A2AModel,
    A2ASkillModel,
    A2ATaskStoreModel,
    AgentModel,
    AppConfig,
    AppModel,
    DatabaseModel,
    DeploymentTarget,
    InferenceEndpointModel,
    ProviderModel,
)


def _minimal_config(
    *,
    a2a: A2AModel | None = None,
    agents: list[AgentModel] | None = None,
    description: str = "test agent",
) -> AppConfig:
    if agents is None:
        agents = [
            AgentModel(
                name="greeter",
                description="says hi",
                model=InferenceEndpointModel(name="databricks-gpt-5-4-mini"),
            ),
        ]
    extra: dict = {}
    if a2a is not None:
        extra["a2a"] = a2a
    return AppConfig(
        app=AppModel(
            name="dao-ai-test",
            description=description,
            deployment_target=DeploymentTarget.APPS,
            agents=agents,
            **extra,
        ),
    )


@pytest.mark.unit
def test_effective_a2a_defaults_when_unset():
    """``effective_a2a`` returns a fresh A2AModel when app.a2a is None."""
    cfg = _minimal_config(a2a=None)
    a2a = effective_a2a(cfg)
    assert a2a.enabled is True
    assert isinstance(a2a.task_store, A2ATaskStoreModel)
    assert a2a.task_store.database is None
    assert a2a.task_store.table == "dao_ai_a2a_tasks"
    # on_behalf_of_user defaults to None (auto-derive from resources).
    assert a2a.on_behalf_of_user is None


@pytest.mark.unit
def test_effective_a2a_returns_existing():
    """``effective_a2a`` returns the configured A2AModel when set."""
    custom = A2AModel(enabled=False, task_store=A2ATaskStoreModel())
    cfg = _minimal_config(a2a=custom)
    assert effective_a2a(cfg) is custom


@pytest.mark.unit
def test_agent_card_basic_fields():
    """A default config produces a syntactically valid Agent Card."""
    cfg = _minimal_config(description="a sporting goods agent")
    card = build_agent_card(cfg)
    assert card.name == "dao-ai-test"
    assert card.description == "a sporting goods agent"
    assert card.url.endswith(DEFAULT_A2A_RPC_PATH)
    assert card.version  # dao-ai version string
    assert card.capabilities.streaming is True
    # Default config has no A2A task store database → no persisted task
    # state-transition history is retained.
    assert card.capabilities.state_transition_history is False
    assert card.capabilities.push_notifications is False
    assert card.default_input_modes == ["text/plain", "application/json"]
    assert card.default_output_modes == ["text/plain", "application/json"]
    # Top-level description has no trailing whitespace.
    assert card.description == card.description.rstrip()


@pytest.mark.unit
def test_agent_card_skills_default_to_sub_agents():
    """When ``a2a.skills`` is unset, derive one AgentSkill per sub-agent."""
    agents = [
        AgentModel(
            name="greeter",
            description="says hi",
            model=InferenceEndpointModel(name="databricks-gpt-5-4-mini"),
        ),
        AgentModel(
            name="echoer",
            description="echoes",
            model=InferenceEndpointModel(name="databricks-gpt-5-4-mini"),
        ),
    ]
    cfg = _minimal_config(agents=agents)
    card = build_agent_card(cfg)
    skill_ids = [s.id for s in card.skills]
    assert skill_ids == ["greeter", "echoer"]
    assert card.skills[0].description == "says hi"


@pytest.mark.unit
def test_agent_card_skills_explicit_override():
    """``a2a.skills`` wins over auto-derivation."""
    custom = A2AModel(
        skills=[
            A2ASkillModel(id="solve", name="solver", description="solves things"),
        ]
    )
    cfg = _minimal_config(a2a=custom)
    card = build_agent_card(cfg)
    assert [s.id for s in card.skills] == ["solve"]
    assert card.skills[0].description == "solves things"


@pytest.mark.unit
def test_agent_card_security_default_bearer():
    """No OBO → bearer scheme advertising PAT/M2M."""
    cfg = _minimal_config(a2a=A2AModel(on_behalf_of_user=False))
    card = build_agent_card(cfg)
    assert card.security_schemes is not None
    assert list(card.security_schemes.keys()) == ["bearer"]
    bearer = card.security_schemes["bearer"].root
    assert bearer.type == "http"
    assert bearer.scheme == "bearer"
    assert "PAT" in (bearer.bearer_format or "") or "M2M" in (
        bearer.bearer_format or ""
    )


def _mocked_workspace_client(host: str = "https://test.cloud.databricks.com"):
    """Patch context that makes `WorkspaceClient()` return a stub with the given host.

    Bypasses any real network / auth probing the SDK may do at construction.
    """
    from unittest.mock import patch as mock_patch

    fake = type(
        "FakeWsClient",
        (),
        {"config": type("C", (), {"host": host})()},
    )
    return mock_patch("databricks.sdk.WorkspaceClient", return_value=fake())


@pytest.mark.unit
def test_agent_card_security_obo_hint():
    """With ``a2a.on_behalf_of_user=True`` → Agent Card emits oauth2 + bearer schemes."""
    with _mocked_workspace_client():
        cfg = _minimal_config(a2a=A2AModel(on_behalf_of_user=True))
        card = build_agent_card(cfg)
    assert set(card.security_schemes.keys()) == {"oauth2", "bearer"}
    bearer = card.security_schemes["bearer"].root
    assert "OBO" in (bearer.bearer_format or "")
    oauth2 = card.security_schemes["oauth2"].root
    assert oauth2.type == "oauth2"
    flow = oauth2.flows.authorization_code
    assert flow is not None
    assert "user_impersonation" in flow.scopes


@pytest.mark.unit
def test_obo_auto_derived_from_resource_model():
    """Resource-level ``on_behalf_of_user=True`` alone triggers OBO advertisement."""
    agents = [
        AgentModel(
            name="greeter",
            description="says hi",
            model=InferenceEndpointModel(
                name="databricks-gpt-5-4-mini",
                on_behalf_of_user=True,
            ),
        ),
    ]
    with _mocked_workspace_client():
        cfg = _minimal_config(agents=agents)  # NO a2a block at all
        card = build_agent_card(cfg)
    assert set(card.security_schemes.keys()) == {"oauth2", "bearer"}


@pytest.mark.unit
def test_obo_explicit_false_suppresses_resource_derivation():
    """``a2a.on_behalf_of_user=False`` wins over a resource carrying OBO."""
    agents = [
        AgentModel(
            name="greeter",
            description="says hi",
            model=InferenceEndpointModel(
                name="databricks-gpt-5-4-mini",
                on_behalf_of_user=True,
            ),
        ),
    ]
    cfg = _minimal_config(agents=agents, a2a=A2AModel(on_behalf_of_user=False))
    card = build_agent_card(cfg)
    assert list(card.security_schemes.keys()) == ["bearer"]
    bearer = card.security_schemes["bearer"].root
    assert "PAT" in (bearer.bearer_format or "") or "M2M" in (
        bearer.bearer_format or ""
    )


@pytest.mark.unit
def test_obo_fallback_to_bearer_only_when_host_unresolvable(monkeypatch):
    """When OBO is on but no host resolves, fall back to bearer-only with OBO description."""
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    from unittest.mock import patch as mock_patch

    with mock_patch(
        "databricks.sdk.WorkspaceClient",
        side_effect=RuntimeError("no profile"),
    ):
        cfg = _minimal_config(a2a=A2AModel(on_behalf_of_user=True))
        card = build_agent_card(cfg)
    assert list(card.security_schemes.keys()) == ["bearer"]
    bearer = card.security_schemes["bearer"].root
    assert "OBO" in (bearer.bearer_format or "")


@pytest.mark.unit
def test_agent_card_security_explicit_override():
    """``a2a.security_schemes`` wins over the derived bearer scheme."""
    custom = A2AModel(
        security_schemes={
            "api_key": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
            }
        }
    )
    cfg = _minimal_config(a2a=custom)
    card = build_agent_card(cfg)
    assert list(card.security_schemes.keys()) == ["api_key"]


@pytest.mark.unit
def test_agent_card_security_schemes_typed_construction():
    """A2AModel accepts a2a-sdk typed SecurityScheme instances directly."""
    typed_bearer = HTTPAuthSecurityScheme(
        scheme="bearer",
        bearer_format="Custom Bearer",
        description="custom",
    )
    typed_api_key = APIKeySecurityScheme(
        type="apiKey",
        name="X-Custom-Key",
        **{"in": "header"},
    )
    custom = A2AModel(
        security_schemes={
            "bearer": SecurityScheme(typed_bearer),
            "api_key": SecurityScheme(typed_api_key),
        }
    )
    cfg = _minimal_config(a2a=custom)
    card = build_agent_card(cfg)
    assert set(card.security_schemes.keys()) == {"bearer", "api_key"}
    assert card.security_schemes["bearer"].root.bearer_format == "Custom Bearer"


@pytest.mark.unit
def test_agent_card_security_schemes_invalid_dict_fails_at_config_load():
    """Malformed scheme dicts fail at A2AModel construction, not at agent-card build."""
    with pytest.raises(ValidationError):
        # `apiKey` requires both `name` and `in`; missing `in` fails the union.
        A2AModel(
            security_schemes={
                "bad": {"type": "apiKey", "name": "X-Missing-In"},
            }
        )


@pytest.mark.unit
def test_agent_card_url_from_env(monkeypatch):
    """``$DATABRICKS_APP_URL`` is honored when ``a2a.server_url`` is unset."""
    monkeypatch.setenv("DATABRICKS_APP_URL", "https://app.example.com")
    cfg = _minimal_config()
    card = build_agent_card(cfg)
    assert card.url == "https://app.example.com/a2a"


@pytest.mark.unit
def test_agent_card_url_explicit_override_wins(monkeypatch):
    """``a2a.server_url`` overrides the env var."""
    monkeypatch.setenv("DATABRICKS_APP_URL", "https://app.example.com")
    cfg = _minimal_config(a2a=A2AModel(server_url="https://override.example.com/a2a"))
    card = build_agent_card(cfg)
    assert card.url == "https://override.example.com/a2a"


@pytest.mark.unit
def test_agent_card_url_falls_back_to_relative(monkeypatch):
    """No env, no override → relative ``/a2a``."""
    monkeypatch.delenv("DATABRICKS_APP_URL", raising=False)
    cfg = _minimal_config()
    card = build_agent_card(cfg)
    assert card.url == DEFAULT_A2A_RPC_PATH


@pytest.mark.unit
def test_agent_card_security_requirement_lists_schemes():
    """``security`` array has one OR-alternative per declared scheme (OpenAPI 3 semantics).

    Each entry maps a single scheme to its required scope list. Non-OAuth2
    schemes get an empty list (no scopes required).
    """
    custom = A2AModel(
        security_schemes={
            "bearer": {"type": "http", "scheme": "bearer"},
            "api_key": {"type": "apiKey", "in": "header", "name": "X-Foo"},
        }
    )
    cfg = _minimal_config(a2a=custom)
    card = build_agent_card(cfg)
    assert card.security is not None
    # One requirement object per scheme (OR-of-AND form).
    keys = [list(r.keys())[0] for r in card.security]
    assert sorted(keys) == ["api_key", "bearer"]
    # Neither scheme is OAuth2 → all scope lists empty.
    for req in card.security:
        for scopes in req.values():
            assert scopes == []


@pytest.mark.unit
def test_agent_card_security_obo_lists_user_impersonation_scope():
    """OBO Agent Card lists ``user_impersonation`` under the oauth2 requirement."""
    with _mocked_workspace_client():
        cfg = _minimal_config(a2a=A2AModel(on_behalf_of_user=True))
        card = build_agent_card(cfg)
    assert card.security is not None
    # Two requirements: one per scheme.
    assert len(card.security) == 2
    by_scheme = {list(r.keys())[0]: list(r.values())[0] for r in card.security}
    assert by_scheme["oauth2"] == ["user_impersonation"]
    assert by_scheme["bearer"] == []


@pytest.mark.unit
def test_agent_card_provider_and_docs_flow_through():
    """``provider``, ``documentation_url``, ``icon_url`` on A2AModel surface on the card."""
    cfg = _minimal_config(
        a2a=A2AModel(
            provider=ProviderModel(
                organization="Databricks Field Engineering",
                url="https://github.com/databrickslabs/dao-ai",
            ),
            documentation_url="https://example.com/docs",
            icon_url="https://example.com/icon.png",
        )
    )
    card = build_agent_card(cfg)
    assert card.provider is not None
    assert card.provider.organization == "Databricks Field Engineering"
    assert card.provider.url == "https://github.com/databrickslabs/dao-ai"
    assert card.documentation_url == "https://example.com/docs"
    assert card.icon_url == "https://example.com/icon.png"


@pytest.mark.unit
def test_agent_card_capability_overrides():
    """Capability flags on A2AModel override the auto-derived defaults."""
    cfg = _minimal_config(
        a2a=A2AModel(
            streaming=False,
            push_notifications=True,
            state_transition_history=True,
        )
    )
    card = build_agent_card(cfg)
    assert card.capabilities.streaming is False
    assert card.capabilities.push_notifications is True
    assert card.capabilities.state_transition_history is True


@pytest.mark.unit
def test_agent_card_state_transition_history_auto_derived_from_task_store():
    """STH auto-derives True when the A2A task store has a database configured."""
    cfg = _minimal_config(
        a2a=A2AModel(
            task_store=A2ATaskStoreModel(
                database=DatabaseModel(project="dao-ai-test-lakebase"),
            ),
        )
    )
    card = build_agent_card(cfg)
    assert card.capabilities.state_transition_history is True


@pytest.mark.unit
def test_agent_card_skill_description_trims_trailing_whitespace():
    """YAML ``>`` block-folded descriptions land on the card without trailing newlines."""
    agents = [
        AgentModel(
            name="greeter",
            # Mimic YAML ``>`` folding which appends a final newline.
            description="says hi\n",
            model=InferenceEndpointModel(name="databricks-gpt-5-4-mini"),
        ),
    ]
    cfg = _minimal_config(agents=agents)
    card = build_agent_card(cfg)
    assert card.skills[0].description == "says hi"
    assert not card.skills[0].description.endswith("\n")


@pytest.mark.unit
def test_agent_card_top_level_description_trims_trailing_whitespace():
    """App-level descriptions are rstripped on the card."""
    cfg = _minimal_config(description="a sporting goods agent\n")
    card = build_agent_card(cfg)
    assert card.description == "a sporting goods agent"
