"""Unit tests for :mod:`dao_ai.apps.a2a.security`.

Covers each constant + factory in the convenience module:

* Type assertion (each is the right a2a-sdk type).
* Module-level construction implicitly tests Pydantic validation.
* End-to-end Agent Card embedding (a2a-sdk's ``AgentCard`` accepts the
  scheme inside ``security_schemes``).
* Host resolution precedence for factories.
"""

from unittest.mock import patch

import pytest
from a2a.types import (
    APIKeySecurityScheme,
    HTTPAuthSecurityScheme,
    In,
    OAuth2SecurityScheme,
    OpenIdConnectSecurityScheme,
)

from dao_ai.apps.a2a.agent_card import build_agent_card
from dao_ai.apps.a2a.security import (
    BEARER_DATABRICKS_M2M,
    BEARER_DATABRICKS_OBO,
    BEARER_DATABRICKS_PAT,
    api_key_header,
    oauth2_databricks_authorization_code,
    oauth2_databricks_client_credentials,
    oauth2_databricks_obo,
    openid_connect_databricks,
)
from dao_ai.config import (
    A2AModel,
    AgentModel,
    AppConfig,
    AppModel,
    DeploymentTarget,
    InferenceEndpointModel,
)


def _config_with_schemes(schemes: dict) -> AppConfig:
    return AppConfig(
        app=AppModel(
            name="dao-ai-test",
            description="test",
            deployment_target=DeploymentTarget.APPS,
            a2a=A2AModel(security_schemes=schemes),
            agents=[
                AgentModel(
                    name="greeter",
                    description="says hi",
                    model=InferenceEndpointModel(name="databricks-gpt-5-4-mini"),
                ),
            ],
        ),
    )


@pytest.mark.unit
def test_bearer_pat_constant_shape():
    assert isinstance(BEARER_DATABRICKS_PAT, HTTPAuthSecurityScheme)
    assert BEARER_DATABRICKS_PAT.scheme == "bearer"
    assert "PAT" in (BEARER_DATABRICKS_PAT.bearer_format or "")


@pytest.mark.unit
def test_bearer_m2m_constant_shape():
    assert isinstance(BEARER_DATABRICKS_M2M, HTTPAuthSecurityScheme)
    assert "M2M" in (BEARER_DATABRICKS_M2M.bearer_format or "")


@pytest.mark.unit
def test_bearer_obo_constant_shape():
    assert isinstance(BEARER_DATABRICKS_OBO, HTTPAuthSecurityScheme)
    assert "OBO" in (BEARER_DATABRICKS_OBO.bearer_format or "")
    assert BEARER_DATABRICKS_OBO.description is not None
    assert "x-forwarded-access-token" in BEARER_DATABRICKS_OBO.description


@pytest.mark.unit
def test_api_key_header_factory():
    scheme = api_key_header("X-Custom-Key")
    assert isinstance(scheme, APIKeySecurityScheme)
    assert scheme.name == "X-Custom-Key"
    assert scheme.in_ == In.header


@pytest.mark.unit
def test_oauth2_databricks_authorization_code_factory():
    scheme = oauth2_databricks_authorization_code("https://my-ws.cloud.databricks.com")
    assert isinstance(scheme, OAuth2SecurityScheme)
    flow = scheme.flows.authorization_code
    assert flow is not None
    assert (
        flow.authorization_url == "https://my-ws.cloud.databricks.com/oidc/v1/authorize"
    )
    assert flow.token_url == "https://my-ws.cloud.databricks.com/oidc/v1/token"
    assert "all-apis" in flow.scopes


@pytest.mark.unit
def test_oauth2_databricks_client_credentials_factory():
    scheme = oauth2_databricks_client_credentials("https://my-ws.cloud.databricks.com")
    flow = scheme.flows.client_credentials
    assert flow is not None
    assert flow.token_url == "https://my-ws.cloud.databricks.com/oidc/v1/token"


@pytest.mark.unit
def test_oauth2_databricks_obo_factory():
    scheme = oauth2_databricks_obo("https://my-ws.cloud.databricks.com")
    flow = scheme.flows.authorization_code
    assert flow is not None
    assert "user_impersonation" in flow.scopes


@pytest.mark.unit
def test_openid_connect_databricks_factory():
    scheme = openid_connect_databricks("https://my-ws.cloud.databricks.com")
    assert isinstance(scheme, OpenIdConnectSecurityScheme)
    assert scheme.open_id_connect_url == (
        "https://my-ws.cloud.databricks.com/oidc/.well-known/openid-configuration"
    )


@pytest.mark.unit
def test_host_resolution_normalizes_protocol():
    """Hosts without ``https://`` are normalized."""
    scheme = oauth2_databricks_authorization_code("my-ws.cloud.databricks.com")
    flow = scheme.flows.authorization_code
    assert flow.authorization_url.startswith("https://")


@pytest.mark.unit
def test_host_resolution_strips_trailing_slash():
    scheme = oauth2_databricks_authorization_code("https://my-ws.cloud.databricks.com/")
    flow = scheme.flows.authorization_code
    assert (
        flow.authorization_url == "https://my-ws.cloud.databricks.com/oidc/v1/authorize"
    )


@pytest.mark.unit
def test_host_resolution_uses_env_var(monkeypatch):
    """When `host` is omitted, $DATABRICKS_HOST wins."""
    monkeypatch.setenv("DATABRICKS_HOST", "https://from-env.cloud.databricks.com")
    scheme = oauth2_databricks_authorization_code()
    flow = scheme.flows.authorization_code
    assert (
        flow.authorization_url
        == "https://from-env.cloud.databricks.com/oidc/v1/authorize"
    )


@pytest.mark.unit
def test_host_resolution_raises_when_unresolvable(monkeypatch):
    """No host arg + no env var + no ambient → ValueError with a clear message."""
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    # Force get_default_databricks_host to return None.
    with patch(
        "dao_ai.apps.a2a.security.get_default_databricks_host", return_value=None
    ):
        with pytest.raises(ValueError, match="Workspace host could not be resolved"):
            oauth2_databricks_authorization_code()


@pytest.mark.unit
@pytest.mark.parametrize(
    "scheme_factory",
    [
        lambda: BEARER_DATABRICKS_PAT,
        lambda: BEARER_DATABRICKS_M2M,
        lambda: BEARER_DATABRICKS_OBO,
        lambda: api_key_header("X-Foo"),
        lambda: oauth2_databricks_authorization_code("https://ws.example.com"),
        lambda: oauth2_databricks_client_credentials("https://ws.example.com"),
        lambda: oauth2_databricks_obo("https://ws.example.com"),
        lambda: openid_connect_databricks("https://ws.example.com"),
    ],
)
def test_scheme_round_trips_through_agent_card(scheme_factory):
    """Every constant/factory output is accepted by the Agent Card builder."""
    scheme = scheme_factory()
    cfg = _config_with_schemes({"x": scheme})
    card = build_agent_card(cfg)
    assert card.security_schemes is not None
    assert "x" in card.security_schemes
