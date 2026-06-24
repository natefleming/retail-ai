"""Unit tests for ``dao_ai.tools.a2a_agent.create_a2a_agent_tool``.

Covers the two new configuration modes (Mode 1 = manual endpoint,
Mode 2 = Databricks AppResource) and the five auth modes:

* bearer (existing)
* none (existing)
* gcp_service_account (existing — light coverage; deep coverage lives
  alongside the GCP credential loader tests)
* forwarded_user_token (new)
* databricks_app_sp (new)

Following dao-ai convention this file does not use pytest-asyncio —
async paths are exercised indirectly by spot-checking the synchronous
factory wiring (auth_type resolution, header provider closure, endpoint
derivation). End-to-end stream-drain coverage already lives in
``test_a2a_executor.py`` / ``test_a2a_integration.py`` on the **server**
side; here we just validate the **client tool** factory.
"""

from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from dao_ai.config import DatabricksAppModel
from dao_ai.state import Context
from dao_ai.tools.a2a_agent import (
    _VALID_AUTH_MODES,
    _app_sp_headers,
    _coerce_app,
    _forwarded_user_token_headers,
    _no_auth_headers,
    create_a2a_agent_tool,
)

# ---------------------------------------------------------------------------
# _coerce_app
# ---------------------------------------------------------------------------


def test_coerce_app_passes_through_model_instance() -> None:
    model = DatabricksAppModel(name="some-app")
    assert _coerce_app(model) is model


def test_coerce_app_validates_dict_into_model() -> None:
    coerced = _coerce_app({"name": "some-app", "on_behalf_of_user": True})
    assert isinstance(coerced, DatabricksAppModel)
    assert coerced.name == "some-app"
    assert coerced.on_behalf_of_user is True


def test_coerce_app_none_returns_none() -> None:
    assert _coerce_app(None) is None


def test_coerce_app_rejects_other_types() -> None:
    with pytest.raises(TypeError, match="DatabricksAppModel"):
        _coerce_app("just-a-string")


# ---------------------------------------------------------------------------
# Header providers
# ---------------------------------------------------------------------------


def test_no_auth_headers_returns_empty() -> None:
    assert _no_auth_headers() == {}
    ctx = Context(thread_id="t", user_id="u", headers={"foo": "bar"})
    assert _no_auth_headers(ctx) == {}


def test_forwarded_user_token_reads_lowercase_header() -> None:
    ctx = Context(
        thread_id="t",
        user_id="u",
        headers={"x-forwarded-access-token": "USER_TOKEN_A"},
    )
    assert _forwarded_user_token_headers(ctx) == {
        "Authorization": "Bearer USER_TOKEN_A"
    }


def test_forwarded_user_token_reads_titlecase_header() -> None:
    ctx = Context(
        thread_id="t",
        user_id="u",
        headers={"X-Forwarded-Access-Token": "USER_TOKEN_B"},
    )
    assert _forwarded_user_token_headers(ctx) == {
        "Authorization": "Bearer USER_TOKEN_B"
    }


def test_forwarded_user_token_lowercase_wins_over_titlecase() -> None:
    ctx = Context(
        thread_id="t",
        user_id="u",
        headers={
            "x-forwarded-access-token": "LOWER",
            "X-Forwarded-Access-Token": "UPPER",
        },
    )
    assert _forwarded_user_token_headers(ctx) == {"Authorization": "Bearer LOWER"}


def test_forwarded_user_token_missing_raises_helpful_error() -> None:
    ctx = Context(thread_id="t", user_id="u", headers={})
    with pytest.raises(RuntimeError, match="x-forwarded-access-token"):
        _forwarded_user_token_headers(ctx)


def test_forwarded_user_token_none_context_raises() -> None:
    with pytest.raises(RuntimeError, match="x-forwarded-access-token"):
        _forwarded_user_token_headers(None)


def test_app_sp_headers_delegates_to_workspace_client_authenticate() -> None:
    wc = MagicMock()
    wc.config.authenticate.return_value = {"Authorization": "Bearer M2M_TOKEN"}
    headers = _app_sp_headers(wc)
    assert headers == {"Authorization": "Bearer M2M_TOKEN"}
    wc.config.authenticate.assert_called_once()


def test_app_sp_headers_refreshes_per_call() -> None:
    """``config.authenticate`` is called every time, allowing SDK token rotation."""
    wc = MagicMock()
    wc.config.authenticate.side_effect = [
        {"Authorization": "Bearer first"},
        {"Authorization": "Bearer second"},
    ]
    assert _app_sp_headers(wc) == {"Authorization": "Bearer first"}
    assert _app_sp_headers(wc) == {"Authorization": "Bearer second"}
    assert wc.config.authenticate.call_count == 2


# ---------------------------------------------------------------------------
# Factory — Mode 1 (manual endpoint)
# ---------------------------------------------------------------------------


def test_mode1_bearer_creates_tool_with_default_name() -> None:
    tool = create_a2a_agent_tool(
        endpoint="https://example.com",
        auth="static_token",
        auth_type="bearer",
    )
    assert tool.name == "a2a_agent"


def test_mode1_bearer_missing_auth_raises() -> None:
    with pytest.raises(ValueError, match="bearer"):
        create_a2a_agent_tool(endpoint="https://example.com", auth_type="bearer")


def test_mode1_bearer_empty_token_raises() -> None:
    with pytest.raises(ValueError, match="bearer token is empty"):
        create_a2a_agent_tool(
            endpoint="https://example.com", auth="", auth_type="bearer"
        )


def test_mode1_none_skips_auth_arg() -> None:
    tool = create_a2a_agent_tool(endpoint="https://example.com", auth_type="none")
    assert tool.name == "a2a_agent"


def test_mode1_unknown_auth_type_raises() -> None:
    with pytest.raises(ValueError, match="auth_type must be one of"):
        create_a2a_agent_tool(endpoint="https://example.com", auth_type="bogus_mode")


def test_mode1_strips_trailing_slash_from_endpoint() -> None:
    # No direct getter for the resolved endpoint on the StructuredTool,
    # but we can at least confirm trailing-slash strip doesn't blow up.
    tool = create_a2a_agent_tool(endpoint="https://example.com/", auth_type="none")
    assert tool.name == "a2a_agent"


# ---------------------------------------------------------------------------
# Factory — Mode 2 (AppResource)
# ---------------------------------------------------------------------------


def _mocked_url(url: str = "https://supplier-app.databricksapps.com"):
    return patch.object(
        DatabricksAppModel, "url", new_callable=PropertyMock, return_value=url
    )


def test_mode2_app_obo_defaults_to_forwarded_user_token() -> None:
    with _mocked_url():
        app = DatabricksAppModel(name="supplier", on_behalf_of_user=True)
        # If the default did NOT resolve to a valid mode, factory would raise.
        tool = create_a2a_agent_tool(app=app)
        assert tool.name == "a2a_agent"


def test_mode2_app_no_obo_defaults_to_databricks_app_sp() -> None:
    with _mocked_url():
        with patch("dao_ai.tools.a2a_agent._ambient_workspace_client") as ambient:
            ambient.return_value.config.authenticate.return_value = {
                "Authorization": "Bearer M2M"
            }
            app = DatabricksAppModel(name="supplier", on_behalf_of_user=False)
            tool = create_a2a_agent_tool(app=app)
            assert tool.name == "a2a_agent"


def test_mode2_app_unset_obo_defaults_to_databricks_app_sp() -> None:
    with _mocked_url():
        with patch("dao_ai.tools.a2a_agent._ambient_workspace_client") as ambient:
            ambient.return_value.config.authenticate.return_value = {
                "Authorization": "Bearer M2M"
            }
            app = DatabricksAppModel(name="supplier")  # on_behalf_of_user unset
            tool = create_a2a_agent_tool(app=app)
            assert tool.name == "a2a_agent"


def test_mode2_dict_input_coerced_to_model() -> None:
    with _mocked_url():
        tool = create_a2a_agent_tool(
            app={"name": "supplier", "on_behalf_of_user": True}
        )
        assert tool.name == "a2a_agent"


def test_mode2_explicit_auth_type_overrides_app_default() -> None:
    """Pinning auth_type='databricks_app_sp' on an OBO-flagged app works."""
    with _mocked_url():
        with patch("dao_ai.tools.a2a_agent._ambient_workspace_client") as ambient:
            ambient.return_value.config.authenticate.return_value = {
                "Authorization": "Bearer M2M"
            }
            app = DatabricksAppModel(name="supplier", on_behalf_of_user=True)
            tool = create_a2a_agent_tool(app=app, auth_type="databricks_app_sp")
            assert tool.name == "a2a_agent"


def test_mode2_app_and_endpoint_warns_and_app_wins(caplog) -> None:
    """Both args set → loguru warning + ``app`` wins (no exception)."""
    from loguru import logger

    sink_handler_id = logger.add(
        lambda msg: caplog.records.append(msg), level="WARNING"
    )
    try:
        with _mocked_url("https://from-app.example.com"):
            app = DatabricksAppModel(name="supplier", on_behalf_of_user=True)
            tool = create_a2a_agent_tool(
                endpoint="https://from-endpoint.example.com", app=app
            )
            assert tool.name == "a2a_agent"
    finally:
        logger.remove(sink_handler_id)
    # Confirm a warning was emitted mentioning the resolution.
    assert any("app" in str(r) and "wins" in str(r) for r in caplog.records)


def test_neither_endpoint_nor_app_raises() -> None:
    with pytest.raises(ValueError, match="endpoint.*app"):
        create_a2a_agent_tool()


# ---------------------------------------------------------------------------
# Auth mode registry
# ---------------------------------------------------------------------------


def test_mode2_factory_does_not_resolve_app_url_eagerly() -> None:
    """``dao-ai validate`` must not require the bound app to be deployed."""
    with patch.object(DatabricksAppModel, "url", new_callable=PropertyMock) as url_prop:
        url_prop.side_effect = AssertionError(
            "app.url should NOT be accessed at factory build time"
        )
        app = DatabricksAppModel(name="supplier", on_behalf_of_user=True)
        # Must not raise — URL resolution is deferred to first tool call.
        tool = create_a2a_agent_tool(app=app)
        assert tool.name == "a2a_agent"


def test_valid_auth_modes_includes_new_modes() -> None:
    assert "forwarded_user_token" in _VALID_AUTH_MODES
    assert "databricks_app_sp" in _VALID_AUTH_MODES
    # Existing modes remain.
    assert {"bearer", "gcp_service_account", "none"}.issubset(set(_VALID_AUTH_MODES))
