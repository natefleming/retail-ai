"""Unit tests for the ``strict`` kwarg on ``IsDatabricksResource.workspace_client_from``."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from dao_ai.auth import OBONotAvailableError
from dao_ai.config import DatabricksAppModel, LLMModel
from dao_ai.state import Context


def _ws_with_authenticate(token: str = "Bearer ambient") -> MagicMock:
    ws = MagicMock(name="WorkspaceClient")
    ws.config.authenticate.return_value = {"Authorization": token}
    return ws


class TestStrictModeOBO:
    def test_strict_true_raises_when_obo_set_and_no_context(self) -> None:
        """`strict=True` + on_behalf_of_user + context=None → raise."""
        model = DatabricksAppModel(name="r1", on_behalf_of_user=True)
        with (
            patch.object(
                DatabricksAppModel,
                "workspace_client",
                new=_ws_with_authenticate(),
            ),
            pytest.raises(OBONotAvailableError) as exc,
        ):
            model.workspace_client_from(None, strict=True)
        assert exc.value.resource_name == "r1"

    def test_strict_true_raises_when_obo_set_and_context_has_no_headers(self) -> None:
        model = DatabricksAppModel(name="r2", on_behalf_of_user=True)
        ctx = Context(headers=None)
        with (
            patch.object(
                DatabricksAppModel,
                "workspace_client",
                new=_ws_with_authenticate(),
            ),
            pytest.raises(OBONotAvailableError),
        ):
            model.workspace_client_from(ctx, strict=True)

    def test_strict_true_raises_when_headers_missing_forwarded_token(self) -> None:
        model = DatabricksAppModel(name="r3", on_behalf_of_user=True)
        ctx = Context(headers={"unrelated-header": "value"})
        with (
            patch.object(
                DatabricksAppModel,
                "workspace_client",
                new=_ws_with_authenticate(),
            ),
            pytest.raises(OBONotAvailableError),
        ):
            model.workspace_client_from(ctx, strict=True)

    def test_strict_true_succeeds_when_forwarded_token_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Happy path: strict=True + OBO + token → builds a PAT-auth WC.

        Patches WorkspaceClient construction so the SDK doesn't try DNS
        host-metadata resolution against a fake hostname.
        """
        model = DatabricksAppModel(name="r4", on_behalf_of_user=True)
        ctx = Context(headers={"x-forwarded-access-token": "user-token-xyz"})

        captured_kwargs: dict = {}

        def fake_ws_init(**kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock(name="OBO-WC")

        monkeypatch.setattr("dao_ai.config.WorkspaceClient", fake_ws_init)

        ws = model.workspace_client_from(ctx, strict=True)
        assert ws is not None
        # Confirm the right auth shape was passed to WorkspaceClient(...)
        assert captured_kwargs["token"] == "user-token-xyz"
        assert captured_kwargs["auth_type"] == "pat"

    def test_strict_true_with_obo_false_falls_back(self) -> None:
        """When the resource doesn't ask for OBO, strict has no effect."""
        model = DatabricksAppModel(name="r5", on_behalf_of_user=False)
        fake_ws = _ws_with_authenticate()
        with patch.object(DatabricksAppModel, "workspace_client", new=fake_ws):
            ws = model.workspace_client_from(None, strict=True)
        assert ws is fake_ws

    def test_strict_false_silently_falls_back_when_obo_set_no_token(self) -> None:
        """Default lenient behavior preserved when strict=False (the default)."""
        model = DatabricksAppModel(name="r6", on_behalf_of_user=True)
        fake_ws = _ws_with_authenticate()
        with patch.object(DatabricksAppModel, "workspace_client", new=fake_ws):
            # Note: no strict kwarg → defaults to False
            ws = model.workspace_client_from(None)
        # Falls back silently — historical behavior preserved
        assert ws is fake_ws


class TestStrictModeAppliesAcrossResources:
    """The strict kwarg is on IsDatabricksResource so every resource subclass inherits it."""

    def test_llm_model_inherits_strict_mode(self) -> None:
        llm = LLMModel(name="endpoint-x", on_behalf_of_user=True)
        with pytest.raises(OBONotAvailableError):
            llm.workspace_client_from(None, strict=True)
