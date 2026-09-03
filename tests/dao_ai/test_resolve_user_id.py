"""Tests for resolve_user_id_from_headers — prefers the login name over the
numeric x-forwarded-user id so user_id is human-readable and consistent across
Databricks Apps and model serving."""

import pytest

from dao_ai.models import resolve_user_id_from_headers


class TestResolveUserIdFromHeaders:
    @pytest.mark.unit
    def test_prefers_preferred_username(self) -> None:
        headers = {
            "x-forwarded-preferred-username": "nate.fleming",
            "x-forwarded-email": "nate.fleming@databricks.com",
            "x-forwarded-user": "8687208780835288@7474649850651417",
        }
        assert resolve_user_id_from_headers(headers) == "nate.fleming"

    @pytest.mark.unit
    def test_falls_back_to_email(self) -> None:
        headers = {
            "x-forwarded-email": "nate.fleming@databricks.com",
            "x-forwarded-user": "8687208780835288@7474649850651417",
        }
        assert resolve_user_id_from_headers(headers) == "nate.fleming@databricks.com"

    @pytest.mark.unit
    def test_falls_back_to_numeric_user(self) -> None:
        headers = {"X-Forwarded-User": "8687208780835288@7474649850651417"}
        assert resolve_user_id_from_headers(headers) == "8687208780835288@7474649850651417"

    @pytest.mark.unit
    def test_case_insensitive(self) -> None:
        headers = {"X-Forwarded-Preferred-Username": "someone"}
        assert resolve_user_id_from_headers(headers) == "someone"

    @pytest.mark.unit
    def test_none_when_absent(self) -> None:
        assert resolve_user_id_from_headers({}) is None
        assert resolve_user_id_from_headers(None) is None
