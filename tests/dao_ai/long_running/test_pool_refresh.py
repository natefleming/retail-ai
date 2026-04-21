"""Verify the Postgres pool calls back into DatabaseModel.connection_params
on every new connection so Lakebase OAuth tokens stay fresh."""

from __future__ import annotations

from unittest.mock import MagicMock

from dao_ai.memory.postgres import _make_kwargs_provider


def test_kwargs_provider_reresolves_per_call():
    """The kwargs provider calls DatabaseModel.connection_params every invocation.

    The pool uses this to mint a fresh Lakebase OAuth token per new connection.
    """
    call_counter = {"count": 0}

    class _FakeDatabase:
        @property
        def connection_params(self) -> dict:
            call_counter["count"] += 1
            return {
                "host": "example.com",
                "port": 5432,
                "dbname": "db",
                "user": "sp",
                "password": f"token_{call_counter['count']}",
            }

    provider = _make_kwargs_provider(
        database=_FakeDatabase(),
        static_kwargs={"autocommit": True},
        fallback_params={},
    )

    first = provider()
    second = provider()

    # The provider is invoked on each call.
    assert call_counter["count"] == 2
    # Fresh password per call.
    assert first["password"] == "token_1"
    assert second["password"] == "token_2"
    # Static psycopg session options preserved.
    assert first["autocommit"] is True
    assert second["autocommit"] is True


def test_kwargs_provider_uses_fallback_when_database_is_none():
    """Without a DatabaseModel, the provider returns the fallback statically."""
    provider = _make_kwargs_provider(
        database=None,
        static_kwargs={"autocommit": True},
        fallback_params={"password": "static", "host": "h", "port": 5432},
    )

    first = provider()
    second = provider()

    assert first == second
    assert first["password"] == "static"
    assert first["autocommit"] is True


def test_kwargs_provider_database_overrides_static_on_conflict():
    """DatabaseModel params override static kwargs when keys overlap."""
    db = MagicMock()
    db.connection_params = {"password": "from-db", "host": "h"}

    provider = _make_kwargs_provider(
        database=db,
        static_kwargs={"password": "from-static", "autocommit": True},
        fallback_params={},
    )

    result = provider()
    assert result["password"] == "from-db"
    assert result["autocommit"] is True
