"""Tests for the memory viewer + session-metadata helpers (Console batch)."""

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from dao_ai.apps.memory import load_user_memory
from dao_ai.apps.sessions import load_session_meta, user_id_from_headers


def _run(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        asyncio.set_event_loop(asyncio.new_event_loop())


class TestUserIdFromHeaders:
    @pytest.mark.unit
    def test_derives_and_normalizes(self) -> None:
        assert user_id_from_headers({"x-forwarded-user": "nate.fleming"}) == "nate_fleming"

    @pytest.mark.unit
    def test_case_insensitive_header(self) -> None:
        assert user_id_from_headers({"X-Forwarded-User": "a.b"}) == "a_b"

    @pytest.mark.unit
    def test_absent_returns_none(self) -> None:
        assert user_id_from_headers({}) is None


class _FakeStore:
    def __init__(self, items: list[Any]) -> None:
        self._items = items

    async def asearch(self, prefix, *, query=None, limit=10, **kw):  # noqa: ANN001
        return self._items


class TestLoadUserMemory:
    @pytest.mark.unit
    def test_groups_items_by_namespace(self) -> None:
        items = [
            SimpleNamespace(
                namespace=("memory", "nate", "user_profile"),
                key="default",
                value={"name": "Nate"},
                created_at=None,
                updated_at=None,
            ),
            SimpleNamespace(
                namespace=("memory", "nate", "episodes"),
                key="ep1",
                value={"situation": "asked about X"},
                created_at=None,
                updated_at=None,
            ),
        ]
        out = _run(load_user_memory(_FakeStore(items), "nate"))
        assert out["user_id"] == "nate"
        assert "memory/nate/user_profile" in out["memory"]
        assert "memory/nate/episodes" in out["memory"]
        assert out["memory"]["memory/nate/user_profile"][0]["value"] == {"name": "Nate"}
        assert set(out["namespaces"]) == {
            "memory/nate/user_profile",
            "memory/nate/episodes",
        }


class _FakeGraph:
    def __init__(self, snapshot) -> None:  # noqa: ANN001
        self._snapshot = snapshot

    async def aget_state(self, config):  # noqa: ANN001
        return self._snapshot


class TestLoadSessionMeta:
    @pytest.mark.unit
    def test_extracts_checkpoint_metadata(self) -> None:
        snapshot = SimpleNamespace(
            values={"messages": [1, 2, 3]},
            config={"configurable": {"checkpoint_id": "ckpt-9"}},
            metadata={"step": 4},
            created_at="2026-09-02T18:00:00+00:00",
        )
        meta = _run(load_session_meta(_FakeGraph(snapshot), "t-1"))
        assert meta["thread_id"] == "t-1"
        assert meta["checkpoint_id"] == "ckpt-9"
        assert meta["last_modified"] == "2026-09-02T18:00:00+00:00"
        assert meta["step"] == 4
        assert meta["message_count"] == 3
