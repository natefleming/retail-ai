"""Tests for the memory viewer + session-metadata helpers (Console batch)."""

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from dao_ai.apps.memory import load_user_memory
from dao_ai.apps.sessions import (
    list_user_sessions,
    load_session_meta,
    register_session,
    user_id_from_headers,
)


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


class _FakeIndexStore:
    """Minimal BaseStore stand-in: namespace-keyed aput + prefix asearch."""

    def __init__(self) -> None:
        self.data: dict[tuple, dict[str, Any]] = {}

    async def aput(self, namespace, key, value):  # noqa: ANN001
        self.data.setdefault(tuple(namespace), {})[key] = value

    async def asearch(self, namespace_prefix, *, query=None, limit=10, **kw):  # noqa: ANN001
        return [
            SimpleNamespace(namespace=tuple(namespace_prefix), key=k, value=v, updated_at=None)
            for k, v in self.data.get(tuple(namespace_prefix), {}).items()
        ][:limit]


class TestSessionIndex:
    @pytest.mark.unit
    def test_register_and_list_is_user_scoped(self) -> None:
        store = _FakeIndexStore()
        _run(register_session(store, "u1", "t1", "First"))
        _run(register_session(store, "u1", "t2", "Second"))
        _run(register_session(store, "u2", "tX", "Other"))
        rows = _run(list_user_sessions(store, "u1"))
        assert {r["thread_id"] for r in rows} == {"t1", "t2"}
        # u2's thread must never surface for u1.
        assert _run(list_user_sessions(store, "u2"))[0]["thread_id"] == "tX"

    @pytest.mark.unit
    def test_list_orders_most_recent_first(self) -> None:
        store = _FakeIndexStore()
        store.data[("sessions", "u1")] = {
            "old": {"title": "Old", "updated_at": "2026-01-01T00:00:00+00:00"},
            "new": {"title": "New", "updated_at": "2026-09-01T00:00:00+00:00"},
        }
        rows = _run(list_user_sessions(store, "u1"))
        assert [r["thread_id"] for r in rows] == ["new", "old"]

    @pytest.mark.unit
    def test_register_writes_under_sessions_namespace(self) -> None:
        store = _FakeIndexStore()
        _run(register_session(store, "u1", "t1", "Hi"))
        # Kept out of the ("memory", …) namespace so the memory viewer is unaffected.
        assert ("sessions", "u1") in store.data
        assert store.data[("sessions", "u1")]["t1"]["title"] == "Hi"


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
