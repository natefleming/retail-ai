"""Unit tests for the Databricks Managed Agent Memory ``BaseStore`` backend.

The live UC round-trip is covered by fevm e2e; here we drive
:class:`~dao_ai.memory.agent_memory.AgentMemoryStore` against an in-memory fake
of the memory-store REST API, and check the config selection / validation logic.

The repo doesn't use pytest-asyncio, so coroutines are driven via
:func:`asyncio.run` (matching ``test_best_of_n.py``).
"""

import asyncio

import pytest
from databricks.sdk.errors import AlreadyExists, NotFound
from langgraph.store.base import (
    NOT_PROVIDED,
    GetOp,
    ListNamespacesOp,
    MatchCondition,
    PutOp,
    SearchItem,
    SearchOp,
)

from dao_ai.config import MemoryStoreModel, StorageType, StoreModel
from dao_ai.memory.agent_memory import (
    AgentMemoryStore,
    AgentMemoryStoreManager,
    _decode_path,
    _encode_path,
    _matches_namespace,
    _namespace_prefix_path,
)


class _FakeApiClient:
    """Minimal in-memory emulation of the memory-store entries REST surface."""

    def __init__(self) -> None:
        # (scope, path) -> entry dict
        self.entries: dict[tuple[str, str], dict] = {}
        self.calls: list[tuple[str, str]] = []
        self.search_bodies: list[dict] = []
        # optionally force PATCH to raise NotFound (simulate concurrent delete)
        self.patch_raises_notfound = False

    def do(self, method, path, *, query=None, body=None, **_):
        query = query or {}
        body = body or {}
        self.calls.append((method, path))
        scope = query.get("scope") or body.get("scope")

        if method == "POST" and path.endswith("/entries:search"):
            # Real shape: {"results": [{"memory_entry": {...}, "score": f}]}.
            # The real API scopes by path_prefix server-side.
            self.search_bodies.append(dict(body))
            q = (body.get("query") or "").lower()
            pref = body.get("path_prefix")
            results = [
                {"memory_entry": dict(e), "score": 0.5}
                for (s, p), e in self.entries.items()
                if s == scope
                and q in e["contents"].lower()
                and (pref is None or p.startswith(pref))
            ]
            return {"results": results}

        if method == "GET" and path.endswith("/entries:get"):
            key = (scope, query["path"])
            if key not in self.entries:
                raise NotFound(f"no entry {key}")
            return dict(self.entries[key])

        if method == "POST" and path.endswith("/entries"):
            key = (scope, body["path"])
            if key in self.entries:
                raise AlreadyExists(f"entry exists {key}")
            self.entries[key] = {
                "path": body["path"],
                "contents": body["contents"],
                "create_time": "2026-01-01T00:00:00Z",
                "update_time": "2026-01-01T00:00:00Z",
            }
            return dict(self.entries[key])

        if method == "PATCH" and path.endswith("/entries"):
            key = (scope, body["path"])
            if self.patch_raises_notfound:
                # simulate a concurrent delete: the entry is gone, PATCH 404s
                self.entries.pop(key, None)
                raise NotFound(f"no entry {key}")
            if key not in self.entries:
                raise NotFound(f"no entry {key}")
            self.entries[key]["contents"] = body["replace_all"]["contents"]
            return dict(self.entries[key])

        if method == "DELETE" and path.endswith("/entries"):
            key = (scope, query["path"])
            if key not in self.entries:
                raise NotFound(f"no entry {key}")
            del self.entries[key]
            return {}

        if method == "GET" and path.endswith("/entries"):
            # Real shape: {"entries": [...]} and the list omits `contents`.
            prefix = query.get("path_prefix", "/")
            hits = [
                {k: v for k, v in e.items() if k != "contents"}
                for (s, p), e in self.entries.items()
                if s == scope and p.startswith(prefix)
            ]
            return {"entries": hits}

        raise AssertionError(f"unexpected call {method} {path}")


def _store(scope: str = "user-1") -> tuple[AgentMemoryStore, _FakeApiClient]:
    model = MemoryStoreModel(name="main.default.mem")
    store = AgentMemoryStore(memory_store=model, scope=scope, log_extra={})
    fake = _FakeApiClient()

    class _W:
        api_client = fake

    store._cached_client = _W()  # bypass real auth
    return store, fake


# --------------------------------------------------------------------------
# path <-> namespace encoding
# --------------------------------------------------------------------------


def test_path_roundtrip():
    # paths are rooted at /memories/ (a Managed Memory API requirement)
    ns, key = ("profiles", "user-123"), "pref1"
    p = _encode_path(ns, key)
    assert p == "/memories/profiles/user-123/pref1"
    assert _decode_path(p) == (ns, key)
    # a namespace whose first element is literally "memories" still round-trips
    assert _decode_path(_encode_path(("memories", "u1"), "k")) == (
        ("memories", "u1"),
        "k",
    )


def test_namespace_prefix_path():
    assert _namespace_prefix_path(("a", "b")) == "/memories/a/b/"
    assert _namespace_prefix_path(()) == "/memories/"


def test_encode_path_drops_empty_segments():
    # An empty namespace element (e.g. unresolved {user_id}) must not produce an
    # API-invalid path with an empty segment (`//`). Regression: a live aget with
    # namespace ("memory", "") key "default" produced "/memories/memory//default"
    # and the Managed Memory API rejected it ("no ... empty segments").
    assert _encode_path(("memory", ""), "default") == "/memories/memory/default"
    assert _namespace_prefix_path(("memory", "")) == "/memories/memory/"


def test_matches_namespace():
    assert _matches_namespace(("a", "b"), ("a",), None, None)
    assert not _matches_namespace(("a", "b"), ("x",), None, None)
    assert _matches_namespace(("a", "b", "c"), ("a",), ("c",), None)
    assert not _matches_namespace(("a", "b", "c"), None, None, 2)
    # wildcard segment
    assert _matches_namespace(("a", "b"), ("*", "b"), None, None)


# --------------------------------------------------------------------------
# CRUD round-trips
# --------------------------------------------------------------------------


def test_put_get_roundtrip():
    store, _ = _store()

    async def run():
        await store.aput(("memories",), "k1", {"fact": "likes email", "n": 3})
        item = await store.aget(("memories",), "k1")
        return item

    item = asyncio.run(run())
    assert item is not None
    assert item.namespace == ("memories",)
    assert item.key == "k1"
    assert item.value == {"fact": "likes email", "n": 3}


def test_get_missing_returns_none():
    store, _ = _store()
    assert asyncio.run(store.aget(("memories",), "nope")) is None


def test_put_upsert_replaces():
    store, fake = _store()

    async def run():
        await store.aput(("m",), "k", {"v": 1})
        await store.aput(("m",), "k", {"v": 2})  # triggers delete+recreate
        return await store.aget(("m",), "k")

    item = asyncio.run(run())
    assert item.value == {"v": 2}
    # upsert path issued a PATCH (replace_all), not a delete
    assert any(m == "PATCH" for m, _ in fake.calls)


def test_delete_then_get_none():
    store, _ = _store()

    async def run():
        await store.aput(("m",), "k", {"v": 1})
        await store.adelete(("m",), "k")
        return await store.aget(("m",), "k")

    assert asyncio.run(run()) is None


def test_delete_missing_is_noop():
    store, _ = _store()
    # NotFound is swallowed
    assert asyncio.run(store.adelete(("m",), "ghost")) is None


# --------------------------------------------------------------------------
# search + list_namespaces
# --------------------------------------------------------------------------


def test_search_query_scoped_server_side_by_path_prefix():
    store, fake = _store()

    async def run():
        await store.aput(("memories", "u1"), "a", {"text": "prefers dark mode"})
        await store.aput(("memories", "u1"), "b", {"text": "timezone is PST"})
        await store.aput(("other", "u1"), "c", {"text": "prefers dark mode too"})
        return await store.asearch(("memories",), query="prefers")

    results = asyncio.run(run())
    assert all(isinstance(r, SearchItem) for r in results)
    # the service's relevance score is passed through, not discarded
    assert all(r.score == 0.5 for r in results)
    # the query search is scoped server-side by path_prefix, so another
    # namespace's keyword-matching entry ("other") is never even returned
    assert fake.search_bodies[-1].get("path_prefix") == "/memories/memories/"
    assert {r.namespace for r in results} == {("memories", "u1")}
    assert {r.key for r in results} == {"a"}
    assert results[0].value == {"text": "prefers dark mode"}


def test_search_no_query_no_filter_hydrates_only_the_slice():
    store, fake = _store()

    async def run():
        for i in range(10):
            await store.aput(("m",), f"k{i}", {"i": i})
        fake.calls.clear()
        return await store.asearch(("m",), limit=3, offset=0)

    results = asyncio.run(run())
    assert len(results) == 3
    # only the requested slice is hydrated with :get (not all 10 entries)
    gets = sum(1 for m, p in fake.calls if m == "GET" and p.endswith("/entries:get"))
    assert gets == 3, f"expected 3 hydration gets, got {gets}"


def test_search_no_query_lists_by_prefix_with_filter():
    store, _ = _store()

    async def run():
        await store.aput(("m", "u1"), "a", {"kind": "pref"})
        await store.aput(("m", "u1"), "b", {"kind": "episode"})
        await store.aput(("m", "u2"), "c", {"kind": "pref"})
        return await store.asearch(("m", "u1"), filter={"kind": "pref"})

    results = asyncio.run(run())
    assert {r.key for r in results} == {"a"}


def test_search_offset_limit():
    store, _ = _store()

    async def run():
        for i in range(5):
            await store.aput(("m",), f"k{i}", {"i": i})
        return await store.asearch(("m",), limit=2, offset=1)

    results = asyncio.run(run())
    assert len(results) == 2


def test_list_namespaces_and_max_depth():
    store, _ = _store()

    async def run():
        await store.aput(("m", "u1"), "a", {"v": 1})
        await store.aput(("m", "u2"), "b", {"v": 2})
        await store.aput(("m", "u1", "deep"), "c", {"v": 3})
        full = await store.alist_namespaces()
        shallow = await store.alist_namespaces(max_depth=2)
        return full, shallow

    full, shallow = asyncio.run(run())
    assert ("m", "u1") in full and ("m", "u1", "deep") in full
    # max_depth truncates and dedups
    assert ("m", "u1", "deep") not in shallow
    assert ("m", "u1") in shallow and ("m", "u2") in shallow


def test_list_namespaces_suffix_matched_before_max_depth_truncation():
    store, _ = _store()

    async def run():
        await store.aput(("m", "u1", "deep"), "c", {"v": 1})
        await store.aput(("m", "u2"), "b", {"v": 2})
        # suffix must match the FULL namespace (…,'deep'); max_depth only
        # truncates the returned value. Truncating before matching would drop it.
        return await store.alist_namespaces(suffix=("deep",), max_depth=2)

    result = asyncio.run(run())
    # ('m','u1','deep') matches suffix on the full ns, returned truncated to depth 2
    assert ("m", "u1") in result
    assert ("m", "u2") not in result


def test_upsert_patch_notfound_falls_back_to_create():
    store, fake = _store()

    async def run():
        await store.aput(("m",), "k", {"v": 1})
        # simulate a concurrent delete: the replace_all PATCH hits a missing entry
        fake.patch_raises_notfound = True
        await store.aput(
            ("m",), "k", {"v": 2}
        )  # POST->AlreadyExists->PATCH(NotFound)->POST
        fake.patch_raises_notfound = False
        return await store.aget(("m",), "k")

    item = asyncio.run(run())
    # the fallback re-create wins; no exception propagated
    assert item is not None and item.value == {"v": 2}


# --------------------------------------------------------------------------
# TTL + sync-batch contracts
# --------------------------------------------------------------------------


def test_ttl_not_provided_ok_but_real_ttl_raises():
    store, _ = _store()
    # NOT_PROVIDED (the langgraph sentinel) is accepted
    asyncio.run(store.aput(("m",), "k", {"v": 1}, ttl=NOT_PROVIDED))
    # None accepted
    asyncio.run(store.aput(("m",), "k2", {"v": 1}, ttl=None))
    # a real TTL is rejected
    with pytest.raises(NotImplementedError):
        asyncio.run(store.aput(("m",), "k3", {"v": 1}, ttl=5.0))


def test_batch_sync_raises():
    store, _ = _store()
    with pytest.raises(NotImplementedError):
        store.batch([])


def test_abatch_dispatch():
    store, _ = _store()

    async def run():
        await store.abatch(
            [
                PutOp(("m", "u1"), "a", {"text": "hello world"}, None, NOT_PROVIDED),
                PutOp(("m", "u2"), "b", {"text": "goodbye"}, None, NOT_PROVIDED),
            ]
        )
        results = await store.abatch(
            [
                GetOp(("m", "u1"), "a", True),
                SearchOp(("m",), None, 10, 0, "hello", True),
                ListNamespacesOp(
                    (MatchCondition(match_type="prefix", path=("m",)),), None, 100, 0
                ),
                # value None = delete
                PutOp(("m", "u1"), "a", None, None, NOT_PROVIDED),
            ]
        )
        after = await store.aget(("m", "u1"), "a")
        return results, after

    results, after = asyncio.run(run())
    get_res, search_res, ns_res, put_res = results
    assert get_res.value == {"text": "hello world"}
    assert {r.key for r in search_res} == {"a"}
    assert set(ns_res) == {("m", "u1"), ("m", "u2")}
    assert put_res is None
    assert after is None  # deleted by the None-value PutOp


# --------------------------------------------------------------------------
# config selection / validation
# --------------------------------------------------------------------------


def test_storage_type_selects_agent_memory():
    sm = StoreModel(name="s", memory_store={"name": "main.default.mem"})
    assert sm.storage_type == StorageType.AGENT_MEMORY
    assert sm.memory_store.full_name == "main.default.mem"


def test_full_name_from_schema_and_short_name():
    sm = StoreModel(
        name="s",
        memory_store={
            "schema": {"catalog_name": "main", "schema_name": "default"},
            "name": "mem",
        },
    )
    assert sm.memory_store.full_name == "main.default.mem"


def test_database_and_memory_store_mutually_exclusive():
    with pytest.raises(Exception):
        StoreModel(name="s", database={"project": "p"}, memory_store={"name": "a.b.c"})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "just_short"},  # no schema, not fully qualified
        {
            "schema": {"catalog_name": "c", "schema_name": "s"},
            "name": "a.b.c",  # dots not allowed alongside schema
        },
    ],
)
def test_memory_store_identity_validation(kwargs):
    with pytest.raises(Exception):
        MemoryStoreModel(**kwargs)


def test_manager_ignores_embedding_model_and_defaults_scope(caplog):
    sm = StoreModel(
        name="s",
        namespace="tenant-a",
        memory_store={"name": "main.default.mem"},
    )
    mgr = AgentMemoryStoreManager(sm)
    mgr._setup()
    store = mgr.store()
    assert isinstance(store, AgentMemoryStore)
    assert store._scope == "tenant-a"  # falls back to StoreModel.namespace


def test_store_manager_distinguishes_scopes_for_same_store():
    # Two configs target the SAME UC memory store under DIFFERENT scopes; the
    # manager cache must not collide them (regression: keyed on full_name only).
    from dao_ai.memory.core import StoreManager

    a = StoreModel(
        name="store_scope_x",
        memory_store={"name": "main.default.shared_mem", "scope_value": "team-x"},
    )
    b = StoreModel(
        name="store_scope_y",
        memory_store={"name": "main.default.shared_mem", "scope_value": "team-y"},
    )
    store_a = StoreManager.instance(a).store()
    store_b = StoreManager.instance(b).store()
    assert store_a is not store_b
    assert store_a._scope == "team-x"
    assert store_b._scope == "team-y"
