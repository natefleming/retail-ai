"""Databricks Managed Agent Memory store backend for dao-ai long-term memory.

Implements LangGraph's :class:`~langgraph.store.base.BaseStore` on top of
Databricks' **Managed Agent Memory** API (Beta, ~2026): Unity-Catalog-governed
memory stores addressed by a 3-part name (``catalog.schema.store``) holding
memory entries keyed by ``(scope, path) -> free-form text contents``, with
key-based get, keyword search, and list-by-prefix.

Mapping to ``BaseStore`` (single configured scope; the namespace tuple is encoded
into the entry ``path`` so that prefix search / namespace listing — which carry no
scope argument — work uniformly):

- ``namespace`` tuple + ``key`` -> entry ``path`` ``/memories/ns0/ns1/.../key``
  (the Managed Memory API requires entry paths to be rooted at ``/memories/``)
- ``value`` dict            -> entry ``contents`` (JSON)
- ``search(query=...)``     -> the store's ``entries:search`` endpoint
- ``search()`` / ``list_namespaces`` -> list entries by ``path_prefix``

NOT a checkpointer: the Managed Memory API has no graph-state checkpoint concept
(channel blobs, versions, pending writes, parent chains), so it cannot back a
``BaseCheckpointSaver``. Pair this store with a Lakebase / Postgres / in-memory
checkpointer.

Fidelity notes:

- ``search(query=...)`` returns the Managed Memory service's relevance ``score``
  on each ``SearchItem`` (its ``entries:search`` endpoint ranks results). The
  ``index`` argument to ``put`` is ignored — indexing is managed by the service.
- The list endpoint omits entry ``contents`` (returns only ``has_contents``), so
  the no-query ``search`` path fetches each matched entry's contents individually.
- Writes are **eventually consistent**: create / ``replace_all`` / delete return
  the authoritative result, but a read immediately afterward can briefly observe
  the prior state (typically sub-second). Fine for cross-session long-term memory;
  a write-then-immediate-read in the same turn may lag.
- TTL is unsupported (``supports_ttl = False``): a real ``ttl`` raises, while the
  langgraph ``NOT_PROVIDED`` sentinel and ``None`` are accepted and ignored.
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Iterable, Literal

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import AlreadyExists, NotFound
from langgraph.store.base import (
    NOT_PROVIDED,
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    NotProvided,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)
from loguru import logger

from dao_ai.config import MemoryStoreModel, StoreModel, value_of
from dao_ai.memory.base import StoreManagerBase

_API_BASE = "/api/2.1/unity-catalog/memory-stores"
_PAGE_SIZE = 200
# The Managed Memory API requires entry paths to be rooted here; the BaseStore
# namespace is encoded beneath it.
_PATH_ROOT = "memories"


def _encode_path(namespace: tuple[str, ...], key: str) -> str:
    """Encode a ``(namespace, key)`` pair into an entry path ``/memories/ns.../key``.

    Empty segments are dropped: the Managed Memory API rejects paths with empty
    segments (``//``), which arise when a namespace element is "" (e.g. an
    unresolved ``{user_id}``). Dropping them keeps the path valid and consistent
    with :func:`_decode_path`, which also ignores empty segments.
    """
    segments = [s for s in (_PATH_ROOT, *namespace, key) if s != ""]
    return "/" + "/".join(segments)


def _namespace_prefix_path(namespace_prefix: tuple[str, ...]) -> str:
    """Path prefix for listing/searching entries under a namespace prefix."""
    segments = [s for s in (_PATH_ROOT, *namespace_prefix) if s != ""]
    return "/" + "/".join(segments) + "/"


def _decode_path(path: str) -> tuple[tuple[str, ...], str]:
    """Decode an entry path (``/memories/ns.../key``) back into ``(namespace, key)``."""
    parts = [p for p in path.split("/") if p != ""]
    if parts and parts[0] == _PATH_ROOT:
        parts = parts[1:]
    if not parts:
        return (), ""
    return tuple(parts[:-1]), parts[-1]


def _parse_contents(contents: str | None) -> dict[str, Any]:
    """Parse an entry's ``contents`` back into a value dict."""
    if not contents:
        return {}
    try:
        parsed = json.loads(contents)
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}
    except (json.JSONDecodeError, TypeError):
        # Not written by this store (e.g. a human-authored markdown memory).
        return {"content": contents}


def _parse_ts(entry: dict[str, Any], *keys: str) -> datetime:
    for k in keys:
        raw = entry.get(k)
        if isinstance(raw, str) and raw:
            try:
                return datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                continue
    return datetime.now(timezone.utc)


def _matches_namespace(
    namespace: tuple[str, ...],
    prefix: tuple[str, ...] | None,
    suffix: tuple[str, ...] | None,
    max_depth: int | None,
) -> bool:
    def _seg_eq(a: str, b: str) -> bool:
        return b == "*" or a == b

    if prefix:
        if len(namespace) < len(prefix):
            return False
        if not all(_seg_eq(a, b) for a, b in zip(namespace[: len(prefix)], prefix)):
            return False
    if suffix:
        if len(namespace) < len(suffix):
            return False
        if not all(_seg_eq(a, b) for a, b in zip(namespace[-len(suffix) :], suffix)):
            return False
    if max_depth is not None and len(namespace) > max_depth:
        return False
    return True


class AgentMemoryStore(BaseStore):
    """``BaseStore`` backed by a Databricks Managed Agent Memory store.

    Async-only: ``batch`` raises. All I/O goes through the Databricks SDK HTTP
    client (which handles OBO/SP/PAT/ambient auth) wrapped in
    :func:`asyncio.to_thread` so the event loop is never blocked.
    """

    supports_ttl: bool = False

    def __init__(
        self,
        memory_store: MemoryStoreModel,
        scope: str,
        log_extra: dict[str, Any],
    ):
        # BaseStore is abstract on ``batch`` / ``abatch``; we satisfy those via
        # the overrides below, so we intentionally skip super().__init__().
        self._memory_store = memory_store
        self._scope = scope
        self._log_extra = log_extra
        self._full_name = memory_store.full_name
        self._entries_path = f"{_API_BASE}/{self._full_name}/entries"
        self._cached_client: WorkspaceClient | None = None

    # -- auth / transport ---------------------------------------------------

    def _client(self) -> WorkspaceClient:
        # OBO tokens are request-scoped, so resolve a fresh client each call;
        # SP/PAT/ambient are stable and cheap to cache.
        if self._memory_store.on_behalf_of_user:
            return self._memory_store.workspace_client
        if self._cached_client is None:
            self._cached_client = self._memory_store.workspace_client
        return self._cached_client

    async def _do(
        self,
        method: str,
        path: str,
        *,
        query: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> Any:
        client = self._client()
        return await asyncio.to_thread(
            client.api_client.do, method, path, query=query, body=body
        )

    async def _list_entries(self, path_prefix: str) -> list[dict[str, Any]]:
        """List all entries under ``path_prefix`` in the configured scope."""
        entries: list[dict[str, Any]] = []
        page_token: str | None = None
        while True:
            query: dict[str, Any] = {
                "scope": self._scope,
                "path_prefix": path_prefix,
                "page_size": _PAGE_SIZE,
            }
            if page_token:
                query["page_token"] = page_token
            resp = await self._do("GET", self._entries_path, query=query)
            resp = resp or {}
            page = resp.get("memory_entries") or resp.get("entries") or []
            entries.extend(page)
            page_token = resp.get("next_page_token") or resp.get("page_token")
            if not page_token:
                break
        return entries

    def _to_item(self, entry: dict[str, Any]) -> tuple[tuple[str, ...], str, Item]:
        namespace, key = _decode_path(entry.get("path", ""))
        item = Item(
            value=_parse_contents(entry.get("contents")),
            key=key,
            namespace=namespace,
            created_at=_parse_ts(entry, "created_at", "create_time"),
            updated_at=_parse_ts(entry, "updated_at", "update_time", "last_updated"),
        )
        return namespace, key, item

    # -- BaseStore async surface -------------------------------------------

    async def aget(
        self,
        namespace: tuple[str, ...],
        key: str,
        *,
        refresh_ttl: bool | None = None,
    ) -> Item | None:
        path = _encode_path(namespace, key)
        with mlflow.start_span(name="agent_memory.aget", span_type="RETRIEVER") as span:
            span.set_attributes({"memory_store": self._full_name, "path": path})
            try:
                entry = await self._do(
                    "GET",
                    f"{self._entries_path}:get",
                    query={"scope": self._scope, "path": path},
                )
            except NotFound:
                return None
        if not entry:
            return None
        _, _, item = self._to_item({**entry, "path": path})
        return item

    async def aput(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Literal[False] | list[str] | None = None,
        *,
        ttl: float | None | NotProvided = NOT_PROVIDED,
    ) -> None:
        # supports_ttl is False: accept the langgraph NOT_PROVIDED sentinel (and
        # None) and ignore, but reject a real TTL rather than silently drop it.
        if ttl is not NOT_PROVIDED and ttl is not None:
            raise NotImplementedError(
                "AgentMemoryStore does not support TTL; the Managed Agent Memory "
                "API has no per-entry expiry."
            )
        path = _encode_path(namespace, key)
        contents = json.dumps(value)
        with mlflow.start_span(name="agent_memory.aput", span_type="RETRIEVER") as span:
            span.set_attributes({"memory_store": self._full_name, "path": path})
            try:
                await self._do(
                    "POST",
                    self._entries_path,
                    query={"scope": self._scope},
                    body={"path": path, "contents": contents},
                )
            except AlreadyExists:
                # Upsert: overwrite the existing entry via the partial-edit
                # endpoint's replace_all operation. If a concurrent delete removed
                # the entry between the failed POST and this PATCH (writes are
                # eventually consistent), recreate it instead of failing.
                try:
                    await self._do(
                        "PATCH",
                        self._entries_path,
                        body={
                            "scope": self._scope,
                            "path": path,
                            "replace_all": {"contents": contents},
                        },
                    )
                except NotFound:
                    await self._do(
                        "POST",
                        self._entries_path,
                        query={"scope": self._scope},
                        body={"path": path, "contents": contents},
                    )

    async def adelete(self, namespace: tuple[str, ...], key: str) -> None:
        path = _encode_path(namespace, key)
        with mlflow.start_span(
            name="agent_memory.adelete", span_type="RETRIEVER"
        ) as span:
            span.set_attributes({"memory_store": self._full_name, "path": path})
            try:
                await self._do(
                    "DELETE",
                    self._entries_path,
                    query={"scope": self._scope, "path": path},
                )
            except NotFound:
                return None

    async def asearch(
        self,
        namespace_prefix: tuple[str, ...],
        /,
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool | None = None,
    ) -> list[SearchItem]:
        prefix_path = _namespace_prefix_path(namespace_prefix)
        with mlflow.start_span(
            name="agent_memory.asearch", span_type="RETRIEVER"
        ) as span:
            span.set_attributes(
                {
                    "memory_store": self._full_name,
                    "namespace_prefix": prefix_path,
                    "query": query or "",
                    "keyword_only": True,
                }
            )
            if query:
                # Scope the search to the namespace server-side via path_prefix
                # (the API supports it), so a shared scope can't leak other
                # namespaces' entries into a caller's results.
                resp = await self._do(
                    "POST",
                    f"{self._entries_path}:search",
                    body={
                        "scope": self._scope,
                        "query": query,
                        "path_prefix": prefix_path,
                    },
                )
                resp = resp or {}
                # entries:search -> {"results": [{"memory_entry": {...}, "score": f}]}
                scored = [
                    (r.get("memory_entry") or {}, r.get("score"))
                    for r in (resp.get("results") or [])
                ]
            else:
                # The list endpoint omits contents, so entries must be hydrated
                # with a per-entry :get. Without a value filter only the requested
                # slice is needed, so hydrate just that; with a filter, values are
                # needed to evaluate it. Hydrate concurrently either way.
                listed = await self._list_entries(prefix_path)
                candidates = [
                    e
                    for e in listed
                    if _matches_namespace(
                        _decode_path(e.get("path", ""))[0],
                        namespace_prefix,
                        None,
                        None,
                    )
                ]
                to_hydrate = (
                    candidates if filter else candidates[offset : offset + limit]
                )
                fulls = await asyncio.gather(
                    *(
                        self._do(
                            "GET",
                            f"{self._entries_path}:get",
                            query={"scope": self._scope, "path": e.get("path", "")},
                        )
                        for e in to_hydrate
                    )
                )
                scored = [(full or e, None) for full, e in zip(fulls, to_hydrate)]

        results: list[SearchItem] = []
        for entry, score in scored:
            namespace, key, item = self._to_item(entry)
            # Safety net (query path already scoped server-side by path_prefix).
            if not _matches_namespace(namespace, namespace_prefix, None, None):
                continue
            if filter and not all(
                item.value.get(fk) == fv for fk, fv in filter.items()
            ):
                continue
            results.append(
                SearchItem(
                    namespace=namespace,
                    key=key,
                    value=item.value,
                    created_at=item.created_at,
                    updated_at=item.updated_at,
                    score=score,
                )
            )
        # The no-query, no-filter path already applied offset/limit to candidates.
        if query is None and not filter:
            return results
        return results[offset : offset + limit]

    async def alist_namespaces(
        self,
        *,
        prefix: tuple[str, ...] | None = None,
        suffix: tuple[str, ...] | None = None,
        max_depth: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        prefix_path = _namespace_prefix_path(prefix or ())
        entries = await self._list_entries(prefix_path)
        seen: dict[tuple[str, ...], None] = {}
        for entry in entries:
            namespace, _ = _decode_path(entry.get("path", ""))
            # Match prefix/suffix against the full namespace, then truncate to
            # max_depth for the returned (deduped) result.
            if not _matches_namespace(namespace, prefix, suffix, None):
                continue
            if max_depth is not None:
                namespace = namespace[:max_depth]
            seen[namespace] = None
        ordered = list(seen.keys())
        return ordered[offset : offset + limit]

    # -- batch dispatch -----------------------------------------------------

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        results: list[Result] = []
        for op in ops:
            if isinstance(op, GetOp):
                results.append(await self.aget(op.namespace, op.key))
            elif isinstance(op, PutOp):
                if op.value is None:
                    await self.adelete(op.namespace, op.key)
                else:
                    await self.aput(op.namespace, op.key, op.value, op.index)
                results.append(None)
            elif isinstance(op, SearchOp):
                results.append(
                    await self.asearch(
                        op.namespace_prefix,
                        query=op.query,
                        filter=op.filter,
                        limit=op.limit,
                        offset=op.offset,
                    )
                )
            elif isinstance(op, ListNamespacesOp):
                prefix: tuple[str, ...] | None = None
                suffix: tuple[str, ...] | None = None
                for cond in op.match_conditions or ():
                    if cond.match_type == "prefix":
                        prefix = cond.path
                    elif cond.match_type == "suffix":
                        suffix = cond.path
                results.append(
                    await self.alist_namespaces(
                        prefix=prefix,
                        suffix=suffix,
                        max_depth=op.max_depth,
                        limit=op.limit,
                        offset=op.offset,
                    )
                )
            else:  # pragma: no cover - defensive
                raise ValueError(f"Unsupported store op: {type(op).__name__}")
        return results

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        # The Managed Memory REST client is driven async; the agent never calls
        # the store from a sync context.
        raise NotImplementedError(
            "AgentMemoryStore is async-only; use abatch from an async context"
        )


class AgentMemoryStoreManager(StoreManagerBase):
    """Store backed by the Databricks Managed Agent Memory API.

    Sync ``_setup`` only validates config and resolves the scope; the store does
    all I/O lazily on first ``await`` via the SDK HTTP client.
    """

    def __init__(self, store_model: StoreModel):
        self.store_model = store_model
        self._store: BaseStore | None = None
        self._setup_complete = False

    def store(self) -> BaseStore:
        if not self._setup_complete or not self._store:
            self._setup()
        if not self._store:
            raise RuntimeError("AgentMemoryStoreManager initialization failed")
        return self._store

    def _setup(self) -> None:
        if self._setup_complete:
            return

        memory_store = self.store_model.memory_store
        if memory_store is None:
            raise ValueError(
                "memory_store configuration is required for the Agent Memory store"
            )

        if self.store_model.embedding_model is not None:
            logger.warning(
                "Agent Memory store ignores embedding_model: the Managed Agent "
                "Memory API provides keyword search only, not vector search.",
                store=self.store_model.name,
            )

        scope = (
            value_of(memory_store.scope_value)
            if memory_store.scope_value is not None
            else (self.store_model.namespace or "default")
        )

        self._store = AgentMemoryStore(
            memory_store=memory_store,
            scope=scope,
            log_extra={
                "store": self.store_model.name,
                "memory_store": memory_store.full_name,
                "scope": scope,
            },
        )
        self._setup_complete = True
        logger.debug(
            "Agent Memory store registered",
            store=self.store_model.name,
            memory_store=memory_store.full_name,
            scope=scope,
        )
