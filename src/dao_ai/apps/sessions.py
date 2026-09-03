"""Session reload for the dao-ai Console.

``GET /v1/sessions/{thread_id}`` reconstructs a past conversation from the
LangGraph checkpointer's persisted state so the Console can reopen a thread.
``session_items_from_messages`` is the pure converter (LangChain messages →
UI conversation items) — reasoning is split from the answer text so a reloaded
transcript matches what the live turn streamed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from dao_ai.models import _split_content, resolve_user_id_from_headers

# The user→thread session index is kept in the configured LangGraph ``BaseStore``
# (``graph.store``) under this namespace root, addressed through the store's own
# ``aput``/``asearch`` API rather than native SQL — so it works unchanged across
# any store backend (Postgres, Lakebase, Managed Agent Memory, in-memory) and
# stays isolated from the memory namespace (``("memory", user_id)``).
_SESSION_NS_ROOT = "sessions"


def session_items_from_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Convert checkpointer ``state.values["messages"]`` into conversation items.

    Item shapes:

    - user: ``{"role": "user", "content": str}``
    - assistant: ``{"role": "assistant", "content": str[, "reasoning": str]}``
    - tool: ``{"role": "tool", "name": str, "content": str}``

    Messages with no renderable content are skipped.
    """
    items: list[dict[str, Any]] = []
    for message in messages or []:
        if isinstance(message, HumanMessage):
            text, _ = _split_content(message.content)
            if text:
                items.append({"role": "user", "content": text})
        elif isinstance(message, ToolMessage):
            text, _ = _split_content(message.content)
            items.append(
                {
                    "role": "tool",
                    "name": getattr(message, "name", None) or "tool",
                    "content": text,
                    "tool_call_id": getattr(message, "tool_call_id", None),
                }
            )
        elif isinstance(message, AIMessage):
            text, reasoning = _split_content(message.content)
            tool_calls: list[dict[str, Any]] = [
                {
                    "call_id": tc.get("id"),
                    "name": tc.get("name"),
                    "arguments": tc.get("args"),
                }
                for tc in (getattr(message, "tool_calls", None) or [])
                if isinstance(tc, dict)
            ]
            # Keep the message if it carries an answer, reasoning, OR tool calls,
            # so the reconstructed turn can rebuild the tool/handoff flow.
            if not text and not reasoning and not tool_calls:
                continue
            item: dict[str, Any] = {"role": "assistant", "content": text}
            if reasoning:
                item["reasoning"] = reasoning
            if tool_calls:
                item["tool_calls"] = tool_calls
            items.append(item)
    return items


def user_id_from_headers(headers: Any) -> Optional[str]:
    """Resolve the calling user from the OBO identity headers.

    Uses the same resolver + normalization as ``models.py`` (login name
    preferred over the numeric ``x-forwarded-user`` id; ``.`` → ``_``) so the
    Console's sessions/memory scope matches the ``user_id`` the agent runs
    under. Derived server-side (never a client value). Returns None when no
    identity header is present (local/dev).
    """
    # Support both a mapping and a Starlette Headers object.
    raw_headers = dict(headers.items()) if hasattr(headers, "items") else headers
    resolved = resolve_user_id_from_headers(raw_headers)
    return resolved.replace(".", "_") if resolved else None


async def register_session(
    store: Any, user_id: str, thread_id: str, title: Optional[str]
) -> None:
    """Record/refresh a user's thread in the configured ``BaseStore``.

    Uses ``store.aput`` (the implementation object) so any store backend records
    the index identically — no backend-specific query. Keyed by ``thread_id``
    under the ``("sessions", user_id)`` namespace, so listing is user-scoped.
    """
    await store.aput(
        (_SESSION_NS_ROOT, user_id),
        thread_id,
        {"title": title, "updated_at": datetime.now(timezone.utc).isoformat()},
    )


async def list_user_sessions(
    store: Any, user_id: str, *, limit: int = 50
) -> list[dict[str, Any]]:
    """List a user's threads (most-recently-updated first) from the ``BaseStore``.

    Reads through ``store.asearch`` over the user's ``("sessions", user_id)``
    namespace — the same implementation-object path the memory viewer uses — so
    the sidebar works for every store backend without native queries.
    """
    items: list[Any] = await store.asearch((_SESSION_NS_ROOT, user_id), limit=limit)
    rows: list[dict[str, Any]] = []
    for item in items or []:
        value: dict[str, Any] = getattr(item, "value", None) or {}
        updated_at: Optional[str] = value.get("updated_at")
        if updated_at is None:
            stamp = getattr(item, "updated_at", None)
            updated_at = stamp.isoformat() if hasattr(stamp, "isoformat") else None
        rows.append(
            {
                "thread_id": getattr(item, "key", None),
                "title": value.get("title"),
                "updated_at": updated_at,
            }
        )
    # asearch has no cross-backend ordering guarantee, so order here.
    rows.sort(key=lambda r: r["updated_at"] or "", reverse=True)
    return rows[:limit]


async def user_owns_thread(store: Any, user_id: Optional[str], thread_id: str) -> bool:
    """True when ``thread_id`` is registered to ``user_id`` in the session index.

    The ``("sessions", user_id)`` namespace is the authoritative user→thread
    mapping (written by :func:`register_session`), so the checkpointer-backed
    reload/meta/trace routes use this to scope access to the owning user — the
    checkpointer itself is keyed only by ``thread_id`` and would otherwise return
    any user's conversation for a known/guessed id. Fails closed (``False``) on a
    missing store/id or any store error.
    """
    if not (store is not None and user_id and thread_id):
        return False
    try:
        item = await store.aget((_SESSION_NS_ROOT, user_id), thread_id)
        return item is not None
    except Exception:  # noqa: BLE001 — fail closed on any store error
        return False


async def load_session_meta(graph: Any, thread_id: str) -> dict[str, Any]:
    """Return checkpoint metadata for a thread from the checkpoint API.

    Reads the latest ``StateSnapshot`` and surfaces the identifiers and
    last-modified time the Console's session-info popover shows.
    """
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    snapshot: Any = await graph.aget_state(config)
    values: dict[str, Any] = getattr(snapshot, "values", None) or {}
    messages: list[Any] = values.get("messages", []) or []
    snap_config: dict[str, Any] = getattr(snapshot, "config", None) or {}
    checkpoint_id: Optional[str] = (snap_config.get("configurable") or {}).get(
        "checkpoint_id"
    )
    metadata: dict[str, Any] = getattr(snapshot, "metadata", None) or {}
    return {
        "thread_id": thread_id,
        "checkpoint_id": checkpoint_id,
        "last_modified": getattr(snapshot, "created_at", None),
        "step": metadata.get("step"),
        "message_count": len(messages),
    }


async def load_session(graph: Any, thread_id: str) -> dict[str, Any]:
    """Reconstruct a conversation from the checkpointer for ``thread_id``.

    Reads the persisted LangGraph state and maps its message history into
    Console conversation items. Returns an empty ``messages`` list when the
    thread has no persisted state yet.
    """
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    snapshot: Any = await graph.aget_state(config)
    values: dict[str, Any] = getattr(snapshot, "values", None) or {}
    messages: list[Any] = values.get("messages", []) or []
    return {
        "thread_id": thread_id,
        "messages": session_items_from_messages(messages),
    }
