"""Session reload for the dao-ai Console.

``GET /v1/sessions/{thread_id}`` reconstructs a past conversation from the
LangGraph checkpointer's persisted state so the Console can reopen a thread.
``session_items_from_messages`` is the pure converter (LangChain messages →
UI conversation items) — reasoning is split from the answer text so a reloaded
transcript matches what the live turn streamed.
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from dao_ai.models import _split_content


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
            if text:
                items.append(
                    {
                        "role": "tool",
                        "name": getattr(message, "name", None) or "tool",
                        "content": text,
                    }
                )
        elif isinstance(message, AIMessage):
            text, reasoning = _split_content(message.content)
            if not text and not reasoning:
                continue
            item: dict[str, Any] = {"role": "assistant", "content": text}
            if reasoning:
                item["reasoning"] = reasoning
            items.append(item)
    return items


def user_id_from_headers(headers: Any) -> Optional[str]:
    """Resolve the calling user from the OBO ``x-forwarded-user`` header.

    Derived server-side (never trusted from a client value) and normalized the
    same way ``models.py`` does (``.`` → ``_``) so it matches how sessions and
    memory are keyed. Returns None when the header is absent (local/dev).
    """
    raw = headers.get("x-forwarded-user") or headers.get("X-Forwarded-User")
    return raw.replace(".", "_") if raw else None


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
