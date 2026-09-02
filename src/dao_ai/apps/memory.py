"""Read a user's long-term memory for the dao-ai Console memory viewer.

Reads the configured LangGraph ``BaseStore`` (dao-ai's ``AgentMemoryStore`` over
the Databricks Managed Agent Memory API) for the current user's stored profile /
preferences / episodic memories. Read-only, scoped to a single user's namespace
(``("memory", user_id)``) so a viewer only sees their own memory.
"""

from __future__ import annotations

from typing import Any, Optional

from loguru import logger


def _iso(value: Any) -> Optional[str]:
    return value.isoformat() if hasattr(value, "isoformat") else None


async def load_user_memory(store: Any, user_id: str) -> dict[str, Any]:
    """Return the user's memory grouped by namespace.

    Uses the store's prefix search under ``("memory", user_id)`` and groups the
    returned items by their full namespace (e.g. ``memory/<uid>/user_profile``,
    ``.../preferences``, ``.../episodes``). Shapes follow ``memory/schemas.py``.
    """
    prefix = ("memory", user_id)
    items = await store.asearch(prefix, query=None, limit=500)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        ns = "/".join(getattr(item, "namespace", ()) or prefix)
        grouped.setdefault(ns, []).append(
            {
                "key": getattr(item, "key", None),
                "value": getattr(item, "value", None),
                "created_at": _iso(getattr(item, "created_at", None)),
                "updated_at": _iso(getattr(item, "updated_at", None)),
            }
        )

    return {
        "user_id": user_id,
        "namespaces": sorted(grouped.keys()),
        "memory": grouped,
    }


async def safe_load_user_memory(store: Any, user_id: str) -> dict[str, Any]:
    """`load_user_memory` guarded so a store hiccup returns empty, not a 500."""
    try:
        return await load_user_memory(store, user_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load user memory", user_id=user_id, error=str(exc))
        return {"user_id": user_id, "namespaces": [], "memory": {}}
