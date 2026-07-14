"""
Cache-keyed factory for audit sinks.

Multiple audited tools may reference the same ``AuditModel`` YAML anchor,
which resolves to identical ``(database, table)`` pairs. We return a
single ``LakebaseAuditSink`` instance per unique pair so all audited tools
share one connection pool and one hash-chain state.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from dao_ai.audit.lakebase import LakebaseAuditSink

if TYPE_CHECKING:
    from dao_ai.config import AuditModel, DatabaseModel


SinkKey = tuple[str, str]


class AuditSinkManager:
    """Process-wide singleton cache keyed on ``(database identity, table)``."""

    _sinks: dict[SinkKey, LakebaseAuditSink] = {}
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def for_config(cls, config: "AuditModel") -> LakebaseAuditSink:
        key: SinkKey = cls._key_for(config)
        with cls._lock:
            existing: LakebaseAuditSink | None = cls._sinks.get(key)
            if existing is not None:
                return existing
            sink: LakebaseAuditSink = LakebaseAuditSink(config)
            cls._sinks[key] = sink
            return sink

    @classmethod
    def _key_for(cls, config: "AuditModel") -> SinkKey:
        db: "DatabaseModel" = config.database
        identity: str = db.project or db.name or "unknown-database"
        return (identity, config.table)

    @classmethod
    def reset(cls) -> None:
        """Clear cached sinks. Test-only helper."""
        with cls._lock:
            cls._sinks.clear()
