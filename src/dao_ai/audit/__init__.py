"""
dao-ai audit subsystem.

Public surface:

- ``AuditReceipt`` — canonical receipt schema.
- ``ReceiptKind`` / ``ExecutionStatus`` — enum discriminators.
- ``args_hash_of`` / ``canonical_jcs`` / ``sha256_hex`` — hashing helpers.
- ``AuditSinkManager`` — factory that returns a cached ``LakebaseAuditSink``.
- ``LakebaseAuditSink`` — concrete Lakebase-backed sink implementation.
- ``AuditNonceError`` — raised on any nonce validation failure.

Importing this package is safe on the audit-disabled path: nothing here
loads ``psycopg`` or Lakebase infrastructure at module import time.
Backend I/O happens only when a sink method is awaited.
"""

from dao_ai.audit.base import (
    AuditReceipt,
    ExecutionStatus,
    ReceiptKind,
    args_hash_of,
    canonical_jcs,
    sha256_hex,
)
from dao_ai.audit.lakebase import LakebaseAuditSink
from dao_ai.audit.manager import AuditSinkManager
from dao_ai.audit.nonces import AuditNonceError

__all__ = [
    "AuditNonceError",
    "AuditReceipt",
    "AuditSinkManager",
    "ExecutionStatus",
    "LakebaseAuditSink",
    "ReceiptKind",
    "args_hash_of",
    "canonical_jcs",
    "sha256_hex",
]
