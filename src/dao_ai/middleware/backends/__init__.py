"""Pluggable backends for Deep Agents middleware in DAO AI."""

__all__ = [
    "DatabricksVolumeBackend",
]


def __getattr__(name: str):
    """Lazily import the deepagents-backed volume backend (PEP 562).

    ``DatabricksVolumeBackend`` subclasses a ``deepagents`` protocol, so its
    module cannot be imported when the ``deepagents`` extra is absent. Defer the
    import to attribute access so ``import dao_ai.middleware.backends`` works
    without the extra; the friendly missing-extra error surfaces from
    ``resolve_backend`` / factory usage.
    """
    if name == "DatabricksVolumeBackend":
        from dao_ai._extras import require_extra

        require_extra("deepagents", feature="Databricks Volume backend")
        from dao_ai.middleware.backends.volume import DatabricksVolumeBackend

        return DatabricksVolumeBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
