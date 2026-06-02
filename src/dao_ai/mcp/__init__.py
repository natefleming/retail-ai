"""dao-ai MCP server — exposes dao-ai's Genie cache + advanced retrieval as MCP tools.

Tools are discovered dynamically from ``AppConfig.tools`` via a pluggable
adapter registry (``dao_ai.mcp.adapters``). Each adapter knows how to turn one
dao-ai tool factory (e.g. ``dao_ai.tools.create_genie_toolkit``) into one or
more MCP tools, lifting name and description from the underlying dao-ai
factory output rather than hardcoding them on the MCP side.

Public entrypoints:

* :func:`dao_ai.mcp.server.main` — uvicorn-launched script (see
  ``dao-ai-mcp-server`` console entry in ``pyproject.toml``).
* :func:`dao_ai.mcp.server.build_app` — in-process ASGI app construction for
  tests.

Deployment artifacts (``databricks.yml``, ``app.yaml``, etc.) are produced by
``dao-ai generate-mcp`` — see :func:`dao_ai.mcp.generate.write_mcp_bundle`.

Optional install: ``pip install 'dao-ai[mcp]'`` pulls fastapi + uvicorn.
"""

from __future__ import annotations

try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("dao-ai")
except Exception:
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
