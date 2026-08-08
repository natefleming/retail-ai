"""dao-ai MCP server — exposes a dao-ai agent as a single MCP tool.

The MCP server registers exactly one tool per deployment: the whole
dao-ai agent graph built by :meth:`AppConfig.as_responses_agent`. The
tool's name is derived from ``config.app.name`` (slugified); its
description is ``config.app.description``. See
:func:`dao_ai.mcp.agent_tool.register_agent_as_tool`.

Public entrypoints:

* :func:`dao_ai.mcp.server.main` — uvicorn-launched entrypoint. Generated
  Apps bundles invoke it as ``python -m dao_ai.mcp.server`` (see
  ``dao_ai.mcp.generate._MCP_APP_COMMAND``).
* :func:`dao_ai.mcp.server.build_app` — in-process ASGI app construction for
  tests.

Deployment artifacts (``databricks.yml``, ``app.yaml``, etc.) are produced by
``dao-ai agent build --as-mcp`` — see :func:`dao_ai.mcp.generate.write_mcp_bundle`.

Optional install: ``pip install 'dao-ai[mcp]'`` pulls fastapi + uvicorn.
"""

from __future__ import annotations

try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("dao-ai")
except Exception:
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
