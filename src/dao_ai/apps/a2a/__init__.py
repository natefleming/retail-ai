"""Google A2A (Agent2Agent) protocol support for dao-ai Databricks Apps.

When a dao-ai agent is deployed to Databricks Apps, this package mounts a
fully spec-compliant A2A v0.3 endpoint alongside the existing OpenAI
Responses contract:

* ``GET  /.well-known/agent-card.json``  — Agent Card discovery.
* ``POST /a2a``                          — JSON-RPC 2.0 (``message/send``,
  ``message/stream``, ``tasks/get``, ``tasks/cancel``, ``tasks/list``).

Both contracts share the same :class:`langgraph.graph.state.CompiledStateGraph`
instance (via :meth:`dao_ai.config.AppConfig.as_graph`) and the same
LangGraph checkpointer, so conversations are consistent regardless of which
protocol the caller uses.

Public entry point:

* :func:`mount_a2a_routes` — call from FastAPI app startup; idempotent and
  honors the ``app.a2a.enabled`` opt-out.

See :class:`dao_ai.config.A2AModel` for the full configuration surface.
"""

from dao_ai.apps.a2a.agent_card import build_agent_card, effective_a2a
from dao_ai.apps.a2a.executor import A2AAgentExecutor
from dao_ai.apps.a2a.routes import mount_a2a_routes
from dao_ai.apps.a2a.security import (
    BEARER_DATABRICKS_M2M,
    BEARER_DATABRICKS_OBO,
    BEARER_DATABRICKS_PAT,
    api_key_header,
    oauth2_databricks_authorization_code,
    oauth2_databricks_client_credentials,
    oauth2_databricks_obo,
    openid_connect_databricks,
)
from dao_ai.apps.a2a.task_store import LakebaseTaskStore, build_task_store

__all__ = [
    "A2AAgentExecutor",
    "BEARER_DATABRICKS_M2M",
    "BEARER_DATABRICKS_OBO",
    "BEARER_DATABRICKS_PAT",
    "LakebaseTaskStore",
    "api_key_header",
    "build_agent_card",
    "build_task_store",
    "effective_a2a",
    "mount_a2a_routes",
    "oauth2_databricks_authorization_code",
    "oauth2_databricks_client_credentials",
    "oauth2_databricks_obo",
    "openid_connect_databricks",
]
