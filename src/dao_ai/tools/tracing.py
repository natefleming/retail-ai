"""
Resource tracing utilities for DAO AI tools and middleware.

Provides a lightweight ``ResourceInfo`` DTO and a helper to stamp
MLflow span attributes that describe which Databricks resource was
used and whether it ran on behalf of the calling user (OBO).

All attribute keys follow OpenTelemetry naming conventions with the
``dao_ai.`` application prefix to avoid collisions with OTel
semantic conventions (e.g. OTel's own ``resource.*`` concept).

Attributes set on each span:

- ``dao_ai.resource.type``  -- e.g. ``"vector_search"``, ``"model_serving"``
- ``dao_ai.resource.on_behalf_of_user`` -- ``True`` or ``False``
- ``dao_ai.resource.name`` -- optional resource identifier

Usage::

    from dao_ai.tools.tracing import ResourceInfo, set_resource_attributes

    # In a top-level tool (config in closure scope):
    set_resource_attributes(
        ResourceInfo("vector_search", vector_store.on_behalf_of_user, index_name)
    )

    # In an inner function (passed as optional param):
    def route_query(..., resource_info: ResourceInfo | None = None):
        ...
        if resource_info:
            set_resource_attributes(resource_info)
"""

from __future__ import annotations

from dataclasses import dataclass

import mlflow

# ---------------------------------------------------------------------------
# Attribute key constants -- single source of truth for span attribute names.
# Uses ``dao_ai.*`` namespace per OTel guidance for application-specific
# attributes, with dot-delimited hierarchy and snake_case components.
# ---------------------------------------------------------------------------

ATTR_RESOURCE_TYPE = "dao_ai.resource.type"
ATTR_RESOURCE_NAME = "dao_ai.resource.name"
ATTR_RESOURCE_OBO = "dao_ai.resource.on_behalf_of_user"

ATTR_ROUTER_MODE = "dao_ai.router.mode"
ATTR_ROUTER_FALLBACK = "dao_ai.router.fallback"
ATTR_ROUTER_BYPASSED = "dao_ai.router.bypassed_stages"

ATTR_VERIFIER_OUTCOME = "dao_ai.verifier.outcome"
ATTR_VERIFIER_RETRIES = "dao_ai.verifier.retry_count"
ATTR_VERIFIER_DETAIL = "dao_ai.verifier.detail"

ATTR_RERANKER_AVG_SCORE = "dao_ai.reranker.avg_score"

ATTR_DECOMPOSITION_COUNT = "dao_ai.decomposition.subquery_count"
ATTR_DECOMPOSITION_DETAIL = "dao_ai.decomposition.detail"

# Vertex AI Agent Engine tool
ATTR_VERTEX_ENDPOINT_URL = "dao_ai.vertex.endpoint_url"
ATTR_VERTEX_CLASS_METHOD = "dao_ai.vertex.class_method"
ATTR_VERTEX_HTTP_METHOD = "dao_ai.vertex.http_method"
ATTR_VERTEX_AUTH_MODE = "dao_ai.vertex.auth_mode"
ATTR_VERTEX_USER_ID = "dao_ai.vertex.user_id"
ATTR_VERTEX_SESSION_ID = "dao_ai.vertex.session_id"
ATTR_VERTEX_SESSION_PASSED = "dao_ai.vertex.session_id_passed"
ATTR_VERTEX_RETRIED_WITHOUT_SESSION = "dao_ai.vertex.retried_without_session"
ATTR_VERTEX_RETRY_REASON = "dao_ai.vertex.retry_reason"
ATTR_VERTEX_PROMPT_CHARS = "dao_ai.vertex.prompt_chars"
ATTR_VERTEX_RESPONSE_CHARS = "dao_ai.vertex.response_chars"
ATTR_VERTEX_STREAM_TOTAL_LINES = "dao_ai.vertex.stream.total_lines"
ATTR_VERTEX_STREAM_MODEL_EVENTS = "dao_ai.vertex.stream.model_events"
ATTR_VERTEX_STREAM_STATE_DELTA_EVENTS = "dao_ai.vertex.stream.state_delta_events"
ATTR_VERTEX_STREAM_NONJSON_LINES = "dao_ai.vertex.stream.non_json_lines"

# A2A Agent tool (Google Agent-to-Agent protocol)
ATTR_A2A_ENDPOINT_URL = "dao_ai.a2a.endpoint_url"
ATTR_A2A_AGENT_NAME = "dao_ai.a2a.agent_name"
ATTR_A2A_AGENT_VERSION = "dao_ai.a2a.agent_version"
ATTR_A2A_AUTH_MODE = "dao_ai.a2a.auth_mode"
ATTR_A2A_STREAMING = "dao_ai.a2a.streaming"
ATTR_A2A_USER_ID = "dao_ai.a2a.user_id"
ATTR_A2A_CONTEXT_ID = "dao_ai.a2a.context_id"
ATTR_A2A_CONTEXT_PASSED = "dao_ai.a2a.context_id_passed"
ATTR_A2A_TASK_ID = "dao_ai.a2a.task_id"
ATTR_A2A_PROMPT_CHARS = "dao_ai.a2a.prompt_chars"
ATTR_A2A_RESPONSE_CHARS = "dao_ai.a2a.response_chars"
ATTR_A2A_RETRIED_WITHOUT_CONTEXT = "dao_ai.a2a.retried_without_context"
ATTR_A2A_RETRY_REASON = "dao_ai.a2a.retry_reason"
ATTR_A2A_CARD_PATH_USED = "dao_ai.a2a.card.path_used"
ATTR_A2A_STREAM_TOTAL_EVENTS = "dao_ai.a2a.stream.total_events"
ATTR_A2A_STREAM_AGENT_EVENTS = "dao_ai.a2a.stream.agent_events"
ATTR_A2A_STREAM_TASK_EVENTS = "dao_ai.a2a.stream.task_events"
ATTR_A2A_STREAM_NON_TEXT_PARTS = "dao_ai.a2a.stream.non_text_parts"
ATTR_A2A_STREAM_TERMINAL_STATE = "dao_ai.a2a.stream.terminal_state"

# Databricks-App agent tools (Responses API + Chat Completions API)
ATTR_APP_AGENT_APP_NAME = "dao_ai.app_agent.app_name"
ATTR_APP_AGENT_API = "dao_ai.app_agent.api"  # "responses" or "chat_completions"
ATTR_APP_AGENT_MODEL = "dao_ai.app_agent.model"  # f"apps/{app_name}"
ATTR_APP_AGENT_OBO = "dao_ai.app_agent.on_behalf_of_user"
ATTR_APP_AGENT_PROMPT_CHARS = "dao_ai.app_agent.prompt_chars"
ATTR_APP_AGENT_RESPONSE_CHARS = "dao_ai.app_agent.response_chars"

# OpenTelemetry HTTP semantic conventions (used on the _post child span)
ATTR_HTTP_METHOD = "http.method"
ATTR_HTTP_URL = "http.url"
ATTR_HTTP_STATUS = "http.status_code"
ATTR_HTTP_RESP_LEN = "http.response_content_length"


@dataclass(frozen=True)
class ResourceInfo:
    """Trace metadata for a Databricks resource usage."""

    resource_type: str
    on_behalf_of_user: bool
    name: str | None = None


def set_resource_attributes(info: ResourceInfo) -> None:
    """Set resource trace attributes on the current active MLflow span."""
    span = mlflow.get_current_active_span()
    if span:
        span.set_attribute(ATTR_RESOURCE_TYPE, info.resource_type)
        span.set_attribute(ATTR_RESOURCE_OBO, info.on_behalf_of_user)
        if info.name:
            span.set_attribute(ATTR_RESOURCE_NAME, info.name)
