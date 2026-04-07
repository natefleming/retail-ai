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
