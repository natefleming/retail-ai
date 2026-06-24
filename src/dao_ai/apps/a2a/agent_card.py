"""Build an A2A Agent Card from a dao-ai :class:`AppConfig`.

The Agent Card is the discovery document served at
``/.well-known/agent-card.json``. Clients fetch it to learn the agent's
name, capabilities, supported skills, and security requirements before
calling the JSON-RPC endpoint.

Defaults (used when ``config.app.a2a`` is unset or fields are unset):

* ``skills``         — one per entry in ``config.agents`` (sub-agent name +
  description). Empty list when no sub-agents are configured.
* ``security_schemes`` — single ``bearer`` HTTP scheme; the human-readable
  description is conditioned on ``config.app.on_behalf_of_user`` to indicate
  whether OBO is supported.
* ``url`` — ``$DATABRICKS_APP_URL/a2a`` if the env var is set at startup,
  otherwise the relative path ``"/a2a"`` (the canonical caller-relative
  reference).
"""

from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from typing import TYPE_CHECKING

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentProvider,
    AgentSkill,
    HTTPAuthSecurityScheme,
    OAuth2SecurityScheme,
    SecurityScheme,
)
from loguru import logger
from pydantic import BaseModel

if TYPE_CHECKING:
    from dao_ai.config import A2AModel, AppConfig


# Default routes — exposed here so routes.py and the Agent Card stay in sync.
DEFAULT_A2A_RPC_PATH = "/a2a"
DEFAULT_AGENT_CARD_PATH = "/.well-known/agent-card.json"


def effective_a2a(config: "AppConfig") -> "A2AModel":
    """Return the :class:`A2AModel` to use for this config.

    ``AppModel.a2a`` is concrete-by-default (``default_factory=A2AModel``),
    so it's always populated when ``config.app`` exists. This helper only
    handles the edge case of ``config.app`` itself being unset — every other
    A2A code path can read ``config.app.a2a`` directly.
    """
    from dao_ai.config import A2AModel

    if config.app is None:
        return A2AModel()
    return config.app.a2a


def _dao_ai_version() -> str:
    try:
        return pkg_version("dao-ai")
    except PackageNotFoundError:
        return "0.0.0"


def _resolve_server_url(a2a: "A2AModel", rpc_path: str) -> str:
    """Pick the URL advertised on the Agent Card.

    Order of preference:
      1. ``a2a.server_url`` (explicit override).
      2. ``$DATABRICKS_APP_URL`` + ``rpc_path`` (Databricks Apps runtime).
      3. ``rpc_path`` (relative — clients use the host they fetched the
         card from).
    """
    if a2a.server_url:
        return a2a.server_url
    base = os.environ.get("DATABRICKS_APP_URL")
    if base:
        return base.rstrip("/") + rpc_path
    return rpc_path


def _derive_skills(config: "AppConfig", a2a: "A2AModel") -> list[AgentSkill]:
    """Derive Agent Card skills from config.

    Priority:
      1. Explicit ``a2a.skills`` override.
      2. One skill per ``AgentModel`` registered under ``config.agents`` (if
         present at AppConfig level).
      3. Single fallback skill named after ``config.app.name``.
    """
    if a2a.skills:
        logger.trace(
            "Agent Card skills from explicit override",
            count=len(a2a.skills),
            skill_ids=[s.id for s in a2a.skills],
        )
        return [
            AgentSkill(
                id=s.id,
                name=s.name,
                description=(s.description or s.name).rstrip(),
                tags=list(s.tags),
                examples=list(s.examples) if s.examples else None,
                input_modes=list(s.input_modes) if s.input_modes else None,
                output_modes=list(s.output_modes) if s.output_modes else None,
            )
            for s in a2a.skills
        ]

    derived: list[AgentSkill] = []
    # ``agents`` lives in two places: AppConfig.agents (dict[name, AgentModel])
    # is the canonical lookup-by-name view, and AppModel.agents (list) is the
    # original input form. AppConfig.agents is normally populated from
    # AppModel.agents via a validator, but we read both to stay robust.
    sub_agent_iter: list = []
    config_agents = getattr(config, "agents", None)
    if isinstance(config_agents, dict) and config_agents:
        sub_agent_iter = list(config_agents.values())
    elif isinstance(config_agents, list) and config_agents:
        sub_agent_iter = list(config_agents)
    elif config.app is not None:
        app_agents = getattr(config.app, "agents", None) or []
        if isinstance(app_agents, list):
            sub_agent_iter = list(app_agents)
    for sub in sub_agent_iter:
        sub_name = str(getattr(sub, "name", ""))
        raw_desc = (
            getattr(sub, "description", None)
            or f"dao-ai sub-agent: {sub_name or 'agent'}"
        )
        # YAML ``>`` folding appends a trailing newline; strip so the card
        # doesn't carry visible whitespace into consumers.
        description = raw_desc.rstrip()
        # Pick up tags / examples from the AgentModel only if they exist as
        # real list-valued attributes — keeps backward compatibility while
        # future-proofing for an AgentModel that adopts these fields.
        extra_tags = getattr(sub, "tags", None)
        extra_examples = getattr(sub, "examples", None)
        tags = ["dao-ai", "sub-agent"]
        if isinstance(extra_tags, list) and all(isinstance(t, str) for t in extra_tags):
            tags = list(dict.fromkeys([*tags, *extra_tags]))
        examples = None
        if isinstance(extra_examples, list) and all(
            isinstance(e, str) for e in extra_examples
        ):
            examples = list(extra_examples) or None
        derived.append(
            AgentSkill(
                id=sub_name,
                name=sub_name,
                description=description,
                tags=tags,
                examples=examples,
            )
        )

    if derived:
        logger.trace(
            "Agent Card skills derived from sub-agents",
            count=len(derived),
            skill_ids=[s.id for s in derived],
        )
        return derived

    # Last-resort fallback.
    name = config.app.name if config.app else "agent"
    logger.warning(
        "Agent Card skills fell back to single app-level entry",
        app_name=name,
        reason="no a2a.skills and no config.agents",
    )
    return [
        AgentSkill(
            id=name,
            name=name,
            description=config.app.description if config.app else "dao-ai agent",
            tags=["dao-ai"],
        )
    ]


def _any_resource_obo(config: "AppConfig") -> bool:
    """True iff any ``IsDatabricksResource`` in the config has ``on_behalf_of_user=True``.

    Walks the AppConfig Pydantic tree (BaseModel fields, lists, dicts, tuples,
    sets) and short-circuits on the first OBO-enabled resource. Cycle-safe
    via an identity-tracking ``seen`` set.

    Anchored on the :class:`dao_ai.config.IsDatabricksResource` base class so
    every current and future OBO-capable subclass (InferenceEndpointModel,
    GenieRoomModel, VectorStoreModel, WarehouseModel, FunctionModel, etc.)
    is covered without enumeration.
    """
    from dao_ai.config import IsDatabricksResource

    seen: set[int] = set()

    def visit(obj: object) -> bool:
        if obj is None:
            return False
        oid = id(obj)
        if oid in seen:
            return False
        seen.add(oid)
        if isinstance(obj, IsDatabricksResource) and obj.on_behalf_of_user is True:
            return True
        if isinstance(obj, BaseModel):
            return any(visit(getattr(obj, f, None)) for f in obj.model_fields)
        if isinstance(obj, (list, tuple, set)):
            return any(visit(v) for v in obj)
        if isinstance(obj, dict):
            return any(visit(v) for v in obj.values())
        return False

    return visit(config)


def _derive_security_schemes(
    config: "AppConfig", a2a: "A2AModel"
) -> dict[str, SecurityScheme] | None:
    """Derive Agent Card security_schemes.

    Resolution:

    1. Explicit ``a2a.security_schemes`` wins (already validated against
       a2a-sdk's SecurityScheme discriminated union at config-load time).
    2. Otherwise resolve effective OBO:
       * ``a2a.on_behalf_of_user`` is True/False — use that.
       * ``a2a.on_behalf_of_user`` is None (default) — scan resources for
         any ``on_behalf_of_user=True`` (see :func:`_any_resource_obo`).
    3. Effective OBO True → emit BOTH an ``oauth2`` declaration (the actual
       authorization-code flow the Apps proxy carries out) and a
       ``bearer`` scheme (the wire-level shape clients send). If the
       workspace host can't be resolved at boot, fall back to the
       OBO-aware bearer constant alone and log a warning.
    4. Effective OBO False → single bearer scheme advertising PAT / M2M.
    """
    if a2a.security_schemes is not None:
        return a2a.security_schemes or None

    obo = a2a.on_behalf_of_user
    if obo is None:
        obo = _any_resource_obo(config)

    if not obo:
        return {
            "bearer": HTTPAuthSecurityScheme(
                scheme="bearer",
                bearer_format="Databricks PAT or OAuth M2M token",
                description=(
                    "Databricks Apps forwards the caller's bearer token via "
                    "``x-forwarded-access-token`` to the dao-ai agent."
                ),
            )
        }

    # OBO in play — full OAuth2 declaration + bearer wire-shape, paired.
    from dao_ai.apps.a2a.security import (
        BEARER_DATABRICKS_OBO,
        oauth2_databricks_obo,
    )

    try:
        return {
            "oauth2": oauth2_databricks_obo(),
            "bearer": BEARER_DATABRICKS_OBO,
        }
    except ValueError as exc:
        logger.warning(
            "A2A Agent Card: workspace host unresolvable; "
            "falling back to OBO bearer-only scheme. {err}",
            err=exc,
        )
        return {"bearer": BEARER_DATABRICKS_OBO}


def _scheme_inner(scheme: object) -> object:
    """Unwrap an ``a2a.types.SecurityScheme`` RootModel to its concrete
    discriminated-union member. Pass-through for already-concrete schemes."""
    return getattr(scheme, "root", scheme)


def _scheme_scopes(scheme: object) -> list[str]:
    """Return the scope names declared on a scheme, in spec-correct order.

    For OAuth2 schemes that publish a ``flows.authorization_code.scopes``
    map, returns its keys. For all other scheme kinds, returns an empty
    list (matches OpenAPI 3 'no scopes required').
    """
    inner = _scheme_inner(scheme)
    if isinstance(inner, OAuth2SecurityScheme):
        flows = getattr(inner, "flows", None)
        ac = getattr(flows, "authorization_code", None) if flows else None
        if ac is not None and getattr(ac, "scopes", None):
            return list(ac.scopes.keys())
    return []


def _derive_state_transition_history(config: "AppConfig", a2a: "A2AModel") -> bool:
    """Auto-derive whether the agent retains A2A task state transitions.

    True iff the A2A task store is backed by a database — i.e., tasks
    persist across restarts and clients can fetch their state history.
    """
    task_store = getattr(a2a, "task_store", None)
    if task_store is None:
        return False
    return getattr(task_store, "database", None) is not None


def build_agent_card(config: "AppConfig") -> AgentCard:
    """Build the public A2A Agent Card for this config.

    Pure function — no side effects, no I/O. Suitable for serving from a
    request handler. Reads ``$DATABRICKS_APP_URL`` once at call time.
    """
    a2a = effective_a2a(config)

    name = config.app.name if config.app else "dao-ai-agent"
    raw_description = (
        config.app.description if config.app else None
    ) or f"dao-ai agent: {name}"
    # YAML ``>`` block-folding always appends a trailing newline; strip it
    # so consumers don't render a visible blank line.
    description = raw_description.rstrip()

    skills = _derive_skills(config, a2a)
    security_schemes = _derive_security_schemes(config, a2a)
    security: list[dict[str, list[str]]] | None = None
    if security_schemes:
        # One requirement object per scheme so the array is interpreted as
        # OR-of-AND (per OpenAPI 3 / A2A spec): callers can satisfy any
        # single entry. For oauth2 schemes, advertise the required scope
        # names declared on the scheme's authorization_code flow.
        security = [
            {key: _scheme_scopes(scheme)} for key, scheme in security_schemes.items()
        ]

    sth = a2a.state_transition_history
    if sth is None:
        sth = _derive_state_transition_history(config, a2a)
    capabilities = AgentCapabilities(
        streaming=a2a.streaming,
        push_notifications=a2a.push_notifications,
        state_transition_history=sth,
    )

    provider = None
    if a2a.provider is not None:
        provider = AgentProvider(
            organization=a2a.provider.organization,
            url=a2a.provider.url,
        )

    card = AgentCard(
        name=name,
        description=description,
        url=_resolve_server_url(a2a, DEFAULT_A2A_RPC_PATH),
        version=_dao_ai_version(),
        capabilities=capabilities,
        default_input_modes=list(a2a.default_input_modes),
        default_output_modes=list(a2a.default_output_modes),
        skills=skills,
        security_schemes=security_schemes,
        security=security,
        provider=provider,
        documentation_url=a2a.documentation_url,
        icon_url=a2a.icon_url,
    )

    logger.debug(
        "Built A2A Agent Card",
        name=name,
        skills_count=len(skills),
        security_schemes=list(security_schemes.keys()) if security_schemes else [],
        url=card.url,
    )
    return card
