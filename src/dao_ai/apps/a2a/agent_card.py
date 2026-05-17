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
    AgentSkill,
    SecurityScheme,
)
from loguru import logger

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
                description=s.description or s.name,
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
        derived.append(
            AgentSkill(
                id=str(getattr(sub, "name", "")),
                name=str(getattr(sub, "name", "")),
                description=getattr(sub, "description", None)
                or f"dao-ai sub-agent: {getattr(sub, 'name', 'agent')}",
                tags=["dao-ai", "sub-agent"],
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


def _derive_security_schemes(
    config: "AppConfig", a2a: "A2AModel"
) -> dict[str, SecurityScheme] | None:
    """Derive Agent Card security_schemes.

    Explicit ``a2a.security_schemes`` wins (already validated against
    a2a-sdk's SecurityScheme discriminated union at config-load time).
    Otherwise, emit a single ``bearer`` HTTP scheme whose ``bearer_format``
    documents whether the deployment supports OBO (when
    ``a2a.on_behalf_of_user`` is True).
    """
    if a2a.security_schemes is not None:
        return a2a.security_schemes or None

    bearer_format = (
        "Databricks OAuth (forwarded by Apps proxy via x-forwarded-access-token; OBO supported)"
        if a2a.on_behalf_of_user
        else "Databricks PAT or OAuth M2M token"
    )
    scheme_dict: dict[str, object] = {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": bearer_format,
        "description": (
            "Databricks Apps forwards the caller's bearer token via "
            "``x-forwarded-access-token`` to the dao-ai agent."
        ),
    }
    return {"bearer": SecurityScheme.model_validate(scheme_dict)}


def build_agent_card(config: "AppConfig") -> AgentCard:
    """Build the public A2A Agent Card for this config.

    Pure function — no side effects, no I/O. Suitable for serving from a
    request handler. Reads ``$DATABRICKS_APP_URL`` once at call time.
    """
    a2a = effective_a2a(config)

    name = config.app.name if config.app else "dao-ai-agent"
    description = (
        config.app.description if config.app else None
    ) or f"dao-ai agent: {name}"

    skills = _derive_skills(config, a2a)
    security_schemes = _derive_security_schemes(config, a2a)
    security: list[dict[str, list[str]]] | None = None
    if security_schemes:
        # Single requirement OR over the declared schemes, no extra scopes.
        security = [{key: [] for key in security_schemes}]

    capabilities = AgentCapabilities(
        streaming=True,
        push_notifications=False,
        state_transition_history=True,
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
    )

    logger.debug(
        "Built A2A Agent Card",
        name=name,
        skills_count=len(skills),
        security_schemes=list(security_schemes.keys()) if security_schemes else [],
        url=card.url,
    )
    return card
