"""
Deep_agent pattern for single-agent orchestration with deepagents.

The deep_agent pattern wraps `langchain-ai/deepagents` so its full feature set —
todo planning, filesystem ops, shell execution, sub-agent delegation via the
``task`` tool, skills (deepagents' SkillsMiddleware), AGENTS.md memory, and
human-in-the-loop interrupts — is expressible declaratively in YAML and
composable with dao-ai primitives (InferenceEndpointModel, ToolModel, MiddlewareModel,
AgentModel, PromptModel, SkillModel).

Based on: https://github.com/langchain-ai/deepagents
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, cast

from langchain.agents.middleware import AgentMiddleware as LangchainAgentMiddleware
from langchain.agents.middleware.human_in_the_loop import InterruptOnConfig
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore
from loguru import logger

from dao_ai.config import (
    AgentModel,
    AppConfig,
    BackendModel,
    DeepAgentModel,
    FilesystemPermissionModel,
    HumanInTheLoopModel,
    InferenceEndpointModel,
    MiddlewareModel,
    OrchestrationModel,
    PromptModel,
    ResponseFormatModel,
    SubAgentModel,
    ToolModel,
)
from dao_ai.middleware.core import create_factory_middleware
from dao_ai.middleware.human_in_the_loop import _config_to_interrupt_on_entry
from dao_ai.orchestration import (
    create_checkpointer,
    create_store,
)
from dao_ai.skills import (
    resolve_instruction_file_runtime_paths,
    resolve_skill_runtime_paths,
)
from dao_ai.tools import create_tools

if TYPE_CHECKING:
    from deepagents import SubAgent as DeepAgentsSubAgent
    from deepagents.middleware.permissions import FilesystemPermission


def _resolve_model(
    spec: InferenceEndpointModel | str | None,
) -> BaseChatModel | str | None:
    """Resolve a deep_agent ``model`` field to what ``create_deep_agent`` expects.

    deepagents accepts ``str | BaseChatModel | None``. We give it:
    * a fully initialized chat model when an ``InferenceEndpointModel`` is supplied (so
      dao-ai's temperature/max_tokens/fallbacks/best_of_n configuration is honored)
    * the raw string when a string is supplied (deepagents passes it to
      ``init_chat_model`` itself)
    * ``None`` to fall through to deepagents' default.
    """
    if spec is None:
        return None
    if isinstance(spec, InferenceEndpointModel):
        return spec.as_chat_model()
    return spec


def _resolve_tool_spec(spec: ToolModel | str, config: AppConfig) -> ToolModel:
    """Resolve a tool spec entry to a concrete ToolModel.

    Strings are looked up in ``config.tools``. Inline ``ToolModel`` entries
    pass through unchanged.
    """
    if isinstance(spec, str):
        if spec not in config.tools:
            raise ValueError(
                f"Tool reference '{spec}' not found in config.tools. "
                f"Available: {sorted(config.tools.keys())}"
            )
        return config.tools[spec]
    return spec


def _resolve_tools(specs: list[ToolModel | str], config: AppConfig) -> list[BaseTool]:
    """Resolve a mixed list of ToolModel/string entries into BaseTool instances."""
    if not specs:
        return []
    tool_models: list[ToolModel] = [_resolve_tool_spec(s, config) for s in specs]
    return list(create_tools(tool_models))


def _resolve_system_prompt(spec: str | PromptModel | None) -> str | None:
    """Resolve a system_prompt spec to the string deepagents expects.

    For inline strings: pass through.
    For ``PromptModel``: return ``template`` as a plain string. Users who need
    template-variable substitution should keep using inline strings and add a
    ``dynamic_prompt`` middleware to ``middleware``.
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        return spec

    return spec.template


def _resolve_middleware(
    specs: list[MiddlewareModel],
) -> list[LangchainAgentMiddleware]:
    """Resolve MiddlewareModel entries to instantiated middleware.

    Reuses ``create_factory_middleware`` for parity with swarm/supervisor.
    """
    middlewares: list[LangchainAgentMiddleware] = []
    for spec in specs or []:
        logger.trace("Creating deep_agent middleware", middleware_name=spec.name)
        middleware = create_factory_middleware(
            function_name=spec.name,
            args=spec.args,
        )
        middlewares.append(middleware)
    return middlewares


def _resolve_permissions(
    specs: list[FilesystemPermissionModel],
) -> list[FilesystemPermission]:
    """Translate FilesystemPermissionModel into deepagents' FilesystemPermission dataclass.

    deepagents' FilesystemPermission is a ``@dataclass`` (NOT a TypedDict) with
    fields ``operations``, ``paths``, ``mode`` and a ``__post_init__`` validator
    that requires all paths to start with ``/``. Defaults ``operations`` to both
    read and write when the user omits it.
    """
    if not specs:
        return []

    from deepagents.middleware.permissions import FilesystemPermission

    resolved: list[FilesystemPermission] = []
    for p in specs:
        resolved.append(
            FilesystemPermission(
                operations=list(p.operations) if p.operations else ["read", "write"],
                paths=list(p.paths),
                mode=p.mode,
            )
        )
    return resolved


def _resolve_interrupt_on(
    specs: dict[str, bool | HumanInTheLoopModel],
) -> dict[str, bool | InterruptOnConfig]:
    """Translate per-tool interrupt config to deepagents' shape.

    Delegates to ``_config_to_interrupt_on_entry`` (the same converter used
    by tool-level ``human_in_the_loop:`` annotations) so there is exactly
    one place that knows how to translate a ``HumanInTheLoopModel`` into
    LangChain's ``InterruptOnConfig`` TypedDict.
    """
    if not specs:
        return {}
    return {
        tool_name: _config_to_interrupt_on_entry(value)
        for tool_name, value in specs.items()
    }


def _resolve_response_format(
    spec: ResponseFormatModel | type | str | None,
) -> Any:
    """Pass response_format through to deepagents.

    AgentModel's existing validator already converts strings/types into
    ResponseFormatModel; for deep_agent, downstream LangChain will accept
    ResponseFormatModel.response_schema directly.
    """
    if spec is None:
        return None
    if isinstance(spec, ResponseFormatModel):
        return spec.response_schema
    return spec


def _resolve_context_schema(spec: str | None) -> type | None:
    """Resolve a fully-qualified class name to its type, if provided."""
    if not spec:
        return None
    module_name, _, attr = spec.rpartition(".")
    if not module_name:
        raise ValueError(
            f"context_schema '{spec}' must be a fully qualified name like 'mymod.MyContext'"
        )
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def _resolve_backend(
    spec: BackendModel | None, *, needs_filesystem: bool = False
) -> Any:
    """Instantiate a backend factory from its FQN.

    When ``spec`` is None:

    * If skills **or instruction files** are declared **anywhere** — on the
      deep_agent or on any one of its sub-agents — default to
      ``FilesystemBackend(virtual_mode=False)``. deepagents' default
      ``StateBackend`` cannot read files from disk or UC volumes, so a
      filesystem-aware backend is required for skill discovery and for
      ``MemoryMiddleware`` to load an ``AGENTS.md``.
    * Otherwise, return None so deepagents picks its default StateBackend.

    Instruction files count for the same reason skills do, and forgetting them
    made a config with ``instruction_files`` and no skills unfixable from the
    path side: ``StateBackend`` reads from graph state, so no path — however
    correct — could ever load.

    ``virtual_mode=False`` is passed **explicitly and is load-bearing**, not a
    restatement of a default. deepagents flipped this default to ``True`` in
    0.7.0 (its deprecation notice announced 0.6.0, but 0.6.x still defaults to
    ``False``), and under ``virtual_mode=True`` every incoming path is treated as a
    *virtual* path anchored at ``root_dir`` (here the CWD): the leading slash is
    stripped and the remainder joined underneath, so the real absolute path
    ``/app/src/instructions/AGENTS.md`` is looked up at
    ``<cwd>/app/src/instructions/AGENTS.md`` and never found. dao-ai's runtime
    resolvers deal in real host paths — a config directory, an mlflow artifact's
    ``code`` dir, a ``/Volumes`` mount — none of which are reachable that way,
    and ``/Volumes`` is rejected outright as "outside root directory". Since
    ``pyproject.toml`` allows ``deepagents>=0.5.7``, leaving the default implicit
    means the behaviour changes under the user's feet on upgrade: skills log a
    load error and instruction files fail in complete silence. Observed exactly
    that way — the committed lock pins 0.6.12 and works, while a freshly locked
    Apps bundle resolved 0.7.5 and served an agent with no memory at all.

    Override by setting ``orchestration.deep_agent.backend`` explicitly. An
    explicit ``FilesystemBackend`` that omits ``virtual_mode`` from its ``args``
    inherits deepagents' default and so carries the version-dependent behaviour
    described above into a user's own config, where none of the reasoning above is
    visible; that case is warned about rather than corrected, because overriding a
    value the user wrote out by hand is worse than telling them about it.
    """
    if spec is None:
        if needs_filesystem:
            from deepagents.backends import FilesystemBackend

            # Log the swap: it is the decision that determines whether skills and
            # instruction files can load at all, and leaving it silent means an
            # operator reading endpoint logs cannot tell a filesystem-backed
            # agent from deepagents' default one.
            logger.info(
                "Defaulting deep_agent backend to FilesystemBackend",
                virtual_mode=False,
                reason="skills or instruction files are declared",
            )
            return FilesystemBackend(virtual_mode=False)
        logger.debug(
            "No skills or instruction files declared; leaving the backend to "
            "deepagents (StateBackend)"
        )
        return None
    module_name, _, attr = spec.name.rpartition(".")
    if not module_name:
        raise ValueError(
            f"backend.name '{spec.name}' must be fully qualified (e.g. 'deepagents.backends.StateBackend')"
        )
    module = importlib.import_module(module_name)
    factory = getattr(module, attr)
    if attr == "FilesystemBackend" and "virtual_mode" not in spec.args:
        logger.warning(
            "backend declares FilesystemBackend without virtual_mode, so the "
            "value comes from whichever deepagents is installed: <0.7.0 defaults "
            "to False (real host paths) and >=0.7.0 to True, which re-anchors "
            "every absolute path under root_dir and cannot reach /Volumes at all. "
            "Set virtual_mode explicitly in backend.args — False to keep real "
            "host paths, which is what dao-ai's skill and instruction-file "
            "resolvers hand the backend.",
            backend=spec.name,
            args=sorted(spec.args),
        )
    return factory(**spec.args) if spec.args else factory()


def _agent_to_subagent(agent: AgentModel, config: AppConfig) -> DeepAgentsSubAgent:
    """Translate a dao-ai ``AgentModel`` to deepagents' ``SubAgent`` TypedDict.

    Carries over: name, description, prompt → system_prompt, tools, model, middleware,
    response_format. Drops ``requires`` and ``recursion_limit`` (no analog in deepagents)
    and ``handoff_prompt`` (only meaningful in swarm/supervisor handoffs). Logs at
    DEBUG when fields are dropped so the user can audit.
    """
    if agent.requires or agent.recursion_limit is not None or agent.handoff_prompt:
        logger.debug(
            "Dropping non-deepagents AgentModel fields when used as a sub_agent",
            agent=agent.name,
            dropped_requires=agent.requires,
            dropped_recursion_limit=agent.recursion_limit,
            dropped_handoff_prompt=bool(agent.handoff_prompt),
        )

    sub: dict[str, Any] = {
        "name": agent.name,
        "description": agent.description or f"{agent.name} agent",
        "system_prompt": _resolve_system_prompt(agent.prompt) or "",
    }
    if agent.tools:
        sub["tools"] = _resolve_tools(list(agent.tools), config)
    if agent.model:
        sub["model"] = _resolve_model(agent.model)
    if agent.middleware:
        sub["middleware"] = _resolve_middleware(agent.middleware)
    if agent.response_format is not None:
        sub["response_format"] = _resolve_response_format(agent.response_format)
    return cast("DeepAgentsSubAgent", sub)


def _resolve_subagent(
    spec: SubAgentModel | AgentModel | str, config: AppConfig
) -> DeepAgentsSubAgent:
    """Resolve a sub-agent spec (any of three forms) to deepagents' SubAgent dict."""
    if isinstance(spec, str):
        # Look up in app.agents (a list) by name.
        if config.app is None or not config.app.agents:
            raise ValueError(
                f"sub_agent reference '{spec}' provided but no agents are declared in app.agents"
            )
        match = next((a for a in config.app.agents if a.name == spec), None)
        if match is None:
            available = [a.name for a in config.app.agents]
            raise ValueError(
                f"sub_agent reference '{spec}' not found in app.agents. Available: {available}"
            )
        return _agent_to_subagent(match, config)

    if isinstance(spec, AgentModel):
        return _agent_to_subagent(spec, config)

    # SubAgentModel → translate field-by-field.
    sub: dict[str, Any] = {
        "name": spec.name,
        "description": spec.description,
        "system_prompt": _resolve_system_prompt(spec.system_prompt) or "",
    }
    if spec.tools:
        sub["tools"] = _resolve_tools(list(spec.tools), config)
    if spec.model is not None:
        sub["model"] = _resolve_model(spec.model)
    if spec.middleware:
        sub["middleware"] = _resolve_middleware(spec.middleware)
    if spec.interrupt_on:
        sub["interrupt_on"] = _resolve_interrupt_on(spec.interrupt_on)
    if spec.skills:
        sub["skills"] = resolve_skill_runtime_paths(list(spec.skills), config)
    if spec.permissions:
        sub["permissions"] = _resolve_permissions(spec.permissions)
    if spec.response_format is not None:
        sub["response_format"] = _resolve_response_format(spec.response_format)
    return cast("DeepAgentsSubAgent", sub)


def create_deep_agent_graph(config: AppConfig) -> CompiledStateGraph:
    """
    Build a deepagents-style compiled graph from dao-ai config.

    Mirrors the structure of ``create_supervisor_graph`` and ``create_swarm_graph``:
    pulls all configuration from ``AppConfig``, wires checkpointer/store from the
    shared ``OrchestrationModel.memory`` block, and returns a ``CompiledStateGraph``
    that drops in wherever the supervisor and swarm graphs are used.

    Args:
        config: The application configuration. Must have
            ``config.app.orchestration.deep_agent`` set.

    Returns:
        A compiled LangGraph state machine produced by ``deepagents.create_deep_agent``.

    Raises:
        ValueError: if no ``deep_agent`` block is configured.
    """
    from dao_ai._extras import require_extra

    require_extra("deepagents", feature="Deep Agent orchestration")
    from deepagents import create_deep_agent

    orchestration: OrchestrationModel = config.app.orchestration
    deep_agent: DeepAgentModel | None = orchestration.deep_agent
    if deep_agent is None:
        raise ValueError(
            "create_deep_agent_graph called but config.app.orchestration.deep_agent is not set"
        )

    # ``skills_count`` alone reads 0 for a config whose skills live only on a
    # sub-agent, which is a shape that loads skills perfectly well — so report
    # those too rather than leaving the log implying nothing was declared. Only
    # ``SubAgentModel`` is counted because ``_agent_to_subagent`` does not carry
    # an ``AgentModel``'s skills into deepagents, so those never load.
    subagent_skills_count: int = sum(
        len(spec.skills or [])
        for spec in deep_agent.subagents or []
        if isinstance(spec, SubAgentModel)
    )
    logger.info(
        "Creating deep_agent graph",
        pattern="deepagents",
        agents_count=len(config.app.agents) if config.app.agents else 0,
        subagents_count=len(deep_agent.subagents or []),
        skills_count=len(deep_agent.skills or []),
        subagent_skills_count=subagent_skills_count,
        instruction_files_count=len(deep_agent.instruction_files or []),
        tools_count=len(deep_agent.tools or []),
    )

    store: BaseStore | None = create_store(orchestration)
    checkpointer: BaseCheckpointSaver | None = create_checkpointer(orchestration)

    # Sub-agents: explicit ``deep_agent.subagents`` entries are processed
    # first so they claim names. Then every AgentModel in ``app.agents``
    # that isn't already claimed is added as an implicit sub-agent. This
    # removes the need to re-list agents under ``subagents:`` once they
    # appear under ``app.agents``.
    subagents: list[DeepAgentsSubAgent] = []
    seen_names: set[str] = set()
    for spec in deep_agent.subagents or []:
        sub = _resolve_subagent(spec, config)
        subagents.append(sub)
        seen_names.add(sub["name"])
    for agent in config.app.agents or []:
        if agent.name in seen_names:
            continue
        subagents.append(_agent_to_subagent(agent, config))
        seen_names.add(agent.name)

    skills: list[str] = resolve_skill_runtime_paths(
        list(deep_agent.skills or []), config
    )
    instruction_files: list[str] = resolve_instruction_file_runtime_paths(
        list(deep_agent.instruction_files or []), config
    )

    graph: CompiledStateGraph = create_deep_agent(
        model=_resolve_model(deep_agent.model),
        tools=_resolve_tools(list(deep_agent.tools or []), config),
        system_prompt=_resolve_system_prompt(deep_agent.system_prompt),
        middleware=_resolve_middleware(deep_agent.middleware),
        subagents=subagents or None,
        skills=skills or None,
        # NB: dao-ai exposes deepagents' ``memory=`` parameter (AGENTS.md
        # instruction files) as ``instruction_files`` on DeepAgentModel to
        # avoid collision with OrchestrationModel.memory (runtime
        # checkpointer/store/extraction). The keyword we pass to deepagents
        # remains ``memory`` because that's the upstream API name.
        memory=instruction_files or None,
        permissions=_resolve_permissions(deep_agent.permissions) or None,
        response_format=_resolve_response_format(deep_agent.response_format),
        context_schema=_resolve_context_schema(deep_agent.context_schema),
        checkpointer=checkpointer,
        store=store,
        # Sub-agent skills count too: they are loaded by a SkillsMiddleware that
        # runs on the backend configured here, so a config whose skills live only
        # on a sub-agent would otherwise get deepagents' StateBackend and could
        # never load them by any path. Read off the resolved sub-agent dicts
        # rather than the specs, so this asks exactly what deepagents receives.
        backend=_resolve_backend(
            deep_agent.backend,
            needs_filesystem=bool(skills or instruction_files)
            or any(sub.get("skills") for sub in subagents),
        ),
        interrupt_on=_resolve_interrupt_on(deep_agent.interrupt_on) or None,
        debug=deep_agent.debug,
        name=deep_agent.name,
    )

    if deep_agent.recursion_limit is not None:
        # LangGraph recursion_limit is set on the runnable config at invoke
        # time, but we can also stamp it on the compiled graph for callers
        # that use graph.with_config(...). Mirrors swarm.max_hops behavior.
        graph = graph.with_config({"recursion_limit": deep_agent.recursion_limit})

    return graph
