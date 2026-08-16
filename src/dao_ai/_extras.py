"""Optional-dependency ("extras") management for dao-ai.

dao-ai ships a lean core install. Features that pull heavyweight, rarely-used
dependency trees are gated behind pip *extras* so ``pip install dao-ai`` stays
small and fast:

===========  ==================  ====================================
Extra        Package             Feature
===========  ==================  ====================================
``a2a``      ``a2a-sdk``         Google Agent-to-Agent protocol
``rerank``   ``flashrank``       FlashRank cross-encoder reranking
``deepagents``  ``deepagents``   Deep-agent middleware (skills, subagents, ...)
``memory``   ``langmem``         LangMem long-term memory + extraction
``search``   ``ddgs``            DuckDuckGo web search tool
``excel``    ``openpyxl``        Excel dataset ingest
===========  ==================  ====================================

This module is the single source of truth for that mapping. It provides:

* :func:`require_extra` — a runtime guard raising a friendly, actionable
  ``ImportError`` when a feature is used without its extra installed.
* :func:`resolve_required_extras` — inspect a loaded :class:`AppConfig` and
  return exactly the extras that config exercises, so every ``generate-*`` /
  ``deploy`` path can pin ``dao-ai[<extras>]`` precisely.
* :func:`resolve_required_extras_or_all` — the same, but returns ``{"all"}`` in
  notebook sessions where installing everything is the convenient default.
* :func:`format_extras_suffix` / :func:`expand_all` — helpers for emitting the
  ``[a2a,rerank]`` requirement suffix and expanding ``all`` to its members.

The module is intentionally import-light: it must be importable without any of
the optional packages present, and :class:`AppConfig` is referenced only under
``TYPE_CHECKING`` so importing ``dao_ai._extras`` never triggers
``dao_ai.config``'s heavy import chain.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from types import ModuleType
from typing import TYPE_CHECKING, Any, Final, Literal

from loguru import logger

if TYPE_CHECKING:
    from dao_ai.config import (
        AppConfig,
        BaseRetrieverModel,
        MemoryModel,
        MiddlewareModel,
    )

# Deployment targets the resolver distinguishes. Only Model Serving omits the
# always-on A2A routes (they mount on the Apps FastAPI stack, never on MS).
DeployTarget = Literal["apps", "mcp", "pipeline", "model_serving"]

# Extra name → the importable top-level module that the extra provides. Probing
# this module tells us whether the extra is installed.
_EXTRA_TO_IMPORT: Final[dict[str, str]] = {
    "a2a": "a2a",
    "rerank": "flashrank",
    "deepagents": "deepagents",
    "memory": "langmem",
    "search": "ddgs",
    "excel": "openpyxl",
}

# The concrete feature extras bundled by ``dao-ai[all]`` (deterministic order).
ALL_EXTRAS: Final[tuple[str, ...]] = (
    "a2a",
    "rerank",
    "deepagents",
    "memory",
    "search",
    "excel",
)

# Middleware factory FQNs that require the ``deepagents`` package. Referenced by
# ``MiddlewareModel.name`` in config. ``create_summarization_middleware`` is
# deliberately absent — the plain summarization path is deepagents-free; only
# ``create_deep_summarization_middleware`` needs it.
_DEEPAGENT_MIDDLEWARE_FQNS: Final[frozenset[str]] = frozenset(
    {
        "dao_ai.middleware.skills.create_skills_middleware",
        "dao_ai.middleware.subagent.create_subagent_middleware",
        "dao_ai.middleware.filesystem.create_filesystem_middleware",
        "dao_ai.middleware.memory_agents.create_agents_memory_middleware",
        "dao_ai.middleware.summarization.create_deep_summarization_middleware",
    }
)


def _iter_middleware(config: "AppConfig") -> Iterator["MiddlewareModel"]:
    """Every middleware entry a deploy will actually build.

    The top-level ``middleware`` registry is only half of it. Each agent carries
    its own ``middleware`` list, and that is where ``agent.skills`` end up —
    ``AppConfig._translate_agent_skills_to_middleware`` rewrites every skill into
    a ``create_skills_middleware`` entry on the agent and clears
    ``agent.skills``. Walking only the registry therefore missed every skill, so
    a config whose agent uses skills deployed without the ``deepagents`` extra
    and died at model load with "Skills middleware requires the 'deepagents'
    extra". Orchestration patterns hold middleware lists of their own.

    Agents are reachable through two registries that a YAML anchor makes the
    same object (``config.agents`` and ``config.app.agents``), so dedupe by
    identity — the same traversal the translating validator uses.
    """
    yield from config.middleware.values()

    seen: set[int] = set()
    holders: list[Any] = list((config.agents or {}).values())
    app = config.app
    if app is not None and app.agents:
        holders.extend(app.agents)
    orchestration = app.orchestration if app is not None else None
    if orchestration is not None:
        # ``swarm: true`` is a Literal, not a model — it carries no middleware.
        holders.extend(
            holder
            for holder in (
                orchestration.supervisor,
                orchestration.swarm,
                orchestration.deep_agent,
            )
            if holder is not None and holder is not True
        )
        if orchestration.deep_agent is not None:
            holders.extend(orchestration.deep_agent.subagents or [])

    for holder in holders:
        if id(holder) in seen:
            continue
        seen.add(id(holder))
        yield from holder.middleware or []


def require_extra(
    extra: str,
    *,
    feature: str,
    package: str | None = None,
) -> ModuleType:
    """Import an optional package, raising a friendly error if it is missing.

    Call this at the top of a function that uses an optional feature, then use
    the returned module (or import the concrete names afterwards, now that the
    package is known to be importable).

    Args:
        extra: The pip extra that provides the package (e.g. ``"a2a"``).
        feature: Human-readable feature name used in the error message
            (e.g. ``"Agent-to-Agent (A2A) tools"``).
        package: The importable module to probe. Defaults to the module mapped
            from ``extra`` in :data:`_EXTRA_TO_IMPORT`.

    Returns:
        The imported module.

    Raises:
        ImportError: If the package is not installed, with an actionable
            ``pip install 'dao-ai[<extra>]'`` hint.
    """
    module_name: str = package or _EXTRA_TO_IMPORT.get(extra, extra)
    try:
        return importlib.import_module(module_name)
    except ImportError as err:
        raise ImportError(
            f"{feature} requires the '{extra}' extra, which is not installed. "
            f"Install it with: pip install 'dao-ai[{extra}]' (or "
            f"'dao-ai[all]' for every optional feature)."
        ) from err


def resolve_required_extras(
    config: "AppConfig", target: DeployTarget = "apps"
) -> set[str]:
    """Return the set of extras a given config actually exercises.

    Detection is intentionally conservative — biased toward *including* an
    extra when in doubt. A false positive merely installs a small extra
    package; a false negative ships a deployment whose feature crashes at
    runtime with a missing-dependency error.

    Args:
        config: A loaded :class:`AppConfig`.
        target: The deployment target the extras are resolved for. Governs
            whether the always-on A2A server routes count toward the ``a2a``
            extra. A2A routes mount only on Databricks Apps (and the MCP
            server / provisioning pipeline that reuse the Apps FastAPI stack),
            not on Model Serving — so for ``"model_serving"`` the ``a2a`` extra
            is included only when an explicit A2A *tool* is configured.
            Defaults to ``"apps"`` (the most inclusive, route-bearing target).

    Returns:
        A set of extra names drawn from :data:`ALL_EXTRAS`.
    """
    # Import the config models here (not at module scope) so ``dao_ai._extras``
    # stays import-light; by the time a config is resolved these are loaded.
    from dao_ai.config import (
        AiSearchToolModel,
        DatasetFormat,
        FunctionType,
        LakebaseSearchToolModel,
    )

    extras: set[str] = set()

    # Tools. ``function`` may be a bare reference string (unresolved) or a
    # concrete BaseFunctionModel; only the latter carries a type / retriever.
    for tool in config.tools.values():
        function = tool.function
        if isinstance(function, str):
            continue
        if function.type == FunctionType.A2A:
            extras.add("a2a")
        elif function.type == FunctionType.SEARCH:
            extras.add("search")
        # Search-family tools carry a retriever that may declare reranking.
        if isinstance(function, (AiSearchToolModel, LakebaseSearchToolModel)):
            if _retriever_needs_flashrank(function.retriever):
                extras.add("rerank")

    # Standalone retrievers declared at the top level.
    for retriever in config.retrievers.values():
        if _retriever_needs_flashrank(retriever):
            extras.add("rerank")

    app = config.app

    # A2A server routes. ``AppModel.a2a`` defaults to enabled, but the routes
    # mount only on the Apps FastAPI stack (Apps / MCP server / pipeline),
    # never on Model Serving. So the default-enabled routes pull the ``a2a``
    # extra everywhere EXCEPT Model Serving, where only an explicit A2A tool
    # (handled above) does — keeping the serving image lean.
    if target != "model_serving" and app is not None and app.a2a.enabled:
        extras.add("a2a")

    # Deep-agent orchestration or deepagents-backed middleware.
    # Orchestration is an App-level setting (``config.app.orchestration``).
    orchestration = app.orchestration if app is not None else None
    if orchestration is not None and orchestration.deep_agent is not None:
        extras.add("deepagents")
    for mw in _iter_middleware(config):
        if mw.name in _DEEPAGENT_MIDDLEWARE_FQNS:
            extras.add("deepagents")

    # LangMem long-term memory store / extraction (a checkpointer alone is core
    # — it uses langgraph-checkpoint-postgres, not langmem).
    if _memory_needs_langmem(config.memory):
        extras.add("memory")
    if orchestration is not None and _memory_needs_langmem(orchestration.memory):
        extras.add("memory")

    # Excel dataset ingest.
    for dataset in config.datasets or []:
        if dataset.format == DatasetFormat.EXCEL:
            extras.add("excel")

    logger.debug(
        "Resolved required extras from config",
        extras=sorted(extras),
        target=target,
    )
    return extras


def resolve_required_extras_or_all(
    config: "AppConfig", target: DeployTarget = "apps"
) -> set[str]:
    """Resolve extras precisely, or return ``{"all"}`` in notebook sessions.

    In an interactive notebook, installing every extra is the convenient
    default (the user is iterating, not optimizing a production image), so we
    short-circuit to ``all``. Every non-notebook path (CLI, Model Serving,
    generated bundles) gets the precise, size-minimal set.

    Args:
        config: A loaded :class:`AppConfig`.
        target: Deployment target, forwarded to :func:`resolve_required_extras`
            (see its docs — governs the always-on A2A routes for Model Serving).
    """
    from dao_ai.utils import is_in_notebook

    if is_in_notebook():
        logger.debug("Notebook session detected — requiring all extras")
        return {"all"}
    return resolve_required_extras(config, target=target)


def expand_all(extras: set[str]) -> set[str]:
    """Expand a set that may contain ``"all"`` into concrete feature extras."""
    if "all" not in extras:
        return set(extras)
    expanded: set[str] = set(extras)
    expanded.discard("all")
    expanded.update(ALL_EXTRAS)
    return expanded


def format_extras_suffix(extras: set[str]) -> str:
    """Format extras as a requirement suffix, e.g. ``"[a2a,rerank]"``.

    Returns an empty string when ``extras`` is empty, so callers can build
    ``f"dao-ai{format_extras_suffix(extras)}"`` unconditionally. The extras are
    sorted for deterministic, diff-stable output.
    """
    if not extras:
        return ""
    return "[" + ",".join(sorted(extras)) + "]"


def _retriever_needs_flashrank(retriever: "BaseRetrieverModel | None") -> bool:
    """Return True if a retriever config requires the FlashRank package.

    FlashRank is needed for a local cross-encoder rerank pass (``rerank: true``
    or ``rerank.model``) or for any instruction-aware rerank stage. It is NOT
    needed for Databricks server-side reranking (``rerank.columns`` only).
    """
    if retriever is None:
        return False

    rerank = retriever.rerank
    # ``rerank: true`` (a bare bool that survived validation) → FlashRank.
    if rerank is True:
        return True
    # A RerankParametersModel with a FlashRank model set → FlashRank.
    # (``rerank`` is ``RerankParametersModel | bool | None``; a False bool and
    # a columns-only model both fall through to the instruction check.)
    if rerank is not None and not isinstance(rerank, bool) and rerank.model:
        return True

    # Instruction-aware reranking runs a FlashRank stage first.
    if retriever.instructed is not None and retriever.instructed.rerank is not None:
        return True

    return False


def _memory_needs_langmem(memory: "MemoryModel | None") -> bool:
    """Return True if a memory config requires the LangMem package.

    A long-term ``store`` or ``extraction`` block is backed by langmem. A
    ``checkpointer`` alone is core (langgraph-checkpoint-postgres).
    """
    if memory is None:
        return False
    return memory.store is not None or memory.extraction is not None
