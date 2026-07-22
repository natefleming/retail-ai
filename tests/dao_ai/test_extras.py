"""Unit tests for dao_ai._extras — the optional-dependency (extras) resolver.

Covers the friendly ``require_extra`` guard, config→extras detection with its
conservative edge cases, deployment-target awareness (A2A routes vs Model
Serving), and the requirement-suffix / expand helpers. The resolver now uses
typed ``isinstance`` checks against the real config models, so these tests
build genuine model instances rather than duck types.
"""

import pytest

from dao_ai import _extras
from dao_ai.config import (
    A2AModel,
    A2AToolModel,
    AgentModel,
    AiSearchRetrieverModel,
    AiSearchToolModel,
    AppConfig,
    AppModel,
    ColumnInfo,
    DatasetModel,
    DeepAgentModel,
    IndexModel,
    InferenceEndpointModel,
    InstructedRetrieverModel,
    InstructionAwareRerankModel,
    MemoryModel,
    MiddlewareModel,
    OrchestrationModel,
    RerankParametersModel,
    SearchToolModel,
    StoreModel,
    ToolModel,
    VectorStoreModel,
)


def _app(**kwargs) -> AppModel:
    """A minimal valid AppModel (requires a name + at least one agent)."""
    kwargs.setdefault("name", "my-app")
    # deployment_target=apps avoids the registered_model requirement; the
    # resolver's target arg is independent of this field.
    kwargs.setdefault("deployment_target", "apps")
    kwargs.setdefault(
        "agents",
        [AgentModel(name="ag", model=InferenceEndpointModel(name="databricks-gpt-oss-120b"))],
    )
    return AppModel(**kwargs)


def _ai_search_retriever(*, rerank=None, instructed=None) -> AiSearchRetrieverModel:
    """A minimal valid AiSearchRetrieverModel (requires a vector store index)."""
    return AiSearchRetrieverModel(
        vector_store=VectorStoreModel(index=IndexModel(name="cat.sch.idx")),
        rerank=rerank,
        instructed=instructed,
    )


# ---------------------------------------------------------------------------
# require_extra
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_require_extra_missing_raises_friendly_error() -> None:
    with pytest.raises(ImportError) as exc_info:
        _extras.require_extra(
            "a2a",
            feature="A2A tools",
            package="dao_ai_definitely_not_installed_xyz",
        )
    msg = str(exc_info.value)
    assert "A2A tools" in msg
    assert "pip install 'dao-ai[a2a]'" in msg
    assert "dao-ai[all]" in msg


@pytest.mark.unit
def test_require_extra_present_returns_module() -> None:
    # A stdlib module always importable — stands in for an installed extra.
    mod = _extras.require_extra("search", feature="Search", package="json")
    assert mod.__name__ == "json"


# ---------------------------------------------------------------------------
# format_extras_suffix / expand_all
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_format_extras_suffix_empty() -> None:
    assert _extras.format_extras_suffix(set()) == ""


@pytest.mark.unit
def test_format_extras_suffix_sorted_deterministic() -> None:
    assert _extras.format_extras_suffix({"rerank", "a2a"}) == "[a2a,rerank]"
    assert _extras.format_extras_suffix({"a2a", "rerank"}) == "[a2a,rerank]"


@pytest.mark.unit
def test_expand_all_expands_sentinel() -> None:
    assert _extras.expand_all({"all"}) == set(_extras.ALL_EXTRAS)


@pytest.mark.unit
def test_expand_all_passthrough_without_sentinel() -> None:
    assert _extras.expand_all({"a2a", "rerank"}) == {"a2a", "rerank"}


@pytest.mark.unit
def test_expand_all_merges_sentinel_and_concrete() -> None:
    assert _extras.expand_all({"all", "memory"}) == set(_extras.ALL_EXTRAS)


# ---------------------------------------------------------------------------
# _retriever_needs_flashrank — real BaseRetrieverModel instances
# ---------------------------------------------------------------------------
def _retriever(*, rerank=None, instructed=None) -> AiSearchRetrieverModel:
    return _ai_search_retriever(rerank=rerank, instructed=instructed)


@pytest.mark.unit
def test_flashrank_columns_only_does_not_trigger() -> None:
    # Databricks server-side rerank (columns only) does NOT need flashrank.
    retriever = _retriever(rerank=RerankParametersModel(columns=["product_name"]))
    assert _extras._retriever_needs_flashrank(retriever) is False


@pytest.mark.unit
def test_flashrank_model_triggers() -> None:
    retriever = _retriever(rerank=RerankParametersModel(model="ms-marco-MiniLM-L-12-v2"))
    assert _extras._retriever_needs_flashrank(retriever) is True


@pytest.mark.unit
def test_flashrank_bool_true_triggers() -> None:
    # ``rerank: true`` is normalized to a default FlashRank model by a
    # validator, so it also needs flashrank.
    retriever = _retriever(rerank=True)
    assert _extras._retriever_needs_flashrank(retriever) is True


@pytest.mark.unit
def test_flashrank_instructed_triggers() -> None:
    instructed = InstructedRetrieverModel(
        columns=[ColumnInfo(name="c")],
        rerank=InstructionAwareRerankModel(),
    )
    retriever = _retriever(instructed=instructed)
    assert _extras._retriever_needs_flashrank(retriever) is True


@pytest.mark.unit
def test_flashrank_none_and_absent() -> None:
    assert _extras._retriever_needs_flashrank(None) is False
    assert _extras._retriever_needs_flashrank(_retriever()) is False


# ---------------------------------------------------------------------------
# _memory_needs_langmem — real MemoryModel instances
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_memory_checkpointer_only_is_core() -> None:
    # A bare MemoryModel (no store / no extraction) uses langgraph checkpointing,
    # NOT langmem.
    assert _extras._memory_needs_langmem(MemoryModel()) is False


@pytest.mark.unit
def test_memory_store_triggers_langmem() -> None:
    assert _extras._memory_needs_langmem(MemoryModel(store=StoreModel(name="s"))) is True


@pytest.mark.unit
def test_memory_none() -> None:
    assert _extras._memory_needs_langmem(None) is False


# ---------------------------------------------------------------------------
# resolve_required_extras — real AppConfig, target awareness
# ---------------------------------------------------------------------------
def _config(*, tools=None, middleware=None, memory=None, datasets=None, app=None) -> AppConfig:
    """Build a minimal real AppConfig for the resolver."""
    return AppConfig(
        tools=tools or {},
        middleware=middleware or {},
        memory=memory,
        datasets=datasets,
        app=app,
    )


@pytest.mark.unit
def test_a2a_default_routes_included_for_apps_not_model_serving() -> None:
    # AppModel.a2a defaults to enabled=True; routes mount on Apps, not MS.
    cfg = _config(app=_app(a2a=A2AModel(enabled=True)))
    assert _extras.resolve_required_extras(cfg, target="apps") == {"a2a"}
    assert _extras.resolve_required_extras(cfg, target="model_serving") == set()


@pytest.mark.unit
def test_a2a_tool_included_on_every_target() -> None:
    # An explicit A2A tool needs a2a-sdk regardless of deployment target.
    tools = {
        "remote": ToolModel(
            name="remote", function=A2AToolModel(type="a2a", endpoint="https://x")
        )
    }
    cfg = _config(tools=tools, app=_app(a2a=A2AModel(enabled=False)))
    assert "a2a" in _extras.resolve_required_extras(cfg, target="apps")
    assert "a2a" in _extras.resolve_required_extras(cfg, target="model_serving")


@pytest.mark.unit
def test_a2a_disabled_and_no_tool_omits_extra() -> None:
    cfg = _config(app=_app(a2a=A2AModel(enabled=False)))
    assert "a2a" not in _extras.resolve_required_extras(cfg, target="apps")


@pytest.mark.unit
def test_search_tool_triggers_search_extra() -> None:
    tools = {"web": ToolModel(name="web", function=SearchToolModel(type="search"))}
    cfg = _config(tools=tools, app=_app(a2a=A2AModel(enabled=False)))
    assert _extras.resolve_required_extras(cfg, target="model_serving") == {"search"}


@pytest.mark.unit
def test_rerank_tool_triggers_rerank_extra() -> None:
    retriever = _ai_search_retriever(
        rerank=RerankParametersModel(model="ms-marco-MiniLM-L-12-v2")
    )
    tools = {
        "vs": ToolModel(
            name="vs",
            function=AiSearchToolModel(type="ai_search", retriever=retriever),
        )
    }
    cfg = _config(tools=tools, app=_app(a2a=A2AModel(enabled=False)))
    assert _extras.resolve_required_extras(cfg, target="model_serving") == {"rerank"}


@pytest.mark.unit
def test_deep_agent_orchestration_triggers_deepagents() -> None:
    app = _app(
        a2a=A2AModel(enabled=False),
        orchestration=OrchestrationModel(deep_agent=DeepAgentModel()),
    )
    cfg = _config(app=app)
    assert _extras.resolve_required_extras(cfg, target="model_serving") == {"deepagents"}


@pytest.mark.unit
def test_deepagents_middleware_fqn_triggers_deepagents() -> None:
    fqn = "dao_ai.middleware.skills.create_skills_middleware"
    cfg = _config(
        middleware={"m": MiddlewareModel(name=fqn)},
        app=_app(a2a=A2AModel(enabled=False)),
    )
    assert _extras.resolve_required_extras(cfg, target="model_serving") == {"deepagents"}


@pytest.mark.unit
def test_plain_summarization_does_not_trigger_deepagents() -> None:
    fqn = "dao_ai.middleware.summarization.create_summarization_middleware"
    cfg = _config(
        middleware={"m": MiddlewareModel(name=fqn)},
        app=_app(a2a=A2AModel(enabled=False)),
    )
    assert _extras.resolve_required_extras(cfg, target="model_serving") == set()


@pytest.mark.unit
def test_memory_store_triggers_memory_extra() -> None:
    cfg = _config(
        memory=MemoryModel(store=StoreModel(name="s")),
        app=_app(a2a=A2AModel(enabled=False)),
    )
    assert _extras.resolve_required_extras(cfg, target="model_serving") == {"memory"}


@pytest.mark.unit
def test_excel_dataset_triggers_excel_extra() -> None:
    cfg = _config(
        datasets=[DatasetModel(format="excel")],
        app=_app(a2a=A2AModel(enabled=False)),
    )
    assert _extras.resolve_required_extras(cfg, target="model_serving") == {"excel"}


# ---------------------------------------------------------------------------
# resolve_required_extras_or_all — notebook override
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_notebook_override_returns_all(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dao_ai.utils.is_in_notebook", lambda: True)
    cfg = _config(app=_app(a2a=A2AModel(enabled=False)))
    assert _extras.resolve_required_extras_or_all(cfg) == {"all"}


@pytest.mark.unit
def test_non_notebook_returns_precise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("dao_ai.utils.is_in_notebook", lambda: False)
    cfg = _config(app=_app(a2a=A2AModel(enabled=True)))
    assert _extras.resolve_required_extras_or_all(cfg, target="apps") == {"a2a"}
