"""Tests for the R1+R2 refinements to deep_agent orchestration.

R1: ``AgentModel.skills`` + ``SkillModel.as_middleware()``
R2: Implicit subagents from ``app.agents`` + dict-form ``deep_agent.subagents``

Run with:
    pytest tests/dao_ai/test_deep_agent_refinements.py -v -m unit
"""

from __future__ import annotations

import pytest
from langgraph.graph.state import CompiledStateGraph
from pydantic import ValidationError

from dao_ai.config import (
    AgentModel,
    AppConfig,
    AppModel,
    DeepAgentModel,
    LLMModel,
    OrchestrationModel,
    ResourcesModel,
    SkillModel,
    SubAgentModel,
)

# ---------------------------------------------------------------------------
# R1: AgentModel.skills translation to middleware
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAgentSkillsTranslation:
    """``AgentModel.skills`` is translated to ``MiddlewareModel`` entries at AppConfig load."""

    def test_inline_skill_translated_to_middleware(self) -> None:
        skill = SkillModel(name="research", path="skills/research")
        agent = AgentModel(name="researcher", model=LLMModel(name="x"), skills=[skill])
        AppConfig(
            agents={"researcher": agent},
            resources=ResourcesModel(),
            app=AppModel(
                name="test_app",
                agents=[agent],
                orchestration=OrchestrationModel(deep_agent=DeepAgentModel()),
            ),
        )
        # After AppConfig validator runs, skills are cleared and middleware
        # has one entry pointing at the SkillsMiddleware factory.
        assert agent.skills == []
        assert len(agent.middleware) == 1
        assert (
            agent.middleware[0].name
            == "dao_ai.middleware.skills.create_skills_middleware"
        )
        # ``sources`` is the parent of the resolved skill leaf (deepagents
        # source-dir convention). When the runtime resolver can't find the
        # leaf in this test environment, the parent of the raw path is used.
        assert len(agent.middleware[0].args["sources"]) == 1
        assert agent.middleware[0].args["backend_type"] == "filesystem"

    def test_string_skill_resolved_via_resources(self) -> None:
        skill = SkillModel(name="research", path="skills/research")
        agent = AgentModel(
            name="researcher",
            model=LLMModel(name="x"),
            skills=["research"],  # string ref
        )
        AppConfig(
            agents={"researcher": agent},
            resources=ResourcesModel(skills={"research": skill}),
            app=AppModel(
                name="test_app",
                agents=[agent],
                orchestration=OrchestrationModel(deep_agent=DeepAgentModel()),
            ),
        )
        assert agent.skills == []
        assert agent.middleware[0].args["backend_type"] == "filesystem"
        assert len(agent.middleware[0].args["sources"]) == 1

    def test_unknown_skill_name_errors(self) -> None:
        agent = AgentModel(
            name="researcher",
            model=LLMModel(name="x"),
            skills=["does_not_exist"],
        )
        with pytest.raises(ValidationError) as exc:
            AppConfig(
                agents={"researcher": agent},
                resources=ResourcesModel(),
                app=AppModel(
                    name="test_app",
                    agents=[agent],
                ),
            )
        assert "does_not_exist" in str(exc.value)

    def test_multiple_skills_produce_multiple_middleware(self) -> None:
        agent = AgentModel(
            name="x",
            model=LLMModel(name="x"),
            skills=[
                SkillModel(name="a", path="skills/a"),
                SkillModel(name="b", path="skills/b"),
            ],
        )
        AppConfig(
            agents={"x": agent},
            resources=ResourcesModel(),
            app=AppModel(
                name="test_app",
                agents=[agent],
                orchestration=OrchestrationModel(deep_agent=DeepAgentModel()),
            ),
        )
        # Each SkillModel produces one middleware entry — the actual
        # SkillsMiddleware instance dedup/aggregation can happen at
        # construction time but the config-level translation is 1:1.
        assert len(agent.middleware) == 2

    def test_idempotent_on_repeated_load(self) -> None:
        """Running the AppConfig validator twice on the same agent must not double-append."""
        skill = SkillModel(name="research", path="skills/research")
        agent = AgentModel(name="researcher", model=LLMModel(name="x"), skills=[skill])
        AppConfig(
            agents={"researcher": agent},
            resources=ResourcesModel(),
            app=AppModel(
                name="test_app",
                agents=[agent],
                orchestration=OrchestrationModel(deep_agent=DeepAgentModel()),
            ),
        )
        # Re-running validator (model_rebuild via repeated init).
        AppConfig(
            agents={"researcher": agent},
            resources=ResourcesModel(),
            app=AppModel(
                name="test_app",
                agents=[agent],
                orchestration=OrchestrationModel(deep_agent=DeepAgentModel()),
            ),
        )
        # Only one middleware entry survives even with two AppConfig loads.
        assert len(agent.middleware) == 1


# ---------------------------------------------------------------------------
# R2: dict-form subagents
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDictFormSubagents:
    def test_dict_form_injects_name_from_key(self) -> None:
        d = DeepAgentModel.model_validate(
            {
                "system_prompt": "plan",
                "subagents": {
                    "research": {
                        "description": "Research products",
                        "system_prompt": "You research.",
                    },
                    "coder": {
                        "description": "Write code",
                        "system_prompt": "You code.",
                    },
                },
            }
        )
        assert len(d.subagents) == 2
        names = {s.name for s in d.subagents}
        assert names == {"research", "coder"}

    def test_dict_form_explicit_name_overrides_key(self) -> None:
        d = DeepAgentModel.model_validate(
            {
                "subagents": {
                    "research": {
                        "name": "explicit_name",
                        "description": "x",
                        "system_prompt": "x",
                    },
                },
            }
        )
        assert d.subagents[0].name == "explicit_name"

    def test_list_form_still_works(self) -> None:
        d = DeepAgentModel.model_validate(
            {
                "subagents": [
                    {"name": "a", "description": "x", "system_prompt": "x"},
                    "main_agent",
                ],
            }
        )
        assert len(d.subagents) == 2
        assert d.subagents[0].name == "a"
        assert d.subagents[1] == "main_agent"


# ---------------------------------------------------------------------------
# R2: Implicit subagent inclusion + validator relaxation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestImplicitSubagents:
    def test_empty_app_agents_allowed_with_deep_agent(self) -> None:
        AppConfig(
            resources=ResourcesModel(),
            app=AppModel(
                name="test_app",
                agents=[],
                orchestration=OrchestrationModel(
                    deep_agent=DeepAgentModel(system_prompt="hi")
                ),
            ),
        )

    def test_empty_app_agents_rejected_without_deep_agent(self) -> None:
        with pytest.raises(ValidationError):
            AppConfig(
                resources=ResourcesModel(),
                app=AppModel(
                    name="test_app",
                    agents=[],
                ),
            )

    def test_implicit_subagent_from_app_agents(self, tmp_path, monkeypatch) -> None:
        """An AgentModel in ``app.agents`` is auto-included as a sub-agent
        when deep_agent is the orchestration mode."""
        from dao_ai.orchestration.deep_agent import create_deep_agent_graph

        monkeypatch.delenv("DAO_AI_PROJECT_ROOT", raising=False)
        researcher = AgentModel(
            name="researcher",
            description="web research",
            model=LLMModel(name="x"),
            prompt="You research.",
        )
        cfg = AppConfig(
            agents={"researcher": researcher},
            resources=ResourcesModel(),
            app=AppModel(
                name="test_app",
                agents=[researcher],
                orchestration=OrchestrationModel(
                    deep_agent=DeepAgentModel(system_prompt="plan")
                ),
            ),
        )
        graph = create_deep_agent_graph(cfg)
        # The grpah builds (researcher was carried in as a sub-agent
        # without being re-listed under deep_agent.subagents).
        assert isinstance(graph, CompiledStateGraph)

    def test_explicit_subagent_wins_over_app_agents(
        self, tmp_path, monkeypatch
    ) -> None:
        """When the same name appears in both ``app.agents`` and
        ``deep_agent.subagents``, the explicit entry takes precedence."""
        from dao_ai.orchestration.deep_agent import (
            _agent_to_subagent,
            _resolve_subagent,
        )

        monkeypatch.delenv("DAO_AI_PROJECT_ROOT", raising=False)
        # app.agents version
        base = AgentModel(
            name="research",
            description="generic researcher",
            model=LLMModel(name="x"),
            prompt="generic prompt",
        )
        # explicit subagent override with different prompt
        override = SubAgentModel(
            name="research",
            description="sporting goods researcher",
            system_prompt="You research SPORTING GOODS specifically.",
        )
        cfg = AppConfig(
            agents={"research": base},
            resources=ResourcesModel(),
            app=AppModel(
                name="test_app",
                agents=[base],
                orchestration=OrchestrationModel(
                    deep_agent=DeepAgentModel(subagents=[override])
                ),
            ),
        )
        # Manually walk the merge logic to assert who wins
        seen: set[str] = set()
        subs = []
        for spec in cfg.app.orchestration.deep_agent.subagents or []:
            sub = _resolve_subagent(spec, cfg)
            subs.append(sub)
            seen.add(sub["name"])
        for agent in cfg.app.agents or []:
            if agent.name in seen:
                continue
            subs.append(_agent_to_subagent(agent, cfg))

        # Only ONE sub-agent named 'research' — the explicit one.
        research_subs = [s for s in subs if s["name"] == "research"]
        assert len(research_subs) == 1
        assert "SPORTING GOODS" in research_subs[0]["system_prompt"]
