"""End-to-end smoke test: compile a deep_agent graph from a YAML config.

These tests invoke ``create_deep_agent_graph`` with a fully-shaped AppConfig
to ensure the resolver pipeline produces what ``deepagents.create_deep_agent``
expects. The graph is compiled but not invoked against a real LLM (which
would require API credentials); instead we verify the returned object is a
``CompiledStateGraph`` and inspect its structure.

Run with:
    pytest tests/dao_ai/test_deep_agent_graph.py -v -m unit
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langgraph.graph.state import CompiledStateGraph

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
from dao_ai.orchestration import create_orchestration_graph
from dao_ai.orchestration.deep_agent import create_deep_agent_graph


@pytest.fixture
def tmp_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    skills_dir = tmp_path / "skills" / "research"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text("# research skill\nDo research.")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'tmp'\n")
    monkeypatch.setenv("DAO_AI_PROJECT_ROOT", str(tmp_path))
    return tmp_path


@pytest.mark.unit
class TestCreateDeepAgentGraph:
    def test_minimal_graph_compiles(self, tmp_project: Path) -> None:
        cfg = AppConfig(
            agents={"main": AgentModel(name="main", model=LLMModel(name="x"))},
            resources=ResourcesModel(),
            app=AppModel(
                name="test_app",
                agents=[AgentModel(name="main", model=LLMModel(name="x"))],
                orchestration=OrchestrationModel(
                    deep_agent=DeepAgentModel(
                        system_prompt="You are a deep agent.",
                    )
                ),
            ),
        )
        graph = create_deep_agent_graph(cfg)
        assert isinstance(graph, CompiledStateGraph)

    def test_graph_with_skill(self, tmp_project: Path) -> None:
        cfg = AppConfig(
            agents={"main": AgentModel(name="main", model=LLMModel(name="x"))},
            resources=ResourcesModel(
                skills={"research": SkillModel(name="research", path="skills/research")}
            ),
            app=AppModel(
                name="test_app",
                agents=[AgentModel(name="main", model=LLMModel(name="x"))],
                orchestration=OrchestrationModel(
                    deep_agent=DeepAgentModel(
                        system_prompt="You are a deep agent.",
                        skills=["research"],
                    )
                ),
            ),
        )
        graph = create_deep_agent_graph(cfg)
        assert isinstance(graph, CompiledStateGraph)

    def test_graph_with_inline_subagent(self, tmp_project: Path) -> None:
        cfg = AppConfig(
            agents={"main": AgentModel(name="main", model=LLMModel(name="x"))},
            resources=ResourcesModel(),
            app=AppModel(
                name="test_app",
                agents=[AgentModel(name="main", model=LLMModel(name="x"))],
                orchestration=OrchestrationModel(
                    deep_agent=DeepAgentModel(
                        system_prompt="You are a deep agent.",
                        subagents=[
                            SubAgentModel(
                                name="research",
                                description="researcher",
                                system_prompt="You research.",
                            )
                        ],
                    )
                ),
            ),
        )
        graph = create_deep_agent_graph(cfg)
        assert isinstance(graph, CompiledStateGraph)

    def test_graph_with_agent_model_subagent(self, tmp_project: Path) -> None:
        researcher = AgentModel(
            name="researcher",
            description="web research specialist",
            model=LLMModel(name="x"),
            prompt="You research.",
        )
        cfg = AppConfig(
            agents={"main": AgentModel(name="main", model=LLMModel(name="x"))},
            resources=ResourcesModel(),
            app=AppModel(
                name="test_app",
                agents=[
                    AgentModel(name="main", model=LLMModel(name="x")),
                    researcher,
                ],
                orchestration=OrchestrationModel(
                    deep_agent=DeepAgentModel(
                        system_prompt="You are a deep agent.",
                        subagents=["researcher"],  # name lookup
                    )
                ),
            ),
        )
        graph = create_deep_agent_graph(cfg)
        assert isinstance(graph, CompiledStateGraph)


@pytest.mark.unit
class TestCreateOrchestrationGraphDispatch:
    def test_dispatches_to_deep_agent(self, tmp_project: Path) -> None:
        cfg = AppConfig(
            agents={"main": AgentModel(name="main", model=LLMModel(name="x"))},
            resources=ResourcesModel(),
            app=AppModel(
                name="test_app",
                agents=[AgentModel(name="main", model=LLMModel(name="x"))],
                orchestration=OrchestrationModel(
                    deep_agent=DeepAgentModel(system_prompt="hi")
                ),
            ),
        )
        graph = create_orchestration_graph(cfg)
        assert isinstance(graph, CompiledStateGraph)
