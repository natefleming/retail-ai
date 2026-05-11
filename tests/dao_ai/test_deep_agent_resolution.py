"""Tests for the deep_agent resolver helpers.

These exercise the translation from dao-ai primitives (LLMModel, ToolModel,
SubAgentModel, AgentModel, SkillModel) into the shapes that
``deepagents.create_deep_agent`` consumes. The full graph build is covered by
the integration test; here we keep things fast and pure.

Run with:
    pytest tests/dao_ai/test_deep_agent_resolution.py -v -m unit
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dao_ai.config import (
    AgentModel,
    AppConfig,
    AppModel,
    DeepAgentModel,
    FilesystemPermissionModel,
    HumanInTheLoopModel,
    LLMModel,
    OrchestrationModel,
    ResourcesModel,
    SkillModel,
    SubAgentModel,
)
from dao_ai.orchestration.deep_agent import (
    _agent_to_subagent,
    _resolve_interrupt_on,
    _resolve_model,
    _resolve_permissions,
    _resolve_subagent,
    _resolve_system_prompt,
)


@pytest.fixture
def tmp_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "skills" / "research").mkdir(parents=True)
    (tmp_path / "skills" / "research" / "SKILL.md").write_text("# r")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'tmp'\n")
    monkeypatch.setenv("DAO_AI_PROJECT_ROOT", str(tmp_path))
    return tmp_path


def _minimal_config(*, agents: list[AgentModel] | None = None) -> AppConfig:
    agents = agents or [AgentModel(name="default", model=LLMModel(name="x"))]
    return AppConfig(
        agents={a.name: a for a in agents},
        resources=ResourcesModel(),
        app=AppModel(
            name="test_app",
            deployment_target="apps",
            agents=agents,
            orchestration=OrchestrationModel(deep_agent=DeepAgentModel()),
        ),
    )


# ---------------------------------------------------------------------------
# _resolve_model
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResolveModel:
    def test_none(self) -> None:
        assert _resolve_model(None) is None

    def test_string(self) -> None:
        assert (
            _resolve_model("anthropic:claude-sonnet-4-6")
            == "anthropic:claude-sonnet-4-6"
        )

    def test_llm_model_instantiates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sentinel = object()
        monkeypatch.setattr(LLMModel, "as_chat_model", lambda self: sentinel)
        assert _resolve_model(LLMModel(name="x")) is sentinel


# ---------------------------------------------------------------------------
# _resolve_system_prompt
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResolveSystemPrompt:
    def test_none(self) -> None:
        assert _resolve_system_prompt(None) is None

    def test_string_passthrough(self) -> None:
        assert _resolve_system_prompt("hello") == "hello"


# ---------------------------------------------------------------------------
# _resolve_permissions, _resolve_interrupt_on
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResolvePermissions:
    def test_empty(self) -> None:
        assert _resolve_permissions([]) == []

    def test_translates_fields(self) -> None:
        # FilesystemPermission is a dataclass (not TypedDict) — attribute access.
        result = _resolve_permissions(
            [
                FilesystemPermissionModel(
                    paths=["/skills/**"], mode="deny", operations=["write"]
                )
            ]
        )
        assert len(result) == 1
        assert result[0].paths == ["/skills/**"]
        assert result[0].mode == "deny"
        assert result[0].operations == ["write"]

    def test_default_operations_both(self) -> None:
        result = _resolve_permissions([FilesystemPermissionModel(paths=["/skills/**"])])
        assert result[0].operations == ["read", "write"]


@pytest.mark.unit
class TestResolveInterruptOn:
    def test_empty(self) -> None:
        assert _resolve_interrupt_on({}) == {}

    def test_bool_true_expands_to_default_decisions(self) -> None:
        """``True`` is expanded to the default decisions by the shared converter,
        matching the behavior of tool-level ``human_in_the_loop: true`` annotations."""
        result = _resolve_interrupt_on({"write_file": True})
        assert result["write_file"] == {
            "allowed_decisions": ["approve", "edit", "reject"]
        }

    def test_bool_false_disables(self) -> None:
        result = _resolve_interrupt_on({"write_file": False})
        assert result["write_file"] is False

    def test_config_translated(self) -> None:
        # Reuses the existing HumanInTheLoopModel; review_prompt becomes
        # description in the LangChain InterruptOnConfig TypedDict.
        result = _resolve_interrupt_on(
            {
                "write_file": HumanInTheLoopModel(
                    allowed_decisions=["approve", "respond"],
                    review_prompt="hitl",
                )
            }
        )
        assert "respond" in result["write_file"]["allowed_decisions"]
        assert result["write_file"]["description"] == "hitl"


# ---------------------------------------------------------------------------
# _agent_to_subagent and _resolve_subagent
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAgentToSubagent:
    def test_minimal_carry_over(self) -> None:
        cfg = _minimal_config()
        agent = AgentModel(
            name="research",
            description="researcher",
            model=LLMModel(name="x"),
            prompt="You are a researcher.",
        )
        sub = _agent_to_subagent(agent, cfg)
        assert sub["name"] == "research"
        assert sub["description"] == "researcher"
        assert sub["system_prompt"] == "You are a researcher."

    def test_drops_requires_and_recursion_limit(self) -> None:
        cfg = _minimal_config()
        agent = AgentModel(
            name="research",
            description="x",
            model=LLMModel(name="x"),
            prompt="p",
            recursion_limit=10,
        )
        # Allow requires field by adding another agent
        sub = _agent_to_subagent(agent, cfg)
        assert "recursion_limit" not in sub
        assert "requires" not in sub
        assert "handoff_prompt" not in sub


@pytest.mark.unit
class TestResolveSubagent:
    def test_string_form_resolves_against_app_agents(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agents = [
            AgentModel(
                name="research",
                description="r",
                model=LLMModel(name="x"),
                prompt="p",
            ),
        ]
        cfg = _minimal_config(agents=agents)
        sub = _resolve_subagent("research", cfg)
        assert sub["name"] == "research"

    def test_string_form_unknown_name_errors(self) -> None:
        cfg = _minimal_config()
        with pytest.raises(ValueError) as exc:
            _resolve_subagent("not_an_agent", cfg)
        assert "not_an_agent" in str(exc.value)

    def test_inline_subagent_model(self) -> None:
        cfg = _minimal_config()
        spec = SubAgentModel(
            name="r",
            description="researcher",
            system_prompt="You research.",
        )
        sub = _resolve_subagent(spec, cfg)
        assert sub["name"] == "r"
        assert sub["description"] == "researcher"
        assert sub["system_prompt"] == "You research."

    def test_agent_model_form(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = _minimal_config()
        agent = AgentModel(
            name="r",
            description="x",
            model=LLMModel(name="x"),
            prompt="p",
        )
        sub = _resolve_subagent(agent, cfg)
        assert sub["name"] == "r"
        assert sub["system_prompt"] == "p"

    def test_inline_subagent_with_skills(self, tmp_project: Path) -> None:
        cfg = _minimal_config()
        spec = SubAgentModel(
            name="r",
            description="x",
            system_prompt="p",
            skills=[SkillModel(name="research", path="skills/research")],
        )
        sub = _resolve_subagent(spec, cfg)
        assert "skills" in sub
        assert len(sub["skills"]) == 1
        # deepagents SkillsMiddleware wants source dirs (parent of leaf), not
        # the skill folder itself, so the resolver returns the parent.
        assert sub["skills"][0].endswith("/skills")
