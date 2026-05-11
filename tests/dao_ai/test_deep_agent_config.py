"""Pydantic model tests for the deep_agent orchestration block.

Run with:
    pytest tests/dao_ai/test_deep_agent_config.py -v -m unit
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dao_ai.config import (
    BackendModel,
    DeepAgentModel,
    FilesystemPermissionModel,
    HumanInTheLoopModel,
    OrchestrationModel,
    SkillModel,
    SubAgentModel,
    SupervisorModel,
    SwarmModel,
    VolumeModel,
    VolumePathModel,
)

# ---------------------------------------------------------------------------
# SkillModel
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSkillModel:
    def test_local_skill_minimal(self) -> None:
        s = SkillModel(name="research", path="skills/research")
        assert s.is_volume_backed is False
        assert s.runtime_path == "skills/research"
        assert s.as_resources() == []

    def test_as_middleware_local(self) -> None:
        s = SkillModel(name="research", path="skills/research")
        mw = s.as_middleware()
        assert mw.name == "dao_ai.middleware.skills.create_skills_middleware"
        assert mw.args["backend_type"] == "filesystem"
        assert mw.args["root_dir"] == "/"
        # sources is the PARENT of the resolved leaf (deepagents source-dir
        # convention). When the runtime resolver can't find the leaf, sources
        # falls back to the parent of the raw declared path.
        assert len(mw.args["sources"]) == 1

    def test_as_middleware_volume(self) -> None:
        s = SkillModel(name="g", path="/Volumes/c/s/v/r")
        mw = s.as_middleware()
        assert mw.args["backend_type"] == "volume"
        assert "volume_path" in mw.args
        # Volume sources is the parent of the volume leaf path.
        assert mw.args["sources"] == ["/Volumes/c/s/v"]

    def test_local_skill_with_description(self) -> None:
        s = SkillModel(name="research", path="skills/research", description="docs")
        assert s.description == "docs"

    def test_volume_skill_via_structured_path(self) -> None:
        vol = VolumeModel(name="skills_vol")
        s = SkillModel(
            name="research",
            path=VolumePathModel(volume=vol, path="research"),
        )
        assert s.is_volume_backed is True
        # full_name composes /Volumes/<cat>/<schema>/<vol>/<sub> when the
        # volume has a schema attached; without a schema it falls back to
        # the bare path on the model.
        assert "research" in s.runtime_path

    def test_volume_skill_string_auto_promoted(self) -> None:
        s = SkillModel(name="x", path="/Volumes/a/b/c/d")
        assert s.is_volume_backed is True
        assert isinstance(s.path, VolumePathModel)
        assert s.runtime_path == "/Volumes/a/b/c/d"

    def test_volume_path_dict_form(self) -> None:
        s = SkillModel(
            name="x",
            path={"path": "/Volumes/cat/schema/vol/sub"},  # type: ignore[arg-type]
        )
        assert s.is_volume_backed is True

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SkillModel(name="x", path="skills/x", bogus="nope")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# FilesystemPermissionModel + BackendModel
# (HumanInTheLoopModel reuse for interrupt_on covered in test_hitl_config_model.py)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFilesystemPermissionModel:
    def test_minimal(self) -> None:
        p = FilesystemPermissionModel(paths=["/skills/**"])
        assert p.mode == "allow"
        assert p.operations is None

    def test_deny(self) -> None:
        p = FilesystemPermissionModel(
            paths=["/etc/**"], mode="deny", operations=["write"]
        )
        assert p.mode == "deny"
        assert p.operations == ["write"]


@pytest.mark.unit
class TestHumanInTheLoopModelInInterruptOn:
    """Confirm HumanInTheLoopModel works as the interrupt_on value type
    (i.e. that the consolidation onto a single HITL model is wired correctly)."""

    def test_defaults(self) -> None:
        c = HumanInTheLoopModel()
        assert c.allowed_decisions == ["approve", "edit", "reject"]
        assert c.review_prompt is None

    def test_custom_with_respond(self) -> None:
        c = HumanInTheLoopModel(
            allowed_decisions=["approve", "respond"],
            review_prompt="hitl review",
        )
        assert "respond" in c.allowed_decisions
        assert c.review_prompt == "hitl review"

    def test_deep_agent_accepts_human_in_the_loop_model(self) -> None:
        d = DeepAgentModel(
            interrupt_on={
                "write_file": HumanInTheLoopModel(allowed_decisions=["approve"]),
                "execute": True,
            }
        )
        assert isinstance(d.interrupt_on["write_file"], HumanInTheLoopModel)
        assert d.interrupt_on["execute"] is True


@pytest.mark.unit
class TestBackendModel:
    def test_minimal(self) -> None:
        b = BackendModel(name="deepagents.backends.StateBackend")
        assert b.args == {}

    def test_with_args(self) -> None:
        b = BackendModel(name="some.backend", args={"foo": "bar"})
        assert b.args["foo"] == "bar"


# ---------------------------------------------------------------------------
# SubAgentModel
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSubAgentModel:
    def test_required_fields(self) -> None:
        s = SubAgentModel(
            name="research",
            description="web research",
            system_prompt="You are a researcher.",
        )
        assert s.tools == []
        assert s.skills == []

    def test_string_tool_refs(self) -> None:
        s = SubAgentModel(
            name="research",
            description="x",
            system_prompt="prompt",
            tools=["search", "weather"],
        )
        assert s.tools == ["search", "weather"]

    def test_skill_refs(self) -> None:
        s = SubAgentModel(
            name="research",
            description="x",
            system_prompt="prompt",
            skills=[
                SkillModel(name="r", path="skills/research"),
                "skills/coding",
            ],
        )
        assert len(s.skills) == 2


# ---------------------------------------------------------------------------
# DeepAgentModel
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDeepAgentModel:
    def test_minimal(self) -> None:
        d = DeepAgentModel()
        assert d.tools == []
        assert d.subagents == []
        assert d.skills == []
        assert d.debug is False

    def test_with_string_model(self) -> None:
        d = DeepAgentModel(model="anthropic:claude-sonnet-4-6")
        assert d.model == "anthropic:claude-sonnet-4-6"

    def test_with_inline_subagent(self) -> None:
        d = DeepAgentModel(
            subagents=[
                SubAgentModel(
                    name="research",
                    description="x",
                    system_prompt="p",
                ),
            ]
        )
        assert len(d.subagents) == 1

    def test_with_string_subagent(self) -> None:
        d = DeepAgentModel(subagents=["product_agent"])
        assert d.subagents == ["product_agent"]

    def test_recursion_limit_validation(self) -> None:
        with pytest.raises(ValidationError):
            DeepAgentModel(recursion_limit=0)


# ---------------------------------------------------------------------------
# OrchestrationModel mutual exclusion
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOrchestrationModelMutualExclusion:
    def test_supervisor_only(self) -> None:
        o = OrchestrationModel(supervisor=SupervisorModel(model={"name": "x"}))
        assert o.supervisor is not None
        assert o.swarm is None
        assert o.deep_agent is None

    def test_swarm_only(self) -> None:
        o = OrchestrationModel(swarm=True)
        assert o.swarm is not None
        assert o.supervisor is None
        assert o.deep_agent is None

    def test_deep_agent_only(self) -> None:
        o = OrchestrationModel(deep_agent=DeepAgentModel())
        assert o.deep_agent is not None
        assert o.supervisor is None
        assert o.swarm is None

    def test_supervisor_plus_deep_agent_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc:
            OrchestrationModel(
                supervisor=SupervisorModel(model={"name": "x"}),
                deep_agent=DeepAgentModel(),
            )
        assert "supervisor" in str(exc.value)
        assert "deep_agent" in str(exc.value)

    def test_swarm_plus_deep_agent_rejected(self) -> None:
        with pytest.raises(ValidationError) as exc:
            OrchestrationModel(swarm=SwarmModel(), deep_agent=DeepAgentModel())
        assert "swarm" in str(exc.value)
        assert "deep_agent" in str(exc.value)

    def test_all_three_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OrchestrationModel(
                supervisor=SupervisorModel(model={"name": "x"}),
                swarm=SwarmModel(),
                deep_agent=DeepAgentModel(),
            )

    def test_none_set_allowed(self) -> None:
        # Existing behavior: AppConfig auto-picks a router when none specified.
        o = OrchestrationModel()
        assert o.supervisor is None
        assert o.swarm is None
        assert o.deep_agent is None
