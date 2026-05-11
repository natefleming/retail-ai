"""Tests for the skills module — code_paths collection and runtime resolution.

Run with:
    pytest tests/dao_ai/test_skills.py -v -m unit
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

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
    VolumeModel,
    VolumePathModel,
)
from dao_ai.skills import (
    _iter_deep_agent_skills,
    _project_root,
    collect_skills_code_paths,
    resolve_skill_runtime_paths,
)

if TYPE_CHECKING:
    pass


@pytest.fixture
def tmp_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a temp project dir with skills/ subdirs and point _project_root at it."""
    skills_dir = tmp_path / "skills" / "research"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text("# research skill")

    other = tmp_path / "skills" / "coding"
    other.mkdir(parents=True)
    (other / "SKILL.md").write_text("# coding skill")

    # Mark this dir as a project root so _project_root() picks it up.
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'tmp'\n")

    monkeypatch.setenv("DAO_AI_PROJECT_ROOT", str(tmp_path))
    return tmp_path


def _config_with_deep_agent(
    *,
    skills: list,
    subagent_skills: list | None = None,
    resource_skills: dict | None = None,
) -> AppConfig:
    deep = DeepAgentModel(skills=skills)
    if subagent_skills is not None:
        deep.subagents = [
            SubAgentModel(
                name="research",
                description="x",
                system_prompt="p",
                skills=subagent_skills,
            )
        ]
    return AppConfig(
        agents={"a": AgentModel(name="a", model=LLMModel(name="x"))},
        resources=ResourcesModel(skills=resource_skills or {}),
        app=AppModel(
            name="test_app",
            deployment_target="apps",
            agents=[AgentModel(name="a", model=LLMModel(name="x"))],
            orchestration=OrchestrationModel(deep_agent=deep),
        ),
    )


@pytest.mark.unit
class TestProjectRoot:
    def test_env_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DAO_AI_PROJECT_ROOT", str(tmp_path))
        assert _project_root() == tmp_path.resolve()


@pytest.mark.unit
class TestIterDeepAgentSkills:
    def test_no_orchestration(self) -> None:
        cfg = AppConfig()
        assert _iter_deep_agent_skills(cfg) == []

    def test_no_deep_agent(self, tmp_project: Path) -> None:
        cfg = AppConfig(
            agents={"a": AgentModel(name="a", model=LLMModel(name="x"))},
            app=AppModel(
                name="test_app",
                deployment_target="apps",
                agents=[AgentModel(name="a", model=LLMModel(name="x"))],
                orchestration=OrchestrationModel(),
            ),
        )
        assert _iter_deep_agent_skills(cfg) == []

    def test_top_level_only(self, tmp_project: Path) -> None:
        cfg = _config_with_deep_agent(skills=["skills/research"])
        result = _iter_deep_agent_skills(cfg)
        assert result == ["skills/research"]

    def test_includes_subagent_skills(self, tmp_project: Path) -> None:
        cfg = _config_with_deep_agent(
            skills=["skills/research"], subagent_skills=["skills/coding"]
        )
        result = _iter_deep_agent_skills(cfg)
        assert "skills/research" in result
        assert "skills/coding" in result


@pytest.mark.unit
class TestCollectSkillsCodePaths:
    """``collect_skills_code_paths`` returns the *parent* skills/ directory so
    mlflow preserves the full ``skills/<vertical>/<skill>/`` layout at
    ``<model_dir>/code/skills/...`` instead of flattening to ``<model_dir>/code/<skill>/``.
    """

    def test_local_skill_returns_skills_root(self, tmp_project: Path) -> None:
        cfg = _config_with_deep_agent(skills=["skills/research"])
        paths = collect_skills_code_paths(cfg)
        assert len(paths) == 1
        assert Path(paths[0]).is_absolute()
        # Returns the parent skills/ dir (so mlflow preserves the full layout).
        assert paths[0].endswith("/skills")
        assert (Path(paths[0]) / "research" / "SKILL.md").exists()

    def test_volume_skill_excluded(self, tmp_project: Path) -> None:
        skill = SkillModel(name="g", path="/Volumes/c/s/v/g")
        cfg = _config_with_deep_agent(skills=[skill])
        paths = collect_skills_code_paths(cfg)
        assert paths == []

    def test_mixed_local_and_volume(self, tmp_project: Path) -> None:
        cfg = _config_with_deep_agent(
            skills=[
                "skills/research",
                SkillModel(name="g", path="/Volumes/c/s/v/g"),
            ]
        )
        paths = collect_skills_code_paths(cfg)
        # One entry: the parent skills/ dir for the local skill.
        assert len(paths) == 1
        assert paths[0].endswith("/skills")

    def test_only_volume_skills_returns_empty(self, tmp_project: Path) -> None:
        cfg = _config_with_deep_agent(
            skills=[SkillModel(name="g", path="/Volumes/c/s/v/g")]
        )
        assert collect_skills_code_paths(cfg) == []

    def test_dedup(self, tmp_project: Path) -> None:
        # Multiple references to local skills still produce exactly one
        # code_paths entry (the parent skills/ dir).
        cfg = _config_with_deep_agent(
            skills=["skills/research"], subagent_skills=["skills/research"]
        )
        paths = collect_skills_code_paths(cfg)
        assert len(paths) == 1

    def test_missing_dir_returns_empty(self, tmp_project: Path) -> None:
        cfg = _config_with_deep_agent(skills=["skills/does_not_exist"])
        paths = collect_skills_code_paths(cfg)
        assert paths == []

    def test_named_resource_lookup(self, tmp_project: Path) -> None:
        skill = SkillModel(name="research", path="skills/research")
        cfg = _config_with_deep_agent(
            skills=["research"],  # string ref, looked up in resources.skills
            resource_skills={"research": skill},
        )
        paths = collect_skills_code_paths(cfg)
        assert len(paths) == 1
        assert paths[0].endswith("/skills")


@pytest.mark.unit
class TestCollectLocalSkillDirs:
    """``collect_local_skill_dirs`` returns each leaf skill dir (used by Apps bundle copy)."""

    def test_returns_each_local_dir(self, tmp_project: Path) -> None:
        from dao_ai.skills import collect_local_skill_dirs

        cfg = _config_with_deep_agent(
            skills=["skills/research"], subagent_skills=["skills/coding"]
        )
        paths = collect_local_skill_dirs(cfg)
        assert len(paths) == 2
        assert any(p.endswith("/skills/research") for p in paths)
        assert any(p.endswith("/skills/coding") for p in paths)

    def test_excludes_volume_skills(self, tmp_project: Path) -> None:
        from dao_ai.skills import collect_local_skill_dirs

        cfg = _config_with_deep_agent(
            skills=[
                "skills/research",
                SkillModel(name="g", path="/Volumes/c/s/v/g"),
            ]
        )
        paths = collect_local_skill_dirs(cfg)
        assert len(paths) == 1
        assert paths[0].endswith("/skills/research")


@pytest.mark.unit
class TestRuntimeAnchorResolution:
    """Verify the runtime anchor resolver finds skills under both Apps and
    Model Serving deployment layouts using the SAME yaml config.

    These tests simulate the two production layouts:

    * Apps:    ``<bundle_root>/skills/<vertical>/<skill>/`` with CWD == bundle root
    * Serving: ``<model_dir>/code/skills/<vertical>/<skill>/`` with
               ``<model_dir>/code`` on ``sys.path`` (mlflow ``code_paths`` extraction)
    """

    def test_apps_layout_via_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Apps deploys with skills/ at the bundle root and CWD == bundle root.

        The resolver returns the *parent* of the skill leaf (the deepagents
        ``SkillsMiddleware`` source-dir convention).
        """
        bundle_root = tmp_path / "bundle"
        skill_dir = bundle_root / "skills" / "sg" / "research"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# r")

        monkeypatch.chdir(bundle_root)
        # No DAO_AI_PROJECT_ROOT — simulate the deployed env.
        monkeypatch.delenv("DAO_AI_PROJECT_ROOT", raising=False)

        skill = SkillModel(name="research", path="skills/sg/research")
        from dao_ai.skills import resolve_skill_runtime_paths

        paths = resolve_skill_runtime_paths([skill], AppConfig())
        assert len(paths) == 1
        # Parent dir: skills/sg/  (contains research/SKILL.md as a subdir)
        assert paths[0].endswith("skills/sg")
        assert (Path(paths[0]) / "research" / "SKILL.md").exists()

    def test_model_serving_layout_via_sys_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Model Serving extracts code_paths into <model_dir>/code/skills/...
        and prepends <model_dir>/code to sys.path. The resolver must find the
        skill via that sys.path entry, not CWD, then return the parent.
        """
        # Simulate the mlflow extraction layout.
        model_dir = tmp_path / "model"
        code_dir = model_dir / "code"
        skill_dir = code_dir / "skills" / "sg" / "research"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# r served via mlflow")

        # CWD is somewhere unrelated (mimics serving worker).
        unrelated = tmp_path / "unrelated"
        unrelated.mkdir()
        monkeypatch.chdir(unrelated)
        monkeypatch.delenv("DAO_AI_PROJECT_ROOT", raising=False)

        # Mlflow prepends <model_dir>/code to sys.path at model-load time.
        monkeypatch.syspath_prepend(str(code_dir))

        skill = SkillModel(name="research", path="skills/sg/research")
        from dao_ai.skills import resolve_skill_runtime_paths

        paths = resolve_skill_runtime_paths([skill], AppConfig())
        assert len(paths) == 1
        # Parent dir: <model>/code/skills/sg
        assert paths[0].endswith("skills/sg")
        assert "model/code/skills/sg" in paths[0]
        assert (Path(paths[0]) / "research" / "SKILL.md").exists()

    def test_dedup_when_skills_share_parent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two SkillModel entries under the same parent dir produce ONE entry."""
        bundle_root = tmp_path / "bundle"
        for sub in ("research", "product_lookup"):
            d = bundle_root / "skills" / "sg" / sub
            d.mkdir(parents=True)
            (d / "SKILL.md").write_text(f"# {sub}")

        monkeypatch.chdir(bundle_root)
        monkeypatch.delenv("DAO_AI_PROJECT_ROOT", raising=False)

        from dao_ai.skills import resolve_skill_runtime_paths

        paths = resolve_skill_runtime_paths(
            [
                SkillModel(name="research", path="skills/sg/research"),
                SkillModel(name="product_lookup", path="skills/sg/product_lookup"),
            ],
            AppConfig(),
        )
        # Both skills share parent skills/sg → one source path.
        assert len(paths) == 1
        assert paths[0].endswith("skills/sg")

    def test_volume_path_returns_parent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Volume-backed skills return the *parent* of the volume path (the
        deepagents source-dir convention) so the middleware can list the
        volume root and discover skill subdirs."""
        unrelated = tmp_path / "nothing"
        unrelated.mkdir()
        monkeypatch.chdir(unrelated)
        monkeypatch.delenv("DAO_AI_PROJECT_ROOT", raising=False)

        skill = SkillModel(name="g", path="/Volumes/cat/sch/vol/skill_v1")
        from dao_ai.skills import resolve_skill_runtime_paths

        paths = resolve_skill_runtime_paths([skill], AppConfig())
        assert paths == ["/Volumes/cat/sch/vol"]

    def test_mixed_local_and_volume_runtime(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both shapes resolve to parent dirs in the same config."""
        bundle_root = tmp_path / "bundle"
        local_skill = bundle_root / "skills" / "sg" / "research"
        local_skill.mkdir(parents=True)
        (local_skill / "SKILL.md").write_text("# local")

        monkeypatch.chdir(bundle_root)
        monkeypatch.delenv("DAO_AI_PROJECT_ROOT", raising=False)

        from dao_ai.skills import resolve_skill_runtime_paths

        paths = resolve_skill_runtime_paths(
            [
                SkillModel(name="local", path="skills/sg/research"),
                SkillModel(name="vol", path="/Volumes/cat/sch/vol/skill"),
            ],
            AppConfig(),
        )
        assert len(paths) == 2
        # Each becomes the parent (source dir).
        assert any(p.endswith("skills/sg") for p in paths)
        assert "/Volumes/cat/sch/vol" in paths

    def test_missing_local_skipped_with_anchors_logged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog
    ) -> None:
        """When no anchor finds the skill, the resolver skips and logs anchors tried."""
        bundle_root = tmp_path / "bundle"
        bundle_root.mkdir()
        monkeypatch.chdir(bundle_root)
        monkeypatch.delenv("DAO_AI_PROJECT_ROOT", raising=False)

        skill = SkillModel(name="missing", path="skills/sg/missing")
        from dao_ai.skills import resolve_skill_runtime_paths

        paths = resolve_skill_runtime_paths([skill], AppConfig())
        assert paths == []


@pytest.mark.unit
class TestResolveSkillRuntimePaths:
    def test_volume_returns_parent_string_form(self, tmp_project: Path) -> None:
        skill = SkillModel(name="g", path="/Volumes/c/s/v/g")
        paths = resolve_skill_runtime_paths([skill], AppConfig())
        # Parent of the volume path becomes the source dir.
        assert paths == ["/Volumes/c/s/v"]

    def test_volume_returns_parent_structured_form(self, tmp_project: Path) -> None:
        vol = VolumeModel(name="vol_name")
        skill = SkillModel(
            name="g",
            path=VolumePathModel(volume=vol, path="research"),
        )
        paths = resolve_skill_runtime_paths([skill], AppConfig())
        # Parent of the composed full_name. Without a schema attached the
        # composition returns the bare subpath; either way the resolver
        # consistently strips one path component for deepagents.
        assert len(paths) == 1

    def test_local_resolves_to_parent(self, tmp_project: Path) -> None:
        cfg = _config_with_deep_agent(skills=[])
        paths = resolve_skill_runtime_paths(
            [SkillModel(name="r", path="skills/research")], cfg
        )
        assert len(paths) == 1
        assert Path(paths[0]).is_absolute()
        # Parent of skills/research is skills/
        assert paths[0].endswith("/skills")

    def test_missing_local_skipped(self, tmp_project: Path) -> None:
        cfg = _config_with_deep_agent(skills=[])
        paths = resolve_skill_runtime_paths(
            [SkillModel(name="r", path="skills/missing")], cfg
        )
        assert paths == []
