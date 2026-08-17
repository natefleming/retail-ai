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
    _runtime_anchors,
    assert_skills_resolvable,
    collect_skills_code_paths,
    config_skill_anchor,
    resolve_skill_runtime_paths,
    resolve_skill_source_dir,
    skill_anchors,
    unresolvable_skills,
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


@pytest.mark.unit
class TestAsMiddlewareIsPure:
    """``SkillModel.as_middleware`` must not touch the filesystem.

    It runs during pydantic validation — i.e. on whichever machine *loaded* the
    config, which for a provisioning job or a git-sourced deploy is not the
    machine that will run the agent. When it probed the filesystem and stored what
    it found, ``create_agent`` serialized the loader's own directory into the model
    artifact and the endpoint came up healthy with no skills at all.
    """

    def test_local_source_stays_relative(self, tmp_project: Path) -> None:
        mw = SkillModel(name="r", path="skills/research").as_middleware()
        assert mw.args["sources"] == ["skills"]
        assert mw.args["backend_type"] == "filesystem"

    def test_identical_regardless_of_cwd_or_project_root(
        self, tmp_project: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole point: two machines must serialize the same config."""
        on_loader = SkillModel(name="r", path="skills/research").as_middleware()

        elsewhere = tmp_path / "nothing_here"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        monkeypatch.delenv("DAO_AI_PROJECT_ROOT", raising=False)
        on_runner = SkillModel(name="r", path="skills/research").as_middleware()

        assert on_loader.args == on_runner.args

    def test_no_absolute_path_even_when_the_dir_exists(self, tmp_project: Path) -> None:
        """``skills/research`` really is present under ``tmp_project`` — the old
        code found it and baked it in. Existence must not change the output."""
        assert (tmp_project / "skills" / "research" / "SKILL.md").is_file()
        mw = SkillModel(name="r", path="skills/research").as_middleware()
        assert not Path(mw.args["sources"][0]).is_absolute()

    def test_bare_leaf_yields_dot_not_filesystem_root(self, tmp_project: Path) -> None:
        """A leaf with no directory component. ``rsplit("/")`` would produce ``/``
        here and point the middleware at the filesystem root."""
        mw = SkillModel(name="r", path="research").as_middleware()
        assert mw.args["sources"] == ["."]

    def test_trailing_slash_ignored(self, tmp_project: Path) -> None:
        mw = SkillModel(name="r", path="skills/research/").as_middleware()
        assert mw.args["sources"] == ["skills"]

    def test_volume_backed_still_absolute(self) -> None:
        """Volume paths are already machine-independent, so nothing is deferred."""
        mw = SkillModel(name="g", path="/Volumes/c/s/v/research").as_middleware()
        assert mw.args["sources"] == ["/Volumes/c/s/v"]
        assert mw.args["backend_type"] == "volume"


@pytest.mark.unit
class TestResolveSkillSourceDir:
    def test_absolute_returns_none_for_passthrough(self, tmp_project: Path) -> None:
        """Absolute sources are the caller's to use verbatim."""
        assert resolve_skill_source_dir("/Volumes/c/s/v/skills") is None
        assert resolve_skill_source_dir("/abs/elsewhere") is None

    def test_leading_slash_is_not_root_dir_relative(self, tmp_project: Path) -> None:
        """``/skills`` under a backend rooted elsewhere is not "relative to the
        root" and cannot be repaired by stripping the slash — it must pass
        through, or resolution would silently invent a different directory."""
        assert resolve_skill_source_dir("/skills") is None

    def test_resolves_against_extra_anchor(self, tmp_path: Path) -> None:
        base = tmp_path / "cfgdir"
        (base / "skills" / "research").mkdir(parents=True)
        (base / "skills" / "research" / "SKILL.md").write_text("# s")
        assert resolve_skill_source_dir("skills", (base,)) == (base / "skills")

    def test_dot_means_the_anchor_itself(self, tmp_path: Path) -> None:
        base = tmp_path / "cfgdir"
        (base / "research").mkdir(parents=True)
        (base / "research" / "SKILL.md").write_text("# s")
        assert resolve_skill_source_dir(".", (base,)) == base

    def test_miss_returns_none(self, tmp_path: Path) -> None:
        assert resolve_skill_source_dir("no/such/dir", (tmp_path,)) is None

    def test_prefers_the_anchor_that_actually_holds_skills(
        self, tmp_path: Path
    ) -> None:
        """An empty same-named dir under an earlier anchor must not shadow the
        real one — a bare CWD in a monorepo is enough to cause this, and the
        symptom is an agent with no skills and no error."""
        decoy = tmp_path / "decoy"
        (decoy / "skills").mkdir(parents=True)
        real = tmp_path / "real"
        (real / "skills" / "research").mkdir(parents=True)
        (real / "skills" / "research" / "SKILL.md").write_text("# s")

        assert resolve_skill_source_dir("skills", (decoy, real)) == (real / "skills")

    def test_falls_back_to_first_existing_when_none_hold_skills(
        self, tmp_path: Path
    ) -> None:
        first = tmp_path / "first"
        (first / "skills").mkdir(parents=True)
        second = tmp_path / "second"
        (second / "skills").mkdir(parents=True)
        assert resolve_skill_source_dir("skills", (first, second)) == (first / "skills")


@pytest.mark.unit
class TestSkillAnchors:
    def test_scope_is_restored(self, tmp_path: Path) -> None:
        """A build must not change how the *next* build resolves."""
        before = _runtime_anchors()
        with skill_anchors(tmp_path):
            assert _runtime_anchors()[0] == tmp_path.resolve()
        assert _runtime_anchors() == before

    def test_restored_even_on_exception(self, tmp_path: Path) -> None:
        before = _runtime_anchors()
        with pytest.raises(RuntimeError):
            with skill_anchors(tmp_path):
                raise RuntimeError("boom")
        assert _runtime_anchors() == before

    def test_config_anchor_empty_without_local_path(self) -> None:
        """A config with no local path — a URL-loaded one, or one built in
        Python — must not be anchored on a guessed local directory. That is the
        boundary ``_reject_relative_assets_for_remote_config`` defends."""
        cfg = AppConfig()
        assert cfg.local_config_path is None
        assert config_skill_anchor(cfg) == ()


@pytest.mark.unit
class TestUnresolvableSkills:
    """Deploy-time paths fail loudly; serve time only warns."""

    def test_clean_config_reports_nothing(self, tmp_project: Path) -> None:
        cfg = _config_with_deep_agent(
            skills=[SkillModel(name="r", path="skills/research")]
        )
        assert unresolvable_skills(cfg) == []

    def test_missing_skill_is_named(self, tmp_project: Path) -> None:
        cfg = _config_with_deep_agent(
            skills=[SkillModel(name="ghost", path="skills/ghost")]
        )
        problems = unresolvable_skills(cfg)
        assert len(problems) == 1
        assert "ghost" in problems[0]

    def test_assert_raises_naming_target_and_skill(self, tmp_project: Path) -> None:
        cfg = _config_with_deep_agent(
            skills=[SkillModel(name="ghost", path="skills/ghost")]
        )
        with pytest.raises(ValueError, match="Model Serving deploy") as err:
            assert_skills_resolvable(cfg, target="Model Serving deploy")
        assert "ghost" in str(err.value)

    def test_volume_backed_never_flagged(self, tmp_project: Path) -> None:
        """A volume path is not checkable from the deploying machine."""
        cfg = _config_with_deep_agent(
            skills=[SkillModel(name="g", path="/Volumes/c/s/v/research")]
        )
        assert unresolvable_skills(cfg) == []


@pytest.mark.unit
class TestRemoteConfigRejectsRelativeSkills:
    """A config fetched over HTTP has no directory of its own, so a relative skill
    path in it can only be resolved against whatever local tree the process
    happens to be standing in — and that Markdown goes straight into the agent's
    prompt. The guard's docstring always claimed to cover skills; now it does."""

    def _reject(self, config: AppConfig) -> list[str]:
        from dao_ai.config import _reject_relative_assets_for_remote_config

        try:
            _reject_relative_assets_for_remote_config(
                config, source="https://example.com/agent.yaml"
            )
        except ValueError as exc:
            return str(exc).splitlines()
        return []

    def test_relative_deep_agent_skill_is_rejected(self) -> None:
        config = _config_with_deep_agent(
            skills=[SkillModel(name="research", path="skills/research")]
        )
        lines = self._reject(config)
        assert any("skills/research" in line for line in lines), lines

    def test_volume_backed_skill_is_accepted(self) -> None:
        """Governed skills are the right way to ship skills with a remote config:
        the path is absolute and names nothing on the local disk."""
        config = _config_with_deep_agent(
            skills=[
                SkillModel(
                    name="governed",
                    path=VolumePathModel(
                        volume=VolumeModel(name="skills_library"), path="research"
                    ),
                )
            ]
        )
        assert self._reject(config) == []

    def test_relative_agent_skill_is_rejected_after_translation(
        self, tmp_project: Path
    ) -> None:
        """Agent-level skills have become middleware ``sources`` by this point, so
        the guard has to look where they landed, not where they were declared."""
        config = AppConfig(
            agents={
                "researcher": AgentModel(
                    name="researcher",
                    model=LLMModel(name="dbx"),
                    skills=[SkillModel(name="research", path="skills/research")],
                )
            }
        )
        lines = self._reject(config)
        assert any("skills" in line for line in lines), lines
