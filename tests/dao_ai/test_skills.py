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
    MiddlewareModel,
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
    assert_skill_assets_resolvable,
    collect_instruction_file_code_paths,
    collect_skills_code_paths,
    config_skill_anchor,
    iter_instruction_file_stagings,
    resolve_instruction_file_path,
    resolve_instruction_file_runtime_paths,
    resolve_skill_runtime_paths,
    resolve_skill_source_dir,
    skill_anchors,
    unresolvable_instruction_files,
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
            assert_skill_assets_resolvable(cfg, target="Model Serving deploy")
        assert "ghost" in str(err.value)

    def test_volume_backed_never_flagged(self, tmp_project: Path) -> None:
        """A volume path is not checkable from the deploying machine."""
        cfg = _config_with_deep_agent(
            skills=[SkillModel(name="g", path="/Volumes/c/s/v/research")]
        )
        assert unresolvable_skills(cfg) == []

    @pytest.mark.parametrize(
        "path",
        [
            "/Volumes/c/s/v/research",
            "/Workspace/Shared/skills/research",
            "/dbfs/FileStore/skills/research",
        ],
    )
    def test_fuse_paths_never_flagged(self, path: str, tmp_project: Path) -> None:
        """Every Databricks FUSE root gets the same exemption, not just /Volumes.

        All three are mounted inside Databricks compute and absent from the laptop
        running the CLI, so checking whether one exists says nothing about runtime —
        and failing the deploy over it refuses a config that would have served.
        """
        cfg = _config_with_deep_agent(skills=[SkillModel(name="g", path=path)])
        assert unresolvable_skills(cfg) == []

    @pytest.mark.parametrize(
        "path", ["/Volumesnotreally/research", "/WorkspaceFoo/research"]
    )
    def test_near_miss_prefixes_are_still_checked(
        self, path: str, tmp_project: Path
    ) -> None:
        """The exemption is per path segment, not a substring match."""
        cfg = _config_with_deep_agent(skills=[SkillModel(name="g", path=path)])
        assert len(unresolvable_skills(cfg)) == 1

    def test_bare_string_fuse_path_never_flagged(self, tmp_project: Path) -> None:
        """The inline-string spelling gets the same exemption as the model one."""
        cfg = _config_with_deep_agent(skills=["/Workspace/Shared/skills/research"])
        assert unresolvable_skills(cfg) == []

    @pytest.mark.parametrize("backend_type", [None, "state", "store", "volume"])
    def test_non_filesystem_sources_are_not_gated(
        self, backend_type: str | None, tmp_project: Path
    ) -> None:
        """The gate must agree with the runtime about what a "source" means.

        ``create_skills_middleware`` resolves sources against the filesystem for
        ``backend_type="filesystem"`` only, passing them through verbatim for the
        rest — under those a source is a key into graph state, a Store namespace, or
        a volume path, and nothing about it belongs on the deploying machine's disk.
        Gating them anyway made the gate contradict the runtime and hard-fail
        ``examples/12_middleware/deepagents_middleware.yaml``, which declares
        ``sources: [/skills/base/, /skills/user/]`` with the default ``state``
        backend, blocking every deploy path for a config that serves correctly.

        ``None`` covers the shipped example's shape exactly: no ``backend_type`` at
        all, so the factory's ``"state"`` default applies.
        """
        args: dict = {"sources": ["/skills/base/", "/skills/user/"]}
        if backend_type is not None:
            args["backend_type"] = backend_type
        cfg = AppConfig(
            agents={"a": AgentModel(name="a", model=LLMModel(name="x"))},
            app=AppModel(
                name="test_app",
                agents=[
                    AgentModel(
                        name="a",
                        model=LLMModel(name="x"),
                        middleware=[
                            MiddlewareModel(
                                name="dao_ai.middleware.skills.create_skills_middleware",
                                args=args,
                            )
                        ],
                    )
                ],
            ),
        )
        assert unresolvable_skills(cfg) == []

    def test_filesystem_sources_are_still_gated(self, tmp_project: Path) -> None:
        """The counterpart: an explicitly filesystem-backed source keeps its gate,
        which is the whole point of having one."""
        cfg = AppConfig(
            agents={"a": AgentModel(name="a", model=LLMModel(name="x"))},
            app=AppModel(
                name="test_app",
                agents=[
                    AgentModel(
                        name="a",
                        model=LLMModel(name="x"),
                        middleware=[
                            MiddlewareModel(
                                name="dao_ai.middleware.skills.create_skills_middleware",
                                args={
                                    "sources": ["skills/ghost"],
                                    "backend_type": "filesystem",
                                },
                            )
                        ],
                    )
                ],
            ),
        )
        problems = unresolvable_skills(cfg)
        assert len(problems) == 1
        assert "skills/ghost" in problems[0]



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

    def test_relative_bare_string_skill_is_rejected(self) -> None:
        """The shorter spelling of the same declaration must not be a way around
        the guard. ``skills: [skills/research]`` is a relative local path just as
        much as the ``SkillModel`` form is, and narrowing the check to
        ``SkillModel`` let it through — a remote document naming a local directory,
        which is the one thing this guard exists to stop."""
        config = _config_with_deep_agent(skills=["skills/research"])
        lines = self._reject(config)
        assert any("skills/research" in line for line in lines), lines

    def test_relative_bare_string_subagent_skill_is_rejected(self) -> None:
        config = _config_with_deep_agent(skills=[], subagent_skills=["skills/research"])
        lines = self._reject(config)
        assert any("skills/research" in line for line in lines), lines

    def test_bare_string_naming_a_registry_entry_is_judged_by_its_target(self) -> None:
        """A bare string is two different things: a key into ``resources.skills``
        or an inline path. When it is a key, the path that matters is the one the
        registered skill carries, so that is what gets checked."""
        config = _config_with_deep_agent(
            skills=["research"],
            resource_skills={
                "research": SkillModel(name="research", path="skills/research")
            },
        )
        lines = self._reject(config)
        assert any("skills/research" in line for line in lines), lines

    def test_bare_string_naming_a_volume_registry_entry_is_accepted(self) -> None:
        config = _config_with_deep_agent(
            skills=["governed"],
            resource_skills={
                "governed": SkillModel(
                    name="governed",
                    path=VolumePathModel(
                        volume=VolumeModel(name="skills_library"), path="research"
                    ),
                )
            },
        )
        assert self._reject(config) == []

    def test_absolute_bare_string_skill_is_accepted(self) -> None:
        """Absolute paths are not the failure mode here — they name a specific
        location rather than resolving against whatever tree is nearby."""
        config = _config_with_deep_agent(skills=["/abs/skills/research"])
        assert self._reject(config) == []

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


def _config_with_instruction_files(
    *, instruction_files: list[str], skills: list | None = None
) -> AppConfig:
    deep = DeepAgentModel(skills=skills or [], instruction_files=instruction_files)
    return AppConfig(
        agents={"a": AgentModel(name="a", model=LLMModel(name="x"))},
        app=AppModel(
            name="test_app",
            agents=[AgentModel(name="a", model=LLMModel(name="x"))],
            orchestration=OrchestrationModel(deep_agent=deep),
        ),
    )


@pytest.fixture
def tmp_project_with_instructions(tmp_project: Path) -> Path:
    """Add both instruction-file shapes to the temp project.

    ``instructions/AGENTS.md`` is the standalone case (nothing else stages it),
    and ``skills/research/AGENTS.md`` is the common one — an ``AGENTS.md`` living
    inside the skill it documents, which the skill staging already carries.
    """
    instructions = tmp_project / "instructions"
    instructions.mkdir()
    (instructions / "AGENTS.md").write_text("# house style\nAlways cite sources.\n")
    (tmp_project / "skills" / "research" / "AGENTS.md").write_text("# research notes\n")
    (tmp_project / "AGENTS.md").write_text("# root instructions\n")
    return tmp_project


@pytest.mark.unit
class TestResolveInstructionFilePath:
    """The file-shaped sibling of ``resolve_skill_source_dir``."""

    def test_absolute_passes_through(self, tmp_project_with_instructions: Path) -> None:
        """``None`` means "use the source verbatim" — including ``/Volumes/...``."""
        assert resolve_instruction_file_path("/Volumes/c/s/v/AGENTS.md") is None

    def test_relative_resolves_against_anchor(
        self, tmp_project_with_instructions: Path
    ) -> None:
        found = resolve_instruction_file_path("instructions/AGENTS.md")
        assert found == tmp_project_with_instructions / "instructions" / "AGENTS.md"

    def test_missing_returns_none(self, tmp_project_with_instructions: Path) -> None:
        assert resolve_instruction_file_path("instructions/GHOST.md") is None

    def test_directory_is_not_a_match(
        self, tmp_project_with_instructions: Path
    ) -> None:
        """A directory would be accepted by an ``exists()`` test and then dropped
        by MemoryMiddleware without a word, so the predicate is ``is_file()``."""
        assert resolve_instruction_file_path("instructions") is None

    def test_config_anchor_wins_over_cwd(
        self, tmp_project_with_instructions: Path, tmp_path: Path
    ) -> None:
        """The config directory leads the anchor list: an unrelated CWD holding a
        same-named file must not shadow the one the author meant."""
        elsewhere = tmp_path / "elsewhere" / "instructions"
        elsewhere.mkdir(parents=True)
        (elsewhere / "AGENTS.md").write_text("# wrong file\n")

        found = resolve_instruction_file_path(
            "instructions/AGENTS.md", (tmp_project_with_instructions,)
        )
        assert found == tmp_project_with_instructions / "instructions" / "AGENTS.md"


@pytest.mark.unit
class TestResolveInstructionFileRuntimePaths:
    """What ``create_deep_agent(memory=...)`` actually receives."""

    def test_relative_becomes_absolute(
        self, tmp_project_with_instructions: Path
    ) -> None:
        cfg = _config_with_instruction_files(
            instruction_files=["instructions/AGENTS.md"]
        )
        resolved = resolve_instruction_file_runtime_paths(
            cfg.app.orchestration.deep_agent.instruction_files, cfg
        )
        assert resolved == [
            str(tmp_project_with_instructions / "instructions" / "AGENTS.md")
        ]

    def test_absolute_passes_through_unchanged(
        self, tmp_project_with_instructions: Path
    ) -> None:
        cfg = _config_with_instruction_files(
            instruction_files=["/Volumes/c/s/v/AGENTS.md"]
        )
        resolved = resolve_instruction_file_runtime_paths(
            cfg.app.orchestration.deep_agent.instruction_files, cfg
        )
        assert resolved == ["/Volumes/c/s/v/AGENTS.md"]

    def test_missing_is_dropped_not_passed_on(
        self, tmp_project_with_instructions: Path
    ) -> None:
        """Passing a bad path through would have deepagents skip it silently and
        render ``(No memory loaded)``; dropping it here at least logs the anchors."""
        cfg = _config_with_instruction_files(
            instruction_files=["instructions/GHOST.md"]
        )
        assert (
            resolve_instruction_file_runtime_paths(
                cfg.app.orchestration.deep_agent.instruction_files, cfg
            )
            == []
        )

    def test_duplicates_collapse(self, tmp_project_with_instructions: Path) -> None:
        cfg = _config_with_instruction_files(
            instruction_files=["instructions/AGENTS.md", "instructions/AGENTS.md"]
        )
        assert (
            len(
                resolve_instruction_file_runtime_paths(
                    cfg.app.orchestration.deep_agent.instruction_files, cfg
                )
            )
            == 1
        )


@pytest.mark.unit
class TestUnresolvableInstructionFiles:
    """Fail at deploy, where a human is watching."""

    def test_clean_config_reports_nothing(
        self, tmp_project_with_instructions: Path
    ) -> None:
        cfg = _config_with_instruction_files(
            instruction_files=["instructions/AGENTS.md"]
        )
        assert unresolvable_instruction_files(cfg) == []

    def test_missing_file_is_named(self, tmp_project_with_instructions: Path) -> None:
        cfg = _config_with_instruction_files(
            instruction_files=["instructions/GHOST.md"]
        )
        problems = unresolvable_instruction_files(cfg)
        assert len(problems) == 1
        assert "GHOST.md" in problems[0]

    @pytest.mark.parametrize(
        "entry",
        [
            "/Volumes/c/s/v/AGENTS.md",
            "/Workspace/Shared/AGENTS.md",
            "/dbfs/FileStore/AGENTS.md",
        ],
    )
    def test_fuse_paths_never_flagged(
        self, entry: str, tmp_project_with_instructions: Path
    ) -> None:
        """A FUSE mount is not checkable from the deploying machine.

        All three roots are mounted inside Databricks compute and absent from the
        laptop running the CLI, so a bundle-time existence check on one says
        nothing about runtime. Exempting only ``/Volumes`` failed the deploy for a
        ``/Workspace/Shared/AGENTS.md`` that would have loaded fine.
        """
        cfg = _config_with_instruction_files(instruction_files=[entry])
        assert unresolvable_instruction_files(cfg) == []

    @pytest.mark.parametrize(
        "entry", ["/Volumesnotreally/AGENTS.md", "/WorkspaceFoo/AGENTS.md"]
    )
    def test_near_miss_prefixes_are_still_checked(
        self, entry: str, tmp_project_with_instructions: Path
    ) -> None:
        """The exemption is per path segment, not a substring match."""
        cfg = _config_with_instruction_files(instruction_files=[entry])
        assert len(unresolvable_instruction_files(cfg)) == 1

    def test_missing_local_absolute_is_named(
        self, tmp_project_with_instructions: Path
    ) -> None:
        """Only ``/Volumes`` is exempt — every other absolute is still checked.

        Exempting all absolutes let the worst case through the gate untouched: a
        hand-written host path, staged nowhere, that renders ``(No memory loaded)``
        in the container. A typo in one now stops the deploy.
        """
        ghost: Path = tmp_project_with_instructions / "elsewhere" / "GHOST.md"
        cfg = _config_with_instruction_files(instruction_files=[str(ghost)])
        problems = unresolvable_instruction_files(cfg)
        assert len(problems) == 1
        assert "GHOST.md" in problems[0]

    def test_existing_local_absolute_passes(
        self, tmp_project_with_instructions: Path
    ) -> None:
        """The gate checks existence, not portability.

        An absolute path that exists is the author's call — it works for a local run
        and a notebook. It still cannot reproduce itself inside a serving container,
        which is what the staging collectors warn about separately.
        """
        real: Path = tmp_project_with_instructions / "instructions" / "AGENTS.md"
        cfg = _config_with_instruction_files(instruction_files=[str(real)])
        assert unresolvable_instruction_files(cfg) == []

    def test_deploy_gate_covers_instruction_files(
        self, tmp_project_with_instructions: Path
    ) -> None:
        """One gate for both asset families, so a new deploy path only has to
        remember one call."""
        cfg = _config_with_instruction_files(
            instruction_files=["instructions/GHOST.md"]
        )
        with pytest.raises(ValueError, match="Apps deploy") as err:
            assert_skill_assets_resolvable(cfg, target="Apps deploy")
        assert "GHOST.md" in str(err.value)


@pytest.mark.unit
class TestInstructionFileCodePaths:
    """mlflow flattens a file entry to ``code/<basename>``, so what gets shipped
    is the entry's top-level directory, not the file."""

    def test_nested_entry_ships_its_top_level_dir(
        self, tmp_project_with_instructions: Path
    ) -> None:
        cfg = _config_with_instruction_files(
            instruction_files=["instructions/AGENTS.md"]
        )
        assert collect_instruction_file_code_paths(cfg) == [
            str(tmp_project_with_instructions / "instructions")
        ]

    def test_bare_filename_ships_the_file(
        self, tmp_project_with_instructions: Path
    ) -> None:
        """No directory component, so flattening is a no-op and the file itself
        lands at ``code/AGENTS.md`` — exactly where the config's path looks."""
        cfg = _config_with_instruction_files(instruction_files=["AGENTS.md"])
        assert collect_instruction_file_code_paths(cfg) == [
            str(tmp_project_with_instructions / "AGENTS.md")
        ]

    def test_absolute_entry_ships_nothing(
        self, tmp_project_with_instructions: Path
    ) -> None:
        cfg = _config_with_instruction_files(
            instruction_files=["/Volumes/c/s/v/AGENTS.md"]
        )
        assert collect_instruction_file_code_paths(cfg) == []

    def test_entry_inside_skills_dedupes_against_the_skills_root(
        self, tmp_project_with_instructions: Path
    ) -> None:
        """An ``AGENTS.md`` inside a skill ships as part of ``skills/`` — which
        ``collect_skills_code_paths`` already returns, so the deploy site's
        dedup collapses the two into one entry."""
        cfg = _config_with_instruction_files(
            instruction_files=["skills/research/AGENTS.md"],
            skills=[SkillModel(name="r", path="skills/research")],
        )
        instr = collect_instruction_file_code_paths(cfg)
        assert instr == [str(tmp_project_with_instructions / "skills")]
        assert instr == collect_skills_code_paths(cfg)


@pytest.mark.unit
class TestInstructionFileStagings:
    """The plan every bundler and the direct Apps upload share."""

    def test_dest_preserves_the_declared_layout(
        self, tmp_project_with_instructions: Path
    ) -> None:
        cfg = _config_with_instruction_files(
            instruction_files=["instructions/AGENTS.md"]
        )
        stagings = iter_instruction_file_stagings(cfg)
        assert stagings == [
            (
                tmp_project_with_instructions / "instructions" / "AGENTS.md",
                "instructions/AGENTS.md",
            )
        ]

    def test_entry_already_inside_a_staged_skill_is_dropped(
        self, tmp_project_with_instructions: Path
    ) -> None:
        """The common case: the skill staging copied the directory wholesale, so
        re-planning the file would make the bundle report claim it skipped a file
        it had just written."""
        cfg = _config_with_instruction_files(
            instruction_files=["skills/research/AGENTS.md"],
            skills=[SkillModel(name="r", path="skills/research")],
        )
        assert iter_instruction_file_stagings(cfg) == []

    def test_missing_entry_is_not_planned(
        self, tmp_project_with_instructions: Path
    ) -> None:
        cfg = _config_with_instruction_files(
            instruction_files=["instructions/GHOST.md"]
        )
        assert iter_instruction_file_stagings(cfg) == []

    def test_absolute_entry_is_not_planned(
        self, tmp_project_with_instructions: Path
    ) -> None:
        cfg = _config_with_instruction_files(
            instruction_files=["/Volumes/c/s/v/AGENTS.md"]
        )
        assert iter_instruction_file_stagings(cfg) == []


@pytest.mark.unit
class TestRemoteConfigRejectsRelativeInstructionFiles:
    """An instruction file is spliced into the system prompt verbatim, so a remote
    config naming a relative one is the untrusted-document-reads-local-Markdown
    problem with none of the indirection."""

    def _reject(self, config: AppConfig) -> list[str]:
        from dao_ai.config import _reject_relative_assets_for_remote_config

        try:
            _reject_relative_assets_for_remote_config(
                config, source="https://example.com/agent.yaml"
            )
        except ValueError as err:
            return str(err).splitlines()
        return []

    def test_relative_instruction_file_is_rejected(self, tmp_project: Path) -> None:
        cfg = _config_with_instruction_files(
            instruction_files=["instructions/AGENTS.md"]
        )
        lines = self._reject(cfg)
        assert any("instruction_files" in line for line in lines), lines

    def test_volume_instruction_file_is_allowed(self, tmp_project: Path) -> None:
        cfg = _config_with_instruction_files(
            instruction_files=["/Volumes/c/s/v/AGENTS.md"]
        )
        assert self._reject(cfg) == []
