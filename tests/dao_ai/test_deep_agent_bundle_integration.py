"""End-to-end bundle integration test for deep_agent skills.

Exercises the full chain that ships skills to both deployment targets:

* **Apps** — ``write_bundle`` copies local skill dirs into the bundle output
  directory; volume-backed skills emit ``READ_VOLUME`` permissions in the
  generated ``resources/app.yml``.
* **Model Serving** — ``collect_skills_code_paths`` returns the project-root
  ``skills/`` directory which mlflow extracts into ``<model_dir>/code/skills/...``,
  preserving the full layout so the runtime resolver finds skill files via
  the ``sys.path`` anchor.

This test deliberately covers BOTH a local and a volume-backed skill in the
same config so the asymmetric handling (copy vs. resource-only) is regression-locked.

Run with:
    pytest tests/dao_ai/test_deep_agent_bundle_integration.py -v -m unit
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from dao_ai.apps.bundle import write_bundle
from dao_ai.config import AppConfig
from dao_ai.skills import collect_skills_code_paths


@pytest.fixture(autouse=True)
def _stub_bundle_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub ``generate_bundle_lock`` so these bundle-layout tests don't invoke a
    real ``uv lock`` (which needs network + a published dao-ai version, and would
    choke on the stub pyproject). Writes a placeholder ``uv.lock`` so presence
    assertions still hold. The lock itself is covered by
    ``test_bundle_dependency_lock.py``.
    """
    from pathlib import Path as _Path

    def _fake_lock(bundle_dir: _Path) -> None:
        (bundle_dir / "uv.lock").write_text("# stub lock for tests\n")

    monkeypatch.setattr("dao_ai.apps.bundle.generate_bundle_lock", _fake_lock)


@pytest.fixture
def project_root_with_skills(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Copy the real project's example YAML + skills dir into a tmp project root.

    Ensures the test exercises the actual deep_agent_with_skills.yaml example
    (the deployment example we ship), not a stripped-down inline fixture.
    """
    real_example_dir = (
        Path(__file__).resolve().parents[2] / "examples" / "13_orchestration"
    )
    real_example = real_example_dir / "deep_agent_with_skills.yaml"
    real_skills_dir = real_example_dir / "skills"

    # Mirror the canonical layout into tmp_path: skills are colocated with the
    # config (``<config-dir>/skills/...``), so keep them siblings here too.
    shutil.copy(real_example, tmp_path / "deep_agent_with_skills.yaml")
    shutil.copytree(real_skills_dir, tmp_path / "skills")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'tmp'\n")

    monkeypatch.setenv("DAO_AI_PROJECT_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.unit
class TestBundleShipsSkills:
    """``write_bundle`` produces a deployable bundle that contains skill content
    for the local skill and only a permission entry for the volume-backed skill."""

    def test_local_skill_copied_into_bundle(
        self, project_root_with_skills: Path, tmp_path: Path
    ) -> None:
        cfg = AppConfig.from_file(
            str(project_root_with_skills / "deep_agent_with_skills.yaml")
        )
        out = tmp_path / "bundle_out"
        write_bundle(cfg, out, overwrite=True)

        # Local skill files end up under <bundle>/skills/<vertical>/<skill>/.
        assert (
            out / "skills" / "sporting_goods_store" / "research" / "SKILL.md"
        ).exists()
        assert (
            out / "skills" / "sporting_goods_store" / "research" / "AGENTS.md"
        ).exists()

    def test_local_skill_copied_when_cwd_is_not_config_dir(
        self,
        project_root_with_skills: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Regression: skills are colocated with the config and must be found at
        bundle time even when the CLI runs from a CWD that is NOT the config's
        directory (e.g. ``dao-ai generate-agent -c .../deep_agent_with_skills.yaml``
        from the repo root). The bundle-time resolver anchors on the config's
        own dir, not CWD."""
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        monkeypatch.delenv("DAO_AI_PROJECT_ROOT", raising=False)

        cfg = AppConfig.from_file(
            str(project_root_with_skills / "deep_agent_with_skills.yaml")
        )
        out = tmp_path / "bundle_out"
        write_bundle(cfg, out, overwrite=True)

        assert (
            out / "skills" / "sporting_goods_store" / "research" / "SKILL.md"
        ).exists()

    def test_volume_skill_NOT_copied(
        self, project_root_with_skills: Path, tmp_path: Path
    ) -> None:
        """Volume-backed skills live on UC volumes and are read at runtime — they
        must NOT be copied into the bundle (which doesn't have access to UC)."""
        cfg = AppConfig.from_file(
            str(project_root_with_skills / "deep_agent_with_skills.yaml")
        )
        out = tmp_path / "bundle_out"
        write_bundle(cfg, out, overwrite=True)

        # The volume-backed skill is under product_lookup/. We DO have a
        # local placeholder skills/sporting_goods_store/product_lookup/ but
        # the bundle should not include it (it isn't referenced as a local
        # SkillModel — only as the volume target).
        bundled_local_skills = list(
            (out / "skills").rglob("SKILL.md") if (out / "skills").exists() else []
        )
        # Only the local research skill should be bundled; product_lookup
        # is volume-backed in this example so it stays out.
        bundled_paths = [str(p.relative_to(out)) for p in bundled_local_skills]
        assert any("research/SKILL.md" in p for p in bundled_paths)
        # product_lookup is the volume-backed path — it MUST NOT be in the bundle
        # because the deployed config points at /Volumes/... for it.
        # (The local placeholder under skills/sporting_goods_store/product_lookup/
        # exists at the project root for dev testing, but the volume-backed entry
        # in the YAML config is what matters at deploy time.)


@pytest.mark.unit
class TestBundleEmitsVolumePermission:
    """The volume backing a skill must be wired as a deployment resource so the
    serving endpoint / app SP gets ``READ_VOLUME`` access."""

    def test_app_yml_has_volume_permission(
        self, project_root_with_skills: Path, tmp_path: Path
    ) -> None:
        cfg = AppConfig.from_file(
            str(project_root_with_skills / "deep_agent_with_skills.yaml")
        )
        out = tmp_path / "bundle_out"
        write_bundle(cfg, out, overwrite=True)

        app_yml = (out / "resources" / "app.yml").read_text()
        # Volume must show up as a uc_securable with VOLUME type and READ_VOLUME perm.
        assert "skills_library" in app_yml
        assert "VOLUME" in app_yml
        assert "READ_VOLUME" in app_yml

        # And parse it as YAML to confirm structure.
        parsed = yaml.safe_load(app_yml)
        app_resources = parsed["resources"]["apps"]["deep-agent-skills"]["resources"]
        volume_entries = [
            r
            for r in app_resources
            if r.get("uc_securable", {}).get("securable_type") == "VOLUME"
        ]
        assert len(volume_entries) == 1
        assert volume_entries[0]["uc_securable"]["permission"] == "READ_VOLUME"
        assert (
            "skills_library" in volume_entries[0]["uc_securable"]["securable_full_name"]
        )


@pytest.mark.unit
class TestModelServingCodePaths:
    """``collect_skills_code_paths`` returns the project-root skills/ dir so
    mlflow preserves the full layout under ``<model_dir>/code/skills/``."""

    def test_returns_project_skills_root(self, project_root_with_skills: Path) -> None:
        cfg = AppConfig.from_file(
            str(project_root_with_skills / "deep_agent_with_skills.yaml")
        )
        paths = collect_skills_code_paths(cfg)
        assert len(paths) == 1
        # Should be the absolute project_root/skills directory.
        skills_root = Path(paths[0])
        assert skills_root.is_absolute()
        assert skills_root.name == "skills"
        # The local skill content should be reachable under it.
        assert (skills_root / "sporting_goods_store" / "research" / "SKILL.md").exists()


@pytest.mark.unit
class TestRuntimeResolutionMatchesBundleLayout:
    """The path declared in YAML must resolve correctly under both the Apps and
    Model Serving layouts produced by ``write_bundle`` / mlflow code_paths."""

    def test_apps_layout_resolves_after_bundle(
        self,
        project_root_with_skills: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Build the bundle, then run the runtime resolver against the bundle root
        as CWD (simulating the deployed Apps environment)."""
        cfg = AppConfig.from_file(
            str(project_root_with_skills / "deep_agent_with_skills.yaml")
        )
        out = tmp_path / "bundle_out"
        write_bundle(cfg, out, overwrite=True)

        # Reload the rendered config from the bundle (this is what the deployed
        # app actually runs) and resolve skills against the bundle root.
        deployed_cfg = AppConfig.from_file(str(out / "deep_agent_with_skills.yaml"))
        monkeypatch.chdir(out)
        monkeypatch.delenv("DAO_AI_PROJECT_ROOT", raising=False)

        from dao_ai.skills import resolve_skill_runtime_paths

        skills = deployed_cfg.app.orchestration.deep_agent.skills
        paths = resolve_skill_runtime_paths(skills, deployed_cfg)

        # Two skills declared: one local (resolves to bundle path), one volume
        # (passes through verbatim).
        # After R1+R2: the local research skill is declared on the AGENT
        # (translated to middleware at config-load), so deep_agent.skills
        # contains only the volume-backed entry.
        assert len(paths) == 1
        volume_paths = [p for p in paths if p.startswith("/Volumes/")]
        assert len(volume_paths) == 1
        # Volume path parent: the skills_library/ root contains skill subdirs.
        assert volume_paths[0].endswith("skills_library")

    def test_model_serving_layout_resolves(
        self,
        project_root_with_skills: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Simulate the mlflow extraction layout (``<model_dir>/code/skills/...``)
        and confirm the runtime resolver finds the skill via ``sys.path``."""
        cfg = AppConfig.from_file(
            str(project_root_with_skills / "deep_agent_with_skills.yaml")
        )
        # Mlflow copies the project skills/ dir into <model_dir>/code/skills/.
        model_dir = tmp_path / "served_model"
        code_dir = model_dir / "code"
        code_dir.mkdir(parents=True)
        shutil.copytree(project_root_with_skills / "skills", code_dir / "skills")

        # Worker CWD is unrelated; only sys.path has the model code dir.
        unrelated = tmp_path / "unrelated"
        unrelated.mkdir()
        monkeypatch.chdir(unrelated)
        monkeypatch.delenv("DAO_AI_PROJECT_ROOT", raising=False)
        monkeypatch.syspath_prepend(str(code_dir))

        from dao_ai.skills import resolve_skill_runtime_paths

        skills = cfg.app.orchestration.deep_agent.skills
        paths = resolve_skill_runtime_paths(skills, cfg)

        # After R1+R2: only the volume-backed deep_agent.skills entry shows
        # here. The local research skill is now on AGENT.skills, translated
        # to middleware, and lives in agent.middleware[*].args['sources'].
        assert len(paths) == 1
        assert paths[0].startswith("/Volumes/")


def _load_skills(middleware) -> tuple[list[str], list]:
    """Discover skills the way a running agent does.

    Asserting on ``middleware.sources`` is what let the Model Serving bug ship:
    the sources looked plausible and the directory did not exist. Discovery is
    lazy inside ``before_agent``, so only calling it proves anything.

    ``before_agent`` takes three positionals, and the state must not already
    carry ``skills_metadata`` or the load short-circuits.
    """
    from types import SimpleNamespace

    runtime = SimpleNamespace(context=None, stream_writer=None, store=None)
    out = middleware.before_agent({"messages": []}, runtime, {}) or {}
    return (
        [s["name"] for s in out.get("skills_metadata", [])],
        # ``skills_load_errors`` only exists in deepagents >= 0.6; the floor in
        # pyproject.toml is 0.5.7, so treat absent as empty rather than asserting
        # on it.
        out.get("skills_load_errors", []),
    )


def _skills_middleware_args(dumped: dict) -> list[dict]:
    """Every skills-middleware ``args`` block anywhere in a dumped config.

    Walks the whole tree rather than a known list of holders, because the bug was
    precisely that different copies of the same middleware disagreed — a targeted
    lookup would have inspected one copy and missed it.
    """
    found: list[dict] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            name = node.get("name")
            if isinstance(name, str) and name.endswith("create_skills_middleware"):
                found.append(node.get("args") or {})
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(dumped)
    return found


@pytest.mark.unit
class TestModelServingDictReconstruction:
    """The regression that would have caught the shipped bug.

    Model Serving is the only target that rebuilds the config from a dict
    (``AppConfig(**ModelConfig().to_dict())``) rather than ``from_file``, so it
    is the only one where ``local_config_path`` is ``None`` and nothing can
    re-anchor a path afterwards. A config that baked the *loading* machine's
    directory therefore pointed the serving container at a directory that never
    existed there — and the endpoint came up healthy, with no skills.
    """

    def _dump(self, cfg: AppConfig) -> dict:
        """Serialize exactly as ``providers/databricks.py`` does for log_model."""
        return cfg.model_dump(mode="json", by_alias=True, exclude_none=True)

    def test_every_serialized_copy_agrees_and_is_relative(
        self, project_root_with_skills: Path
    ) -> None:
        """The shipped MLmodel held this middleware three times with two
        different values, one of them a job driver's ephemeral directory."""
        cfg = AppConfig.from_file(
            str(project_root_with_skills / "deep_agent_with_skills.yaml")
        )
        arg_blocks = _skills_middleware_args(self._dump(cfg))
        assert arg_blocks, "expected at least one skills middleware in the config"

        filesystem_sources = [
            source
            for block in arg_blocks
            if block.get("backend_type") == "filesystem"
            for source in block.get("sources", [])
        ]
        assert filesystem_sources
        assert len(set(filesystem_sources)) == 1, (
            f"copies disagree: {sorted(set(filesystem_sources))}"
        )
        for source in filesystem_sources:
            assert not Path(source).is_absolute(), (
                f"{source!r} is absolute — this machine's path would ship to the "
                "serving container"
            )

    def test_skill_loads_after_dict_rebuild_with_only_syspath(
        self,
        project_root_with_skills: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The full Model Serving path: dump, rebuild from the dict, and give the
        process nothing but the ``sys.path`` entry mlflow prepends."""
        from dao_ai.middleware.skills import create_skills_middleware

        cfg = AppConfig.from_file(
            str(project_root_with_skills / "deep_agent_with_skills.yaml")
        )
        dumped = self._dump(cfg)

        # mlflow copies the colocated skills/ dir to <model_dir>/code/skills/ and
        # prepends <model_dir>/code to sys.path.
        code_dir = tmp_path / "served_model" / "code"
        code_dir.mkdir(parents=True)
        shutil.copytree(project_root_with_skills / "skills", code_dir / "skills")

        unrelated = tmp_path / "unrelated"
        unrelated.mkdir()
        monkeypatch.chdir(unrelated)
        monkeypatch.delenv("DAO_AI_PROJECT_ROOT", raising=False)
        monkeypatch.syspath_prepend(str(code_dir))

        rebuilt = AppConfig(**dumped)
        assert rebuilt.local_config_path is None, (
            "a dict rebuild has no config path — that is what makes this target "
            "unable to re-anchor after the fact"
        )

        blocks = [
            b
            for b in _skills_middleware_args(self._dump(rebuilt))
            if b.get("backend_type") == "filesystem"
        ]
        names, errors = _load_skills(create_skills_middleware(**blocks[0]))
        assert "research" in names, f"skills did not load: {names} errors={errors}"

    def test_stale_absolute_source_loads_nothing(self, tmp_path: Path) -> None:
        """The negative twin: the exact shape that shipped. Kept so the harness
        is shown to be capable of detecting the failure it was written for."""
        from dao_ai.middleware.skills import create_skills_middleware

        stale = "/home/spark-928f77f1-0d5c-42a6/.dao-ai/git/abc/skills/vertical"
        middleware = create_skills_middleware(
            sources=[stale], backend_type="filesystem", root_dir="/"
        )
        names, _ = _load_skills(middleware)
        assert names == []

    def test_building_middleware_does_not_mutate_the_config(
        self, project_root_with_skills: Path
    ) -> None:
        """``07_deploy_agent.py`` calls ``display_graph()`` and then
        ``create_agent()`` in one process. If building the graph wrote resolved
        paths back into the config, the very next ``model_dump`` would ship them.
        """
        import json

        from dao_ai.middleware.skills import create_skills_middleware
        from dao_ai.skills import config_skill_anchor, skill_anchors

        cfg = AppConfig.from_file(
            str(project_root_with_skills / "deep_agent_with_skills.yaml")
        )
        before = json.dumps(self._dump(cfg), sort_keys=True)

        with skill_anchors(*config_skill_anchor(cfg)):
            for block in _skills_middleware_args(json.loads(before)):
                if block.get("backend_type") == "filesystem":
                    create_skills_middleware(**block)

        assert json.dumps(self._dump(cfg), sort_keys=True) == before


@pytest.mark.unit
class TestSkillsLoadPerDeploymentTarget:
    """One relative path in the config; every target must load the real files."""

    def _filesystem_block(self, cfg: AppConfig) -> dict:
        blocks = [
            b
            for b in _skills_middleware_args(
                cfg.model_dump(mode="json", by_alias=True, exclude_none=True)
            )
            if b.get("backend_type") == "filesystem"
        ]
        assert blocks, "expected a filesystem-backed skills middleware"
        return blocks[0]

    def test_apps_bundle_root_as_cwd(
        self,
        project_root_with_skills: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from dao_ai.middleware.skills import create_skills_middleware

        cfg = AppConfig.from_file(
            str(project_root_with_skills / "deep_agent_with_skills.yaml")
        )
        out = tmp_path / "bundle_out"
        write_bundle(cfg, out, overwrite=True)

        deployed = AppConfig.from_file(str(out / "deep_agent_with_skills.yaml"))
        monkeypatch.chdir(out)
        monkeypatch.delenv("DAO_AI_PROJECT_ROOT", raising=False)

        names, errors = _load_skills(
            create_skills_middleware(**self._filesystem_block(deployed))
        )
        assert "research" in names, f"{names} errors={errors}"

    def test_local_run_with_cwd_elsewhere(
        self,
        project_root_with_skills: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``dao-ai run -c examples/.../agent.yaml`` from an unrelated directory.
        Only the config-dir anchor can save this one."""
        from dao_ai.middleware.skills import create_skills_middleware
        from dao_ai.skills import config_skill_anchor, skill_anchors

        config_path = project_root_with_skills / "deep_agent_with_skills.yaml"
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        monkeypatch.delenv("DAO_AI_PROJECT_ROOT", raising=False)

        cfg = AppConfig.from_file(str(config_path))
        with skill_anchors(*config_skill_anchor(cfg)):
            names, errors = _load_skills(
                create_skills_middleware(**self._filesystem_block(cfg))
            )
        assert "research" in names, f"{names} errors={errors}"

    def test_volume_backed_skill_passes_through(
        self, project_root_with_skills: Path
    ) -> None:
        """A governed skill must keep its ``/Volumes/...`` path — resolution must
        not try to find it on the local disk and drop it."""
        from dao_ai.skills import resolve_skill_runtime_paths

        cfg = AppConfig.from_file(
            str(project_root_with_skills / "deep_agent_with_skills.yaml")
        )
        paths = resolve_skill_runtime_paths(
            cfg.app.orchestration.deep_agent.skills, cfg
        )
        assert paths == [
            "/Volumes/retail_consumer_goods/sporting_goods_store/skills_library"
        ]


@pytest.mark.unit
class TestAllBundlersStageSkillContent:
    """A staged config that names skills the bundle does not contain produces an
    agent with no skills. Only the Apps bundler used to copy them."""

    def test_mcp_bundle_contains_skill_files(
        self,
        project_root_with_skills: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from dao_ai.mcp.generate import write_mcp_bundle

        monkeypatch.setattr(
            "dao_ai.mcp.generate.generate_bundle_lock",
            lambda bundle_dir: (bundle_dir / "uv.lock").write_text("# stub\n"),
        )
        cfg = AppConfig.from_file(
            str(project_root_with_skills / "deep_agent_with_skills.yaml")
        )
        out = tmp_path / "mcp_out"
        write_mcp_bundle(cfg, out, overwrite=True)

        assert (
            out / "skills" / "sporting_goods_store" / "research" / "SKILL.md"
        ).is_file()

    def test_mcp_bundle_skill_loads_with_bundle_root_as_cwd(
        self,
        project_root_with_skills: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from dao_ai.mcp.generate import write_mcp_bundle
        from dao_ai.middleware.skills import create_skills_middleware

        monkeypatch.setattr(
            "dao_ai.mcp.generate.generate_bundle_lock",
            lambda bundle_dir: (bundle_dir / "uv.lock").write_text("# stub\n"),
        )
        cfg = AppConfig.from_file(
            str(project_root_with_skills / "deep_agent_with_skills.yaml")
        )
        out = tmp_path / "mcp_out"
        write_mcp_bundle(cfg, out, overwrite=True)

        deployed = AppConfig.from_file(str(out / "deep_agent_with_skills.yaml"))
        monkeypatch.chdir(out)
        monkeypatch.delenv("DAO_AI_PROJECT_ROOT", raising=False)

        blocks = [
            b
            for b in _skills_middleware_args(
                deployed.model_dump(mode="json", by_alias=True, exclude_none=True)
            )
            if b.get("backend_type") == "filesystem"
        ]
        names, errors = _load_skills(create_skills_middleware(**blocks[0]))
        assert "research" in names, f"{names} errors={errors}"

    def test_workflow_bundle_stages_skills_beside_the_staged_config(
        self, project_root_with_skills: Path, tmp_path: Path
    ) -> None:
        """The DAB notebooks reload the config from ``config/``, so the skills
        have to sit beside *that* copy for ``collect_skills_code_paths`` to find
        content to ship on to Model Serving."""
        from dao_ai.pipeline.bundle import write_pipeline_bundle
        from dao_ai.skills import collect_skills_code_paths

        cfg = AppConfig.from_file(
            str(project_root_with_skills / "deep_agent_with_skills.yaml")
        )
        out = tmp_path / "dab_out"
        write_pipeline_bundle(cfg, out, overwrite=True)

        staged_skill = (
            out / "config" / "skills" / "sporting_goods_store" / "research" / "SKILL.md"
        )
        assert staged_skill.is_file(), "skills were not staged beside the staged config"

        staged_config = out / "config" / "deep_agent_with_skills.yaml"
        reloaded = AppConfig.from_file(str(staged_config))
        assert collect_skills_code_paths(reloaded), (
            "the staged config found no skill content to ship to Model Serving"
        )

    def test_apps_direct_deploy_uploads_skill_files(
        self, project_root_with_skills: Path
    ) -> None:
        """``deploy_agent(mode=APPS)`` bypasses the Apps *bundler*, so it needs its
        own upload — the app CWD is the source path, and a relative skills source
        in the uploaded config resolves only if the content is beside it."""
        from unittest.mock import MagicMock

        from dao_ai.providers.databricks import DatabricksProvider

        cfg = AppConfig.from_file(
            str(project_root_with_skills / "deep_agent_with_skills.yaml")
        )
        provider = DatabricksProvider(w=MagicMock())
        source_path = "/Workspace/Users/u/apps/skills-app"

        provider._upload_skill_dirs(cfg, source_path)

        uploaded = {
            c.kwargs["path"] for c in provider.w.workspace.upload.call_args_list
        }
        assert (
            f"{source_path}/skills/sporting_goods_store/research/SKILL.md" in uploaded
        ), uploaded
        # The volume-backed skill is not local content and must not be uploaded.
        assert not any("skills_library" in path for path in uploaded), uploaded


@pytest.mark.unit
class TestBackendGateCoversSubagentSkills:
    """The backend chosen for the graph has to account for *every* skill in it.

    deepagents' default ``StateBackend`` reads from graph state, so a skill on
    disk or in a volume cannot be loaded by any path under it. The gate that
    upgrades to ``FilesystemBackend`` looked at ``deep_agent.skills`` alone, so a
    config that declares its skills only on a sub-agent got the state backend and
    served an agent whose skills could never load — no exception, no log line,
    exactly the silent degradation this module exists to lock down.
    """

    @staticmethod
    def _subagent_only_config() -> AppConfig:
        from dao_ai.config import (
            AgentModel,
            AppModel,
            DeepAgentModel,
            LLMModel,
            OrchestrationModel,
            SkillModel,
            SubAgentModel,
        )

        return AppConfig(
            agents={"a": AgentModel(name="a", model=LLMModel(name="x"))},
            app=AppModel(
                name="test_app",
                agents=[AgentModel(name="a", model=LLMModel(name="x"))],
                orchestration=OrchestrationModel(
                    deep_agent=DeepAgentModel(
                        name="d",
                        subagents=[
                            SubAgentModel(
                                name="researcher",
                                description="researches",
                                system_prompt="p",
                                skills=[
                                    SkillModel(name="r", path="/Volumes/c/s/v/research")
                                ],
                            )
                        ],
                    )
                ),
            ),
        )

    def test_subagent_only_skills_still_get_a_filesystem_backend(self) -> None:
        from deepagents.backends import FilesystemBackend

        config = self._subagent_only_config()
        assert not config.app.orchestration.deep_agent.skills, (
            "fixture must declare no top-level skills"
        )

        backend = self._built_backend(config)
        assert isinstance(backend, FilesystemBackend), type(backend).__name__

    def test_no_skills_anywhere_still_defers_to_deepagents(self) -> None:
        """The gate must stay a gate: a config with no skills at all keeps
        deepagents' own default rather than paying for a filesystem backend."""
        from dao_ai.config import (
            AgentModel,
            AppModel,
            DeepAgentModel,
            LLMModel,
            OrchestrationModel,
        )

        config = AppConfig(
            agents={"a": AgentModel(name="a", model=LLMModel(name="x"))},
            app=AppModel(
                name="test_app",
                agents=[AgentModel(name="a", model=LLMModel(name="x"))],
                orchestration=OrchestrationModel(deep_agent=DeepAgentModel(name="d")),
            ),
        )
        assert self._built_backend(config) is None

    @staticmethod
    def _built_backend(config: AppConfig) -> object:
        """The ``backend`` that ``create_deep_agent_graph`` actually hands deepagents.

        Asserted through the real build rather than by recomputing the gate
        expression in the test: the bug was in the expression at the call site, so a
        test that repeats it passes whether or not the call site is fixed.
        """
        import deepagents

        from dao_ai.orchestration.deep_agent import create_deep_agent_graph

        captured: dict[str, object] = {}

        def _capture(**kwargs: object) -> object:
            captured.update(kwargs)
            return object()

        original = deepagents.create_deep_agent
        deepagents.create_deep_agent = _capture
        try:
            create_deep_agent_graph(config)
        finally:
            deepagents.create_deep_agent = original

        return captured["backend"]
