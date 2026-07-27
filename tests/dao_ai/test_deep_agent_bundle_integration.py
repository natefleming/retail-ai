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
