"""Tests for the wheel-only pipeline staging bundle (dao_ai.pipeline.bundle).

The pipeline subcommand stages a self-contained Lakeflow-job bundle from the
installed dao-ai wheel's packaged assets — no source checkout required. These
tests cover:

- the packaged assets are reachable via importlib.resources,
- write_pipeline_bundle materializes databricks.yaml, requirements.txt, the 8
  step notebooks, and the resolved config into the staging dir,
- _referenced_asset_paths picks up relative ddl/data paths and ignores
  Volume-backed / absolute ones,
- config-referenced data/functions files are copied into the bundle,
  preserving the relative layout notebooks resolve against.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import pytest

from dao_ai.config import (
    AppConfig,
    DatasetModel,
    UnityCatalogFunctionSqlModel,
    VolumeModel,
)
from dao_ai.pipeline.bundle import (
    _materialize_notebooks,
    _packaged_text,
    _referenced_asset_paths,
    generate_pipeline_databricks_yaml,
    write_pipeline_bundle,
)

_MINIMAL_CONFIG = """\
resources:
  models:
    default_llm: &default_llm
      name: databricks-gpt-5-4-mini
agents:
  greeter: &greeter
    name: greeter
    description: A friendly assistant.
    model: *default_llm
    prompt: You are a concise assistant.
app:
  name: pipeline_test_app
  deployment_target: apps
  agents:
    - *greeter
"""


# ---------------------------------------------------------------------------
# Packaged assets ship in the wheel
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPackagedAssets:
    def test_all_eight_step_notebooks_are_packaged(self) -> None:
        names = sorted(
            p.name
            for p in files("dao_ai.pipeline.notebooks").iterdir()
            if p.name.endswith(".py") and p.name != "__init__.py"
        )
        assert len(names) == 8, f"expected 8 step notebooks, got {names}"
        assert names[0].startswith("01_")
        assert names[-1].startswith("08_")

    def test_requirements_are_packaged_without_dao_ai_pin(self) -> None:
        requirements = _packaged_text("dao_ai.pipeline", "requirements.txt")
        # The pinned lock intentionally does NOT pin dao-ai (installed by the
        # notebooks from the bundled wheel or PyPI).
        assert "dao-ai" not in requirements.splitlines()

    def test_notebooks_have_no_source_path_fallback(self) -> None:
        """The wheel-only refactor drops the `../src` sys.path fallback."""
        for p in files("dao_ai.pipeline.notebooks").iterdir():
            if not p.name.endswith(".py") or p.name == "__init__.py":
                continue
            text = p.read_text(encoding="utf-8")
            assert 'sys.path.insert(0, "../src")' not in text, (
                f"{p.name} still has the ../src fallback"
            )


# ---------------------------------------------------------------------------
# generate_pipeline_databricks_yaml — programmatic DAB (dict -> YAML)
# ---------------------------------------------------------------------------


class _AppStub:
    """Minimal stand-in — the generator only reads ``config.app.name``."""

    def __init__(self, name: str) -> None:
        self.name = name


@pytest.mark.unit
class TestGeneratePipelineDatabricksYaml:
    @staticmethod
    def _config() -> AppConfig:
        return AppConfig.model_construct(app=_AppStub("pipeline_test_app"))

    def _doc(self, development: bool) -> dict:
        import yaml

        return yaml.safe_load(
            generate_pipeline_databricks_yaml(self._config(), development=development)
        )

    def test_bundle_name_and_targets_use_normalized_app_name(self) -> None:
        doc = self._doc(development=False)
        assert doc["bundle"]["name"] == "pipeline_test_app"
        assert set(doc["targets"]) == {
            "pipeline_test_app-azure",
            "pipeline_test_app-aws",
            "pipeline_test_app-gcp",
        }

    def test_eight_task_dag_with_dependencies(self) -> None:
        tasks = self._doc(development=False)["resources"]["jobs"]["deploy_job"]["tasks"]
        assert len(tasks) == 8
        by_key = {t["task_key"]: t for t in tasks}
        # run-evaluation fans in from deploy-agents + generate-evaluation-data.
        deps = {d["task_key"] for d in by_key["run-evaluation"]["depends_on"]}
        assert deps == {"deploy-agents", "generate-evaluation-data"}
        # deploy-agents forwards the deployment_target + development vars.
        params = by_key["deploy-agents"]["notebook_task"]["base_parameters"]
        assert params["deployment-target"] == "${var.deployment_target}"
        assert params["development"] == "${var.development}"

    def test_development_includes_wheel_in_sync(self) -> None:
        assert "dist/*.whl" in self._doc(development=True)["sync"]["include"]

    def test_published_omits_wheel_from_sync(self) -> None:
        assert "dist/*.whl" not in self._doc(development=False)["sync"]["include"]

    def test_no_artifacts_block(self) -> None:
        # The dao-ai wheel is never built at bundle-deploy time.
        assert "artifacts" not in self._doc(development=True)


# ---------------------------------------------------------------------------
# _referenced_asset_paths — which config values need staging
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestReferencedAssetPaths:
    """Exercises the pure path-collection helper.

    ``_referenced_asset_paths`` only reads ``config.datasets`` and
    ``config.unity_catalog_functions``, so these build a minimal config via
    ``model_construct`` (bypassing full AppConfig validation) with just those
    two attributes set.
    """

    @staticmethod
    def _config(**kwargs: object) -> AppConfig:
        kwargs.setdefault("datasets", [])
        kwargs.setdefault("unity_catalog_functions", [])
        return AppConfig.model_construct(**kwargs)

    def test_relative_ddl_and_data_collected(self) -> None:
        config = self._config(
            datasets=[
                DatasetModel(
                    ddl="../functions/x/table.sql",
                    data="../data/x/seed.parquet",
                    format="parquet",
                )
            ],
        )
        paths = _referenced_asset_paths(config)
        assert "../functions/x/table.sql" in paths
        assert "../data/x/seed.parquet" in paths

    def test_volume_backed_ddl_ignored(self) -> None:
        """Volume references live on UC volumes and need no staging."""
        config = self._config(
            datasets=[
                DatasetModel(ddl=VolumeModel(name="c.s.v"), data=None)
            ],
        )
        assert _referenced_asset_paths(config) == []

    def test_absolute_paths_ignored(self) -> None:
        config = self._config(
            unity_catalog_functions=[
                UnityCatalogFunctionSqlModel(
                    function={"name": "c.s.f"},
                    ddl="/abs/path/fn.sql",
                )
            ],
        )
        assert _referenced_asset_paths(config) == []

    def test_dedupes_repeated_paths(self) -> None:
        config = self._config(
            datasets=[
                DatasetModel(ddl="../functions/shared.sql", data=None),
                DatasetModel(ddl="../functions/shared.sql", data=None),
            ],
        )
        assert _referenced_asset_paths(config) == ["../functions/shared.sql"]


# ---------------------------------------------------------------------------
# write_pipeline_bundle — end-to-end staging
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestWritePipelineBundle:
    def _load(self, tmp_path: Path, body: str) -> AppConfig:
        cfg_path = tmp_path / "my_config.yaml"
        cfg_path.write_text(body)
        return AppConfig.from_file(cfg_path, initialize=False)

    def test_stages_core_assets(self, tmp_path: Path) -> None:
        config = self._load(tmp_path, _MINIMAL_CONFIG)
        out = tmp_path / "bundle"
        write_pipeline_bundle(config, out, source_root=tmp_path)

        assert (out / "databricks.yaml").exists()
        assert (out / "requirements.txt").exists()
        assert (out / "config" / "my_config.yaml").exists()
        notebooks = sorted((out / "notebooks").glob("*.py"))
        assert len(notebooks) == 8

    def test_databricks_yaml_substitutes_app_name(self, tmp_path: Path) -> None:
        config = self._load(tmp_path, _MINIMAL_CONFIG)
        out = tmp_path / "bundle"
        write_pipeline_bundle(config, out, source_root=tmp_path)

        text = (out / "databricks.yaml").read_text()
        assert "__APP_NAME__" not in text
        assert "pipeline_test_app" in text

    def test_no_dev_wheel_without_development_flag(self, tmp_path: Path) -> None:
        config = self._load(tmp_path, _MINIMAL_CONFIG)
        out = tmp_path / "bundle"
        write_pipeline_bundle(config, out, source_root=tmp_path, development=False)
        assert not (out / "dist").exists()

    def test_stages_referenced_assets_preserving_layout(
        self, tmp_path: Path
    ) -> None:
        # Lay out a source tree: <root>/notebooks (CWD) + <root>/data + functions
        (tmp_path / "notebooks").mkdir()
        (tmp_path / "data" / "hw").mkdir(parents=True)
        (tmp_path / "functions" / "hw").mkdir(parents=True)
        (tmp_path / "data" / "hw" / "seed.sql").write_text("SELECT 1;")
        (tmp_path / "functions" / "hw" / "fn.sql").write_text("CREATE FUNCTION f();")

        body = (
            "resources:\n"
            "  models:\n"
            "    default_llm: &default_llm\n"
            "      name: databricks-gpt-5-4-mini\n"
            "agents:\n"
            "  greeter: &greeter\n"
            "    name: greeter\n"
            "    description: A friendly assistant.\n"
            "    model: *default_llm\n"
            "    prompt: You are a concise assistant.\n"
            "app:\n"
            "  name: hw_app\n"
            "  deployment_target: apps\n"
            "  agents:\n"
            "    - *greeter\n"
            "datasets:\n"
            "  - ddl: ../data/hw/seed.sql\n"
            "    data: null\n"
            "unity_catalog_functions:\n"
            "  - function:\n"
            "      name: c.s.f\n"
            "    ddl: ../functions/hw/fn.sql\n"
        )
        config = self._load(tmp_path, body)
        out = tmp_path / "bundle"
        write_pipeline_bundle(config, out, source_root=tmp_path)

        # Copied to the same relative location so `notebooks/../data/...` resolves.
        assert (out / "data" / "hw" / "seed.sql").exists()
        assert (out / "functions" / "hw" / "fn.sql").exists()

    def test_derived_artifacts_always_regenerate(self, tmp_path: Path) -> None:
        """databricks.yaml is derived, not user content: it must be rewritten on
        every stage regardless of ``overwrite`` so it always matches the config
        being deployed (a stale bundle name/targets would break bundle deploy)."""
        config = self._load(tmp_path, _MINIMAL_CONFIG)
        out = tmp_path / "bundle"
        write_pipeline_bundle(config, out, source_root=tmp_path)
        (out / "databricks.yaml").write_text("SENTINEL")
        # overwrite=False must STILL regenerate the derived databricks.yaml.
        write_pipeline_bundle(config, out, source_root=tmp_path, overwrite=False)
        assert (out / "databricks.yaml").read_text() != "SENTINEL"
        assert "pipeline_test_app" in (out / "databricks.yaml").read_text()

    def test_overwrite_false_preserves_user_config(self, tmp_path: Path) -> None:
        """The copied-in config IS user content: overwrite=False preserves it."""
        config = self._load(tmp_path, _MINIMAL_CONFIG)
        out = tmp_path / "bundle"
        write_pipeline_bundle(config, out, source_root=tmp_path)
        staged_config = out / "config" / "my_config.yaml"
        staged_config.write_text("SENTINEL")
        write_pipeline_bundle(config, out, source_root=tmp_path, overwrite=False)
        assert staged_config.read_text() == "SENTINEL"

    def test_overwrite_true_replaces_user_config(self, tmp_path: Path) -> None:
        config = self._load(tmp_path, _MINIMAL_CONFIG)
        out = tmp_path / "bundle"
        write_pipeline_bundle(config, out, source_root=tmp_path)
        staged_config = out / "config" / "my_config.yaml"
        staged_config.write_text("SENTINEL")
        write_pipeline_bundle(config, out, source_root=tmp_path, overwrite=True)
        assert staged_config.read_text() != "SENTINEL"


    def test_two_configs_isolated_dirs_no_collision(self, tmp_path: Path) -> None:
        """Two different configs staged into their own dirs each get a
        databricks.yaml whose bundle name matches that config — no bleed-over."""
        cfg_a = _MINIMAL_CONFIG  # app.name: pipeline_test_app
        cfg_b = _MINIMAL_CONFIG.replace(
            "name: pipeline_test_app", "name: other_app"
        )
        a = AppConfig.from_file(
            self._write(tmp_path, "a.yaml", cfg_a), initialize=False
        )
        b = AppConfig.from_file(
            self._write(tmp_path, "b.yaml", cfg_b), initialize=False
        )
        out_a, out_b = tmp_path / "A", tmp_path / "B"
        write_pipeline_bundle(a, out_a, source_root=tmp_path)
        write_pipeline_bundle(b, out_b, source_root=tmp_path)

        assert "pipeline_test_app" in (out_a / "databricks.yaml").read_text()
        assert "other_app" in (out_b / "databricks.yaml").read_text()
        assert "other_app" not in (out_a / "databricks.yaml").read_text()

    @staticmethod
    def _write(tmp_path: Path, name: str, body: str) -> Path:
        p = tmp_path / name
        p.write_text(body)
        return p


@pytest.mark.unit
def test_materialize_notebooks_skips_init(tmp_path: Path) -> None:
    written = _materialize_notebooks(tmp_path, overwrite=True)
    assert all("__init__" not in w for w in written)
    assert len(written) == 8
