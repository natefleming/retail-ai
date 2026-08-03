"""Tests for the wheel-only pipeline staging bundle (dao_ai.pipeline.bundle).

The pipeline subcommand stages a self-contained Lakeflow-job bundle from the
installed dao-ai wheel's packaged assets — no source checkout required. These
tests cover:

- the packaged assets are reachable via importlib.resources,
- write_pipeline_bundle materializes databricks.yaml, the 8 step notebooks, and
  the resolved config into the staging dir (dao-ai installs via the serverless
  environment's dao_ai_dep dependency; no requirements.txt),
- _referenced_asset_paths picks up relative ddl/data paths and ignores
  Volume-backed / absolute ones,
- config-referenced data/functions files are copied into the bundle next to
  the staged config, resolved against the config's own directory.
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
    _referenced_asset_paths,
    generate_model_serving_agent_databricks_yaml,
    generate_pipeline_databricks_yaml,
    write_pipeline_bundle,
)
from dao_ai.utils import dao_ai_version

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

    def test_notebooks_have_no_source_path_fallback(self) -> None:
        """The wheel-only refactor drops the `../src` sys.path fallback."""
        for p in files("dao_ai.pipeline.notebooks").iterdir():
            if not p.name.endswith(".py") or p.name == "__init__.py":
                continue
            text = p.read_text(encoding="utf-8")
            assert 'sys.path.insert(0, "../src")' not in text, (
                f"{p.name} still has the ../src fallback"
            )

    def test_notebooks_bootstrap_extras_suffix(self) -> None:
        """Each step notebook's ``%uv pip install`` bootstrap must install the
        feature extras its own body can exercise.

        Graph-building notebooks (06_deploy_agent, 08_run_evaluation) build the
        agent before the config is known, so they install ``[all]``;
        01_ingest_and_transform may read EXCEL datasets so it installs
        ``[excel]``; the pure provisioning notebooks install bare dao-ai. Every
        notebook single-quotes the interpolated spec so a dev wheel's ``+local``
        version tag and any ``[extras]`` bracket survive shell expansion.
        """
        # notebook filename prefix -> the extras suffix its bootstrap must append
        # ("" means no suffix — bare core install).
        expected_suffix: dict[str, str] = {
            "01_": "[excel]",
            "02_": "",
            "03_": "",
            "04_": "",
            "05_": "",
            "06_": "[all]",
            "07_": "",
            "08_": "[all]",
        }
        seen: set[str] = set()
        for p in files("dao_ai.pipeline.notebooks").iterdir():
            if not p.name.endswith(".py") or p.name == "__init__.py":
                continue
            prefix = p.name[:3]
            assert prefix in expected_suffix, f"unmapped notebook {p.name}"
            seen.add(prefix)
            text = p.read_text(encoding="utf-8")

            # The magic must single-quote the interpolated spec (glob-safe).
            assert "# MAGIC %uv pip install --quiet '{_dao_ai_dep}'" in text, (
                f"{p.name} must single-quote the %uv install spec"
            )

            suffix = expected_suffix[prefix]
            if suffix:
                assert f'+ "{suffix}"' in text, (
                    f"{p.name} must append the {suffix} extras suffix"
                )
            else:
                # Core provisioning notebooks must not append any extras suffix.
                assert '+ "[' not in text, (
                    f"{p.name} must not append an extras suffix"
                )
        assert seen == set(expected_suffix), (
            f"notebook set changed: {sorted(seen)}"
        )


# ---------------------------------------------------------------------------
# generate_pipeline_databricks_yaml — programmatic DAB (dict -> YAML)
# ---------------------------------------------------------------------------


class _A2AStub:
    """Stand-in for AppModel.a2a — the extras resolver reads ``.enabled``."""

    enabled = False


class _AppStub:
    """Minimal stand-in — the generator reads ``config.app.name``,
    ``config.app.pip_requirements``, ``config.app.resource_paths``, and (in
    dev mode, via the extras resolver) ``config.app.a2a`` /
    ``config.app.orchestration``."""

    def __init__(self, name: str, pip_requirements: list[str] | None = None) -> None:
        self.name = name
        self.pip_requirements = pip_requirements or []
        self.resource_paths: list[str] = []
        # Resolver-touched attributes (dev-mode extras resolution).
        self.a2a = _A2AStub()
        self.orchestration = None


@pytest.mark.unit
class TestGeneratePipelineDatabricksYaml:
    @staticmethod
    def _config() -> AppConfig:
        # ``model_construct`` bypasses validation; set the collections the
        # extras resolver iterates so dev-mode resolution finds them empty.
        return AppConfig.model_construct(
            app=_AppStub("pipeline_test_app"),
            datasets=[],
            unity_catalog_functions=[],
            tools={},
            retrievers={},
            middleware={},
            memory=None,
        )

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
        # deploy-agents forwards the mode + development vars.
        params = by_key["deploy-agents"]["notebook_task"]["base_parameters"]
        assert params["mode"] == "${var.mode}"
        assert params["development"] == "${var.development}"
        # run-evaluation also forwards mode so it can pick the eval source
        # (registry for model_serving, config for apps/mcp). Without this the
        # eval task defaults to the registry and fails on apps-mode deploys,
        # which register no model.
        eval_params = by_key["run-evaluation"]["notebook_task"]["base_parameters"]
        assert eval_params["mode"] == "${var.mode}"

    def test_development_includes_wheel_in_sync(self) -> None:
        assert "dist/*.whl" in self._doc(development=True)["sync"]["include"]

    def test_published_omits_wheel_from_sync(self) -> None:
        assert "dist/*.whl" not in self._doc(development=False)["sync"]["include"]

    def test_no_requirements_txt_in_sync(self) -> None:
        # requirements.txt is retired: dao-ai (with its transitive deps) is
        # installed via the serverless environment's ``dao_ai_dep`` dependency.
        assert "requirements.txt" not in self._doc(development=False)["sync"]["include"]

    def test_env_spec_installs_dao_ai_via_bundle_var(self) -> None:
        doc = self._doc(development=False)
        # The dao_ai_dep variable exists and defaults to the version-pinned
        # PyPI spec so a raw ``bundle deploy`` (no CLI override) is reproducible.
        assert (
            doc["variables"]["dao_ai_dep"]["default"] == f"dao-ai=={dao_ai_version()}"
        )
        # The serverless environment installs exactly the dao_ai_dep dependency.
        env = doc["resources"]["jobs"]["deploy_job"]["environments"][0]
        assert env["spec"]["dependencies"] == ["${var.dao_ai_dep}"]

    def test_dev_env_deps_are_glob_safe_no_extras_on_wheel(self) -> None:
        # Regression guard: databricks bundle globs local-path env deps, so the
        # dev-mode wheel dep must NOT carry an ``[extras]`` suffix (a glob char
        # class → "no files match pattern"). Extras' backing packages are pinned
        # as separate glob-safe PyPI deps instead.
        env = self._doc(development=True)["resources"]["jobs"]["deploy_job"][
            "environments"
        ][0]
        deps = env["spec"]["dependencies"]
        # First dep is the wheel var; no dependency entry may contain "[" (extras
        # bracket) on a local path.
        for dep in deps:
            if dep.endswith(".whl") or "dist/" in dep or dep == "${var.dao_ai_dep}":
                assert "[" not in dep, (
                    f"wheel dep must be glob-safe (no extras): {dep!r}"
                )
        # In development, the extra-feature package pins are present (glob-safe
        # PyPI specs like "a2a-sdk==..."). The stub config exercises no optional
        # features, so at minimum the core pins (e.g. mlflow) appear.
        joined = " ".join(deps)
        assert "mlflow" in joined or "databricks-agents" in joined, (
            f"dev env deps should include glob-safe PyPI pins; got {deps}"
        )

    def test_no_artifacts_block(self) -> None:
        # The dao-ai wheel is never built at bundle-deploy time.
        assert "artifacts" not in self._doc(development=True)


# ---------------------------------------------------------------------------
# Thin model_serving agent Job bundle (single deploy-agent task)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGenerateModelServingAgentDatabricksYaml:
    """The thin model_serving agent bundle: same Job shape as the pipeline but a
    single deploy-agent task and a model_serving default mode."""

    @staticmethod
    def _config() -> AppConfig:
        return AppConfig.model_construct(
            app=_AppStub("ms_agent_app"),
            datasets=[],
            unity_catalog_functions=[],
            tools={},
            retrievers={},
            middleware={},
            memory=None,
        )

    def _doc(self, development: bool) -> dict:
        import yaml

        return yaml.safe_load(
            generate_model_serving_agent_databricks_yaml(
                self._config(), development=development
            )
        )

    def test_single_deploy_agent_task(self) -> None:
        tasks = self._doc(development=False)["resources"]["jobs"]["deploy_job"]["tasks"]
        assert len(tasks) == 1, (
            f"expected one task, got {[t['task_key'] for t in tasks]}"
        )
        task = tasks[0]
        assert task["task_key"] == "deploy-agent"
        assert (
            task["notebook_task"]["notebook_path"] == "./notebooks/06_deploy_agent.py"
        )
        # A lone task has no upstream dependency.
        assert "depends_on" not in task
        params = task["notebook_task"]["base_parameters"]
        assert params["config-path"] == "${var.config_path}"
        assert params["mode"] == "${var.mode}"
        assert params["development"] == "${var.development}"

    def test_mode_variable_defaults_to_model_serving(self) -> None:
        # So a raw `databricks bundle run` (no --var mode=…) deploys the endpoint.
        assert self._doc(development=False)["variables"]["mode"]["default"] == (
            "model_serving"
        )

    def test_no_app_or_experiment_resource(self) -> None:
        # It is a Job bundle — no Databricks App / experiment resource blocks
        # (the deploy notebook creates the experiment at run time).
        resources = self._doc(development=False)["resources"]
        assert set(resources) == {"jobs"}

    def test_per_cloud_targets_and_env_spec_shared(self) -> None:
        doc = self._doc(development=False)
        assert set(doc["targets"]) == {
            "ms_agent_app-azure",
            "ms_agent_app-aws",
            "ms_agent_app-gcp",
        }
        env = doc["resources"]["jobs"]["deploy_job"]["environments"][0]
        assert env["environment_key"] == "dao-ai-env"
        assert env["spec"]["dependencies"] == ["${var.dao_ai_dep}"]

    def test_development_includes_wheel_no_artifacts(self) -> None:
        doc = self._doc(development=True)
        assert "dist/*.whl" in doc["sync"]["include"]
        assert "artifacts" not in doc


@pytest.mark.unit
class TestWriteModelServingAgentBundle:
    """The MS writer stages only the deploy-agent notebook + baked config."""

    @staticmethod
    def _config(tmp_path: Path) -> AppConfig:
        cfg = tmp_path / "ms.yaml"
        cfg.write_text(_MINIMAL_CONFIG)
        return AppConfig.from_file(str(cfg), initialize=False)

    def test_stages_only_deploy_agent_notebook(self, tmp_path: Path) -> None:
        from dao_ai.pipeline.bundle import write_model_serving_agent_bundle

        out = tmp_path / "ms_out"
        write_model_serving_agent_bundle(
            self._config(tmp_path), out, overwrite=True
        )
        staged_notebooks = sorted(p.name for p in (out / "notebooks").glob("*.py"))
        assert staged_notebooks == ["06_deploy_agent.py"], staged_notebooks
        # databricks.yaml + the one notebook + the staged config are all written.
        assert (out / "databricks.yaml").exists()
        assert (out / "notebooks" / "06_deploy_agent.py").exists()
        assert (out / "config" / "ms.yaml").exists()

    def test_baked_config_has_no_parameters_block(self, tmp_path: Path) -> None:
        # MS has no provisioning task to fill deferred params, so params are baked
        # (parameters: block stripped) — like the Apps bundle.
        import yaml

        from dao_ai.pipeline.bundle import write_model_serving_agent_bundle

        out = tmp_path / "ms_out"
        write_model_serving_agent_bundle(self._config(tmp_path), out, overwrite=True)
        staged = yaml.safe_load((out / "config" / "ms.yaml").read_text())
        assert "parameters" not in staged


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
                    ddl="functions/x/table.sql",
                    data="data/x/seed.parquet",
                    format="parquet",
                )
            ],
        )
        paths = _referenced_asset_paths(config)
        assert "functions/x/table.sql" in paths
        assert "data/x/seed.parquet" in paths

    def test_volume_backed_ddl_ignored(self) -> None:
        """Volume references live on UC volumes and need no staging."""
        config = self._config(
            datasets=[DatasetModel(ddl=VolumeModel(name="c.s.v"), data=None)],
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
        write_pipeline_bundle(config, out)

        assert (out / "databricks.yaml").exists()
        # requirements.txt is retired — deps install via the env-spec dao_ai_dep.
        assert not (out / "requirements.txt").exists()
        assert (out / "config" / "my_config.yaml").exists()
        notebooks = sorted((out / "notebooks").glob("*.py"))
        assert len(notebooks) == 8

    def test_staged_config_written(self, tmp_path: Path) -> None:
        # The config is staged under config/ next to the notebooks so the job
        # reloads it at run time.
        config = self._load(tmp_path, _MINIMAL_CONFIG)
        out = tmp_path / "bundle"
        write_pipeline_bundle(config, out)
        assert (out / "config" / "my_config.yaml").exists()

    def test_databricks_yaml_substitutes_app_name(self, tmp_path: Path) -> None:
        config = self._load(tmp_path, _MINIMAL_CONFIG)
        out = tmp_path / "bundle"
        write_pipeline_bundle(config, out)

        text = (out / "databricks.yaml").read_text()
        assert "__APP_NAME__" not in text
        assert "pipeline_test_app" in text

    def test_no_dev_wheel_without_development_flag(self, tmp_path: Path) -> None:
        config = self._load(tmp_path, _MINIMAL_CONFIG)
        out = tmp_path / "bundle"
        write_pipeline_bundle(config, out, development=False)
        assert not (out / "dist").exists()

    _ASSET_CONFIG = (
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
        "  agents:\n"
        "    - *greeter\n"
        "datasets:\n"
        "  - ddl: {ddl}\n"
        "    data: null\n"
        "unity_catalog_functions:\n"
        "  - function:\n"
        "      name: c.s.f\n"
        "    ddl: {fn}\n"
    )

    def test_stages_config_relative_assets_next_to_config(self, tmp_path: Path) -> None:
        # Assets colocated with the config (config-relative bare paths).
        (tmp_path / "data").mkdir()
        (tmp_path / "functions").mkdir()
        (tmp_path / "data" / "seed.sql").write_text("SELECT 1;")
        (tmp_path / "functions" / "fn.sql").write_text("CREATE FUNCTION f();")

        body = self._ASSET_CONFIG.format(ddl="data/seed.sql", fn="functions/fn.sql")
        config = self._load(tmp_path, body)
        out = tmp_path / "bundle"
        write_pipeline_bundle(config, out)

        # Copied next to the staged config so bare `data/...` resolves against
        # the staged config's own directory at run time.
        assert (out / "config" / "data" / "seed.sql").exists()
        assert (out / "config" / "functions" / "fn.sql").exists()

    _CODE_PATHS_CONFIG = (
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
        "  name: cp_app\n"
        "  code_paths:\n"
        "    - tools/custom_tool.py\n"
        "  agents:\n"
        "    - *greeter\n"
    )

    def test_stages_code_paths_next_to_config(self, tmp_path: Path) -> None:
        # Custom code colocated with the config stages under config/ so the
        # deploy notebook resolves it against the staged config directory.
        (tmp_path / "tools").mkdir()
        (tmp_path / "tools" / "custom_tool.py").write_text(
            "def my_tool():\n    return 'custom'\n"
        )
        config = self._load(tmp_path, self._CODE_PATHS_CONFIG)
        out = tmp_path / "bundle"
        write_pipeline_bundle(config, out)

        assert (out / "config" / "tools" / "custom_tool.py").exists()
        # config/** already covers it; no extra sync glob needed.
        import yaml

        doc = yaml.safe_load((out / "databricks.yaml").read_text())
        assert "config/**" in doc["sync"]["include"]

    def test_stages_src_packages_under_config_src(self, tmp_path: Path) -> None:
        # The src/ convention: a colocated src/<pkg> stages under config/src/<pkg>
        # so the deploy notebook's src anchor (config/src) yields foo.bar.
        (tmp_path / "src" / "foo").mkdir(parents=True)
        (tmp_path / "src" / "foo" / "bar.py").write_text(
            "def my_tool():\n    return 'src-custom'\n"
        )
        config = self._load(tmp_path, _MINIMAL_CONFIG)
        out = tmp_path / "bundle"
        write_pipeline_bundle(config, out)

        assert (out / "config" / "src" / "foo" / "bar.py").exists()

    _RESOURCE_PATHS_CONFIG = (
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
        "  name: overlay_wf_app\n"
        "  resource_paths:\n"
        "    - overlays/extra_job.yml\n"
        "  agents:\n"
        "    - *greeter\n"
    )

    def test_resource_paths_overlay_parity_with_agent(
        self, tmp_path: Path
    ) -> None:
        # Parity: app.resource_paths works on the workflow noun exactly as on
        # agent/mcp — the overlay lands in resources/ and the generated
        # databricks.yaml merges it via include: [resources/*.yml].
        import yaml

        (tmp_path / "overlays").mkdir()
        (tmp_path / "overlays" / "extra_job.yml").write_text(
            "resources:\n  jobs:\n    extra:\n      name: extra\n"
        )
        config = self._load(tmp_path, self._RESOURCE_PATHS_CONFIG)
        out = tmp_path / "bundle"
        write_pipeline_bundle(config, out)

        assert (out / "resources" / "extra_job.yml").exists()
        doc = yaml.safe_load((out / "databricks.yaml").read_text())
        assert doc.get("include") == ["resources/*.yml"]
        assert "resources/**" in doc["sync"]["include"]

    def test_implicit_resources_dir_parity(self, tmp_path: Path) -> None:
        # The resources/ convention works on the workflow noun too: a *.yml
        # dropped in resources/ (no resource_paths declared) is staged + merged.
        import yaml

        (tmp_path / "resources").mkdir()
        (tmp_path / "resources" / "nightly.yml").write_text(
            "resources:\n  jobs:\n    nightly:\n      name: nightly\n"
        )
        config = self._load(tmp_path, _MINIMAL_CONFIG)
        out = tmp_path / "bundle"
        write_pipeline_bundle(config, out)

        assert (out / "resources" / "nightly.yml").exists()
        doc = yaml.safe_load((out / "databricks.yaml").read_text())
        assert doc.get("include") == ["resources/*.yml"]

    def test_no_include_when_no_overlays(self, tmp_path: Path) -> None:
        # A config with no resource_paths and no resources/ dir emits no dangling
        # include: key.
        import yaml

        config = self._load(tmp_path, _MINIMAL_CONFIG)
        out = tmp_path / "bundle"
        write_pipeline_bundle(config, out)
        doc = yaml.safe_load((out / "databricks.yaml").read_text())
        assert "include" not in doc

    def test_sibling_use_case_assets_stage_and_get_sync_glob(
        self, tmp_path: Path
    ) -> None:
        # A config that reaches a sibling use case's shared assets via `../`.
        cfg_dir = tmp_path / "use_case_b"
        cfg_dir.mkdir()
        shared = tmp_path / "use_case_a" / "functions"
        shared.mkdir(parents=True)
        (shared / "shared.sql").write_text("CREATE FUNCTION f();")

        body = self._ASSET_CONFIG.format(
            ddl="../use_case_a/functions/shared.sql",
            fn="../use_case_a/functions/shared.sql",
        )
        cfg_path = cfg_dir / "b.yaml"
        cfg_path.write_text(body)
        config = AppConfig.from_file(cfg_path, initialize=False)
        out = tmp_path / "bundle"
        write_pipeline_bundle(config, out)

        # Staged outside config/ (config/../use_case_a → bundle-root use_case_a).
        assert (out / "use_case_a" / "functions" / "shared.sql").exists()
        # ...and the databricks.yaml sync list covers that top-level dir.
        import yaml

        doc = yaml.safe_load((out / "databricks.yaml").read_text())
        assert "use_case_a/**" in doc["sync"]["include"]

    def test_derived_artifacts_always_regenerate(self, tmp_path: Path) -> None:
        """databricks.yaml is derived, not user content: it must be rewritten on
        every stage regardless of ``overwrite`` so it always matches the config
        being deployed (a stale bundle name/targets would break bundle deploy)."""
        config = self._load(tmp_path, _MINIMAL_CONFIG)
        out = tmp_path / "bundle"
        write_pipeline_bundle(config, out)
        (out / "databricks.yaml").write_text("SENTINEL")
        # overwrite=False must STILL regenerate the derived databricks.yaml.
        write_pipeline_bundle(config, out, overwrite=False)
        assert (out / "databricks.yaml").read_text() != "SENTINEL"
        assert "pipeline_test_app" in (out / "databricks.yaml").read_text()

    def test_overwrite_false_preserves_user_config(self, tmp_path: Path) -> None:
        """The copied-in config IS user content: overwrite=False preserves it."""
        config = self._load(tmp_path, _MINIMAL_CONFIG)
        out = tmp_path / "bundle"
        write_pipeline_bundle(config, out)
        staged_config = out / "config" / "my_config.yaml"
        staged_config.write_text("SENTINEL")
        write_pipeline_bundle(config, out, overwrite=False)
        assert staged_config.read_text() == "SENTINEL"

    def test_overwrite_true_replaces_user_config(self, tmp_path: Path) -> None:
        config = self._load(tmp_path, _MINIMAL_CONFIG)
        out = tmp_path / "bundle"
        write_pipeline_bundle(config, out)
        staged_config = out / "config" / "my_config.yaml"
        staged_config.write_text("SENTINEL")
        write_pipeline_bundle(config, out, overwrite=True)
        assert staged_config.read_text() != "SENTINEL"

    def test_two_configs_isolated_dirs_no_collision(self, tmp_path: Path) -> None:
        """Two different configs staged into their own dirs each get a
        databricks.yaml whose bundle name matches that config — no bleed-over."""
        cfg_a = _MINIMAL_CONFIG  # app.name: pipeline_test_app
        cfg_b = _MINIMAL_CONFIG.replace("name: pipeline_test_app", "name: other_app")
        a = AppConfig.from_file(
            self._write(tmp_path, "a.yaml", cfg_a), initialize=False
        )
        b = AppConfig.from_file(
            self._write(tmp_path, "b.yaml", cfg_b), initialize=False
        )
        out_a, out_b = tmp_path / "A", tmp_path / "B"
        write_pipeline_bundle(a, out_a)
        write_pipeline_bundle(b, out_b)

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


@pytest.mark.unit
class TestGenieProvisioningStaging:
    """Workflow staging must PRESERVE a Genie room's ``space_id: ${var.X}`` binding
    (+ its declaration) when the operator did NOT supply X, so
    05_provision_genie can provision and 06_deploy_agent can inject the id.
    A supplied X bakes to a literal (reuse). Non-genie params always bake."""

    _CFG = (
        "parameters:\n"
        "  catalog:\n    default: main\n"
        "  genie_space_id:\n    description: provisioned by workflow\n    provided: true\n"
        "schemas:\n  s: &s\n    catalog_name: ${var.catalog}\n    schema_name: dao_ai\n"
        "resources:\n"
        "  warehouses:\n    wh: &wh\n      name: Serverless Starter Warehouse\n"
        "  genie_rooms:\n"
        "    room: &room\n"
        "      name: test room\n"
        '      space_id: "${var.genie_space_id}"\n'
        "      warehouse: *wh\n"
        "  llms:\n    m: &m\n      name: databricks-test-llm\n"
        "agents:\n  a: &a\n    name: a\n    description: a\n    model: *m\n    prompt: hi\n"
        "app:\n  name: prov_test\n  agents:\n    - *a\n"
    )

    def _cfg(self, tmp_path: Path) -> Path:
        p = tmp_path / "prov.yaml"
        p.write_text(self._CFG)
        return p

    def _staged_text(self, tmp_path: Path, out_name: str, **from_file_kwargs) -> str:
        cfg = AppConfig.from_file(
            str(self._cfg(tmp_path)), initialize=False, **from_file_kwargs
        )
        out = tmp_path / out_name
        write_pipeline_bundle(cfg, out)
        return (out / "config" / "prov.yaml").read_text()

    def test_unprovided_genie_param_ref_survives_staging(self, tmp_path: Path) -> None:
        staged = self._staged_text(tmp_path, "omit")
        # The ${var.genie_space_id} ref is preserved, and its declaration retained.
        assert "${var.genie_space_id}" in staged
        assert "genie_space_id:" in staged
        # Non-genie params are baked; their decls stripped.
        assert "${var.catalog}" not in staged
        assert "catalog_name: main" in staged
        assert "catalog:" not in staged.split("resources:")[0]

    def test_supplied_genie_param_is_baked_no_provision(self, tmp_path: Path) -> None:
        staged = self._staged_text(
            tmp_path, "passid", params={"genie_space_id": "01fEXISTING"}
        )
        # Operator supplied it → baked to a literal, declaration stripped (reuse).
        assert "${var.genie_space_id}" not in staged
        assert "01fEXISTING" in staged
        assert "parameters:" not in staged

    def test_staged_provision_config_round_trips(self, tmp_path: Path) -> None:
        """The staged (deferred) config reloads so 05's gate fires and 06 injects."""
        from dao_ai.config import is_parameter, parameter_name, value_of

        cfg = AppConfig.from_file(str(self._cfg(tmp_path)), initialize=False)
        out = tmp_path / "rt"
        write_pipeline_bundle(cfg, out)
        staged_path = str(out / "config" / "prov.yaml")

        # 05 gate
        reloaded = AppConfig.from_file(staged_path, initialize=False)
        room = list(reloaded.resources.genie_rooms.values())[0]
        assert is_parameter(room.raw_space_id) is True
        assert parameter_name(room.raw_space_id) == "genie_space_id"
        assert not value_of(room.space_id)

        # 06 injection via taskValues
        class _FakeTV:
            def get(self, taskKey, key, default="", debugValue=""):
                return "PROVISIONED_123" if key == "genie_space_id" else default

        injected = AppConfig.from_file(
            staged_path,
            task_values=_FakeTV(),
            task_key="provision-genie",
            initialize=False,
        )
        room2 = list(injected.resources.genie_rooms.values())[0]
        assert value_of(room2.space_id) == "PROVISIONED_123"
