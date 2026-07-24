"""``write_bundle`` (generate-agent) copies ``app.code_paths`` into the bundle.

Custom code declared via ``app.code_paths`` must be copied next to the config in
the generated Apps bundle so it imports at runtime (bundle root is the app CWD;
``add_code_paths_to_sys_path`` puts each entry's parent on ``sys.path``). This is
additive to the manual ``src/<package>`` wheel route, which still works.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from dao_ai.apps.bundle import write_bundle
from dao_ai.config import AppConfig

_CONFIG = textwrap.dedent(
    """
    resources:
      models:
        default_llm: &default_llm
          name: databricks-claude-sonnet-5
    agents:
      greeter: &greeter
        name: greeter
        description: A friendly assistant.
        model: *default_llm
        prompt: You are concise.
    app:
      name: cp_bundle_app
      code_paths:
        - tools/custom_tool.py
      registered_model:
        schema:
          catalog_name: cat
          schema_name: sch
        name: cp_bundle_model
      agents:
        - *greeter
    """
)


@pytest.fixture(autouse=True)
def _stub_bundle_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid a real ``uv lock`` (needs network / published dao-ai)."""

    def _fake_lock(bundle_dir: Path) -> None:
        (bundle_dir / "uv.lock").write_text("# stub lock for tests\n")

    monkeypatch.setattr("dao_ai.apps.bundle.generate_bundle_lock", _fake_lock)


@pytest.mark.unit
class TestBundleShipsCodePaths:
    def _config(self, tmp_path: Path) -> AppConfig:
        (tmp_path / "tools").mkdir()
        (tmp_path / "tools" / "custom_tool.py").write_text(
            "def my_tool():\n    return 'custom'\n"
        )
        cfg_path = tmp_path / "dao_ai.yaml"
        cfg_path.write_text(_CONFIG)
        return AppConfig.from_file(str(cfg_path))

    def test_code_path_copied_next_to_config(self, tmp_path: Path) -> None:
        config = self._config(tmp_path)
        out = tmp_path / "bundle_out"
        write_bundle(config, out, overwrite=True)

        # Custom code lands at its config-relative layout under the bundle root
        # (the app CWD at runtime), alongside the copied dao_ai.yaml.
        assert (out / "tools" / "custom_tool.py").exists()
        assert (out / "dao_ai.yaml").exists()

    def test_databricks_yaml_syncs_code_path_dir(self, tmp_path: Path) -> None:
        import yaml

        config = self._config(tmp_path)
        out = tmp_path / "bundle_out"
        write_bundle(config, out, overwrite=True)

        doc = yaml.safe_load((out / "databricks.yaml").read_text())
        assert "tools/**" in doc["sync"]["include"]


@pytest.mark.unit
class TestDeployAppsUploadsCodePaths:
    """The Apps direct-deploy uploads code_paths per-file under the source path."""

    def _config(self, tmp_path: Path) -> AppConfig:
        (tmp_path / "tools").mkdir()
        (tmp_path / "tools" / "custom_tool.py").write_text(
            "def my_tool():\n    return 'custom'\n"
        )
        cfg_path = tmp_path / "dao_ai.yaml"
        cfg_path.write_text(_CONFIG)
        return AppConfig.from_file(str(cfg_path))

    def test_upload_planner_walks_and_uploads_each_file(
        self, tmp_path: Path
    ) -> None:
        from unittest.mock import MagicMock

        from dao_ai.providers.databricks import DatabricksProvider

        config = self._config(tmp_path)
        provider = DatabricksProvider(w=MagicMock())
        source_path = "/Workspace/Users/u/apps/cp-bundle-app"

        provider._upload_code_paths(config, source_path)

        upload_paths = [
            c.kwargs["path"]
            for c in provider.w.workspace.upload.call_args_list
        ]
        assert upload_paths == [f"{source_path}/tools/custom_tool.py"]
        # Parent dir created before upload.
        mkdir_args = [
            c.args[0] if c.args else c.kwargs.get("path")
            for c in provider.w.workspace.mkdirs.call_args_list
        ]
        assert f"{source_path}/tools" in mkdir_args

    def test_no_uploads_when_no_code_paths(self, tmp_path: Path) -> None:
        from unittest.mock import MagicMock

        from dao_ai.providers.databricks import DatabricksProvider

        config = self._config(tmp_path)
        config.app.code_paths = []
        provider = DatabricksProvider(w=MagicMock())

        provider._upload_code_paths(config, "/Workspace/Users/u/apps/x")
        provider.w.workspace.upload.assert_not_called()


_SRC_CONFIG = textwrap.dedent(
    """
    resources:
      models:
        default_llm: &default_llm
          name: databricks-claude-sonnet-5
    agents:
      greeter: &greeter
        name: greeter
        description: A friendly assistant.
        model: *default_llm
        prompt: You are concise.
    app:
      name: src_bundle_app
      registered_model:
        schema:
          catalog_name: cat
          schema_name: sch
        name: src_bundle_model
      agents:
        - *greeter
    """
)


@pytest.mark.unit
class TestPyprojectTemplatesUseSrcConvention:
    def test_apps_templates_package_src_prefix_free(self) -> None:
        from dao_ai.apps.bundle import _PYPROJECT_DEV_TEMPLATE, _PYPROJECT_TEMPLATE

        pub = _PYPROJECT_TEMPLATE.format(
            name="x", package_name="x", dao_ai_version="0.2.4", extras="", extra_deps=""
        )
        dev = _PYPROJECT_DEV_TEMPLATE.format(
            name="x", package_name="x", wheel_filename="w.whl", extras="", extra_deps=""
        )
        for rendered in (pub, dev):
            assert 'packages = ["src"]' in rendered
            assert 'sources = ["src"]' in rendered

    def test_mcp_templates_package_src_prefix_free(self) -> None:
        from dao_ai.mcp.generate import _PYPROJECT_DEV_TEMPLATE, _PYPROJECT_TEMPLATE

        pub = _PYPROJECT_TEMPLATE.format(
            name="x",
            package_name="x",
            dao_ai_version="0.2.4",
            extras="mcp",
            extra_deps="",
        )
        dev = _PYPROJECT_DEV_TEMPLATE.format(
            name="x",
            package_name="x",
            wheel_filename="w.whl",
            extras="mcp",
            extra_deps="",
        )
        for rendered in (pub, dev):
            assert 'packages = ["src"]' in rendered
            # sources=["src"] was MISSING from MCP templates before the convention;
            # required for prefix-free (foo.bar, not src.foo.bar) imports.
            assert 'sources = ["src"]' in rendered


@pytest.mark.unit
class TestBundleShipsSrcPackages:
    def _config(self, tmp_path: Path) -> AppConfig:
        (tmp_path / "src" / "foo").mkdir(parents=True)
        (tmp_path / "src" / "foo" / "bar.py").write_text("def t():\n    return 'x'\n")
        cfg_path = tmp_path / "dao_ai.yaml"
        cfg_path.write_text(_SRC_CONFIG)
        return AppConfig.from_file(str(cfg_path))

    def test_src_package_copied_into_bundle(self, tmp_path: Path) -> None:
        config = self._config(tmp_path)
        out = tmp_path / "bundle_out"
        write_bundle(config, out, overwrite=True)

        # Copied under the bundle's src/ so hatch (packages=["src"]) builds it.
        assert (out / "src" / "foo" / "bar.py").exists()

    def test_upload_src_planner_uploads_under_src(self, tmp_path: Path) -> None:
        from unittest.mock import MagicMock

        from dao_ai.providers.databricks import DatabricksProvider

        config = self._config(tmp_path)
        provider = DatabricksProvider(w=MagicMock())
        source_path = "/Workspace/Users/u/apps/src-bundle-app"

        provider._upload_src_packages(config, source_path)

        upload_paths = [
            c.kwargs["path"] for c in provider.w.workspace.upload.call_args_list
        ]
        assert upload_paths == [f"{source_path}/src/foo/bar.py"]
