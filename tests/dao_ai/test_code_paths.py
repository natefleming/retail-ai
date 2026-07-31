"""Unit tests for the shared ``code_paths`` helpers.

These cover the resolver, the Model Serving collector, the shared staging
planner (used by both bundle generators and the Apps upload), and the sync-glob
helper. Path-semantics contract: relative entries resolve against the config
file's directory, with a legacy CWD fallback; absolute / ``../``-climbing entries
fall back to a ``code/<basename>`` bundle dest.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from dao_ai import code_paths as cp
from dao_ai.config import AppConfig

_CONFIG_TEMPLATE = textwrap.dedent(
    """
    schemas:
      s: &s
        catalog_name: cat
        schema_name: sch
    resources:
      models:
        m: &m
          name: databricks-claude-sonnet-5
    agents:
      a: &a
        name: agent_one
        model: *m
        prompt: hi
    app:
      name: test_app
      code_paths:
    {code_paths_block}
      registered_model:
        schema: *s
        name: test_model
      agents:
      - *a
    """
)


def _write_config(tmp_path: Path, code_paths: list[str]) -> AppConfig:
    block = "\n".join(f"    - {p}" for p in code_paths)
    (tmp_path / "cfg.yaml").write_text(_CONFIG_TEMPLATE.format(code_paths_block=block))
    return AppConfig.from_file(str(tmp_path / "cfg.yaml"))


_CONFIG_NO_CODE_PATHS = textwrap.dedent(
    """
    schemas:
      s: &s
        catalog_name: cat
        schema_name: sch
    resources:
      models:
        m: &m
          name: databricks-claude-sonnet-5
    agents:
      a: &a
        name: agent_one
        model: *m
        prompt: hi
    app:
      name: test_app
      registered_model:
        schema: *s
        name: test_model
      agents:
      - *a
    """
)


def _write_config_no_code_paths(tmp_path: Path) -> AppConfig:
    (tmp_path / "cfg.yaml").write_text(_CONFIG_NO_CODE_PATHS)
    return AppConfig.from_file(str(tmp_path / "cfg.yaml"))


class TestResolveCodePath:
    def test_config_relative_resolves_against_config_dir(self, tmp_path: Path) -> None:
        (tmp_path / "tools").mkdir()
        target = tmp_path / "tools" / "custom_tool.py"
        target.write_text("def my_tool():\n    return 'x'\n")
        config = _write_config(tmp_path, ["tools/custom_tool.py"])

        resolved = cp.resolve_code_path("tools/custom_tool.py", config)
        assert resolved == target.resolve()

    def test_absolute_passes_through(self, tmp_path: Path) -> None:
        target = tmp_path / "abs_tool.py"
        target.write_text("x = 1\n")
        config = _write_config(tmp_path, ["tools/placeholder.py"])
        (tmp_path / "tools").mkdir()
        (tmp_path / "tools" / "placeholder.py").write_text("x = 1\n")

        resolved = cp.resolve_code_path(str(target), config)
        assert resolved == target

    def test_cwd_fallback_for_legacy_relative(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Config lives in one dir; the code file only exists relative to CWD —
        # the legacy Model Serving behavior. The fallback must still find it.
        (tmp_path / "tools").mkdir()
        (tmp_path / "tools" / "placeholder.py").write_text("x = 1\n")
        config = _write_config(tmp_path, ["tools/placeholder.py"])

        cwd = tmp_path / "elsewhere"
        (cwd / "legacy").mkdir(parents=True)
        (cwd / "legacy" / "mod.py").write_text("x = 1\n")
        monkeypatch.chdir(cwd)

        resolved = cp.resolve_code_path("legacy/mod.py", config)
        assert resolved == (cwd / "legacy" / "mod.py").resolve()

    def test_missing_returns_none(self, tmp_path: Path) -> None:
        (tmp_path / "tools").mkdir()
        (tmp_path / "tools" / "placeholder.py").write_text("x = 1\n")
        config = _write_config(tmp_path, ["tools/placeholder.py"])
        assert cp.resolve_code_path("tools/nope.py", config) is None


class TestCollectCodePaths:
    def test_dedupes_and_preserves_order(self, tmp_path: Path) -> None:
        (tmp_path / "tools").mkdir()
        (tmp_path / "tools" / "a.py").write_text("x = 1\n")
        (tmp_path / "tools" / "b.py").write_text("x = 1\n")
        config = _write_config(tmp_path, ["tools/a.py", "tools/b.py", "tools/a.py"])

        collected = cp.collect_code_paths(config)
        assert [Path(p).name for p in collected] == ["a.py", "b.py"]

    def test_raises_on_missing(self, tmp_path: Path) -> None:
        (tmp_path / "tools").mkdir()
        (tmp_path / "tools" / "placeholder.py").write_text("x = 1\n")
        config = _write_config(tmp_path, ["tools/placeholder.py"])
        config.app.code_paths = ["tools/missing.py"]

        with pytest.raises(FileNotFoundError, match="missing.py"):
            cp.collect_code_paths(config)

    def test_empty_when_no_code_paths(self, tmp_path: Path) -> None:
        (tmp_path / "tools").mkdir()
        (tmp_path / "tools" / "placeholder.py").write_text("x = 1\n")
        config = _write_config(tmp_path, ["tools/placeholder.py"])
        config.app.code_paths = []
        assert cp.collect_code_paths(config) == []


class TestIterCodePathStagings:
    def test_file_entry_preserves_layout(self, tmp_path: Path) -> None:
        (tmp_path / "tools").mkdir()
        target = tmp_path / "tools" / "custom_tool.py"
        target.write_text("x = 1\n")
        config = _write_config(tmp_path, ["tools/custom_tool.py"])

        stagings = cp.iter_code_path_stagings(config)
        assert stagings == [(target.resolve(), "tools/custom_tool.py")]

    def test_directory_entry_single_pair(self, tmp_path: Path) -> None:
        pkg = tmp_path / "mypkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "mod.py").write_text("x = 1\n")
        config = _write_config(tmp_path, ["mypkg"])

        stagings = cp.iter_code_path_stagings(config)
        assert stagings == [(pkg.resolve(), "mypkg")]

    def test_absolute_entry_falls_back_to_code_prefix(self, tmp_path: Path) -> None:
        (tmp_path / "tools").mkdir()
        (tmp_path / "tools" / "placeholder.py").write_text("x = 1\n")
        abs_file = tmp_path / "outside.py"
        abs_file.write_text("x = 1\n")
        config = _write_config(tmp_path, [str(abs_file)])

        stagings = cp.iter_code_path_stagings(config)
        assert stagings == [(abs_file, "code/outside.py")]


_CONFIG_RESOURCE_PATHS = textwrap.dedent(
    """
    schemas:
      s: &s
        catalog_name: cat
        schema_name: sch
    resources:
      models:
        m: &m
          name: databricks-claude-sonnet-5
    agents:
      a: &a
        name: agent_one
        model: *m
        prompt: hi
    app:
      name: test_app
      resource_paths:
    {block}
      registered_model:
        schema: *s
        name: test_model
      agents:
      - *a
    """
)


def _write_config_resource_paths(tmp_path: Path, entries: list[str]) -> AppConfig:
    block = "\n".join(f"    - {e}" for e in entries)
    (tmp_path / "cfg.yaml").write_text(_CONFIG_RESOURCE_PATHS.format(block=block))
    return AppConfig.from_file(str(tmp_path / "cfg.yaml"))


class TestIterResourcePathStagings:
    def test_config_relative_lands_flat_in_resources(self, tmp_path: Path) -> None:
        (tmp_path / "overlays").mkdir()
        target = tmp_path / "overlays" / "jobs.yml"
        target.write_text("resources: {}\n")
        config = _write_config_resource_paths(tmp_path, ["overlays/jobs.yml"])

        # Dest is resources/<basename> — the declared subpath is NOT preserved,
        # because DABs' ``include: [resources/*.yml]`` merges a flat directory.
        stagings = cp.iter_resource_path_stagings(config)
        assert stagings == [(target.resolve(), "resources/jobs.yml")]

    def test_missing_entry_fails_loud(self, tmp_path: Path) -> None:
        # A typo'd/absent overlay must NOT silently ship a bundle missing the
        # user's resource — fail loud like collect_code_paths does for code.
        config = _write_config_resource_paths(tmp_path, ["overlays/absent.yml"])
        with pytest.raises(FileNotFoundError, match="does not exist"):
            cp.iter_resource_path_stagings(config)

    def test_duplicate_basename_raises(self, tmp_path: Path) -> None:
        # Two entries flattening to the same resources/<name> would silently drop
        # one — reject the collision instead.
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        (tmp_path / "a" / "jobs.yml").write_text("resources: {}\n")
        (tmp_path / "b" / "jobs.yml").write_text("resources: {}\n")
        config = _write_config_resource_paths(tmp_path, ["a/jobs.yml", "b/jobs.yml"])
        with pytest.raises(ValueError, match="resources/jobs.yml"):
            cp.iter_resource_path_stagings(config)

    def test_reserved_app_yml_basename_raises(self, tmp_path: Path) -> None:
        # An overlay named app.yml would collide with the generated
        # resources/app.yml and be silently shadowed — reject it.
        (tmp_path / "o").mkdir()
        (tmp_path / "o" / "app.yml").write_text("resources: {}\n")
        config = _write_config_resource_paths(tmp_path, ["o/app.yml"])
        with pytest.raises(ValueError, match="reserved basename"):
            cp.iter_resource_path_stagings(config)

    def test_empty_when_unset(self, tmp_path: Path) -> None:
        config = _write_config_no_code_paths(tmp_path)
        assert cp.iter_resource_path_stagings(config) == []


class TestResourcesConvention:
    """The colocated ``resources/`` dir auto-ships *.yml overlays (like src/)."""

    def test_implicit_resources_dir_discovered(self, tmp_path: Path) -> None:
        (tmp_path / "resources").mkdir()
        target = tmp_path / "resources" / "nightly.yml"
        target.write_text("resources: {}\n")
        # No resource_paths declared — the convention picks it up.
        config = _write_config_no_code_paths(tmp_path)
        assert cp.discover_resource_overlays(config) == [target.resolve()]
        assert cp.iter_resource_path_stagings(config) == [
            (target.resolve(), "resources/nightly.yml")
        ]

    def test_reserved_and_non_yaml_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "resources").mkdir()
        (tmp_path / "resources" / "app.yml").write_text("resources: {}\n")  # reserved
        (tmp_path / "resources" / "notes.txt").write_text("hi\n")  # non-yaml
        (tmp_path / "resources" / "jobs.yaml").write_text("resources: {}\n")  # kept
        config = _write_config_no_code_paths(tmp_path)
        names = [p.name for p in cp.discover_resource_overlays(config)]
        assert names == ["jobs.yaml"]

    def test_explicit_and_implicit_dedup_by_path(self, tmp_path: Path) -> None:
        # An explicit resource_paths entry that points at a file already under the
        # resources/ dir must not be staged twice.
        (tmp_path / "resources").mkdir()
        (tmp_path / "resources" / "jobs.yml").write_text("resources: {}\n")
        config = _write_config_resource_paths(tmp_path, ["resources/jobs.yml"])
        stagings = cp.iter_resource_path_stagings(config)
        assert stagings == [
            ((tmp_path / "resources" / "jobs.yml").resolve(), "resources/jobs.yml")
        ]


class TestWalkCodePathFiles:
    def test_file_yields_itself(self, tmp_path: Path) -> None:
        f = tmp_path / "a.py"
        f.write_text("x = 1\n")
        assert cp.walk_code_path_files(f, "tools/a.py") == [(f, "tools/a.py")]

    def test_directory_walks_and_skips_pycache(self, tmp_path: Path) -> None:
        pkg = tmp_path / "pkg"
        (pkg / "__pycache__").mkdir(parents=True)
        (pkg / "__pycache__" / "x.pyc").write_text("junk")
        (pkg / "__init__.py").write_text("")
        (pkg / "mod.py").write_text("x = 1\n")

        walked = cp.walk_code_path_files(pkg, "pkg")
        dests = sorted(d for _s, d in walked)
        assert dests == ["pkg/__init__.py", "pkg/mod.py"]


class TestPrependCodePathsToSysPath:
    def test_makes_config_relative_module_importable_from_other_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Custom module colocated with the config; process CWD is elsewhere
        # (mimics deploy-time log_model validation running from the repo root).
        (tmp_path / "custom_pkg").mkdir()
        (tmp_path / "custom_pkg" / "mod.py").write_text("VALUE = 'cp-ok'\n")
        config = _write_config(tmp_path, ["custom_pkg"])

        other = tmp_path / "elsewhere"
        other.mkdir()
        monkeypatch.chdir(other)

        import sys as _sys

        _sys.modules.pop("custom_pkg", None)
        _sys.modules.pop("custom_pkg.mod", None)
        cp.prepend_code_paths_to_sys_path(config)
        import importlib

        mod = importlib.import_module("custom_pkg.mod")
        assert mod.VALUE == "cp-ok"

    def test_noop_when_no_code_paths(self, tmp_path: Path) -> None:
        (tmp_path / "tools").mkdir()
        (tmp_path / "tools" / "placeholder.py").write_text("x = 1\n")
        config = _write_config(tmp_path, ["tools/placeholder.py"])
        config.app.code_paths = []
        # Should not raise.
        cp.prepend_code_paths_to_sys_path(config)


class TestCodePathSyncGlobs:
    def test_config_relative_entries_contribute_nothing(self, tmp_path: Path) -> None:
        (tmp_path / "tools").mkdir()
        (tmp_path / "tools" / "a.py").write_text("x = 1\n")
        config = _write_config(tmp_path, ["tools/a.py"])
        # 'tools' is a top-level dir next to the config, not 'config'
        assert cp.code_path_sync_globs(config) == ["tools/**"]

    def test_absolute_entry_yields_code_glob(self, tmp_path: Path) -> None:
        (tmp_path / "tools").mkdir()
        (tmp_path / "tools" / "placeholder.py").write_text("x = 1\n")
        abs_file = tmp_path / "outside.py"
        abs_file.write_text("x = 1\n")
        config = _write_config(tmp_path, [str(abs_file)])
        assert cp.code_path_sync_globs(config) == ["code/**"]


class TestDiscoverSrcPackages:
    def test_discovers_top_level_packages(self, tmp_path: Path) -> None:
        (tmp_path / "src" / "foo").mkdir(parents=True)
        (tmp_path / "src" / "baz").mkdir()
        (tmp_path / "src" / "foo" / "bar.py").write_text("x = 1\n")  # namespace pkg
        (tmp_path / "src" / "baz" / "__init__.py").write_text("")  # regular pkg
        config = _write_config_no_code_paths(tmp_path)

        names = [p.name for p in cp.discover_src_packages(config)]
        assert names == ["baz", "foo"]  # sorted

    def test_absent_src_returns_empty(self, tmp_path: Path) -> None:
        config = _write_config_no_code_paths(tmp_path)
        assert cp.discover_src_packages(config) == []

    def test_empty_src_returns_empty(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        config = _write_config_no_code_paths(tmp_path)
        assert cp.discover_src_packages(config) == []

    def test_skips_loose_files_pycache_egginfo(self, tmp_path: Path) -> None:
        (tmp_path / "src" / "foo").mkdir(parents=True)
        (tmp_path / "src" / "foo" / "__init__.py").write_text("")
        (tmp_path / "src" / "loose.py").write_text("x = 1\n")  # loose file → ignored
        (tmp_path / "src" / "__pycache__").mkdir()
        (tmp_path / "src" / "pkg.egg-info").mkdir()
        config = _write_config_no_code_paths(tmp_path)

        assert [p.name for p in cp.discover_src_packages(config)] == ["foo"]


class TestCollectServingCodePaths:
    def test_unions_code_paths_and_src(self, tmp_path: Path) -> None:
        (tmp_path / "tools").mkdir()
        (tmp_path / "tools" / "t.py").write_text("x = 1\n")
        (tmp_path / "src" / "foo").mkdir(parents=True)
        (tmp_path / "src" / "foo" / "bar.py").write_text("x = 1\n")
        config = _write_config(tmp_path, ["tools/t.py"])

        names = sorted(Path(p).name for p in cp.collect_serving_code_paths(config))
        assert names == ["foo", "t.py"]

    def test_dedupes_code_path_pointing_into_src(self, tmp_path: Path) -> None:
        # A code_paths entry that IS a src package must not be shipped twice.
        (tmp_path / "src" / "foo").mkdir(parents=True)
        (tmp_path / "src" / "foo" / "bar.py").write_text("x = 1\n")
        config = _write_config(tmp_path, ["src/foo"])

        resolved = cp.collect_serving_code_paths(config)
        assert len(resolved) == 1
        assert Path(resolved[0]).name == "foo"


class TestPrependSrcToSysPath:
    def test_puts_src_on_path_import_prefix_free(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "src" / "cpkg").mkdir(parents=True)
        (tmp_path / "src" / "cpkg" / "mod.py").write_text("VALUE = 'src-ok'\n")
        config = _write_config_no_code_paths(tmp_path)

        other = tmp_path / "elsewhere"
        other.mkdir()
        monkeypatch.chdir(other)

        import importlib
        import sys as _sys

        _sys.modules.pop("cpkg", None)
        _sys.modules.pop("cpkg.mod", None)
        cp.prepend_src_to_sys_path(config)
        # FQN is prefix-free: cpkg.mod, NOT src.cpkg.mod
        assert importlib.import_module("cpkg.mod").VALUE == "src-ok"

    def test_noop_when_src_absent(self, tmp_path: Path) -> None:
        config = _write_config_no_code_paths(tmp_path)
        cp.prepend_src_to_sys_path(config)  # must not raise
