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
    (tmp_path / "cfg.yaml").write_text(
        _CONFIG_TEMPLATE.format(code_paths_block=block)
    )
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
        config = _write_config(
            tmp_path, ["tools/a.py", "tools/b.py", "tools/a.py"]
        )

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
