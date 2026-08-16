"""Guard the hand-run demo notebooks under the repo's top-level ``notebooks/``.

These are not packaged in the wheel and are not importable, so — like
``test_pipeline_bundle.TestPackagedAssets`` does for the step notebooks — the
only way to keep them runnable is to scan their source for the references that
broke them:

- ``%pip install -r ../requirements.txt``, where ``requirements.txt`` was deleted
  in d88c0c53 and six of the seven notebooks were never updated. They died in
  cell 1, before anything else could run.
- the ``config-paths`` dropdown, fed by a local ``find_yaml_files_os_walk`` copy
  that *raises* ``FileNotFoundError`` on a missing ``../config`` (there is no
  such directory in a repo checkout), so these three never even reached the
  dropdown's own empty-``choices`` crash.

Skipped when ``notebooks/`` is absent, so an installed-wheel test run is fine.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from dao_ai.git_source import is_git_locator

_REPO_ROOT: Path = Path(__file__).parents[2]
_NOTEBOOKS: Path = _REPO_ROOT / "notebooks"
_CELL_SEP = "# COMMAND ----------"


def _notebook_sources() -> list[Path]:
    if not _NOTEBOOKS.is_dir():
        pytest.skip("repo notebooks/ dir not present (installed-wheel test run)")
    paths = sorted(_NOTEBOOKS.glob("*.py"))
    assert paths, "notebooks/ exists but contains no .py notebooks"
    return paths


@pytest.mark.unit
class TestRepoNotebooks:
    def test_no_dead_requirements_reference(self) -> None:
        """``requirements.txt`` does not exist; nothing may install from it."""
        assert not (_REPO_ROOT / "requirements.txt").exists(), (
            "requirements.txt is back — update or delete this test"
        )
        for path in _notebook_sources():
            text = path.read_text(encoding="utf-8")
            assert "requirements.txt" not in text, (
                f"{path.name} installs from ../requirements.txt, which does not exist"
            )

    def test_bootstrap_installs_dao_ai(self) -> None:
        """A notebook that imports dao_ai must install it the way the step
        notebooks do: the newest ``../dist`` wheel if present, else the published
        PyPI package.

        This includes imports inside an optional branch —
        ``06_background_agents`` installed only the HTTP clients, so its
        "deploy the app first" cell raised ImportError.
        """
        for path in _notebook_sources():
            text = path.read_text(encoding="utf-8")
            if "import dao_ai" not in text and "from dao_ai" not in text:
                continue
            assert 'glob.glob("../dist/dao_ai-*.whl")' in text, (
                f"{path.name} must bootstrap dao-ai from ../dist or PyPI"
            )
            assert "# MAGIC %uv pip install --quiet '{_dao_ai_dep}'" in text, (
                f"{path.name} must single-quote the interpolated install spec"
            )

    def test_no_config_discovery(self) -> None:
        """No notebook may discover its own config, by dropdown or by walk."""
        for path in _notebook_sources():
            text = path.read_text(encoding="utf-8")
            assert "config-paths" not in text, (
                f"{path.name} still declares the config-paths dropdown"
            )
            assert "find_yaml_files_os_walk" not in text, (
                f"{path.name} still walks ../config for candidate configs"
            )

    def test_config_path_widget_has_a_usable_default(self) -> None:
        """These are hand-run demos, so the widget points at a shipped example
        rather than starting empty — the single biggest usability win here. The
        default must actually resolve to a config in this repo.

        Two spellings resolve: a path relative to ``notebooks/``, and a git
        locator naming an in-repo path (``08_provision_from_git`` demonstrates
        loading with nothing checked out, so a relative path would defeat it).
        Either way the target file has to exist, which is what stops a default
        from rotting when an example is renamed.
        """
        checked: list[str] = []
        for path in _notebook_sources():
            text = path.read_text(encoding="utf-8")
            if 'dbutils.widgets.text(\n    name="config-path"' not in text:
                continue
            checked.append(path.name)
            default = text.split('name="config-path",\n    defaultValue="', 1)[1]
            default = default.split('"', 1)[0]
            if is_git_locator(default):
                in_repo: str = default.partition("#")[2]
                assert in_repo, (
                    f"{path.name} defaults config-path to the locator {default!r} "
                    "with no `#path`, so it relies on config discovery"
                )
                target: Path = (_REPO_ROOT / in_repo).resolve()
            else:
                assert default.startswith("../examples/"), (
                    f"{path.name} should default config-path to a shipped example "
                    f"or a git locator, got {default!r}"
                )
                target = (_NOTEBOOKS / default).resolve()
            assert target.is_file(), (
                f"{path.name} defaults config-path to {default!r}, which does not "
                f"resolve to a config in this repo ({target})"
            )
        # Guard against the string match silently drifting and the loop becoming
        # a no-op.
        assert len(checked) >= 4, f"only inspected {checked}"

    def test_wheel_globs_are_sorted_by_version(self) -> None:
        """Every ``../dist`` wheel glob must be sorted by PEP 440 version.

        A lexical sort puts ``0.2.8`` above ``0.2.10``, so a stale wheel wins
        the moment the minor version reaches double digits. Five of these
        notebooks glob twice — bootstrap plus a force-reinstall cell — and a
        cell is its own execution unit, so ``_wheel_version``'s imports have to
        sit beside each copy. Twin of ``_assert_wheel_selection`` in
        ``test_pipeline_bundle.py``, which guards the packaged step notebooks.
        """
        globbed: list[str] = []
        for path in _notebook_sources():
            for cell in path.read_text(encoding="utf-8").split(_CELL_SEP):
                if "glob.glob(" not in cell or "dao_ai-*.whl" not in cell:
                    continue
                globbed.append(path.name)
                assert "key=_wheel_version" in cell, (
                    f"{path.name} sorts the ../dist wheel glob lexically — "
                    "0.2.8 would beat 0.2.10; sort with key=_wheel_version"
                )
                assert "from packaging.version import Version" in cell, (
                    f"{path.name} uses _wheel_version without importing "
                    "Version in that cell"
                )
                assert re.search(r"^import .*\bos\b", cell, re.M), (
                    f"{path.name} uses _wheel_version without importing os in "
                    "that cell"
                )
        assert len(globbed) >= 7, f"only inspected {globbed}"
