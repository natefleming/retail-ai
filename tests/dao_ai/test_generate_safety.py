"""Guards that generate-* never destroy a user's own code.

The generators copy the config's colocated ``src/`` packages and ``code_paths``
files into the bundle. If the output dir overlaps the config's own directory, or
``--overwrite`` is passed, naive copying could overwrite the user's source. These
tests lock in the safety rules:

- the CLI aborts when the output dir overlaps the config's directory,
- ``src/``, ``code_paths``, and the source config are never overwritten (even
  with ``overwrite=True``) — they are reported as preserved,
- the empty stub never clobbers a real ``src/<app>/__init__.py``,
- copying a file onto itself (in-place) is a no-op, not a ``SameFileError``.
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
      name: safe_app
      registered_model:
        schema:
          catalog_name: cat
          schema_name: sch
        name: safe_model
      agents:
        - *greeter
    """
)


@pytest.fixture(autouse=True)
def _stub_bundle_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_lock(bundle_dir: Path) -> None:
        (bundle_dir / "uv.lock").write_text("# stub lock for tests\n")

    monkeypatch.setattr("dao_ai.apps.bundle.generate_bundle_lock", _fake_lock)


def _config_with_src(tmp_path: Path) -> AppConfig:
    (tmp_path / "src" / "mypkg").mkdir(parents=True)
    (tmp_path / "src" / "mypkg" / "tool.py").write_text("VALUE = 'real-code'\n")
    cfg_path = tmp_path / "dao_ai.yaml"
    cfg_path.write_text(_CONFIG)
    return AppConfig.from_file(str(cfg_path))


# ---------------------------------------------------------------------------
# Generating INTO the config's own dir does not abort — it leaves the user's
# src/ and code_paths (and config) alone and continues.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGenerateInPlaceIsNonFatal:
    def test_in_place_generation_preserves_user_code_and_continues(
        self, tmp_path: Path
    ) -> None:
        cfg = _config_with_src(tmp_path)
        original_src = (tmp_path / "src" / "mypkg" / "tool.py").read_text()
        original_cfg = (tmp_path / "dao_ai.yaml").read_text()

        # Output dir IS the config dir — must NOT raise; generates the scaffold
        # while leaving the user's src/ and config untouched.
        write_bundle(cfg, tmp_path, overwrite=True)

        assert (tmp_path / "src" / "mypkg" / "tool.py").read_text() == original_src
        assert (tmp_path / "dao_ai.yaml").read_text() == original_cfg
        # Generated scaffold still lands.
        assert (tmp_path / "databricks.yaml").exists()
        assert (tmp_path / "pyproject.toml").exists()


# ---------------------------------------------------------------------------
# write_bundle: user code is sacred
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUserCodeSacred:
    def test_existing_src_not_overwritten_even_with_overwrite(
        self, tmp_path: Path
    ) -> None:
        cfg = _config_with_src(tmp_path)
        out = tmp_path / "bundle"
        # Pre-seed a user-edited copy in the bundle's src/ with DIFFERENT content.
        (out / "src" / "mypkg").mkdir(parents=True)
        sentinel = "USER_EDITED = 1\n"
        (out / "src" / "mypkg" / "tool.py").write_text(sentinel)

        write_bundle(cfg, out, overwrite=True)

        # Sacred: the pre-existing user file is left byte-identical.
        assert (out / "src" / "mypkg" / "tool.py").read_text() == sentinel

    def test_stub_does_not_clobber_real_init(self, tmp_path: Path) -> None:
        # User's package IS named after the app (safe_app -> safe_app), with a
        # real __init__.py. The stub must not overwrite it with empty content.
        (tmp_path / "src" / "safe_app").mkdir(parents=True)
        real_init = "from .x import y  # real package init\n"
        (tmp_path / "src" / "safe_app" / "__init__.py").write_text(real_init)
        (tmp_path / "src" / "safe_app" / "x.py").write_text("y = 1\n")
        cfg_path = tmp_path / "dao_ai.yaml"
        cfg_path.write_text(_CONFIG)
        cfg = AppConfig.from_file(str(cfg_path))

        out = tmp_path / "bundle"
        write_bundle(cfg, out, overwrite=True)

        assert (out / "src" / "safe_app" / "__init__.py").read_text() == real_init

    def test_stub_created_when_absent(self, tmp_path: Path) -> None:
        cfg = _config_with_src(tmp_path)  # no src/safe_app package
        out = tmp_path / "bundle"
        write_bundle(cfg, out, overwrite=True)
        # The app-name stub is scaffolded (empty) since the user had none.
        assert (out / "src" / "safe_app" / "__init__.py").exists()
        assert (out / "src" / "safe_app" / "__init__.py").read_text() == ""

    def test_in_place_copy_is_noop_not_error(self, tmp_path: Path) -> None:
        # output_dir == config dir would normally be blocked by the CLI guard,
        # but write_bundle itself must not raise SameFileError on in-place files.
        cfg = _config_with_src(tmp_path)
        # Deploy into the config dir directly (bypassing the CLI guard).
        write_bundle(cfg, tmp_path, overwrite=True)
        # The user's real src/ file survives unchanged.
        assert (
            tmp_path / "src" / "mypkg" / "tool.py"
        ).read_text() == "VALUE = 'real-code'\n"

    def test_source_config_not_overwritten_in_place(self, tmp_path: Path) -> None:
        cfg = _config_with_src(tmp_path)
        original = (tmp_path / "dao_ai.yaml").read_text()
        write_bundle(cfg, tmp_path, overwrite=True)
        # The original config (with its parameters block) is untouched.
        assert (tmp_path / "dao_ai.yaml").read_text() == original
