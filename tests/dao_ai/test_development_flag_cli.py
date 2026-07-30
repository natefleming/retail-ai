"""Tests for the --development / --no-development tri-state across CLI commands.

The flag must resolve identically on every build-capable command:
``<noun> build`` (agent/workflow) — ``--development`` -> True,
``--no-development`` -> False, omitted -> None (auto-detect). For the workflow
it is additionally forwarded to the deploy notebook as a
``--var development=auto|true|false`` bundle variable.

The removed commands ``deploy``, ``generate-agent``, ``generate-workflow``, and
``generate-mcp`` must now be rejected by the parser (the noun lifecycle verbs
are now ``build``/``sync``/``start``/``down``).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import call, patch

import pytest

from dao_ai import cli
from dao_ai.cli import deploy_app_bundle, parse_args, run_databricks_command
from dao_ai.config import AppConfig


def _base(cmd: str) -> list[str]:
    """Build argv for a command, splitting a nested `"agent build"` on space."""
    return cmd.split() + ["-c", "config.yaml"]


def _stamp_manifest(
    bundle_dir: Path, *, fingerprint: str = "", files: list[str] | None = None
) -> None:
    """Write a `.dao-ai-manifest.yaml` recording the given files' current hashes.

    Mirrors what the bundle writers hand `_write_staging_manifest`: only the
    listed (generated) files are registered, so edits to unlisted user code are
    ignored by `_staging_dir_has_local_edits`.
    """
    from dao_ai.apps.bundle import _sha256_file

    registry = {rel: _sha256_file(bundle_dir / rel) for rel in (files or [])}
    cli._write_staging_manifest(
        bundle_dir, is_default=True, fingerprint=fingerprint, registry=registry
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "command",
    [
        "workflow build",
        "agent build",
        # sync and up also have source-selection flags (sync can auto-build
        # and handles model_serving create_agent/deploy_agent).
        "agent sync",
        "agent up",
        "workflow sync",
        "workflow up",
    ],
)
class TestDevelopmentTriState:
    def test_development_true(self, command: str) -> None:
        opts = parse_args(_base(command) + ["--development"])
        assert opts.development is True

    def test_no_development_false(self, command: str) -> None:
        opts = parse_args(_base(command) + ["--no-development"])
        assert opts.development is False

    def test_omitted_none(self, command: str) -> None:
        opts = parse_args(_base(command))
        assert opts.development is None


@pytest.mark.unit
class TestNounVerbParsing:
    """Nested `<noun> <verb>` parses to (command=noun, subcommand=verb);
    removed top-level names are rejected by the parser.
    """

    @pytest.mark.parametrize("noun", ["agent", "workflow"])
    @pytest.mark.parametrize("verb", ["up", "build", "sync", "start", "down"])
    def test_nested_verbs_parse(self, noun: str, verb: str) -> None:
        opts = parse_args([noun, verb, "-c", "config.yaml"])
        assert opts.command == noun
        assert opts.subcommand == verb

    @pytest.mark.parametrize("noun", ["agent", "workflow"])
    @pytest.mark.parametrize("old_verb", ["generate", "deploy", "run", "destroy"])
    def test_old_verb_names_rejected(self, noun: str, old_verb: str) -> None:
        # Renamed to build/sync/start/down (breaking) — old names are invalid.
        with pytest.raises(SystemExit):
            parse_args([noun, old_verb, "-c", "config.yaml"])

    @pytest.mark.parametrize("noun", ["agent", "workflow"])
    def test_bare_noun_requires_verb(self, noun: str) -> None:
        with pytest.raises(SystemExit):
            parse_args([noun, "-c", "config.yaml"])

    def test_sync_verb_rejects_run(self) -> None:
        # sync has no --run; use `up` to build+sync+start in one command.
        with pytest.raises(SystemExit):
            parse_args(["agent", "sync", "-c", "c.yaml", "--run"])

    def test_sync_verb_rejects_direct(self) -> None:
        # --direct moved to `up`; sync is the pure push verb.
        with pytest.raises(SystemExit):
            parse_args(["agent", "sync", "-c", "c.yaml", "--direct"])

    def test_up_verb_parses(self) -> None:
        opts = parse_args(["agent", "up", "-c", "c.yaml", "--mode", "mcp"])
        assert opts.command == "agent"
        assert opts.subcommand == "up"
        assert opts.mode == "mcp"

    def test_up_verb_direct_flag(self) -> None:
        opts = parse_args(["agent", "up", "-c", "c.yaml", "--direct"])
        assert opts.direct is True

    def test_workflow_up_rejects_direct(self) -> None:
        # --direct is agent-only: workflow up runs the provisioning DAB job, which
        # requires the bundle, so a bundle-less SDK path is meaningless. The flag
        # is not registered on the workflow noun (was a silent no-op before).
        with pytest.raises(SystemExit):
            parse_args(["workflow", "up", "-c", "c.yaml", "--direct"])

    def test_up_verb_accepts_direct_with_model_serving(self) -> None:
        # --direct now means "SDK path, no bundle" for EVERY mode; for
        # model_serving it is the register + deploy-endpoint SDK path (the old
        # default). The combo parses (the former rejection was removed when the
        # bundle path became the model_serving default).
        opts = parse_args(
            ["agent", "up", "-c", "c.yaml", "--direct", "--mode", "model_serving"]
        )
        assert opts.direct is True
        assert opts.mode == "model_serving"

    def test_up_verb_accepts_direct_with_model_serving_alias(self) -> None:
        # The `ms` alias normalizes to model_serving and still parses with --direct.
        opts = parse_args(["agent", "up", "-c", "c.yaml", "--direct", "-m", "ms"])
        assert opts.direct is True
        assert opts.mode == "model_serving"

    @pytest.mark.parametrize(
        "argv",
        [
            ["deploy", "-c", "c.yaml"],
            ["generate-agent", "-c", "c.yaml"],
            ["generate-mcp", "-c", "c.yaml"],
            ["generate-workflow", "-c", "c.yaml"],
            ["generate-bundle", "-c", "c.yaml"],
            ["pipeline", "-c", "c.yaml"],
            ["tools", "-c", "c.yaml"],  # moved to `mcp tools`
        ],
    )
    def test_removed_commands_rejected(self, argv: list[str]) -> None:
        # argparse rejects unknown subcommands with SystemExit(2).
        with pytest.raises(SystemExit):
            parse_args(argv)


@pytest.mark.unit
class TestDefaultBundleDir:
    """Default bundle dir: $DAO_AI_BUNDLE_DIR base or `.dao-ai/bundle`, then
    `<kind>/<app>` appended (per-app isolation)."""

    def test_default_base_when_env_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DAO_AI_BUNDLE_DIR", raising=False)
        assert cli._default_bundle_dir("agent", "my_app") == Path(
            ".dao-ai/bundle/agent/my_app"
        )
        assert cli._default_bundle_dir("workflow", "My_App") == Path(
            ".dao-ai/bundle/workflow/my_app"
        )  # app name normalized

    def test_env_var_overrides_base(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DAO_AI_BUNDLE_DIR", "/tmp/custom-base")
        assert cli._default_bundle_dir("mcp", "my_app") == Path(
            "/tmp/custom-base/mcp/my_app"
        )

    def test_kind_app_structure_preserved_under_env_base(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DAO_AI_BUNDLE_DIR", "artifacts")
        for kind in ("workflow", "agent", "mcp"):
            assert cli._default_bundle_dir(kind, "app") == Path(f"artifacts/{kind}/app")

    def test_agent_mode_nests_under_app(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Agent bundles nest the serving mode under the app so apps/mcp/ms
        never clobber one another (mode subdir from _mode_subdir)."""
        monkeypatch.delenv("DAO_AI_BUNDLE_DIR", raising=False)
        assert cli._default_bundle_dir(
            "agent", "my_app", cli._mode_subdir("apps")
        ) == Path(".dao-ai/bundle/agent/my_app/apps")
        assert cli._default_bundle_dir(
            "agent", "my_app", cli._mode_subdir("mcp")
        ) == Path(".dao-ai/bundle/agent/my_app/mcp")
        assert cli._default_bundle_dir(
            "agent", "my_app", cli._mode_subdir("model_serving")
        ) == Path(".dao-ai/bundle/agent/my_app/ms")

    def test_workflow_dir_has_no_mode_subdir(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Workflow's artifact is mode-agnostic (mode is a runtime job var), so it
        # passes no mode subdir.
        monkeypatch.delenv("DAO_AI_BUNDLE_DIR", raising=False)
        assert cli._default_bundle_dir("workflow", "my_app") == Path(
            ".dao-ai/bundle/workflow/my_app"
        )

    def test_clean_guard_respects_env_base(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_clean_default_staging_dir wipes under the (env) base, never outside."""
        base = tmp_path / "mybase"
        monkeypatch.setenv("DAO_AI_BUNDLE_DIR", str(base))
        under = base / "agent" / "app"
        under.mkdir(parents=True)
        (under / "sentinel").write_text("x")
        _stamp_manifest(under, files=["sentinel"])  # mark as our output
        cli._clean_default_staging_dir(
            under, is_default=True, overwrite=False, noun="agent"
        )
        assert not under.exists(), "path under env base must be cleaned"

        outside = tmp_path / "user_dir"
        outside.mkdir()
        (outside / "keep").write_text("x")
        cli._clean_default_staging_dir(
            outside, is_default=True, overwrite=False, noun="agent"
        )
        assert (outside / "keep").exists(), "path outside base must be protected"


@pytest.mark.unit
class TestEditSafety:
    """A default staging dir with hand-edits is protected from re-generate."""

    def _staged(self, tmp_path: Path) -> Path:
        base = tmp_path / "base"
        d = base / "agent" / "app"
        d.mkdir(parents=True)
        (d / "app.yaml").write_text("orig\n")  # generated (tracked in `files`)
        (d / "src").mkdir()
        (d / "src" / "mine.py").write_text("print('mine')\n")  # user code (in `tracked`)
        # Only app.yaml is a generated file; src/mine.py is present at stamp time,
        # so it lands in `tracked` (not `files`) and is safe to edit later.
        _stamp_manifest(d, files=["app.yaml"])
        return d

    def test_untouched_dir_wiped_silently(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DAO_AI_BUNDLE_DIR", str(tmp_path / "base"))
        d = self._staged(tmp_path)
        cli._clean_default_staging_dir(
            d, is_default=True, overwrite=False, noun="agent"
        )
        assert not d.exists()

    def test_hand_edit_refuses_without_overwrite(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DAO_AI_BUNDLE_DIR", str(tmp_path / "base"))
        d = self._staged(tmp_path)
        # Change a registered file's CONTENT — hash no longer matches the manifest.
        (d / "app.yaml").write_text("edited\n")
        with pytest.raises(SystemExit) as exc:
            cli._clean_default_staging_dir(
                d, is_default=True, overwrite=False, noun="agent"
            )
        assert exc.value.code == 1
        assert d.exists(), "edited dir must be preserved"

    def test_overwrite_wipes_edited_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DAO_AI_BUNDLE_DIR", str(tmp_path / "base"))
        d = self._staged(tmp_path)
        (d / "app.yaml").write_text("edited\n")
        cli._clean_default_staging_dir(d, is_default=True, overwrite=True, noun="agent")
        assert not d.exists()

    def test_missing_marker_treated_as_edited(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DAO_AI_BUNDLE_DIR", str(tmp_path / "base"))
        d = tmp_path / "base" / "agent" / "app"
        d.mkdir(parents=True)
        (d / "app.yaml").write_text("user file\n")  # no manifest written
        with pytest.raises(SystemExit):
            cli._clean_default_staging_dir(
                d, is_default=True, overwrite=False, noun="agent"
            )
        assert d.exists()

    def test_editing_existing_user_code_does_not_trigger(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Editing user code that existed at generate time must never flag edits.

        Only dao-ai-generated files are hashed; a user's in-place edits to src/
        or code_paths (present in ``tracked`` but not ``files``) are invisible to
        edit-detection, so the dir regenerates cleanly.
        """
        monkeypatch.setenv("DAO_AI_BUNDLE_DIR", str(tmp_path / "base"))
        d = self._staged(tmp_path)  # src/mine.py present at stamp time
        (d / "src" / "mine.py").write_text(
            "print('edited')\n"
        )  # edit existing user code
        assert cli._staging_dir_has_local_edits(d) is False
        # A dir with no edited generated files + no strays is wiped silently.
        cli._clean_default_staging_dir(
            d, is_default=True, overwrite=False, noun="agent"
        )
        assert not d.exists()

    def test_user_added_stray_file_triggers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A NEW user file added after generate (e.g. a hand-dropped resources/
        jobs.yml) must be protected — the dir must not be silently wiped."""
        monkeypatch.setenv("DAO_AI_BUNDLE_DIR", str(tmp_path / "base"))
        d = self._staged(tmp_path)
        (d / "resources").mkdir()
        (d / "resources" / "jobs.yml").write_text(
            "resources: {}\n"
        )  # stray, not in `tracked`
        assert cli._staging_dir_has_local_edits(d) is True
        with pytest.raises(SystemExit):
            cli._clean_default_staging_dir(
                d, is_default=True, overwrite=False, noun="agent"
            )
        assert d.exists(), "stray user file must be preserved"

    def test_build_noise_not_treated_as_stray(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """__pycache__/.venv/dist churn appearing after generate is ignored."""
        monkeypatch.setenv("DAO_AI_BUNDLE_DIR", str(tmp_path / "base"))
        d = self._staged(tmp_path)
        (d / "__pycache__").mkdir()
        (d / "__pycache__" / "x.cpython-311.pyc").write_text("noise")
        (d / "src" / "app.pyc").write_text("noise")
        (d / "dist").mkdir()
        (d / "dist" / "app-0.1.0.whl").write_text("noise")
        assert cli._staging_dir_has_local_edits(d) is False

    def test_deleted_generated_file_triggers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DAO_AI_BUNDLE_DIR", str(tmp_path / "base"))
        d = self._staged(tmp_path)
        (d / "app.yaml").unlink()  # a tracked file vanished
        assert cli._staging_dir_has_local_edits(d) is True

    def test_legacy_marker_mtime_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pre-manifest dir (legacy marker, no manifest) uses the mtime heuristic."""
        import os

        monkeypatch.setenv("DAO_AI_BUNDLE_DIR", str(tmp_path / "base"))
        d = tmp_path / "base" / "agent" / "app"
        d.mkdir(parents=True)
        (d / "app.yaml").write_text("orig\n")
        (d / cli._STAGING_MARKER).write_text("legacy-fp")
        marker_mtime = (d / cli._STAGING_MARKER).stat().st_mtime
        # Untouched (all files older-or-equal to marker) -> no edits.
        os.utime(d / "app.yaml", (marker_mtime - 10, marker_mtime - 10))
        assert cli._staging_dir_has_local_edits(d) is False
        # A file newer than the marker -> edited.
        os.utime(d / "app.yaml", (marker_mtime + 10, marker_mtime + 10))
        assert cli._staging_dir_has_local_edits(d) is True

    def test_legacy_known_key_still_detects_stray(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A manifest staged before the ``known`` -> ``tracked`` rename still
        drives stray detection via the legacy key."""
        import yaml

        monkeypatch.setenv("DAO_AI_BUNDLE_DIR", str(tmp_path / "base"))
        d = tmp_path / "base" / "agent" / "app"
        d.mkdir(parents=True)
        (d / "app.yaml").write_text("orig\n")
        # Hand-write a legacy-shaped manifest that uses the old ``known`` key.
        (d / cli._STAGING_MANIFEST).write_text(
            yaml.safe_dump(
                {
                    "version": cli._MANIFEST_VERSION,
                    "config_fingerprint": "",
                    "files": {},
                    "known": ["app.yaml"],
                }
            )
        )
        assert cli._staging_dir_has_local_edits(d) is False
        (d / "stray.txt").write_text("added later\n")  # not in legacy `known`
        assert cli._staging_dir_has_local_edits(d) is True


@pytest.mark.unit
class TestConfigFingerprint:
    """The staging marker records a config fingerprint for staleness detection."""

    def _config(self, name: str = "greeter") -> AppConfig:
        from dao_ai.config import AgentModel, AppModel, InferenceEndpointModel

        return AppConfig(
            app=AppModel(
                name="fp-test",
                agents=[
                    AgentModel(
                        name=name,
                        description="says hi",
                        model=InferenceEndpointModel(name="databricks-gpt-5-4-mini"),
                    )
                ],
            )
        )

    def test_fingerprint_stable_for_identical_config(self) -> None:
        a = cli._config_fingerprint(self._config(), development=False)
        b = cli._config_fingerprint(self._config(), development=False)
        assert a == b

    def test_fingerprint_changes_with_config(self) -> None:
        a = cli._config_fingerprint(self._config(name="greeter"), development=False)
        b = cli._config_fingerprint(self._config(name="farewell"), development=False)
        assert a != b

    def test_fingerprint_changes_with_development_flag(self) -> None:
        a = cli._config_fingerprint(self._config(), development=False)
        b = cli._config_fingerprint(self._config(), development=True)
        assert a != b

    def test_manifest_records_fingerprint_and_files(self, tmp_path: Path) -> None:
        d = tmp_path / "base" / "agent" / "app"
        d.mkdir(parents=True)
        (d / "databricks.yaml").write_text("bundle: x\n")
        from dao_ai.apps.bundle import _sha256_file

        registry = {"databricks.yaml": _sha256_file(d / "databricks.yaml")}
        cli._write_staging_manifest(
            d, is_default=True, fingerprint="abc123", registry=registry
        )
        manifest = cli._read_staging_manifest(d)
        assert manifest is not None
        assert manifest["version"] == cli._MANIFEST_VERSION
        assert manifest["config_fingerprint"] == "abc123"
        assert manifest["files"] == registry
        # `tracked` snapshots every non-ignored file on disk at stamp time.
        assert "databricks.yaml" in manifest["tracked"]

    def test_manifest_retires_legacy_marker(self, tmp_path: Path) -> None:
        d = tmp_path / "base" / "agent" / "app"
        d.mkdir(parents=True)
        (d / cli._STAGING_MARKER).write_text("old-fp")
        cli._write_staging_manifest(
            d, is_default=True, fingerprint="new-fp", registry={}
        )
        assert not (d / cli._STAGING_MARKER).exists()
        assert (d / cli._STAGING_MANIFEST).exists()

    def test_malformed_manifest_files_does_not_crash(self, tmp_path: Path) -> None:
        """A manifest whose `files` is a non-dict must not raise on detection."""
        d = tmp_path / "base" / "agent" / "app"
        d.mkdir(parents=True)
        (d / cli._STAGING_MANIFEST).write_text(
            "version: 1\nconfig_fingerprint: fp\nfiles:\n  - a\n  - b\n"
        )
        # `files` is a list, not a dict -> treated as empty, no AttributeError.
        assert cli._staging_dir_has_local_edits(d) is False
        assert cli._staged_config_is_stale(d, "fp") is False


@pytest.mark.unit
class TestStagedConfigStaleness:
    """_staged_config_is_stale compares current config vs the stamped fingerprint."""

    def _marked(self, tmp_path: Path, fingerprint: str) -> Path:
        d = tmp_path / "base" / "agent" / "app"
        d.mkdir(parents=True)
        cli._write_staging_manifest(
            d, is_default=True, fingerprint=fingerprint, registry={}
        )
        return d

    def test_matching_fingerprint_not_stale(self, tmp_path: Path) -> None:
        d = self._marked(tmp_path, "fp-1")
        assert cli._staged_config_is_stale(d, "fp-1") is False

    def test_differing_fingerprint_is_stale(self, tmp_path: Path) -> None:
        d = self._marked(tmp_path, "fp-1")
        assert cli._staged_config_is_stale(d, "fp-2") is True

    def test_missing_manifest_not_stale(self, tmp_path: Path) -> None:
        d = tmp_path / "base" / "agent" / "app"
        d.mkdir(parents=True)  # no manifest (legacy / -o dir)
        assert cli._staged_config_is_stale(d, "fp-1") is False

    def test_empty_fingerprint_not_stale(self, tmp_path: Path) -> None:
        # A workflow bundle stamps an empty fingerprint; never treat it as stale.
        d = self._marked(tmp_path, "")
        assert cli._staged_config_is_stale(d, "fp-1") is False

    def test_legacy_marker_fingerprint_fallback(self, tmp_path: Path) -> None:
        # A pre-manifest dir still exposes its fingerprint via the legacy marker.
        d = tmp_path / "base" / "agent" / "app"
        d.mkdir(parents=True)
        (d / cli._STAGING_MARKER).write_text("legacy-fp")
        assert cli._staged_config_is_stale(d, "legacy-fp") is False
        assert cli._staged_config_is_stale(d, "other-fp") is True


@pytest.mark.unit
class TestPipelineEmitsDevelopmentVar:
    """run_databricks_command forwards development as a bundle --var."""

    @pytest.mark.parametrize(
        "development,expected",
        [
            (True, "development=true"),
            (False, "development=false"),
            (None, "development=auto"),
        ],
    )
    def test_var_emitted(self, development: bool | None, expected: str) -> None:
        captured: dict[str, str] = {}

        def _capture(msg: str) -> None:
            if msg.startswith("[DRY RUN]"):
                captured["cmd"] = msg

        # config=None skips the config/template machinery; mock cloud detection
        # so we deterministically reach the var-emission + dry-run return.
        with (
            patch.object(cli, "detect_cloud_provider", return_value="aws"),
            patch.object(cli.logger, "info", side_effect=_capture),
        ):
            run_databricks_command(
                ["bundle", "deploy"],
                config=None,
                dry_run=True,
                development=development,
            )

        assert expected in captured["cmd"]

    def test_default_is_auto(self) -> None:
        captured: dict[str, str] = {}

        def _capture(msg: str) -> None:
            if msg.startswith("[DRY RUN]"):
                captured["cmd"] = msg

        with (
            patch.object(cli, "detect_cloud_provider", return_value="aws"),
            patch.object(cli.logger, "info", side_effect=_capture),
        ):
            run_databricks_command(["bundle", "deploy"], config=None, dry_run=True)

        assert "development=auto" in captured["cmd"]

    def test_command_none_stages_without_executing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`generate-workflow` with no action flag stages the bundle but must
        NOT shell out to `databricks bundle` (regression: it used to no-op)."""
        cfg = tmp_path / "c.yaml"
        cfg.write_text(
            "resources:\n  models:\n    m: &m\n      name: databricks-gpt-5-4-mini\n"
            "agents:\n  g: &g\n    name: g\n    description: d\n    model: *m\n"
            "    prompt: p\n"
            "app:\n  name: stage_only_app\n  agents:\n    - *g\n"
        )
        staged: dict[str, object] = {}
        monkeypatch.setattr(cli, "_apply_profile_context", lambda p: None)
        monkeypatch.setattr(cli, "detect_cloud_provider", lambda p: "aws")
        monkeypatch.setattr(
            "dao_ai.pipeline.bundle.write_pipeline_bundle",
            lambda *a, **k: staged.setdefault("wrote", True),
        )
        exec_mock = patch.object(cli, "_exec_bundle_command").start()

        run_databricks_command(
            None,
            config=str(cfg),
            staging_dir=str(tmp_path / "out"),
            # Published mode: the stubbed bundle-write stages no wheel; this
            # test asserts staging + no-exec, not local-wheel resolution.
            development=False,
        )
        patch.stopall()

        assert staged.get("wrote") is True, "bundle must be staged"
        exec_mock.assert_not_called()

    def test_stage_false_skips_write_and_execs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`workflow deploy` (stage=False) runs the bundle verb WITHOUT
        re-staging: write_pipeline_bundle must not be called, and the existing
        staged databricks.yaml is executed against in place."""
        cfg = tmp_path / "c.yaml"
        cfg.write_text(
            "resources:\n  models:\n    m: &m\n      name: databricks-gpt-5-4-mini\n"
            "agents:\n  g: &g\n    name: g\n    description: d\n    model: *m\n"
            "    prompt: p\n"
            "app:\n  name: stage_only_app\n  agents:\n    - *g\n"
        )
        out = tmp_path / "out"
        out.mkdir()
        (out / "databricks.yaml").write_text("bundle: {}\n")  # pretend pre-staged

        wrote: dict[str, object] = {}
        monkeypatch.setattr(cli, "_apply_profile_context", lambda p: None)
        monkeypatch.setattr(cli, "detect_cloud_provider", lambda p: "aws")
        monkeypatch.setattr(
            "dao_ai.pipeline.bundle.write_pipeline_bundle",
            lambda *a, **k: wrote.setdefault("wrote", True),
        )
        exec_mock = patch.object(cli, "_exec_bundle_command").start()

        run_databricks_command(
            ["bundle", "deploy"],
            config=str(cfg),
            staging_dir=str(out),
            development=False,
            stage=False,
        )
        patch.stopall()

        assert "wrote" not in wrote, "stage=False must NOT regenerate the bundle"
        exec_mock.assert_called_once()
        assert exec_mock.call_args.kwargs["cwd"] == out

    def test_stage_false_errors_when_not_staged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`workflow deploy` against an unstaged dir exits with guidance."""
        cfg = tmp_path / "c.yaml"
        cfg.write_text(
            "resources:\n  models:\n    m: &m\n      name: databricks-gpt-5-4-mini\n"
            "agents:\n  g: &g\n    name: g\n    description: d\n    model: *m\n"
            "    prompt: p\n"
            "app:\n  name: stage_only_app\n  agents:\n    - *g\n"
        )
        monkeypatch.setattr(cli, "_apply_profile_context", lambda p: None)
        monkeypatch.setattr(cli, "detect_cloud_provider", lambda p: "aws")
        with pytest.raises(SystemExit) as exc:
            run_databricks_command(
                ["bundle", "deploy"],
                config=str(cfg),
                staging_dir=str(tmp_path / "nonexistent"),
                development=False,
                stage=False,
            )
        assert exc.value.code == 1


@pytest.mark.unit
class TestConfigVarsForwardedOnlyIfDeclared:
    """Regression: a dao-ai --param is forwarded to `databricks bundle` as --var
    ONLY when the staged databricks.yaml declares a matching bundle variable.

    A dao-ai config parameter (e.g. genie_space_id) is baked into the staged
    config, not declared as a bundle variable — forwarding it made
    `databricks bundle deploy` hard-fail "variable X has not been defined".
    """

    _CFG = (
        "resources:\n  models:\n    m: &m\n      name: databricks-gpt-5-4-mini\n"
        "agents:\n  g: &g\n    name: g\n    description: d\n    model: *m\n"
        "    prompt: p\n"
        "app:\n  name: stage_only_app\n  agents:\n    - *g\n"
    )

    def _run_and_capture(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        declared_vars: list[str],
        config_vars: dict[str, str],
    ) -> list[str]:
        cfg = tmp_path / "c.yaml"
        cfg.write_text(self._CFG)
        out = tmp_path / "out"
        out.mkdir()
        # Pre-staged bundle whose databricks.yaml declares `declared_vars`.
        variables_block = "".join(
            f"  {name}:\n    description: d\n" for name in declared_vars
        )
        (out / "databricks.yaml").write_text(f"variables:\n{variables_block}")

        captured: dict[str, list[str]] = {}

        def _fake_exec(command, *, extra_vars=None, **kwargs):
            captured["extra_vars"] = extra_vars or []

        monkeypatch.setattr(cli, "_apply_profile_context", lambda p: None)
        monkeypatch.setattr(cli, "detect_cloud_provider", lambda p: "aws")
        monkeypatch.setattr(cli, "_exec_bundle_command", _fake_exec)

        run_databricks_command(
            ["bundle", "deploy"],
            config=str(cfg),
            staging_dir=str(out),
            development=False,
            stage=False,
            config_vars=config_vars,
        )
        return captured["extra_vars"]

    def test_undeclared_param_not_forwarded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        extra = self._run_and_capture(
            tmp_path,
            monkeypatch,
            declared_vars=["config_path", "mode", "development", "dao_ai_dep"],
            config_vars={"genie_space_id": "01f153"},
        )
        joined = " ".join(extra)
        assert "genie_space_id" not in joined, (
            "an undeclared dao-ai param must NOT be forwarded as a bundle --var"
        )

    def test_declared_param_is_forwarded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        extra = self._run_and_capture(
            tmp_path,
            monkeypatch,
            declared_vars=[
                "config_path",
                "mode",
                "development",
                "dao_ai_dep",
                "catalog",
            ],
            config_vars={"catalog": "main", "genie_space_id": "01f153"},
        )
        joined = " ".join(extra)
        assert '--var="catalog=main"' in joined, "a declared overlap must be forwarded"
        assert "genie_space_id" not in joined, "the undeclared one is still dropped"


@pytest.mark.unit
class TestDeclaredBundleVariables:
    """_declared_bundle_variables reads the staged databricks.yaml safely."""

    def test_reads_declared_names(self, tmp_path: Path) -> None:
        (tmp_path / "databricks.yaml").write_text(
            "variables:\n  a:\n    description: x\n  b:\n    default: y\n"
        )
        assert cli._declared_bundle_variables(tmp_path) == {"a", "b"}

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert cli._declared_bundle_variables(tmp_path) == set()

    def test_no_variables_block_returns_empty(self, tmp_path: Path) -> None:
        (tmp_path / "databricks.yaml").write_text("bundle:\n  name: x\n")
        assert cli._declared_bundle_variables(tmp_path) == set()

    def test_malformed_yaml_returns_empty(self, tmp_path: Path) -> None:
        (tmp_path / "databricks.yaml").write_text("variables: [not, a, mapping\n")
        assert cli._declared_bundle_variables(tmp_path) == set()


@pytest.mark.unit
class TestBuildIsStageOnly:
    """``build`` is a pure staging verb — ``--deploy``/``--run``/``--destroy``
    are rejected by the parser (use ``up`` for orchestration).
    """

    @pytest.mark.parametrize("noun", ["agent", "workflow"])
    @pytest.mark.parametrize("flag", ["--deploy", "--run", "--destroy"])
    def test_build_rejects_action_flags(self, noun: str, flag: str) -> None:
        with pytest.raises(SystemExit):
            parse_args([noun, "build", "-c", "config.yaml", flag])

    @pytest.mark.parametrize("noun", ["agent", "workflow"])
    def test_build_without_action_flags_parses(self, noun: str) -> None:
        opts = parse_args([noun, "build", "-c", "config.yaml"])
        assert opts.subcommand == "build"


def _app_config(*, trace: bool) -> AppConfig:
    """Minimal AppConfig stub for deploy_app_bundle (needs app.name + trace_location)."""

    class _App:
        name = "my_app"
        trace_location = object() if trace else None

    return AppConfig.model_construct(app=_App())


@pytest.mark.unit
class TestModeChoices:
    def test_sync_accepts_all_three(self) -> None:
        for m in ("model_serving", "apps", "mcp"):
            assert parse_args(["agent", "sync", "-c", "c.yaml", "--mode", m]).mode == m

    def test_build_accepts_model_serving(self) -> None:
        # Agent build now stages a bundle for every mode, including the thin
        # model_serving Job bundle.
        assert (
            parse_args(
                ["agent", "build", "-c", "c.yaml", "--mode", "model_serving"]
            ).mode
            == "model_serving"
        )

    def test_build_accepts_apps_and_mcp(self) -> None:
        for m in ("apps", "mcp"):
            assert parse_args(["agent", "build", "-c", "c.yaml", "--mode", m]).mode == m

    @pytest.mark.parametrize("verb", ["build", "start", "down"])
    def test_agent_verbs_accept_model_serving(self, verb: str) -> None:
        # Every agent verb now resolves a model_serving staging dir (agent/<app>/ms).
        assert (
            parse_args(["agent", verb, "-c", "c.yaml", "--mode", "model_serving"]).mode
            == "model_serving"
        )

    def test_mode_defaults_to_apps(self) -> None:
        assert parse_args(["agent", "sync", "-c", "c.yaml"]).mode == "apps"

    def test_short_alias_m(self) -> None:
        assert parse_args(["agent", "sync", "-c", "c.yaml", "-m", "mcp"]).mode == "mcp"

    @pytest.mark.parametrize("alias", ["ms", "model-serving", "model_serving"])
    def test_model_serving_aliases_normalize_on_sync(self, alias: str) -> None:
        # ms / model-serving normalize to the canonical model_serving value.
        assert (
            parse_args(["agent", "sync", "-c", "c.yaml", "-m", alias]).mode
            == "model_serving"
        )

    @pytest.mark.parametrize("alias", ["ms", "model-serving", "model_serving"])
    def test_model_serving_aliases_normalize_on_build(self, alias: str) -> None:
        # model_serving (and its aliases) are now valid on agent build and
        # normalize to the canonical value.
        assert (
            parse_args(["agent", "build", "-c", "c.yaml", "--mode", alias]).mode
            == "model_serving"
        )


@pytest.mark.unit
class TestWorkflowModeChoices:
    """Every workflow verb accepts the same three modes as every agent verb —
    a uniform surface (ADR §2.7: workflow deploys the agent in any valid mode,
    incl. model_serving, forwarding `--mode` to the deploy-agent job step). No
    verb may reject a mode another verb on the same noun accepts.
    """

    @pytest.mark.parametrize("verb", ["up", "build", "sync", "start", "down"])
    @pytest.mark.parametrize("mode", ["model_serving", "apps", "mcp"])
    def test_workflow_verb_accepts_all_modes(self, verb: str, mode: str) -> None:
        assert (
            parse_args(["workflow", verb, "-c", "c.yaml", "--mode", mode]).mode == mode
        )

    @pytest.mark.parametrize("verb", ["up", "build", "sync", "start", "down"])
    @pytest.mark.parametrize("mode", ["model_serving", "apps", "mcp"])
    def test_agent_verb_accepts_all_modes(self, verb: str, mode: str) -> None:
        # Agent parity — the two nouns expose identical mode choices per verb.
        assert parse_args(["agent", verb, "-c", "c.yaml", "--mode", mode]).mode == mode

    @pytest.mark.parametrize("noun", ["agent", "workflow"])
    @pytest.mark.parametrize("verb", ["up", "build", "sync", "start", "down"])
    def test_ms_alias_normalizes_on_every_verb(self, noun: str, verb: str) -> None:
        # `ms` normalizes to model_serving uniformly across nouns + verbs.
        assert (
            parse_args([noun, verb, "-c", "c.yaml", "-m", "ms"]).mode == "model_serving"
        )


@pytest.mark.unit
class TestMcpNounRemoved:
    """The `mcp` noun has been removed; its modes are now routed via `agent --mode mcp`."""

    def test_mcp_noun_removed(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["mcp", "generate", "-c", "c.yaml"])


@pytest.mark.unit
class TestMonitorScorersRegroup:
    """`monitor <action>` is regrouped under `monitor scorers <action>` (breaking)."""

    @pytest.mark.parametrize("action", ["enable", "status", "disable"])
    def test_scorers_subcommand_parses(self, action: str) -> None:
        opts = parse_args(["monitor", "scorers", action, "-c", "c.yaml"])
        assert opts.command == "monitor"
        assert opts.subcommand == "scorers"
        assert opts.action == action

    @pytest.mark.parametrize("action", ["enable", "status", "disable"])
    def test_flat_monitor_action_rejected(self, action: str) -> None:
        # The old flat form `monitor enable` is gone; the action is now a
        # positional on the `scorers` sub-verb, so `enable` is an unknown verb.
        with pytest.raises(SystemExit):
            parse_args(["monitor", action, "-c", "c.yaml"])

    def test_bare_monitor_requires_subcommand(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["monitor", "-c", "c.yaml"])

    def test_scorers_rejects_unknown_action(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["monitor", "scorers", "bogus", "-c", "c.yaml"])


_MINIMAL_CONFIG_YAML = (
    "resources:\n  models:\n    m: &m\n      name: databricks-gpt-5-4-mini\n"
    "agents:\n  g: &g\n    name: g\n    description: d\n    model: *m\n"
    "    prompt: p\n"
    "app:\n  name: test_app\n  agents:\n    - *g\n"
)


@pytest.mark.unit
class TestAgentModeWriterSelection:
    """handle_agent_command selects the correct bundle writer based on --mode."""

    def _write_config(self, tmp_path: Path) -> Path:
        cfg = tmp_path / "c.yaml"
        cfg.write_text(_MINIMAL_CONFIG_YAML)
        return cfg

    def test_agent_generate_mcp_uses_mcp_writer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--mode mcp must call write_mcp_bundle, not write_bundle."""
        called: dict[str, bool] = {}
        monkeypatch.setattr(
            "dao_ai.mcp.generate.write_mcp_bundle",
            lambda *a, **k: called.setdefault("mcp", True),
        )
        monkeypatch.setattr(
            "dao_ai.apps.bundle.write_bundle",
            lambda *a, **k: called.setdefault("apps", True),
        )
        monkeypatch.setattr(cli, "_apply_profile_context", lambda p: None)
        monkeypatch.setattr(cli, "detect_cloud_provider", lambda p: "aws")

        cfg = self._write_config(tmp_path)
        opts = parse_args(
            [
                "agent",
                "build",
                "-c",
                str(cfg),
                "-s",
                str(tmp_path / "out"),
                "--mode",
                "mcp",
            ]
        )
        cli.handle_agent_command(opts)

        assert called == {"mcp": True}, (
            f"Expected only write_mcp_bundle to be called; got called={called}"
        )

    def test_agent_generate_apps_uses_default_writer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mode omitted (defaults to apps) must call write_bundle, not write_mcp_bundle."""
        called: dict[str, bool] = {}
        monkeypatch.setattr(
            "dao_ai.mcp.generate.write_mcp_bundle",
            lambda *a, **k: called.setdefault("mcp", True),
        )
        monkeypatch.setattr(
            "dao_ai.apps.bundle.write_bundle",
            lambda *a, **k: called.setdefault("apps", True),
        )
        monkeypatch.setattr(cli, "_apply_profile_context", lambda p: None)
        monkeypatch.setattr(cli, "detect_cloud_provider", lambda p: "aws")

        cfg = self._write_config(tmp_path)
        opts = parse_args(
            ["agent", "build", "-c", str(cfg), "-s", str(tmp_path / "out")]
        )
        cli.handle_agent_command(opts)

        assert called == {"apps": True}, (
            f"Expected only write_bundle to be called; got called={called}"
        )

    def test_agent_deploy_mcp_uses_mcp_staging_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """agent deploy --mode mcp resolves the agent/<app>/mcp staging dir."""
        cfg = self._write_config(tmp_path)
        # Capture the mode_subdir _resolve_bundle_dir is asked for.
        resolved: dict[str, Path] = {}

        def mock_resolve_bundle_dir(
            kind: str,
            config: object,
            staging_dir: str | None,
            mode_subdir: str | None = None,
        ) -> tuple[Path, bool]:
            staged = tmp_path / "staged" / kind / "test_app" / (mode_subdir or "")
            staged.mkdir(parents=True, exist_ok=True)
            (staged / "databricks.yaml").write_text("bundle: {}\n")
            resolved[mode_subdir or "<none>"] = staged
            return staged, staging_dir is None

        monkeypatch.setattr(cli, "_apply_profile_context", lambda p: None)

        with patch.object(
            cli, "_resolve_bundle_dir", side_effect=mock_resolve_bundle_dir
        ):
            with patch.object(cli, "deploy_app_bundle"):
                opts = parse_args(["agent", "sync", "-c", str(cfg), "--mode", "mcp"])
                cli.handle_agent_command(opts)

        assert "mcp" in resolved, (
            "deploy must resolve the mcp mode subdir when --mode mcp"
        )
        assert "/mcp" in str(resolved["mcp"]), (
            f"Expected mcp staging dir path to contain /mcp, got {resolved['mcp']}"
        )

    def test_agent_deploy_model_serving_uses_ms_writer_and_job_driver(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """agent deploy --mode model_serving stages via the MS writer and runs
        the Job driver (_run_ms_job_bundle), NOT the App driver."""
        called: dict[str, bool] = {}
        monkeypatch.setattr(
            "dao_ai.pipeline.bundle.write_model_serving_agent_bundle",
            lambda *a, **k: called.setdefault("ms_writer", True) or {},
        )
        monkeypatch.setattr(cli, "_apply_profile_context", lambda p: None)
        monkeypatch.setattr(cli, "detect_cloud_provider", lambda p: "aws")
        monkeypatch.setattr(AppConfig, "_resolve_all_resources", lambda self: None)
        monkeypatch.setattr(
            AppConfig, "assert_provided_params_satisfied", lambda self: None
        )

        cfg = self._write_config(tmp_path)
        with patch.object(cli, "_run_ms_job_bundle") as job_driver:
            with patch.object(cli, "deploy_app_bundle") as app_driver:
                opts = parse_args(
                    [
                        "agent",
                        "sync",
                        "-c",
                        str(cfg),
                        "-s",
                        str(tmp_path / "out"),
                        "--mode",
                        "model_serving",
                    ]
                )
                cli.handle_agent_command(opts)

        assert called.get("ms_writer"), "MS writer must stage the bundle"
        job_driver.assert_called_once()
        app_driver.assert_not_called()


@pytest.mark.unit
class TestDeployRestagesOnConfigDrift:
    """agent deploy re-stages an already-staged bundle when the source config drifts."""

    def _write_config(self, tmp_path: Path) -> Path:
        cfg = tmp_path / "c.yaml"
        cfg.write_text(_MINIMAL_CONFIG_YAML)
        return cfg

    def _run_deploy(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        marker_fingerprint: str | None,
        has_edits: bool = False,
        overwrite: bool = False,
    ) -> bool:
        """Drive `agent deploy` against a pre-staged dir. Returns True if re-staged.

        ``marker_fingerprint`` seeds the staged manifest's config_fingerprint:
        None omits the manifest (legacy dir), "" is a workflow-style empty
        fingerprint, a string is a recorded hash. ``has_edits`` simulates
        hand-edits; ``overwrite`` passes --overwrite.
        """
        cfg = self._write_config(tmp_path)
        staged = tmp_path / "staged" / "agent" / "test_app"
        staged.mkdir(parents=True)
        (staged / "databricks.yaml").write_text("bundle: {}\n")
        if marker_fingerprint is not None:
            cli._write_staging_manifest(
                staged,
                is_default=True,
                fingerprint=marker_fingerprint,
                registry={},
            )

        restaged: dict[str, bool] = {}
        monkeypatch.setattr(cli, "_apply_profile_context", lambda p: None)
        monkeypatch.setattr(
            cli,
            "_resolve_bundle_dir",
            lambda kind, config, staging_dir, mode_subdir=None: (staged, True),
        )
        monkeypatch.setattr(cli, "_staging_dir_has_local_edits", lambda d: has_edits)
        monkeypatch.setattr(cli, "deploy_app_bundle", lambda *a, **k: None)
        monkeypatch.setattr(AppConfig, "_resolve_all_resources", lambda self: None)
        monkeypatch.setattr(
            cli,
            "_stage_app_bundle",
            lambda *a, **k: restaged.setdefault("staged", True),
        )

        argv = ["agent", "sync", "-c", str(cfg)]
        if overwrite:
            argv.append("--overwrite")
        opts = parse_args(argv)
        cli.handle_agent_command(opts)
        return restaged.get("staged", False)

    def test_restages_when_manifest_fingerprint_differs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A fingerprint that cannot match the current config -> stale -> re-stage.
        assert (
            self._run_deploy(tmp_path, monkeypatch, marker_fingerprint="stale-hash")
            is True
        )

    def test_deploys_in_place_when_fingerprint_matches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Stamp the marker with the CURRENT config's real fingerprint -> not stale.
        # Resolve development exactly as the deploy path does (options.development
        # is None here -> auto-detect), so the seeded hash matches.
        from dao_ai.utils import resolve_use_local_source

        cfg = self._write_config(tmp_path)
        config = AppConfig.from_file(str(cfg), initialize=False)
        current_fp = cli._config_fingerprint(
            config, development=resolve_use_local_source(None)
        )
        assert (
            self._run_deploy(tmp_path, monkeypatch, marker_fingerprint=current_fp)
            is False
        )

    def test_deploys_in_place_when_manifest_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Staged dir with no manifest -> no recorded fingerprint -> in place.
        assert self._run_deploy(tmp_path, monkeypatch, marker_fingerprint=None) is False

    def test_stale_with_edits_deploys_in_place(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Config drifted but the dir has hand-edits and no --overwrite -> in place.
        assert (
            self._run_deploy(
                tmp_path, monkeypatch, marker_fingerprint="stale", has_edits=True
            )
            is False
        )

    def test_stale_with_edits_and_overwrite_regenerates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # --overwrite forces regen even over hand-edits when the config drifted.
        assert (
            self._run_deploy(
                tmp_path,
                monkeypatch,
                marker_fingerprint="stale",
                has_edits=True,
                overwrite=True,
            )
            is True
        )


@pytest.mark.unit
class TestDeployAppBundle:
    """Shared App deploy driver: deploy -> (link+grant) -> run, by app name."""

    @pytest.fixture(autouse=True)
    def _no_link_lookup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Keep tests hermetic: don't hit the Apps API for the URL print.
        monkeypatch.setattr(cli, "_print_app_link", lambda _n: None)

    def test_deploy_then_run_order_and_target(self, tmp_path: Path) -> None:
        cfg = _app_config(trace=False)
        with (
            patch.object(cli, "_exec_bundle_command") as exec_cmd,
            patch.object(cli, "_link_and_grant_trace") as link,
        ):
            deploy_app_bundle(
                cfg,
                staging_dir=tmp_path,
                deploy=True,
                run=True,
                destroy=False,
                profile="fevm",
            )
        # deploy first, then run <app-name>, both --target dev, cwd == bundle dir.
        assert exec_cmd.call_args_list == [
            call(
                ["bundle", "deploy"],
                profile="fevm",
                target="dev",
                cwd=tmp_path,
                dry_run=False,
            ),
            call(
                ["bundle", "run", "my-app"],
                profile="fevm",
                target="dev",
                cwd=tmp_path,
                dry_run=False,
            ),
        ]
        # No trace_location -> link/grant is still called but no-ops internally.
        link.assert_called_once()

    def test_link_grant_runs_between_deploy_and_run(self, tmp_path: Path) -> None:
        """Ordering guarantee: link must happen after deploy, before run."""
        cfg = _app_config(trace=True)
        events: list[str] = []
        with (
            patch.object(
                cli,
                "_exec_bundle_command",
                side_effect=lambda command, **_k: events.append(command[1]),
            ),
            patch.object(
                cli,
                "_link_and_grant_trace",
                side_effect=lambda *_a, **_k: events.append("link"),
            ),
        ):
            deploy_app_bundle(
                cfg,
                staging_dir=tmp_path,
                deploy=True,
                run=True,
                destroy=False,
                profile=None,
            )
        assert events == ["deploy", "link", "run"]

    def test_destroy_skips_deploy_and_link(self, tmp_path: Path) -> None:
        cfg = _app_config(trace=True)
        with (
            patch.object(cli, "_exec_bundle_command") as exec_cmd,
            patch.object(cli, "_link_and_grant_trace") as link,
        ):
            deploy_app_bundle(
                cfg,
                staging_dir=tmp_path,
                deploy=False,
                run=False,
                destroy=True,
                profile=None,
            )
        exec_cmd.assert_called_once_with(
            ["bundle", "destroy", "--auto-approve"],
            profile=None,
            target="dev",
            cwd=tmp_path,
            dry_run=False,
        )
        link.assert_not_called()

    def test_run_only_skips_deploy_and_link(self, tmp_path: Path) -> None:
        cfg = _app_config(trace=True)
        with (
            patch.object(cli, "_exec_bundle_command") as exec_cmd,
            patch.object(cli, "_link_and_grant_trace") as link,
        ):
            deploy_app_bundle(
                cfg,
                staging_dir=tmp_path,
                deploy=False,
                run=True,
                destroy=False,
                profile=None,
            )
        exec_cmd.assert_called_once_with(
            ["bundle", "run", "my-app"],
            profile=None,
            target="dev",
            cwd=tmp_path,
            dry_run=False,
        )
        link.assert_not_called()


@pytest.mark.unit
class TestDeleteServingEndpoint:
    """`agent down --mode model_serving` deletes the (non-DAB) serving endpoint
    after `bundle destroy`, so the endpoint isn't left orphaned/billing.
    """

    @staticmethod
    def _ms_config(endpoint: str | None = "my_ep") -> AppConfig:
        class _App:
            name = "my_app"
            endpoint_name = endpoint

        return AppConfig.model_construct(app=_App())

    def test_deletes_endpoint_by_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        deleted: list[str] = []

        class _WC:
            class serving_endpoints:
                @staticmethod
                def delete(name: str) -> None:
                    deleted.append(name)

        import databricks.sdk as sdk

        monkeypatch.setattr(sdk, "WorkspaceClient", lambda *a, **k: _WC())
        cli._delete_serving_endpoint(
            self._ms_config("my_ep"), profile=None, dry_run=False
        )
        assert deleted == ["my_ep"]

    def test_dry_run_does_not_delete(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called: list[str] = []

        class _WC:
            class serving_endpoints:
                @staticmethod
                def delete(name: str) -> None:
                    called.append(name)

        import databricks.sdk as sdk

        monkeypatch.setattr(sdk, "WorkspaceClient", lambda *a, **k: _WC())
        cli._delete_serving_endpoint(
            self._ms_config("my_ep"), profile=None, dry_run=True
        )
        assert called == [], "dry-run must not delete the endpoint"

    def test_missing_endpoint_is_not_an_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from databricks.sdk.errors import NotFound

        class _WC:
            class serving_endpoints:
                @staticmethod
                def delete(name: str) -> None:
                    raise NotFound("gone")

        import databricks.sdk as sdk

        monkeypatch.setattr(sdk, "WorkspaceClient", lambda *a, **k: _WC())
        # Must not raise — an already-deleted endpoint is fine.
        cli._delete_serving_endpoint(
            self._ms_config("my_ep"), profile=None, dry_run=False
        )

    def test_ms_down_calls_endpoint_delete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_run_ms_job_bundle(destroy=True) runs bundle destroy THEN deletes the endpoint."""
        order: list[str] = []
        monkeypatch.setattr(
            cli, "_exec_job_bundle", lambda **k: order.append("bundle_destroy")
        )
        monkeypatch.setattr(
            cli,
            "_delete_serving_endpoint",
            lambda config, *, profile, dry_run: order.append("endpoint_delete"),
        )
        monkeypatch.setattr(cli, "_resolve_job_dao_ai_dep", lambda *a, **k: "dao-ai")
        monkeypatch.setattr(cli, "detect_cloud_provider", lambda p: "aws")
        cli._run_ms_job_bundle(
            self._ms_config("my_ep"),
            staging_dir=tmp_path,
            deploy=False,
            run=False,
            destroy=True,
            dry_run=False,
            profile=None,
            development=None,
            target=None,
            cloud=None,
            config_vars={},
        )
        assert order == ["bundle_destroy", "endpoint_delete"], order


@pytest.mark.unit
class TestLinkAndGrantTrace:
    """_link_and_grant_trace no-ops without trace_location, links with it."""

    def test_noop_without_trace_location(self) -> None:
        cfg = _app_config(trace=False)
        with patch.object(cli, "_resolve_experiment_id_for_link") as resolve:
            cli._link_and_grant_trace(cfg, dry_run=False)
        resolve.assert_not_called()

    def test_dry_run_does_not_resolve_or_link(self) -> None:
        cfg = _app_config(trace=True)
        with patch.object(cli, "_resolve_experiment_id_for_link") as resolve:
            cli._link_and_grant_trace(cfg, dry_run=True)
        resolve.assert_not_called()


@pytest.mark.unit
class TestDeployLinks:
    """Post-deploy link printing for app and workflow-job bundles."""

    def test_app_link_printed(self, capsys: pytest.CaptureFixture[str]) -> None:
        class _App:
            url = "https://my-app-123.aws.databricksapps.com"

        class _WC:
            class apps:
                @staticmethod
                def get(name: str) -> object:
                    return _App()

        with patch("databricks.sdk.WorkspaceClient", return_value=_WC()):
            cli._print_app_link("my-app")
        assert "https://my-app-123.aws.databricksapps.com" in capsys.readouterr().out

    def test_app_link_best_effort_on_error(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with patch("databricks.sdk.WorkspaceClient", side_effect=RuntimeError("boom")):
            cli._print_app_link("my-app")  # must not raise
        assert "databricksapps" not in capsys.readouterr().out

    def test_endpoint_link_printed(self, capsys: pytest.CaptureFixture[str]) -> None:
        class _Cfg:
            host = "https://ws.cloud.databricks.com/"

        class _WC:
            config = _Cfg()

        with patch("databricks.sdk.WorkspaceClient", return_value=_WC()):
            cli._print_endpoint_link("my_endpoint")
        out = capsys.readouterr().out
        assert "https://ws.cloud.databricks.com/ml/endpoints/my_endpoint" in out

    def test_endpoint_link_best_effort_on_error(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with patch("databricks.sdk.WorkspaceClient", side_effect=RuntimeError("x")):
            cli._print_endpoint_link("my_endpoint")  # must not raise
        assert "ml/endpoints" not in capsys.readouterr().out

    def test_job_link_printed_from_summary(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import json
        import subprocess as _sp

        summary = {
            "resources": {"jobs": {"deploy_job": {"url": "https://ws/jobs/42?w=1"}}}
        }

        class _R:
            returncode = 0
            stdout = json.dumps(summary)

        with patch.object(_sp, "run", return_value=_R()):
            cli._print_job_link(
                tmp_path, profile="fevm", target="app-aws", extra_vars=[]
            )
        assert "https://ws/jobs/42?w=1" in capsys.readouterr().out

    def test_job_link_best_effort_on_summary_failure(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import subprocess as _sp

        class _R:
            returncode = 1
            stdout = ""

        with patch.object(_sp, "run", return_value=_R()):
            cli._print_job_link(tmp_path, profile=None, target=None)  # no raise
        assert "Job URL" not in capsys.readouterr().out


@pytest.mark.unit
class TestNounVerbDispatch:
    """main() routes nouns to per-noun handlers; aliases share the generate path;
    standalone deploy/run act on the staged dir without regenerating."""

    def test_generate_verb_routes_to_bundle(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`agent generate` reaches _generate_app_bundle with the correct command
        and subcommand."""
        seen: list[tuple[str, str]] = []
        monkeypatch.setattr(
            cli,
            "_generate_app_bundle",
            lambda opts, **kw: seen.append((opts.command, opts.subcommand)),
        )
        monkeypatch.setattr(cli, "setup_logging", lambda v: None)

        monkeypatch.setattr("sys.argv", ["dao-ai", "agent", "build", "-c", "c.yaml"])
        cli.main()
        assert seen == [("agent", "build")]

    def test_deploy_verb_no_regenerate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`agent deploy` on a staged dir calls deploy_app_bundle and never a
        bundle writer; the sentinel on disk survives."""
        cfg = tmp_path / "c.yaml"
        cfg.write_text(
            "resources:\n  models:\n    m: &m\n      name: databricks-gpt-5-4-mini\n"
            "agents:\n  g: &g\n    name: g\n    description: d\n    model: *m\n"
            "    prompt: p\n"
            "app:\n  name: my_app\n  agents:\n    - *g\n"
        )
        out = tmp_path / "staged"
        out.mkdir()
        (out / "databricks.yaml").write_text("bundle: {}\n")
        (out / "sentinel").write_text("keep\n")

        monkeypatch.setattr(cli, "_apply_profile_context", lambda p: None)
        wrote: dict[str, object] = {}
        monkeypatch.setattr(
            "dao_ai.apps.bundle.write_bundle",
            lambda *a, **k: wrote.setdefault("wrote", True),
        )
        with patch.object(cli, "deploy_app_bundle") as dep:
            opts = parse_args(["agent", "sync", "-c", str(cfg), "-s", str(out)])
            cli.handle_agent_command(opts)

        assert "wrote" not in wrote, "deploy must not regenerate"
        dep.assert_called_once()
        assert dep.call_args.kwargs["staging_dir"] == out
        assert (out / "sentinel").read_text() == "keep\n"

    def test_deploy_verb_errors_when_not_staged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = tmp_path / "c.yaml"
        cfg.write_text(
            "resources:\n  models:\n    m: &m\n      name: databricks-gpt-5-4-mini\n"
            "agents:\n  g: &g\n    name: g\n    description: d\n    model: *m\n"
            "    prompt: p\n"
            "app:\n  name: my_app\n  agents:\n    - *g\n"
        )
        monkeypatch.setattr(cli, "_apply_profile_context", lambda p: None)
        opts = parse_args(
            ["agent", "sync", "-c", str(cfg), "-s", str(tmp_path / "nope")]
        )
        with pytest.raises(SystemExit) as exc:
            cli.handle_agent_command(opts)
        assert exc.value.code == 1


_MINIMAL_CONFIG_NO_TARGET = (
    "resources:\n  models:\n    m: &m\n      name: databricks-gpt-5-4-mini\n"
    "agents:\n  g: &g\n    name: g\n    description: d\n    model: *m\n"
    "    prompt: p\n"
    "app:\n  name: my_app\n  agents:\n    - *g\n"
)


@pytest.mark.unit
class TestDeployAutoGenerate:
    """Task 6: deploy auto-generates when unstaged; --direct; model_serving routing."""

    def _write_cfg(self, tmp_path: Path) -> Path:
        cfg = tmp_path / "c.yaml"
        cfg.write_text(_MINIMAL_CONFIG_NO_TARGET)
        return cfg

    def test_deploy_autogenerates_when_unstaged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`agent deploy` on an empty dir stages first, then deploys the bundle."""
        cfg = self._write_cfg(tmp_path)
        out = tmp_path / "out"  # does NOT pre-exist with databricks.yaml

        wrote: dict[str, bool] = {}

        def fake_writer(config: object, bundle_dir: object, **kw: object) -> None:
            wrote["called"] = True
            # Create databricks.yaml so deploy_app_bundle finds it staged.
            import pathlib

            pathlib.Path(str(bundle_dir)).mkdir(parents=True, exist_ok=True)
            (pathlib.Path(str(bundle_dir)) / "databricks.yaml").write_text(
                "bundle: {}\n"
            )

        monkeypatch.setattr(cli, "_apply_profile_context", lambda p: None)
        monkeypatch.setattr(
            "dao_ai.apps.bundle.write_bundle",
            fake_writer,
        )
        with patch.object(cli, "deploy_app_bundle") as dep:
            opts = parse_args(["agent", "sync", "-c", str(cfg), "-s", str(out)])
            cli.handle_agent_command(opts)

        assert wrote.get("called"), "writer must be called to auto-generate the bundle"
        dep.assert_called_once()

    def test_deploy_in_place_when_staged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`agent deploy` on an already-staged dir deploys in-place without regenerating."""
        cfg = self._write_cfg(tmp_path)
        out = tmp_path / "staged"
        out.mkdir()
        (out / "databricks.yaml").write_text("bundle: {}\n")
        (out / "sentinel").write_text("keep\n")

        wrote: dict[str, bool] = {}

        monkeypatch.setattr(cli, "_apply_profile_context", lambda p: None)
        monkeypatch.setattr(
            "dao_ai.apps.bundle.write_bundle",
            lambda *a, **k: wrote.setdefault("called", True),
        )
        with patch.object(cli, "deploy_app_bundle") as dep:
            opts = parse_args(["agent", "sync", "-c", str(cfg), "-s", str(out)])
            cli.handle_agent_command(opts)

        assert not wrote, "deploy must NOT regenerate when already staged"
        dep.assert_called_once()
        assert (out / "sentinel").read_text() == "keep\n"

    def test_direct_flag_uses_sdk_not_bundle(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`agent up --direct` calls deploy_agent (SDK path) without touching bundles."""
        cfg = self._write_cfg(tmp_path)

        deploy_agent_calls: list[dict[str, object]] = []

        def fake_deploy_agent(
            self_config: object,
            target: object = None,
            development: object = None,
        ) -> None:
            deploy_agent_calls.append({"target": target, "development": development})

        monkeypatch.setattr(cli, "_apply_profile_context", lambda p: None)
        monkeypatch.setattr(
            "dao_ai.config.AppConfig.deploy_agent",
            fake_deploy_agent,
        )
        with (
            patch.object(cli, "deploy_app_bundle") as dep,
            patch.object(cli, "_exec_bundle_command") as exec_cmd,
        ):
            opts = parse_args(["agent", "up", "-c", str(cfg), "--direct"])
            cli.handle_agent_command(opts)

        assert deploy_agent_calls, "--direct must call deploy_agent"
        dep.assert_not_called()
        exec_cmd.assert_not_called()

    def test_model_serving_direct_registers_then_deploys_no_bundle(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`agent deploy --mode model_serving --direct` must register the model
        (create_agent) BEFORE deploy_agent via the SDK, and NOT touch a bundle.

        This is the escape-hatch SDK path (the former model_serving default).
        Skipping create_agent would deploy a stale/nonexistent model version.
        """
        cfg = self._write_cfg(tmp_path)

        calls: list[str] = []

        def fake_create_agent(self_config: object, development: object = None) -> None:
            calls.append("create")

        deploy_agent_calls: list[dict[str, object]] = []

        def fake_deploy_agent(
            self_config: object,
            target: object = None,
            development: object = None,
        ) -> None:
            calls.append("deploy")
            deploy_agent_calls.append({"target": target, "development": development})

        monkeypatch.setattr(cli, "_apply_profile_context", lambda p: None)
        monkeypatch.setattr(AppConfig, "_resolve_all_resources", lambda self: None)
        monkeypatch.setattr("dao_ai.config.AppConfig.create_agent", fake_create_agent)
        monkeypatch.setattr("dao_ai.config.AppConfig.deploy_agent", fake_deploy_agent)
        # --direct lives on the `up` verb (deploy is the pure push verb).
        with (
            patch.object(cli, "deploy_app_bundle") as dep,
            patch.object(cli, "_run_ms_job_bundle") as job_driver,
        ):
            opts = parse_args(
                ["agent", "up", "-c", str(cfg), "--mode", "model_serving", "--direct"]
            )
            cli.handle_agent_command(opts)

        from dao_ai.config import ServingMode

        # create_agent (register) must run BEFORE deploy_agent, and NO bundle
        # driver (App or Job) is touched on the --direct path.
        assert calls == ["create", "deploy"], calls
        assert deploy_agent_calls[0]["target"] == ServingMode.MODEL_SERVING
        dep.assert_not_called()
        job_driver.assert_not_called()

    def test_model_serving_default_uses_bundle_job_driver_not_sdk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`agent deploy --mode model_serving` (no --direct) now stages the thin
        MS Job bundle and runs the Job driver — it must NOT call create_agent/
        deploy_agent directly (that is the --direct path)."""
        cfg = self._write_cfg(tmp_path)

        sdk_calls: list[str] = []
        wrote: dict[str, bool] = {}

        def fake_ms_writer(config: object, bundle_dir: object, **kw: object) -> dict:
            wrote["called"] = True
            import pathlib

            pathlib.Path(str(bundle_dir)).mkdir(parents=True, exist_ok=True)
            (pathlib.Path(str(bundle_dir)) / "databricks.yaml").write_text(
                "bundle: {}\n"
            )
            return {}

        monkeypatch.setattr(cli, "_apply_profile_context", lambda p: None)
        monkeypatch.setattr(AppConfig, "_resolve_all_resources", lambda self: None)
        monkeypatch.setattr(cli, "detect_cloud_provider", lambda p: "aws")
        # Skip real dao_ai_dep resolution (needs a staged wheel in dev mode); the
        # test asserts writer + driver dispatch, not dependency pinning.
        monkeypatch.setattr(cli, "_resolve_job_dao_ai_dep", lambda *a, **k: "dao-ai")
        monkeypatch.setattr(
            "dao_ai.pipeline.bundle.write_model_serving_agent_bundle", fake_ms_writer
        )
        monkeypatch.setattr(
            "dao_ai.config.AppConfig.create_agent",
            lambda self, development=None: sdk_calls.append("create"),
        )
        monkeypatch.setattr(
            "dao_ai.config.AppConfig.deploy_agent",
            lambda self, target=None, development=None: sdk_calls.append("deploy"),
        )
        with (
            patch.object(cli, "_exec_job_bundle") as exec_job,
            patch.object(cli, "deploy_app_bundle") as app_driver,
        ):
            opts = parse_args(
                [
                    "agent",
                    "sync",
                    "-c",
                    str(cfg),
                    "-s",
                    str(tmp_path / "out"),
                    "--mode",
                    "model_serving",
                ]
            )
            cli.handle_agent_command(opts)

        assert wrote.get("called"), "MS writer must stage the Job bundle"
        assert sdk_calls == [], "bundle path must NOT call create/deploy_agent directly"
        exec_job.assert_called()  # Job driver ran bundle deploy
        app_driver.assert_not_called()

    def _write_cfg_with_provided(self, tmp_path: Path) -> Path:
        """A config whose genie room binds space_id to an unsupplied `provided` param."""
        cfg = tmp_path / "prov.yaml"
        cfg.write_text(
            "parameters:\n"
            "  gsid:\n    provided: true\n"
            "resources:\n"
            "  models:\n    m: &m\n      name: databricks-gpt-5-4-mini\n"
            "  genie_rooms:\n"
            "    room: &room\n      name: r\n      space_id: ${var.gsid}\n"
            "tools:\n"
            "  gt: &gt\n    name: genie\n    function:\n      type: genie\n"
            "      name: q\n      description: d\n      genie_room: *room\n"
            "agents:\n"
            "  a: &a\n    name: aa\n    description: aa\n    model: *m\n"
            "    tools: [*gt]\n    prompt: hi\n"
            "app:\n  name: prov_guardrail_app\n  agents: [*a]\n"
        )
        return cfg

    @pytest.mark.parametrize("argv_tail", [["--mode", "model_serving"], ["--direct"]])
    def test_unsatisfied_provided_param_blocks_sdk_deploy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, argv_tail: list[str]
    ) -> None:
        """model_serving + --direct deploys must hard-error (guardrail) on an
        unsupplied `provided` param BEFORE calling create_agent/deploy_agent."""
        cfg = self._write_cfg_with_provided(tmp_path)

        called: list[str] = []
        monkeypatch.setattr(cli, "_apply_profile_context", lambda p: None)
        monkeypatch.setattr(AppConfig, "_resolve_all_resources", lambda self: None)
        monkeypatch.setattr(
            "dao_ai.config.AppConfig.create_agent",
            lambda self, development=None: called.append("create"),
        )
        monkeypatch.setattr(
            "dao_ai.config.AppConfig.deploy_agent",
            lambda self, target=None, development=None: called.append("deploy"),
        )

        opts = parse_args(["agent", "up", "-c", str(cfg), *argv_tail])
        with pytest.raises(ValueError, match="provided"):
            cli.handle_agent_command(opts)
        assert called == [], "guardrail must fire before create/deploy"

    def test_supplied_provided_param_allows_sdk_deploy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Supplying the `provided` param via --param satisfies the guardrail."""
        cfg = self._write_cfg_with_provided(tmp_path)

        called: list[str] = []
        monkeypatch.setattr(cli, "_apply_profile_context", lambda p: None)
        monkeypatch.setattr(AppConfig, "_resolve_all_resources", lambda self: None)
        monkeypatch.setattr(
            "dao_ai.config.AppConfig.deploy_agent",
            lambda self, target=None, development=None: called.append("deploy"),
        )
        opts = parse_args(
            ["agent", "up", "-c", str(cfg), "--direct", "--param", "gsid=01fREAL"]
        )
        cli.handle_agent_command(opts)
        assert called == ["deploy"]

    def test_direct_flag_parses_on_up(self) -> None:
        """`--direct` is accepted on the `up` verb (moved from deploy)."""
        opts = parse_args(["agent", "up", "-c", "c.yaml", "--direct"])
        assert opts.direct is True

    def test_direct_flag_rejected_on_deploy(self) -> None:
        """`--direct` is NOT accepted on `deploy` — use `up --direct` instead."""
        with pytest.raises(SystemExit):
            parse_args(["agent", "sync", "-c", "c.yaml", "--direct"])

    def test_run_still_errors_when_unstaged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`agent run` on an empty dir must still error (auto-generate is deploy-only)."""
        cfg = self._write_cfg(tmp_path)
        monkeypatch.setattr(cli, "_apply_profile_context", lambda p: None)
        opts = parse_args(
            ["agent", "start", "-c", str(cfg), "-s", str(tmp_path / "nope")]
        )
        with pytest.raises(SystemExit) as exc:
            cli.handle_agent_command(opts)
        assert exc.value.code == 1

    def test_deploy_autogenerate_calls_resolve_all_resources(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On auto-generate (unstaged deploy), _resolve_all_resources must be called."""
        cfg = self._write_cfg(tmp_path)
        out = tmp_path / "out"

        resolve_calls: list[str] = []

        def fake_writer(config: object, bundle_dir: object, **kw: object) -> None:
            import pathlib

            pathlib.Path(str(bundle_dir)).mkdir(parents=True, exist_ok=True)
            (pathlib.Path(str(bundle_dir)) / "databricks.yaml").write_text(
                "bundle: {}\n"
            )

        def track_resolve(self_config: object) -> None:
            resolve_calls.append("called")

        monkeypatch.setattr(cli, "_apply_profile_context", lambda p: None)
        monkeypatch.setattr(
            "dao_ai.apps.bundle.write_bundle",
            fake_writer,
        )
        monkeypatch.setattr(
            "dao_ai.config.AppConfig._resolve_all_resources",
            track_resolve,
        )
        with patch.object(cli, "deploy_app_bundle"):
            opts = parse_args(["agent", "sync", "-c", str(cfg), "-s", str(out)])
            cli.handle_agent_command(opts)

        assert resolve_calls == ["called"], (
            "auto-generate path must call _resolve_all_resources"
        )

    def test_deploy_inplace_does_not_call_resolve_all_resources(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On in-place deploy (already staged), _resolve_all_resources must NOT be called."""
        cfg = self._write_cfg(tmp_path)
        out = tmp_path / "staged"
        out.mkdir()
        (out / "databricks.yaml").write_text("bundle: {}\n")

        resolve_calls: list[str] = []

        def track_resolve(self_config: object) -> None:
            resolve_calls.append("called")

        monkeypatch.setattr(cli, "_apply_profile_context", lambda p: None)
        monkeypatch.setattr(
            "dao_ai.config.AppConfig._resolve_all_resources",
            track_resolve,
        )
        with patch.object(cli, "deploy_app_bundle"):
            opts = parse_args(["agent", "sync", "-c", str(cfg), "-s", str(out)])
            cli.handle_agent_command(opts)

        assert resolve_calls == [], (
            "in-place deploy path must NOT call _resolve_all_resources"
        )

    def test_up_deploys_and_runs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`agent up --mode apps` bundle path calls deploy_app_bundle with deploy=True AND run=True."""
        cfg = self._write_cfg(tmp_path)
        out = tmp_path / "staged"
        out.mkdir()
        (out / "databricks.yaml").write_text("bundle: {}\n")

        monkeypatch.setattr(cli, "_apply_profile_context", lambda p: None)
        with patch.object(cli, "deploy_app_bundle") as dep:
            opts = parse_args(
                ["agent", "up", "-c", str(cfg), "-s", str(out), "--mode", "apps"]
            )
            cli.handle_agent_command(opts)

        dep.assert_called_once()
        call_kwargs = dep.call_args.kwargs
        assert call_kwargs["deploy"] is True, (
            "up must call deploy_app_bundle(deploy=True)"
        )
        assert call_kwargs["run"] is True, "up must call deploy_app_bundle(run=True)"


@pytest.mark.unit
class TestToolsCommandRenamed:
    """MCP tool listing now lives under `mcp tools`; older names are rejected."""

    def test_mcp_tools_parses(self) -> None:
        """`mcp tools` parses to (command=mcp, subcommand=tools)."""
        o = parse_args(["mcp", "tools", "-c", "c.yaml"])
        assert o.command == "mcp"
        assert o.subcommand == "tools"

    def test_top_level_tools_rejected(self) -> None:
        """The interim top-level `tools` name is rejected (moved to `mcp tools`)."""
        with pytest.raises(SystemExit):
            parse_args(["tools", "-c", "c.yaml"])

    def test_list_mcp_tools_rejected(self) -> None:
        """Old list-mcp-tools name is rejected by the parser."""
        with pytest.raises(SystemExit):
            parse_args(["list-mcp-tools", "-c", "c.yaml"])


@pytest.mark.unit
class TestTraceNoun:
    """``dao-ai trace create|link|grant`` is the new surface; old flat names are gone."""

    def test_trace_create_parses(self) -> None:
        o = parse_args(["trace", "create", "--name", "/Shared/my-exp"])
        assert o.command == "trace"
        assert o.subcommand == "create"
        assert o.name == "/Shared/my-exp"

    def test_trace_create_id_flag(self) -> None:
        o = parse_args(["trace", "create", "--id", "1234"])
        assert o.command == "trace"
        assert o.subcommand == "create"
        assert o.id == "1234"

    def test_trace_create_output_default(self) -> None:
        o = parse_args(["trace", "create", "--name", "/foo"])
        assert o.output == "text"

    def test_trace_create_output_json(self) -> None:
        o = parse_args(["trace", "create", "--name", "/foo", "-o", "json"])
        assert o.output == "json"

    def test_trace_create_no_create_flag(self) -> None:
        o = parse_args(["trace", "create", "--name", "/foo", "--no-create"])
        assert o.no_create is True

    def test_trace_link_parses(self) -> None:
        o = parse_args(["trace", "link", "-c", "config.yaml"])
        assert o.command == "trace"
        assert o.subcommand == "link"
        assert o.config == "config.yaml"

    def test_trace_link_optional_flags(self) -> None:
        o = parse_args(
            [
                "trace",
                "link",
                "-c",
                "c.yaml",
                "--experiment-id",
                "9999",
                "--app-sp",
                "uuid-1",
            ]
        )
        assert o.experiment_id == "9999"
        assert o.app_sp == "uuid-1"

    def test_trace_grant_parses(self) -> None:
        o = parse_args(["trace", "grant", "-c", "config.yaml"])
        assert o.command == "trace"
        assert o.subcommand == "grant"
        assert o.config == "config.yaml"

    def test_trace_grant_optional_flags(self) -> None:
        o = parse_args(
            [
                "trace",
                "grant",
                "-c",
                "c.yaml",
                "--experiment-id",
                "8888",
                "--app-sp",
                "uuid-2",
            ]
        )
        assert o.experiment_id == "8888"
        assert o.app_sp == "uuid-2"

    @pytest.mark.parametrize(
        "old",
        ["create-experiment", "link-trace-destination", "grant-trace-permissions"],
    )
    def test_old_flat_names_rejected(self, old: str) -> None:
        with pytest.raises(SystemExit):
            parse_args([old])

    def test_bare_trace_requires_verb(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["trace"])

    def test_trace_link_var_flag(self) -> None:
        o = parse_args(["trace", "link", "-c", "c.yaml", "--var", "k=v"])
        assert o.var == ["k=v"]

    def test_trace_grant_var_flag(self) -> None:
        o = parse_args(["trace", "grant", "-c", "c.yaml", "--var", "k=v"])
        assert o.var == ["k=v"]


@pytest.mark.unit
class TestTraceNounDispatch:
    """main() routes ``trace <verb>`` to the correct handler via handle_trace_command."""

    def _invoke(
        self, monkeypatch: pytest.MonkeyPatch, argv: list[str]
    ) -> dict[str, int]:
        called: dict[str, int] = {}
        monkeypatch.setattr(
            cli,
            "handle_create_experiment_command",
            lambda o: called.update(create=called.get("create", 0) + 1),
        )
        monkeypatch.setattr(
            cli,
            "handle_link_trace_destination_command",
            lambda o: called.update(link=called.get("link", 0) + 1),
        )
        monkeypatch.setattr(
            cli,
            "handle_grant_trace_permissions_command",
            lambda o: called.update(grant=called.get("grant", 0) + 1),
        )
        monkeypatch.setattr(cli, "setup_logging", lambda v: None)
        monkeypatch.setattr("sys.argv", ["dao-ai"] + argv)
        cli.main()
        return called

    def test_trace_create_routes_to_create_handler(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        called = self._invoke(monkeypatch, ["trace", "create", "--name", "/foo"])
        assert called == {"create": 1}

    def test_trace_link_routes_to_link_handler(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        called = self._invoke(monkeypatch, ["trace", "link", "-c", "c.yaml"])
        assert called == {"link": 1}

    def test_trace_grant_routes_to_grant_handler(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        called = self._invoke(monkeypatch, ["trace", "grant", "-c", "c.yaml"])
        assert called == {"grant": 1}


@pytest.mark.unit
class TestMcpNoun:
    """``dao-ai mcp tools|inspect|call`` — utilities noun; top-level `tools` is gone."""

    def test_mcp_tools_parses(self) -> None:
        o = parse_args(["mcp", "tools", "-c", "c.yaml"])
        assert o.command == "mcp"
        assert o.subcommand == "tools"
        assert o.config == "c.yaml"
        assert o.apply_filters is False

    def test_mcp_tools_apply_filters(self) -> None:
        o = parse_args(["mcp", "tools", "-c", "c.yaml", "--apply-filters"])
        assert o.apply_filters is True

    def test_mcp_tools_var_flag(self) -> None:
        o = parse_args(["mcp", "tools", "-c", "c.yaml", "--var", "k=v"])
        assert o.var == ["k=v"]

    def test_mcp_inspect_url(self) -> None:
        o = parse_args(["mcp", "inspect", "--url", "https://x/mcp"])
        assert o.command == "mcp"
        assert o.subcommand == "inspect"
        assert o.url == "https://x/mcp"
        assert o.app is None

    def test_mcp_inspect_app(self) -> None:
        o = parse_args(["mcp", "inspect", "--app", "my-app"])
        assert o.app == "my-app"
        assert o.url is None

    def test_mcp_inspect_requires_target(self) -> None:
        # --url/--app are a required mutually-exclusive group.
        with pytest.raises(SystemExit):
            parse_args(["mcp", "inspect"])

    def test_mcp_inspect_rejects_both_targets(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["mcp", "inspect", "--url", "https://x/mcp", "--app", "a"])

    def test_mcp_call_parses(self) -> None:
        o = parse_args(
            ["mcp", "call", "mytool", "--url", "https://x/mcp", "--args", '{"a":1}']
        )
        assert o.command == "mcp"
        assert o.subcommand == "call"
        assert o.tool == "mytool"
        assert o.url == "https://x/mcp"
        assert o.args == '{"a":1}'

    def test_mcp_call_args_default(self) -> None:
        o = parse_args(["mcp", "call", "mytool", "--app", "a"])
        assert o.args == "{}"

    def test_mcp_call_requires_target(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["mcp", "call", "mytool"])

    def test_bare_mcp_requires_verb(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["mcp"])


@pytest.mark.unit
class TestMcpNounDispatch:
    """main() routes ``mcp <verb>`` to the correct handler via handle_mcp_command."""

    def _invoke(
        self, monkeypatch: pytest.MonkeyPatch, argv: list[str]
    ) -> dict[str, int]:
        called: dict[str, int] = {}
        monkeypatch.setattr(
            cli,
            "_handle_mcp_tools",
            lambda o: called.update(tools=called.get("tools", 0) + 1),
        )
        monkeypatch.setattr(
            cli,
            "_handle_mcp_inspect",
            lambda o: called.update(inspect=called.get("inspect", 0) + 1),
        )
        monkeypatch.setattr(
            cli,
            "_handle_mcp_call",
            lambda o: called.update(call=called.get("call", 0) + 1),
        )
        monkeypatch.setattr(cli, "setup_logging", lambda v: None)
        monkeypatch.setattr("sys.argv", ["dao-ai"] + argv)
        cli.main()
        return called

    def test_mcp_tools_routes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self._invoke(monkeypatch, ["mcp", "tools", "-c", "c.yaml"]) == {
            "tools": 1
        }

    def test_mcp_inspect_routes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self._invoke(monkeypatch, ["mcp", "inspect", "--app", "a"]) == {
            "inspect": 1
        }

    def test_mcp_call_routes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self._invoke(monkeypatch, ["mcp", "call", "t", "--app", "a"]) == {
            "call": 1
        }


@pytest.mark.unit
class TestGlobalProfileVerbose:
    """``-p/--profile`` and ``-v/--verbose`` work at both top level and subcommand level."""

    def test_profile_before_subcommand(self) -> None:
        o = parse_args(["-p", "fevm", "agent", "sync", "-c", "c.yaml"])
        assert o.profile == "fevm"

    def test_profile_after_subcommand(self) -> None:
        o = parse_args(["agent", "sync", "-c", "c.yaml", "-p", "fevm"])
        assert o.profile == "fevm"

    def test_profile_subcommand_wins_when_both(self) -> None:
        o = parse_args(["-p", "top", "agent", "sync", "-c", "c.yaml", "-p", "sub"])
        assert o.profile == "sub"

    def test_profile_absent_defaults_none(self) -> None:
        o = parse_args(["agent", "build", "-c", "c.yaml"])
        assert getattr(o, "profile", None) is None

    def test_verbose_before_subcommand(self) -> None:
        o = parse_args(["-v", "agent", "sync", "-c", "c.yaml"])
        assert o.verbose >= 1

    def test_verbose_after_subcommand(self) -> None:
        o = parse_args(["agent", "sync", "-c", "c.yaml", "-vv"])
        assert o.verbose >= 2

    def test_verbose_absent_defaults_zero(self) -> None:
        o = parse_args(["agent", "build", "-c", "c.yaml"])
        assert getattr(o, "verbose", 0) == 0

    def test_profile_before_trace_subcommand(self) -> None:
        o = parse_args(["-p", "fevm", "trace", "link", "-c", "c.yaml"])
        assert o.profile == "fevm"

    def test_profile_after_trace_subcommand(self) -> None:
        o = parse_args(["trace", "link", "-c", "c.yaml", "-p", "fevm"])
        assert o.profile == "fevm"

    def test_verbose_before_workflow_subcommand(self) -> None:
        o = parse_args(["-vv", "workflow", "build", "-c", "c.yaml"])
        assert o.verbose >= 2

    def test_verbose_after_workflow_subcommand(self) -> None:
        o = parse_args(["workflow", "sync", "-c", "c.yaml", "-v"])
        assert o.verbose >= 1


@pytest.mark.unit
class TestParametersSubcommand:
    """`dao-ai parameters` (alias `vars`): action defaults to list; get takes a name."""

    def test_bare_defaults_to_list(self) -> None:
        o = parse_args(["parameters", "-c", "c.yaml"])
        assert o.action == "list"
        assert o.name is None

    def test_explicit_list_backcompat(self) -> None:
        o = parse_args(["parameters", "list", "-c", "c.yaml"])
        assert o.action == "list"

    def test_vars_alias_bare(self) -> None:
        o = parse_args(["vars", "-c", "c.yaml"])
        assert o.action == "list"

    def test_get_with_name(self) -> None:
        o = parse_args(["parameters", "get", "catalog", "-c", "c.yaml"])
        assert o.action == "get"
        assert o.name == "catalog"

    def test_invalid_action_rejected(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["parameters", "bogus", "-c", "c.yaml"])

    def test_config_still_required(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["parameters", "list"])


# ---------------------------------------------------------------------------
# Auto-adopt: bind an existing-but-untracked workspace resource before deploy
# so `bundle deploy` UPDATEs instead of 409-ing on a blind CREATE.
# ---------------------------------------------------------------------------


import json as _json  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402


def _plan_json(*resources: tuple[str, str, dict]) -> str:
    """Build a `bundle plan --output json` payload.

    Each resource is (node_type, short_key, extra_entry_fields). The node key is
    ``resources.<type>.<short_key>``; extra fields (e.g. action, new_state) merge
    into the entry.
    """
    plan: dict = {}
    for rtype, key, fields in resources:
        plan[f"resources.{rtype}.{key}"] = fields
    return _json.dumps({"plan": plan})


def _completed(returncode: int, stdout: str = "", stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


@pytest.mark.unit
class TestAdoptUntrackedBundleResources:
    _APP_CREATE = (
        "apps",
        "my-app",
        {"action": "create", "new_state": {"value": {"name": "my-app"}}},
    )

    def _run_adopt(
        self, tmp_path: Path, *, plan: str, target: str = "dev",
        extra_vars=None, dry_run: bool = False, app_exists: bool = True,
    ):
        """Drive the helper with subprocess + WorkspaceClient patched.

        Returns the list of argv lists passed to subprocess.run so the test can
        assert which commands (plan/bind) ran.
        """
        calls: list[list[str]] = []

        def _fake_run(cmd, **kwargs):
            calls.append(cmd)
            if cmd[-3:] == ["bundle", "plan", "--output"] or "plan" in cmd:
                return _completed(0, stdout=plan)
            return _completed(0)  # bind

        w = MagicMock()
        if app_exists:
            w.apps.get.return_value = SimpleNamespace(name="my-app")
        else:
            from databricks.sdk.errors import NotFound

            w.apps.get.side_effect = NotFound("no such app")

        with patch.object(cli.subprocess, "run", side_effect=_fake_run), patch(
            "databricks.sdk.WorkspaceClient", return_value=w
        ):
            cli._adopt_untracked_bundle_resources(
                staging_dir=tmp_path, profile="fevm", target=target,
                extra_vars=extra_vars, dry_run=dry_run,
            )
        return calls

    def test_app_would_create_and_exists_binds(self, tmp_path: Path) -> None:
        calls = self._run_adopt(tmp_path, plan=_plan_json(self._APP_CREATE))
        bind_calls = [c for c in calls if "bind" in c]
        assert len(bind_calls) == 1
        b = bind_calls[0]
        assert b[-4:] == ["bind", "my-app", "my-app", "--auto-approve"] or (
            "bind" in b and "my-app" in b and "--auto-approve" in b
        )

    def test_app_would_create_but_absent_no_bind(self, tmp_path: Path) -> None:
        calls = self._run_adopt(
            tmp_path, plan=_plan_json(self._APP_CREATE), app_exists=False
        )
        assert not [c for c in calls if "bind" in c]

    def test_resource_skip_action_no_bind(self, tmp_path: Path) -> None:
        entry = ("apps", "my-app", {"action": "skip"})
        calls = self._run_adopt(tmp_path, plan=_plan_json(entry))
        assert not [c for c in calls if "bind" in c]

    def test_dry_run_prints_no_bind(self, tmp_path: Path) -> None:
        calls = self._run_adopt(
            tmp_path, plan=_plan_json(self._APP_CREATE), dry_run=True
        )
        # plan still runs (read-only); bind is NOT executed as a subprocess.
        assert not [c for c in calls if "bind" in c]
        assert any("plan" in c for c in calls)

    def test_bind_failure_is_swallowed(self, tmp_path: Path) -> None:
        def _fake_run(cmd, **kwargs):
            if "plan" in cmd:
                return _completed(0, stdout=_plan_json(self._APP_CREATE))
            return _completed(1, stderr="bind boom")  # bind fails

        w = MagicMock()
        w.apps.get.return_value = SimpleNamespace(name="my-app")
        with patch.object(cli.subprocess, "run", side_effect=_fake_run), patch(
            "databricks.sdk.WorkspaceClient", return_value=w
        ):
            # Must not raise.
            cli._adopt_untracked_bundle_resources(
                staging_dir=tmp_path, profile="fevm", target="dev",
            )

    def test_plan_failure_is_noop(self, tmp_path: Path) -> None:
        def _fake_run(cmd, **kwargs):
            return _completed(1, stderr="plan boom")

        with patch.object(cli.subprocess, "run", side_effect=_fake_run):
            cli._adopt_untracked_bundle_resources(
                staging_dir=tmp_path, profile="fevm", target="dev",
            )  # no raise, no bind

    def test_plan_non_json_is_noop(self, tmp_path: Path) -> None:
        def _fake_run(cmd, **kwargs):
            return _completed(0, stdout="not json")

        with patch.object(cli.subprocess, "run", side_effect=_fake_run):
            cli._adopt_untracked_bundle_resources(
                staging_dir=tmp_path, profile="fevm", target="dev",
            )

    def test_job_target_and_vars_forwarded(self, tmp_path: Path) -> None:
        calls = self._run_adopt(
            tmp_path,
            plan=_plan_json(),  # empty plan; we only assert the plan argv
            target="myapp-aws",
            extra_vars=['--var="config_path=../config/x.yaml"'],
        )
        plan_call = next(c for c in calls if "plan" in c)
        assert "--target" in plan_call and "myapp-aws" in plan_call
        assert '--var="config_path=../config/x.yaml"' in plan_call


@pytest.mark.unit
class TestDeployTriggersAdopt:
    """The adopt step fires only for `bundle deploy`, across both exec paths."""

    def _fake_popen(self):
        """A Popen stand-in whose stdout.readline drains cleanly (empty output)."""
        proc = MagicMock()
        proc.stdout.readline.side_effect = [""]  # iter(readline, "") stops at once
        proc.wait.return_value = None
        proc.returncode = 0
        return proc

    def test_deploy_triggers_adopt(self, tmp_path: Path) -> None:
        with patch.object(cli, "_adopt_untracked_bundle_resources") as adopt, patch.object(
            cli.subprocess, "Popen", return_value=self._fake_popen()
        ):
            cli._exec_bundle_command(
                ["bundle", "deploy"], profile="fevm", target="dev", cwd=tmp_path,
            )
        adopt.assert_called_once()
        assert adopt.call_args.kwargs["target"] == "dev"
        assert adopt.call_args.kwargs["staging_dir"] == tmp_path

    def test_run_skips_adopt(self, tmp_path: Path) -> None:
        with patch.object(cli, "_adopt_untracked_bundle_resources") as adopt, patch.object(
            cli.subprocess, "Popen", return_value=self._fake_popen()
        ):
            cli._exec_bundle_command(
                ["bundle", "run", "my-app"], profile="fevm", target="dev", cwd=tmp_path,
            )
        adopt.assert_not_called()

    def test_destroy_skips_adopt(self, tmp_path: Path) -> None:
        with patch.object(cli, "_adopt_untracked_bundle_resources") as adopt, patch.object(
            cli.subprocess, "Popen", return_value=self._fake_popen()
        ):
            cli._exec_bundle_command(
                ["bundle", "destroy", "--auto-approve"], profile="fevm",
                target="dev", cwd=tmp_path,
            )
        adopt.assert_not_called()

    def test_job_deploy_forwards_target_and_vars(self, tmp_path: Path) -> None:
        with patch.object(cli, "_adopt_untracked_bundle_resources") as adopt, patch.object(
            cli.subprocess, "Popen", return_value=self._fake_popen()
        ):
            cli._exec_bundle_command(
                ["bundle", "deploy"], profile="fevm", target="myapp-aws",
                cwd=tmp_path, extra_vars=['--var="mode=apps"'],
            )
        adopt.assert_called_once()
        assert adopt.call_args.kwargs["target"] == "myapp-aws"
        assert adopt.call_args.kwargs["extra_vars"] == ['--var="mode=apps"']
