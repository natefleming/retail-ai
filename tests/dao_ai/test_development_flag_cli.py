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


def _stamp_manifest(bundle_dir: Path, *, checksum: str = "") -> None:
    """Write a `.manifest.yaml` recording the config checksum.

    The staging dir is ephemeral build output, so the manifest records only the
    ``checksum`` used for drift detection (no per-file registry).
    """
    cli._write_staging_manifest(bundle_dir, is_default=True, checksum=checksum)


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
        cli._clean_default_staging_dir(under, is_default=True)
        assert not under.exists(), "path under env base must be cleaned"

        outside = tmp_path / "user_dir"
        outside.mkdir()
        (outside / "keep").write_text("x")
        cli._clean_default_staging_dir(outside, is_default=True)
        assert (outside / "keep").exists(), "path outside base must be protected"


@pytest.mark.unit
class TestCleanDefaultStagingDir:
    """The default staging dir is ephemeral build output — always wiped for a
    clean regenerate; user-supplied / outside-base dirs are never touched."""

    def _staged(self, tmp_path: Path) -> Path:
        base = tmp_path / "base"
        d = base / "agent" / "app"
        d.mkdir(parents=True)
        (d / "app.yaml").write_text("orig\n")  # generated file
        (d / "src").mkdir()
        (d / "src" / "mine.py").write_text("print('mine')\n")  # user code
        _stamp_manifest(d)
        return d

    def test_default_dir_wiped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DAO_AI_BUNDLE_DIR", str(tmp_path / "base"))
        d = self._staged(tmp_path)
        cli._clean_default_staging_dir(d, is_default=True)
        assert not d.exists()

    def test_default_dir_wiped_even_with_hand_edits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No edit-detection: a default dir is ephemeral, so it's always wiped for
        a clean regenerate (user intent lives in the config, not the staging dir)."""
        monkeypatch.setenv("DAO_AI_BUNDLE_DIR", str(tmp_path / "base"))
        d = self._staged(tmp_path)
        (d / "app.yaml").write_text("edited\n")
        (d / "extra.txt").write_text("hand-dropped\n")
        cli._clean_default_staging_dir(d, is_default=True)
        assert not d.exists()

    def test_non_default_dir_never_wiped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A user-supplied ``-o`` dir (is_default=False) is left entirely alone."""
        monkeypatch.setenv("DAO_AI_BUNDLE_DIR", str(tmp_path / "base"))
        d = self._staged(tmp_path)
        cli._clean_default_staging_dir(d, is_default=False)
        assert d.exists()

    def test_dir_outside_base_never_wiped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Even is_default=True only wipes paths strictly under the owned base."""
        monkeypatch.setenv("DAO_AI_BUNDLE_DIR", str(tmp_path / "base"))
        outside = tmp_path / "user_dir"
        outside.mkdir()
        (outside / "keep").write_text("x")
        cli._clean_default_staging_dir(outside, is_default=True)
        assert (outside / "keep").exists()


@pytest.mark.unit
class TestConfigChecksum:
    """The staging manifest records a config checksum for staleness detection."""

    def _config(self, name: str = "greeter") -> AppConfig:
        from dao_ai.config import AgentModel, AppModel, InferenceEndpointModel

        return AppConfig(
            app=AppModel(
                name="cksum-test",
                agents=[
                    AgentModel(
                        name=name,
                        description="says hi",
                        model=InferenceEndpointModel(name="databricks-gpt-5-4-mini"),
                    )
                ],
            )
        )

    def test_checksum_stable_for_identical_config(self) -> None:
        a = cli._config_checksum(self._config(), development=False)
        b = cli._config_checksum(self._config(), development=False)
        assert a == b

    def test_checksum_changes_with_config(self) -> None:
        a = cli._config_checksum(self._config(name="greeter"), development=False)
        b = cli._config_checksum(self._config(name="farewell"), development=False)
        assert a != b

    def test_checksum_changes_with_development_flag(self) -> None:
        a = cli._config_checksum(self._config(), development=False)
        b = cli._config_checksum(self._config(), development=True)
        assert a != b

    def test_checksum_stable_despite_input_example_uuid(self) -> None:
        """Regression: ``app.input_example`` auto-injects a fresh conversation_id
        UUID on every load (ChatPayload.ensure_thread_id). The checksum must
        exclude it so a config with an input_example still hashes identically
        across loads — otherwise the idempotent-skip never triggers."""
        from dao_ai.config import (
            AgentModel,
            AppModel,
            ChatPayload,
            InferenceEndpointModel,
        )

        def _cfg() -> AppConfig:
            return AppConfig(
                app=AppModel(
                    name="cksum-ie",
                    agents=[
                        AgentModel(
                            name="g",
                            description="hi",
                            model=InferenceEndpointModel(
                                name="databricks-gpt-5-4-mini"
                            ),
                        )
                    ],
                    # No conversation_id/thread_id → a UUID is injected each load.
                    input_example=ChatPayload(
                        messages=[{"role": "user", "content": "hi"}]
                    ),
                )
            )

        # Sanity: the two loads really do differ in the injected UUID.
        a_cfg, b_cfg = _cfg(), _cfg()
        a_cid = a_cfg.app.input_example.custom_inputs["configurable"][
            "conversation_id"
        ]
        b_cid = b_cfg.app.input_example.custom_inputs["configurable"][
            "conversation_id"
        ]
        assert a_cid != b_cid, "precondition: input_example UUIDs should differ"
        # ...yet the checksum is stable (input_example excluded from the hash).
        assert cli._config_checksum(
            a_cfg, development=False
        ) == cli._config_checksum(b_cfg, development=False)

    def test_checksum_changes_when_custom_code_edited(self, tmp_path: Path) -> None:
        """Regression: editing a code_paths/src file WITHOUT touching the config
        must change the checksum, so the idempotent-skip never ships stale code."""
        (tmp_path / "src" / "mypkg").mkdir(parents=True)
        tool = tmp_path / "src" / "mypkg" / "tool.py"
        tool.write_text("VERSION = 1\n")
        cfg_path = tmp_path / "dao_ai.yaml"
        cfg_path.write_text(
            "resources:\n  models:\n    m: &m {name: databricks-gpt-5-4-mini}\n"
            "agents:\n  g: &g {name: g, description: d, model: *m, prompt: p}\n"
            "app:\n  name: cksum_code\n"
            "  registered_model: {schema: {catalog_name: c, schema_name: s}, "
            "name: m}\n  agents: [*g]\n"
        )

        def _checksum() -> str:
            return cli._config_checksum(
                AppConfig.from_file(str(cfg_path), initialize=False),
                development=False,
            )

        before = _checksum()
        # Edit ONLY the src file — config bytes are untouched.
        tool.write_text("VERSION = 2\n")
        after = _checksum()
        assert before != after, (
            "editing custom code must change the checksum (else stale code ships)"
        )

    def test_checksum_fails_loud_on_missing_resource_path(
        self, tmp_path: Path
    ) -> None:
        """A bad resource_paths entry must surface through _config_checksum (which
        every build/up/sync computes), not be silently checksummed away — locks in
        the fail-loud contract the checksum folds in via iter_resource_path_stagings."""
        cfg_path = tmp_path / "dao_ai.yaml"
        cfg_path.write_text(
            "resources:\n  models:\n    m: &m {name: databricks-gpt-5-4-mini}\n"
            "agents:\n  g: &g {name: g, description: d, model: *m, prompt: p}\n"
            "app:\n  name: cksum_badres\n"
            "  resource_paths: [does/not/exist.yml]\n"
            "  registered_model: {schema: {catalog_name: c, schema_name: s}, "
            "name: m}\n  agents: [*g]\n"
        )
        with pytest.raises(FileNotFoundError, match="does not exist"):
            cli._config_checksum(
                AppConfig.from_file(str(cfg_path), initialize=False),
                development=False,
            )

    def test_manifest_records_only_checksum(self, tmp_path: Path) -> None:
        d = tmp_path / "base" / "agent" / "app"
        d.mkdir(parents=True)
        cli._write_staging_manifest(d, is_default=True, checksum="abc123")
        manifest = cli._read_staging_manifest(d)
        assert manifest is not None
        assert manifest["version"] == cli._MANIFEST_VERSION
        assert manifest["checksum"] == "abc123"
        # Ephemeral model: no per-file edit-detection registry.
        assert "files" not in manifest
        assert "tracked" not in manifest

    def test_manifest_only_written_for_default_dir(self, tmp_path: Path) -> None:
        d = tmp_path / "user_dir"
        d.mkdir(parents=True)
        # is_default=False → a user -o dir is never stamped.
        cli._write_staging_manifest(d, is_default=False, checksum="cksum")
        assert not (d / cli._STAGING_MANIFEST).exists()


@pytest.mark.unit
class TestStagedConfigStaleness:
    """_staged_config_is_stale compares current config vs the stamped checksum."""

    def _marked(self, tmp_path: Path, checksum: str) -> Path:
        d = tmp_path / "base" / "agent" / "app"
        d.mkdir(parents=True)
        cli._write_staging_manifest(d, is_default=True, checksum=checksum)
        return d

    def test_matching_checksum_not_stale(self, tmp_path: Path) -> None:
        d = self._marked(tmp_path, "cksum-1")
        assert cli._staged_config_is_stale(d, "cksum-1") is False

    def test_differing_checksum_is_stale(self, tmp_path: Path) -> None:
        d = self._marked(tmp_path, "cksum-1")
        assert cli._staged_config_is_stale(d, "cksum-2") is True

    def test_missing_manifest_not_stale(self, tmp_path: Path) -> None:
        d = tmp_path / "base" / "agent" / "app"
        d.mkdir(parents=True)  # no manifest (user -o dir)
        assert cli._staged_config_is_stale(d, "cksum-1") is False

    def test_empty_checksum_not_stale(self, tmp_path: Path) -> None:
        # A workflow bundle may stamp an empty checksum; never treat it as stale.
        d = self._marked(tmp_path, "")
        assert cli._staged_config_is_stale(d, "cksum-1") is False


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

    def test_agent_up_model_serving_uses_ms_writer_and_job_driver(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """agent up --mode model_serving stages via the MS writer and runs
        the Job driver (_run_ms_job_bundle), NOT the App driver. (Only `up`
        builds; unstaged `sync` errors.)"""
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
                        "up",
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
    """agent `up` re-stages an already-staged bundle when the source config drifts.
    (Re-staging on drift is an `up`-only concern — `sync` never rebuilds.)"""

    def _write_config(self, tmp_path: Path) -> Path:
        cfg = tmp_path / "c.yaml"
        cfg.write_text(_MINIMAL_CONFIG_YAML)
        return cfg

    def _run_deploy(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        marker_checksum: str | None,
        overwrite: bool = False,
    ) -> bool:
        """Drive `agent deploy` against a pre-staged dir. Returns True if re-staged.

        ``marker_checksum`` seeds the staged manifest's checksum: None omits the
        manifest (user -o dir), "" is a workflow-style empty checksum, a string is
        a recorded hash. ``overwrite`` passes --overwrite.
        """
        cfg = self._write_config(tmp_path)
        staged = tmp_path / "staged" / "agent" / "test_app"
        staged.mkdir(parents=True, exist_ok=True)
        (staged / "databricks.yaml").write_text("bundle: {}\n")
        if marker_checksum is not None:
            cli._write_staging_manifest(
                staged,
                is_default=True,
                checksum=marker_checksum,
            )

        restaged: dict[str, bool] = {}
        monkeypatch.setattr(cli, "_apply_profile_context", lambda p: None)
        monkeypatch.setattr(
            cli,
            "_resolve_bundle_dir",
            lambda kind, config, staging_dir, mode_subdir=None: (staged, True),
        )
        monkeypatch.setattr(cli, "deploy_app_bundle", lambda *a, **k: None)
        monkeypatch.setattr(AppConfig, "_resolve_all_resources", lambda self: None)
        monkeypatch.setattr(
            cli,
            "_stage_app_bundle",
            lambda *a, **k: restaged.setdefault("staged", True),
        )

        argv = ["agent", "up", "-c", str(cfg)]
        if overwrite:
            argv.append("--overwrite")
        opts = parse_args(argv)
        cli.handle_agent_command(opts)
        return restaged.get("staged", False)

    def test_restages_when_manifest_checksum_differs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A checksum that cannot match the current config -> stale -> re-stage.
        assert (
            self._run_deploy(tmp_path, monkeypatch, marker_checksum="stale-hash")
            is True
        )

    def test_deploys_in_place_when_checksum_matches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Stamp the marker with the CURRENT config's real checksum -> not stale.
        # Resolve development exactly as the deploy path does (options.development
        # is None here -> auto-detect), so the seeded hash matches.
        from dao_ai.utils import resolve_use_local_source

        cfg = self._write_config(tmp_path)
        config = AppConfig.from_file(str(cfg), initialize=False)
        current_checksum = cli._config_checksum(
            config, development=resolve_use_local_source(None)
        )
        assert (
            self._run_deploy(tmp_path, monkeypatch, marker_checksum=current_checksum)
            is False
        )

    def test_deploys_in_place_when_manifest_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Staged dir with no manifest -> no recorded checksum -> in place.
        assert self._run_deploy(tmp_path, monkeypatch, marker_checksum=None) is False

    def test_stale_default_dir_always_regenerates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Ephemeral model: a stale DEFAULT dir always re-stages — there is no
        # edit-detection guard, so the outcome is identical with or without
        # --overwrite (user intent lives in the config, not the staging dir).
        assert (
            self._run_deploy(tmp_path, monkeypatch, marker_checksum="stale")
            is True
        )
        assert (
            self._run_deploy(
                tmp_path, monkeypatch, marker_checksum="stale", overwrite=True
            )
            is True
        )


_STRICT_CFG = (
    "resources:\n  models:\n    m: &m\n      name: databricks-gpt-5-4-mini\n"
    "agents:\n  g: &g\n    name: g\n    description: d\n    model: *m\n"
    "    prompt: p\n"
    "app:\n  name: my_app\n  agents:\n    - *g\n"
)


@pytest.mark.unit
class TestStrictPrimitivesErrorWhenUnstaged:
    """Model B: only `up` builds. sync/start/down on an unstaged dir error with
    the exact next command — consistent across agent (apps/mcp/ms) and workflow.
    `up` on the same unstaged dir succeeds (auto-builds)."""

    def _cfg(self, tmp_path: Path) -> Path:
        cfg = tmp_path / "c.yaml"
        cfg.write_text(_STRICT_CFG)
        return cfg

    @pytest.mark.parametrize("verb", ["sync", "start", "down"])
    @pytest.mark.parametrize("mode", ["apps", "mcp", "model_serving"])
    def test_agent_primitive_errors_when_unstaged(
        self, verb: str, mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = self._cfg(tmp_path)
        out = tmp_path / "out"  # no databricks.yaml
        monkeypatch.setattr(cli, "_apply_profile_context", lambda p: None)
        # Guard must fire BEFORE any writer/driver — patch them to blow up if hit.
        monkeypatch.setattr(
            "dao_ai.apps.bundle.write_bundle",
            lambda *a, **k: pytest.fail("must not build a primitive"),
        )
        with patch.object(cli, "deploy_app_bundle") as dep, patch.object(
            cli, "_run_ms_job_bundle"
        ) as job:
            opts = parse_args(
                ["agent", verb, "-c", str(cfg), "-s", str(out), "--mode", mode]
            )
            with pytest.raises(SystemExit) as exc:
                cli.handle_agent_command(opts)
        assert exc.value.code == 1
        dep.assert_not_called()
        job.assert_not_called()

    @pytest.mark.parametrize("mode", ["apps", "mcp", "model_serving"])
    def test_agent_up_autobuilds_when_unstaged(
        self, mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = self._cfg(tmp_path)
        out = tmp_path / "out"
        built: dict[str, bool] = {}

        def fake_writer(config: object, bundle_dir: object, **kw: object) -> dict:
            built["yes"] = True
            import pathlib

            p = pathlib.Path(str(bundle_dir))
            p.mkdir(parents=True, exist_ok=True)
            (p / "databricks.yaml").write_text("bundle: {}\n")
            return {}

        monkeypatch.setattr(cli, "_apply_profile_context", lambda p: None)
        monkeypatch.setattr(cli, "detect_cloud_provider", lambda p: "aws")
        monkeypatch.setattr(AppConfig, "_resolve_all_resources", lambda self: None)
        monkeypatch.setattr(
            AppConfig, "assert_provided_params_satisfied", lambda self: None
        )
        monkeypatch.setattr(cli, "_resolve_job_dao_ai_dep", lambda *a, **k: "dao-ai")
        # _mode_writer imports the writer per mode at call time — patch each source.
        monkeypatch.setattr("dao_ai.apps.bundle.write_bundle", fake_writer)
        monkeypatch.setattr("dao_ai.mcp.generate.write_mcp_bundle", fake_writer)
        monkeypatch.setattr(
            "dao_ai.pipeline.bundle.write_model_serving_agent_bundle", fake_writer
        )
        with patch.object(cli, "deploy_app_bundle"), patch.object(
            cli, "_run_ms_job_bundle"
        ):
            opts = parse_args(
                ["agent", "up", "-c", str(cfg), "-s", str(out), "--mode", mode]
            )
            cli.handle_agent_command(opts)
        assert built.get("yes"), f"up --mode {mode} must auto-build when unstaged"

    @pytest.mark.parametrize("verb", ["sync", "start", "down"])
    def test_workflow_primitive_errors_when_unstaged(
        self, verb: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = self._cfg(tmp_path)
        monkeypatch.setattr(cli, "_apply_profile_context", lambda p: None)
        monkeypatch.setattr(cli, "detect_cloud_provider", lambda p: "aws")
        monkeypatch.setattr(
            "dao_ai.pipeline.bundle.write_pipeline_bundle",
            lambda *a, **k: pytest.fail("must not build a primitive"),
        )
        with patch.object(cli, "_exec_bundle_command"):
            opts = parse_args(
                ["workflow", verb, "-c", str(cfg), "-s", str(tmp_path / "out")]
            )
            with pytest.raises(SystemExit) as exc:
                cli.handle_workflow_command(opts)
        assert exc.value.code == 1

    def test_workflow_up_builds_exactly_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """workflow up = build → sync → start. It must stage ONCE (the deploy
        call builds; the run call passes stage=False), not twice."""
        cfg = self._cfg(tmp_path)
        out = tmp_path / "out"
        builds: list[int] = []

        def fake_writer(config: object, bundle_dir: object, **kw: object) -> dict:
            builds.append(1)
            import pathlib

            p = pathlib.Path(str(bundle_dir))
            p.mkdir(parents=True, exist_ok=True)
            (p / "databricks.yaml").write_text("bundle: {}\n")
            return {}

        monkeypatch.setattr(cli, "_apply_profile_context", lambda p: None)
        monkeypatch.setattr(cli, "detect_cloud_provider", lambda p: "aws")
        monkeypatch.setattr(AppConfig, "_resolve_all_resources", lambda self: None)
        monkeypatch.setattr(cli, "_resolve_job_dao_ai_dep", lambda *a, **k: "dao-ai")
        monkeypatch.setattr(cli, "_clean_default_staging_dir", lambda *a, **k: None)
        monkeypatch.setattr(cli, "_write_staging_manifest", lambda *a, **k: None)
        monkeypatch.setattr(
            "dao_ai.pipeline.bundle.write_pipeline_bundle", fake_writer
        )
        with patch.object(cli, "_exec_bundle_command"):
            opts = parse_args(["workflow", "up", "-c", str(cfg), "-s", str(out)])
            cli.handle_workflow_command(opts)
        assert len(builds) == 1, f"workflow up must build once, built {len(builds)}x"

    def test_workflow_up_skips_rebuild_on_unchanged_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Parity with agent up: a second `workflow up` against an unchanged
        config in a DEFAULT staging dir is an idempotent no-op skip (the writer
        is not called again), not an unconditional rebuild."""
        monkeypatch.setenv("DAO_AI_BUNDLE_DIR", str(tmp_path / "base"))
        cfg = self._cfg(tmp_path)
        builds: list[int] = []

        def fake_writer(config: object, bundle_dir: object, **kw: object) -> None:
            builds.append(1)
            import pathlib

            p = pathlib.Path(str(bundle_dir))
            p.mkdir(parents=True, exist_ok=True)
            (p / "databricks.yaml").write_text("bundle: {}\n")

        monkeypatch.setattr(cli, "_apply_profile_context", lambda p: None)
        monkeypatch.setattr(cli, "detect_cloud_provider", lambda p: "aws")
        monkeypatch.setattr(AppConfig, "_resolve_all_resources", lambda self: None)
        monkeypatch.setattr(cli, "_resolve_job_dao_ai_dep", lambda *a, **k: "dao-ai")
        monkeypatch.setattr(
            "dao_ai.pipeline.bundle.write_pipeline_bundle", fake_writer
        )
        with patch.object(cli, "_exec_bundle_command"):
            # First up builds; second up (same config) must skip the rebuild.
            for _ in range(2):
                opts = parse_args(["workflow", "up", "-c", str(cfg)])
                cli.handle_workflow_command(opts)
        assert len(builds) == 1, (
            f"workflow up must skip rebuild on unchanged config; built {len(builds)}x"
        )

    def test_agent_build_skips_and_overwrite_forces(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`agent build` is idempotent like `workflow build`: a second build of an
        unchanged config in a DEFAULT dir skips (writer not re-invoked); passing
        --overwrite forces a rebuild."""
        monkeypatch.setenv("DAO_AI_BUNDLE_DIR", str(tmp_path / "base"))
        cfg = self._cfg(tmp_path)
        builds: list[int] = []

        def fake_stage(*a: object, **k: object) -> None:
            builds.append(1)

        monkeypatch.setattr(cli, "_apply_profile_context", lambda p: None)
        monkeypatch.setattr(cli, "detect_cloud_provider", lambda p: "aws")
        monkeypatch.setattr(AppConfig, "_resolve_all_resources", lambda self: None)
        monkeypatch.setattr(
            AppConfig, "assert_provided_params_satisfied", lambda self: None
        )
        # Stamp a manifest so the 2nd/3rd builds see a "current" dir. The real
        # _stage_app_bundle is replaced, so emulate its manifest write here.
        orig_checksum = cli._config_checksum

        def stage_and_stamp(config, bundle_dir, *, is_default_dir, checksum, **k):  # type: ignore[no-untyped-def]
            builds.append(1)
            bundle_dir.mkdir(parents=True, exist_ok=True)
            (bundle_dir / "databricks.yaml").write_text("bundle: {}\n")
            cli._write_staging_manifest(
                bundle_dir, is_default=is_default_dir, checksum=checksum
            )

        monkeypatch.setattr(cli, "_stage_app_bundle", stage_and_stamp)
        monkeypatch.setattr(cli, "_config_checksum", orig_checksum)

        # 1st build stages; 2nd (unchanged) skips; 3rd with --overwrite forces.
        for argv in (
            ["agent", "build", "-c", str(cfg)],
            ["agent", "build", "-c", str(cfg)],
            ["agent", "build", "-c", str(cfg), "--overwrite"],
        ):
            cli.handle_agent_command(parse_args(argv))
        assert len(builds) == 2, (
            f"expected build, skip, force = 2 stages; got {len(builds)}"
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

    def test_up_wait_calls_ready_poller_for_app(self, tmp_path: Path) -> None:
        # `up --wait` (wait_timeout set) blocks on the App readiness poller with
        # kind="app" and the app_resource_name (my_app -> my-app).
        cfg = _app_config(trace=False)
        with (
            patch.object(cli, "_exec_bundle_command"),
            patch.object(cli, "_link_and_grant_trace"),
            patch.object(cli, "_wait_for_resource_ready") as ready,
        ):
            deploy_app_bundle(
                cfg,
                staging_dir=tmp_path,
                deploy=True,
                run=True,
                destroy=False,
                profile="fevm",
                wait_timeout=120,
            )
        ready.assert_called_once_with("app", "my-app", "fevm", 120)

    def test_up_without_wait_does_not_poll(self, tmp_path: Path) -> None:
        cfg = _app_config(trace=False)
        with (
            patch.object(cli, "_exec_bundle_command"),
            patch.object(cli, "_link_and_grant_trace"),
            patch.object(cli, "_wait_for_resource_ready") as ready,
        ):
            deploy_app_bundle(
                cfg,
                staging_dir=tmp_path,
                deploy=True,
                run=True,
                destroy=False,
                profile="fevm",
                wait_timeout=None,
            )
        ready.assert_not_called()


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
            self._ms_config("my_ep"), profile=None, dry_run=False, wait_timeout=None
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
            lambda *a, **k: order.append("endpoint_delete"),
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

    def test_ms_up_wait_calls_ready_poller_for_endpoint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_run_ms_job_bundle(run=True, wait_timeout=N) blocks on the endpoint
        readiness poller with kind="endpoint" and the endpoint name."""
        monkeypatch.setattr(cli, "_exec_job_bundle", lambda **k: None)
        monkeypatch.setattr(cli, "_resolve_job_dao_ai_dep", lambda *a, **k: "dao-ai")
        monkeypatch.setattr(cli, "detect_cloud_provider", lambda p: "aws")
        monkeypatch.setattr(cli, "_print_endpoint_link", lambda _n: None)
        with patch.object(cli, "_wait_for_resource_ready") as ready:
            cli._run_ms_job_bundle(
                self._ms_config("my_ep"),
                staging_dir=tmp_path,
                deploy=True,
                run=True,
                destroy=False,
                dry_run=False,
                profile="fevm",
                development=None,
                target=None,
                cloud=None,
                config_vars={},
                wait_timeout=200,
            )
        ready.assert_called_once_with("endpoint", "my_ep", "fevm", 200)


@pytest.mark.unit
class TestDeleteApp:
    """_delete_app removes the imperatively-deployed App on workflow down."""

    @staticmethod
    def _app_config(name: str = "My_App") -> AppConfig:
        class _App:
            def __init__(self, n: str) -> None:
                self.name = n

            @property
            def app_resource_name(self) -> str:
                return self.name.lower().replace("_", "-")

        return AppConfig.model_construct(app=_App(name))

    def test_deletes_app_by_normalized_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        deleted: list[str] = []

        class _WC:
            class apps:
                @staticmethod
                def delete(name: str) -> None:
                    deleted.append(name)

        import databricks.sdk as sdk

        monkeypatch.setattr(sdk, "WorkspaceClient", lambda *a, **k: _WC())
        cli._delete_app(
            self._app_config("My_App"), profile=None, dry_run=False, wait_timeout=None
        )
        # App name is the lower/hyphen form the App was deployed under.
        assert deleted == ["my-app"]

    def test_dry_run_does_not_delete(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called: list[str] = []

        class _WC:
            class apps:
                @staticmethod
                def delete(name: str) -> None:
                    called.append(name)

        import databricks.sdk as sdk

        monkeypatch.setattr(sdk, "WorkspaceClient", lambda *a, **k: _WC())
        cli._delete_app(self._app_config(), profile=None, dry_run=True)
        assert called == [], "dry-run must not delete the app"

    def test_missing_app_is_not_an_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from databricks.sdk.errors import NotFound

        class _WC:
            class apps:
                @staticmethod
                def delete(name: str) -> None:
                    raise NotFound("gone")

        import databricks.sdk as sdk

        monkeypatch.setattr(sdk, "WorkspaceClient", lambda *a, **k: _WC())
        # Must not raise — an already-deleted app is fine.
        cli._delete_app(self._app_config(), profile=None, dry_run=False)

    def test_delete_failure_is_non_fatal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _WC:
            class apps:
                @staticmethod
                def delete(name: str) -> None:
                    raise RuntimeError("boom")

        import databricks.sdk as sdk

        monkeypatch.setattr(sdk, "WorkspaceClient", lambda *a, **k: _WC())
        # Best-effort: any failure is logged, not raised.
        cli._delete_app(self._app_config(), profile=None, dry_run=False)


@pytest.mark.unit
class TestWorkflowDownRemovesDeployedAgent:
    """workflow down destroys the Job THEN removes the imperatively-deployed
    App (apps/mcp) or serving endpoint (model_serving)."""

    _CFG = (
        "resources:\n  models:\n    m: &m\n      name: databricks-gpt-5-4-mini\n"
        "agents:\n  g: &g\n    name: g\n    description: d\n    model: *m\n"
        "    prompt: p\n"
        "app:\n  name: my_app\n  agents:\n    - *g\n"
    )

    def _cfg(self, tmp_path: Path) -> Path:
        cfg = tmp_path / "c.yaml"
        cfg.write_text(self._CFG)
        (tmp_path / "databricks.yaml").write_text("bundle: {}\n")  # pretend staged
        return cfg

    def _run_down(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, mode_argv: list[str]
    ) -> list[str]:
        order: list[str] = []
        monkeypatch.setattr(cli, "_apply_profile_context", lambda p: None)
        monkeypatch.setattr(cli, "detect_cloud_provider", lambda p: "aws")
        monkeypatch.setattr(cli, "_resolve_job_dao_ai_dep", lambda *a, **k: "dao-ai")
        monkeypatch.setattr(
            cli, "_exec_job_bundle", lambda **k: order.append("bundle_destroy")
        )
        monkeypatch.setattr(
            cli,
            "_delete_app",
            lambda *a, **k: order.append("app_delete"),
        )
        monkeypatch.setattr(
            cli,
            "_delete_serving_endpoint",
            lambda *a, **k: order.append("endpoint_delete"),
        )
        cfg = self._cfg(tmp_path)
        argv = ["workflow", "down", "-c", str(cfg), "-s", str(tmp_path)] + mode_argv
        opts = parse_args(argv)
        cli.handle_workflow_command(opts)
        return order

    def test_apps_mode_deletes_app_after_destroy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        order = self._run_down(tmp_path, monkeypatch, mode_argv=["--mode", "apps"])
        assert order == ["bundle_destroy", "app_delete"], order

    def test_no_mode_defaults_to_app_delete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Unspecified mode → the workflow deployed apps (notebook default), so
        # cleanup must target the App, NOT a serving endpoint.
        order = self._run_down(tmp_path, monkeypatch, mode_argv=[])
        assert order == ["bundle_destroy", "app_delete"], order

    def test_model_serving_deletes_endpoint_after_destroy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        order = self._run_down(
            tmp_path, monkeypatch, mode_argv=["--mode", "model_serving"]
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

    def test_up_autogenerates_when_unstaged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`agent up` on an empty dir stages first, then deploys the bundle.
        (Only `up` builds; `sync` on an unstaged dir errors — see
        TestStrictPrimitivesErrorWhenUnstaged.)"""
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
            opts = parse_args(["agent", "up", "-c", str(cfg), "-s", str(out)])
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

    def test_direct_apps_up_wait_gates_on_health(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`agent up --direct --wait` (apps) must block on the App readiness poller
        AFTER deploy_agent — deploy_and_wait only reaches deployment SUCCEEDED, but
        the app process can still 502, so we gate on compute ACTIVE + /health 200.
        """
        cfg = self._write_cfg(tmp_path)
        monkeypatch.setattr(cli, "_apply_profile_context", lambda p: None)
        monkeypatch.setattr(
            "dao_ai.config.AppConfig.deploy_agent",
            lambda self_config, target=None, development=None: None,
        )
        with (
            patch.object(cli, "deploy_app_bundle"),
            patch.object(cli, "_exec_bundle_command"),
            patch.object(cli, "_wait_for_resource_ready") as ready,
        ):
            opts = parse_args(
                ["agent", "up", "-c", str(cfg), "--direct", "--wait", "120"]
            )
            cli.handle_agent_command(opts)
        ready.assert_called_once_with("app", "my-app", None, 120)

    def test_direct_apps_up_without_wait_does_not_poll(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = self._write_cfg(tmp_path)
        monkeypatch.setattr(cli, "_apply_profile_context", lambda p: None)
        monkeypatch.setattr(
            "dao_ai.config.AppConfig.deploy_agent",
            lambda self_config, target=None, development=None: None,
        )
        with (
            patch.object(cli, "deploy_app_bundle"),
            patch.object(cli, "_exec_bundle_command"),
            patch.object(cli, "_wait_for_resource_ready") as ready,
        ):
            opts = parse_args(["agent", "up", "-c", str(cfg), "--direct"])
            cli.handle_agent_command(opts)
        ready.assert_not_called()

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
                    "up",
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
            opts = parse_args(["agent", "up", "-c", str(cfg), "-s", str(out)])
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


@pytest.mark.unit
class TestWaitForResourceDeleted:
    """`_wait_for_resource_deleted` polls get() until NotFound, else times out."""

    def _patch_ws(
        self, monkeypatch: pytest.MonkeyPatch, kind: str, get_side_effect
    ) -> None:
        """Install a fake WorkspaceClient whose <kind>.get uses get_side_effect."""
        from unittest.mock import MagicMock

        fake = MagicMock()
        getter = fake.apps.get if kind == "app" else fake.serving_endpoints.get
        getter.side_effect = get_side_effect
        monkeypatch.setattr(cli, "_apply_profile_context", lambda p: None)
        # Patch the symbols the helper imports lazily (databricks.sdk.WorkspaceClient
        # + the stdlib time.sleep) at their source modules.
        import time as _time

        import databricks.sdk as _sdk

        monkeypatch.setattr(_sdk, "WorkspaceClient", lambda *a, **k: fake)
        monkeypatch.setattr(_time, "sleep", lambda *_a, **_k: None)

    @pytest.mark.parametrize("kind", ["app", "endpoint"])
    def test_returns_when_absent_after_polls(
        self, kind: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from databricks.sdk.errors import NotFound

        calls = {"n": 0}

        def _get(name: str):
            calls["n"] += 1
            if calls["n"] < 3:
                return object()  # still present
            raise NotFound("gone")

        self._patch_ws(monkeypatch, kind, _get)
        cli._wait_for_resource_deleted(kind, "res", profile=None, timeout_seconds=60)
        assert calls["n"] == 3

    @pytest.mark.parametrize("kind", ["app", "endpoint"])
    def test_returns_immediately_when_already_absent(
        self, kind: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from databricks.sdk.errors import NotFound

        self._patch_ws(monkeypatch, kind, NotFound("gone"))
        cli._wait_for_resource_deleted(kind, "res", profile=None, timeout_seconds=60)

    def test_timeout_warns_and_returns_when_never_deleted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A resource that never disappears must be POLLED (>=1 get) and then time
        # out gracefully (warn + return, no hang/raise). Drive a fake monotonic
        # clock so the deadline is reached after a couple of real poll iterations
        # without waiting wall-clock time.
        calls = {"n": 0}

        def _get(name: str):  # never raises NotFound -> "still present"
            calls["n"] += 1
            return object()

        self._patch_ws(monkeypatch, "app", _get)
        import time as _time

        ticks = iter([0.0, 0.5, 1.0, 5.0, 999.0])  # advances past a 2s timeout
        monkeypatch.setattr(_time, "monotonic", lambda: next(ticks))
        cli._wait_for_resource_deleted("app", "res", profile=None, timeout_seconds=2)
        assert calls["n"] >= 1, "must poll at least once before timing out"

    def test_unknown_kind_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A bad kind must fail loud, not silently poll the endpoint getter.
        self._patch_ws(monkeypatch, "app", lambda name: object())
        with pytest.raises(ValueError, match="unknown resource kind"):
            cli._wait_for_resource_deleted("App", "res", profile=None, timeout_seconds=1)


@pytest.mark.unit
class TestWaitForResourceReady:
    """`_wait_for_resource_ready` blocks until servable; exits non-zero on
    terminal failure/timeout. Apps: SDK compute-ACTIVE then GET /health 200.
    Model Serving: endpoint READY + served-model DEPLOYMENT_READY (no HTTP)."""

    @staticmethod
    def _app(compute=None, app_state=None, deployment=None, url="https://app.example"):
        from types import SimpleNamespace

        return SimpleNamespace(
            url=url,
            compute_status=SimpleNamespace(state=compute, message=""),
            app_status=SimpleNamespace(state=app_state, message=""),
            active_deployment=SimpleNamespace(status=SimpleNamespace(state=deployment)),
        )

    @staticmethod
    def _endpoint(ready=None, config_update=None, deployments=()):
        from types import SimpleNamespace

        served = [
            SimpleNamespace(state=SimpleNamespace(deployment=d))
            for d in deployments
        ]
        return SimpleNamespace(
            state=SimpleNamespace(ready=ready, config_update=config_update),
            config=SimpleNamespace(served_entities=served),
        )

    def _patch(self, monkeypatch, *, app_get=None, ep_get=None, http_get=None):
        from unittest.mock import MagicMock

        fake = MagicMock()
        fake.config.authenticate.return_value = {"Authorization": "Bearer x"}
        if app_get is not None:
            fake.apps.get.side_effect = app_get
        if ep_get is not None:
            fake.serving_endpoints.get.side_effect = ep_get
        monkeypatch.setattr(cli, "_apply_profile_context", lambda p: None)
        import time as _time

        import databricks.sdk as _sdk

        monkeypatch.setattr(_sdk, "WorkspaceClient", lambda *a, **k: fake)
        monkeypatch.setattr(_time, "sleep", lambda *_a, **_k: None)
        if http_get is not None:
            import httpx

            monkeypatch.setattr(httpx, "get", http_get)
        return fake

    # --- Apps ---------------------------------------------------------------
    def test_app_ready_when_compute_active_and_health_200(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from databricks.sdk.service.apps import ApplicationState, ComputeState

        # compute becomes ACTIVE on 2nd poll, then /health 200 on 2nd HTTP poll.
        app_calls = {"n": 0}

        def _app_get(name: str):
            app_calls["n"] += 1
            compute = (
                ComputeState.ACTIVE if app_calls["n"] >= 2 else ComputeState.STARTING
            )
            return self._app(compute=compute, app_state=ApplicationState.RUNNING)

        http_calls = {"n": 0}

        def _http(url, **kw):
            http_calls["n"] += 1
            from types import SimpleNamespace

            return SimpleNamespace(status_code=200 if http_calls["n"] >= 2 else 503)

        fake = self._patch(monkeypatch, app_get=_app_get, http_get=_http)
        cli._wait_for_resource_ready("app", "my-app", profile=None, timeout_seconds=60)
        # Health was polled at the resolved app URL with auth headers.
        assert http_calls["n"] >= 2
        assert fake.config.authenticate.called

    def test_app_crashed_exits_nonzero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from databricks.sdk.service.apps import ApplicationState, ComputeState

        def _app_get(name: str):
            return self._app(
                compute=ComputeState.ACTIVE, app_state=ApplicationState.CRASHED
            )

        # No http_get: must fail before/without needing a 200.
        self._patch(monkeypatch, app_get=_app_get, http_get=lambda *a, **k: None)
        with pytest.raises(SystemExit):
            cli._wait_for_resource_ready(
                "app", "my-app", profile=None, timeout_seconds=60
            )

    def test_app_compute_never_active_times_out(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from databricks.sdk.service.apps import ApplicationState, ComputeState

        def _app_get(name: str):
            return self._app(
                compute=ComputeState.STARTING, app_state=ApplicationState.DEPLOYING
            )

        self._patch(monkeypatch, app_get=_app_get)
        import time as _time

        ticks = iter([0.0, 0.5, 1.0, 5.0, 999.0])
        monkeypatch.setattr(_time, "monotonic", lambda: next(ticks))
        with pytest.raises(SystemExit):
            cli._wait_for_resource_ready(
                "app", "my-app", profile=None, timeout_seconds=2
            )

    def test_app_health_never_200_times_out(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from databricks.sdk.service.apps import ApplicationState, ComputeState

        def _app_get(name: str):
            return self._app(
                compute=ComputeState.ACTIVE, app_state=ApplicationState.RUNNING
            )

        def _http(url, **kw):
            from types import SimpleNamespace

            return SimpleNamespace(status_code=503)

        self._patch(monkeypatch, app_get=_app_get, http_get=_http)
        import time as _time

        ticks = iter([0.0, 0.1, 0.2, 0.3, 1.0, 5.0, 999.0])
        monkeypatch.setattr(_time, "monotonic", lambda: next(ticks))
        with pytest.raises(SystemExit):
            cli._wait_for_resource_ready(
                "app", "my-app", profile=None, timeout_seconds=2
            )

    # --- Model Serving ------------------------------------------------------
    def test_endpoint_ready_when_all_served_models_ready(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from databricks.sdk.service.serving import (
            EndpointStateConfigUpdate,
            EndpointStateReady,
            ServedModelStateDeployment,
        )

        calls = {"n": 0}

        def _ep_get(name: str):
            calls["n"] += 1
            if calls["n"] < 2:
                return self._endpoint(
                    ready=EndpointStateReady.NOT_READY,
                    config_update=EndpointStateConfigUpdate.IN_PROGRESS,
                    deployments=[ServedModelStateDeployment.DEPLOYMENT_CREATING],
                )
            return self._endpoint(
                ready=EndpointStateReady.READY,
                config_update=EndpointStateConfigUpdate.NOT_UPDATING,
                deployments=[ServedModelStateDeployment.DEPLOYMENT_READY],
            )

        self._patch(monkeypatch, ep_get=_ep_get)
        cli._wait_for_resource_ready("endpoint", "ep", profile=None, timeout_seconds=60)
        assert calls["n"] >= 2

    def test_endpoint_served_model_failed_exits_nonzero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from databricks.sdk.service.serving import (
            EndpointStateConfigUpdate,
            EndpointStateReady,
            ServedModelStateDeployment,
        )

        def _ep_get(name: str):
            return self._endpoint(
                ready=EndpointStateReady.NOT_READY,
                config_update=EndpointStateConfigUpdate.NOT_UPDATING,
                deployments=[ServedModelStateDeployment.DEPLOYMENT_FAILED],
            )

        self._patch(monkeypatch, ep_get=_ep_get)
        with pytest.raises(SystemExit):
            cli._wait_for_resource_ready(
                "endpoint", "ep", profile=None, timeout_seconds=60
            )

    def test_endpoint_ready_but_model_still_creating_keeps_polling(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Endpoint READY but served model still CREATING → NOT ready yet; must
        # keep polling (and here, time out rather than falsely returning).
        from databricks.sdk.service.serving import (
            EndpointStateConfigUpdate,
            EndpointStateReady,
            ServedModelStateDeployment,
        )

        calls = {"n": 0}

        def _ep_get(name: str):
            calls["n"] += 1
            return self._endpoint(
                ready=EndpointStateReady.READY,
                config_update=EndpointStateConfigUpdate.NOT_UPDATING,
                deployments=[ServedModelStateDeployment.DEPLOYMENT_CREATING],
            )

        self._patch(monkeypatch, ep_get=_ep_get)
        import time as _time

        ticks = iter([0.0, 0.5, 1.0, 5.0, 999.0])
        monkeypatch.setattr(_time, "monotonic", lambda: next(ticks))
        with pytest.raises(SystemExit):
            cli._wait_for_resource_ready(
                "endpoint", "ep", profile=None, timeout_seconds=2
            )
        assert calls["n"] >= 1

    def test_unknown_kind_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch(monkeypatch)
        with pytest.raises(ValueError, match="unknown resource kind"):
            cli._wait_for_resource_ready(
                "App", "res", profile=None, timeout_seconds=1
            )


@pytest.mark.unit
class TestWaitFlagParsing:
    """`--wait [SECONDS]` is opt-in on the `up` AND `down` verbs of BOTH nouns."""

    @pytest.mark.parametrize("noun", ["agent", "workflow"])
    @pytest.mark.parametrize("verb", ["up", "down"])
    def test_bare_wait_uses_default(self, noun: str, verb: str) -> None:
        opts = parse_args([noun, verb, "-c", "c.yaml", "--wait"])
        assert opts.wait == cli._DEFAULT_WAIT_SECONDS

    @pytest.mark.parametrize("noun", ["agent", "workflow"])
    @pytest.mark.parametrize("verb", ["up", "down"])
    def test_wait_explicit_count(self, noun: str, verb: str) -> None:
        opts = parse_args([noun, verb, "-c", "c.yaml", "--wait", "90"])
        assert opts.wait == 90

    @pytest.mark.parametrize("noun", ["agent", "workflow"])
    @pytest.mark.parametrize("verb", ["up", "down"])
    def test_without_wait_is_none(self, noun: str, verb: str) -> None:
        opts = parse_args([noun, verb, "-c", "c.yaml"])
        assert opts.wait is None
        # And the typed accessor reflects "don't wait".
        assert cli._wait_timeout_of(opts) is None

    @pytest.mark.parametrize("noun", ["agent", "workflow"])
    @pytest.mark.parametrize("verb", ["sync", "build", "start"])
    def test_non_up_down_verbs_reject_wait(self, noun: str, verb: str) -> None:
        with pytest.raises(SystemExit):
            parse_args([noun, verb, "-c", "c.yaml", "--wait"])

    @pytest.mark.parametrize("verb", ["sync", "build", "start"])
    def test_wait_timeout_of_none_for_verbs_without_flag(self, verb: str) -> None:
        # Non-up/down verbs never define --wait; the accessor returns None, not raise.
        opts = parse_args(["agent", verb, "-c", "c.yaml"])
        assert cli._wait_timeout_of(opts) is None


@pytest.mark.unit
class TestPurgeFlagParsing:
    """`--purge` is opt-in on the `down` verb of BOTH nouns."""

    @pytest.mark.parametrize("noun", ["agent", "workflow"])
    def test_down_purge_sets_true(self, noun: str) -> None:
        opts = parse_args([noun, "down", "-c", "c.yaml", "--purge"])
        assert opts.purge is True
        assert cli._purge_of(opts) is True

    @pytest.mark.parametrize("noun", ["agent", "workflow"])
    def test_down_without_purge_is_false(self, noun: str) -> None:
        opts = parse_args([noun, "down", "-c", "c.yaml"])
        assert opts.purge is False
        assert cli._purge_of(opts) is False

    @pytest.mark.parametrize("noun", ["agent", "workflow"])
    @pytest.mark.parametrize("verb", ["up", "sync", "build", "start"])
    def test_non_down_verbs_reject_purge(self, noun: str, verb: str) -> None:
        with pytest.raises(SystemExit):
            parse_args([noun, verb, "-c", "c.yaml", "--purge"])

    @pytest.mark.parametrize("verb", ["up", "sync", "build", "start"])
    def test_purge_of_false_for_verbs_without_flag(self, verb: str) -> None:
        # Non-down verbs never define --purge; the accessor must return False.
        opts = parse_args(["agent", verb, "-c", "c.yaml"])
        assert cli._purge_of(opts) is False


@pytest.mark.unit
class TestRestoreTrashedExperimentOnUpdate:
    """A tracked experiment trashed out-of-band is restored before deploy so the
    bundle UPDATE isn't blocked by `Cannot move node ... in trash folder`.

    These tests model the MLflow trash-RENAME: the deleted node's remote_state
    carries the tracked id and lifecycle_stage=="deleted" (its `name` is the
    ``Trash/[dev ...] <name>-<ts>`` path), and the fix keys on that id — NOT the
    clean name, which no longer resolves once trashed (the reverted bug).
    """

    _TRASHED_UPDATE = (
        "experiments",
        "my-app-experiment",
        {
            "action": "update",
            "new_state": {
                "value": {"name": "/Users/u@d.com/[dev u] my_app"}
            },
            "remote_state": {
                "experiment_id": "12345",
                "lifecycle_stage": "deleted",
                "name": "/Users/u@d.com/Trash/[dev u] my_app-2026-08-02 01:33:48",
            },
        },
    )

    def _run_adopt(self, tmp_path: Path, *, plan: str):
        client = MagicMock()
        with patch.object(
            cli.subprocess, "run",
            side_effect=lambda cmd, **kw: _completed(0, stdout=plan),
        ), patch("mlflow.MlflowClient", return_value=client):
            cli._adopt_untracked_bundle_resources(
                staging_dir=tmp_path, profile="fevm", target="dev",
            )
        return client

    def test_trashed_update_restores_by_id(self, tmp_path: Path) -> None:
        client = self._run_adopt(tmp_path, plan=_plan_json(self._TRASHED_UPDATE))
        # Restored by the tracked id from remote_state, never the clean name.
        client.restore_experiment.assert_called_once_with("12345")

    def test_live_update_not_restored(self, tmp_path: Path) -> None:
        # Same UPDATE but the remote is ACTIVE — nothing to un-trash.
        live = (
            "experiments",
            "my-app-experiment",
            {
                "action": "update",
                "remote_state": {
                    "experiment_id": "12345",
                    "lifecycle_stage": "active",
                    "name": "/Users/u@d.com/[dev u] my_app",
                },
            },
        )
        client = self._run_adopt(tmp_path, plan=_plan_json(live))
        client.restore_experiment.assert_not_called()

    def test_create_experiment_not_restored(self, tmp_path: Path) -> None:
        # A CREATE (fresh deploy, state cleared by destroy) never restores.
        create = (
            "experiments",
            "my-app-experiment",
            {
                "action": "create",
                "new_state": {"value": {"name": "/Users/u@d.com/my_app"}},
            },
        )
        client = self._run_adopt(tmp_path, plan=_plan_json(create))
        client.restore_experiment.assert_not_called()

    def test_restore_failure_is_swallowed(self, tmp_path: Path) -> None:
        client = MagicMock()
        client.restore_experiment.side_effect = RuntimeError("boom")
        with patch.object(
            cli.subprocess, "run",
            side_effect=lambda cmd, **kw: _completed(
                0, stdout=_plan_json(self._TRASHED_UPDATE)
            ),
        ), patch("mlflow.MlflowClient", return_value=client):
            # Must not raise — best-effort, so deploy still runs.
            cli._adopt_untracked_bundle_resources(
                staging_dir=tmp_path, profile="fevm", target="dev",
            )


@pytest.mark.unit
class TestPurgeExperiment:
    """`down --purge` permanently deletes experiment nodes by their (Trash) path,
    found via search_experiments — NOT a clean-name workspace.delete (the
    reverted bug, which no-oped because the clean path no longer exists)."""

    def _config(self, *, external_id: str | None = None):
        app = MagicMock()
        app.experiment = SimpleNamespace(id=external_id) if external_id else None
        # No trace_location by default, so the existing purge tests exercise only
        # the experiment-node path (OTEL cleanup is covered in its own class).
        app.trace_location = None
        cfg = MagicMock()
        cfg.app = app
        return cfg

    def test_purge_deletes_each_matching_node_by_path(self) -> None:
        w = MagicMock()
        client = MagicMock()
        trashed = SimpleNamespace(
            experiment_id="1",
            name="/Users/u@d.com/Trash/[dev u] my_app-2026-08-02 01:33:48",
        )
        active = SimpleNamespace(
            experiment_id="2", name="/Users/u@d.com/[dev u] my_app"
        )
        client.search_experiments.return_value = [trashed, active]
        provider = MagicMock()
        provider.experiment_name.return_value = "/Users/u@d.com/my_app"
        with patch("databricks.sdk.WorkspaceClient", return_value=w), patch(
            "mlflow.MlflowClient", return_value=client
        ), patch(
            "dao_ai.providers.databricks.DatabricksProvider", return_value=provider
        ):
            cli._purge_experiment(self._config(), profile="fevm")
        deleted = {c.args[0] for c in w.workspace.delete.call_args_list}
        assert deleted == {trashed.name, active.name}
        for c in w.workspace.delete.call_args_list:
            assert c.kwargs.get("recursive") is True

    def test_purge_does_not_touch_sibling_app_experiment(self) -> None:
        # Purging `my_app` must NOT delete a different app whose name merely shares
        # the prefix (`my_app_2`) — the LIKE prefilter matches it, the basename
        # regex must reject it.
        w = MagicMock()
        client = MagicMock()
        mine = SimpleNamespace(experiment_id="1", name="/Users/u@d.com/my_app")
        sibling = SimpleNamespace(experiment_id="2", name="/Users/u@d.com/my_app_2")
        sibling_trash = SimpleNamespace(
            experiment_id="3",
            name="/Users/u@d.com/Trash/[dev u] my_app_v2-2026-08-02 01:33:48",
        )
        client.search_experiments.return_value = [mine, sibling, sibling_trash]
        provider = MagicMock()
        provider.experiment_name.return_value = "/Users/u@d.com/my_app"
        with patch("databricks.sdk.WorkspaceClient", return_value=w), patch(
            "mlflow.MlflowClient", return_value=client
        ), patch(
            "dao_ai.providers.databricks.DatabricksProvider", return_value=provider
        ):
            cli._purge_experiment(self._config(), profile="fevm")
        deleted = {c.args[0] for c in w.workspace.delete.call_args_list}
        assert deleted == {mine.name}

    def test_purge_matches_hyphenated_apps_leaf(self) -> None:
        # apps/mcp DABs name the node with the HYPHENATED leaf; the underscored
        # clean leaf from experiment_name must still match it (separator-agnostic).
        w = MagicMock()
        client = MagicMock()
        node = SimpleNamespace(
            experiment_id="1", name="/Users/u@d.com/[dev u] ai-gateway-example"
        )
        client.search_experiments.return_value = [node]
        provider = MagicMock()
        provider.experiment_name.return_value = "/Users/u@d.com/ai_gateway_example"
        with patch("databricks.sdk.WorkspaceClient", return_value=w), patch(
            "mlflow.MlflowClient", return_value=client
        ), patch(
            "dao_ai.providers.databricks.DatabricksProvider", return_value=provider
        ):
            cli._purge_experiment(self._config(), profile="fevm")
        deleted = {c.args[0] for c in w.workspace.delete.call_args_list}
        assert deleted == {node.name}

    def test_purge_search_pattern_is_separator_agnostic(self) -> None:
        # apps/mcp DABs name the experiment with a HYPHENATED leaf while
        # model_serving/workflow use the UNDERSCORED one; the search LIKE pattern
        # must map both separators to the `_` wildcard so one query finds either.
        w = MagicMock()
        client = MagicMock()
        client.search_experiments.return_value = []
        provider = MagicMock()
        provider.experiment_name.return_value = "/Users/u@d.com/ai-gateway-example"
        with patch("databricks.sdk.WorkspaceClient", return_value=w), patch(
            "mlflow.MlflowClient", return_value=client
        ), patch(
            "dao_ai.providers.databricks.DatabricksProvider", return_value=provider
        ):
            cli._purge_experiment(self._config(), profile="fevm")
        (_, kwargs) = client.search_experiments.call_args
        assert kwargs["filter_string"] == "name LIKE '%ai_gateway_example%'"

    def test_purge_skips_external_experiment_id(self) -> None:
        w = MagicMock()
        with patch("databricks.sdk.WorkspaceClient", return_value=w):
            cli._purge_experiment(self._config(external_id="999"), profile="fevm")
        w.workspace.delete.assert_not_called()

    def test_purge_no_matches_is_noop(self) -> None:
        w = MagicMock()
        client = MagicMock()
        client.search_experiments.return_value = []
        provider = MagicMock()
        provider.experiment_name.return_value = "/Users/u@d.com/my_app"
        with patch("databricks.sdk.WorkspaceClient", return_value=w), patch(
            "mlflow.MlflowClient", return_value=client
        ), patch(
            "dao_ai.providers.databricks.DatabricksProvider", return_value=provider
        ):
            cli._purge_experiment(self._config(), profile="fevm")
        w.workspace.delete.assert_not_called()

    def test_purge_delete_failure_is_swallowed(self) -> None:
        w = MagicMock()
        w.workspace.delete.side_effect = RuntimeError("boom")
        client = MagicMock()
        client.search_experiments.return_value = [
            SimpleNamespace(experiment_id="1", name="/Users/u@d.com/Trash/x")
        ]
        provider = MagicMock()
        provider.experiment_name.return_value = "/Users/u@d.com/my_app"
        with patch("databricks.sdk.WorkspaceClient", return_value=w), patch(
            "mlflow.MlflowClient", return_value=client
        ), patch(
            "dao_ai.providers.databricks.DatabricksProvider", return_value=provider
        ):
            # Must not raise.
            cli._purge_experiment(self._config(), profile="fevm")


@pytest.mark.unit
class TestPurgeOtelTraceTables:
    """`down --purge` also drops the app's OTEL trace tables — but ONLY when the
    prefix is per-experiment (unset ``table_prefix``). An explicit prefix may be
    shared across agents, so it's left in place with a manual-drop hint."""

    @staticmethod
    def _config(*, trace_location, external_id: str | None = None):
        app = MagicMock()
        app.experiment = SimpleNamespace(id=external_id) if external_id else None
        app.trace_location = trace_location
        cfg = MagicMock()
        cfg.app = app
        return cfg

    @staticmethod
    def _trace_location(*, prefix, catalog="cat", schema="sch"):
        loc = MagicMock()
        loc.resolved_table_prefix = prefix
        loc.catalog_name = catalog
        loc.schema_name = schema
        return loc

    def test_unset_prefix_drops_per_experiment_tables(self) -> None:
        cfg = self._config(trace_location=self._trace_location(prefix=None))
        with patch(
            "dao_ai.providers.databricks._drop_uc_otel_tables"
        ) as drop:
            cli._purge_otel_trace_tables(cfg, ["111", "222"], profile="fevm")
        # One drop call per purged experiment id, keyed on that id as the prefix.
        assert drop.call_count == 2
        called_prefixes = {c.args[2] for c in drop.call_args_list}
        assert called_prefixes == {"111", "222"}
        for c in drop.call_args_list:
            assert c.args[0] == "cat" and c.args[1] == "sch"  # catalog, schema

    def test_explicit_prefix_does_not_drop_but_logs(self) -> None:
        cfg = self._config(trace_location=self._trace_location(prefix="shared_x"))
        with patch(
            "dao_ai.providers.databricks._drop_uc_otel_tables"
        ) as drop, patch.object(cli.logger, "warning") as warn:
            cli._purge_otel_trace_tables(cfg, ["111"], profile="fevm")
        drop.assert_not_called()
        # The four shared table names are surfaced for manual cleanup, with EACH
        # identifier segment backticked (`cat`.`sch`.`tbl`) — NOT the whole dotted
        # name in one pair, which SQL parses as a single literal identifier and
        # silently no-ops. This is the runnable form.
        logged = " ".join(str(c.args[0]) for c in warn.call_args_list)
        for suffix in ("spans", "logs", "metrics", "annotations"):
            assert f"`cat`.`sch`.`shared_x_otel_{suffix}`" in logged
        # Guard against a regression to the broken single-backtick-pair form.
        assert "`cat.sch." not in logged

    def test_no_trace_location_is_noop(self) -> None:
        cfg = self._config(trace_location=None)
        with patch("dao_ai.providers.databricks._drop_uc_otel_tables") as drop:
            cli._purge_otel_trace_tables(cfg, ["111"], profile="fevm")
        drop.assert_not_called()

    def test_no_purged_ids_is_noop(self) -> None:
        cfg = self._config(trace_location=self._trace_location(prefix=None))
        with patch("dao_ai.providers.databricks._drop_uc_otel_tables") as drop:
            cli._purge_otel_trace_tables(cfg, [], profile="fevm")
        drop.assert_not_called()

    def test_drop_failure_is_swallowed(self) -> None:
        cfg = self._config(trace_location=self._trace_location(prefix=None))
        with patch(
            "dao_ai.providers.databricks._drop_uc_otel_tables",
            side_effect=RuntimeError("boom"),
        ):
            # Must not raise — best-effort so `down` still completes.
            cli._purge_otel_trace_tables(cfg, ["111"], profile="fevm")

    def test_purge_experiment_passes_purged_ids_to_otel_cleanup(self) -> None:
        # Integration: _purge_experiment forwards the ids it actually deleted.
        w = MagicMock()
        client = MagicMock()
        node = SimpleNamespace(experiment_id="777", name="/Users/u@d.com/my_app")
        client.search_experiments.return_value = [node]
        provider = MagicMock()
        provider.experiment_name.return_value = "/Users/u@d.com/my_app"
        cfg = self._config(trace_location=self._trace_location(prefix=None))
        with patch("databricks.sdk.WorkspaceClient", return_value=w), patch(
            "mlflow.MlflowClient", return_value=client
        ), patch(
            "dao_ai.providers.databricks.DatabricksProvider", return_value=provider
        ), patch.object(cli, "_purge_otel_trace_tables") as otel:
            cli._purge_experiment(cfg, profile="fevm")
        otel.assert_called_once()
        assert otel.call_args.args[1] == ["777"]