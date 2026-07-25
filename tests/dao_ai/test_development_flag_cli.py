"""Tests for the --development / --no-development tri-state across CLI commands.

The flag must resolve identically on every deploy-capable command:
``<noun> generate`` (agent/workflow) — ``--development`` -> True,
``--no-development`` -> False, omitted -> None (auto-detect). For the workflow
it is additionally forwarded to the deploy notebook as a
``--var development=auto|true|false`` bundle variable.

The removed commands ``deploy``, ``generate-agent``, ``generate-workflow``, and
``generate-mcp`` must now be rejected by the parser.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import call, patch

import pytest

from dao_ai import cli
from dao_ai.cli import deploy_app_bundle, parse_args, run_databricks_command
from dao_ai.config import AppConfig


def _base(cmd: str) -> list[str]:
    """Build argv for a command, splitting a nested `"agent generate"` on space."""
    return cmd.split() + ["-c", "config.yaml"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "command",
    [
        "workflow generate",
        "agent generate",
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
    @pytest.mark.parametrize("verb", ["generate", "deploy", "run", "destroy"])
    def test_nested_verbs_parse(self, noun: str, verb: str) -> None:
        opts = parse_args([noun, verb, "-c", "config.yaml"])
        assert opts.command == noun
        assert opts.subcommand == verb

    @pytest.mark.parametrize("noun", ["agent", "workflow"])
    def test_bare_noun_requires_verb(self, noun: str) -> None:
        with pytest.raises(SystemExit):
            parse_args([noun, "-c", "config.yaml"])

    def test_deploy_verb_accepts_run_chain(self) -> None:
        # `<noun> deploy --run` deploys then runs; run flag is on the deploy verb.
        assert parse_args(["agent", "deploy", "-c", "c.yaml", "--run"]).run is True

    @pytest.mark.parametrize(
        "argv",
        [
            ["deploy", "-c", "c.yaml"],
            ["generate-agent", "-c", "c.yaml"],
            ["generate-mcp", "-c", "c.yaml"],
            ["generate-workflow", "-c", "c.yaml"],
            ["generate-bundle", "-c", "c.yaml"],
            ["pipeline", "-c", "c.yaml"],
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

    def test_default_base_when_env_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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
            assert cli._default_bundle_dir(kind, "app") == Path(
                f"artifacts/{kind}/app"
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
        cli._write_staging_marker(under, is_default=True)  # mark as our output
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
        (d / "app.yaml").write_text("orig\n")
        cli._write_staging_marker(d, is_default=True)
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
        # Edit a file so its mtime post-dates the marker.
        marker_mtime = (d / cli._STAGING_MARKER).stat().st_mtime
        import os

        (d / "app.yaml").write_text("edited\n")
        os.utime(d / "app.yaml", (marker_mtime + 10, marker_mtime + 10))
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
        cli._clean_default_staging_dir(
            d, is_default=True, overwrite=True, noun="agent"
        )
        assert not d.exists()

    def test_missing_marker_treated_as_edited(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DAO_AI_BUNDLE_DIR", str(tmp_path / "base"))
        d = tmp_path / "base" / "agent" / "app"
        d.mkdir(parents=True)
        (d / "app.yaml").write_text("user file\n")  # no marker written
        with pytest.raises(SystemExit):
            cli._clean_default_staging_dir(
                d, is_default=True, overwrite=False, noun="agent"
            )
        assert d.exists()


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
            output_dir=str(tmp_path / "out"),
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
            output_dir=str(out),
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
                output_dir=str(tmp_path / "nonexistent"),
                development=False,
                stage=False,
            )
        assert exc.value.code == 1


@pytest.mark.unit
class TestValidateBundleActionFlags:
    """``--destroy`` + ``--deploy``/``--run`` is contradictory and must be
    rejected uniformly across all three generate-* commands (previously the
    workflow handler deployed-then-destroyed while the App driver destroyed-only).
    """

    @pytest.mark.parametrize(
        "cmd", ["workflow generate", "agent generate"]
    )
    @pytest.mark.parametrize("bad", [["--deploy", "--destroy"], ["--run", "--destroy"], ["--deploy", "--run", "--destroy"]])
    def test_destroy_with_deploy_or_run_exits(self, cmd: str, bad: list[str]) -> None:
        options = parse_args(_base(cmd) + bad)
        with pytest.raises(SystemExit) as exc:
            cli._validate_bundle_action_flags(options)
        assert exc.value.code == 1

    @pytest.mark.parametrize(
        "cmd", ["workflow generate", "agent generate"]
    )
    @pytest.mark.parametrize("ok", [["--deploy"], ["--run"], ["--deploy", "--run"], ["--destroy"], []])
    def test_valid_combinations_pass(self, cmd: str, ok: list[str]) -> None:
        options = parse_args(_base(cmd) + ok)
        cli._validate_bundle_action_flags(options)  # must not raise


def _app_config(*, trace: bool) -> AppConfig:
    """Minimal AppConfig stub for deploy_app_bundle (needs app.name + trace_location)."""

    class _App:
        name = "my_app"
        trace_location = object() if trace else None

    return AppConfig.model_construct(app=_App())


@pytest.mark.unit
class TestModeChoices:
    def test_deploy_accepts_all_three(self) -> None:
        for m in ("model_serving", "apps", "mcp"):
            assert parse_args(["agent", "deploy", "-c", "c.yaml", "--mode", m]).mode == m

    def test_generate_rejects_model_serving(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["agent", "generate", "-c", "c.yaml", "--mode", "model_serving"])

    def test_generate_accepts_apps_and_mcp(self) -> None:
        for m in ("apps", "mcp"):
            assert parse_args(["agent", "generate", "-c", "c.yaml", "--mode", m]).mode == m

    def test_mode_defaults_to_apps(self) -> None:
        assert parse_args(["agent", "deploy", "-c", "c.yaml"]).mode == "apps"


@pytest.mark.unit
class TestWorkflowModeChoices:
    """workflow verbs must accept --mode with the same choices as agent verbs."""

    def test_workflow_deploy_accepts_all_three(self) -> None:
        for m in ("model_serving", "apps", "mcp"):
            assert (
                parse_args(["workflow", "deploy", "-c", "c.yaml", "--mode", m]).mode
                == m
            )

    def test_workflow_generate_accepts_apps_and_mcp(self) -> None:
        for m in ("apps", "mcp"):
            assert (
                parse_args(["workflow", "generate", "-c", "c.yaml", "--mode", m]).mode
                == m
            )

    def test_workflow_run_accepts_apps_and_mcp(self) -> None:
        for m in ("apps", "mcp"):
            assert (
                parse_args(["workflow", "run", "-c", "c.yaml", "--mode", m]).mode == m
            )

    def test_workflow_destroy_accepts_apps_and_mcp(self) -> None:
        for m in ("apps", "mcp"):
            assert (
                parse_args(["workflow", "destroy", "-c", "c.yaml", "--mode", m]).mode
                == m
            )

    def test_workflow_run_rejects_model_serving(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["workflow", "run", "-c", "c.yaml", "--mode", "model_serving"])

    def test_workflow_destroy_rejects_model_serving(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(
                ["workflow", "destroy", "-c", "c.yaml", "--mode", "model_serving"]
            )

    def test_workflow_generate_rejects_model_serving(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(
                ["workflow", "generate", "-c", "c.yaml", "--mode", "model_serving"]
            )


@pytest.mark.unit
class TestMcpNounRemoved:
    """The `mcp` noun has been removed; its modes are now routed via `agent --mode mcp`."""

    def test_mcp_noun_removed(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["mcp", "generate", "-c", "c.yaml"])


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
            ["agent", "generate", "-c", str(cfg), "-o", str(tmp_path / "out"), "--mode", "mcp"]
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
            ["agent", "generate", "-c", str(cfg), "-o", str(tmp_path / "out")]
        )
        cli.handle_agent_command(opts)

        assert called == {"apps": True}, (
            f"Expected only write_bundle to be called; got called={called}"
        )

    def test_agent_deploy_mcp_uses_mcp_staging_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """agent deploy --mode mcp passes the mcp-kind staging dir to deploy_app_bundle."""
        cfg = self._write_config(tmp_path)
        # Capture what _resolve_bundle_dir is asked to return: (Path, is_default).
        resolved_bundle_dirs: dict[str, tuple[str, Path]] = {}

        def mock_resolve_bundle_dir(
            kind: str, config: object, output_dir: str | None
        ) -> tuple[Path, bool]:
            # Record: kind -> (output_dir passed in, resolved path)
            mcp_staging = tmp_path / "staged" / kind / "test_app"
            mcp_staging.mkdir(parents=True, exist_ok=True)
            (mcp_staging / "databricks.yaml").write_text("bundle: {}\n")
            resolved_bundle_dirs[kind] = (output_dir or "default", mcp_staging)
            return mcp_staging, output_dir is None

        monkeypatch.setattr(cli, "_apply_profile_context", lambda p: None)

        with patch.object(cli, "_resolve_bundle_dir", side_effect=mock_resolve_bundle_dir):
            with patch.object(cli, "deploy_app_bundle"):
                opts = parse_args(
                    ["agent", "deploy", "-c", str(cfg), "--mode", "mcp"]
                )
                cli.handle_agent_command(opts)

        assert "mcp" in resolved_bundle_dirs, (
            "deploy must call _resolve_bundle_dir with kind='mcp' when --mode mcp"
        )
        _, resolved_path = resolved_bundle_dirs["mcp"]
        assert "/mcp/" in str(resolved_path), (
            f"Expected mcp staging dir path to contain /mcp/, got {resolved_path}"
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
                output_dir=tmp_path,
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
                output_dir=tmp_path,
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
                output_dir=tmp_path,
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
                output_dir=tmp_path,
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

    def test_endpoint_link_printed(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
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
            "resources": {
                "jobs": {"deploy_job": {"url": "https://ws/jobs/42?w=1"}}
            }
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

        monkeypatch.setattr("sys.argv", ["dao-ai", "agent", "generate", "-c", "c.yaml"])
        cli.main()
        assert seen == [("agent", "generate")]

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
            opts = parse_args(["agent", "deploy", "-c", str(cfg), "-o", str(out)])
            cli.handle_agent_command(opts)

        assert "wrote" not in wrote, "deploy must not regenerate"
        dep.assert_called_once()
        assert dep.call_args.kwargs["output_dir"] == out
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
            ["agent", "deploy", "-c", str(cfg), "-o", str(tmp_path / "nope")]
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
            (pathlib.Path(str(bundle_dir)) / "databricks.yaml").write_text("bundle: {}\n")

        monkeypatch.setattr(cli, "_apply_profile_context", lambda p: None)
        monkeypatch.setattr(
            "dao_ai.apps.bundle.write_bundle",
            fake_writer,
        )
        with patch.object(cli, "deploy_app_bundle") as dep:
            opts = parse_args(["agent", "deploy", "-c", str(cfg), "-o", str(out)])
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
            opts = parse_args(["agent", "deploy", "-c", str(cfg), "-o", str(out)])
            cli.handle_agent_command(opts)

        assert not wrote, "deploy must NOT regenerate when already staged"
        dep.assert_called_once()
        assert (out / "sentinel").read_text() == "keep\n"

    def test_direct_flag_uses_sdk_not_bundle(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`agent deploy --direct` calls deploy_agent (SDK path) without touching bundles."""
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
        with patch.object(cli, "deploy_app_bundle") as dep, patch.object(
            cli, "_exec_bundle_command"
        ) as exec_cmd:
            opts = parse_args(["agent", "deploy", "-c", str(cfg), "--direct"])
            cli.handle_agent_command(opts)

        assert deploy_agent_calls, "--direct must call deploy_agent"
        dep.assert_not_called()
        exec_cmd.assert_not_called()

    def test_model_serving_deploy_uses_provider_not_bundle(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`agent deploy --mode model_serving` routes to deploy_agent, not the bundle path."""
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
        with patch.object(cli, "deploy_app_bundle") as dep:
            opts = parse_args(["agent", "deploy", "-c", str(cfg), "--mode", "model_serving"])
            cli.handle_agent_command(opts)

        from dao_ai.config import DeploymentTarget

        assert deploy_agent_calls, "--mode model_serving must call deploy_agent"
        assert deploy_agent_calls[0]["target"] == DeploymentTarget.MODEL_SERVING
        dep.assert_not_called()

    def test_direct_flag_parses(self) -> None:
        """`--direct` is accepted on the deploy verb."""
        opts = parse_args(["agent", "deploy", "-c", "c.yaml", "--direct"])
        assert opts.direct is True

    def test_run_still_errors_when_unstaged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`agent run` on an empty dir must still error (auto-generate is deploy-only)."""
        cfg = self._write_cfg(tmp_path)
        monkeypatch.setattr(cli, "_apply_profile_context", lambda p: None)
        opts = parse_args(["agent", "run", "-c", str(cfg), "-o", str(tmp_path / "nope")])
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
            (pathlib.Path(str(bundle_dir)) / "databricks.yaml").write_text("bundle: {}\n")

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
            opts = parse_args(["agent", "deploy", "-c", str(cfg), "-o", str(out)])
            cli.handle_agent_command(opts)

        assert resolve_calls == [
            "called"
        ], "auto-generate path must call _resolve_all_resources"

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
            opts = parse_args(["agent", "deploy", "-c", str(cfg), "-o", str(out)])
            cli.handle_agent_command(opts)

        assert (
            resolve_calls == []
        ), "in-place deploy path must NOT call _resolve_all_resources"


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
            ["trace", "link", "-c", "c.yaml", "--experiment-id", "9999", "--app-sp", "uuid-1"]
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
            ["trace", "grant", "-c", "c.yaml", "--experiment-id", "8888", "--app-sp", "uuid-2"]
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
