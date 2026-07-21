"""Tests for the --development / --no-development tri-state across CLI commands.

The flag must resolve identically on every deploy-capable command:
``deploy``, ``generate-workflow``, ``generate-agent``, ``generate-mcp`` —
``--development`` -> True, ``--no-development`` -> False, omitted -> None
(auto-detect). For ``generate-workflow`` it is additionally forwarded to the
deploy notebook as a ``--var development=auto|true|false`` bundle variable.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import call, patch

import pytest

from dao_ai import cli
from dao_ai.cli import deploy_app_bundle, parse_args, run_databricks_command
from dao_ai.config import AppConfig


def _base(cmd: str) -> list[str]:
    return [cmd, "-c", "config.yaml"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "command",
    ["deploy", "generate-workflow", "generate-agent", "generate-mcp"],
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
class TestRemovedCommandAliases:
    """0.2.0 removed the old command names; only the new verbs are accepted."""

    def test_new_names_parse(self) -> None:
        assert parse_args(_base("generate-agent")).command == "generate-agent"
        assert (
            parse_args(_base("generate-workflow")).command == "generate-workflow"
        )

    @pytest.mark.parametrize("old", ["generate-bundle", "pipeline"])
    def test_old_names_rejected(self, old: str) -> None:
        # argparse rejects an unknown subcommand with SystemExit(2).
        with pytest.raises(SystemExit):
            parse_args(_base(old))


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
        cli._clean_default_staging_dir(under, is_default=True)
        assert not under.exists(), "path under env base must be cleaned"

        outside = tmp_path / "user_dir"
        outside.mkdir()
        (outside / "keep").write_text("x")
        cli._clean_default_staging_dir(outside, is_default=True)
        assert (outside / "keep").exists(), "path outside base must be protected"


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
            "app:\n  name: stage_only_app\n  deployment_target: apps\n  agents:\n    - *g\n"
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
            None, config=str(cfg), output_dir=str(tmp_path / "out")
        )
        patch.stopall()

        assert staged.get("wrote") is True, "bundle must be staged"
        exec_mock.assert_not_called()


@pytest.mark.unit
class TestValidateBundleActionFlags:
    """``--destroy`` + ``--deploy``/``--run`` is contradictory and must be
    rejected uniformly across all three generate-* commands (previously the
    workflow handler deployed-then-destroyed while the App driver destroyed-only).
    """

    @pytest.mark.parametrize("cmd", ["generate-workflow", "generate-agent", "generate-mcp"])
    @pytest.mark.parametrize("bad", [["--deploy", "--destroy"], ["--run", "--destroy"], ["--deploy", "--run", "--destroy"]])
    def test_destroy_with_deploy_or_run_exits(self, cmd: str, bad: list[str]) -> None:
        options = parse_args(_base(cmd) + bad)
        with pytest.raises(SystemExit) as exc:
            cli._validate_bundle_action_flags(options)
        assert exc.value.code == 1

    @pytest.mark.parametrize("cmd", ["generate-workflow", "generate-agent", "generate-mcp"])
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
