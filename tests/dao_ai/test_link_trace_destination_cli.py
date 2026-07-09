"""Unit tests for `dao-ai link-trace-destination` CLI verb.

Focus is on the CLI orchestration: argument parsing, experiment-id
resolution (explicit override, `app.experiment.resolved_id`, bundle-name
lookup), and clean stderr output on failure paths. The idempotent linker
itself is exercised by `test_trace_location_linkage.py` in PR #164.
"""

from unittest.mock import MagicMock, patch

import pytest


def _cfg_with_trace(
    catalog: str = "cat",
    schema: str = "sch",
    prefix: str | None = None,
    experiment_resolved_id: str | None = None,
    app_name: str = "my-app",
) -> MagicMock:
    """Build a mock AppConfig for the resolver / handler under test.

    Real AppConfig instantiation drags too many required fields (secrets,
    resources, etc.); the CLI handler only touches
    `config.app.{name,trace_location,experiment}`.
    """
    cfg = MagicMock()
    cfg.app = MagicMock()
    cfg.app.name = app_name
    cfg.app.trace_location.catalog_name = catalog
    cfg.app.trace_location.schema_name = schema
    cfg.app.trace_location.resolved_table_prefix = prefix
    if experiment_resolved_id is None:
        cfg.app.experiment = None
    else:
        cfg.app.experiment.resolved_id = experiment_resolved_id
    return cfg


@pytest.mark.unit
class TestResolveExperimentId:
    def test_override_wins(self) -> None:
        from dao_ai.cli import _resolve_experiment_id_for_link

        cfg = _cfg_with_trace(experiment_resolved_id="cfg_id")
        assert _resolve_experiment_id_for_link(cfg, "override_id") == "override_id"

    def test_uses_config_experiment_when_set(self) -> None:
        from dao_ai.cli import _resolve_experiment_id_for_link

        cfg = _cfg_with_trace(experiment_resolved_id="cfg_id")
        assert _resolve_experiment_id_for_link(cfg, None) == "cfg_id"

    def test_returns_none_when_experiment_resolved_id_missing(self, capsys) -> None:
        from dao_ai.cli import _resolve_experiment_id_for_link

        cfg = MagicMock()
        cfg.app = MagicMock()
        cfg.app.experiment.resolved_id = None
        assert _resolve_experiment_id_for_link(cfg, None) is None
        assert "resolved_id is None" in capsys.readouterr().err

    def test_falls_back_to_bundle_name_lookup_prod(self) -> None:
        """Prod-mode deploy: unprefixed experiment name matches first."""
        from dao_ai.cli import _resolve_experiment_id_for_link

        cfg = _cfg_with_trace(app_name="my_app")  # underscore → hyphen
        exp = MagicMock()
        exp.experiment_id = "bundle_id"
        with (
            patch("databricks.sdk.WorkspaceClient") as wc,
            patch("mlflow.tracking.MlflowClient") as mc,
        ):
            wc.return_value.current_user.me.return_value.user_name = "u@x.com"
            mc.return_value.get_experiment_by_name.return_value = exp
            assert _resolve_experiment_id_for_link(cfg, None) == "bundle_id"
            # First candidate is the unprefixed path.
            first_call = mc.return_value.get_experiment_by_name.call_args_list[0]
            assert first_call.args[0] == "/Users/u@x.com/my-app"

    def test_falls_back_to_dev_prefixed_name(self) -> None:
        """DABs `--target dev` prefix: first lookup misses, second hits."""
        from dao_ai.cli import _resolve_experiment_id_for_link

        cfg = _cfg_with_trace(app_name="my_app")
        exp = MagicMock()
        exp.experiment_id = "dev_bundle_id"
        with (
            patch("databricks.sdk.WorkspaceClient") as wc,
            patch("mlflow.tracking.MlflowClient") as mc,
        ):
            wc.return_value.current_user.me.return_value.user_name = "u@x.com"
            # Return None for unprefixed, actual exp for dev-prefixed.
            mc.return_value.get_experiment_by_name.side_effect = [None, exp]
            assert _resolve_experiment_id_for_link(cfg, None) == "dev_bundle_id"
            calls = mc.return_value.get_experiment_by_name.call_args_list
            assert calls[0].args[0] == "/Users/u@x.com/my-app"
            # dev-tag: split '@', lowercase, non-alnum → underscore.
            assert calls[1].args[0] == "/Users/u@x.com/[dev u] my-app"

    def test_bundle_lookup_missing_returns_none(self, capsys) -> None:
        from dao_ai.cli import _resolve_experiment_id_for_link

        cfg = _cfg_with_trace()
        with (
            patch("databricks.sdk.WorkspaceClient") as wc,
            patch("mlflow.tracking.MlflowClient") as mc,
        ):
            wc.return_value.current_user.me.return_value.user_name = "u@x.com"
            # Both candidates miss.
            mc.return_value.get_experiment_by_name.return_value = None
            assert _resolve_experiment_id_for_link(cfg, None) is None
        err = capsys.readouterr().err
        assert "not found" in err
        assert "databricks bundle deploy" in err
        # Both candidates surfaced in the diagnostic
        assert "[dev " in err

    def test_bundle_lookup_error_returns_none(self, capsys) -> None:
        from dao_ai.cli import _resolve_experiment_id_for_link

        cfg = _cfg_with_trace()
        with patch("databricks.sdk.WorkspaceClient") as wc:
            wc.side_effect = RuntimeError("no auth")
            assert _resolve_experiment_id_for_link(cfg, None) is None
        assert "RuntimeError: no auth" in capsys.readouterr().err


@pytest.mark.unit
class TestHandleCommand:
    def test_noop_when_no_trace_location(self, capsys) -> None:
        from argparse import Namespace

        from dao_ai.cli import handle_link_trace_destination_command

        cfg = MagicMock()
        cfg.app.trace_location = None
        with (
            patch("dao_ai.cli._apply_profile_context"),
            patch("dao_ai.cli.AppConfig") as ac,
        ):
            ac.from_file.return_value = cfg
            handle_link_trace_destination_command(
                Namespace(
                    config="cfg.yaml", profile=None, experiment_id=None, var=None
                )
            )
        assert "nothing to link" in capsys.readouterr().err

    def test_exit_1_when_resolver_returns_none(self) -> None:
        from argparse import Namespace

        from dao_ai.cli import handle_link_trace_destination_command

        cfg = _cfg_with_trace()
        with (
            patch("dao_ai.cli._apply_profile_context"),
            patch("dao_ai.cli.AppConfig") as ac,
            patch(
                "dao_ai.cli._resolve_experiment_id_for_link", return_value=None
            ),
        ):
            ac.from_file.return_value = cfg
            with pytest.raises(SystemExit) as exc:
                handle_link_trace_destination_command(
                    Namespace(
                        config="cfg.yaml",
                        profile=None,
                        experiment_id=None,
                        var=None,
                    )
                )
        assert exc.value.code == 1

    def test_success_calls_linker_and_prints(self, capsys) -> None:
        from argparse import Namespace

        from dao_ai.cli import handle_link_trace_destination_command

        cfg = _cfg_with_trace(catalog="cat", schema="sch", prefix="myp")
        with (
            patch("dao_ai.cli._apply_profile_context"),
            patch("dao_ai.cli.AppConfig") as ac,
            patch(
                "dao_ai.cli._resolve_experiment_id_for_link", return_value="exp42"
            ),
            patch(
                "dao_ai.providers.databricks._link_experiment_trace_location"
            ) as link,
        ):
            ac.from_file.return_value = cfg
            handle_link_trace_destination_command(
                Namespace(
                    config="cfg.yaml", profile=None, experiment_id=None, var=None
                )
            )
        link.assert_called_once_with(cfg, "exp42")
        out = capsys.readouterr().out
        assert "exp42" in out
        assert "cat.sch" in out
        assert "table_prefix=myp" in out

    def test_link_failure_exits_1_with_diagnostic(self, capsys) -> None:
        from argparse import Namespace

        from dao_ai.cli import handle_link_trace_destination_command

        cfg = _cfg_with_trace()
        with (
            patch("dao_ai.cli._apply_profile_context"),
            patch("dao_ai.cli.AppConfig") as ac,
            patch(
                "dao_ai.cli._resolve_experiment_id_for_link", return_value="exp42"
            ),
            patch(
                "dao_ai.providers.databricks._link_experiment_trace_location",
                side_effect=RuntimeError("warehouse timeout"),
            ),
        ):
            ac.from_file.return_value = cfg
            with pytest.raises(SystemExit) as exc:
                handle_link_trace_destination_command(
                    Namespace(
                        config="cfg.yaml",
                        profile=None,
                        experiment_id=None,
                        var=None,
                    )
                )
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "warehouse timeout" in err
        assert "exp42" in err
