"""Tests for the ``dao-ai monitor logs`` subcommand.

Covers argument parsing (mutually-exclusive name source, mode default, flags)
and handler dispatch (Apps -> ``stream_app_logs``, Model Serving snapshot ->
``fetch_model_serving_logs``, and the ``--follow`` rejection for model_serving).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from dao_ai.cli import parse_args
from dao_ai.monitoring import stream_app_logs


@pytest.mark.unit
class TestMonitorLogsParsing:
    def test_mode_defaults_to_apps(self) -> None:
        opts = parse_args(["monitor", "logs", "-c", "config.yaml"])
        assert opts.subcommand == "logs"
        assert opts.mode == "apps"
        assert opts.lines == 200
        assert opts.follow is False

    def test_name_source_used_literally(self) -> None:
        opts = parse_args(["monitor", "logs", "--name", "my-app"])
        assert opts.name == "my-app"
        assert opts.config is None

    def test_mode_alias_normalized(self) -> None:
        opts = parse_args(["monitor", "logs", "--name", "ep", "-m", "ms"])
        assert opts.mode == "model_serving"

    def test_config_and_name_mutually_exclusive(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["monitor", "logs", "-c", "config.yaml", "--name", "x"])

    def test_one_name_source_required(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["monitor", "logs"])


@pytest.mark.unit
class TestMonitorLogsDispatch:
    def _run(self, argv: list[str]):
        """Parse ``argv`` and run the monitor handler, returning the SystemExit."""
        from dao_ai.cli import handle_monitor_command

        opts = parse_args(argv)
        with pytest.raises(SystemExit) as exc:
            handle_monitor_command(opts)
        return exc.value

    def test_model_serving_follow_rejected(self) -> None:
        exc = self._run(
            ["monitor", "logs", "--name", "ep", "-m", "model_serving", "--follow"]
        )
        assert exc.code == 1

    @patch("dao_ai.monitoring.fetch_model_serving_logs", return_value="hello logs")
    def test_model_serving_snapshot_prints(
        self, mock_fetch: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        exc = self._run(
            ["monitor", "logs", "--name", "ep", "-m", "model_serving", "--lines", "50"]
        )
        assert exc.code == 0
        mock_fetch.assert_called_once_with(endpoint_name="ep", lines=50)
        assert "hello logs" in capsys.readouterr().out

    @patch("dao_ai.monitoring.stream_app_logs", return_value=0)
    def test_apps_streams_via_name(self, mock_stream: MagicMock) -> None:
        exc = self._run(
            ["monitor", "logs", "--name", "my-app", "-p", "fevm", "--follow"]
        )
        assert exc.code == 0
        mock_stream.assert_called_once_with(
            app_name="my-app", lines=200, follow=True, profile="fevm"
        )

    @patch("dao_ai.monitoring.stream_app_logs", return_value=0)
    @patch("dao_ai.cli.AppConfig.from_file")
    def test_apps_derives_name_from_config(
        self, mock_from_file: MagicMock, mock_stream: MagicMock
    ) -> None:
        # app.name "minimal_dao" -> workspace app "minimal-dao" (underscore->hyphen)
        cfg = MagicMock()
        cfg.app.app_resource_name = "minimal-dao"
        mock_from_file.return_value = cfg
        exc = self._run(["monitor", "logs", "-c", "config.yaml", "-p", "fevm"])
        assert exc.code == 0
        mock_stream.assert_called_once_with(
            app_name="minimal-dao", lines=200, follow=False, profile="fevm"
        )

    def test_negative_lines_rejected(self) -> None:
        exc = self._run(["monitor", "logs", "--name", "x", "--lines", "-3"])
        assert exc.code == 1

    @patch("dao_ai.monitoring.stream_app_logs", side_effect=KeyboardInterrupt)
    def test_follow_keyboard_interrupt_exits_cleanly(
        self, mock_stream: MagicMock
    ) -> None:
        # Ctrl-C on a follow stream must not escape as a raw traceback.
        exc = self._run(["monitor", "logs", "--name", "x", "--follow"])
        assert exc.code == 130


@pytest.mark.unit
class TestStreamAppLogs:
    @patch("dao_ai.monitoring.shutil.which", return_value=None)
    def test_missing_databricks_cli_raises(self, _which: MagicMock) -> None:
        with pytest.raises(RuntimeError, match="databricks.*CLI"):
            stream_app_logs(app_name="x")
