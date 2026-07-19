"""Tests for the --development / --no-development tri-state across CLI commands.

The flag must resolve identically on every deploy-capable command:
``deploy``, ``pipeline``, ``generate-bundle``, ``generate-mcp`` —
``--development`` -> True, ``--no-development`` -> False, omitted -> None
(auto-detect). For ``pipeline`` it is additionally forwarded to the deploy
notebook as a ``--var development=auto|true|false`` bundle variable.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from dao_ai import cli
from dao_ai.cli import parse_args, run_databricks_command


def _base(cmd: str) -> list[str]:
    return [cmd, "-c", "config.yaml"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "command", ["deploy", "pipeline", "generate-bundle", "generate-mcp"]
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
