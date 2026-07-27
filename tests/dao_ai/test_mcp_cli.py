"""Handler-level tests for the ``dao-ai mcp`` noun.

Parse-level coverage lives in test_development_flag_cli.py (TestMcpNoun /
TestMcpNounDispatch). These tests exercise the handler bodies with the MCP
network layer mocked.
"""

from argparse import Namespace
from unittest.mock import patch

import pytest

from dao_ai.cli import _handle_mcp_call, _mcp_function_from_args
from dao_ai.config import McpFunctionModel


@pytest.mark.unit
class TestMcpFunctionFromArgs:
    """_mcp_function_from_args builds a McpFunctionModel from --url or --app."""

    def test_url_builds_url_model(self) -> None:
        opts = Namespace(url="https://host/mcp", app=None)
        fn = _mcp_function_from_args(opts)
        assert isinstance(fn, McpFunctionModel)
        assert str(fn.url) == "https://host/mcp"
        assert fn.app is None

    def test_app_builds_app_model(self) -> None:
        opts = Namespace(url=None, app="my-mcp-app")
        fn = _mcp_function_from_args(opts)
        assert isinstance(fn, McpFunctionModel)
        assert fn.app is not None
        assert fn.app.name == "my-mcp-app"


@pytest.mark.unit
class TestHandleMcpCall:
    """_handle_mcp_call parses --args, invokes call_mcp_tool, prints the result."""

    def test_returns_result_and_exits_zero(self, capsys: pytest.CaptureFixture) -> None:
        opts = Namespace(
            url="https://host/mcp",
            app=None,
            tool="ask",
            args='{"input": "hi"}',
            profile=None,
        )
        with patch("dao_ai.tools.mcp.call_mcp_tool", return_value="pong") as mock_call:
            with pytest.raises(SystemExit) as exc:
                _handle_mcp_call(opts)
        assert exc.value.code == 0
        mock_call.assert_called_once()
        # tool name + parsed args are forwarded
        assert mock_call.call_args.args[1] == "ask"
        assert mock_call.call_args.args[2] == {"input": "hi"}
        assert "pong" in capsys.readouterr().out

    def test_malformed_json_exits_one(self, capsys: pytest.CaptureFixture) -> None:
        opts = Namespace(
            url="https://host/mcp",
            app=None,
            tool="ask",
            args="{not json}",
            profile=None,
        )
        with pytest.raises(SystemExit) as exc:
            _handle_mcp_call(opts)
        assert exc.value.code == 1
        assert "must be a JSON object" in capsys.readouterr().err

    def test_non_object_json_exits_one(self, capsys: pytest.CaptureFixture) -> None:
        opts = Namespace(
            url="https://host/mcp",
            app=None,
            tool="ask",
            args="[1, 2, 3]",
            profile=None,
        )
        with pytest.raises(SystemExit) as exc:
            _handle_mcp_call(opts)
        assert exc.value.code == 1
        assert "must be a JSON object" in capsys.readouterr().err
