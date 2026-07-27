"""Handler-level tests for the ``dao-ai mcp`` noun.

Parse-level coverage lives in test_development_flag_cli.py (TestMcpNoun /
TestMcpNounDispatch). These tests exercise the handler bodies with the MCP
network layer mocked.
"""

from argparse import Namespace
from unittest.mock import MagicMock, patch

import pytest

from dao_ai.cli import (
    _handle_mcp_call,
    _handle_mcp_inspect,
    _mcp_function_from_args,
)
from dao_ai.config import McpFunctionModel


def _mock_function(mcp_url: str = "https://host/mcp") -> MagicMock:
    """A stand-in McpFunctionModel: exposes mcp_url + workspace_client auth."""
    fn = MagicMock()
    fn.mcp_url = mcp_url
    fn.workspace_client.config.authenticate.return_value = {"Authorization": "Bearer x"}
    return fn


def _mock_health(status_code: int, body: str = "") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = body
    if body:
        import json as _json

        resp.json.return_value = _json.loads(body)
    else:
        resp.json.side_effect = ValueError("no json")
    return resp


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


@pytest.mark.unit
class TestHandleMcpInspect:
    """_handle_mcp_inspect: health probe + tool listing, with both mocked."""

    def _opts(self) -> Namespace:
        return Namespace(url="https://host/mcp", app=None, profile=None)

    def test_healthy_server_lists_tools(self, capsys: pytest.CaptureFixture) -> None:
        tool = MagicMock()
        tool.name = "agent_tool"
        tool.description = "desc"
        tool.input_schema = {}
        with (
            patch("dao_ai.cli._mcp_function_from_args", return_value=_mock_function()),
            patch("httpx.get", return_value=_mock_health(200, '{"ok": true}')),
            patch("dao_ai.tools.mcp.list_mcp_tools", return_value=[tool]),
        ):
            with pytest.raises(SystemExit) as exc:
                _handle_mcp_inspect(self._opts())
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "Health: ✓ 200" in out
        assert "agent_tool" in out

    def test_empty_body_health_is_ok(self, capsys: pytest.CaptureFixture) -> None:
        # 200 with an empty body must NOT be reported as "no /healthz endpoint".
        with (
            patch("dao_ai.cli._mcp_function_from_args", return_value=_mock_function()),
            patch("httpx.get", return_value=_mock_health(200, "")),
            patch("dao_ai.tools.mcp.list_mcp_tools", return_value=[]),
        ):
            with pytest.raises(SystemExit) as exc:
                _handle_mcp_inspect(self._opts())
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "Health: ✓ 200" in out
        assert "no /healthz endpoint" not in out

    def test_health_404_reports_no_endpoint(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        with (
            patch("dao_ai.cli._mcp_function_from_args", return_value=_mock_function()),
            patch("httpx.get", return_value=_mock_health(404)),
            patch("dao_ai.tools.mcp.list_mcp_tools", return_value=[]),
        ):
            with pytest.raises(SystemExit) as exc:
                _handle_mcp_inspect(self._opts())
        assert exc.value.code == 0
        assert "no /healthz endpoint" in capsys.readouterr().out

    def test_non_mcp_server_gives_guidance_not_crash(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        # Tool listing fails (e.g. a plain agent App) -> friendly guidance, exit 1.
        with (
            patch("dao_ai.cli._mcp_function_from_args", return_value=_mock_function()),
            patch("httpx.get", return_value=_mock_health(200, "")),
            patch(
                "dao_ai.tools.mcp.list_mcp_tools",
                side_effect=RuntimeError("TaskGroup error"),
            ),
        ):
            with pytest.raises(SystemExit) as exc:
                _handle_mcp_inspect(self._opts())
        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "did not respond as an MCP server" in out
