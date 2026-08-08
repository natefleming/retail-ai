"""End-to-end validator for a dao-ai agent deployed as an MCP server.

Exercises the MCP surface of a live ``dao-ai agent up --as-mcp`` deployment:

  1. Resolves the App's ``/mcp`` endpoint (``--app``) or uses ``--url`` verbatim.
  2. Lists the tools the server advertises (MCP ``tools/list``).
  3. Invokes the agent-as-tool (MCP ``tools/call``) and asserts a non-empty
     text response.

Exits non-zero on any failure, so it works as a deploy gate in CI.

Run against a deployed MCP App:

  uv run python examples/02_mcp/client/validate_mcp.py \\
      --app mcp-uc-fn-trace-val-dao --profile fevm

Or against any MCP server URL:

  uv run python examples/02_mcp/client/validate_mcp.py \\
      --url https://<app-host>.cloud.databricksapps.com/mcp --profile fevm

Auth and URL resolution are delegated to dao-ai's own ``McpFunctionModel`` —
the same code path the CLI's ``dao-ai mcp inspect`` / ``mcp call`` verbs use — so
the App-name-to-endpoint lookup and the OBO -> SP -> PAT -> ambient credential
chain behave identically here and in production. Nothing is hand-rolled.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a deployed dao-ai MCP server end to end.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--app",
        type=str,
        metavar="NAME",
        help="Databricks App name (e.g. mcp-my-agent); its /mcp endpoint is "
        "resolved via the SDK.",
    )
    target.add_argument(
        "--url",
        type=str,
        metavar="URL",
        help="Direct MCP server URL (e.g. https://host/.../mcp).",
    )
    parser.add_argument(
        "-p",
        "--profile",
        type=str,
        default=None,
        help="Databricks CLI profile used for App lookup + auth.",
    )
    parser.add_argument(
        "--tool",
        type=str,
        default=None,
        help="Tool to invoke. Defaults to the single agent-as-tool the server "
        "advertises (fails if the server exposes more than one and none is given).",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="What can you help me with?",
        help="Value for the tool's `input` argument.",
    )
    return parser.parse_args()


async def _run(options: argparse.Namespace) -> int:
    # Reuse dao-ai's own MCP client machinery: McpFunctionModel.mcp_url resolves
    # an App name to its /mcp endpoint via the SDK, and the inherited
    # IsDatabricksResource auth chain supplies credentials.
    from dao_ai.config import DatabricksAppModel, McpFunctionModel
    from dao_ai.tools.mcp import acreate_mcp_tools

    if options.app:
        function = McpFunctionModel(app=DatabricksAppModel(name=options.app))
    else:
        function = McpFunctionModel(url=options.url)

    print(f"MCP endpoint: {function.mcp_url}")

    # 1 + 2. Connect and discover the advertised tool surface.
    tools = await acreate_mcp_tools(function)
    names: list[str] = [t.name for t in tools]
    if not names:
        print("FAIL: server advertised no tools", file=sys.stderr)
        return 1
    print(f"tools/list -> {names}")

    # 3. Pick the tool to invoke.
    if options.tool:
        tool_name = options.tool
        if tool_name not in names:
            print(
                f"FAIL: requested tool {tool_name!r} not advertised; got {names}",
                file=sys.stderr,
            )
            return 1
    elif len(names) == 1:
        tool_name = names[0]
    else:
        print(
            f"FAIL: server advertises {len(names)} tools {names}; pass --tool",
            file=sys.stderr,
        )
        return 1

    tool = next(t for t in tools if t.name == tool_name)
    print(f"tools/call -> {tool_name}(input={options.input!r})")

    result: Any = await tool.ainvoke({"input": options.input})
    text: str = result if isinstance(result, str) else json.dumps(result, default=str)

    if not text or not text.strip():
        print("FAIL: tool returned an empty response", file=sys.stderr)
        return 1

    preview = text if len(text) <= 600 else text[:600] + " …"
    print(f"\nresponse ({len(text)} chars):\n{preview}\n")
    print(f"PASS: {tool_name} returned a non-empty response")
    return 0


def main() -> None:
    options = _parse_args()
    if options.profile:
        # Scope every SDK call in this process to the requested profile, matching
        # how the CLI applies -p/--profile.
        import os

        os.environ["DATABRICKS_CONFIG_PROFILE"] = options.profile

    try:
        sys.exit(asyncio.run(_run(options)))
    except Exception as exc:  # noqa: BLE001 - top-level validator boundary
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
