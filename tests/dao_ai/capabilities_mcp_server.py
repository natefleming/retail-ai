"""In-process FastMCP fixture that exercises every advanced capability
we wire from ``McpCapabilitiesModel``.

Run standalone for manual smoke:
    uv run python tests/dao_ai/capabilities_mcp_server.py

Integration tests import ``build_server()`` and mount it on an ephemeral
port via uvicorn in a background thread.
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import Context, FastMCP


def build_server(name: str = "capabilities_probe") -> FastMCP:
    mcp = FastMCP(name)

    @mcp.tool()
    async def long_task(steps: int, ctx: Context) -> str:
        """Emit `steps` progress notifications, return final message."""
        for i in range(1, steps + 1):
            await ctx.report_progress(
                progress=float(i),
                total=float(steps),
                message=f"step {i}/{steps}",
            )
        return f"completed {steps} steps"

    @mcp.tool()
    async def noisy_task(ctx: Context) -> str:
        """Emit one message at each of the four standard log levels."""
        for level in ("debug", "info", "warning", "error"):
            await ctx.log(level=level, message=f"probe {level}")
        return "logged 4 levels"

    @mcp.tool()
    async def list_documents(ctx: Context) -> list[dict[str, Any]]:
        """Return content array containing a resource_link.

        FastMCP will wrap the returned Python object into TextContent by
        default; to emit a real ResourceLink we return the raw content list
        via the low-level API.
        """
        # FastMCP JSON-serializes structured returns; the client sees the
        # structuredContent path. For a bona-fide resource_link, use the
        # `raw_content` return path below.
        return [
            {
                "type": "resource_link",
                "uri": "https://example.com/probe/doc-1.txt",
                "name": "doc-1.txt",
                "mimeType": "text/plain",
            }
        ]

    @mcp.tool()
    async def structured_result(query: str, ctx: Context) -> dict[str, Any]:
        """Return a dict — FastMCP surfaces this as `structuredContent`."""
        return {"answer": f"echo:{query}", "confidence": 0.87}

    @mcp.tool()
    async def raise_tool_error(ctx: Context) -> str:
        """Signal an error via raise — surfaces as CallToolResult(isError=True)."""
        raise ValueError("intentional error from raise_tool_error")

    return mcp


if __name__ == "__main__":
    build_server().run(transport="streamable-http")
