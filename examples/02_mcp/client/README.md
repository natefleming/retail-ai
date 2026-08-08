# MCP validation client

End-to-end validator for a dao-ai agent deployed as an MCP server
(`dao-ai agent up --as-mcp`). Lists the advertised tools, invokes the
agent-as-tool, and asserts a non-empty response — exiting non-zero on failure,
so it works as a deploy gate in CI.

```bash
# Deploy the MCP server (deploys as mcp-<app.name>)
uv run dao-ai agent up -c <config>.yaml -p <profile> --as-mcp --wait

# Validate it
uv run python examples/02_mcp/client/validate_mcp.py \
    --app mcp-<app-name> -p <profile>
```

Target a raw URL instead of an App name with `--url https://<host>/mcp`.
Override the prompt with `--input`, and pick a specific tool with `--tool` when a
server advertises more than one.

URL resolution and the auth chain (OBO → service principal → PAT → ambient) are
delegated to dao-ai's own `McpFunctionModel` — the same code path used by
`dao-ai mcp inspect` and `dao-ai mcp call` — so behavior matches production
rather than a hand-rolled client.

## Related

- `dao-ai mcp inspect --app <name>` — health + tool list for a live server
- `dao-ai mcp call <tool> --app <name> --args '{...}'` — invoke a single tool
- `docs/mcp_server.md` — deployment, wire shape, and capability reference
