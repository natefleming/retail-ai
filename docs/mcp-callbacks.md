# Surfacing MCP callbacks to the caller

MCP servers can send `notifications/progress` while a tool call is running.
dao-ai's client-side callback translates them into MLflow span events for
post-hoc tracing **and** — when the tool is running under a streaming
ResponsesAgent — forwards normalized envelopes to the outer response stream
so callers see in-flight status.

> **Deprecated capabilities removed.** The MCP `logging`, `sampling`, and
> `roots` capabilities were deprecated together under
> [SEP-2577](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2577)
> and are no longer supported by dao-ai. The `logging` migration path is
> OpenTelemetry — dao-ai already emits MLflow/OTEL traces, and W3C trace
> context now propagates client→server via `_meta` (see
> [MLflow tracing](mlflow-tracing.md) and `mcp_server.md`). Configs that set
> `capabilities.logging` / `capabilities.sampling` / `capabilities.roots`
> will fail validation (`extra="forbid"`).

## Configuration

```yaml
tools:
  - name: genie_mcp
    type: mcp
    mcp_url: https://<workspace>/api/2.0/mcp/genie/<space>
    capabilities:
      progress: true            # opt into progress notifications
      structured_output: true   # observe structuredContent / resource_link (default)
      elicitation: hitl         # optional: handle server-initiated elicitation
```

When `progress` is set, dao-ai wires the MCP progress callback and begins
dual-emitting envelopes to both MLflow spans and the outer stream. No
separate stream-toggle: opting in via `progress` opts in to both surfaces.

## Wire format on the response stream

Every progress notification the server emits becomes one
`ResponsesAgentStreamEvent` on the SSE stream:

```json
{
  "type": "response.output_item.added",
  "item": {
    "id": "mcp_<server>_<msg_id>_<seq>",
    "type": "custom_tool_call",
    "status": "in_progress",
    "name": "mcp.progress",
    "input": {
      "channel": "mcp.progress",
      "server_name": "genie",
      "tool_name": "run_genie_query",
      "progress": 0.3,
      "total": 1.0,
      "message": "Fetched 3/10 docs"
    }
  }
}
```

The stable `id` shape (`mcp_<server>_<msg_id>_<seq>`) lets a UI overwrite
same-line status updates idempotently. Audit receipts flow on the same
stream under the `dao_ai.audit.*` channel.

The event type (`response.output_item.added` with `status="in_progress"`)
is the OpenAI Responses SSE convention. Any client that already renders
Responses-API streams (Vercel AI SDK, OpenAI clients,
e2e-chatbot-app-next) will surface these envelopes as generic status
items without dao-ai-specific code.

## Transport under the hood

The MCP callback layer dispatches envelopes via **LangChain's callback
manager**, not via LangGraph's `stream_mode="custom"` channel. `apredict_stream`
attaches an `AsyncCallbackHandler` (`_McpEventCollector`) to
`config["callbacks"]` before invoking `graph.astream(...)`. Tools inside
the graph call `adispatch_custom_event(channel, envelope, config=...)`;
the handler pushes envelopes onto a per-request `asyncio.Queue` that
`apredict_stream` drains between astream chunks and yields as
`response.output_item.added` events.

This choice was empirically driven — the `get_stream_writer()` +
`stream_mode="custom"` route silently drops writes when the tool lives
inside a `create_agent`-built subgraph
([LangGraph #6447](https://github.com/langchain-ai/langgraph/issues/6447)).
The callback-manager route is proven end-to-end by langchain-core's own
`test_custom_event_root_dispatch_with_in_tool` and doesn't share that
propagation gap. See
`/Users/nate.fleming/Documents/SSA-brain/40-reference/dao-ai/mcp-callback-streaming-2026-07-13.md`
for the full analysis + citations.

## Non-streaming (batch) fallback

Non-streaming callers (batch `predict()`, `/invocations` with
`stream: false`) do not receive individual events on the wire.
`apredict_stream` additionally mirrors the full envelope timeline into
`custom_outputs["mcp_events"]` on the final `response.output_item.done`
event so a replay client can reconstruct what happened. Batch callers
who want the timeline should hit the streaming endpoint once.

## Elicitation

`capabilities.elicitation: hitl` raises a LangGraph interrupt whose value
carries `{"type": "mcp.elicitation", "server_name", "tool_name",
"message", "requestedSchema"}`. That interrupt surfaces via the existing
HITL path — `custom_outputs["interrupts"]` on the response — and resumes
via `custom_inputs["decisions"]` on the next request. No additional
stream plumbing.

## MCP notifications currently wired

| MCP notification              | Wired? | Channel        |
|-------------------------------|--------|----------------|
| `notifications/progress`      | Yes    | `mcp.progress` |
| `elicitation/create`          | Yes    | HITL interrupt |
| `notifications/message`       | Removed — MCP `logging` deprecated (SEP-2577); use OTEL tracing |
| `sampling/createMessage`      | Removed — MCP `sampling` deprecated (SEP-2577) |
| `roots/list`                  | Removed — MCP `roots` deprecated (SEP-2577) |
| `notifications/resources/*`   | Not wired — extensible via new callback class + channel |
| `notifications/tools/*`       | Not wired |
| `notifications/prompts/*`     | Not wired |
| `notifications/cancelled`     | Not wired |

## W3C trace-context propagation (`_meta`)

dao-ai injects W3C trace context (`traceparent`, `baggage`) into the
`_meta` block of every `tools/call` on the capabilities path
([SEP-414](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/414)),
so a downstream MCP server can continue the caller's distributed trace. The
`traceparent` is minted from the active MLflow span's `trace_id`/`span_id`
(the OTel-native hex that also lands in the UC `_otel_spans` table); the
MLflow trace id rides in `baggage` for dao-ai-native correlation. This
replaces the former custom `x-dao-ai-trace-id` header. See
`dao_ai.tools.mcp_trace_context`.
