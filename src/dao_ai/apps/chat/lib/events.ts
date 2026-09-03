/**
 * Derivations shared by the live and reloaded inspector views:
 *
 * - `provisionalTrace` — a span waterfall built from a turn's tool-call
 *   lifecycle, so the Timeline fills in *during* a live turn (before the
 *   authoritative MLflow trace is fetched at completion).
 * - `eventsFromSteps` — a chronological event log from a turn's reconstructed
 *   steps/tool calls, so Events populate on reload with no trace dependency.
 * - `eventsFromTrace` — a chronological event log from a fetched MLflow trace's
 *   spans (real offsets/durations + MCP/lifecycle detail), used to upgrade the
 *   Events log once a trace lands (live completion or reload).
 *
 * All pure functions (no React, no fetch) so they're trivially testable.
 */
import type { SpanNode, TraceTree } from "@/lib/contract";
import type { TurnStep, UIEvent, UIToolCall } from "@/runtime/useConsole";

/** A tool whose name is a supervisor/swarm handoff renders as an agent transfer. */
function isHandoff(name: string): boolean {
  return name.startsWith("handoff_to") || name.startsWith("transfer_to");
}

/**
 * Build a provisional `TraceTree` from a turn's tool calls so the Timeline shows
 * a waterfall while the turn streams. Offsets are relative to the earliest
 * `started_at`; an in-progress call is timed up to now. Returns null when no
 * call carries timing (e.g. reloaded turns, whose calls have no timestamps —
 * those rely on the fetched trace instead).
 */
export function provisionalTrace(
  toolCalls: UIToolCall[],
  traceId = "live",
): TraceTree | null {
  const timed = toolCalls.filter((c) => c.started_at);
  if (!timed.length) return null;

  const starts = timed.map((c) => Date.parse(c.started_at as string));
  const t0 = Math.min(...starts);
  const now = Date.now();

  const spans: SpanNode[] = timed.map((c, i) => {
    const start = starts[i] - t0;
    const duration =
      c.duration_ms != null
        ? c.duration_ms
        : Math.max(now - (t0 + start), 0); // in-progress → elapsed so far
    return {
      span_id: c.call_id,
      parent_id: null,
      name: c.name,
      span_type: isHandoff(c.name) ? "AGENT" : "TOOL",
      status: c.status === "error" ? "ERROR" : "OK",
      start_offset_ms: start,
      duration_ms: duration,
      inputs: (c.arguments as Record<string, unknown>) ?? {},
      outputs: {},
      attributes: {},
      events: [],
      children: [],
    };
  });

  const total = Math.max(...spans.map((s) => s.start_offset_ms + s.duration_ms), 0.001);
  return { trace_id: traceId, root_span_id: null, duration_ms: total, spans };
}

/**
 * Partition an ordered run of tool call_ids into concurrency groups: a group is
 * a maximal run of calls whose execution windows overlap in time, so the inline
 * conversation can flag tools that ran in parallel (the Timeline shows this via
 * overlapping bars; the flat card list otherwise can't). A call's window is
 * `[started_at, started_at + duration_ms]`; an in-progress call (no duration) is
 * treated as still running to `nowMs`. Calls arrive in roughly start order, so a
 * greedy envelope merge suffices: a call joins the current group when it starts
 * before the group's latest end. A call with no `started_at` can't be timed and
 * becomes its own singleton group (never merged). Sequential tools yield
 * singleton groups (no parallel wrapper); only groups of 2+ are concurrent.
 */
export function toolConcurrencyGroups(
  callIds: string[],
  toolCalls: UIToolCall[],
  nowMs: number = Date.now(),
): string[][] {
  const byId = new Map(toolCalls.map((c) => [c.call_id, c]));
  const groups: string[][] = [];
  let cur: string[] = [];
  let envEnd = -Infinity;
  const flush = () => {
    if (cur.length) groups.push(cur);
    cur = [];
    envEnd = -Infinity;
  };
  for (const id of callIds) {
    const c = byId.get(id);
    if (!c?.started_at) {
      flush();
      groups.push([id]);
      continue;
    }
    const start = Date.parse(c.started_at);
    const end = c.duration_ms != null ? start + c.duration_ms : nowMs;
    if (cur.length && start < envEnd) {
      cur.push(id);
      envEnd = Math.max(envEnd, end);
    } else {
      flush();
      cur = [id];
      envEnd = end;
    }
  }
  flush();
  return groups;
}

/**
 * Derive a chronological event log from a turn's reconstructed steps + tool
 * calls (no trace needed) — used so a reloaded turn's Events tab isn't empty.
 * `at` is ordinal (step index) since reconstruction has no real timestamps; the
 * Events view shows relative deltas, so ordering reads correctly without
 * implying false latencies.
 */
export function eventsFromSteps(
  steps: TurnStep[],
  toolCalls: UIToolCall[],
): UIEvent[] {
  const events: UIEvent[] = [];
  let at = 0;
  for (const step of steps) {
    if (step.kind === "reasoning") {
      events.push({ at: at++, kind: "reasoning", label: "reasoning" });
    } else if (step.kind === "tool") {
      const call = toolCalls.find((c) => c.call_id === step.callId);
      const name = call?.name ?? "tool";
      const kind = call && isHandoff(name) ? "lifecycle" : "tool";
      events.push({ at: at++, kind, label: `${name} started` });
      if (call?.result != null || call?.error != null) {
        events.push({
          at: at++,
          kind,
          label: `${name} completed`,
          detail: call?.error ? "error" : undefined,
        });
      }
    }
  }
  return events;
}

/**
 * Derive a chronological event log from a fetched MLflow trace's spans — real
 * offsets/durations plus per-span events (e.g. mcp.progress). Flattens the tree
 * and sorts by start offset.
 */
export function eventsFromTrace(trace: TraceTree): UIEvent[] {
  const events: UIEvent[] = [];

  const walk = (span: SpanNode): void => {
    const type = span.span_type ?? "";
    const errored = span.status && span.status !== "OK" && span.status !== "UNSET";
    if (type === "TOOL" || type === "RETRIEVER") {
      events.push({
        at: span.start_offset_ms,
        kind: "tool",
        label: `${span.name} started`,
      });
      events.push({
        at: span.start_offset_ms + span.duration_ms,
        kind: "tool",
        label: errored ? `${span.name} error` : `${span.name} completed`,
        detail: `${span.duration_ms.toFixed(0)} ms`,
      });
    } else if (type === "AGENT" || isHandoff(span.name)) {
      events.push({ at: span.start_offset_ms, kind: "lifecycle", label: span.name });
    }
    for (const ev of span.events ?? []) {
      events.push({
        at: ev.timestamp_ms ?? span.start_offset_ms,
        kind: "mcp",
        label: ev.name,
      });
    }
    for (const child of span.children ?? []) walk(child);
  };

  for (const root of trace.spans) walk(root);
  events.sort((a, b) => a.at - b.at);
  return events;
}
