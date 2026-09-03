import { useState } from "react";
import { ChevronRight } from "lucide-react";
import { clsx } from "clsx";

import type { SpanNode } from "@/lib/contract";
import type { Turn } from "@/runtime/useConsole";

const SPAN_COLOR: Record<string, string> = {
  LLM: "var(--color-span-llm)",
  CHAT_MODEL: "var(--color-span-llm)",
  TOOL: "var(--color-span-tool)",
  RETRIEVER: "var(--color-span-retrieval)",
  AGENT: "var(--color-span-agent)",
  CHAIN: "var(--color-span-agent)",
};

function spanColor(node: SpanNode): string {
  if (node.status && node.status !== "OK" && node.status !== "UNSET") {
    return "var(--color-span-error)";
  }
  return SPAN_COLOR[node.span_type ?? ""] ?? "var(--color-fg-subtle)";
}

interface Row {
  node: SpanNode;
  depth: number;
  hasChildren: boolean;
  collapsed: boolean;
}

function flatten(
  spans: SpanNode[],
  collapsedIds: Set<string>,
  depth = 0,
  acc: Row[] = [],
): Row[] {
  for (const node of spans) {
    const hasChildren = !!node.children?.length;
    const collapsed = collapsedIds.has(node.span_id);
    acc.push({ node, depth, hasChildren, collapsed });
    if (hasChildren && !collapsed) {
      flatten(node.children, collapsedIds, depth + 1, acc);
    }
  }
  return acc;
}

function countDescendants(node: SpanNode): number {
  return (node.children ?? []).reduce(
    (sum, child) => sum + 1 + countDescendants(child),
    0,
  );
}

export function Timeline({ turn }: { turn: Turn | undefined }) {
  const [selected, setSelected] = useState<SpanNode | null>(null);
  const [collapsedIds, setCollapsedIds] = useState<Set<string>>(new Set());

  const toggle = (spanId: string) =>
    setCollapsedIds((prev) => {
      const next = new Set(prev);
      if (next.has(spanId)) next.delete(spanId);
      else next.add(spanId);
      return next;
    });

  const trace = turn?.trace;
  const rows = trace ? flatten(trace.spans, collapsedIds) : [];
  // No spans — whether the trace came back null or empty, the cause on Apps is
  // the same (spans weren't exported), so explain it either way.
  if (!trace || !rows.length) {
    return <Empty label={emptyLabel(turn)} />;
  }
  const total = Math.max(trace.duration_ms, 0.001);

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between px-3 py-2 text-[11px] text-[var(--color-fg-subtle)]">
        <span className="font-mono">{trace.trace_id}</span>
        <span className="tabular-nums">{trace.duration_ms.toFixed(1)} ms</span>
      </div>
      <div className="flex-1 space-y-1 overflow-y-auto px-3 pb-3">
        {rows.map(({ node, depth, hasChildren, collapsed }) => {
          const left = (node.start_offset_ms / total) * 100;
          const width = Math.max((node.duration_ms / total) * 100, 0.8);
          return (
            <div
              key={node.span_id}
              title={`${node.name} · ${node.duration_ms} ms`}
            >
              <div
                className="mb-0.5 flex items-center gap-1 font-mono text-[11px] text-[var(--color-fg-muted)]"
                style={{ paddingLeft: depth * 10 }}
              >
                {hasChildren ? (
                  <button
                    onClick={() => toggle(node.span_id)}
                    className="shrink-0 text-[var(--color-fg-subtle)] hover:text-[var(--color-fg)]"
                    aria-label={collapsed ? "Expand span" : "Collapse span"}
                    aria-expanded={!collapsed}
                  >
                    <ChevronRight
                      size={12}
                      className={clsx("transition-transform", !collapsed && "rotate-90")}
                    />
                  </button>
                ) : (
                  <span className="inline-block w-3 shrink-0" aria-hidden />
                )}
                <button
                  onClick={() => setSelected(node)}
                  className="truncate text-left hover:text-[var(--color-fg)]"
                >
                  {node.name}
                  {collapsed && hasChildren && (
                    <span className="ml-1 text-[var(--color-fg-subtle)]">
                      (+{countDescendants(node)})
                    </span>
                  )}
                </button>
              </div>
              <button
                onClick={() => setSelected(node)}
                className="block w-full"
                aria-label={`${node.name} span bar`}
              >
                <div className="relative h-2 w-full rounded bg-[var(--color-ink-850)]">
                  <div
                    className={clsx(
                      "absolute h-2 rounded",
                      selected?.span_id === node.span_id && "ring-1 ring-white/40",
                    )}
                    style={{
                      left: `${left}%`,
                      width: `${width}%`,
                      background: spanColor(node),
                    }}
                  />
                </div>
              </button>
            </div>
          );
        })}
      </div>
      {selected && <SpanDetail node={selected} onClose={() => setSelected(null)} />}
    </div>
  );
}

function SpanDetail({ node, onClose }: { node: SpanNode; onClose: () => void }) {
  return (
    <div className="max-h-[45%] overflow-y-auto border-t border-[var(--color-line)] bg-[var(--color-ink-900)] px-3 py-2">
      <div className="mb-1 flex items-center justify-between">
        <span className="font-mono text-xs text-[var(--color-fg)]">{node.name}</span>
        <button onClick={onClose} className="text-[11px] text-[var(--color-fg-subtle)]">
          close
        </button>
      </div>
      <div className="mb-2 flex gap-3 font-mono text-[11px] text-[var(--color-fg-subtle)]">
        <span>{node.span_type}</span>
        <span>{node.status}</span>
        <span className="tabular-nums">{node.duration_ms} ms</span>
      </div>
      {!!Object.keys(node.inputs ?? {}).length && (
        <Payload label="inputs" value={node.inputs} />
      )}
      {!!Object.keys(node.outputs ?? {}).length && (
        <Payload label="outputs" value={node.outputs} />
      )}
    </div>
  );
}

function Payload({ label, value }: { label: string; value: unknown }) {
  return (
    <div className="mb-2">
      <div className="text-[10px] uppercase tracking-wide text-[var(--color-fg-subtle)]">
        {label}
      </div>
      <pre className="overflow-x-auto whitespace-pre-wrap font-mono text-[11px] text-[var(--color-fg-muted)]">
        {JSON.stringify(value, null, 2)}
      </pre>
    </div>
  );
}

/** Explain *why* there's no waterfall — pending vs. genuinely unavailable —
 * so a missing trace_location reads as a config note, not a broken panel. */
function emptyLabel(turn: Turn | undefined): string {
  if (!turn || (!turn.traceId && turn.status !== "done")) {
    return "No trace yet — run a turn to see the span waterfall.";
  }
  // A completed turn whose trace came back null or with no spans. The Console
  // can't tell which cause applies, so name the common ones: trace_location not
  // set, spans still propagating (UC is eventually consistent), or the app's
  // identity lacking read access to the trace warehouse.
  if (turn.traceId && (turn.trace === null || turn.status === "done")) {
    return (
      "No trace spans to show yet. Traces are recorded to the workspace " +
      "(control plane) by default, or to Unity Catalog when app.trace_location " +
      "is set — this one may still be propagating, or this runtime can't reach " +
      "the trace store. Try reopening shortly."
    );
  }
  if (turn.traceId) return "Fetching trace…";
  return "This turn produced no trace.";
}

function Empty({ label }: { label: string }) {
  return (
    <div className="flex h-full items-center justify-center px-6 text-center text-xs text-[var(--color-fg-subtle)]">
      {label}
    </div>
  );
}
