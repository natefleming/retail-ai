import { useState } from "react";
import { clsx } from "clsx";

import type { SpanNode, TraceTree } from "@/lib/contract";

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
}

function flatten(spans: SpanNode[], depth = 0, acc: Row[] = []): Row[] {
  for (const node of spans) {
    acc.push({ node, depth });
    if (node.children?.length) flatten(node.children, depth + 1, acc);
  }
  return acc;
}

export function Timeline({ trace }: { trace: TraceTree | null | undefined }) {
  const [selected, setSelected] = useState<SpanNode | null>(null);

  if (!trace) {
    return <Empty label="No trace yet — run a turn to see the span waterfall." />;
  }
  const rows = flatten(trace.spans);
  if (!rows.length) return <Empty label="Trace has no spans." />;
  const total = Math.max(trace.duration_ms, 0.001);

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between px-3 py-2 text-[11px] text-[var(--color-fg-subtle)]">
        <span className="font-mono">{trace.trace_id}</span>
        <span className="tabular-nums">{trace.duration_ms.toFixed(1)} ms</span>
      </div>
      <div className="flex-1 space-y-1 overflow-y-auto px-3 pb-3">
        {rows.map(({ node, depth }) => {
          const left = (node.start_offset_ms / total) * 100;
          const width = Math.max((node.duration_ms / total) * 100, 0.8);
          return (
            <button
              key={node.span_id}
              onClick={() => setSelected(node)}
              className="block w-full text-left"
              title={`${node.name} · ${node.duration_ms} ms`}
            >
              <div
                className="mb-0.5 truncate font-mono text-[11px] text-[var(--color-fg-muted)]"
                style={{ paddingLeft: depth * 10 }}
              >
                {node.name}
              </div>
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

function Empty({ label }: { label: string }) {
  return (
    <div className="flex h-full items-center justify-center px-6 text-center text-xs text-[var(--color-fg-subtle)]">
      {label}
    </div>
  );
}
