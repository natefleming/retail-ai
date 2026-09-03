import { useEffect, useState } from "react";
import { Activity, Braces, ExternalLink, GitBranch, ListTree } from "lucide-react";
import { clsx } from "clsx";

import { Flow } from "@/components/Flow";
import { JsonTree } from "@/components/JsonTree";
import { Timeline } from "@/components/Timeline";
import { fetchTraceUrl } from "@/lib/api";
import { useConsoleContext, type Turn, type UIEvent } from "@/runtime/useConsole";

type Tab = "flow" | "timeline" | "events" | "outputs";

const eventColor: Record<UIEvent["kind"], string> = {
  tool: "var(--color-span-tool)",
  mcp: "var(--color-brand-blue)",
  reasoning: "var(--color-span-llm)",
  lifecycle: "var(--color-fg-subtle)",
};

function EventLog({ turn }: { turn: Turn | undefined }) {
  if (!turn || turn.events.length === 0) {
    return (
      <div className="flex h-full items-center justify-center px-6 text-center text-xs text-[var(--color-fg-subtle)]">
        Event log is empty for this turn.
      </div>
    );
  }
  const t0 = turn.events[0].at;
  return (
    <div className="h-full space-y-1 overflow-y-auto p-3 font-mono text-[11px]">
      {turn.events.map((e, i) => (
        <div key={i} className="flex items-baseline gap-2">
          <span className="tabular-nums text-[var(--color-fg-subtle)]">
            +{e.at - t0}ms
          </span>
          <span style={{ color: eventColor[e.kind] }}>{e.label}</span>
          {e.detail && <span className="text-[var(--color-fg-muted)]">{e.detail}</span>}
        </div>
      ))}
    </div>
  );
}

function TraceLink({ traceId }: { traceId: string }) {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    let cancelled = false;
    setUrl(null);
    void fetchTraceUrl(traceId).then((u) => {
      if (!cancelled) setUrl(u);
    });
    return () => {
      cancelled = true;
    };
  }, [traceId]);

  const label = `trace ${traceId.slice(0, 12)}…`;
  if (!url) return <span title={traceId}>{label}</span>;
  return (
    <a
      href={url}
      target="_blank"
      rel="noreferrer"
      title={`View trace in Databricks: ${traceId}`}
      className="inline-flex items-center gap-1 text-[var(--color-primary)] hover:underline"
    >
      {label}
      <ExternalLink size={10} />
    </a>
  );
}

function Metadata({ turn }: { turn: Turn | undefined }) {
  if (!turn) return null;
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-b border-[var(--color-line)] px-3 py-2 font-mono text-[10px] text-[var(--color-fg-subtle)]">
      {turn.traceId && <TraceLink traceId={turn.traceId} />}
      {turn.trace && <span className="tabular-nums">{turn.trace.duration_ms.toFixed(0)} ms</span>}
      <span>{turn.toolCalls.length} tools</span>
      <span className={turn.status === "error" ? "text-[var(--color-span-error)]" : ""}>
        {turn.status}
      </span>
    </div>
  );
}

export function Inspector() {
  const { turns, selectedTurnId } = useConsoleContext();
  const [tab, setTab] = useState<Tab>("flow");
  const turn =
    turns.find((t) => t.id === selectedTurnId) ??
    [...turns].reverse().find((t) => t.role === "assistant");

  const tabs: { id: Tab; label: string; icon: typeof GitBranch }[] = [
    { id: "flow", label: "Flow", icon: GitBranch },
    { id: "timeline", label: "Timeline", icon: ListTree },
    { id: "events", label: "Events", icon: Activity },
    { id: "outputs", label: "Outputs", icon: Braces },
  ];

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center gap-1 border-b border-[var(--color-line)] px-2 py-1.5">
        {tabs.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setTab(id)}
            className={clsx(
              "flex items-center gap-1.5 rounded px-2.5 py-1 text-xs transition-colors",
              tab === id
                ? "bg-[var(--color-ink-800)] text-[var(--color-fg)]"
                : "text-[var(--color-fg-subtle)] hover:text-[var(--color-fg-muted)]",
            )}
          >
            <Icon size={13} />
            {label}
          </button>
        ))}
      </div>
      <Metadata turn={turn} />
      <div className="min-h-0 flex-1">
        {tab === "flow" && <Flow turn={turn} />}
        {tab === "timeline" && <Timeline turn={turn} />}
        {tab === "events" && <EventLog turn={turn} />}
        {tab === "outputs" && <JsonTree value={turn?.customOutputs} />}
      </div>
    </div>
  );
}
