import { useEffect, useState } from "react";
import { Brain, X } from "lucide-react";

import { fetchMemory } from "@/lib/api";
import type { MemoryEntry, MemoryResponse } from "@/lib/contract";

/** Prettify a namespace path (memory/<uid>/user_profile → "User profile"). */
function nsLabel(ns: string): string {
  const last = ns.split("/").pop() ?? ns;
  return last.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
}

function EntryValue({ value }: { value: unknown }) {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return (
      <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1">
        {Object.entries(value as Record<string, unknown>).map(([k, v]) => (
          <div key={k} className="contents">
            <dt className="font-mono text-[11px] text-[var(--color-fg-subtle)]">{k}</dt>
            <dd className="text-[13px] text-[var(--color-fg)]">
              {typeof v === "object" ? JSON.stringify(v) : String(v)}
            </dd>
          </div>
        ))}
      </dl>
    );
  }
  return <div className="text-[13px] text-[var(--color-fg)]">{String(value)}</div>;
}

export function MemoryPanel({ onClose }: { onClose: () => void }) {
  const [data, setData] = useState<MemoryResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    void fetchMemory().then((res) => {
      if (!cancelled) {
        setData(res);
        setLoading(false);
      }
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const namespaces = data?.memory ? Object.keys(data.memory) : [];

  return (
    <aside className="flex h-full w-[420px] shrink-0 flex-col border-l border-[var(--color-line)] bg-[var(--color-ink-900)]">
      <div className="flex items-center gap-2 border-b border-[var(--color-line)] px-4 py-3">
        <Brain size={15} className="text-[var(--color-span-llm)]" />
        <span className="text-sm font-medium text-[var(--color-fg)]">Memory</span>
        <button
          onClick={onClose}
          className="ml-auto text-[var(--color-fg-subtle)] hover:text-[var(--color-fg)]"
          aria-label="Close memory"
        >
          <X size={16} />
        </button>
      </div>
      <div className="flex-1 space-y-4 overflow-y-auto p-4">
        {loading ? (
          <div className="text-xs text-[var(--color-fg-subtle)]">Loading…</div>
        ) : !data || data.memory === null ? (
          <div className="text-xs text-[var(--color-fg-subtle)]">
            No memory store is configured for this agent.
          </div>
        ) : namespaces.length === 0 ? (
          <div className="text-xs text-[var(--color-fg-subtle)]">
            Nothing stored yet for you.
          </div>
        ) : (
          namespaces.map((ns) => (
            <section key={ns}>
              <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-[var(--color-fg-muted)]">
                {nsLabel(ns)}
              </h3>
              <div className="space-y-2">
                {(data.memory![ns] as MemoryEntry[]).map((entry, i) => (
                  <div
                    key={entry.key ?? i}
                    className="rounded-lg border border-[var(--color-line)] bg-[var(--color-ink-950)] p-3 shadow-[var(--shadow-card)]"
                  >
                    <EntryValue value={entry.value} />
                    {entry.updated_at && (
                      <div className="mt-2 font-mono text-[10px] text-[var(--color-fg-subtle)]">
                        {new Date(entry.updated_at).toLocaleString()}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </section>
          ))
        )}
      </div>
    </aside>
  );
}
