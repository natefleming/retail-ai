import { useState } from "react";
import { Info, MessageSquarePlus } from "lucide-react";
import { clsx } from "clsx";

import { fetchSessionMeta } from "@/lib/api";
import type { SessionMeta } from "@/lib/contract";
import { useConsoleContext } from "@/runtime/useConsole";

function SessionMetaPopover({ meta }: { meta: SessionMeta | null }) {
  if (!meta) {
    return (
      <div className="px-3 py-2 text-[11px] text-[var(--color-fg-subtle)]">
        Loading metadata…
      </div>
    );
  }
  const rows: [string, string][] = [
    ["thread", meta.thread_id],
    ["checkpoint", meta.checkpoint_id ?? "—"],
    [
      "last modified",
      meta.last_modified ? new Date(meta.last_modified).toLocaleString() : "—",
    ],
    ["messages", String(meta.message_count)],
    ["step", meta.step != null ? String(meta.step) : "—"],
  ];
  return (
    <div className="mx-2 mb-1 rounded-md border border-[var(--color-line)] bg-[var(--color-ink-900)] p-2 shadow-[var(--shadow-card)]">
      <dl className="grid grid-cols-[auto_1fr] gap-x-2 gap-y-1 font-mono text-[10px]">
        {rows.map(([k, v]) => (
          <div key={k} className="contents">
            <dt className="text-[var(--color-fg-subtle)]">{k}</dt>
            <dd className="truncate text-[var(--color-fg-muted)]" title={v}>
              {v}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

export function SessionSidebar() {
  const { sessions, threadId, newSession, loadSession, isRunning } =
    useConsoleContext();
  const [metaOpenFor, setMetaOpenFor] = useState<string | null>(null);
  const [meta, setMeta] = useState<SessionMeta | null>(null);

  const toggleMeta = async (tid: string) => {
    if (metaOpenFor === tid) {
      setMetaOpenFor(null);
      return;
    }
    setMeta(null);
    setMetaOpenFor(tid);
    setMeta(await fetchSessionMeta(tid));
  };

  return (
    <div className="flex h-full flex-col">
      <button
        onClick={newSession}
        className="mx-3 mt-3 flex items-center gap-2 rounded-lg border border-[var(--color-line)] px-3 py-2 text-sm text-[var(--color-fg-muted)] transition-colors hover:text-[var(--color-fg)]"
      >
        <MessageSquarePlus size={15} /> New session
      </button>
      <div className="mt-3 flex-1 space-y-0.5 overflow-y-auto px-2 pb-3">
        {sessions.length === 0 ? (
          <div className="px-3 py-6 text-center text-xs text-[var(--color-fg-subtle)]">
            Past sessions appear here.
          </div>
        ) : (
          sessions.map((s) => {
            const active = s.threadId === threadId;
            return (
              <div key={s.threadId}>
                <div
                  className={clsx(
                    "group flex items-center rounded-md transition-colors",
                    active
                      ? "bg-[var(--color-ink-800)]"
                      : "hover:bg-[var(--color-ink-850)]",
                  )}
                >
                  <button
                    disabled={isRunning}
                    onClick={() => loadSession(s.threadId)}
                    className={clsx(
                      "min-w-0 flex-1 truncate px-3 py-2 text-left text-[13px] disabled:opacity-50",
                      active
                        ? "text-[var(--color-fg)]"
                        : "text-[var(--color-fg-muted)]",
                    )}
                    title={s.title}
                  >
                    {s.title}
                  </button>
                  <button
                    onClick={() => toggleMeta(s.threadId)}
                    className={clsx(
                      "mr-1 shrink-0 rounded p-1 text-[var(--color-fg-subtle)] hover:text-[var(--color-fg)]",
                      active ? "opacity-100" : "opacity-0 group-hover:opacity-100",
                    )}
                    aria-label="Session info"
                    title="Session info"
                  >
                    <Info size={13} />
                  </button>
                </div>
                {metaOpenFor === s.threadId && <SessionMetaPopover meta={meta} />}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
