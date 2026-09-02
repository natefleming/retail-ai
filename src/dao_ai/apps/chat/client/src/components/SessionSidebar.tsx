import { MessageSquarePlus } from "lucide-react";
import { clsx } from "clsx";

import { useConsoleContext } from "@/runtime/useConsole";

export function SessionSidebar() {
  const { sessions, threadId, newSession, loadSession, isRunning } =
    useConsoleContext();

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
          sessions.map((s) => (
            <button
              key={s.threadId}
              disabled={isRunning}
              onClick={() => loadSession(s.threadId)}
              className={clsx(
                "block w-full truncate rounded-md px-3 py-2 text-left text-[13px] transition-colors disabled:opacity-50",
                s.threadId === threadId
                  ? "bg-[var(--color-ink-800)] text-[var(--color-fg)]"
                  : "text-[var(--color-fg-muted)] hover:bg-[var(--color-ink-850)]",
              )}
              title={s.title}
            >
              {s.title}
            </button>
          ))
        )}
      </div>
    </div>
  );
}
