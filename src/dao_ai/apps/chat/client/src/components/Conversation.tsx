import { useEffect, useRef, useState } from "react";
import {
  ArrowUp,
  Brain,
  ChevronRight,
  Loader2,
  Square,
  Wrench,
} from "lucide-react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { VegaLite } from "react-vega";

import type { UIConfig } from "@/lib/config";
import type { HITLInterrupt, VisualizationSpec } from "@/lib/contract";
import { useConsoleContext, type Turn, type UIToolCall } from "@/runtime/useConsole";
import { clsx } from "clsx";

const statusColor: Record<UIToolCall["status"], string> = {
  in_progress: "text-[var(--color-span-tool)]",
  completed: "text-[var(--color-ok)]",
  error: "text-[var(--color-span-error)]",
};

function ToolCard({ call }: { call: UIToolCall }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-md border border-[var(--color-line)] bg-[var(--color-ink-850)]">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left font-mono text-xs"
      >
        <Wrench size={13} className={statusColor[call.status]} />
        <span className="text-[var(--color-fg)]">{call.name}</span>
        {call.status === "in_progress" ? (
          <Loader2 size={12} className="animate-spin text-[var(--color-span-tool)]" />
        ) : (
          <span
            className={clsx(
              "ml-auto tabular-nums",
              call.status === "error"
                ? "text-[var(--color-span-error)]"
                : "text-[var(--color-fg-subtle)]",
            )}
          >
            {call.duration_ms != null ? `${call.duration_ms} ms` : call.status}
          </span>
        )}
        <ChevronRight
          size={13}
          className={clsx("text-[var(--color-fg-subtle)] transition-transform", open && "rotate-90")}
        />
      </button>
      {open && (
        <div className="space-y-2 border-t border-[var(--color-line)] px-3 py-2 font-mono text-[11px]">
          {call.arguments != null && (
            <pre className="overflow-x-auto whitespace-pre-wrap text-[var(--color-fg-muted)]">
              {JSON.stringify(call.arguments, null, 2)}
            </pre>
          )}
          {call.result && (
            <pre className="overflow-x-auto whitespace-pre-wrap text-[var(--color-fg)]">
              {call.result}
            </pre>
          )}
          {call.error && (
            <pre className="overflow-x-auto whitespace-pre-wrap text-[var(--color-span-error)]">
              {call.error}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}

function Reasoning({ text, defaultOpen }: { text: string; defaultOpen: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  if (!text) return null;
  return (
    <div className="rounded-md border border-[var(--color-line)] bg-[var(--color-ink-900)]/60">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-xs text-[var(--color-span-llm)]"
      >
        <Brain size={13} />
        <span>Reasoning</span>
        <ChevronRight
          size={13}
          className={clsx("ml-auto transition-transform", open && "rotate-90")}
        />
      </button>
      {open && (
        <p className="whitespace-pre-wrap px-3 pb-3 text-[13px] italic text-[var(--color-fg-muted)]">
          {text}
        </p>
      )}
    </div>
  );
}

function Visualization({ viz }: { viz: VisualizationSpec }) {
  try {
    return (
      <div className="overflow-x-auto rounded-md border border-[var(--color-line)] bg-white p-2">
        <VegaLite spec={viz.spec as never} actions={false} />
      </div>
    );
  } catch {
    return null;
  }
}

function InterruptCard({ interrupt }: { interrupt: HITLInterrupt }) {
  const { respondToInterrupt, isRunning } = useConsoleContext();
  return (
    <div className="rounded-md border border-[var(--color-span-handoff)]/50 bg-[var(--color-span-handoff)]/10 px-3 py-3">
      <div className="text-sm font-medium text-[var(--color-fg)]">
        {interrupt.title ?? "Approval required"}
      </div>
      {interrupt.instructions && (
        <div className="mt-1 text-xs text-[var(--color-fg-muted)]">{interrupt.instructions}</div>
      )}
      <div className="mt-3 flex gap-2">
        <button
          disabled={isRunning}
          onClick={() => respondToInterrupt([{ type: "approve" }])}
          className="rounded bg-[var(--color-ok)]/20 px-3 py-1 text-xs text-[var(--color-ok)] disabled:opacity-50"
        >
          Approve
        </button>
        <button
          disabled={isRunning}
          onClick={() => respondToInterrupt([{ type: "reject", message: "Rejected" }])}
          className="rounded bg-[var(--color-span-error)]/20 px-3 py-1 text-xs text-[var(--color-span-error)] disabled:opacity-50"
        >
          Reject
        </button>
      </div>
    </div>
  );
}

function TurnView({ turn, showReasoning }: { turn: Turn; showReasoning: boolean }) {
  const { selectTurn, selectedTurnId } = useConsoleContext();
  const isUser = turn.role === "user";
  return (
    <div
      onClick={() => selectTurn(turn.id)}
      className={clsx(
        "group flex flex-col gap-2",
        isUser ? "items-end" : "items-start",
      )}
    >
      {isUser ? (
        <div className="max-w-[80%] rounded-2xl rounded-br-sm bg-[var(--color-ink-800)] px-4 py-2 text-[15px]">
          {turn.content}
        </div>
      ) : (
        <div
          className={clsx(
            "w-full max-w-[80%] cursor-pointer space-y-2 rounded-lg border px-1 py-1 transition-colors",
            selectedTurnId === turn.id
              ? "border-[var(--color-line)]"
              : "border-transparent",
          )}
        >
          {showReasoning && <Reasoning text={turn.reasoning} defaultOpen={false} />}
          {turn.toolCalls.map((c) => (
            <ToolCard key={c.call_id} call={c} />
          ))}
          {turn.content && (
            <div className="prose-invert px-2 text-[15px] leading-relaxed text-[var(--color-fg)]">
              <Markdown remarkPlugins={[remarkGfm]}>{turn.content}</Markdown>
            </div>
          )}
          {turn.visualizations.map((v, i) => (
            <Visualization key={i} viz={v} />
          ))}
          {turn.interrupts.map((it, i) => (
            <InterruptCard key={i} interrupt={it} />
          ))}
          {turn.status === "streaming" && !turn.content && !turn.toolCalls.length && (
            <div className="flex items-center gap-2 px-2 text-sm text-[var(--color-fg-subtle)]">
              <Loader2 size={14} className="animate-spin" /> thinking…
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function Conversation({ config }: { config: UIConfig }) {
  const { turns, isRunning, send, cancel } = useConsoleContext();
  const [draft, setDraft] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const showReasoning = config.mode === "developer";

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [turns]);

  const submit = () => {
    if (!draft.trim() || isRunning) return;
    void send(draft);
    setDraft("");
  };

  return (
    <div className="flex h-full flex-col">
      <div ref={scrollRef} className="flex-1 space-y-6 overflow-y-auto px-6 py-6">
        {turns.length === 0 ? (
          <div className="mx-auto mt-24 max-w-md text-center">
            <div className="font-display text-lg text-[var(--color-fg)]">
              {config.title}
            </div>
            <div className="mt-2 text-sm text-[var(--color-fg-muted)]">
              {config.subtitle ?? "Ask anything — watch the answer, the reasoning, and every tool call as it happens."}
            </div>
          </div>
        ) : (
          <div className="mx-auto flex max-w-3xl flex-col gap-6">
            {turns.map((t) => (
              <TurnView key={t.id} turn={t} showReasoning={showReasoning} />
            ))}
          </div>
        )}
      </div>
      <div className="border-t border-[var(--color-line)] px-6 py-4">
        <div className="mx-auto flex max-w-3xl items-end gap-2 rounded-xl border border-[var(--color-line)] bg-[var(--color-ink-850)] px-3 py-2">
          <textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
            rows={1}
            placeholder="Ask anything…"
            className="max-h-40 flex-1 resize-none bg-transparent py-1.5 text-[15px] outline-none placeholder:text-[var(--color-fg-subtle)]"
          />
          {isRunning ? (
            <button
              onClick={cancel}
              className="rounded-lg bg-[var(--color-ink-700)] p-2 text-[var(--color-fg)]"
              title="Stop"
            >
              <Square size={16} />
            </button>
          ) : (
            <button
              onClick={submit}
              disabled={!draft.trim()}
              className="rounded-lg bg-[var(--color-brand-blue)] p-2 text-white disabled:opacity-40"
              title="Send"
            >
              <ArrowUp size={16} />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
