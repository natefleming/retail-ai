import { useEffect, useRef, useState, type ReactNode } from "react";
import {
  ArrowUp,
  Brain,
  ChevronRight,
  Loader2,
  Split,
  Square,
  Wrench,
} from "lucide-react";
import { CustomInputs } from "@/components/CustomInputs";
import { Markdown } from "@/components/Markdown";
import { Visualization } from "@/components/Visualization";

import type { UIConfig } from "@/lib/config";
import type { HITLInterrupt } from "@/lib/contract";
import {
  turnSteps,
  useConsoleContext,
  type Turn,
  type UIToolCall,
} from "@/runtime/useConsole";
import { toolConcurrencyGroups } from "@/lib/events";
import { clsx } from "clsx";

function ToolCard({ call }: { call: UIToolCall }) {
  const [open, setOpen] = useState(false);
  const running = call.status === "in_progress";
  return (
    <div
      className={clsx(
        "rounded-lg border bg-[var(--color-ink-900)] shadow-[var(--shadow-card)]",
        running
          ? "border-[var(--color-span-tool)] ring-1 ring-[var(--color-span-tool)]/40"
          : "border-[var(--color-line)]",
      )}
    >
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-xs"
      >
        {running ? (
          <Loader2 size={13} className="animate-spin text-[var(--color-span-tool)]" />
        ) : (
          <Wrench
            size={13}
            className={
              call.status === "error"
                ? "text-[var(--color-span-error)]"
                : "text-[var(--color-primary)]"
            }
          />
        )}
        <span className="rounded bg-[var(--color-tag-bg)] px-1.5 py-0.5 font-mono text-[11px] text-[var(--color-tag-fg)]">
          {call.name}
        </span>
        {running ? (
          <span className="ml-auto animate-pulse text-[var(--color-span-tool)]">
            running…
          </span>
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

/** A cluster of tools whose execution overlapped in time, rendered with a
 * header that flags the concurrency the flat card list otherwise hides. */
function ParallelGroup({ calls }: { calls: UIToolCall[] }) {
  const running = calls.some((c) => c.status === "in_progress");
  return (
    <div className="space-y-2 rounded-lg border border-dashed border-[var(--color-span-tool)]/40 bg-[var(--color-span-tool)]/5 p-1.5">
      <div className="flex items-center gap-1.5 px-1 text-[11px] font-medium text-[var(--color-span-tool)]">
        {running ? (
          <Loader2 size={12} className="animate-spin" />
        ) : (
          <Split size={12} />
        )}
        <span className={clsx(running && "animate-pulse")}>
          {running
            ? `${calls.length} tools running in parallel`
            : `${calls.length} tools ran in parallel`}
        </span>
      </div>
      {calls.map((c) => (
        <ToolCard key={c.call_id} call={c} />
      ))}
    </div>
  );
}

function Reasoning({
  text,
  defaultOpen,
  live,
}: {
  text: string;
  defaultOpen: boolean;
  live: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  // While the model is still thinking, show the indicator even before any
  // reasoning text has arrived; once text exists it becomes expandable.
  if (!text && !live) return null;
  return (
    <div className="rounded-lg border border-[var(--color-line)] bg-[var(--color-ink-900)] shadow-[var(--shadow-card)]">
      <button
        onClick={() => setOpen((o) => !o)}
        disabled={!text}
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-xs text-[var(--color-span-llm)] disabled:cursor-default"
      >
        {live ? (
          <Loader2 size={13} className="animate-spin" />
        ) : (
          <Brain size={13} />
        )}
        <span className={clsx(live && "animate-pulse")}>
          {live ? "Thinking…" : "Thinking"}
        </span>
        {text && (
          <ChevronRight
            size={13}
            className={clsx("ml-auto transition-transform", open && "rotate-90")}
          />
        )}
      </button>
      {open && text && (
        <p className="whitespace-pre-wrap px-3 pb-3 text-[13px] italic text-[var(--color-fg-muted)]">
          {text}
        </p>
      )}
    </div>
  );
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

function TurnView({ turn }: { turn: Turn }) {
  const { selectTurn, selectedTurnId } = useConsoleContext();
  const isUser = turn.role === "user";
  const steps = turnSteps(turn);
  // The model is actively reasoning when the turn is still streaming and the
  // trailing step is a reasoning segment (no tool/text has followed it yet).
  const reasoningLive =
    turn.status === "streaming" &&
    steps.length > 0 &&
    steps[steps.length - 1].kind === "reasoning";
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
          {(() => {
            // Render steps in order, but collapse each run of consecutive tool
            // steps into concurrency groups so tools that ran in parallel are
            // flagged (a single tool renders as a plain card, unchanged).
            const out: ReactNode[] = [];
            for (let i = 0; i < steps.length; ) {
              const step = steps[i];
              if (step.kind === "tool") {
                const runIds: string[] = [];
                let j = i;
                while (j < steps.length && steps[j].kind === "tool") {
                  runIds.push((steps[j] as { callId: string }).callId);
                  j++;
                }
                for (const group of toolConcurrencyGroups(runIds, turn.toolCalls)) {
                  const groupCalls = group
                    .map((id) => turn.toolCalls.find((c) => c.call_id === id))
                    .filter((c): c is UIToolCall => c != null);
                  if (groupCalls.length > 1) {
                    out.push(
                      <ParallelGroup key={group.join("+")} calls={groupCalls} />,
                    );
                  } else if (groupCalls.length === 1) {
                    out.push(<ToolCard key={group[0]} call={groupCalls[0]} />);
                  }
                }
                i = j;
                continue;
              }
              if (step.kind === "reasoning") {
                out.push(
                  <Reasoning
                    key={step.id}
                    text={step.text}
                    defaultOpen={false}
                    live={reasoningLive && i === steps.length - 1}
                  />,
                );
              } else {
                out.push(
                  <div
                    key={step.id}
                    className="px-2 text-[15px] leading-relaxed text-[var(--color-fg)]"
                  >
                    <Markdown>{step.text}</Markdown>
                  </div>,
                );
              }
              i++;
            }
            return out;
          })()}
          {turn.visualizations.map((v, i) => (
            <Visualization key={i} viz={v} />
          ))}
          {turn.interrupts.map((it, i) => (
            <InterruptCard key={i} interrupt={it} />
          ))}
          {turn.status === "streaming" && !reasoningLive && (
            <div className="flex items-center gap-2 px-2 text-sm text-[var(--color-fg-subtle)]">
              <Loader2 size={14} className="animate-spin text-[var(--color-primary)]" />
              {(() => {
                const active = turn.toolCalls.find((c) => c.status === "in_progress");
                if (active) return `running ${active.name}…`;
                if (!steps.length) return "thinking…";
                return "working…";
              })()}
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
              <TurnView key={t.id} turn={t} />
            ))}
          </div>
        )}
      </div>
      <div className="border-t border-[var(--color-line)] px-6 py-4">
        <div className="mx-auto flex max-w-3xl items-end gap-2 rounded-xl border border-[var(--color-line)] bg-[var(--color-ink-850)] px-3 py-2">
          <CustomInputs />
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
