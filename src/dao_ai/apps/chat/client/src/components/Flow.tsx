import { ArrowRightLeft, Bot, Database, Wrench } from "lucide-react";
import { clsx } from "clsx";

import type { SpanNode } from "@/lib/contract";
import type { Turn, UIToolCall } from "@/runtime/useConsole";

/**
 * The signature view: the agent system as a node graph that lights up as the
 * prompt propagates. It shows tool calls AND agent→agent handoffs (nested to
 * arbitrary depth — a handoff to another agent that itself hands off), drawn as
 * an indented tree.
 *
 * When the MLflow trace is available it is the faithful source of the
 * hierarchy (agent spans nesting sub-agent and tool spans); while a turn is
 * still streaming — before the trace lands — the graph falls back to the flat
 * tool-call lifecycle so tools still appear live.
 */

type Kind = "agent" | "handoff" | "tool" | "retrieval";

interface FlowNode {
  id: string;
  label: string;
  kind: Kind;
  duration_ms?: number | null;
  status?: string;
  running?: boolean;
  children: FlowNode[];
}

const HANDOFF_RE = /(handoff|transfer|delegate|route_to|to_agent)/i;
const RETRIEVAL_RE = /(genie|sql|vector|search|retriev|index|lookup)/i;

function kindForTool(name: string): Kind {
  if (HANDOFF_RE.test(name)) return "handoff";
  if (RETRIEVAL_RE.test(name)) return "retrieval";
  return "tool";
}

/** Build the flow tree from the MLflow span tree, pruning non-structural
 * spans (LLM calls, parsers) and lifting the children of pruned nodes so the
 * agent → sub-agent → tool shape survives. */
function fromSpans(spans: SpanNode[]): FlowNode[] {
  const out: FlowNode[] = [];
  for (const s of spans) {
    const children = fromSpans(s.children ?? []);
    const t = (s.span_type ?? "").toUpperCase();
    let kind: Kind | null = null;
    if (t === "AGENT" || t === "CHAIN") kind = "agent";
    else if (t === "TOOL") kind = kindForTool(s.name);
    else if (t === "RETRIEVER") kind = "retrieval";

    if (kind) {
      out.push({
        id: s.span_id,
        label: s.name,
        kind,
        duration_ms: s.duration_ms,
        status: s.status,
        children,
      });
    } else {
      out.push(...children); // prune this span, keep its structure
    }
  }
  return out;
}

/** Live fallback: a single agent root with the flat tool calls as children. */
function fromToolCalls(calls: UIToolCall[]): FlowNode[] {
  return [
    {
      id: "agent",
      label: "agent",
      kind: "agent",
      children: calls.map((c) => ({
        id: c.call_id,
        label: c.name,
        kind: c.status === "error" ? "tool" : kindForTool(c.name),
        duration_ms: c.duration_ms,
        status: c.status === "error" ? "ERROR" : undefined,
        running: c.status === "in_progress",
        children: [],
      })),
    },
  ];
}

const KIND_META: Record<Kind, { color: string; Icon: typeof Bot }> = {
  agent: { color: "var(--color-span-agent)", Icon: Bot },
  handoff: { color: "var(--color-span-handoff)", Icon: ArrowRightLeft },
  tool: { color: "var(--color-span-tool)", Icon: Wrench },
  retrieval: { color: "var(--color-span-retrieval)", Icon: Database },
};

function NodePill({ node }: { node: FlowNode }) {
  const { color, Icon } = KIND_META[node.kind];
  const isError = node.status && node.status !== "OK" && node.status !== "UNSET";
  const borderColor = isError ? "var(--color-span-error)" : color;
  return (
    <div
      className={clsx(
        "flex items-center gap-2 rounded-md border bg-[var(--color-ink-850)] px-2.5 py-1.5",
        node.running && "dao-live-gradient",
      )}
      style={node.running ? undefined : { borderColor }}
    >
      <span
        className="flex h-5 w-5 shrink-0 items-center justify-center rounded"
        style={{ color: isError ? "var(--color-span-error)" : color }}
      >
        <Icon size={13} />
      </span>
      <span className="truncate font-mono text-[11px] text-[var(--color-fg)]">
        {node.label}
      </span>
      <span className="ml-auto shrink-0 pl-2 font-mono text-[10px] tabular-nums text-[var(--color-fg-subtle)]">
        {node.running
          ? "running…"
          : node.duration_ms != null
            ? `${node.duration_ms} ms`
            : (node.status ?? "")}
      </span>
    </div>
  );
}

function NodeTree({ node }: { node: FlowNode }) {
  const hasHandoffChild = node.children.some(
    (c) => c.kind === "agent" || c.kind === "handoff",
  );
  return (
    <div>
      <NodePill node={node} />
      {node.children.length > 0 && (
        <div
          className="ml-3 mt-1 space-y-1 border-l pl-3"
          style={{
            borderColor: hasHandoffChild
              ? "var(--color-span-handoff)"
              : "var(--color-line)",
          }}
        >
          {node.children.map((c) => (
            <NodeTree key={c.id} node={c} />
          ))}
        </div>
      )}
    </div>
  );
}

export function Flow({ turn }: { turn: Turn | undefined }) {
  if (!turn) {
    return <Empty label="Select a turn to see its anatomy." />;
  }

  const nodes =
    turn.trace && turn.trace.spans.length
      ? fromSpans(turn.trace.spans)
      : turn.toolCalls.length
        ? fromToolCalls(turn.toolCalls)
        : [];

  if (nodes.length === 0) {
    return (
      <Empty
        label={
          turn.status === "streaming"
            ? "Waiting for the agent to route…"
            : "No tools or handoffs this turn."
        }
      />
    );
  }

  return (
    <div className="h-full space-y-1 overflow-auto p-3">
      {nodes.map((n) => (
        <NodeTree key={n.id} node={n} />
      ))}
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
