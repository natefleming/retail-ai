import { clsx } from "clsx";

import type { Turn, UIToolCall } from "@/runtime/useConsole";

/**
 * The signature view: the agent system as a node graph that lights up as the
 * prompt propagates. The agent node sits at the top; each tool call is a node
 * below it, wired by an edge that animates (traveling dash) while the tool
 * runs and settles to a status color when it completes. Node type is inferred
 * from the tool name (genie/sql → retrieval, search/vector → retrieval,
 * agent/handoff → handoff) so the graph reads as the request's anatomy.
 */

function nodeColor(call: UIToolCall): string {
  if (call.status === "error") return "var(--color-span-error)";
  if (call.status === "in_progress") return "var(--color-span-tool)";
  const n = call.name.toLowerCase();
  if (/(genie|sql|vector|search|retriev|index)/.test(n)) return "var(--color-span-retrieval)";
  if (/(agent|handoff|supervis|transfer)/.test(n)) return "var(--color-span-handoff)";
  return "var(--color-span-tool)";
}

export function Flow({ turn }: { turn: Turn | undefined }) {
  if (!turn || turn.toolCalls.length === 0) {
    return (
      <div className="flex h-full items-center justify-center px-6 text-center text-xs text-[var(--color-fg-subtle)]">
        {turn?.status === "streaming"
          ? "Waiting for the agent to route…"
          : "No tools were called this turn."}
      </div>
    );
  }

  const calls = turn.toolCalls;
  const W = 320;
  const rowH = 66;
  const H = 90 + calls.length * rowH;
  const agent = { x: W / 2, y: 44 };
  const toolX = W / 2;

  return (
    <div className="h-full overflow-y-auto p-3">
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} className="mx-auto block">
        {/* edges */}
        {calls.map((c, i) => {
          const ty = 110 + i * rowH;
          const active = c.status === "in_progress";
          return (
            <path
              key={`e-${c.call_id}`}
              d={`M ${agent.x} ${agent.y + 18} C ${agent.x} ${(agent.y + ty) / 2}, ${toolX} ${(agent.y + ty) / 2}, ${toolX} ${ty - 16}`}
              fill="none"
              strokeWidth={active ? 2 : 1.5}
              className={clsx(active && "flow-edge-active")}
              stroke={active ? undefined : "var(--color-line)"}
            />
          );
        })}

        {/* agent / supervisor node */}
        <g>
          <rect
            x={agent.x - 62}
            y={agent.y - 18}
            width={124}
            height={36}
            rx={18}
            fill="var(--color-ink-850)"
            stroke="var(--color-span-agent)"
            strokeWidth={1.5}
          />
          <text
            x={agent.x}
            y={agent.y + 4}
            textAnchor="middle"
            className="fill-[var(--color-fg)] font-mono"
            fontSize={12}
          >
            agent
          </text>
        </g>

        {/* tool nodes */}
        {calls.map((c, i) => {
          const ty = 110 + i * rowH;
          const color = nodeColor(c);
          return (
            <g key={`n-${c.call_id}`}>
              <rect
                x={toolX - 130}
                y={ty - 16}
                width={260}
                height={40}
                rx={8}
                fill="var(--color-ink-850)"
                stroke={color}
                strokeWidth={1.5}
              />
              <circle cx={toolX - 114} cy={ty + 4} r={4} fill={color} />
              <text
                x={toolX - 100}
                y={ty}
                className="fill-[var(--color-fg)] font-mono"
                fontSize={11}
              >
                {c.name.length > 24 ? `${c.name.slice(0, 24)}…` : c.name}
              </text>
              <text
                x={toolX - 100}
                y={ty + 14}
                className="fill-[var(--color-fg-subtle)] font-mono"
                fontSize={10}
              >
                {c.status === "in_progress"
                  ? "running…"
                  : c.duration_ms != null
                    ? `${c.duration_ms} ms`
                    : c.status}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
