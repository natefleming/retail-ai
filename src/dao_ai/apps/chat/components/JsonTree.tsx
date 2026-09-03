import { useState } from "react";
import { ChevronRight } from "lucide-react";
import { clsx } from "clsx";

/**
 * Compact, collapsible tree for arbitrary JSON — used to show a turn's raw
 * custom_outputs in the inspector. Objects/arrays expand on click; primitives
 * render inline, color-coded by type.
 */
function isContainer(v: unknown): v is Record<string, unknown> | unknown[] {
  return v !== null && typeof v === "object";
}

function summary(v: unknown): string {
  if (Array.isArray(v)) return `[] ${v.length}`;
  if (isContainer(v)) return `{} ${Object.keys(v).length}`;
  return "";
}

function Leaf({ value }: { value: unknown }) {
  let color = "var(--color-fg)";
  let text: string;
  if (value === null) {
    color = "var(--color-fg-subtle)";
    text = "null";
  } else if (typeof value === "string") {
    color = "var(--color-span-retrieval)";
    text = `"${value}"`;
  } else if (typeof value === "number" || typeof value === "boolean") {
    color = "var(--color-primary)";
    text = String(value);
  } else {
    text = String(value);
  }
  return (
    <span className="break-all" style={{ color }}>
      {text}
    </span>
  );
}

function Node({
  name,
  value,
  depth,
  defaultOpen,
}: {
  name: string;
  value: unknown;
  depth: number;
  defaultOpen: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const container = isContainer(value);
  const entries = container
    ? Array.isArray(value)
      ? value.map((v, i) => [String(i), v] as const)
      : Object.entries(value)
    : [];

  return (
    <div style={{ paddingLeft: depth ? 12 : 0 }}>
      <div className="flex items-start gap-1 py-0.5">
        {container ? (
          <button
            onClick={() => setOpen((o) => !o)}
            className="mt-0.5 shrink-0 text-[var(--color-fg-subtle)] hover:text-[var(--color-fg)]"
            aria-label={open ? "Collapse" : "Expand"}
            aria-expanded={open}
          >
            <ChevronRight
              size={11}
              className={clsx("transition-transform", open && "rotate-90")}
            />
          </button>
        ) : (
          <span className="inline-block w-[11px] shrink-0" aria-hidden />
        )}
        <span className="text-[var(--color-fg-muted)]">{name}</span>
        {container ? (
          <span className="text-[var(--color-fg-subtle)]">{summary(value)}</span>
        ) : (
          <>
            <span className="text-[var(--color-fg-subtle)]">:</span>
            <Leaf value={value} />
          </>
        )}
      </div>
      {container && open && (
        <div>
          {entries.length === 0 ? (
            <div
              className="py-0.5 text-[var(--color-fg-subtle)]"
              style={{ paddingLeft: 12 }}
            >
              empty
            </div>
          ) : (
            entries.map(([k, v]) => (
              <Node
                key={k}
                name={k}
                value={v}
                depth={depth + 1}
                defaultOpen={depth < 1}
              />
            ))
          )}
        </div>
      )}
    </div>
  );
}

export function JsonTree({ value }: { value: unknown }) {
  if (!isContainer(value) || Object.keys(value).length === 0) {
    return (
      <div className="flex h-full items-center justify-center px-6 text-center text-xs text-[var(--color-fg-subtle)]">
        No custom_outputs for this turn.
      </div>
    );
  }
  return (
    <div className="h-full overflow-auto p-3 font-mono text-[11px] leading-relaxed">
      {Object.entries(value).map(([k, v]) => (
        <Node key={k} name={k} value={v} depth={0} defaultOpen />
      ))}
    </div>
  );
}
