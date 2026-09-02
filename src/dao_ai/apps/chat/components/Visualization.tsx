import { useState } from "react";
import { ChevronRight } from "lucide-react";
import { clsx } from "clsx";

import type { VisualizationSpec } from "@/lib/contract";

/**
 * Dependency-free renderer for the Vega-Lite specs dao-ai attaches to
 * `custom_outputs.visualizations`. A full Vega runtime pulls a large
 * transitive npm tree (d3, yargs-parser, …) that the Databricks Apps internal
 * npm proxy does not reliably carry, so we render the common cases ourselves:
 * a bar/line chart as inline SVG, with a data table fallback and the raw spec
 * available for inspection.
 */

interface Row {
  [k: string]: unknown;
}

function fieldOf(enc: unknown, channel: string): string | undefined {
  const c = (enc as Record<string, { field?: string }> | undefined)?.[channel];
  return c?.field;
}

export function Visualization({ viz }: { viz: VisualizationSpec }) {
  const [showSpec, setShowSpec] = useState(false);
  const spec = (viz?.spec ?? {}) as Record<string, unknown>;
  const data = spec.data as { values?: Row[] } | undefined;
  const values: Row[] = Array.isArray(data?.values) ? (data!.values as Row[]) : [];
  const enc = spec.encoding;
  const mark = typeof spec.mark === "string" ? spec.mark : (spec.mark as { type?: string })?.type;

  const xField = fieldOf(enc, "x");
  const yField = fieldOf(enc, "y");

  const canChart =
    values.length > 0 &&
    !!xField &&
    !!yField &&
    values.every((r) => typeof r[yField] === "number");

  return (
    <div className="rounded-lg border border-[var(--color-line)] bg-[var(--color-ink-900)] p-3 shadow-[var(--shadow-card)]">
      {canChart ? (
        <BarChart rows={values} xField={xField!} yField={yField!} />
      ) : values.length > 0 ? (
        <DataTable rows={values} />
      ) : (
        <div className="text-xs text-[var(--color-fg-subtle)]">
          Visualization ({typeof mark === "string" ? mark : "spec"})
        </div>
      )}
      <button
        onClick={() => setShowSpec((s) => !s)}
        className="mt-2 flex items-center gap-1 text-[11px] text-[var(--color-fg-subtle)] hover:text-[var(--color-fg-muted)]"
      >
        <ChevronRight size={11} className={clsx("transition-transform", showSpec && "rotate-90")} />
        Vega-Lite spec
      </button>
      {showSpec && (
        <pre className="mt-1 max-h-56 overflow-auto rounded bg-[var(--color-ink-850)] p-2 font-mono text-[10px] text-[var(--color-fg-muted)]">
          {JSON.stringify(spec, null, 2)}
        </pre>
      )}
    </div>
  );
}

function BarChart({
  rows,
  xField,
  yField,
}: {
  rows: Row[];
  xField: string;
  yField: string;
}) {
  const W = 480;
  const H = 220;
  const pad = { l: 40, r: 12, t: 12, b: 46 };
  const innerW = W - pad.l - pad.r;
  const innerH = H - pad.t - pad.b;
  const bars = rows.slice(0, 24);
  const max = Math.max(...bars.map((r) => Number(r[yField])), 0) || 1;
  const bw = innerW / bars.length;

  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="bar chart">
      {/* y axis baseline */}
      <line
        x1={pad.l}
        y1={pad.t + innerH}
        x2={pad.l + innerW}
        y2={pad.t + innerH}
        stroke="var(--color-line)"
      />
      {[0, 0.5, 1].map((t) => (
        <text
          key={t}
          x={pad.l - 6}
          y={pad.t + innerH - t * innerH + 3}
          textAnchor="end"
          className="fill-[var(--color-fg-subtle)]"
          fontSize={9}
        >
          {(max * t).toFixed(max < 10 ? 1 : 0)}
        </text>
      ))}
      {bars.map((r, i) => {
        const v = Number(r[yField]);
        const h = (v / max) * innerH;
        const x = pad.l + i * bw + bw * 0.15;
        const w = bw * 0.7;
        const y = pad.t + innerH - h;
        return (
          <g key={i}>
            <rect x={x} y={y} width={w} height={h} rx={2} fill="var(--color-brand-blue)" />
            <text
              x={x + w / 2}
              y={pad.t + innerH + 12}
              textAnchor="middle"
              className="fill-[var(--color-fg-muted)]"
              fontSize={9}
            >
              {String(r[xField]).slice(0, 8)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function DataTable({ rows }: { rows: Row[] }) {
  const cols = Object.keys(rows[0] ?? {});
  return (
    <div className="max-h-64 overflow-auto">
      <table className="w-full border-collapse text-[11px]">
        <thead>
          <tr>
            {cols.map((c) => (
              <th
                key={c}
                className="border border-[var(--color-line)] px-2 py-1 text-left font-medium text-[var(--color-fg-muted)]"
              >
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.slice(0, 30).map((r, i) => (
            <tr key={i}>
              {cols.map((c) => (
                <td
                  key={c}
                  className="border border-[var(--color-line)] px-2 py-1 font-mono text-[var(--color-fg)]"
                >
                  {String(r[c])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
