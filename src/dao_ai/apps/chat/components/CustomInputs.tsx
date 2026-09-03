import { useEffect, useRef, useState } from "react";
import { Plus, SlidersHorizontal, Trash2, X } from "lucide-react";
import { clsx } from "clsx";

import { fetchCustomInputSchema } from "@/lib/api";
import type { CustomInputField } from "@/lib/contract";
import { useConsoleContext } from "@/runtime/useConsole";

interface Row {
  key: string;
  value: string;
}

type Mode = "fields" | "json";

/** Render any committed value as an editable string. */
function valueToString(v: unknown): string {
  if (v == null) return "";
  return typeof v === "string" ? v : JSON.stringify(v);
}

/** Merge the discovered schema (required first) with committed values into
 * editor rows: schema fields prefilled with committed-or-example values, then
 * any extra committed keys. */
function buildRows(
  schema: CustomInputField[],
  committed: Record<string, unknown>,
): Row[] {
  const rows: Row[] = [];
  const used = new Set<string>();
  const ordered = [...schema].sort(
    (a, b) => Number(b.required) - Number(a.required),
  );
  for (const f of ordered) {
    used.add(f.name);
    const has = Object.prototype.hasOwnProperty.call(committed, f.name);
    rows.push({
      key: f.name,
      value: has ? valueToString(committed[f.name]) : valueToString(f.example_value),
    });
  }
  for (const [k, v] of Object.entries(committed)) {
    if (!used.has(k)) rows.push({ key: k, value: valueToString(v) });
  }
  return rows.length ? rows : [{ key: "", value: "" }];
}

function rowsToObject(rows: Row[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (const { key, value } of rows) {
    if (key.trim()) out[key.trim()] = value;
  }
  return out;
}

/**
 * Composer affordance for the extra `custom_inputs.configurable` fields an
 * agent's config may require (e.g. `store_num`). Prepopulates from the agent's
 * discovered field schema, supports a raw-JSON mode, and commits on Save; the
 * runtime persists the values and merges them into every turn.
 */
export function CustomInputs() {
  const { customInputs, setCustomInputs } = useConsoleContext();
  const [open, setOpen] = useState(false);
  const [mode, setMode] = useState<Mode>("fields");
  const [rows, setRows] = useState<Row[]>(() => buildRows([], customInputs));
  const [jsonText, setJsonText] = useState("{}");
  const [jsonError, setJsonError] = useState<string | null>(null);
  const [schema, setSchema] = useState<CustomInputField[]>([]);
  const wrapRef = useRef<HTMLDivElement>(null);

  const schemaByName = new Map(schema.map((f) => [f.name, f]));
  const activeCount = Object.entries(customInputs).filter(([k, v]) => {
    if (!k.trim() || v == null) return false;
    return typeof v === "string" ? v.trim() !== "" : true;
  }).length;

  // Discover the agent's configurable field schema once.
  useEffect(() => {
    void fetchCustomInputSchema().then(setSchema);
  }, []);

  // Rebuild the draft (merging schema + committed values) each time it opens.
  useEffect(() => {
    if (!open) return;
    setMode("fields");
    setJsonError(null);
    setRows(buildRows(schema, customInputs));
  }, [open, schema, customInputs]);

  // Close on outside click.
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open]);

  const updateRow = (i: number, patch: Partial<Row>) =>
    setRows((prev) => prev.map((r, idx) => (idx === i ? { ...r, ...patch } : r)));

  const goToJson = () => {
    setJsonText(JSON.stringify(rowsToObject(rows), null, 2));
    setJsonError(null);
    setMode("json");
  };

  const goToFields = () => {
    const parsed = parseJsonObject(jsonText);
    if (typeof parsed === "string") {
      setJsonError(parsed);
      return;
    }
    setRows(buildRows(schema, parsed));
    setJsonError(null);
    setMode("fields");
  };

  const save = () => {
    if (mode === "json") {
      const parsed = parseJsonObject(jsonText);
      if (typeof parsed === "string") {
        setJsonError(parsed);
        return;
      }
      setCustomInputs(parsed);
    } else {
      setCustomInputs(rowsToObject(rows));
    }
    setOpen(false);
  };

  const tab = (m: Mode, label: string) => (
    <button
      onClick={() => (m === "json" ? goToJson() : goToFields())}
      className={clsx(
        "rounded px-2 py-0.5 text-[11px]",
        mode === m
          ? "bg-[var(--color-ink-800)] text-[var(--color-fg)]"
          : "text-[var(--color-fg-subtle)] hover:text-[var(--color-fg)]",
      )}
    >
      {label}
    </button>
  );

  return (
    <div ref={wrapRef} className="relative">
      <button
        onClick={() => setOpen((o) => !o)}
        className={clsx(
          "relative rounded-lg p-2 transition-colors",
          activeCount > 0 || open
            ? "text-[var(--color-primary)]"
            : "text-[var(--color-fg-subtle)] hover:text-[var(--color-fg)]",
        )}
        title="Custom inputs"
        aria-label="Custom inputs"
      >
        <SlidersHorizontal size={16} />
        {activeCount > 0 && (
          <span className="absolute -right-0.5 -top-0.5 flex h-3.5 min-w-3.5 items-center justify-center rounded-full bg-[var(--color-primary)] px-1 text-[9px] font-semibold text-white">
            {activeCount}
          </span>
        )}
      </button>

      {open && (
        <div className="absolute bottom-11 left-0 z-20 w-80 rounded-lg border border-[var(--color-line)] bg-[var(--color-ink-900)] p-3 shadow-[var(--shadow-card)]">
          <div className="mb-1 flex items-center gap-2">
            <span className="text-sm font-medium text-[var(--color-fg)]">
              Custom inputs
            </span>
            <div className="ml-auto flex items-center gap-0.5 rounded-md bg-[var(--color-ink-950)] p-0.5">
              {tab("fields", "Fields")}
              {tab("json", "JSON")}
            </div>
            <button
              onClick={() => setOpen(false)}
              className="text-[var(--color-fg-subtle)] hover:text-[var(--color-fg)]"
              aria-label="Close"
            >
              <X size={14} />
            </button>
          </div>
          <p className="mb-3 text-[11px] leading-snug text-[var(--color-fg-subtle)]">
            Sent as <code>custom_inputs.configurable</code> on every turn.
            {schema.length > 0 && " Fields marked * are required by this agent."}
          </p>

          {mode === "fields" ? (
            <>
              <div className="space-y-2">
                {rows.map((row, i) => {
                  const spec = schemaByName.get(row.key.trim());
                  return (
                    <div key={i} className="flex items-center gap-1.5">
                      <div className="relative min-w-0 flex-1">
                        <input
                          value={row.key}
                          onChange={(e) => updateRow(i, { key: e.target.value })}
                          placeholder="key"
                          title={spec?.description ?? undefined}
                          className="w-full rounded border border-[var(--color-line)] bg-[var(--color-ink-950)] px-2 py-1 font-mono text-[12px] text-[var(--color-fg)] outline-none focus:border-[var(--color-primary)]"
                        />
                        {spec?.required && (
                          <span
                            className="pointer-events-none absolute right-1.5 top-1/2 -translate-y-1/2 text-[13px] text-[var(--color-span-error)]"
                            title="Required"
                          >
                            *
                          </span>
                        )}
                      </div>
                      <input
                        value={row.value}
                        onChange={(e) => updateRow(i, { value: e.target.value })}
                        placeholder={
                          spec?.example_value != null
                            ? valueToString(spec.example_value)
                            : "value"
                        }
                        className="min-w-0 flex-1 rounded border border-[var(--color-line)] bg-[var(--color-ink-950)] px-2 py-1 font-mono text-[12px] text-[var(--color-fg)] outline-none focus:border-[var(--color-primary)]"
                      />
                      <button
                        onClick={() =>
                          setRows((prev) =>
                            prev.length > 1
                              ? prev.filter((_, idx) => idx !== i)
                              : [{ key: "", value: "" }],
                          )
                        }
                        className="shrink-0 rounded p-1 text-[var(--color-fg-subtle)] hover:text-[var(--color-span-error)]"
                        aria-label="Remove field"
                      >
                        <Trash2 size={13} />
                      </button>
                    </div>
                  );
                })}
              </div>
              <div className="mt-2">
                <button
                  onClick={() => setRows((prev) => [...prev, { key: "", value: "" }])}
                  className="flex items-center gap-1 text-[12px] text-[var(--color-fg-muted)] hover:text-[var(--color-fg)]"
                >
                  <Plus size={13} /> Add field
                </button>
              </div>
            </>
          ) : (
            <>
              <textarea
                value={jsonText}
                onChange={(e) => setJsonText(e.target.value)}
                spellCheck={false}
                rows={7}
                placeholder={'{\n  "store_num": "12345"\n}'}
                className="w-full resize-y rounded border border-[var(--color-line)] bg-[var(--color-ink-950)] px-2 py-1.5 font-mono text-[12px] leading-snug text-[var(--color-fg)] outline-none focus:border-[var(--color-primary)]"
              />
              {jsonError && (
                <div className="mt-1 text-[11px] text-[var(--color-span-error)]">
                  {jsonError}
                </div>
              )}
            </>
          )}

          <div className="mt-3 flex items-center justify-end">
            <button
              onClick={save}
              className="rounded-lg bg-[var(--color-brand-blue)] px-3 py-1 text-[12px] text-white"
            >
              Save
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

/** Parse a JSON object, returning the object or an error message string. */
function parseJsonObject(text: string): Record<string, unknown> | string {
  const trimmed = text.trim();
  if (!trimmed) return {};
  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch (e) {
    return `Invalid JSON: ${e instanceof Error ? e.message : String(e)}`;
  }
  if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
    return "Must be a JSON object, e.g. { \"store_num\": \"12345\" }";
  }
  return parsed as Record<string, unknown>;
}
