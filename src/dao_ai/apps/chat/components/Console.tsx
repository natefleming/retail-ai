"use client";

import { useEffect, useState } from "react";
import { Brain, PanelRightClose, PanelRightOpen } from "lucide-react";
import { clsx } from "clsx";

import { Conversation } from "@/components/Conversation";
import { Inspector } from "@/components/Inspector";
import { MemoryPanel } from "@/components/MemoryPanel";
import { SessionSidebar } from "@/components/SessionSidebar";
import { fetchMemory } from "@/lib/api";
import { DEFAULT_UI_CONFIG, loadUIConfig, type UIConfig } from "@/lib/config";
import { ConsoleProvider } from "@/runtime/useConsole";

export function Console() {
  const [config, setConfig] = useState<UIConfig>(DEFAULT_UI_CONFIG);
  const [loaded, setLoaded] = useState(false);
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [memoryOpen, setMemoryOpen] = useState(false);
  const [memoryAvailable, setMemoryAvailable] = useState(false);

  useEffect(() => {
    void loadUIConfig().then((c) => {
      setConfig(c);
      setInspectorOpen(c.mode === "developer");
      setLoaded(true);
    });
    // Probe once whether a memory store is configured (gates the Memory button).
    void fetchMemory().then((m) => setMemoryAvailable(!!m && m.memory !== null));
  }, []);

  if (!loaded) {
    return <div className="h-full" />;
  }

  const showInspector = config.inspector;

  return (
    <ConsoleProvider>
      <div className="flex h-full flex-col bg-[var(--color-ink-950)]">
        <header className="flex items-center gap-3 border-b border-[var(--color-line)] px-4 py-2.5">
          <div className="h-4 w-4 rounded-sm dao-live-gradient" aria-hidden />
          <span className="font-display text-sm tracking-tight text-[var(--color-fg)]">
            {config.title}
          </span>
          <span className="text-xs text-[var(--color-fg-subtle)]">
            {config.mode === "developer" ? "developer" : ""}
          </span>
          <div className="ml-auto flex items-center gap-2">
            {memoryAvailable && (
              <button
                onClick={() => setMemoryOpen((o) => !o)}
                className={clsx(
                  "rounded p-1.5 hover:text-[var(--color-fg)]",
                  memoryOpen
                    ? "text-[var(--color-span-llm)]"
                    : "text-[var(--color-fg-subtle)]",
                )}
                title="Memory"
                aria-label="Memory"
              >
                <Brain size={17} />
              </button>
            )}
            {showInspector && (
              <button
                onClick={() => setInspectorOpen((o) => !o)}
                className="rounded p-1.5 text-[var(--color-fg-subtle)] hover:text-[var(--color-fg)]"
                title={inspectorOpen ? "Hide inspector" : "Show inspector"}
              >
                {inspectorOpen ? (
                  <PanelRightClose size={17} />
                ) : (
                  <PanelRightOpen size={17} />
                )}
              </button>
            )}
          </div>
        </header>

        <div className="flex min-h-0 flex-1">
          {config.session_history && (
            <aside className="hidden w-60 shrink-0 border-r border-[var(--color-line)] md:block">
              <SessionSidebar />
            </aside>
          )}

          <main className="min-w-0 flex-1">
            <Conversation config={config} />
          </main>

          {showInspector && inspectorOpen && (
            <aside
              className={clsx(
                "hidden w-[380px] shrink-0 border-l border-[var(--color-line)] lg:block",
              )}
            >
              <Inspector />
            </aside>
          )}

          {memoryOpen && <MemoryPanel onClose={() => setMemoryOpen(false)} />}
        </div>
      </div>
    </ConsoleProvider>
  );
}
