"use client";

/**
 * Console runtime — the single source of truth behind all three views.
 *
 * Consumes the dao-ai SSE contract (see lib/contract) and maintains a list of
 * turns, each carrying the anatomy the Conversation, Flow, and Timeline views
 * render: streamed answer text, a separate reasoning channel, tool-call
 * lifecycle with durations, MCP/event log, Vega-Lite visualizations, HITL
 * interrupts, and the MLflow trace (fetched on turn completion).
 */
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

import {
  fetchSession,
  fetchSessionList,
  fetchTrace,
  registerSession,
  streamChat,
  type ChatMessage,
} from "@/lib/api";
import type {
  CustomOutputs,
  HITLInterrupt,
  StreamEvent,
  TraceTree,
  VisualizationSpec,
} from "@/lib/contract";

export interface UIToolCall {
  call_id: string;
  name: string;
  arguments?: unknown;
  started_at?: string;
  duration_ms?: number | null;
  result?: string;
  error?: string;
  status: "in_progress" | "completed" | "error";
}

export interface UIEvent {
  at: number;
  kind: "tool" | "mcp" | "reasoning" | "lifecycle";
  label: string;
  detail?: string;
}

export interface Turn {
  id: string;
  role: "user" | "assistant";
  content: string;
  reasoning: string;
  toolCalls: UIToolCall[];
  visualizations: VisualizationSpec[];
  interrupts: HITLInterrupt[];
  events: UIEvent[];
  traceId?: string;
  trace?: TraceTree | null;
  /** Raw custom_outputs the agent returned this turn (accumulated), for the
   * inspector's Outputs tree — includes agent-specific keys the Console
   * doesn't otherwise render. */
  customOutputs?: Record<string, unknown>;
  status: "streaming" | "done" | "error";
}

export interface SessionRef {
  threadId: string;
  title: string;
  updatedAt: number;
}

interface ConsoleState {
  turns: Turn[];
  isRunning: boolean;
  threadId?: string;
  userId?: string;
  sessions: SessionRef[];
  selectedTurnId?: string;
  /** Extra `custom_inputs.configurable` fields sent with every turn. */
  customInputs: Record<string, unknown>;
  setCustomInputs: (next: Record<string, unknown>) => void;
  send: (text: string) => Promise<void>;
  respondToInterrupt: (decisions: unknown[]) => Promise<void>;
  cancel: () => void;
  newSession: () => void;
  loadSession: (threadId: string) => Promise<void>;
  selectTurn: (turnId?: string) => void;
}

const ConsoleContext = createContext<ConsoleState | null>(null);

export function useConsoleContext(): ConsoleState {
  const ctx = useContext(ConsoleContext);
  if (!ctx) throw new Error("useConsoleContext must be used within ConsoleProvider");
  return ctx;
}

const SESSIONS_KEY = "dao-ai-console:sessions";
const CUSTOM_INPUTS_KEY = "dao-ai-console:custom-inputs";

function loadStoredSessions(): SessionRef[] {
  try {
    const raw = localStorage.getItem(SESSIONS_KEY);
    return raw ? (JSON.parse(raw) as SessionRef[]) : [];
  } catch {
    return [];
  }
}

function storeSessions(sessions: SessionRef[]): void {
  try {
    localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions.slice(0, 50)));
  } catch {
    /* private mode / disabled storage — sidebar just won't persist */
  }
}

function loadStoredCustomInputs(): Record<string, unknown> {
  try {
    const raw = localStorage.getItem(CUSTOM_INPUTS_KEY);
    return raw ? (JSON.parse(raw) as Record<string, unknown>) : {};
  } catch {
    return {};
  }
}

let idCounter = 0;
const nextId = (prefix: string) => `${prefix}_${Date.now()}_${idCounter++}`;

export function ConsoleProvider({
  userId,
  children,
}: {
  userId?: string;
  children: ReactNode;
}) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [threadId, setThreadId] = useState<string | undefined>(undefined);
  const [sessions, setSessions] = useState<SessionRef[]>(loadStoredSessions);
  const [selectedTurnId, setSelectedTurnId] = useState<string | undefined>();
  const [customInputs, setCustomInputsState] = useState<Record<string, unknown>>(
    loadStoredCustomInputs,
  );
  const abortRef = useRef<AbortController | null>(null);
  // Kept in a ref so runStream always reads the latest inputs without needing
  // to be in its dependency list.
  const customInputsRef = useRef(customInputs);
  customInputsRef.current = customInputs;

  const setCustomInputs = useCallback((next: Record<string, unknown>) => {
    setCustomInputsState(next);
    try {
      localStorage.setItem(CUSTOM_INPUTS_KEY, JSON.stringify(next));
    } catch {
      /* private mode / disabled storage — inputs just won't persist */
    }
  }, []);

  const patchAssistant = useCallback(
    (id: string, fn: (t: Turn) => Turn) => {
      setTurns((prev) => prev.map((t) => (t.id === id ? fn(t) : t)));
    },
    [],
  );

  // Prefer the backend session index (configured persistence) for the sidebar;
  // fall back to the localStorage list when no index is available.
  useEffect(() => {
    let cancelled = false;
    void fetchSessionList().then((rows) => {
      if (cancelled || rows.length === 0) return;
      setSessions(
        rows.map((r) => ({
          threadId: r.thread_id,
          title: r.title ?? "Conversation",
          updatedAt: r.updated_at ? Date.parse(r.updated_at) : Date.now(),
        })),
      );
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const rememberSession = useCallback((tid: string, title: string) => {
    setSessions((prev) => {
      const rest = prev.filter((s) => s.threadId !== tid);
      const next = [{ threadId: tid, title, updatedAt: Date.now() }, ...rest];
      storeSessions(next);
      return next;
    });
    // Mirror to the backend index (no-op when persistence isn't configured).
    void registerSession(tid, title);
  }, []);

  const runStream = useCallback(
    async (messages: ChatMessage[], assistantId: string, decisions?: unknown[]) => {
      const controller = new AbortController();
      abortRef.current = controller;
      setIsRunning(true);
      // Captured directly from the stream so the post-run trace fetch / session
      // persistence don't depend on React state having flushed.
      let capturedTraceId: string | undefined;
      let capturedThread: string | undefined = threadId;
      try {
        for await (const event of streamChat(
          {
            messages,
            threadId,
            userId,
            decisions,
            configurable: nonEmptyInputs(customInputsRef.current),
          },
          controller.signal,
        )) {
          applyEvent(
            assistantId,
            event,
            patchAssistant,
            (tid) => {
              capturedThread = tid;
              setThreadId(tid);
            },
            (trid) => {
              capturedTraceId = trid;
            },
          );
        }
      } catch (err) {
        patchAssistant(assistantId, (t) => ({
          ...t,
          status: "error",
          content:
            t.content ||
            `The request could not be completed: ${
              err instanceof Error ? err.message : String(err)
            }`,
        }));
      } finally {
        setIsRunning(false);
        abortRef.current = null;
        // Fetch the authoritative trace + persist the session for the sidebar.
        if (capturedTraceId) {
          const tree = await fetchTrace(capturedTraceId);
          patchAssistant(assistantId, (t) => ({ ...t, trace: tree }));
        }
        if (capturedThread) {
          const firstUser = messages.find((m) => m.role === "user");
          rememberSession(
            capturedThread,
            firstUser?.content?.slice(0, 60) || "Conversation",
          );
        }
      }
    },
    [threadId, userId, patchAssistant, rememberSession],
  );

  const send = useCallback(
    async (text: string) => {
      if (isRunning || !text.trim()) return;
      const userTurn: Turn = {
        id: nextId("u"),
        role: "user",
        content: text,
        reasoning: "",
        toolCalls: [],
        visualizations: [],
        interrupts: [],
        events: [],
        status: "done",
      };
      const assistantTurn: Turn = {
        id: nextId("a"),
        role: "assistant",
        content: "",
        reasoning: "",
        toolCalls: [],
        visualizations: [],
        interrupts: [],
        events: [],
        status: "streaming",
      };
      const history: ChatMessage[] = [
        ...turns
          .filter((t) => t.content)
          .map((t) => ({ role: t.role, content: t.content }) as ChatMessage),
        { role: "user", content: text },
      ];
      setTurns((prev) => [...prev, userTurn, assistantTurn]);
      setSelectedTurnId(assistantTurn.id);
      await runStream(history, assistantTurn.id);
    },
    [isRunning, turns, runStream],
  );

  const respondToInterrupt = useCallback(
    async (decisions: unknown[]) => {
      if (isRunning) return;
      const assistantTurn: Turn = {
        id: nextId("a"),
        role: "assistant",
        content: "",
        reasoning: "",
        toolCalls: [],
        visualizations: [],
        interrupts: [],
        events: [],
        status: "streaming",
      };
      setTurns((prev) => [...prev, assistantTurn]);
      setSelectedTurnId(assistantTurn.id);
      await runStream([], assistantTurn.id, decisions);
    },
    [isRunning, runStream],
  );

  const cancel = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const newSession = useCallback(() => {
    if (isRunning) abortRef.current?.abort();
    setTurns([]);
    setThreadId(undefined);
    setSelectedTurnId(undefined);
  }, [isRunning]);

  const loadSession = useCallback(async (tid: string) => {
    const thread = await fetchSession(tid);
    if (!thread) return;
    const loaded: Turn[] = thread.messages
      .filter((m) => m.role !== "tool")
      .map((m) => ({
        id: nextId("h"),
        role: m.role === "assistant" ? "assistant" : "user",
        content: m.content,
        reasoning: m.reasoning ?? "",
        toolCalls: [],
        visualizations: [],
        interrupts: [],
        events: [],
        status: "done",
      }));
    setThreadId(tid);
    setTurns(loaded);
    setSelectedTurnId(loaded[loaded.length - 1]?.id);
  }, []);

  const value = useMemo<ConsoleState>(
    () => ({
      turns,
      isRunning,
      threadId,
      userId,
      sessions,
      selectedTurnId,
      customInputs,
      setCustomInputs,
      send,
      respondToInterrupt,
      cancel,
      newSession,
      loadSession,
      selectTurn: setSelectedTurnId,
    }),
    [
      turns,
      isRunning,
      threadId,
      userId,
      sessions,
      selectedTurnId,
      customInputs,
      setCustomInputs,
      send,
      respondToInterrupt,
      cancel,
      newSession,
      loadSession,
    ],
  );

  return <ConsoleContext.Provider value={value}>{children}</ConsoleContext.Provider>;
}

/** Apply a single SSE event to the streaming assistant turn. */
function applyEvent(
  assistantId: string,
  event: StreamEvent,
  patch: (id: string, fn: (t: Turn) => Turn) => void,
  onThread: (tid: string) => void,
  onTrace: (traceId: string) => void,
): void {
  const now = Date.now();

  if (event.type === "response.output_text.delta" && event.delta) {
    patch(assistantId, (t) => ({ ...t, content: t.content + event.delta }));
    return;
  }
  if (event.type === "response.reasoning_summary_text.delta" && event.delta) {
    patch(assistantId, (t) => ({ ...t, reasoning: t.reasoning + event.delta }));
    return;
  }
  if (event.type === "response.output_item.added" && event.item) {
    const item = event.item;
    if (item.type === "function_call") {
      patch(assistantId, (t) => ({
        ...t,
        toolCalls: upsertToolCall(t.toolCalls, {
          call_id: item.call_id,
          name: item.name,
          arguments: safeJson(item.arguments),
          started_at: item.started_at,
          status: "in_progress",
        }),
        events: [
          ...t.events,
          { at: now, kind: "tool", label: `${item.name} started` },
        ],
      }));
    } else if (item.type === "function_call_output") {
      patch(assistantId, (t) => ({
        ...t,
        toolCalls: upsertToolCall(t.toolCalls, {
          call_id: item.call_id,
          result: item.status === "completed" ? item.output : undefined,
          error: item.status === "error" ? item.output : undefined,
          duration_ms: item.duration_ms,
          status: item.status,
        }),
        events: [
          ...t.events,
          {
            at: now,
            kind: "tool",
            label: `tool ${item.status}`,
            detail:
              item.duration_ms != null ? `${item.duration_ms} ms` : undefined,
          },
        ],
      }));
    } else if (item.type === "custom_tool_call") {
      patch(assistantId, (t) => ({
        ...t,
        events: [
          ...t.events,
          { at: now, kind: "mcp", label: item.name, detail: shortJson(item.input) },
        ],
      }));
    } else if (item.type === "reasoning") {
      const text = item.summary?.map((s) => s.text).join(" ") ?? "";
      patch(assistantId, (t) => ({
        ...t,
        reasoning: t.reasoning || text,
        events: [...t.events, { at: now, kind: "reasoning", label: "reasoning" }],
      }));
    }
    return;
  }
  if (event.type === "response.output_item.done") {
    const co = event.custom_outputs ?? {};
    applyCustomOutputs(assistantId, co, patch, onThread, onTrace);
    const text =
      event.item && event.item.type === "message"
        ? event.item.content.map((c) => c.text).join("")
        : undefined;
    patch(assistantId, (t) => ({
      ...t,
      content: text && text.length >= t.content.length ? text : t.content,
      status: "done",
    }));
    return;
  }
  if (
    event.type === "response.created" ||
    event.type === "response.in_progress" ||
    event.type === "response.failed"
  ) {
    const bg = event.custom_outputs?.background;
    patch(assistantId, (t) => ({
      ...t,
      events: [
        ...t.events,
        {
          at: now,
          kind: "lifecycle",
          label: event.type.replace("response.", ""),
          detail: bg?.status,
        },
      ],
      status: event.type === "response.failed" ? "error" : t.status,
    }));
    if (event.custom_outputs) {
      applyCustomOutputs(assistantId, event.custom_outputs, patch, onThread, onTrace);
    }
  }
}

function applyCustomOutputs(
  assistantId: string,
  co: CustomOutputs,
  patch: (id: string, fn: (t: Turn) => Turn) => void,
  onThread: (tid: string) => void,
  onTrace: (traceId: string) => void,
): void {
  const tid =
    co.configurable?.thread_id ??
    (co.session?.conversation_id as string | undefined);
  if (tid) onThread(tid);
  if (co.trace_id) onTrace(co.trace_id);
  patch(assistantId, (t) => ({
    ...t,
    // Keep the raw payload (accumulated across events) for the Outputs tree.
    customOutputs: { ...(t.customOutputs ?? {}), ...co },
    traceId: co.trace_id ?? t.traceId,
    reasoning: co.reasoning ?? t.reasoning,
    visualizations: co.visualizations ?? t.visualizations,
    interrupts: co.interrupts ?? t.interrupts,
    toolCalls: co.tool_calls
      ? co.tool_calls.map((r) => ({
          call_id: r.call_id,
          name: r.name ?? "tool",
          arguments: r.arguments,
          started_at: r.started_at,
          duration_ms: r.duration_ms,
          result: r.result_summary,
          error: r.error,
          status: r.status ?? "completed",
        }))
      : t.toolCalls,
  }));
}

function upsertToolCall(list: UIToolCall[], patch: Partial<UIToolCall> & { call_id: string }): UIToolCall[] {
  const idx = list.findIndex((c) => c.call_id === patch.call_id);
  if (idx === -1) {
    return [
      ...list,
      { name: "tool", status: "in_progress", ...patch } as UIToolCall,
    ];
  }
  const next = [...list];
  next[idx] = { ...next[idx], ...patch };
  return next;
}

/** Drop blank keys and empty values so we never send empty configurable fields. */
function nonEmptyInputs(
  inputs: Record<string, unknown>,
): Record<string, unknown> | undefined {
  const entries = Object.entries(inputs).filter(([k, v]) => {
    if (k.trim() === "" || v == null) return false;
    return typeof v === "string" ? v.trim() !== "" : true;
  });
  return entries.length ? Object.fromEntries(entries) : undefined;
}

function safeJson(raw: string): unknown {
  try {
    return JSON.parse(raw);
  } catch {
    return raw;
  }
}

function shortJson(value: unknown): string {
  const s = JSON.stringify(value);
  return s && s.length > 120 ? `${s.slice(0, 120)}…` : s;
}
