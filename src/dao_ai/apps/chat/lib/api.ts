/** Transport to the dao-ai agent backend (served on the same origin). */

import type {
  CustomInputField,
  MemoryResponse,
  SessionListItem,
  SessionMeta,
  SessionThread,
  StreamEvent,
  TraceTree,
} from "@/lib/contract";

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

export interface ChatRequest {
  messages: ChatMessage[];
  threadId?: string;
  userId?: string;
  /**
   * Extra `custom_inputs.configurable` fields the agent's config requires
   * (e.g. `store_num`). Merged in before the reserved `thread_id`/`user_id`,
   * which always take precedence.
   */
  configurable?: Record<string, unknown>;
  /** Resume a HITL interrupt with approve/reject/edit decisions. */
  decisions?: unknown[];
}

/**
 * Stream a turn from the agent. Posts an OpenAI Responses-style request to
 * `/invocations` with `stream: true` and yields each decoded SSE event
 * (dao-ai emits `data: {ResponsesAgentStreamEvent}\n\n`).
 */
export async function* streamChat(
  req: ChatRequest,
  signal?: AbortSignal,
): AsyncGenerator<StreamEvent> {
  // Config-required fields first; reserved keys below override them so the
  // runtime always controls thread/user identity.
  const configurable: Record<string, unknown> = { ...(req.configurable ?? {}) };
  if (req.threadId) configurable.thread_id = req.threadId;
  if (req.userId) configurable.user_id = req.userId;

  const custom_inputs: Record<string, unknown> = {};
  if (Object.keys(configurable).length) custom_inputs.configurable = configurable;
  if (req.decisions) custom_inputs.decisions = req.decisions;

  const res = await fetch("/invocations", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({
      input: req.messages,
      stream: true,
      ...(Object.keys(custom_inputs).length ? { custom_inputs } : {}),
    }),
    signal,
  });

  if (!res.ok || !res.body) {
    throw new Error(`Agent request failed: ${res.status} ${res.statusText}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line.
    let sep: number;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      const event = parseSseFrame(frame);
      if (event) yield event;
    }
  }
  const tail = parseSseFrame(buffer);
  if (tail) yield tail;
}

function parseSseFrame(frame: string): StreamEvent | null {
  const dataLines = frame
    .split("\n")
    .filter((l) => l.startsWith("data:"))
    .map((l) => l.slice(5).trim());
  if (!dataLines.length) return null;
  const payload = dataLines.join("\n");
  if (!payload || payload === "[DONE]") return null;
  try {
    return JSON.parse(payload) as StreamEvent;
  } catch {
    return null;
  }
}

/** Resolve a Databricks workspace UI deep link for a trace (null if unavailable). */
export async function fetchTraceUrl(traceId: string): Promise<string | null> {
  try {
    const res = await fetch(`/v1/trace-url?trace_id=${encodeURIComponent(traceId)}`);
    if (!res.ok) return null;
    const data = (await res.json()) as { url: string | null };
    return data.url ?? null;
  } catch {
    return null;
  }
}

export async function fetchTrace(traceId: string): Promise<TraceTree | null> {
  // trace_id is passed as a query param: trace_location ids are UC URIs
  // (trace:/<catalog>.<schema>.<prefix>/<id>) whose slashes break a path param.
  const res = await fetch(`/v1/traces?trace_id=${encodeURIComponent(traceId)}`);
  if (!res.ok) return null;
  return (await res.json()) as TraceTree;
}

export async function fetchSession(threadId: string): Promise<SessionThread | null> {
  const res = await fetch(`/v1/sessions/${encodeURIComponent(threadId)}`);
  if (!res.ok) return null;
  return (await res.json()) as SessionThread;
}

/** List the current user's sessions from the configured persistence index. */
export async function fetchSessionList(): Promise<SessionListItem[]> {
  try {
    const res = await fetch("/v1/sessions");
    if (!res.ok) return [];
    return (await res.json()) as SessionListItem[];
  } catch {
    return [];
  }
}

/** Register/refresh a thread in the persistence index (fire-and-forget). */
export async function registerSession(
  threadId: string,
  title: string,
): Promise<void> {
  try {
    await fetch("/v1/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ thread_id: threadId, title }),
    });
  } catch {
    /* index unavailable — sidebar falls back to localStorage */
  }
}

export async function fetchSessionMeta(
  threadId: string,
): Promise<SessionMeta | null> {
  try {
    const res = await fetch(`/v1/sessions/${encodeURIComponent(threadId)}/meta`);
    if (!res.ok) return null;
    return (await res.json()) as SessionMeta;
  } catch {
    return null;
  }
}

/** Fetch the configurable fields the agent's config requires (for prefill). */
export async function fetchCustomInputSchema(): Promise<CustomInputField[]> {
  try {
    const res = await fetch("/v1/custom-inputs");
    if (!res.ok) return [];
    const data = (await res.json()) as { fields?: CustomInputField[] };
    return data.fields ?? [];
  } catch {
    return [];
  }
}

/** Fetch the current user's long-term memory (null when no store configured). */
export async function fetchMemory(): Promise<MemoryResponse | null> {
  try {
    const res = await fetch("/v1/memory");
    if (!res.ok) return null;
    return (await res.json()) as MemoryResponse;
  } catch {
    return null;
  }
}
