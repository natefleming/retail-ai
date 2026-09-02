/** Transport to the dao-ai agent backend (served on the same origin). */

import type {
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
  const configurable: Record<string, unknown> = {};
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

export async function fetchTrace(traceId: string): Promise<TraceTree | null> {
  const res = await fetch(`/v1/traces/${encodeURIComponent(traceId)}`);
  if (!res.ok) return null;
  return (await res.json()) as TraceTree;
}

export async function fetchSession(threadId: string): Promise<SessionThread | null> {
  const res = await fetch(`/v1/sessions/${encodeURIComponent(threadId)}`);
  if (!res.ok) return null;
  return (await res.json()) as SessionThread;
}
