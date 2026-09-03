/**
 * dao-ai backend contract — TypeScript mirror of the streaming/route shapes the
 * Console consumes. Kept in lockstep with the Python runtime:
 *
 *  - SSE events  ← dao_ai.models.LanggraphResponsesAgent.apredict_stream
 *  - custom_outputs ← _build_custom_outputs_async + streaming mirrors
 *  - trace tree  ← dao_ai.apps.traces.build_trace_tree  (GET /v1/traces/{id})
 *  - session     ← dao_ai.apps.sessions.load_session    (GET /v1/sessions/{id})
 *
 * These types describe the OpenAI Responses item taxonomy dao-ai emits, plus
 * the dao-ai-specific extensions (tool duration_ms, reasoning channel, mcp
 * events, visualizations, interrupts, background lifecycle).
 */

// ── Stream item payloads (event.item) ──────────────────────────────────────

/** A tool call as it starts — dao_ai.tool.start → function_call. */
export interface FunctionCallItem {
  type: "function_call";
  id: string;
  call_id: string;
  name: string;
  /** JSON-encoded arguments. */
  arguments: string;
  status: "in_progress";
  started_at?: string;
}

/** A tool call's result — dao_ai.tool.end/error → function_call_output. */
export interface FunctionCallOutputItem {
  type: "function_call_output";
  call_id: string;
  output: string;
  status: "completed" | "error";
  duration_ms: number | null;
}

/** MCP / audit notification, kept as a custom_tool_call for back-compat. */
export interface CustomToolCallItem {
  type: "custom_tool_call";
  id: string;
  status: "in_progress";
  /** The envelope channel, e.g. "mcp.progress". */
  name: string;
  input: Record<string, unknown>;
}

/** Reasoning surfaced as a durable item (summary_text form). */
export interface ReasoningItem {
  type: "reasoning";
  id: string;
  summary: Array<{ type: "summary_text"; text: string }>;
}

/** The terminal assistant message. */
export interface MessageItem {
  type: "message";
  id: string;
  role: "assistant";
  content: Array<{ type: "output_text"; text: string; annotations?: unknown[] }>;
}

export type StreamItem =
  | FunctionCallItem
  | FunctionCallOutputItem
  | CustomToolCallItem
  | ReasoningItem
  | MessageItem;

// ── custom_outputs (attached to the terminal / background events) ───────────

/** One merged tool-call record, mirrored for replay parity. */
export interface ToolCallRecord {
  call_id: string;
  name?: string;
  arguments?: Record<string, unknown>;
  started_at?: string;
  duration_ms?: number | null;
  result_summary?: string;
  error?: string;
  status?: "in_progress" | "completed" | "error";
}

export interface VisualizationSpec {
  /** A full Vega-Lite spec. */
  spec: Record<string, unknown>;
  message_id?: string;
}

export interface HITLInterrupt {
  type: "request_action" | "request_confirmation" | string;
  id?: string;
  title?: string;
  instructions?: string;
  actions?: Array<{
    name: string;
    arguments?: Record<string, unknown>;
    description?: string;
  }>;
}

export interface BackgroundStatus {
  response_id?: string;
  status?: "queued" | "in_progress" | "completed" | "failed" | "cancelled";
  thread_id?: string;
  cursor?: number;
  error?: { type?: string; message?: string; reason?: string };
}

export interface CustomOutputs {
  trace_id?: string;
  configurable?: { thread_id?: string; user_id?: string; [k: string]: unknown };
  session?: Record<string, unknown>;
  tool_calls?: ToolCallRecord[];
  reasoning?: string;
  mcp_events?: Array<Record<string, unknown>>;
  visualizations?: VisualizationSpec[];
  interrupts?: HITLInterrupt[];
  background?: BackgroundStatus;
  [k: string]: unknown;
}

// ── SSE events ──────────────────────────────────────────────────────────────

export type StreamEventType =
  | "response.output_text.delta"
  | "response.reasoning_summary_text.delta"
  | "response.output_item.added"
  | "response.output_item.done"
  | "response.created"
  | "response.in_progress"
  | "response.failed";

/** A single decoded `data: {...}` line from the agent SSE stream. */
export interface StreamEvent {
  type: StreamEventType;
  delta?: string;
  item_id?: string;
  item?: StreamItem;
  custom_outputs?: CustomOutputs;
  id?: string;
}

// ── Trace waterfall (GET /v1/traces/{trace_id}) ─────────────────────────────

export interface SpanEventNode {
  name: string;
  timestamp_ms: number | null;
  attributes: Record<string, unknown>;
}

export interface SpanNode {
  span_id: string;
  parent_id: string | null;
  name: string;
  span_type: string | null;
  status: string;
  start_offset_ms: number;
  duration_ms: number;
  inputs: Record<string, unknown>;
  outputs: Record<string, unknown>;
  attributes: Record<string, unknown>;
  events: SpanEventNode[];
  children: SpanNode[];
}

export interface TraceTree {
  trace_id: string;
  root_span_id: string | null;
  duration_ms: number;
  spans: SpanNode[];
}

// ── Session reload (GET /v1/sessions/{thread_id}) ───────────────────────────

export interface SessionMessage {
  role: "user" | "assistant" | "tool";
  content: string;
  reasoning?: string;
  name?: string;
  /** Tool calls this assistant message issued (reconstructed from the checkpointer). */
  tool_calls?: { call_id?: string; name?: string; arguments?: unknown }[];
  /** For a tool message, the call id it answers (to pair result → call). */
  tool_call_id?: string;
}

export interface SessionThread {
  thread_id: string;
  messages: SessionMessage[];
}

// ── Session index list (GET /v1/sessions) ───────────────────────────────────

export interface SessionListItem {
  thread_id: string;
  title: string | null;
  updated_at: string | null;
}

// ── Session metadata (GET /v1/sessions/{id}/meta) ───────────────────────────

export interface SessionMeta {
  thread_id: string;
  checkpoint_id: string | null;
  last_modified: string | null;
  step: number | null;
  message_count: number;
}

// ── Memory viewer (GET /v1/memory) ──────────────────────────────────────────

export interface MemoryEntry {
  key: string | null;
  value: unknown;
  created_at: string | null;
  updated_at: string | null;
}

export interface MemoryResponse {
  user_id?: string;
  namespaces?: string[];
  /** null when no memory store is configured. */
  memory: Record<string, MemoryEntry[]> | null;
}

/** A `custom_inputs.configurable` field the agent's config requires/accepts,
 * discovered from its CustomFieldValidationMiddleware. */
export interface CustomInputField {
  name: string;
  description: string | null;
  required: boolean;
  example_value: unknown;
}
