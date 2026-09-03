/**
 * Runtime streaming proxy to the agent backend for the chat turn.
 *
 * In a deployed Databricks App the MLflow AgentServer serves `/invocations`
 * itself (this handler is never hit); for `next dev` and any topology where
 * the request reaches Next, forward it to `API_PROXY` and stream the SSE
 * response straight back. Reads `API_PROXY` at request time (unlike
 * next.config rewrites, which bake at build time).
 */
import { forwardedHeaders } from "@/lib/proxy";

export const dynamic = "force-dynamic";

export async function POST(req: Request): Promise<Response> {
  const backend = process.env.API_PROXY;
  if (!backend) {
    return new Response(JSON.stringify({ error: "API_PROXY not set" }), {
      status: 502,
      headers: { "content-type": "application/json" },
    });
  }
  const upstream = await fetch(backend, {
    method: "POST",
    headers: forwardedHeaders(req, {
      "content-type": "application/json",
      accept: "text/event-stream",
    }),
    body: await req.text(),
  });
  return new Response(upstream.body, {
    status: upstream.status,
    headers: {
      "content-type":
        upstream.headers.get("content-type") ?? "text/event-stream",
      "cache-control": "no-cache",
    },
  });
}
