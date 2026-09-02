/**
 * Runtime proxy for the agent's `/v1/*` routes (traces, sessions, responses).
 * Same rationale as the /invocations handler: forwards to the `API_PROXY`
 * backend origin and streams the response, reading env at request time.
 */
export const dynamic = "force-dynamic";

function backendOrigin(): string | undefined {
  const b = process.env.API_PROXY;
  return b ? new URL(b).origin : undefined;
}

async function proxy(req: Request, path: string[]): Promise<Response> {
  const origin = backendOrigin();
  if (!origin) {
    return new Response(JSON.stringify({ error: "API_PROXY not set" }), {
      status: 502,
      headers: { "content-type": "application/json" },
    });
  }
  const search = new URL(req.url).search;
  const target = `${origin}/v1/${path.join("/")}${search}`;
  const upstream = await fetch(target, {
    method: req.method,
    headers: { "content-type": "application/json", accept: "text/event-stream" },
    body:
      req.method !== "GET" && req.method !== "HEAD"
        ? await req.text()
        : undefined,
  });
  return new Response(upstream.body, {
    status: upstream.status,
    headers: {
      "content-type":
        upstream.headers.get("content-type") ?? "application/json",
      "cache-control": "no-cache",
    },
  });
}

export async function GET(
  req: Request,
  { params }: { params: { path: string[] } },
): Promise<Response> {
  return proxy(req, params.path);
}

export async function POST(
  req: Request,
  { params }: { params: { path: string[] } },
): Promise<Response> {
  return proxy(req, params.path);
}
