/**
 * dao-ai Console server.
 *
 * In a deployed Databricks App the MLflow AgentServer (the agent backend) binds
 * the app port, serves the agent contract (`/invocations`, `/v1/responses`,
 * `/v1/traces`, `/v1/sessions`, A2A) directly, and proxies UI routes
 * (`/`, `/assets/*`, `/api/*`) to this server. So this server's job is to serve
 * the built SPA and expose `/api/config` (the injected AppUIModel). It also
 * proxies `/invocations` and `/v1/*` to `API_PROXY` for `npm run dev` and for
 * any topology where those are routed here instead of the backend.
 */
import path from "node:path";
import { fileURLToPath } from "node:url";

import express, {
  type Request,
  type Response,
  type Express,
} from "express";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app: Express = express();
const PORT = Number(process.env.CHAT_APP_PORT || process.env.PORT || 3000);

app.use(express.json({ limit: "10mb" }));

app.get("/ping", (_req: Request, res: Response) => res.status(200).send("pong"));

// AppUIModel (mode / inspector / session_history / title / subtitle), injected
// at deploy time as JSON. Empty object => the client uses its defaults.
app.get("/api/config", (_req: Request, res: Response) => {
  const raw = process.env.DAO_AI_UI_CONFIG;
  try {
    res.json(raw ? JSON.parse(raw) : {});
  } catch {
    res.json({});
  }
});

// Byte-for-byte streaming proxy to the agent backend (dev / fallback topology).
const backend = process.env.API_PROXY;
if (backend) {
  const backendOrigin = new URL(backend).origin;

  const forward = (targetUrl: (req: Request) => string) =>
    async (req: Request, res: Response) => {
      try {
        const headers = { ...req.headers } as Record<string, string>;
        delete headers["content-length"];
        delete headers.host;
        const upstream = await fetch(targetUrl(req), {
          method: req.method,
          headers,
          body:
            req.method !== "GET" && req.method !== "HEAD"
              ? JSON.stringify(req.body)
              : undefined,
        });
        res.status(upstream.status);
        upstream.headers.forEach((value, key) => {
          if (key.toLowerCase() !== "content-length") res.setHeader(key, value);
        });
        if (upstream.body) {
          const reader = upstream.body.getReader();
          for (;;) {
            const { done, value } = await reader.read();
            if (done) break;
            res.write(value);
          }
        }
        res.end();
      } catch (error) {
        res.status(502).json({
          error: "proxy_error",
          message: error instanceof Error ? error.message : String(error),
        });
      }
    };

  app.all("/invocations", forward(() => backend));
  app.all(/^\/v1\/.*/, forward((req) => backendOrigin + req.originalUrl));
  console.log(`Proxying /invocations and /v1/* to ${backendOrigin}`);
}

// Serve the built SPA and fall back to index.html for client routes.
const publicDir = path.join(__dirname, "..", "public");
app.use(express.static(publicDir));
app.get(/^\/(?!api|invocations|v1|ping).*/, (_req: Request, res: Response) => {
  res.sendFile(path.join(publicDir, "index.html"));
});

app.listen(PORT, () => {
  console.log(`dao-ai Console server listening on ${PORT}`);
});
