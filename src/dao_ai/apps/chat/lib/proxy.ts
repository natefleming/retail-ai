/**
 * Header forwarding for the runtime proxy routes.
 *
 * These handlers are a fallback (`next dev` and topologies where the request
 * reaches Next rather than the AgentServer serving `/invocations` and `/v1/*`
 * directly). When they are hit, they must still carry the OBO identity headers
 * the backend derives `user_id` from — otherwise user-scoped routes (sessions,
 * memory) and validation middleware would see no user.
 */
const FORWARDED = [
  "x-forwarded-user",
  "x-forwarded-email",
  "x-forwarded-access-token",
  "authorization",
];

export function forwardedHeaders(
  req: Request,
  base: Record<string, string>,
): Record<string, string> {
  const headers: Record<string, string> = { ...base };
  for (const name of FORWARDED) {
    const value = req.headers.get(name);
    if (value) headers[name] = value;
  }
  return headers;
}
