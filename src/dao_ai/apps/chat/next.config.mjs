/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // The MLflow AgentServer chat proxy decompresses upstream responses (httpx)
  // but forwards the original `content-encoding: gzip` header, so a gzipped
  // page reaches the browser as an empty/undecodable body (blank screen).
  // Serve identity-encoded responses so the proxy passes them through intact.
  compress: false,
};

export default nextConfig;
