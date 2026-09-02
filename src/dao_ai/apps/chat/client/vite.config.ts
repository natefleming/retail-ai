import { fileURLToPath, URL } from "node:url";

import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";

// The client builds to ../server/public; the console server serves that
// directory statically and proxies /api/chat to the dao-ai agent backend.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  build: {
    outDir: "../server/public",
    emptyOutDir: true,
  },
  server: {
    port: 3000,
    // In `npm run dev`, proxy API calls to a locally running agent backend.
    proxy: {
      "/api/chat": { target: "http://localhost:8000", changeOrigin: true },
      "/v1": { target: "http://localhost:8000", changeOrigin: true },
    },
  },
});
