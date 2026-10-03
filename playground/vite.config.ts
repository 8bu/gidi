import { resolve } from "node:path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

// https://vite.dev/config/
export default defineConfig(({ mode }) => ({
  // `public/` only holds the release files synced for the browser build (`pnpm build:web`).
  publicDir: mode === "web" ? "public" : false,
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": resolve(import.meta.dirname, "./src"),
    },
  },
  server: {
    // `pnpm dev` talks to a running `uv run python scripts/demo_ui.py`.
    proxy: { "/api": "http://127.0.0.1:8765" },
  },
}))
