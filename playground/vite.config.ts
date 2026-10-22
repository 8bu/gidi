import { execSync } from "node:child_process"
import { resolve } from "node:path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

/** Short git hash (+ `-dirty`), the build id every error event carries. */
function buildId(): string {
  try {
    const run = (cmd: string) =>
      execSync(cmd, { stdio: ["ignore", "pipe", "ignore"] })
        .toString()
        .trim()
    return (
      run("git rev-parse --short HEAD") +
      (run("git status --porcelain -- .") ? "-dirty" : "")
    )
  } catch {
    return "unknown"
  }
}

// https://vite.dev/config/
export default defineConfig(({ mode }) => ({
  // `public/` only holds the release files synced for the browser build (`pnpm build:web`).
  publicDir: mode === "web" ? "public" : false,
  define: {
    __APP_BUILD__: JSON.stringify(process.env.GIDI_BUILD_ID ?? buildId()),
  },
  build: {
    // `pnpm sourcemaps:upload` sets this: maps are written next to the chunks, uploaded to PostHog
    // and deleted again before anything is deployed (see docs/error-tracking.md).
    sourcemap: process.env.GIDI_SOURCEMAPS === "1" ? "hidden" : false,
  },
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
