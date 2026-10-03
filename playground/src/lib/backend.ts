import type { PredictResponse, RuntimeInfo } from "@/lib/api"

/**
 * `pnpm build:web` (vite `--mode web`) runs the model in the browser; every other build talks to
 * the Python playground server (`scripts/demo_ui.py`). The constant lets the bundler drop the
 * whole browser runtime from the API build.
 */
export const BROWSER_MODE = import.meta.env.MODE === "web"

export type LoadProgress = { loaded: number; total: number | null }

export interface BrowserBackend {
  info: RuntimeInfo
  predict(text: string): Promise<PredictResponse>
  dispose(): Promise<void>
}

export async function loadBrowserBackend(
  onProgress: (progress: LoadProgress) => void,
  signal: AbortSignal
): Promise<BrowserBackend> {
  if (!BROWSER_MODE) throw new Error("The browser runtime is only built with `pnpm build:web`.")
  const { loadBrowserPredictor } = await import("@/lib/browser-backend")
  return loadBrowserPredictor(onProgress, signal)
}
