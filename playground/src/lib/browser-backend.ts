// Loaded only by `build:web` (see backend.ts). The inference itself lives in `@/runtime`.
import {
  parsePredictResponse,
  PlaygroundError,
  type PredictResponse,
  type RuntimeInfo,
} from "@/lib/api"
import type { BrowserBackend, LoadProgress } from "@/lib/backend"
import { highlightSegments, spansOverlap } from "@/lib/segments"
import { loadPredictor, ModelIntegrityError } from "@/runtime"

/** Written by `scripts/sync-release.mjs` from the immutable release manifest. */
const RELEASE_BASE = "/release/gidi-finance-v2/2.0.2"

type WebRelease = {
  model_version: string
  release_version: string
  onnx_opset?: number
  model: { url: string; sha256: string; size_bytes: number }
}

async function fetchWebRelease(signal: AbortSignal): Promise<WebRelease> {
  const response = await fetch(`${RELEASE_BASE}/web-release.json`, { signal })
  if (!response.ok) throw new Error(`web-release.json: HTTP ${response.status}`)
  const body: unknown = await response.json()
  const release = body as Partial<WebRelease> | null
  if (
    !release ||
    typeof release.model_version !== "string" ||
    typeof release.release_version !== "string" ||
    typeof release.model?.url !== "string" ||
    typeof release.model.sha256 !== "string"
  ) {
    throw new Error("web-release.json has an unexpected shape")
  }
  return release as WebRelease
}

function loadError(cause: unknown): PlaygroundError {
  if (cause instanceof PlaygroundError) return cause
  if (cause instanceof ModelIntegrityError) {
    return new PlaygroundError("integrity", cause.message)
  }
  return new PlaygroundError(
    "model_load",
    cause instanceof Error ? cause.message : "Unknown error while loading the model."
  )
}

export async function loadBrowserPredictor(
  onProgress: (progress: LoadProgress) => void,
  signal: AbortSignal
): Promise<BrowserBackend> {
  const loadStarted = performance.now()
  let predictor: Awaited<ReturnType<typeof loadPredictor>>
  let release: WebRelease
  try {
    release = await fetchWebRelease(signal)
    predictor = await loadPredictor({
      baseUrl: `${window.location.origin}${RELEASE_BASE}`,
      modelUrl: new URL(release.model.url, window.location.origin).href,
      modelSha256: release.model.sha256,
      onProgress: (loaded, total) =>
        onProgress({ loaded, total: total ?? release.model.size_bytes }),
      signal,
    })
  } catch (cause) {
    if (signal.aborted) throw cause
    throw loadError(cause)
  }

  const { info } = predictor
  const runtimeInfo: RuntimeInfo = {
    model_version: info.model_version,
    bundle: `${info.model_version} ${info.release_version}`,
    max_length: info.max_length,
    has_value_head: true,
    architecture: info.architecture,
    precision: info.precision,
    backend: info.backend,
    model_file: release.model.url.split("/").pop() ?? null,
    onnx_opset: release.onnx_opset ?? null,
    bundle_path: `release ${info.release_version} (in your browser)`,
    model_sha256: release.model.sha256,
    load_ms: performance.now() - loadStarted,
  }

  return {
    info: runtimeInfo,
    async predict(text: string): Promise<PredictResponse> {
      const started = performance.now()
      let result: Awaited<ReturnType<typeof predictor.predict>>
      try {
        result = await predictor.predict(text)
      } catch (cause) {
        throw new PlaygroundError(
          "inference",
          cause instanceof Error ? cause.message : "Inference failed."
        )
      }
      const latencyMs = performance.now() - started
      return parsePredictResponse(
        {
          ok: true,
          result,
          latency_ms: latencyMs,
          segments: highlightSegments(text, result.target_span, result.value_span),
          spans_overlap: spansOverlap(result.target_span, result.value_span),
        },
        text
      )
    },
    dispose: () => predictor.dispose(),
  }
}
