// Loaded only by `build:web` (see backend.ts). The inference itself lives in `@/runtime`.
import {
  parsePredictResponse,
  PlaygroundError,
  type PredictResponse,
  type RuntimeInfo,
} from "@/lib/api"
import type { BrowserBackend, LoadProgress } from "@/lib/backend"
import type { ModelSpec } from "@/lib/models"
import { highlightSegments, spansOverlap } from "@/lib/segments"
import { loadPredictor, ModelIntegrityError } from "@/runtime"

type WebRelease = {
  model_version: string
  release_version: string
  onnx_opset?: number
  model: {
    sha256: string
    size_bytes: number
    /** Byte ranges of the model file, each its own static asset; they concatenate to the file. */
    parts: { url: string; size_bytes: number }[]
  }
}

/** `web-release.json` is written by `scripts/sync-release.mjs` from the immutable bundle manifest. */
async function fetchWebRelease(
  spec: ModelSpec,
  signal: AbortSignal
): Promise<WebRelease> {
  const response = await fetch(`${spec.releaseBase}/web-release.json`, {
    signal,
  })
  if (!response.ok) throw new Error(`web-release.json: HTTP ${response.status}`)
  const body: unknown = await response.json()
  const release = body as Partial<WebRelease> | null
  if (
    !release ||
    typeof release.model_version !== "string" ||
    typeof release.release_version !== "string" ||
    typeof release.model?.sha256 !== "string" ||
    !Array.isArray(release.model.parts) ||
    release.model.parts.length === 0 ||
    !release.model.parts.every(
      (part) =>
        typeof part?.url === "string" && Number.isInteger(part.size_bytes)
    )
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
    cause instanceof Error
      ? cause.message
      : "Unknown error while loading the model."
  )
}

export async function loadBrowserPredictor(
  spec: ModelSpec,
  onProgress: (progress: LoadProgress) => void,
  signal: AbortSignal
): Promise<BrowserBackend> {
  const loadStarted = performance.now()
  let predictor: Awaited<ReturnType<typeof loadPredictor>>
  let release: WebRelease
  try {
    release = await fetchWebRelease(spec, signal)
    predictor = await loadPredictor({
      baseUrl: `${window.location.origin}${spec.releaseBase}`,
      modelParts: release.model.parts.map((part) => ({
        url: new URL(part.url, window.location.origin).href,
        size: part.size_bytes,
      })),
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
    has_value_head: info.has_value, // the Vietnamese bundle reads the value with the rule-based parser
    architecture: info.architecture,
    precision: info.precision,
    backend: info.backend,
    model_file: "model.int8.onnx",
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
          segments: highlightSegments(
            text,
            result.target_span,
            result.value_span
          ),
          spans_overlap: spansOverlap(result.target_span, result.value_span),
        },
        text
      )
    },
    dispose: () => predictor.dispose(),
  }
}
