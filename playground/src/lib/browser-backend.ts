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
import { captureError } from "@/lib/analytics"
import { setErrorContext, type ErrorStage } from "@/lib/error-report"
import { LANGS } from "@/lib/language"
import { reloadOnce } from "@/lib/reload"
import { loadPredictor, ModelIntegrityError } from "@/runtime"
import {
  ManifestError,
  type ReleaseEntry,
  type ReleasePart,
} from "@/runtime/manifest"
import {
  dropStore,
  prefetchRelease,
  pruneStores,
  readStoredRelease,
} from "@/runtime/model-download"
import {
  cacheNameFor,
  fetchManifestJson,
  resolveRelease,
  type Resolved,
} from "@/runtime/release-manager"

/** Wait this long after the model is up before fetching a newer release. */
const PREFETCH_DELAY_MS = 3000

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

/** Report a non-fatal stage (manifest, Cache Storage, background fetch) to the owner's error tracking. */
function reporter(stage: ErrorStage) {
  return (kind: string, error: unknown, recovery?: string): void =>
    captureError(error, stage, { kind, recovery })
}

const absolute = (parts: readonly ReleasePart[]) =>
  parts.map((part) => ({
    url: new URL(part.url, window.location.origin).href,
    size: part.size_bytes,
  }))

export async function loadBrowserPredictor(
  spec: ModelSpec,
  onProgress: (progress: LoadProgress) => void,
  signal: AbortSignal,
  stage: ErrorStage = "load"
): Promise<BrowserBackend> {
  const loadStarted = performance.now()
  const report = reporter(stage)
  void pruneStores(LANGS.map(cacheNameFor))
  setErrorContext({ language: spec.lang })

  let resolved: Resolved
  try {
    resolved = await resolveRelease(
      spec.lang,
      {
        fetchManifest: (s) => fetchManifestJson(s),
        readStored: readStoredRelease,
        dropStore,
        report,
      },
      signal
    )
  } catch (cause) {
    if (signal.aborted) throw cause
    if (cause instanceof ManifestError && cause.problem === "outdated_client") {
      reloadOnce() // new code is waiting; the page was loaded before the model changed
      throw new PlaygroundError(
        "model_load",
        "A newer version of Gidi is available. Reload the page. / Có bản Gidi mới, hãy tải lại trang."
      )
    }
    if (cause instanceof ManifestError && cause.problem === "disabled") {
      throw new PlaygroundError(
        "model_load",
        "This language's model is temporarily switched off. / Mô hình của ngôn ngữ này đang tạm tắt."
      )
    }
    throw loadError(cause)
  }
  setErrorContext({
    manifest_schema: resolved.manifest?.schema ?? null,
    manifest_active: resolved.manifest?.active ?? null,
  })

  const open = (run: ReleaseEntry) => {
    setErrorContext({ model: `${run.id} ${run.version}` })
    return loadPredictor({
      baseUrl: `${window.location.origin}${run.base}`,
      modelParts: absolute(run.parts),
      modelSha256: run.sha256,
      modelVersion: run.id,
      releaseVersion: run.version,
      tokenizerType: run.tokenizer_type,
      store: { name: resolved.cacheName, release: run },
      onProgress: (loaded, total) =>
        onProgress({ loaded, total: total ?? run.size_bytes }),
      onDiagnostic: (kind, error) =>
        report(
          kind,
          Object.assign(new Error(error.message), { name: error.name })
        ),
      signal,
    })
  }

  let run = resolved.run
  let predictor: Awaited<ReturnType<typeof loadPredictor>>
  try {
    try {
      predictor = await open(run)
    } catch (cause) {
      // The cached older release cannot be read any more: fall back to the active one.
      if (
        signal.aborted ||
        resolved.update === null ||
        cause instanceof ModelIntegrityError
      )
        throw cause
      report("cached_release_unusable", cause, "load_active_release")
      run = resolved.update
      resolved = { ...resolved, update: null }
      predictor = await open(run)
    }
  } catch (cause) {
    if (signal.aborted) throw cause
    throw loadError(cause)
  }

  const update = resolved.update
  if (update !== null) {
    // A newer (or rolled-back) release: fetch it quietly; the next load switches to it, and only
    // after its sha256 passed. A failure changes nothing about the model that is running.
    window.setTimeout(() => {
      void prefetchRelease(absolute(update.parts), update.sha256, signal, {
        name: resolved.cacheName,
        release: update,
        report: (kind, error) =>
          report(kind, error, "keep_running_cached_release"),
      })
    }, PREFETCH_DELAY_MS)
  }

  const release = {
    onnx_opset: run.onnx_opset,
    model: { sha256: run.sha256 },
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
