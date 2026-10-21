import { useMemo, useSyncExternalStore } from "react"

import { track } from "@/lib/analytics"
import { predict, type PredictResponse, type Span } from "@/lib/api"
import {
  BROWSER_MODE,
  loadBrowserBackend,
  type BrowserBackend,
  type LoadProgress,
} from "@/lib/backend"

import { amountToVnd, widenSlangValue } from "./amount"
import {
  TX_TYPES,
  type ModelReading,
  type NoteRecord,
  type TxType,
} from "./types"

export type GidiStatus = "loading" | "ready" | "error"

interface GidiState {
  status: GidiStatus
  /** Model download, 0..1; null until the total size is known. */
  progress: number | null
  /** Bytes received so far and the expected total (null when unknown). */
  loaded: number
  total: number | null
  error: string | null
}

/** "12,3 MB": decimal megabytes with a Vietnamese decimal comma. */
export const formatMegabytes = (bytes: number): string =>
  `${(bytes / 1e6).toFixed(1).replace(".", ",")} MB`

interface NetworkInformation {
  effectiveType?: string
  saveData?: boolean
}

/** Connection and memory hints for diagnosing a failed download; none identify the visitor. */
function environment(): {
  effective_type: string | null
  save_data: boolean | null
  online: boolean
  device_memory: number | null
} {
  const nav = navigator as Navigator & {
    connection?: NetworkInformation
    deviceMemory?: number
  }
  return {
    effective_type: nav.connection?.effectiveType ?? null,
    save_data: nav.connection?.saveData ?? null,
    online: navigator.onLine,
    device_memory: nav.deviceMemory ?? null,
  }
}

type Predictor = (text: string) => Promise<PredictResponse>

const toMessage = (cause: unknown): string =>
  cause instanceof Error ? cause.message : "Không thể tải mô hình."

const copySpan = (span: Span | null | undefined): [number, number] | null =>
  span ? [span[0], span[1]] : null

const isTxType = (value: string): value is TxType =>
  (TX_TYPES as readonly string[]).includes(value)

// One model per page: module-level, so a StrictMode double mount does not load twice.
const idle = { progress: null, loaded: 0, total: null, error: null }
let state: GidiState = { status: "loading", ...idle }
let predictor: Promise<Predictor> | null = null
const listeners = new Set<() => void>()

function setState(next: GidiState): void {
  state = next
  for (const listener of listeners) listener()
}

function reportProgress({ loaded, total }: LoadProgress): void {
  if (state.status !== "loading") return
  const progress =
    total !== null && total > 0
      ? Math.min(1, Math.max(0, loaded / total))
      : null
  setState({ status: "loading", progress, loaded, total, error: null })
}

async function loadPredictor(): Promise<Predictor> {
  if (!BROWSER_MODE) {
    // Dev server: the model runs behind `/api/predict`, nothing to download.
    return predict
  }
  const backend: BrowserBackend = await loadBrowserBackend(
    reportProgress,
    new AbortController().signal
  )
  return (text) => backend.predict(text)
}

function load(): Promise<Predictor> {
  if (predictor === null) {
    setState({ status: "loading", ...idle })
    const started = performance.now()
    const loaded = loadPredictor()
    predictor = loaded
    loaded.then(
      () => {
        track("model_loaded", { ms: Math.round(performance.now() - started) })
        if (predictor === loaded)
          setState({ ...idle, status: "ready", progress: 1 })
      },
      (cause: unknown) => {
        const error = toMessage(cause)
        track("model_load_failed", {
          ms: Math.round(performance.now() - started),
          error,
          error_name: cause instanceof Error ? cause.name : "unknown",
          loaded_bytes: state.loaded,
          total_bytes: state.total,
          ...environment(),
        })
        if (predictor === loaded) {
          setState({ ...idle, status: "error", error })
        }
      }
    )
  }
  return predictor
}

function retry(): void {
  if (state.status !== "error") return
  predictor = null
  void load()
}

async function classify(text: string): Promise<NoteRecord> {
  const note = text.trim()
  if (note === "") throw new Error("Ghi chú đang trống.")
  const { result } = await (await load())(note)
  const { valueText, span: valueSpan } = widenSlangValue(
    note,
    result.value_text ?? null,
    copySpan(result.value_span)
  )
  const type = isTxType(result.type) ? result.type : null
  if (type === null)
    throw new Error(`Mô hình trả về loại không hợp lệ: ${result.type}`)
  const now = Date.now()
  const original: ModelReading = {
    type,
    typeConfidence: result.type_confidence,
    target: result.target,
    targetSpan: copySpan(result.target_span),
    targetConfidence: result.target_confidence,
    valueText,
    valueSpan,
    amount: valueText ? amountToVnd(valueText) : null,
    modelVersion: result.model_version,
  }
  return {
    id: crypto.randomUUID(),
    text: note,
    createdAt: now,
    updatedAt: now,
    type: original.type,
    target: original.target,
    amount: original.amount,
    original,
  }
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener)
  void load()
  return () => {
    listeners.delete(listener)
  }
}

const getSnapshot = (): GidiState => state

/** The on-device model: loads once per page; `classify` waits for it when called early. */
export function useGidi(): {
  status: GidiStatus
  progress: number | null
  loaded: number
  total: number | null
  error: string | null
  retry(): void
  classify(text: string): Promise<NoteRecord>
} {
  const current = useSyncExternalStore(subscribe, getSnapshot, getSnapshot)
  return useMemo(() => ({ ...current, retry, classify }), [current])
}
