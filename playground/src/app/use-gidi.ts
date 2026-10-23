import { useMemo, useSyncExternalStore } from "react"

import { captureError, track } from "@/lib/analytics"
import { predict, type PredictResponse, type Span } from "@/lib/api"
import {
  BROWSER_MODE,
  loadBrowserBackend,
  type BrowserBackend,
  type LoadProgress,
} from "@/lib/backend"
import { registerSensitive } from "@/lib/error-report"
import type { Lang } from "@/lib/language"
import { MODELS } from "@/lib/models"

import { amountToVnd, readEnglishValue, widenSlangValue } from "./amount"
import {
  currencyOf,
  getLanguage,
  getStrings,
  subscribe as onLanguageChange,
} from "./language"
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
  cause instanceof Error ? cause.message : getStrings().loadFailedGeneric

const copySpan = (span: Span | null | undefined): [number, number] | null =>
  span ? [span[0], span[1]] : null

const isTxType = (value: string): value is TxType =>
  (TX_TYPES as readonly string[]).includes(value)

/**
 * One model at a time, the one of the app language: module-level, so a StrictMode double mount
 * does not load twice. Switching language retires the session (aborts a running download, or
 * disposes the worker and its ORT session after the notes already queued) and starts the other.
 */
interface Session {
  lang: Lang
  abort: AbortController
  predictor: Promise<Predictor>
  backend: BrowserBackend | null
}

const idle = { progress: null, loaded: 0, total: null, error: null }
let state: GidiState = { status: "loading", ...idle }
let session: Session | null = null
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

function retire(old: Session | null): void {
  if (old === null) return
  old.abort.abort()
  void old.backend?.dispose()
}

/** Set once the visitor changed language in this page: later load errors are `switch` errors. */
let switched = false

async function loadPredictor(
  lang: Lang,
  abort: AbortController,
  isCurrent: () => boolean,
  keep: (backend: BrowserBackend) => void
): Promise<Predictor> {
  if (!BROWSER_MODE) {
    // Dev server: the model runs behind `/api/predict`, nothing to download.
    return predict
  }
  const backend = await loadBrowserBackend(
    MODELS[lang],
    (progress) => {
      if (isCurrent()) reportProgress(progress)
    },
    abort.signal,
    switched ? "switch" : "load"
  )
  if (!isCurrent()) {
    // Superseded while the last step ran.
    void backend.dispose()
    throw abort.signal.reason ?? new DOMException("Aborted", "AbortError")
  }
  keep(backend)
  return (text) => backend.predict(text)
}

function start(lang: Lang): Session {
  const abort = new AbortController()
  setState({ status: "loading", ...idle })
  const started = performance.now()
  const loaded = loadPredictor(
    lang,
    abort,
    () => session === current,
    (backend) => {
      current.backend = backend
    }
  )
  const current: Session = { lang, abort, predictor: loaded, backend: null }
  loaded.then(
    () => {
      if (session !== current) return
      track("model_loaded", {
        ms: Math.round(performance.now() - started),
        language: lang,
      })
      setState({ ...idle, status: "ready", progress: 1 })
    },
    (cause: unknown) => {
      // A superseded session is not a failure.
      if (session !== current) return
      const error = toMessage(cause)
      captureError(cause, switched ? "switch" : "load", {
        kind: "model_load_failed",
      })
      track("model_load_failed", {
        ms: Math.round(performance.now() - started),
        language: lang,
        error,
        error_name: cause instanceof Error ? cause.name : "unknown",
        loaded_bytes: state.loaded,
        total_bytes: state.total,
        ...environment(),
      })
      setState({ ...idle, status: "error", error })
    }
  )
  return current
}

function load(): Session {
  const lang = getLanguage()
  if (session === null || session.lang !== lang) {
    retire(session)
    session = start(lang)
  }
  return session
}

function retry(): void {
  if (state.status !== "error") return
  session = null
  load()
}

// The visitor switched language: load the other model (the old one is retired first).
onLanguageChange(() => {
  switched = true
  if (listeners.size > 0) load()
})

/**
 * Vietnamese: the rule-based parser's value, read as VND. English: the model's value span (v4)
 * read as US dollars; a model without a value head (v2) or an unreadable span falls back to a
 * deterministic scan of the note (`readEnglishValue`).
 */
function readValue(
  lang: Lang,
  note: string,
  result: PredictResponse["result"],
  hasValueHead: boolean
): {
  valueText: string | null
  span: [number, number] | null
  amount: number | null
} {
  if (lang === "en") {
    return readEnglishValue(note, {
      hasValueHead,
      valueText: result.value_text ?? null,
      span: copySpan(result.value_span),
    })
  }
  const { valueText, span } = widenSlangValue(
    note,
    result.value_text ?? null,
    copySpan(result.value_span)
  )
  return { valueText, span, amount: valueText ? amountToVnd(valueText) : null }
}

async function classify(text: string): Promise<NoteRecord> {
  const note = text.trim()
  registerSensitive(note)
  try {
    return await classifyNote(note)
  } catch (cause) {
    // Messages of these errors may quote the note, so only the error's name and place leave.
    captureError(cause, "classify", { kind: "classify_failed" })
    throw cause
  }
}

async function classifyNote(note: string): Promise<NoteRecord> {
  const t = getStrings()
  if (note === "") throw new Error(t.emptyNote)
  const current = load()
  const { result } = await (await current.predictor)(note)
  const {
    valueText,
    span: valueSpan,
    amount,
  } = readValue(
    current.lang,
    note,
    result,
    current.backend?.info.has_value_head ?? false
  )
  const type = isTxType(result.type) ? result.type : null
  if (type === null) throw new Error(t.badType(result.type))
  const now = Date.now()
  const original: ModelReading = {
    type,
    typeConfidence: result.type_confidence,
    target: result.target,
    targetSpan: copySpan(result.target_span),
    targetConfidence: result.target_confidence,
    valueText,
    valueSpan,
    amount,
    modelVersion: result.model_version,
  }
  registerSensitive(result.target, valueText)
  return {
    id: crypto.randomUUID(),
    text: note,
    createdAt: now,
    updatedAt: now,
    type: original.type,
    target: original.target,
    amount: original.amount,
    currency: currencyOf(current.lang),
    original,
  }
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener)
  load()
  return () => {
    listeners.delete(listener)
  }
}

const getSnapshot = (): GidiState => state

/** The on-device model: loads once per page and language; `classify` waits for it when called early. */
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
