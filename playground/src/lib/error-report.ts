/**
 * What the app tells the owner about a failure, and what it never does. No PostHog import, so
 * `error-report.test.ts` runs it under node.
 *
 * Every error event carries safe context (build, language, model, manifest, connection, a stage
 * tag) and is scrubbed before it leaves: URLs lose their query and hash, and any note text, target
 * or amount the user typed (`registerSensitive`) is replaced. Messages of errors raised while
 * reading or saving a note are not sent at all, only the error's name and where it was thrown.
 * Noise is cut by an ignore list, a per-fingerprint dedupe, a cap per page load and a rate limit.
 */

/** Where in the app the failure happened. */
export type ErrorStage =
  | "boot" // page start, resource and chunk loads, React render
  | "load" // manifest, download, verification, worker, ORT session, Cache Storage
  | "switch" // the same, after the visitor changed language
  | "classify" // tokenizer, model run, decoding, word snap, amount parsing
  | "save" // IndexedDB open, migration, write, quota, import/export

export interface ErrorContext {
  build: string
  client_version: number | null
  language: string | null
  model: string | null
  manifest_schema: number | null
  manifest_active: string | null
}

const context: ErrorContext = {
  build: "dev",
  client_version: null,
  language: null,
  model: null,
  manifest_schema: null,
  manifest_active: null,
}

export function setErrorContext(patch: Partial<ErrorContext>): void {
  Object.assign(context, patch)
}

type ErrorSink = (
  error: unknown,
  stage: ErrorStage,
  extra: { kind?: string; recovery?: string }
) => void

let sink: ErrorSink | null = null

/** `analytics.ts` plugs the PostHog sender in; code that must stay free of it (stores) calls `reportError`. */
export function setErrorSink(next: ErrorSink | null): void {
  sink = next
}

/** Report a caught failure to the owner's error tracking, if analytics is on. Never throws. */
export function reportError(
  error: unknown,
  stage: ErrorStage,
  extra: { kind?: string; recovery?: string } = {}
): void {
  try {
    sink?.(error, stage, extra)
  } catch {
    // Reporting must never break the app.
  }
}

/** Free-text the user typed: never allowed in an event. Kept in memory for the page load only. */
const sensitive = new Set<string>()
const MAX_SENSITIVE = 200

export function registerSensitive(
  ...values: (string | null | undefined)[]
): void {
  for (const value of values) {
    const text = value?.trim()
    if (text === undefined || text.length < 3) continue
    if (sensitive.size >= MAX_SENSITIVE)
      sensitive.delete(sensitive.values().next().value as string)
    sensitive.add(text)
  }
}

const URL_RE = /\b(?:https?|blob|wss?):\/\/[^\s"'<>)]+/gi

/** Drop query and hash from every URL in `text`, then mask user text. */
export function redact(text: string): string {
  let out = text.replace(URL_RE, (url) => url.split(/[?#]/, 1)[0])
  for (const value of sensitive) out = out.split(value).join("[note]")
  return out
}

export function stripUrl(url: string): string {
  return url.split(/[?#]/, 1)[0]
}

/**
 * True for a URL on another origin (the Cloudflare Web Analytics beacon, extension assets).
 * Ad blockers and privacy modes block those; the app cannot fix or act on it, so a failed
 * load is not an app error.
 */
export function isThirdPartyUrl(url: string, origin: string): boolean {
  try {
    return new URL(url, origin).origin !== origin
  } catch {
    return false
  }
}

/** Stages whose messages may quote the note: only the error name and the stack leave. */
const QUIET_STAGES: ReadonlySet<string> = new Set<ErrorStage>([
  "classify",
  "save",
])

const BENIGN_MESSAGES = [
  /ResizeObserver loop/i,
  /^Script error\.?$/i,
  /Non-Error promise rejection captured/i,
]
const EXTENSION_FRAME = /(?:chrome|moz|safari|safari-web)-extension:\/\//i
const NETWORK_NOISE =
  /failed to fetch|load failed|networkerror|network request failed/i

export interface FrameLike {
  filename?: string
}
export interface ExceptionLike {
  type?: string
  value?: string
  stacktrace?: { frames?: FrameLike[] }
}

/** Errors that are noise: aborts, browser quirks, extensions, and fetch failures while offline. */
export function isBenign(
  exception: ExceptionLike,
  online: boolean = typeof navigator === "undefined" ? true : navigator.onLine
): boolean {
  const message = exception.value ?? ""
  if (exception.type === "AbortError" || /\baborted\b/i.test(message))
    return true
  if (BENIGN_MESSAGES.some((re) => re.test(message))) return true
  if (
    exception.stacktrace?.frames?.some((f) =>
      EXTENSION_FRAME.test(f.filename ?? "")
    )
  )
    return true
  if (!online && NETWORK_NOISE.test(message)) return true
  return false
}

/** Same bug, same fingerprint: type, message with digits and ids folded, top frame. */
export function fingerprint(exception: ExceptionLike): string {
  const frame = exception.stacktrace?.frames?.at(-1) as
    { filename?: string; function?: string; lineno?: number } | undefined
  const message = (exception.value ?? "")
    .replace(/[0-9a-f]{8,}|\d+/gi, "#")
    .slice(0, 120)
  return [
    exception.type ?? "Error",
    message,
    stripUrl(frame?.filename ?? ""),
    frame?.function ?? "",
    frame?.lineno ?? "",
  ].join("|")
}

export const ERROR_CAP_PER_PAGE = 10
/** At most this many error events back to back; then one per `REFILL_MS`. */
const BURST = 3
const REFILL_MS = 5_000

export interface ErrorGate {
  /** Whether an error with this fingerprint may be sent now (and counts it if so). */
  admit(print: string): boolean
}

export function createErrorGate(
  now: () => number = () => Date.now()
): ErrorGate {
  const seen = new Set<string>()
  let sent = 0
  let tokens = BURST
  let last = now()
  return {
    admit(print) {
      if (seen.has(print) || sent >= ERROR_CAP_PER_PAGE) return false
      const t = now()
      tokens = Math.min(BURST, tokens + (t - last) / REFILL_MS)
      last = t
      if (tokens < 1) return false
      tokens -= 1
      seen.add(print)
      sent += 1
      return true
    },
  }
}

export function connectionInfo(): Record<string, unknown> {
  if (typeof navigator === "undefined") return {}
  const connection = (
    navigator as Navigator & {
      connection?: { effectiveType?: string; saveData?: boolean }
    }
  ).connection
  return {
    online: navigator.onLine,
    effective_type: connection?.effectiveType ?? null,
    save_data: connection?.saveData ?? null,
  }
}

/** Properties added to every error event (and read back by the scrubber). */
export function errorProperties(
  stage: ErrorStage,
  extra: { kind?: string; recovery?: string } = {}
): Record<string, unknown> {
  return {
    stage,
    kind: extra.kind ?? null,
    recovery: extra.recovery ?? null,
    app_build: context.build,
    client_version: context.client_version,
    language: context.language,
    model: context.model,
    manifest_schema: context.manifest_schema,
    manifest_active: context.manifest_active,
    ...connectionInfo(),
  }
}

interface ExceptionEvent {
  event: string
  properties: Record<string, unknown> & {
    $exception_list?: (ExceptionLike & {
      stacktrace?: { frames?: (FrameLike & Record<string, unknown>)[] }
    })[]
    stage?: string
  }
}

/**
 * The `before_send` step for `$exception` events, whether PostHog made them from `window.onerror`
 * / unhandled rejections or the app asked (`captureError`). Returns `null` to drop the event.
 * Events without a stage (the SDK's own) are tagged `boot`.
 */
export function scrubException<E extends ExceptionEvent>(
  event: E,
  gate: ErrorGate
): E | null {
  const props = event.properties
  const list = props.$exception_list ?? []
  if (list.length === 0) return null
  if (list.some((exception) => isBenign(exception))) return null
  if (!gate.admit(fingerprint(list[0]))) return null
  const stage = typeof props.stage === "string" ? props.stage : "boot"
  for (const [key, value] of Object.entries(
    errorProperties(stage as ErrorStage)
  )) {
    props[key] ??= value
  }
  props.stage = stage
  for (const exception of list) {
    exception.value = QUIET_STAGES.has(stage)
      ? `[message omitted: ${stage}]`
      : redact(exception.value ?? "").slice(0, 300)
    for (const frame of exception.stacktrace?.frames ?? []) {
      if (typeof frame.filename === "string")
        frame.filename = stripUrl(frame.filename)
    }
  }
  return event
}

/** Strip query and hash from the page URLs every event carries. */
export function scrubPageUrls(properties: Record<string, unknown>): void {
  for (const key of [
    "$current_url",
    "$referrer",
    "$initial_current_url",
    "$initial_referrer",
  ]) {
    const value = properties[key]
    if (typeof value === "string") properties[key] = stripUrl(value)
  }
}
