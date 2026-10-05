export type Span = readonly [number, number]
export type SegmentRole = "target" | "value" | null

export type Segment = {
  text: string
  role: SegmentRole
  overlap?: boolean
}

/**
 * `Prediction.to_dict()` of the runtime. The `value_*` keys exist only for a bundle with a value;
 * `value_confidence` is `null` when the value comes from the rule-based parser.
 */
export type Prediction = {
  type: string
  type_confidence: number
  target: string | null
  target_span: Span | null
  target_confidence: number
  value_text?: string | null
  value_span?: Span | null
  value_confidence?: number | null
  truncated: boolean
  model_version: string
}

export type PredictResponse = {
  /** The result object exactly as the server sent it (shown as raw JSON). */
  raw: Record<string, unknown>
  result: Prediction
  latencyMs: number
  segments: Segment[]
  spansOverlap: boolean | null
}

export type RuntimeInfo = {
  model_version: string
  bundle: string
  max_length: number
  has_value_head: boolean
  architecture: string | null
  precision: string | null
  backend: string | null
  model_file: string | null
  onnx_opset: number | null
  bundle_path: string | null
  /** Browser mode only: SHA-256 the downloaded model was verified against. */
  model_sha256?: string | null
  /** Browser mode only: model download + verification + session creation, in ms. */
  load_ms?: number | null
}

export type ErrorKind =
  | "empty_input"
  | "unavailable"
  | "timeout"
  | "inference"
  | "missing_artifact"
  | "malformed"
  | "rejected"
  | "model_load"
  | "integrity"

export const REQUEST_TIMEOUT_MS = 10_000

export class PlaygroundError extends Error {
  kind: ErrorKind
  code: string | null
  status: number | null

  constructor(
    kind: ErrorKind,
    message: string,
    code: string | null = null,
    status: number | null = null
  ) {
    super(message)
    this.name = "PlaygroundError"
    this.kind = kind
    this.code = code
    this.status = status
  }
}

type Json = Record<string, unknown>

function isRecord(value: unknown): value is Json {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

const isNumber = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value)

const isOptionalString = (value: unknown) =>
  value === undefined || value === null || typeof value === "string"

function codePointLength(text: string): number {
  let count = 0
  for (const _ of text) count += 1 // eslint-disable-line @typescript-eslint/no-unused-vars
  return count
}

function malformed(detail: string): PlaygroundError {
  return new PlaygroundError("malformed", detail)
}

function readSpan(value: unknown, field: string, length: number): Span | null {
  if (value === null) return null
  if (
    !Array.isArray(value) ||
    value.length !== 2 ||
    !Number.isInteger(value[0]) ||
    !Number.isInteger(value[1]) ||
    value[0] < 0 ||
    value[0] > value[1] ||
    value[1] > length
  ) {
    throw malformed(
      `${field} must be null or [start, end) within the ${length} input characters`
    )
  }
  return [value[0], value[1]]
}

function readPrediction(raw: unknown, length: number): Prediction {
  if (!isRecord(raw)) throw malformed("result is not an object")
  const need = (ok: boolean, field: string, expected: string) => {
    if (!ok) throw malformed(`result.${field} must be ${expected}`)
  }
  need(typeof raw.type === "string", "type", "a string")
  need(isNumber(raw.type_confidence), "type_confidence", "a number")
  need(
    raw.target === null || typeof raw.target === "string",
    "target",
    "a string or null"
  )
  need(isNumber(raw.target_confidence), "target_confidence", "a number")
  need(typeof raw.truncated === "boolean", "truncated", "a boolean")
  need(typeof raw.model_version === "string", "model_version", "a string")
  const target_span = readSpan(raw.target_span, "result.target_span", length)
  const prediction: Prediction = {
    type: raw.type as string,
    type_confidence: raw.type_confidence as number,
    target: raw.target as string | null,
    target_span,
    target_confidence: raw.target_confidence as number,
    truncated: raw.truncated as boolean,
    model_version: raw.model_version as string,
  }
  const valueKeys = ["value_text", "value_span", "value_confidence"] as const
  if (valueKeys.some((key) => key in raw)) {
    need(
      raw.value_text === null || typeof raw.value_text === "string",
      "value_text",
      "a string or null"
    )
    need(
      raw.value_confidence === null || isNumber(raw.value_confidence),
      "value_confidence",
      "a number or null"
    )
    prediction.value_text = raw.value_text as string | null
    prediction.value_span = readSpan(
      raw.value_span,
      "result.value_span",
      length
    )
    prediction.value_confidence = raw.value_confidence as number | null
  }
  return prediction
}

function readSegments(raw: unknown, text: string): Segment[] {
  if (!Array.isArray(raw)) throw malformed("segments must be an array")
  const segments = raw.map((item, index): Segment => {
    if (
      !isRecord(item) ||
      typeof item.text !== "string" ||
      !(
        item.role === null ||
        item.role === "target" ||
        item.role === "value"
      ) ||
      !(item.overlap === undefined || typeof item.overlap === "boolean")
    ) {
      throw malformed(
        `segments[${index}] must be {text, role: null|"target"|"value"}`
      )
    }
    return {
      text: item.text,
      role: item.role,
      ...(item.overlap ? { overlap: true } : {}),
    }
  })
  if (segments.map((segment) => segment.text).join("") !== text) {
    throw malformed("segments do not add up to the submitted text")
  }
  return segments
}

/** Validate a decoded `/api/predict` success body against the submitted `text`. */
export function parsePredictResponse(
  body: unknown,
  text: string
): PredictResponse {
  if (!isRecord(body) || body.ok !== true)
    throw malformed("response is not an ok payload")
  if (!isNumber(body.latency_ms) || body.latency_ms < 0) {
    throw malformed("latency_ms must be a non-negative number")
  }
  if (
    body.spans_overlap !== undefined &&
    typeof body.spans_overlap !== "boolean"
  ) {
    throw malformed("spans_overlap must be a boolean")
  }
  const result = readPrediction(body.result, codePointLength(text))
  return {
    raw: body.result as Json,
    result,
    latencyMs: body.latency_ms,
    segments: readSegments(body.segments, text),
    spansOverlap:
      typeof body.spans_overlap === "boolean" ? body.spans_overlap : null,
  }
}

export function parseRuntimeInfo(body: unknown): RuntimeInfo {
  if (
    !isRecord(body) ||
    typeof body.model_version !== "string" ||
    typeof body.bundle !== "string" ||
    !isNumber(body.max_length) ||
    typeof body.has_value_head !== "boolean" ||
    !isOptionalString(body.architecture) ||
    !isOptionalString(body.precision) ||
    !isOptionalString(body.backend) ||
    !isOptionalString(body.model_file) ||
    !isOptionalString(body.bundle_path) ||
    !(
      body.onnx_opset === undefined ||
      body.onnx_opset === null ||
      isNumber(body.onnx_opset)
    )
  ) {
    throw malformed("/api/info has an unexpected shape")
  }
  return {
    model_version: body.model_version,
    bundle: body.bundle,
    max_length: body.max_length,
    has_value_head: body.has_value_head,
    architecture: (body.architecture as string | null | undefined) ?? null,
    precision: (body.precision as string | null | undefined) ?? null,
    backend: (body.backend as string | null | undefined) ?? null,
    model_file: (body.model_file as string | null | undefined) ?? null,
    onnx_opset: (body.onnx_opset as number | null | undefined) ?? null,
    bundle_path: (body.bundle_path as string | null | undefined) ?? null,
  }
}

function errorFromResponse(status: number, body: unknown): PlaygroundError {
  const error = isRecord(body) && isRecord(body.error) ? body.error : null
  const code = error && typeof error.code === "string" ? error.code : null
  const message =
    error && typeof error.message === "string"
      ? error.message
      : `HTTP ${status}`
  if (code === null) {
    return malformed(`HTTP ${status} response without an error payload`)
  }
  if (code === "empty_input")
    return new PlaygroundError("empty_input", message, code, status)
  if (code === "missing_artifact") {
    return new PlaygroundError("missing_artifact", message, code, status)
  }
  if (status >= 500)
    return new PlaygroundError("inference", message, code, status)
  return new PlaygroundError("rejected", message, code, status)
}

async function request(
  path: string,
  init: RequestInit,
  timeoutMs: number
): Promise<{ status: number; ok: boolean; body: unknown }> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const response = await fetch(path, { ...init, signal: controller.signal })
    // The timer keeps running while the body streams, so a stalled body also times out.
    const raw = await response.text()
    let body: unknown = null
    try {
      body = JSON.parse(raw)
    } catch {
      body = null
    }
    return { status: response.status, ok: response.ok, body }
  } catch (cause) {
    if (controller.signal.aborted) {
      throw new PlaygroundError(
        "timeout",
        `No response within ${Math.round(timeoutMs / 1000)} s.`
      )
    }
    throw new PlaygroundError(
      "unavailable",
      cause instanceof Error ? cause.message : "Network request failed."
    )
  } finally {
    clearTimeout(timer)
  }
}

export async function predict(
  text: string,
  timeoutMs: number = REQUEST_TIMEOUT_MS
): Promise<PredictResponse> {
  const { status, ok, body } = await request(
    "/api/predict",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    },
    timeoutMs
  )
  if (!ok) throw errorFromResponse(status, body)
  if (body === null) throw malformed("response body is not JSON")
  return parsePredictResponse(body, text)
}

export async function fetchInfo(
  timeoutMs: number = REQUEST_TIMEOUT_MS
): Promise<RuntimeInfo> {
  const { status, ok, body } = await request("/api/info", {}, timeoutMs)
  if (!ok) throw errorFromResponse(status, body)
  return parseRuntimeInfo(body)
}
