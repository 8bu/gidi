/** Public contract of the in-browser runtime. Mirrors `gidi.inference` (Python). */

export interface PredictorInfo {
  model_version: string
  release_version: string
  max_length: number
  architecture: string
  precision: "int8"
  backend: string
}

export interface LoadOptions {
  /** Directory holding `config.json`, `tokenizer.json` and `web-release.json`. */
  baseUrl: string
  modelUrl: string
  modelSha256: string
  onProgress?: (loaded: number, total: number | null) => void
  signal?: AbortSignal
}

/** Exactly `gidi.inference.Prediction.to_dict()` of a v3 bundle (rule-parser value). */
export interface Prediction {
  type: string
  type_confidence: number
  target: string | null
  /** `[start, end)` in Unicode code points of the caller's original string. */
  target_span: [number, number] | null
  target_confidence: number
  value_text: string | null
  value_span: [number, number] | null
  /** Always `null`: the rule-based value parser has no confidence. */
  value_confidence: null
  truncated: boolean
  model_version: string
}

export interface WebPredictor {
  info: PredictorInfo
  predict(text: string): Promise<Prediction>
  dispose(): Promise<void>
}

/** The text is empty or whitespace-only; there is nothing to classify. */
export class EmptyInputError extends Error {
  constructor(message = "text is empty or whitespace-only") {
    super(message)
    this.name = "EmptyInputError"
  }
}

/** The downloaded model bytes do not match the expected sha256. Never recoverable. */
export class ModelIntegrityError extends Error {
  constructor(message: string) {
    super(message)
    this.name = "ModelIntegrityError"
  }
}
