/** `config.json` of a release bundle. Port of the checks in `gidi.inference.bundle.load_config`. */

import type { CrfTransitions } from "./crf.ts"

export interface BundleConfig {
  modelVersion: string
  types: string[]
  maxLength: number
  inputNames: string[]
  outputNames: string[]
  architecture: string
  crf: CrfTransitions
}

const VALUE_LABELS = ["O", "B-VALUE", "I-VALUE"]
const TAGS = ["O", "B-TARGET", "I-TARGET"]
const NUM_TAGS = 3

class ConfigError extends Error {
  constructor(message: string) {
    super(`invalid config.json: ${message}`)
    this.name = "ConfigError"
  }
}

function finiteNumbers(values: unknown, count: number, name: string): number[] {
  const ok =
    Array.isArray(values) &&
    values.length === count &&
    values.every((v) => typeof v === "number" && Number.isFinite(v))
  if (!ok)
    throw new ConfigError(
      `value_decoding.${name} must be ${count} finite numbers`
    )
  return values as number[]
}

function sameList(a: unknown, b: readonly string[]): boolean {
  return (
    Array.isArray(a) && a.length === b.length && a.every((x, i) => x === b[i])
  )
}

/** Validate the parsed JSON and extract what the runtime needs. */
export function parseConfig(raw: unknown): BundleConfig {
  const config = raw as Record<string, any> // eslint-disable-line @typescript-eslint/no-explicit-any
  if (!sameList(config?.tags, TAGS))
    throw new ConfigError("unsupported tag set")
  if (!sameList(config.value_labels, VALUE_LABELS))
    throw new ConfigError("value head required")
  const onnx = config.onnx
  if (!Array.isArray(onnx?.inputs) || !Array.isArray(onnx?.outputs)) {
    throw new ConfigError("missing onnx interface")
  }
  if (onnx.outputs.length !== 3 || onnx.outputs[2] !== "value_logits") {
    throw new ConfigError(
      "value head must be the third ONNX output 'value_logits'"
    )
  }
  const maxLength = Number(config.max_length)
  if (!Number.isInteger(maxLength) || maxLength < 3)
    throw new ConfigError("bad max_length")

  const decoding = config.value_decoding
  if (
    decoding?.method !== "crf_viterbi" ||
    decoding.span_selection !== "highest_confidence"
  ) {
    throw new ConfigError("unsupported value_decoding")
  }
  if (
    !Array.isArray(decoding.transitions) ||
    decoding.transitions.length !== NUM_TAGS
  ) {
    throw new ConfigError("value_decoding.transitions must be 3x3")
  }
  const rows = (decoding.transitions as unknown[]).map((row) =>
    finiteNumbers(row, NUM_TAGS, "transitions")
  )
  const arch = config.architecture
  return {
    modelVersion: String(config.model_version),
    types: config.types as string[],
    maxLength,
    inputNames: onnx.inputs as string[],
    outputNames: onnx.outputs as string[],
    architecture: arch?.path_a
      ? `${arch.kind} (2 x ${arch.path_a.layers}-layer encoders, ${arch.path_a.hidden} hidden)`
      : String(arch?.kind ?? "unknown"),
    crf: {
      numTags: NUM_TAGS,
      start: Float64Array.from(
        finiteNumbers(decoding.start_transitions, NUM_TAGS, "start_transitions")
      ),
      end: Float64Array.from(
        finiteNumbers(decoding.end_transitions, NUM_TAGS, "end_transitions")
      ),
      transitions: Float64Array.from(rows.flat()),
    },
  }
}
