/** `config.json` of a release bundle. Port of the checks in `gidi.inference.bundle.load_config`. */

import type { TokenizerType } from "./tokenizer.ts"

export interface BundleConfig {
  modelVersion: string
  types: string[]
  maxLength: number
  inputNames: string[]
  outputNames: string[]
  architecture: string
  /**
   * `rule-parser`: the value comes from `gidi.value_parser`; `head`: from the model's
   * `value_logits` (per-token argmax, highest-confidence span, whole-word snap); `none`: the
   * bundle has no value.
   */
  valueSource: "rule-parser" | "head" | "none"
  tokenizerType: TokenizerType
}

const TAGS = ["O", "B-TARGET", "I-TARGET"]
const OUTPUTS = ["type_logits", "tag_logits"]
const VALUE_OUTPUTS = [...OUTPUTS, "value_logits"]
const VALUE_LABELS = ["O", "B-VALUE", "I-VALUE"]
const VALUE_PARSER = "gidi.value_parser"

class ConfigError extends Error {
  constructor(message: string) {
    super(`invalid config.json: ${message}`)
    this.name = "ConfigError"
  }
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
  // The runtime hard-wires three contracts: the v3 one (value from the rule parser), the English
  // v2 one (no value at all: neither `value_source`, `value_parser` nor `value_labels`) and the
  // English v4 one (a value head decoded by argmax, spans snapped to words). Target snapped to
  // words in all of them.
  const hasHead = config.value_labels !== undefined
  const valueSource: BundleConfig["valueSource"] = hasHead
    ? "head"
    : config.value_source === undefined
      ? "none"
      : "rule-parser"
  if (valueSource === "rule-parser") {
    if (
      config.value_source !== "rule-parser" ||
      config.value_parser?.name !== VALUE_PARSER
    )
      throw new ConfigError(
        "value_source must be the rule-based gidi.value_parser"
      )
  } else if (valueSource === "head") {
    if (
      config.value_source !== undefined ||
      config.value_parser !== undefined ||
      !sameList(config.value_labels, VALUE_LABELS) ||
      config.value_decoding?.method !== "argmax" ||
      config.value_decoding?.span_selection !== "highest_confidence" ||
      config.value_snap !== "words"
    )
      throw new ConfigError(
        "a value head must declare labels O/B-VALUE/I-VALUE, value_decoding argmax and value_snap words"
      )
  } else if (
    config.value_parser !== undefined ||
    config.value_snap !== undefined
  ) {
    throw new ConfigError("value_parser / value_snap need a value")
  }
  if (config.target_snap !== "words")
    throw new ConfigError("target_snap must be 'words'")
  const tokenizerType = config.tokenizer_type ?? "bpe"
  if (tokenizerType !== "bpe" && tokenizerType !== "wordpiece")
    throw new ConfigError("tokenizer_type must be 'bpe' or 'wordpiece'")
  const onnx = config.onnx
  const outputs = valueSource === "head" ? VALUE_OUTPUTS : OUTPUTS
  if (!sameList(onnx?.outputs, outputs) || !Array.isArray(onnx.inputs)) {
    throw new ConfigError(`onnx outputs must be ${outputs.join(", ")}`)
  }
  const maxLength = Number(config.max_length)
  if (!Number.isInteger(maxLength) || maxLength < 3)
    throw new ConfigError("bad max_length")

  const arch = config.architecture
  return {
    modelVersion: String(config.model_version),
    types: config.types as string[],
    maxLength,
    inputNames: onnx.inputs as string[],
    outputNames: onnx.outputs as string[],
    valueSource,
    tokenizerType,
    architecture:
      arch?.layers && arch.hidden
        ? `${arch.layers}-layer encoder, ${arch.hidden} hidden`
        : String(arch?.kind ?? "unknown"),
  }
}
