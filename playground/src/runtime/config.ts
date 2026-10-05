/** `config.json` of a release bundle. Port of the checks in `gidi.inference.bundle.load_config`. */

export interface BundleConfig {
  modelVersion: string
  types: string[]
  maxLength: number
  inputNames: string[]
  outputNames: string[]
  architecture: string
}

const TAGS = ["O", "B-TARGET", "I-TARGET"]
const OUTPUTS = ["type_logits", "tag_logits"]
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
  // The runtime hard-wires the v3 contract: value from the rule parser, target snapped to words.
  if (
    config.value_source !== "rule-parser" ||
    config.value_parser?.name !== VALUE_PARSER
  ) {
    throw new ConfigError(
      "value_source must be the rule-based gidi.value_parser"
    )
  }
  if (config.target_snap !== "words")
    throw new ConfigError("target_snap must be 'words'")
  const onnx = config.onnx
  if (!sameList(onnx?.outputs, OUTPUTS) || !Array.isArray(onnx.inputs)) {
    throw new ConfigError(`onnx outputs must be ${OUTPUTS.join(", ")}`)
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
    architecture:
      arch?.layers && arch.hidden
        ? `${arch.layers}-layer encoder, ${arch.hidden} hidden`
        : String(arch?.kind ?? "unknown"),
  }
}
