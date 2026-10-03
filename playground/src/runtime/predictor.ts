/**
 * Text in, `Prediction` out. Port of `gidi.inference.predictor` / `schema.Prediction.to_dict`.
 * Framework-free and independent of how the model is executed (`ModelRunner`).
 *
 * The pipeline is split into stages so the parity gate can compare every intermediate:
 * `tokenize` (NFC + tokens) -> `ModelRunner.run` (logits) -> `decodeLogits` (tags, spans).
 */

import type { BundleConfig } from "./config.ts"
import { viterbiMasked } from "./crf.ts"
import {
  argmaxTags,
  bestSpanFromTags,
  decodeFirstSpan,
  type DecodedSpan,
  TAG_O,
  targetConfidence,
  typePrediction,
} from "./decode.ts"
import {
  endToOriginal,
  fromCodePoints,
  isBlank,
  type NormalizedText,
  normalizeNfc,
  startToOriginal,
} from "./text.ts"
import type { BundleTokenizer, TokenizedText } from "./tokenizer.ts"
import { EmptyInputError, type Prediction, type WebPredictor } from "./types.ts"

export interface ModelOutputs {
  typeLogits: Float32Array // [num_types]
  tagLogits: Float32Array // [tokens, 3]
  valueLogits: Float32Array // [tokens, 3]
}

export interface ModelRunner {
  readonly backend: string
  run(
    inputIds: readonly number[],
    attentionMask: readonly number[]
  ): Promise<ModelOutputs>
  dispose(): Promise<void>
}

export interface PredictorParts {
  config: BundleConfig
  tokenizer: BundleTokenizer
  runner: ModelRunner
  releaseVersion: string
}

export interface Tokenization {
  normalized: NormalizedText
  tokens: TokenizedText
  attentionMask: number[]
}

/** Every stage result of one note; `prediction` is the public output. */
export interface Analysis extends Tokenization {
  logits: ModelOutputs
  targetTags: number[]
  valueTags: number[]
  /** Per-token `[start, end)` in the caller's string (`[0, 0]` for specials). */
  originalOffsets: [number, number][]
  prediction: Prediction
}

/** `[text sliced from the caller's string, span in its code points]` of a decoded span. */
function sliceOriginal(
  normalized: NormalizedText,
  span: DecodedSpan | null
): [string | null, [number, number] | null] {
  if (span === null) return [null, null]
  const start = startToOriginal(normalized, span.start)
  const end = endToOriginal(normalized, span.end)
  return [fromCodePoints(normalized.original, start, end), [start, end]]
}

/** Validate, NFC-normalize (with offset maps) and tokenize. */
export function tokenize(
  tokenizer: BundleTokenizer,
  text: string
): Tokenization {
  if (typeof text !== "string")
    throw new TypeError(`text must be string, not ${typeof text}`)
  const normalized = normalizeNfc(text)
  if (isBlank(normalized.original)) throw new EmptyInputError()
  const tokens = tokenizer.encode(normalized.cps)
  return { normalized, tokens, attentionMask: tokens.ids.map(() => 1) }
}

/** Everything after the model: argmax / Viterbi tags, spans, confidences, original offsets. */
export function decodeLogits(
  config: BundleConfig,
  stage: Tokenization,
  logits: ModelOutputs
): Analysis {
  const { normalized, tokens } = stage
  const { typeLogits, tagLogits, valueLogits } = logits

  const [typeIndex, typeConfidence] = typePrediction(typeLogits)
  const real = tokens.specialTokensMask.map((special) => special === 0)
  const targetTags = argmaxTags(tagLogits)
  const targetDecoded = decodeFirstSpan(
    tokens.offsets,
    targetTags,
    normalized.cps
  )
  const targetConf = targetConfidence(
    tagLogits,
    targetTags,
    targetDecoded,
    real
  )
  const [target, targetSpan] = sliceOriginal(normalized, targetDecoded)

  const valueTags = viterbiMasked(
    Float64Array.from(valueLogits),
    real,
    config.crf,
    TAG_O
  )
  const [valueDecoded, valueConf] = bestSpanFromTags(
    valueLogits,
    valueTags,
    tokens.offsets,
    normalized.cps,
    real
  )
  const [valueText, valueSpan] = sliceOriginal(normalized, valueDecoded)

  const originalOffsets = tokens.offsets.map(([s, e], i): [number, number] =>
    tokens.specialTokensMask[i] === 1
      ? [0, 0]
      : [startToOriginal(normalized, s), endToOriginal(normalized, e)]
  )
  return {
    ...stage,
    logits,
    targetTags,
    valueTags,
    originalOffsets,
    prediction: {
      type: config.types[typeIndex],
      type_confidence: typeConfidence,
      target,
      target_span: targetSpan,
      target_confidence: targetConf,
      value_text: valueText,
      value_span: valueSpan,
      value_confidence: valueConf,
      truncated: tokens.truncated,
      model_version: config.modelVersion,
    },
  }
}

export async function analyze(
  parts: PredictorParts,
  text: string
): Promise<Analysis> {
  const stage = tokenize(parts.tokenizer, text)
  const logits = await parts.runner.run(stage.tokens.ids, stage.attentionMask)
  return decodeLogits(parts.config, stage, logits)
}

export function createPredictor(parts: PredictorParts): WebPredictor {
  const { config, runner } = parts
  // Keep results in call order; one ORT session is driven by one call at a time.
  let queue: Promise<unknown> = Promise.resolve()

  return {
    info: {
      model_version: config.modelVersion,
      release_version: parts.releaseVersion,
      max_length: config.maxLength,
      architecture: config.architecture,
      precision: "int8",
      backend: runner.backend,
    },
    predict(text) {
      const result = queue.then(
        async () => (await analyze(parts, text)).prediction
      )
      queue = result.catch(() => undefined)
      return result
    },
    async dispose() {
      await queue
      await runner.dispose()
    },
  }
}
