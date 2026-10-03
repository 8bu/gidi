/**
 * BIO decoding and confidences. Port of `gidi.inference.decode`.
 *
 * The first target span starts at a `B` (or an `I` after a non-span token), consecutive `I`
 * tokens extend it, a later `B` or an `O` ends it; bounds are the min start / max end of the
 * member tokens after trimming whitespace; tokens with empty trimmed offsets are skipped.
 */

import { isPySpace } from "./text.ts"

export const TAG_O = 0
export const TAG_B = 1
export const TAG_I = 2

const NUM_TAGS = 3

export interface DecodedSpan {
  start: number
  end: number
  /** Token indices that make up the span. */
  members: number[]
}

/** Shrink `[start, end)` so it neither starts nor ends on whitespace (may become empty). */
export function trimSpan(
  text: readonly number[],
  start: number,
  end: number
): [number, number] {
  while (start < end && isPySpace(text[start])) start++
  while (end > start && isPySpace(text[end - 1])) end--
  return [start, end]
}

export function decodeFirstSpan(
  offsets: readonly (readonly [number, number])[],
  tagIds: readonly number[],
  text: readonly number[]
): DecodedSpan | null {
  let start: number | null = null
  let end = 0
  const members: number[] = []
  for (let i = 0; i < Math.min(offsets.length, tagIds.length); i++) {
    const [s, e] = trimSpan(text, offsets[i][0], offsets[i][1])
    if (s >= e) continue
    const tag = tagIds[i]
    if (tag === TAG_B) {
      if (start !== null) break
      start = s
      end = e
      members.push(i)
    } else if (tag === TAG_I) {
      if (start === null) {
        start = s
        end = e
      } else {
        start = Math.min(start, s)
        end = Math.max(end, e)
      }
      members.push(i)
    } else if (start !== null) {
      break
    }
  }
  return start === null ? null : { start, end, members }
}

/** Row-wise log-softmax in float64 over `[rows, cols]` row-major logits. */
export function logSoftmax(
  logits: Float32Array | Float64Array,
  cols: number
): Float64Array {
  const out = new Float64Array(logits.length)
  for (let row = 0; row < logits.length / cols; row++) {
    const base = row * cols
    let max = -Infinity
    for (let c = 0; c < cols; c++) max = Math.max(max, logits[base + c])
    let sum = 0
    for (let c = 0; c < cols; c++) sum += Math.exp(logits[base + c] - max)
    const logSum = Math.log(sum)
    for (let c = 0; c < cols; c++)
      out[base + c] = logits[base + c] - max - logSum
  }
  return out
}

/** `[argmax class, softmax probability of it]`; the first maximum wins. */
export function typePrediction(typeLogits: Float32Array): [number, number] {
  const logP = logSoftmax(typeLogits, typeLogits.length)
  let index = 0
  for (let i = 1; i < typeLogits.length; i++) {
    if (typeLogits[i] > typeLogits[index]) index = i
  }
  return [index, Math.exp(logP[index])]
}

/** Per-token argmax over `[tokens, 3]` logits; the first maximum wins. */
export function argmaxTags(logits: Float32Array): number[] {
  const tags: number[] = []
  for (let row = 0; row < logits.length / NUM_TAGS; row++) {
    let best = 0
    for (let c = 1; c < NUM_TAGS; c++) {
      if (logits[row * NUM_TAGS + c] > logits[row * NUM_TAGS + best]) best = c
    }
    tags.push(best)
  }
  return tags
}

/**
 * Geometric mean of P(predicted tag) over the span's tokens. Without a span: the minimum P(O)
 * over the real (non-special) tokens, `1.0` if the note has no real token.
 */
export function targetConfidence(
  tagLogits: Float32Array,
  tagIds: readonly number[],
  span: DecodedSpan | null,
  real: readonly boolean[]
): number {
  const logP = logSoftmax(tagLogits, NUM_TAGS)
  if (span !== null) {
    let sum = 0
    for (const i of span.members) sum += logP[i * NUM_TAGS + tagIds[i]]
    return Math.exp(sum / span.members.length)
  }
  let min = Infinity
  real.forEach((isReal, i) => {
    if (isReal) min = Math.min(min, logP[i * NUM_TAGS + TAG_O])
  })
  return min === Infinity ? 1.0 : Math.exp(min)
}

/** Every BIO span in token order, by the same rule as `decodeFirstSpan`. */
export function decodeAllSpans(
  offsets: readonly (readonly [number, number])[],
  tagIds: readonly number[],
  text: readonly number[]
): DecodedSpan[] {
  const spans: DecodedSpan[] = []
  let current: DecodedSpan | null = null
  const flush = (): void => {
    if (current !== null) spans.push(current)
    current = null
  }
  for (let i = 0; i < Math.min(offsets.length, tagIds.length); i++) {
    const [s, e] = trimSpan(text, offsets[i][0], offsets[i][1])
    if (s >= e) continue
    const tag = tagIds[i]
    if (tag === TAG_B) {
      flush()
      current = { start: s, end: e, members: [i] }
    } else if (tag === TAG_I) {
      if (current === null) {
        current = { start: s, end: e, members: [i] }
      } else {
        current.start = Math.min(current.start, s)
        current.end = Math.max(current.end, e)
        current.members.push(i)
      }
    } else {
      flush()
    }
  }
  flush()
  return spans
}

/**
 * The most confident BIO span of `tagIds` and its confidence (ties go to the earlier span).
 * With no span the confidence is the minimum P(O) over the real tokens.
 */
export function bestSpanFromTags(
  logits: Float32Array,
  tagIds: readonly number[],
  offsets: readonly (readonly [number, number])[],
  text: readonly number[],
  real: readonly boolean[]
): [DecodedSpan | null, number] {
  const spans = decodeAllSpans(offsets, tagIds, text)
  if (spans.length === 0)
    return [null, targetConfidence(logits, tagIds, null, real)]
  let best = spans[0]
  let bestConfidence = targetConfidence(logits, tagIds, best, real)
  for (const span of spans.slice(1)) {
    const confidence = targetConfidence(logits, tagIds, span, real)
    if (confidence > bestConfidence) {
      best = span
      bestConfidence = confidence
    }
  }
  return [best, bestConfidence]
}
