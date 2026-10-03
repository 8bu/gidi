/**
 * Linear-chain CRF Viterbi decoding. Port of `gidi.inference.crf`.
 *
 * `transitions[i * K + j]` scores tag `i` followed by tag `j`. Scores accumulate in float64 in
 * the same order as numpy (`(score + transition) + emission`); ties go to the lowest tag index.
 */

export interface CrfTransitions {
  numTags: number
  start: Float64Array
  end: Float64Array
  transitions: Float64Array
}

/** Highest-scoring tag sequence for `emissions` (`[T, K]` row-major); `[]` when `T == 0`. */
export function viterbi(
  emissions: Float64Array,
  length: number,
  crf: CrfTransitions
): number[] {
  if (length === 0) return []
  const k = crf.numTags
  let score = Float64Array.from(
    { length: k },
    (_, tag) => crf.start[tag] + emissions[tag]
  )
  const back: Int32Array[] = []
  for (let t = 1; t < length; t++) {
    const next = new Float64Array(k)
    const pointers = new Int32Array(k)
    for (let cur = 0; cur < k; cur++) {
      let best = -Infinity
      let bestPrev = 0
      for (let prev = 0; prev < k; prev++) {
        const total =
          score[prev] + crf.transitions[prev * k + cur] + emissions[t * k + cur]
        if (total > best) {
          best = total
          bestPrev = prev
        }
      }
      next[cur] = best
      pointers[cur] = bestPrev
    }
    back.push(pointers)
    score = next
  }
  let best = -Infinity
  let tag = 0
  for (let cur = 0; cur < k; cur++) {
    const total = score[cur] + crf.end[cur]
    if (total > best) {
      best = total
      tag = cur
    }
  }
  const path = [tag]
  for (let t = back.length - 1; t >= 0; t--) {
    tag = back[t][tag]
    path.push(tag)
  }
  return path.reverse()
}

/** Viterbi over the `real` positions only; every other position gets `fill`. */
export function viterbiMasked(
  emissions: Float64Array,
  real: readonly boolean[],
  crf: CrfTransitions,
  fill = 0
): number[] {
  const k = crf.numTags
  const idx = real.flatMap((r, i) => (r ? [i] : []))
  const tags: number[] = real.map(() => fill)
  if (idx.length > 0) {
    const picked = new Float64Array(idx.length * k)
    idx.forEach((position, row) =>
      picked.set(emissions.subarray(position * k, (position + 1) * k), row * k)
    )
    viterbi(picked, idx.length, crf).forEach((tag, row) => {
      tags[idx[row]] = tag
    })
  }
  return tags
}
