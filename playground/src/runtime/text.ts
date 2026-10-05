/**
 * NFC normalization that remembers where every normalized character came from.
 * Port of `gidi.inference.text`; all offsets are Unicode code points, never UTF-16 units.
 *
 * The original string is cut into groups that normalize independently of their neighbours. A
 * group NFC leaves unchanged keeps a per-character map. A group NFC rewrites (composition or
 * reordering) is atomic: a span boundary inside it is rounded outward to the group's edges.
 */

import {
  ALNUM_RANGES,
  ALPHA_RANGES,
  COMBINING_RANGES,
  FOLD_PAIRS,
  PUNCT_RANGES,
  SPACE_RANGES,
  UPPER_RANGES,
} from "./unicode-tables.ts"

function inRanges(ranges: readonly number[], cp: number): boolean {
  let lo = 0
  let hi = ranges.length / 2 - 1
  while (lo <= hi) {
    const mid = (lo + hi) >> 1
    if (cp < ranges[2 * mid]) hi = mid - 1
    else if (cp > ranges[2 * mid + 1]) lo = mid + 1
    else return true
  }
  return false
}

/** `unicodedata.combining(ch) != 0` */
export function isCombining(cp: number): boolean {
  return inRanges(COMBINING_RANGES, cp)
}

/** Python `str.isspace` for one code point (differs from JS `\s`: U+001C..1F yes, U+FEFF no). */
export function isPySpace(cp: number): boolean {
  return inRanges(SPACE_RANGES, cp)
}

/** Python `str.isalpha` for one code point. */
export function isAlpha(cp: number): boolean {
  return inRanges(ALPHA_RANGES, cp)
}

/** Python `str.isalnum` for one code point. */
export function isAlnum(cp: number): boolean {
  return inRanges(ALNUM_RANGES, cp)
}

/** Python `str.isupper` for a one-character string. */
export function isUpper(cp: number): boolean {
  return inRanges(UPPER_RANGES, cp)
}

/** `unicodedata.category(ch).startswith("P")` */
export function isPunct(cp: number): boolean {
  return inRanges(PUNCT_RANGES, cp)
}

const FOLD = new Map<number, number>()
for (let i = 0; i < FOLD_PAIRS.length; i += 2)
  FOLD.set(FOLD_PAIRS[i], FOLD_PAIRS[i + 1])

/** `gidi.value_parser.fold` for one code point (see `fold_pairs` in the table generator). */
export function foldCodePoint(cp: number): number {
  return FOLD.get(cp) ?? cp
}

/** Code points of `text`; a lone surrogate is one element, like a Python `str`. */
export function toCodePoints(text: string): number[] {
  const out: number[] = []
  for (let i = 0; i < text.length; i++) {
    const unit = text.charCodeAt(i)
    if (unit >= 0xd800 && unit <= 0xdbff && i + 1 < text.length) {
      const next = text.charCodeAt(i + 1)
      if (next >= 0xdc00 && next <= 0xdfff) {
        out.push(((unit - 0xd800) << 10) + (next - 0xdc00) + 0x10000)
        i++
        continue
      }
    }
    out.push(unit)
  }
  return out
}

const CHUNK = 4096

/** `String.fromCodePoint` over `cps[start:end]` without blowing the argument limit. */
export function fromCodePoints(
  cps: readonly number[],
  start = 0,
  end = cps.length
): string {
  let out = ""
  for (let i = start; i < end; i += CHUNK) {
    out += String.fromCodePoint(...cps.slice(i, Math.min(i + CHUNK, end)))
  }
  return out
}

/** `text.strip() == ""` with Python's whitespace definition. */
export function isBlank(cps: readonly number[]): boolean {
  return cps.every(isPySpace)
}

export interface NormalizedText {
  /** The caller's string as code points. */
  original: number[]
  /** NFC of the original, as code points. */
  cps: number[]
  /** Length `cps.length + 1`; `null` when the original was already NFC (identity). */
  startMap: Int32Array | null
  endMap: Int32Array | null
}

export function startToOriginal(n: NormalizedText, index: number): number {
  return n.startMap === null ? index : n.startMap[index]
}

export function endToOriginal(n: NormalizedText, index: number): number {
  return n.endMap === null ? index : n.endMap[index]
}

function nfc(cps: readonly number[], start: number, end: number): string {
  return fromCodePoints(cps, start, end).normalize("NFC")
}

/** `[start, end)` groups whose NFC forms concatenate to `NFC(text)`. */
function groups(cps: readonly number[]): [number, number][] {
  const cuts: number[] = []
  for (let i = 0; i < cps.length; i++) {
    if (i === 0 || !isCombining(cps[i])) cuts.push(i)
  }
  cuts.push(cps.length)
  const out: [number, number][] = []
  let groupStart = 0
  let groupNfc = ""
  for (let k = 0; k + 1 < cuts.length; k++) {
    const a = cuts[k]
    const b = cuts[k + 1]
    const clusterNfc = nfc(cps, a, b)
    if (a === 0) {
      groupNfc = clusterNfc
      continue
    }
    const joined = nfc(cps, groupStart, b)
    if (joined === groupNfc + clusterNfc) {
      out.push([groupStart, a])
      groupStart = a
      groupNfc = clusterNfc
    } else {
      groupNfc = joined
    }
  }
  out.push([groupStart, cps.length])
  return out
}

/** NFC-normalize `text` and build the NFC -> original offset maps. */
export function normalizeNfc(text: string): NormalizedText {
  if (text.normalize("NFC") === text) {
    const cps = toCodePoints(text)
    return { original: cps, cps, startMap: null, endMap: null }
  }
  const cps = toCodePoints(text)
  const pieces: string[] = []
  const start: number[] = []
  const end: number[] = [0]
  for (const [a, b] of groups(cps)) {
    const group = fromCodePoints(cps, a, b)
    const groupNfc = group.normalize("NFC")
    pieces.push(groupNfc)
    const groupNfcLength = toCodePoints(groupNfc).length
    if (groupNfc === group) {
      for (let j = a; j < b; j++) {
        start.push(j)
        end.push(j + 1)
      }
    } else {
      for (let j = 0; j < groupNfcLength; j++) {
        start.push(a)
        end.push(b)
      }
    }
  }
  start.push(cps.length)
  const joined = pieces.join("")
  const whole = text.normalize("NFC")
  if (joined !== whole) {
    // The grouping assumption failed; map every position to the whole string rather than
    // return wrong offsets.
    const wholeCps = toCodePoints(whole)
    const n = wholeCps.length
    return {
      original: cps,
      cps: wholeCps,
      startMap: Int32Array.from({ length: n + 1 }, (_, i) =>
        i === n ? cps.length : 0
      ),
      endMap: Int32Array.from({ length: n + 1 }, (_, i) =>
        i === 0 ? 0 : cps.length
      ),
    }
  }
  return {
    original: cps,
    cps: toCodePoints(joined),
    startMap: Int32Array.from(start),
    endMap: Int32Array.from(end),
  }
}
