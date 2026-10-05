/**
 * Deterministic value-span parser: the monetary amount of a short Vietnamese note, as a span.
 * Port of `gidi.value_parser.parser`; offsets are Unicode code points of the caller's string.
 *
 * `parseValue(text)` returns the substring of `text` that is the amount (`1 triệu`, `1tr5`,
 * `80.000đ`, `70`) or `null` when the note holds no amount. No number is ever derived from it.
 *
 * One linear pass over NFC code points and a *folded* shadow of the same length (lowercase,
 * accents removed, `đ` -> `d`). Word lists match on the shadow, spans are cut from the NFC text.
 * A candidate is `number [unit [half | digits]] [currency]`; one candidate is chosen by tier
 * (explicit money unit or money-shaped number, then slang unit, then bare number).
 */

import {
  BILLION,
  CONTINUABLE,
  CURRENCY_SIGN,
  CURRENCY_WORDS,
  HALF,
  HALF_PREFIX,
  inList,
  MILLION,
  NAME_LIKE,
  NUMBER_WORDS,
  PERIOD_BEFORE,
  QUANTITY_AFTER,
  SLANG,
  SLANG_QTY,
  SLANG_QTY_ACCENTED,
  THOUSAND_LETTER,
  THOUSAND_WORDS,
  WEEKDAY_WORD,
  WORD_UNITS,
} from "./lexicon.ts"
import {
  endToOriginal,
  foldCodePoint,
  fromCodePoints,
  isAlnum,
  isAlpha,
  isUpper,
  type NormalizedText,
  normalizeNfc,
  startToOriginal,
} from "./text.ts"

const TIER_MONEY = 1 // explicit unit (k, tr, nghìn, ...), currency suffix, thousand-grouped, >= 5 digits
const TIER_SLANG = 2 // củ, xị, chai, lít
const TIER_BARE = 3 // a bare number that no context explains away

const MAX_PLAIN_DIGITS = 10 // a plain digit run longer than this is an account / id, not an amount
const BARE_PLAIN_MONEY_DIGITS = 5 // a plain digit run of at least this many digits is money-shaped

const SPACE = 0x20
const DOT = 0x2e
const COMMA = 0x2c
const SLASH = 0x2f
const PERCENT = 0x25
const DASH = 0x2d

export interface ValueSpan {
  text: string
  /** `[start, end)` in code points of the caller's string. */
  start: number
  end: number
}

/** A money expression in NFC coordinates. */
interface Candidate {
  start: number
  end: number
  tier: number
}

type Chars = readonly number[]
type Range = [number, number]
type Resume = [Candidate | null, number]

/** Accent-folded lowercase shadow of NFC `text` with the same length. */
export function fold(text: Chars): number[] {
  return text.map(foldCodePoint)
}

const isDigit = (c: number): boolean => c >= 0x30 && c <= 0x39
const isLetter = isAlpha
const sub = (chars: Chars, start: number, end: number): string =>
  fromCodePoints(chars, start, end)
/** Python `str.lower()` of `chars[start:end]`. */
const lowerSub = (chars: Chars, start: number, end: number): string =>
  sub(chars, start, end).toLowerCase()
const oneOf = (c: number, set: string): boolean =>
  set.includes(String.fromCodePoint(c))

// --------------------------------------------------------------------------- scanning helpers

/** End of the letter run starting at `pos` (`pos` when there is none). */
function wordEnd(shadow: Chars, pos: number): number {
  let end = pos
  while (end < shadow.length && isLetter(shadow[end])) end++
  return end
}

/** `[start, end]` of the word ending one space before `pos`; `[pos, pos]` if none. */
function prevWord(shadow: Chars, pos: number): Range {
  if (pos < 2 || shadow[pos - 1] !== SPACE) return [pos, pos]
  const end = pos - 1
  let start = end
  while (start > 0 && isLetter(shadow[start - 1])) start--
  return start < end ? [start, end] : [pos, pos]
}

/** `[start, end]` of the word one space after `pos`; `[pos, pos]` if none. */
function nextWord(shadow: Chars, pos: number): Range {
  if (pos >= shadow.length || shadow[pos] !== SPACE) return [pos, pos]
  const start = pos + 1
  const end = wordEnd(shadow, start)
  return end > start ? [start, end] : [pos, pos]
}

function digitRun(shadow: Chars, pos: number): number {
  let end = pos
  while (end < shadow.length && isDigit(shadow[end])) end++
  return end
}

/**
 * End of the number starting at `pos` and the lengths of its separated digit groups. Groups are
 * joined by `.`/`,` (`80.000`, `1,5`) or, for at least two trailing three-digit groups, by
 * single spaces (`1 000 000`).
 */
function numberEnd(shadow: Chars, pos: number): [number, number[]] {
  const n = shadow.length
  let end = digitRun(shadow, pos)
  const groups = [end - pos]
  while (
    end + 1 < n &&
    (shadow[end] === DOT || shadow[end] === COMMA) &&
    isDigit(shadow[end + 1])
  ) {
    const next = digitRun(shadow, end + 1)
    groups.push(next - end - 1)
    end = next
  }
  if (groups.length === 1 && groups[0] <= 3) {
    let cursor = end
    let spaced = 0
    while (
      cursor + 1 < n &&
      shadow[cursor] === SPACE &&
      isDigit(shadow[cursor + 1])
    ) {
      const next = digitRun(shadow, cursor + 1)
      if (next - cursor - 1 !== 3) break
      spaced++
      cursor = next
    }
    if (spaced >= 2)
      return [cursor, [groups[0], ...Array<number>(spaced).fill(3)]]
  }
  return [end, groups]
}

/** Thousand-grouped: `80.000`, `1.250.000`, `12,500,000` (first group 1-3 digits). */
function isGrouped(groups: number[]): boolean {
  return (
    groups.length > 1 &&
    groups[0] >= 1 &&
    groups[0] <= 3 &&
    groups.slice(1).every((g) => g === 3)
  )
}

/** End of an alphanumeric token starting at `pos` (digits glued to letters). */
function skipIdentifier(shadow: Chars, pos: number): number {
  let end = pos
  while (end < shadow.length && (isLetter(shadow[end]) || isDigit(shadow[end])))
    end++
  return Math.max(end, pos + 1)
}

/**
 * End of a phone-like number at `pos` (9-11 digits, leading 0 or 84), longest match, or `null`.
 * Digit groups of >= 3 digits may be joined by one space, `.` or `-` (`0912 345 678`).
 */
function phoneEnd(shadow: Chars, pos: number): number | null {
  const n = shadow.length
  const groups: [string, number][] = []
  let end = pos
  for (;;) {
    const run = digitRun(shadow, end)
    if (groups.length > 0 && run - end < 3) break
    groups.push([sub(shadow, end, run), run])
    end = run
    if (groups.reduce((sum, [digits]) => sum + digits.length, 0) > 11) break
    if (
      end + 1 < n &&
      (shadow[end] === SPACE || shadow[end] === DOT || shadow[end] === DASH) &&
      isDigit(shadow[end + 1])
    ) {
      end++
      continue
    }
    break
  }
  let best: number | null = null
  let joined = ""
  for (const [digits, groupEnd] of groups) {
    joined += digits
    if (
      joined.length >= 9 &&
      joined.length <= 11 &&
      (joined.startsWith("0") ||
        (joined.startsWith("84") && joined.length >= 10))
    ) {
      best = groupEnd
    }
  }
  return best
}

// --------------------------------------------------------------------------- context rules

/** Number of characters of `shadow[start:end]` that are neither `.` nor `,`. */
function digitCount(shadow: Chars, start: number, end: number): number {
  let count = 0
  for (let i = start; i < end; i++)
    if (shadow[i] !== DOT && shadow[i] !== COMMA) count++
  return count
}

/** Is the number at `[start, end)` a month / period / weekday / year by its left word? */
function explainedBefore(
  text: Chars,
  shadow: Chars,
  start: number,
  end: number,
  groups: number[]
): boolean {
  const [ws, we] = prevWord(shadow, start)
  if (ws === we) return false
  const word = sub(shadow, ws, we)
  if (!inList(PERIOD_BEFORE, word)) return false
  if (inList(NAME_LIKE, word) && isUpper(text[ws])) return false // "Lan 300": a person, not "lần"
  if (groups.length > 1) return false
  const digits = end - start
  if (word === "nam") {
    const accented = lowerSub(text, ws, we) === "n\u0103m" // năm
    const lead = sub(shadow, start, start + 2)
    const year = digits === 4 && (lead === "19" || lead === "20")
    return year || (accented && digits <= 2)
  }
  if (word === WEEKDAY_WORD)
    return digits === 1 && oneOf(shadow[start], "2345678")
  return digits <= 2
}

/** Is the number followed by a counted-thing or time word (`2 tô`, `3 vé`, `5 tháng`)? */
function quantityAfter(shadow: Chars, end: number): boolean {
  const [ws, we] = nextWord(shadow, end)
  return ws !== we && inList(QUANTITY_AFTER, sub(shadow, ws, we))
}

// --------------------------------------------------------------------------- units

/** Is `shadow[start:stop]` a currency word? `dong` must be `đồng`, not `đóng` (pay). */
function isCurrency(
  text: Chars,
  shadow: Chars,
  start: number,
  stop: number
): boolean {
  const word = sub(shadow, start, stop)
  if (!inList(CURRENCY_WORDS, word)) return false
  if (word !== "dong") return true
  const lowered = lowerSub(text, start, stop)
  if (lowered.includes("\u1ed3")) return true // ồ
  if (lowered !== "dong") return false
  for (let i = stop; i < shadow.length; i++)
    if (isAlnum(shadow[i])) return false
  return true
}

/**
 * The money unit right after a number: `[word, start, end]` or `null`. A unit is glued (`50k`,
 * `2tr`) or after one space (`15 K`, `5 củ`). Currency letters need the number to be grouped or
 * long when spaced, so `250 d` stays a bare number.
 */
function readUnit(
  text: Chars,
  shadow: Chars,
  pos: number,
  numberIsBig: boolean
): [string, number, number] | null {
  const sign = String.fromCodePoint(CURRENCY_SIGN)
  const n = shadow.length
  const glued =
    pos < n && (isLetter(shadow[pos]) || shadow[pos] === CURRENCY_SIGN)
  let start = pos
  if (!glued) {
    const [ws, we] = nextWord(shadow, pos)
    if (ws === we) {
      if (
        pos + 1 < n &&
        shadow[pos] === SPACE &&
        shadow[pos + 1] === CURRENCY_SIGN
      ) {
        return [sign, pos + 1, pos + 2]
      }
      return null
    }
    start = ws
  }
  if (shadow[start] === CURRENCY_SIGN) return [sign, start, start + 1]
  const end = wordEnd(shadow, start)
  const word = sub(shadow, start, end)
  const isMoney =
    inList(THOUSAND_LETTER, word) ||
    inList(THOUSAND_WORDS, word) ||
    inList(MILLION, word) ||
    inList(BILLION, word) ||
    inList(SLANG, word) ||
    isCurrency(text, shadow, start, end)
  if (!isMoney) return null
  if (!glued && word === "d" && !numberIsBig) return null
  return [word, start, end]
}

/** Does a goods noun follow a slang unit (`5 lít xăng`, `2 chai bia`)? */
function slangIsQuantity(text: Chars, shadow: Chars, end: number): boolean {
  const [ws, we] = nextWord(shadow, end)
  if (ws === we) return false
  return (
    inList(SLANG_QTY, sub(shadow, ws, we)) ||
    inList(SLANG_QTY_ACCENTED, lowerSub(text, ws, we))
  )
}

/** Extend past a unit: half (`rưỡi`), glued/spaced digits (`1tr5`, `1 triệu 2`). */
function continuation(shadow: Chars, end: number, word: string): number {
  if (!inList(CONTINUABLE, word)) return end
  const n = shadow.length
  const [ws, we] = nextWord(shadow, end)
  if (ws !== we && inList(HALF, sub(shadow, ws, we))) return we
  const glued = end < n && isDigit(shadow[end])
  const spaced =
    !glued && end + 1 < n && shadow[end] === SPACE && isDigit(shadow[end + 1])
  if (!glued && !spaced) return end
  if (spaced && word === "tr") return end // "5tr 20/10": a spaced digit after a bare "tr" is another token
  const ds = end + (spaced ? 1 : 0)
  const de = digitRun(shadow, ds)
  const size = de - ds
  if (size > 3 || (spaced && size !== 1 && size !== 3)) return end
  if (de < n && (oneOf(shadow[de], "/:%") || isLetter(shadow[de]))) return end
  if (
    de + 1 < n &&
    (shadow[de] === DOT || shadow[de] === COMMA) &&
    isDigit(shadow[de + 1])
  ) {
    return end
  }
  if (spaced && quantityAfter(shadow, de)) return end
  return de
}

/** Extend past a currency marker: `250.000đ`, `1.200.000 vnd`, `350.000 đồng`. */
function currencySuffix(text: Chars, shadow: Chars, end: number): number {
  const n = shadow.length
  if (end < n && shadow[end] === CURRENCY_SIGN) return end + 1
  const glued = end < n && isLetter(shadow[end])
  const start = glued ? end : nextWord(shadow, end)[0]
  if (!glued && start === end) {
    if (
      end + 1 < n &&
      shadow[end] === SPACE &&
      shadow[end + 1] === CURRENCY_SIGN
    )
      return end + 2
    return end
  }
  const stop = wordEnd(shadow, start)
  return isCurrency(text, shadow, start, stop) ? stop : end
}

// --------------------------------------------------------------------------- candidates

/** Analyse the digit token at `pos`: `[candidate or null, scan resume position]`. */
function digitCandidate(text: Chars, shadow: Chars, pos: number): Resume {
  const n = shadow.length
  if (pos > 0 && isLetter(shadow[pos - 1]))
    return [null, skipIdentifier(shadow, pos)] // t10, q4, x2
  const phone = phoneEnd(shadow, pos)
  if (phone !== null) return [null, phone]
  const [end, groups] = numberEnd(shadow, pos)
  // dates, times, fractions: 20/10, 9/12/2024, 10:30
  if (
    end < n &&
    oneOf(shadow[end], "/:") &&
    end + 1 < n &&
    isDigit(shadow[end + 1])
  ) {
    let stop = end + 1
    while (stop < n && (isDigit(shadow[stop]) || oneOf(shadow[stop], "/:.")))
      stop++
    return [null, stop]
  }
  if (pos > 0 && shadow[pos - 1] === SLASH) return [null, end]
  // percentages: 20%, 5.5%/năm
  const after = end < n && shadow[end] === SPACE ? end + 1 : end
  if (after < n && shadow[after] === PERCENT) return [null, after + 1]

  const digits = digitCount(shadow, pos, end)
  const grouped = isGrouped(groups)
  const big = grouped || digits >= BARE_PLAIN_MONEY_DIGITS
  if (groups.length === 1 && digits > MAX_PLAIN_DIGITS) return [null, end]

  const unit = readUnit(text, shadow, end, big)
  if (unit === null) {
    // digits glued to a non-money letter run (`10kg`, `5h`, `2x`)
    if (end < n && isLetter(shadow[end]))
      return [null, skipIdentifier(shadow, end)]
    if (big) return [{ start: pos, end, tier: TIER_MONEY }, end]
    if (
      explainedBefore(text, shadow, pos, end, groups) ||
      quantityAfter(shadow, end)
    ) {
      return [null, end]
    }
    return [{ start: pos, end, tier: TIER_BARE }, end]
  }

  const [word, , unitEnd] = unit
  if (inList(SLANG, word)) {
    if (slangIsQuantity(text, shadow, unitEnd)) return [null, unitEnd]
    const stop = currencySuffix(
      text,
      shadow,
      continuation(shadow, unitEnd, word)
    )
    return [{ start: pos, end: stop, tier: TIER_SLANG }, stop]
  }
  let stop = continuation(shadow, unitEnd, word)
  if (!inList(CURRENCY_WORDS, word)) stop = currencySuffix(text, shadow, stop)
  return [{ start: pos, end: stop, tier: TIER_MONEY }, stop]
}

/** A numeral-word amount at `pos` (a word start): `hai trăm nghìn`, `nửa củ`. */
function wordCandidate(text: Chars, shadow: Chars, pos: number): Resume {
  const end = wordEnd(shadow, pos)
  const word = sub(shadow, pos, end)
  let cursor = end
  if (inList(HALF_PREFIX, word)) {
    // "nửa" starts the amount
  } else if (inList(NUMBER_WORDS, word)) {
    for (;;) {
      const [ws, we] = nextWord(shadow, cursor)
      if (ws !== we && inList(NUMBER_WORDS, sub(shadow, ws, we))) {
        cursor = we
        continue
      }
      break
    }
  } else {
    return [null, end]
  }
  const [ws, we] = nextWord(shadow, cursor)
  const unit = ws !== we ? sub(shadow, ws, we) : ""
  if (!inList(WORD_UNITS, unit)) return [null, cursor] // no suffix of this numeral run can end in a unit either
  if (inList(SLANG, unit) && slangIsQuantity(text, shadow, we))
    return [null, we]
  const stop = currencySuffix(text, shadow, continuation(shadow, we, unit))
  const tier = inList(SLANG, unit) ? TIER_SLANG : TIER_MONEY
  return [{ start: pos, end: Math.min(stop, shadow.length), tier }, stop]
}

/** All money candidates of NFC `text`, left to right. */
function findCandidates(text: Chars): Candidate[] {
  const shadow = fold(text)
  const n = shadow.length
  const out: Candidate[] = []
  let pos = 0
  while (pos < n) {
    const char = shadow[pos]
    let found: Resume
    if (isDigit(char)) found = digitCandidate(text, shadow, pos)
    else if (
      isLetter(char) &&
      (pos === 0 || !(isLetter(shadow[pos - 1]) || isDigit(shadow[pos - 1])))
    ) {
      found = wordCandidate(text, shadow, pos)
    } else found = [null, pos + 1]
    if (found[0] !== null) out.push(found[0])
    pos = Math.max(found[1], pos + 1)
  }
  return out
}

/**
 * The amount among `candidates`: best tier; first of the tier, last for bare numbers (several
 * explicit amounts: the leading one is the main one and fees follow; several bare numbers are
 * mostly quantity then price).
 */
function choose(candidates: Candidate[]): Candidate | null {
  if (candidates.length === 0) return null
  const best = Math.min(...candidates.map((c) => c.tier))
  const tied = candidates.filter((c) => c.tier === best)
  return best === TIER_BARE ? tied[tied.length - 1] : tied[0]
}

/** The amount of an already NFC-normalized note, as a span of the caller's string. */
export function parseNormalizedValue(
  normalized: NormalizedText
): ValueSpan | null {
  const chosen = choose(findCandidates(normalized.cps))
  if (chosen === null) return null
  const start = startToOriginal(normalized, chosen.start)
  const end = endToOriginal(normalized, chosen.end)
  return { text: fromCodePoints(normalized.original, start, end), start, end }
}

/** The monetary amount of `text` as a span of the caller's string, or `null`. */
export function parseValue(text: string): ValueSpan | null {
  return parseNormalizedValue(normalizeNfc(text))
}
