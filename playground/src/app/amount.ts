/**
 * Amounts of the notes app, in whole VND.
 *
 * `amountToVnd` reads the value text the deterministic parser returns (see
 * `src/runtime/value-parser.ts`, `src/runtime/lexicon.ts`): digits with `.`/`,`/space grouping,
 * the unit words `k`, `nghìn`, `ngàn`, `tr`, `triệu`, `tỷ`, the slang `củ`/`chai` (1 million),
 * `xị`/`lít`/`cành` (100 thousand) and `tỏi` (1 billion), a half (`2 củ rưỡi`), trailing
 * digits (`1tr2`, `1 triệu 2`, `1tr500k`), numeral words (`hai trăm nghìn`, `nửa củ`) and the
 * currency markers `đ`, `₫`, `vnd`, `đồng`. The app reads `chai` as a million although the
 * released lexicon notes it as 100k; `cành` and `tỏi` are unknown to the released parser, so
 * `widenSlangValue` adds them to the span (the parser is pinned by the v3 bundle).
 */

import type { Currency } from "./types.ts"

const THOUSAND = 1_000
const MILLION = 1_000_000
const SLANG_HUNDRED_THOUSAND = 100_000
const BILLION = 1_000_000_000

/** Folded (accent-free, lowercase) unit word -> value in VND. */
const UNITS: Readonly<Record<string, number>> = {
  k: THOUSAND,
  ng: THOUSAND,
  nghin: THOUSAND,
  ngan: THOUSAND,
  tr: MILLION,
  trieu: MILLION,
  cu: MILLION,
  ty: BILLION,
  xi: SLANG_HUNDRED_THOUSAND,
  lit: SLANG_HUNDRED_THOUSAND,
  canh: SLANG_HUNDRED_THOUSAND,
  chai: MILLION,
  toi: BILLION,
}

const DIGIT_WORDS: Readonly<Record<string, number>> = {
  mot: 1,
  hai: 2,
  ba: 3,
  bon: 4,
  nam: 5,
  lam: 5,
  sau: 6,
  bay: 7,
  tam: 8,
  chin: 9,
}

type Token = { kind: "num"; text: string } | { kind: "word"; word: string }

/** Lowercase, accent-free (`Triệu` -> `trieu`, `đ` -> `d`); `₫` is kept. */
function fold(text: string): string {
  return text
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/đ/g, "d")
    .replace(/\s+/g, " ")
    .trim()
}

function tokenize(text: string): Token[] | null {
  const pattern = /\d{1,3}(?: \d{3}){2,}(?:[.,]\d+)*|\d+(?:[.,]\d+)*|[a-z]+| /y
  const tokens: Token[] = []
  let pos = 0
  while (pos < text.length) {
    pattern.lastIndex = pos
    const match = pattern.exec(text)
    if (match === null) return null
    const piece = match[0]
    pos += piece.length
    if (piece === " ") continue
    tokens.push(
      /\d/.test(piece)
        ? { kind: "num", text: piece }
        : { kind: "word", word: piece }
    )
  }
  return tokens
}

/**
 * A digit token as a number. Thousand-grouped (`80.000`, `1,200,000`, `1 000 000`) is an integer;
 * a single separator before a unit is a decimal point (`1,5tr`); anything else is not an amount.
 */
function parseNumber(text: string, hasUnit: boolean): number | null {
  const parts = text.replace(/ /g, "").split(/[.,]/)
  if (parts.length === 1) return Number(parts[0])
  const grouped =
    parts[0].length <= 3 &&
    Number(parts[0]) > 0 &&
    parts.slice(1).every((part) => part.length === 3)
  if (grouped) return Number(parts.join(""))
  if (hasUnit && parts.length === 2) return Number(`${parts[0]}.${parts[1]}`)
  return null
}

/** Vietnamese numeral words below 1000: `hai trăm năm mươi`, `mười lăm`, `một trăm linh năm`. */
function parseNumeralWords(words: string[]): number | null {
  if (words.length === 0) return null
  let value = 0
  let pending: number | null = null
  let sawZeroMarker = false
  for (const word of words) {
    if (Object.hasOwn(DIGIT_WORDS, word)) {
      if (pending !== null) return null
      pending = DIGIT_WORDS[word]
    } else if (word === "tram") {
      value += (pending ?? 1) * 100
      pending = null
    } else if (word === "muoi") {
      value += (pending ?? 1) * 10
      pending = null
    } else if (word === "linh" || word === "le") {
      sawZeroMarker = true
    } else {
      return null
    }
  }
  if (pending !== null) {
    // "hai trăm năm" is spoken for 250; "hai trăm linh năm" is 205.
    const clippedTens = !sawZeroMarker && value >= 100 && value % 100 === 0
    value += clippedTens ? pending * 10 : pending
  }
  return value
}

const NUMERAL_MARKERS = ["tram", "muoi", "linh", "le"]

const isNumeralWord = (word: string): boolean =>
  Object.hasOwn(DIGIT_WORDS, word) || NUMERAL_MARKERS.includes(word)

function unitValue(word: string): number | undefined {
  return Object.hasOwn(UNITS, word) ? UNITS[word] : undefined
}

function evaluate(tokens: Token[]): number | null {
  let i = 0
  const first = tokens[0]
  if (first === undefined) return null

  // Leading number: digits, numeral words, or "nửa" (a half).
  let lead: number | null
  let bareText: string | null = null
  if (first.kind === "num") {
    bareText = first.text
    lead = null // decided once we know whether a unit follows
    i = 1
  } else if (first.word === "nua") {
    lead = 0.5
    i = 1
  } else if (isNumeralWord(first.word)) {
    const words: string[] = []
    while (i < tokens.length) {
      const token = tokens[i]
      if (token.kind !== "word" || !isNumeralWord(token.word)) break
      words.push(token.word)
      i++
    }
    lead = parseNumeralWords(words)
  } else {
    return null
  }

  const unitToken = tokens[i]
  const mult =
    unitToken !== undefined && unitToken.kind === "word"
      ? unitValue(unitToken.word)
      : undefined

  if (mult === undefined) {
    // No unit: only a plain number or numeral words stand alone.
    if (i !== tokens.length) return null
    if (bareText !== null) return parseNumber(bareText, false)
    return lead === 0.5 ? null : lead // a lone "nửa" is not an amount
  }
  i++

  if (bareText !== null) lead = parseNumber(bareText, true)
  if (lead === null) return null
  let value = lead * mult

  const tail = tokens[i]
  if (tail !== undefined) {
    if (tail.kind === "word" && tail.word === "ruoi") {
      value += mult / 2
      i++
    } else if (tail.kind === "num" && /^\d+$/.test(tail.text)) {
      // `1tr2` = 1.2tr, `1tr25` = 1.25tr, `2k5` = 2.5k; `1tr500k` adds 500 thousand.
      i++
      const next = tokens[i]
      const nextMult =
        next !== undefined && next.kind === "word"
          ? unitValue(next.word)
          : undefined
      if (nextMult !== undefined && nextMult < mult) {
        value += Number(tail.text) * nextMult
        i++
      } else {
        value += (Number(tail.text) * mult) / 10 ** tail.text.length
      }
    }
  }
  if (i !== tokens.length) return null
  return value
}

/**
 * The VND amount a value text stands for, rounded to whole dong; `null` when the text is not a
 * VND amount (foreign currency, unknown words, malformed numbers). A bare amount under 1,000
 * with no unit and no currency suffix counts in thousands, as notes write it (`phở bò 60` = 60k).
 */
export function amountToVnd(valueText: string): number | null {
  const folded = fold(valueText)
  if (folded === "" || folded.includes("$") || folded.includes("usd"))
    return null
  const text = folded.replace(/ ?(?:₫|vnd|dong|d)$/, "")
  const tokens = tokenize(text)
  if (tokens === null) return null
  let value = evaluate(tokens)
  if (value === null || !Number.isFinite(value) || value < 0) return null
  if (text === folded && value < 1000) value *= 1000
  const rounded = Math.round(value)
  return Number.isSafeInteger(rounded) ? rounded : null
}

/**
 * Slang units the released parser does not read, as written: the right accent or none at all
 * (`5 toi` in a no-diacritic note). Other accents are other words (`tôi` I, `cạnh` beside).
 */
const TRAILING_SLANG: Readonly<Record<string, true>> = {
  tỏi: true,
  toi: true,
  cành: true,
  canh: true,
}
/** `cành` followed by one of these (folded) counts branches of flowers, not money. */
const BRANCH_GOODS: Readonly<Record<string, true>> = {
  hoa: true,
  dao: true,
  mai: true,
  lan: true,
  hong: true,
  cuc: true,
  ly: true,
  tre: true,
  cay: true,
}

const BARE_NUMBER = /^\d+(?:[.,]\d+)?$/
const isLetter = (char: string | undefined): boolean =>
  char !== undefined && /\p{L}/u.test(char)

/** The word starting at `pos` of `chars` (code points), or `null` when no letter is there. */
function wordAt(
  chars: string[],
  pos: number
): { word: string; end: number } | null {
  let end = pos
  while (isLetter(chars[end])) end += 1
  return end === pos ? null : { word: chars.slice(pos, end).join(""), end }
}

/**
 * The value span widened over a slang unit the released parser stops before: `1 tỏi`,
 * `50 cành`, `2 tỏi rưỡi`. Only a bare number span is widened; spans are code-point offsets.
 */
export function widenSlangValue(
  note: string,
  valueText: string | null,
  span: [number, number] | null
): { valueText: string | null; span: [number, number] | null } {
  if (valueText === null || span === null || !BARE_NUMBER.test(valueText)) {
    return { valueText, span }
  }
  const chars = Array.from(note)
  if (chars[span[1]] !== " ") return { valueText, span }
  const unit = wordAt(chars, span[1] + 1)
  if (
    unit === null ||
    !Object.hasOwn(TRAILING_SLANG, unit.word.normalize("NFC").toLowerCase())
  ) {
    return { valueText, span }
  }
  let end = unit.end
  const next = chars[end] === " " ? wordAt(chars, end + 1) : null
  if (next !== null && fold(next.word) === "ruoi") end = next.end
  else if (
    next !== null &&
    fold(unit.word) === "canh" &&
    Object.hasOwn(BRANCH_GOODS, fold(next.word))
  ) {
    return { valueText, span }
  }
  const widened: [number, number] = [span[0], end]
  return { valueText: chars.slice(span[0], end).join(""), span: widened }
}

function groupThousands(digits: string): string {
  return digits.replace(/\B(?=(\d{3})+(?!\d))/g, ".")
}

/** `1.200.000 ₫` */
export function formatVnd(n: number): string {
  const rounded = Math.round(n)
  const sign = rounded < 0 ? "-" : ""
  return `${sign}${groupThousands(String(Math.abs(rounded)))} ₫`
}

/** `500k`, `1,2tr`, `15tr`, `1,5 tỷ`; below a thousand the plain number. */
export function formatVndCompact(n: number): string {
  const rounded = Math.round(n)
  const sign = rounded < 0 ? "-" : ""
  const abs = Math.abs(rounded)
  if (abs < THOUSAND) return `${sign}${abs}`

  const roundTenth = (value: number): number => Math.round(value * 10) / 10
  if (abs < MILLION) {
    const k = roundTenth(abs / THOUSAND)
    if (k < 1000) return `${sign}${String(k).replace(".", ",")}k`
  }
  if (abs < BILLION) {
    const tr = roundTenth(abs / MILLION)
    if (tr < 1000) return `${sign}${String(tr).replace(".", ",")}tr`
  }
  const ty = roundTenth(abs / BILLION)
  const whole = Math.trunc(ty)
  const decimal =
    ty === whole ? "" : `,${String(Math.round((ty - whole) * 10))}`
  return `${sign}${groupThousands(String(whole))}${decimal} tỷ`
}

/* -------------------------------------------------------------- English */

/**
 * English notes are US dollars with en-US number syntax: the amount is the number as written (`50`,
 * `4.50`, `1,200`, `2.5k`, `1.2m`, `3 million`, `2 grand`), never scaled by the Vietnamese rules
 * (a bare number is not thousands, `tr`/`củ` mean nothing). Other currencies (`£`, `€`, `quid`,
 * `sgd`) are not converted: the number is kept as is.
 *
 * The English v4 model reads the amount itself (a value span: `$50`, `2 grand`, `twenty bucks`);
 * `amountFromEnglishSpan` turns that span into a number. `findEnglishValue` is the deterministic
 * scan of the note for a model without a value head (gidi-en v2) and the fallback when the
 * model's span cannot be read as an amount: a number with a `$` sign or a money word wins;
 * otherwise the last number (notes end with the amount: `venmo'd mom 50`). Times and dates
 * (`7:30`, `3/5`), ordinals and words with digits (`5pm`, `2nd`, `x2`) are not amounts.
 */
const ENGLISH_NUMBER =
  /(?<![\p{L}\p{N}.,/:$])(\$)?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(k|m|b|bn|\s?(?:mil|million|billion|grand))?(\s?(?:bucks?|dollars?|usd))?(?![\p{L}\p{N}]|[/:]\d)/giu

const ENGLISH_SCALE: Readonly<Record<string, number>> = {
  k: THOUSAND,
  m: MILLION,
  mil: MILLION,
  million: MILLION,
  b: BILLION,
  bn: BILLION,
  billion: BILLION,
  grand: THOUSAND,
}

/** The number a matched English value text stands for; `null` if it is not a finite number. */
export function amountFromEnglish(valueText: string): number | null {
  const match = /^\$?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s?([a-z]*)$/i.exec(
    valueText.trim()
  )
  if (match === null) return null
  const scale = match[3] === "" ? 1 : ENGLISH_SCALE[match[3].toLowerCase()]
  if (scale === undefined) return null
  const value =
    Number(`${match[1].replaceAll(",", "")}${match[2] ?? ""}`) * scale
  if (!Number.isFinite(value)) return null
  const rounded = Math.round(value * 100) / 100
  return Number.isSafeInteger(Math.round(rounded)) ? rounded : null
}

/** The amount of an English note as `valueText` and its code-point span, or all `null`. */
export function findEnglishValue(note: string): {
  valueText: string | null
  span: [number, number] | null
  amount: number | null
} {
  let best: { start: number; end: number; marked: boolean } | null = null
  for (const match of note.matchAll(ENGLISH_NUMBER)) {
    const marked = match[1] !== undefined || match[4] !== undefined
    if (best !== null && best.marked && !marked) continue
    // The span covers the number and its scale (`2.5k`), not the `$` or the money word.
    const start = match.index + (match[1] === undefined ? 0 : 1)
    const end = match.index + match[0].length - (match[4]?.length ?? 0)
    best = { start, end, marked }
  }
  if (best === null) return { valueText: null, span: null, amount: null }
  const valueText = note.slice(best.start, best.end)
  const toCodePoints = (index: number): number =>
    Array.from(note.slice(0, index)).length
  return {
    valueText,
    span: [toCodePoints(best.start), toCodePoints(best.end)],
    amount: amountFromEnglish(valueText),
  }
}

const WORD_UNITS: Readonly<Record<string, number>> = {
  zero: 0,
  one: 1,
  two: 2,
  three: 3,
  four: 4,
  five: 5,
  six: 6,
  seven: 7,
  eight: 8,
  nine: 9,
  ten: 10,
  eleven: 11,
  twelve: 12,
  thirteen: 13,
  fourteen: 14,
  fifteen: 15,
  sixteen: 16,
  seventeen: 17,
  eighteen: 18,
  nineteen: 19,
  fiver: 5,
  tenner: 10,
}
const WORD_TENS: Readonly<Record<string, number>> = {
  twenty: 20,
  thirty: 30,
  forty: 40,
  fifty: 50,
  sixty: 60,
  seventy: 70,
  eighty: 80,
  ninety: 90,
}
const WORD_SCALES: Readonly<Record<string, number>> = {
  thousand: THOUSAND,
  grand: THOUSAND,
  million: MILLION,
  billion: BILLION,
}

/** `twenty five`, `a hundred`, `two thousand`, `fiver`; `null` for anything else. */
function amountFromNumberWords(text: string): number | null {
  const words = text.split(/[\s-]+/).filter((w) => w !== "" && w !== "and")
  if (words.length === 0) return null
  let total = 0
  let current = 0
  let seen = false
  for (const word of words) {
    if (word === "a" || word === "an") continue // `a hundred`: the article only counts as one
    seen = true
    if (word in WORD_UNITS) current += WORD_UNITS[word]
    else if (word in WORD_TENS) current += WORD_TENS[word]
    else if (word === "hundred") current = (current === 0 ? 1 : current) * 100
    else if (word in WORD_SCALES) {
      total += (current === 0 ? 1 : current) * WORD_SCALES[word]
      current = 0
    } else return null
  }
  return seen ? total + current : null
}

/**
 * The number an English value span stands for, or `null` when it is not an amount. The span is the
 * amount as written, currency marker included (`$20`, `20 bucks`, `s$30`, `150s$`, `rs 500`,
 * `2 grand`, `twenty bucks`, `50 cents`): the marker is dropped, the rest read as digits with an
 * optional scale (`5k`, `1.2m`, `3 million`) or as number words. USD only: other currencies keep
 * their number.
 */
export function amountFromEnglishSpan(valueText: string): number | null {
  let text = valueText.toLowerCase().trim()
  const cents = /(?<![a-z])cents?\b/.test(text)
  text = text
    .replace(/(?<=\d)\s?(?:us|s|a|c)?\$/g, "")
    .replace(/\b(?:us|s|a|c)\$/g, "")
    .replace(/[$£€₹¥₱]/g, "")
    .replace(
      /(?<![a-z])(?:bucks?|bux|dollars?|usd|sgd|gbp|eur|inr|quid|pounds?|euros?|rupees?|rs|aed|cents?)\b/g,
      ""
    )
    .trim()
  if (text === "") return null
  const digits = /\d/.test(text)
  const value = digits ? amountFromEnglish(text) : amountFromNumberWords(text)
  if (value === null) return null
  const amount = cents ? Math.round(value) / 100 : value
  return Number.isFinite(amount) && amount >= 0 ? amount : null
}

export interface EnglishValue {
  valueText: string | null
  span: [number, number] | null
  amount: number | null
}

/**
 * The amount of an English note. With a model that has a value head the model decides: its span
 * is used when it reads as an amount, no span means no amount. A span that runs on into the next
 * word (`30 4` for `30 4 bday`, `5000 rs 4`, `1.80 rn`) is cut back to its longest leading part
 * that reads as an amount. The scanner answers only when the model has no value output or no
 * part of its span is an amount.
 */
export function readEnglishValue(
  note: string,
  model: {
    hasValueHead: boolean
    valueText: string | null
    span: [number, number] | null
  }
): EnglishValue {
  if (model.hasValueHead) {
    if (model.valueText === null || model.span === null) {
      return { valueText: null, span: null, amount: null }
    }
    let text = model.valueText
    for (;;) {
      const amount = amountFromEnglishSpan(text)
      if (amount !== null) {
        const end = model.span[0] + Array.from(text).length
        return { valueText: text, span: [model.span[0], end], amount }
      }
      const cut = text.trimEnd().lastIndexOf(" ")
      if (cut <= 0) break
      text = text.slice(0, cut)
    }
  }
  return findEnglishValue(note)
}

const USD = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
})

/** `$1,234.50`, `-$12.00` */
export const formatUsd = (n: number): string =>
  USD.format(Math.round(n * 100) / 100)

/** `$500`, `$1.2k`, `$15M`, `$1.5B`; cents only below a thousand. */
export function formatUsdCompact(n: number): string {
  const sign = n < 0 ? "-" : ""
  const abs = Math.abs(n)
  const tenth = (value: number): string => String(Math.round(value * 10) / 10)
  if (abs < THOUSAND) return `${sign}${formatUsd(abs)}`.replace(/\.00$/, "")
  if (abs < MILLION && Math.round((abs / THOUSAND) * 10) / 10 < 1000)
    return `${sign}$${tenth(abs / THOUSAND)}k`
  if (abs < BILLION && Math.round((abs / MILLION) * 10) / 10 < 1000)
    return `${sign}$${tenth(abs / MILLION)}M`
  return `${sign}$${tenth(abs / BILLION)}B`
}

/** Amount in the currency of the note (or database) it belongs to. */
export const formatAmount = (n: number, currency: Currency): string =>
  currency === "VND" ? formatVnd(n) : formatUsd(n)

export const formatAmountCompact = (n: number, currency: Currency): string =>
  currency === "VND" ? formatVndCompact(n) : formatUsdCompact(n)
