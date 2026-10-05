/**
 * Amounts of the notes app, in whole VND.
 *
 * `amountToVnd` reads the value text the deterministic parser returns (see
 * `src/runtime/value-parser.ts`, `src/runtime/lexicon.ts`): digits with `.`/`,`/space grouping,
 * the unit words `k`, `nghìn`, `ngàn`, `tr`, `triệu`, `tỷ`, the slang `củ` (1 million) and
 * `xị`/`chai`/`lít` (100 thousand, as the lexicon defines them), a half (`2 củ rưỡi`), trailing
 * digits (`1tr2`, `1 triệu 2`, `1tr500k`), numeral words (`hai trăm nghìn`, `nửa củ`) and the
 * currency markers `đ`, `₫`, `vnd`, `đồng`.
 */

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
  chai: SLANG_HUNDRED_THOUSAND,
  lit: SLANG_HUNDRED_THOUSAND,
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
