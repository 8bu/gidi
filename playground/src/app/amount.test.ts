import assert from "node:assert/strict"
import { test } from "node:test"

import {
  amountFromEnglish,
  findEnglishValue,
  formatAmount,
  formatUsd,
  formatUsdCompact,
} from "./amount.ts"

const read = (note: string) => {
  const { valueText, span, amount } = findEnglishValue(note)
  return [valueText, span, amount]
}

test("English notes: the amount is the number as written", () => {
  assert.deepEqual(read("venmo'd mom 50"), ["50", [12, 14], 50])
  assert.deepEqual(read("bro paid me back 20"), ["20", [17, 19], 20])
  assert.deepEqual(read("uber to the airport 42.50"), ["42.50", [20, 25], 42.5])
  assert.deepEqual(read("rent 1,200 due"), ["1,200", [5, 10], 1200])
  assert.deepEqual(read("lent sam 2.5k"), ["2.5k", [9, 13], 2500])
  assert.deepEqual(read("bonus 1.2m"), ["1.2m", [6, 10], 1_200_000])
  assert.deepEqual(read("won 3 million"), ["3 million", [4, 13], 3_000_000])
})

test("English notes: a $ sign or money word wins; the span leaves them out", () => {
  assert.deepEqual(read("2 coffees $9"), ["9", [11, 12], 9])
  assert.deepEqual(read("gave jake 20 bucks for 2 beers"), ["20", [10, 12], 20])
  assert.deepEqual(read("lunch 15 then 30 dollars"), ["30", [14, 16], 30])
})

test("English notes: no amount for times, dates, ordinals and plain words", () => {
  assert.deepEqual(read("got paid"), [null, null, null])
  assert.deepEqual(read("dinner at 7:30"), [null, null, null])
  assert.deepEqual(read("paid on 3/5"), [null, null, null])
  assert.deepEqual(read("my 2nd coffee at 5pm"), [null, null, null])
  assert.deepEqual(read("bought x2"), [null, null, null])
})

test("code-point spans stay correct after an astral character", () => {
  assert.deepEqual(read("🍕 pizza 12"), ["12", [8, 10], 12])
})

test("amountFromEnglish rejects what is not a number", () => {
  assert.equal(amountFromEnglish("abc"), null)
  assert.equal(amountFromEnglish("1,2,3"), null)
  assert.equal(amountFromEnglish("5 zz"), null)
  assert.equal(amountFromEnglish("$5"), 5)
})

test("English amounts are USD with en-US syntax and no Vietnamese scaling", () => {
  assert.deepEqual(read("lunch 12"), ["12", [6, 8], 12])
  assert.deepEqual(read("rent 800"), ["800", [5, 8], 800])
  assert.deepEqual(read("gave jake 5k"), ["5k", [10, 12], 5000])
  assert.deepEqual(read("he owes me 2 grand"), ["2 grand", [11, 18], 2000])
  assert.deepEqual(read("coffee 12.50"), ["12.50", [7, 12], 12.5])
  assert.deepEqual(read("rent 1,200.50"), ["1,200.50", [5, 13], 1200.5])
  // Vietnamese words mean nothing in English: no unit, so the number stays as it is.
  assert.deepEqual(read("lunch 15 tr"), ["15", [6, 8], 15])
  assert.equal(amountFromEnglish("15tr"), null)
  assert.equal(amountFromEnglish("2 cu"), null)
  // Other currencies are out of scope: the number is kept, nothing is converted.
  assert.deepEqual(read("pint £5"), ["5", [6, 7], 5])
  assert.deepEqual(read("taxi 20 quid"), ["20", [5, 7], 20])
})

test("USD formats", () => {
  assert.equal(formatUsd(50), "$50.00")
  assert.equal(formatUsd(1234.5), "$1,234.50")
  assert.equal(formatUsd(-12), "-$12.00")
  assert.equal(formatUsd(0), "$0.00")
  assert.equal(formatUsdCompact(500), "$500")
  assert.equal(formatUsdCompact(12.5), "$12.50")
  assert.equal(formatUsdCompact(1250), "$1.3k")
  assert.equal(formatUsdCompact(15_000_000), "$15M")
  assert.equal(formatUsdCompact(2_500_000_000), "$2.5B")
  assert.equal(formatUsdCompact(-1500), "-$1.5k")
})

test("amounts are formatted by the currency of their database", () => {
  assert.equal(formatAmount(50, "USD"), "$50.00")
  assert.equal(formatAmount(1200000, "VND"), "1.200.000 ₫")
})
