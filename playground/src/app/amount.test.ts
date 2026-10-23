import assert from "node:assert/strict"
import { test } from "node:test"

import {
  amountFromEnglish,
  amountFromEnglishSpan,
  findEnglishValue,
  readEnglishValue,
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

test("English value spans of the model read as dollars", () => {
  for (const [span, amount] of [
    ["50", 50],
    ["$50", 50],
    ["$5.99", 5.99],
    ["20 bucks", 20],
    ["20bucks", 20],
    ["120$", 120],
    ["s$30", 30],
    ["150s$", 150],
    ["us$5", 5],
    ["rs 500", 500],
    ["£45", 45],
    ["€50", 50],
    ["5k", 5000],
    ["5K", 5000],
    ["2.5k", 2500],
    ["1.2m", 1_200_000],
    ["1,200", 1200],
    ["1,200 usd", 1200],
    ["2 grand", 2000],
    ["3 million", 3_000_000],
    ["twenty bucks", 20],
    ["twenty five", 25],
    ["twenty-five dollars", 25],
    ["a hundred", 100],
    ["two hundred and fifty", 250],
    ["a thousand", 1000],
    ["fiver", 5],
    ["tenner", 10],
    ["50 cents", 0.5],
  ] as const) {
    assert.equal(amountFromEnglishSpan(span), amount, span)
  }
})

test("English value spans that are not an amount are refused", () => {
  for (const span of [
    "",
    "  ",
    "$",
    "bucks",
    "a",
    "and",
    "abc",
    "5 zz",
    "1,2,3",
    "-5",
  ]) {
    assert.equal(amountFromEnglishSpan(span), null, JSON.stringify(span))
  }
})

test("a model with a value head decides; the scanner only fills in when it must", () => {
  const model = (valueText: string | null, span: [number, number] | null) => ({
    hasValueHead: true,
    valueText,
    span,
  })
  // the model's span wins over what a scan would pick
  assert.deepEqual(
    readEnglishValue("2 coffees $9 and 4 tips", model("$9", [11, 13])),
    {
      valueText: "$9",
      span: [11, 13],
      amount: 9,
    }
  )
  assert.deepEqual(readEnglishValue("got paid 2400", model("2400", [9, 13])), {
    valueText: "2400",
    span: [9, 13],
    amount: 2400,
  })
  // no span from the model: no amount, even though the note has a number (`4` for `for`)
  assert.deepEqual(readEnglishValue("paid 4 rent", model(null, null)), {
    valueText: null,
    span: null,
    amount: null,
  })
  // a span that runs on into the next word is cut back to its amount
  assert.deepEqual(
    readEnglishValue("paynow sis 30 4 bday", model("30 4", [10, 14])),
    {
      valueText: "30",
      span: [10, 12],
      amount: 30,
    }
  )
  assert.deepEqual(
    readEnglishValue("kopi o 1.80 rn", model("1.80 rn", [7, 14])),
    {
      valueText: "1.80",
      span: [7, 11],
      amount: 1.8,
    }
  )
  // a span that is not an amount: the scan of the note answers
  assert.deepEqual(readEnglishValue("lent bro 20", model("bro", [5, 8])), {
    valueText: "20",
    span: [9, 11],
    amount: 20,
  })
  // a model without a value head (gidi-en v2): always the scan
  assert.deepEqual(
    readEnglishValue("venmo'd mom 50", {
      hasValueHead: false,
      valueText: null,
      span: null,
    }),
    { valueText: "50", span: [12, 14], amount: 50 }
  )
})
