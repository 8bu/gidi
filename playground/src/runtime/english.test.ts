import assert from "node:assert/strict"
import { createHash } from "node:crypto"
import { existsSync, readFileSync } from "node:fs"
import { test } from "node:test"

import { parseConfig } from "./config.ts"
import { snapToWords } from "./decode.ts"
import { normalizeNfc, toCodePoints } from "./text.ts"
import { createTokenizer } from "./tokenizer.ts"

// Written by `scripts/dump_web_parity_en.py --bundle models/gidi-en-v2-bundle --fixture` from the
// Python runtime on gidi-en-v2 (the BPE English bundle).
const fixture = JSON.parse(
  readFileSync(new URL("./english.fixture.json", import.meta.url), "utf8")
) as {
  tokenizer_sha256: string
  max_length: number
  tokens: {
    text: string
    ids: number[]
    offsets: [number, number][]
    special: number[]
    truncated: boolean
  }[]
  snaps: {
    text: string
    start: number
    end: number
    snapped: [number, number]
  }[]
}

// The synced web release (`pnpm build:web` / `node scripts/sync-release.mjs`).
const release = new URL(
  "../../public/release/gidi-en-v2/2.0.1/",
  import.meta.url
)
const synced = existsSync(new URL("tokenizer.json", release))

test("word snap matches gidi.inference.decode.snap_to_words on English cases", () => {
  for (const { text, start, end, snapped } of fixture.snaps) {
    assert.deepEqual(snapToWords(toCodePoints(text), start, end), snapped, text)
  }
})

test(
  "English tokenizer matches the Python tokenizer",
  { skip: !synced },
  () => {
    const bytes = readFileSync(new URL("tokenizer.json", release))
    assert.equal(
      createHash("sha256").update(bytes).digest("hex"),
      fixture.tokenizer_sha256
    )
    const tokenizer = createTokenizer(
      JSON.parse(bytes.toString("utf8")),
      fixture.max_length,
      "bpe"
    )
    for (const expected of fixture.tokens) {
      const encoded = tokenizer.encode(normalizeNfc(expected.text).cps)
      assert.deepEqual(encoded.ids, expected.ids, expected.text)
      assert.deepEqual(encoded.offsets, expected.offsets, expected.text)
      assert.deepEqual(
        encoded.specialTokensMask,
        expected.special,
        expected.text
      )
      assert.equal(encoded.truncated, expected.truncated, expected.text)
    }
  }
)

test(
  "the English config declares no value; the Vietnamese one the rule parser",
  { skip: !synced },
  () => {
    const english = parseConfig(
      JSON.parse(readFileSync(new URL("config.json", release), "utf8"))
    )
    assert.equal(english.valueSource, "none")
    assert.equal(english.modelVersion, "gidi-en-v2")
    const vietnamese = parseConfig(
      JSON.parse(
        readFileSync(
          new URL(
            "../../public/release/gidi-finance-v4/4.0.0/config.json",
            import.meta.url
          ),
          "utf8"
        )
      )
    )
    assert.equal(vietnamese.valueSource, "rule-parser")
    assert.throws(() =>
      parseConfig({
        ...JSON.parse(readFileSync(new URL("config.json", release), "utf8")),
        value_parser: { name: "x" },
      })
    )
  }
)
