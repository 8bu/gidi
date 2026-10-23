import assert from "node:assert/strict"
import { createHash } from "node:crypto"
import { existsSync, readFileSync } from "node:fs"
import { test } from "node:test"

import { parseConfig } from "./config.ts"
import { snapToWords } from "./decode.ts"
import { normalizeNfc, toCodePoints } from "./text.ts"
import { createTokenizer } from "./tokenizer.ts"

// Written by `scripts/dump_web_parity_en.py --fixture` from the Python runtime on gidi-en-v4:
// every test, dev and probe note, the app notes and seeded fuzz strings (emoji, accents, symbols,
// URLs, special tokens, long text), tokenized by the Hugging Face `tokenizers` crate.
const fixture = JSON.parse(
  readFileSync(new URL("./wordpiece.fixture.json", import.meta.url), "utf8")
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
  "../../public/release/gidi-en-v4/4.0.0/",
  import.meta.url
)
const synced = existsSync(new URL("tokenizer.json", release))

test("the fixture covers the eval notes and the fuzz strings", () => {
  assert.ok(fixture.tokens.length > 1000)
  assert.ok(fixture.tokens.some((t) => t.truncated))
  assert.ok(
    fixture.tokens.some((t) => t.ids.includes(1)),
    "an [UNK] case"
  )
})

test("word snap matches gidi.inference.decode.snap_to_words on English cases", () => {
  for (const { text, start, end, snapped } of fixture.snaps) {
    assert.deepEqual(snapToWords(toCodePoints(text), start, end), snapped, text)
  }
})

test(
  "WordPiece tokenizer matches the Python tokenizer: ids, offsets, masks",
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
      "wordpiece"
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
  "the tokenizer type of the file and of the config must agree",
  { skip: !synced },
  () => {
    const json = JSON.parse(
      readFileSync(new URL("tokenizer.json", release), "utf8")
    )
    assert.throws(() => createTokenizer(json, 32, "bpe"), /says bpe/)
    const bpe = JSON.parse(
      readFileSync(
        new URL(
          "../../public/release/gidi-en-v2/2.0.1/tokenizer.json",
          import.meta.url
        ),
        "utf8"
      )
    )
    assert.throws(() => createTokenizer(bpe, 32, "wordpiece"), /says wordpiece/)
  }
)

test(
  "the v4 config declares a value head decoded by argmax; a bad head is refused",
  { skip: !synced },
  () => {
    const raw = JSON.parse(
      readFileSync(new URL("config.json", release), "utf8")
    ) as Record<string, unknown>
    const config = parseConfig(raw)
    assert.equal(config.valueSource, "head")
    assert.equal(config.tokenizerType, "wordpiece")
    assert.equal(config.modelVersion, "gidi-en-v4")
    assert.deepEqual(config.outputNames, [
      "type_logits",
      "tag_logits",
      "value_logits",
    ])
    assert.throws(() =>
      parseConfig({
        ...raw,
        value_decoding: {
          method: "crf_viterbi",
          span_selection: "highest_confidence",
        },
      })
    )
    assert.throws(() => parseConfig({ ...raw, value_snap: undefined }))
    assert.throws(() => parseConfig({ ...raw, tokenizer_type: "unigram" }))
    assert.throws(() =>
      parseConfig({
        ...raw,
        onnx: {
          ...(raw.onnx as object),
          outputs: ["type_logits", "tag_logits"],
        },
      })
    )
  }
)
