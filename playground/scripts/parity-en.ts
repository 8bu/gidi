/**
 * Parity gate for the English model (`gidi-en-v2`): the browser runtime (onnxruntime-web wasm EP,
 * the code the app runs) against the Python reference dumped by `scripts/dump_web_parity_en.py`.
 *
 * Same contract as `parity.ts` for the Vietnamese bundle, minus the value stages (the English
 * bundle has no value): NFC text and offset maps, token ids / offsets / special mask, the
 * ORT logits (reported as max |delta|), BIO tags, type, the word-snapped target span, original
 * offsets, and confidences within 1e-4.
 *
 * Gate (exit code): the tokenizer sweep, every input stage (NFC, ids, offsets, masks), and the
 * decoder fed the Python reference logits must be exact. The ORT logits themselves are only
 * reported: INT8 dynamic quantization differs a little between two ORT builds (the same
 * property documented in experiments/deployment-v4/web-parity.md), so some confidences move
 * and a few decisions flip; they are counted and listed, not gated.
 *
 *     uv run python scripts/dump_web_parity_en.py
 *     pnpm -C playground parity:en
 */

import { createHash } from "node:crypto"
import { readFileSync } from "node:fs"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"

import * as ort from "onnxruntime-web/wasm"

import { parseConfig } from "../src/runtime/config.ts"
import { createOrtRunner } from "../src/runtime/ort-runner.ts"
import {
  decodeLogits,
  tokenize,
  type Analysis,
  type ModelOutputs,
} from "../src/runtime/predictor.ts"
import {
  endToOriginal,
  fromCodePoints,
  normalizeNfc,
  startToOriginal,
} from "../src/runtime/text.ts"
import { BundleTokenizer } from "../src/runtime/tokenizer.ts"
import type { Prediction } from "../src/runtime/types.ts"

const here = dirname(fileURLToPath(import.meta.url))
const root = resolve(here, "../..")
const parityDir = resolve(here, "../.parity/en")
const CONFIDENCE_TOLERANCE = 1e-4
const INPUT_STAGES = new Set([
  "nfc_text",
  "nfc_offset_map",
  "input_ids",
  "attention_mask",
  "token_offsets",
  "special_tokens_mask",
  "truncated",
])

interface Trace {
  nfc: string
  nfc_start_map: number[] | null
  nfc_end_map: number[] | null
  ids: number[]
  attention_mask: number[]
  offsets: [number, number][]
  original_offsets: [number, number][]
  special: number[]
  type_logits: number[]
  tag_logits: number[][]
  target_tags: number[]
}

interface Case {
  source: string
  id: string
  text: string
  expected?: Prediction
  error?: string
  trace?: Trace
}

interface Sweep {
  text: string
  nfc: string
  ids: number[]
  offsets: [number, number][]
  truncated: boolean
  start_map?: number[]
  end_map?: number[]
}

const readJsonl = <T>(path: string): T[] =>
  readFileSync(path, "utf8")
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line) as T)
const sha256 = (bytes: Uint8Array): string =>
  createHash("sha256").update(bytes).digest("hex")
const same = (a: unknown, b: unknown): boolean =>
  JSON.stringify(a) === JSON.stringify(b)

function maxAbs(a: ArrayLike<number>, b: ArrayLike<number>): number {
  if (a.length !== b.length) return Infinity
  let max = 0
  for (let i = 0; i < a.length; i++) max = Math.max(max, Math.abs(a[i] - b[i]))
  return max
}

/** The stages of one note that must match the Python reference exactly. */
function stages(
  a: Analysis,
  t: Trace,
  e: Prediction
): [string, unknown, unknown][] {
  const p = a.prediction
  const n = a.normalized.cps.length
  const maps = (get: (i: number) => number) =>
    Array.from({ length: n + 1 }, (_, i) => get(i))
  const identity = a.normalized.startMap === null
  return [
    ["nfc_text", t.nfc, fromCodePoints(a.normalized.cps)],
    [
      "nfc_offset_map",
      [t.nfc_start_map, t.nfc_end_map],
      identity
        ? [null, null]
        : [
            maps((i) => startToOriginal(a.normalized, i)),
            maps((i) => endToOriginal(a.normalized, i)),
          ],
    ],
    ["input_ids", t.ids, a.tokens.ids],
    ["attention_mask", t.attention_mask, a.attentionMask],
    ["token_offsets", t.offsets, a.tokens.offsets],
    ["special_tokens_mask", t.special, a.tokens.specialTokensMask],
    ["truncated", e.truncated, a.tokens.truncated],
    ["target_bio_tags", t.target_tags, a.targetTags],
    ["type", e.type, p.type],
    ["target", e.target, p.target],
    ["target_span", e.target_span, p.target_span],
    ["original_offsets", t.original_offsets, a.originalOffsets],
    ["model_version", e.model_version, p.model_version],
    [
      "no_value",
      [null, null, null],
      [p.value_text, p.value_span, p.value_confidence],
    ],
  ]
}

const confidenceDelta = (a: Analysis, e: Prediction): number =>
  Math.max(
    Math.abs(a.prediction.type_confidence - e.type_confidence),
    Math.abs(a.prediction.target_confidence - e.target_confidence)
  )

async function main(): Promise<number> {
  const meta = JSON.parse(
    readFileSync(resolve(parityDir, "meta.json"), "utf8")
  ) as {
    bundle: string
    files: Record<string, string>
    inputs: number
    sweep: number
  }
  const bundle = resolve(root, meta.bundle)
  const configBytes = readFileSync(resolve(bundle, "config.json"))
  const tokenizerBytes = readFileSync(resolve(bundle, "tokenizer.json"))
  const modelBytes = readFileSync(resolve(bundle, "model.int8.onnx"))
  const hashes = {
    "config.json": sha256(configBytes),
    "tokenizer.json": sha256(tokenizerBytes),
    "model.int8.onnx": sha256(modelBytes),
  }
  if (!same(hashes, meta.files)) {
    console.error(
      "bundle files differ from the ones the dump was made from; rerun the dump"
    )
    return 1
  }
  const config = parseConfig(JSON.parse(configBytes.toString("utf8")))
  if (config.valueSource !== "none") {
    console.error("the English bundle must declare no value")
    return 1
  }
  const tokenizer = new BundleTokenizer(
    JSON.parse(tokenizerBytes.toString("utf8")),
    config.maxLength
  )

  // Tokenizer alone over the code-point sweep.
  let sweepBad = 0
  const sweep = readJsonl<Sweep>(resolve(parityDir, "sweep.jsonl"))
  for (const rec of sweep) {
    const normalized = normalizeNfc(rec.text)
    const tokens = tokenizer.encode(normalized.cps)
    const n = normalized.cps.length
    const maps =
      rec.start_map === undefined
        ? normalized.startMap === null
        : same(
            Array.from({ length: n + 1 }, (_, i) =>
              startToOriginal(normalized, i)
            ),
            rec.start_map
          ) &&
          same(
            Array.from({ length: n + 1 }, (_, i) =>
              endToOriginal(normalized, i)
            ),
            rec.end_map
          )
    if (
      fromCodePoints(normalized.cps) !== rec.nfc ||
      !same(tokens.ids, rec.ids) ||
      !same(tokens.offsets, rec.offsets) ||
      tokens.truncated !== rec.truncated ||
      !maps
    ) {
      sweepBad++
      if (sweepBad <= 5)
        console.error("sweep mismatch:", JSON.stringify(rec.text))
    }
  }

  ort.env.wasm.numThreads = 1
  ort.env.wasm.proxy = false
  const runner = await createOrtRunner(ort, modelBytes, config)

  const records = readJsonl<Case>(resolve(parityDir, "expected.jsonl"))
  const bySource: Record<
    string,
    { n: number; bad: number; refBad: number; logit: number; conf: number }
  > = {}
  const worst: { delta: number; label: string }[] = []
  const failing: Record<string, number> = {}
  let predictions = 0
  let errors = 0
  let logitDelta = 0
  let confDelta = 0
  let confDeltaRef = 0
  let confBad = 0
  let ortFlips = 0
  const flipList: string[] = []
  const shown: string[] = []

  for (const rec of records) {
    const row = (bySource[rec.source] ??= {
      n: 0,
      bad: 0,
      refBad: 0,
      logit: 0,
      conf: 0,
    })
    row.n++
    if (rec.expected === undefined || rec.trace === undefined) {
      errors++
      let name = "prediction"
      try {
        tokenize(tokenizer, rec.text)
      } catch (e) {
        name = e instanceof Error ? e.name : String(e)
      }
      if (name !== rec.error) {
        row.bad++
        failing.expected_error = (failing.expected_error ?? 0) + 1
      }
      continue
    }
    predictions++
    const stage = tokenize(tokenizer, rec.text)
    const logits = await runner.run(stage.tokens.ids, stage.attentionMask)
    const a = decodeLogits(config, stage, logits)
    const t = rec.trace
    const e = rec.expected
    const all = stages(a, t, e).filter(([, py, ts]) => !same(py, ts))
    const bad = all.filter(([name]) => INPUT_STAGES.has(name))
    const flips = all.filter(([name]) => !INPUT_STAGES.has(name))
    for (const [name] of bad) failing[name] = (failing[name] ?? 0) + 1
    if (flips.length > 0) {
      ortFlips++
      if (flipList.length < 6)
        flipList.push(
          `${rec.source}/${rec.id} ${JSON.stringify(rec.text.slice(0, 70))}: ${flips.map(([n]) => n).join(",")}`
        )
    }
    const delta = Math.max(
      maxAbs(a.logits.typeLogits, t.type_logits),
      maxAbs(a.logits.tagLogits, t.tag_logits.flat())
    )
    row.logit = Math.max(row.logit, delta)
    if (Number.isFinite(delta)) logitDelta = Math.max(logitDelta, delta)
    const conf = confidenceDelta(a, e)
    row.conf = Math.max(row.conf, conf)
    worst.push({
      delta,
      label: `${rec.source}/${rec.id} ${JSON.stringify(rec.text.slice(0, 60))}`,
    })
    confDelta = Math.max(confDelta, conf)
    if (conf > CONFIDENCE_TOLERANCE) confBad++
    if (bad.length > 0) {
      row.bad++
      if (shown.length < 8)
        shown.push(
          `${rec.source}/${rec.id} ${JSON.stringify(rec.text)}: ${bad.map(([n]) => n).join(",")}`
        )
    }

    // The same decoder on the Python reference logits: port errors only, no ORT numerics.
    const reference: ModelOutputs = {
      typeLogits: Float32Array.from(t.type_logits),
      tagLogits: Float32Array.from(t.tag_logits.flat()),
    }
    const onRef = decodeLogits(config, stage, reference)
    const refConf = confidenceDelta(onRef, e)
    confDeltaRef = Math.max(confDeltaRef, refConf)
    const refBad =
      stages(onRef, t, e).some(([, py, ts]) => !same(py, ts)) ||
      refConf > CONFIDENCE_TOLERANCE
    if (refBad) {
      row.refBad++
      failing.decode_on_reference_logits =
        (failing.decode_on_reference_logits ?? 0) + 1
    }
  }
  await runner.dispose()

  const failures =
    sweepBad + Object.values(bySource).reduce((s, r) => s + r.bad + r.refBad, 0)
  console.log(
    `gidi-en-v2 web parity  (sweep ${sweep.length}, notes ${records.length})`
  )
  console.log(
    `  sweep strings: ${sweep.length - sweepBad}/${sweep.length} identical tokenization`
  )
  for (const [source, r] of Object.entries(bySource)) {
    console.log(
      `  ${source.padEnd(16)} ${String(r.n).padStart(5)} notes  mismatches ${r.bad}  on-reference-logits ${r.refBad}  ` +
        `max logit delta ${r.logit.toExponential(1)}  max confidence delta ${r.conf.toExponential(1)}`
    )
  }
  worst.sort(
    (x, y) =>
      (Number.isFinite(y.delta) ? y.delta : 1e9) -
      (Number.isFinite(x.delta) ? x.delta : 1e9)
  )
  for (const w of worst.slice(0, 5))
    console.log(`  worst logit delta ${w.delta.toExponential(2)}  ${w.label}`)
  console.log(
    `  predictions ${predictions}, expected errors ${errors}; max |logit delta| ${logitDelta.toExponential(2)}, ` +
      `max confidence delta ${confDelta.toExponential(2)} (on reference logits ${confDeltaRef.toExponential(2)}), ` +
      `over 1e-4: ${confBad}`
  )
  console.log(
    `  ORT numerics (reported, not gated): ${ortFlips} of ${predictions} notes decide differently ` +
      `(type / target / spans), ${confBad} confidences move by more than 1e-4`
  )
  for (const line of flipList) console.log(`    flip ${line}`)
  for (const line of shown) console.log(`  MISMATCH ${line}`)
  if (Object.keys(failing).length > 0)
    console.log("  failing stages:", JSON.stringify(failing))
  console.log(failures === 0 ? "PASS" : `FAIL (${failures})`)
  return failures === 0 ? 0 : 1
}

process.exitCode = await main()
