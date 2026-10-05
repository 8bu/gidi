/**
 * Parity gate: the browser runtime (onnxruntime-web wasm EP, same code as the browser) against the
 * Python INT8 runtime of the gidi-finance-v3 bundle, stage by stage, on every input of
 * `scripts/dump_web_parity.py`.
 *
 *   uv run python scripts/dump_web_parity.py && pnpm parity
 *
 * Per case, EXACT: NFC text + offset maps, input_ids, attention_mask, token offsets (code points),
 * special-token mask, truncated, target BIO argmax tags, type, target text and span (after the
 * whole-word snap), value text and span (rule parser), per-token offsets in the original string.
 * NUMERIC: max |delta| of the two logit tensors and |delta| <= 1e-4 of the two confidences.
 *
 * Every case is decoded twice by the TS code: from the onnxruntime-web logits (the real pipeline)
 * and from the Python reference logits (isolates the ported tokenizer/decoder/parser from ORT
 * numerics). A mismatch is "float-tolerance" only if the second run is exact; otherwise it is
 * "semantic". The parser is also compared alone on every distinct corpus text (`parser.jsonl`).
 * Exit code 1 on any semantic mismatch. Writes experiments/deployment-v3/web-parity.{md,json}.
 */

import { createHash } from "node:crypto"
import { existsSync, readFileSync, statSync, writeFileSync } from "node:fs"
import { dirname, relative, resolve } from "node:path"
import { fileURLToPath } from "node:url"

import * as ort from "onnxruntime-web/wasm"

import { parseConfig } from "../src/runtime/config.ts"
import { parseValue } from "../src/runtime/value-parser.ts"
import { createOrtRunner } from "../src/runtime/ort-runner.ts"
import {
  decodeLogits,
  tokenize,
  type Analysis,
  type Tokenization,
  type ModelOutputs,
  type ModelRunner,
} from "../src/runtime/predictor.ts"
import {
  endToOriginal,
  foldCodePoint,
  fromCodePoints,
  isAlnum,
  isAlpha,
  isCombining,
  isPunct,
  isPySpace,
  isUpper,
  normalizeNfc,
  startToOriginal,
} from "../src/runtime/text.ts"
import { BundleTokenizer } from "../src/runtime/tokenizer.ts"
import type { Prediction } from "../src/runtime/types.ts"

const here = dirname(fileURLToPath(import.meta.url))
const root = resolve(here, "../..")
const parityDir = resolve(here, "../.parity")
const reportDir = resolve(root, "experiments/deployment-v3")
const CONFIDENCE_TOLERANCE = 1e-4
const CONFIDENCE_FIELDS = ["type_confidence", "target_confidence"] as const
const REQUIRED_COVERAGE = [
  "accented",
  "unaccented",
  "bare_number",
  "slang",
  "multi_number",
  "long_32_tokens",
  "truncated",
  "punctuation",
  "null_target",
  "null_value",
] as const

interface Meta {
  bundle: string
  files: Record<string, string>
  python: string
  unicode: string
  versions: Record<string, string>
  inputs: number
  sweep: number
  parser_texts: number
  unicode_ranges: {
    combining: number[]
    space: number[]
    alpha: number[]
    alnum: number[]
    upper: number[]
    punct: number[]
  }
  fold_pairs: [number, number][]
}

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

interface CaseRecord {
  source: string
  id: string
  text: string
  tags: string[]
  expected?: Prediction
  error?: string
  trace?: Trace
  default?: { expected: Prediction; type_logits: number[] }
}

interface SweepRecord {
  text: string
  nfc: string
  ids: number[]
  offsets: [number, number][]
  truncated: boolean
  start_map?: number[]
  end_map?: number[]
}

interface Stage {
  name: string
  python: unknown
  ts: unknown
}

type Status = "exact" | "float_tolerance" | "semantic"

const readJsonl = <T>(path: string): T[] =>
  readFileSync(path, "utf8")
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line) as T)

const sha256 = (bytes: Uint8Array): string =>
  createHash("sha256").update(bytes).digest("hex")
const same = (a: unknown, b: unknown): boolean =>
  JSON.stringify(a) === JSON.stringify(b)
const flat = (rows: number[][]): number[] => rows.flat()

function maxAbs(a: ArrayLike<number>, b: ArrayLike<number>): number {
  if (a.length !== b.length) return Infinity
  let max = 0
  for (let i = 0; i < a.length; i++) max = Math.max(max, Math.abs(a[i] - b[i]))
  return max
}

function rangesOf(predicate: (cp: number) => boolean): number[] {
  const out: number[] = []
  let start = -1
  for (let cp = 0; cp <= 0x110000; cp++) {
    const hit = cp <= 0x10ffff && predicate(cp)
    if (hit && start < 0) start = cp
    else if (!hit && start >= 0) {
      out.push(start, cp - 1)
      start = -1
    }
  }
  return out
}

/** Stages decided before the model runs: independent of onnxruntime. */
function inputStages(a: Analysis, t: Trace, expected: Prediction): Stage[] {
  const n = a.normalized.cps.length
  const maps = (get: (i: number) => number, ref: number[] | null) =>
    ref === null ? null : Array.from({ length: n + 1 }, (_, i) => get(i))
  const identity = a.normalized.startMap === null
  return [
    { name: "nfc_text", python: t.nfc, ts: fromCodePoints(a.normalized.cps) },
    {
      name: "nfc_offset_map",
      python: { start: t.nfc_start_map, end: t.nfc_end_map },
      ts: identity
        ? { start: null, end: null }
        : {
            start: maps(
              (i) => startToOriginal(a.normalized, i),
              t.nfc_start_map ?? []
            ),
            end: maps(
              (i) => endToOriginal(a.normalized, i),
              t.nfc_end_map ?? []
            ),
          },
    },
    { name: "input_ids", python: t.ids, ts: a.tokens.ids },
    { name: "attention_mask", python: t.attention_mask, ts: a.attentionMask },
    { name: "token_offsets", python: t.offsets, ts: a.tokens.offsets },
    {
      name: "special_tokens_mask",
      python: t.special,
      ts: a.tokens.specialTokensMask,
    },
    { name: "truncated", python: expected.truncated, ts: a.tokens.truncated },
  ]
}

/** Stages decided after the model from the logits. */
function decodeStages(a: Analysis, t: Trace, e: Prediction): Stage[] {
  const p = a.prediction
  return [
    { name: "target_bio_tags", python: t.target_tags, ts: a.targetTags },
    { name: "type", python: e.type, ts: p.type },
    { name: "target", python: e.target, ts: p.target },
    { name: "target_span", python: e.target_span, ts: p.target_span },
    { name: "value_text", python: e.value_text, ts: p.value_text },
    { name: "value_span", python: e.value_span, ts: p.value_span },
    {
      name: "value_confidence",
      python: e.value_confidence,
      ts: p.value_confidence,
    },
    {
      name: "original_offsets",
      python: t.original_offsets,
      ts: a.originalOffsets,
    },
    { name: "model_version", python: e.model_version, ts: p.model_version },
  ]
}

const firstMismatch = (stages: Stage[]): Stage | undefined =>
  stages.find((s) => !same(s.python, s.ts))

const confidenceDeltas = (a: Analysis, e: Prediction): Record<string, number> =>
  Object.fromEntries(
    CONFIDENCE_FIELDS.map((f) => [f, Math.abs(a.prediction[f] - e[f])])
  )

const maxOf = (values: number[]): number =>
  values.reduce((m, v) => Math.max(m, v), 0)

function checkSweep(tokenizer: BundleTokenizer, records: SweepRecord[]) {
  const failures: { text: string; stages: string[] }[] = []
  for (const rec of records) {
    const normalized = normalizeNfc(rec.text)
    const tokens = tokenizer.encode(normalized.cps)
    const bad: string[] = []
    if (fromCodePoints(normalized.cps) !== rec.nfc) bad.push("nfc_text")
    if (!same(tokens.ids, rec.ids)) bad.push("input_ids")
    if (!same(tokens.offsets, rec.offsets)) bad.push("token_offsets")
    if (tokens.truncated !== rec.truncated) bad.push("truncated")
    const n = normalized.cps.length
    if (rec.start_map !== undefined) {
      const starts = Array.from({ length: n + 1 }, (_, i) =>
        startToOriginal(normalized, i)
      )
      const ends = Array.from({ length: n + 1 }, (_, i) =>
        endToOriginal(normalized, i)
      )
      if (!same(starts, rec.start_map) || !same(ends, rec.end_map))
        bad.push("nfc_offset_map")
    } else if (normalized.startMap !== null) bad.push("nfc_offset_map")
    if (bad.length > 0) failures.push({ text: rec.text, stages: bad })
  }
  return failures
}

/** The code points `foldCodePoint` changes, flat `[cp, folded]` pairs (cf. `fold_pairs`). */
function foldPairs(): number[] {
  const out: number[] = []
  for (let cp = 0; cp <= 0x10ffff; cp++) {
    const folded = foldCodePoint(cp)
    if (folded !== cp) out.push(cp, folded)
  }
  return out
}

interface ParserRecord {
  text: string
  value: [number, number] | null
}

/** `parseValue` alone against `gidi.value_parser.parse_value` on every distinct corpus text. */
function checkParser(records: ParserRecord[]) {
  const failures: { text: string; python: unknown; ts: unknown }[] = []
  let withValue = 0
  for (const rec of records) {
    const value = parseValue(rec.text)
    const ts = value === null ? null : [value.start, value.end]
    if (rec.value !== null) withValue++
    const textOk =
      value === null ||
      value.text ===
        fromCodePoints(normalizeNfc(rec.text).original, value.start, value.end)
    if (!same(ts, rec.value) || !textOk)
      failures.push({ text: rec.text, python: rec.value, ts })
  }
  return { texts: records.length, with_value: withValue, failures }
}

function ortEnvironment(releaseVersionNote: string) {
  const resolveFile = (specifier: string): { path: string; bytes: number } => {
    const path = fileURLToPath(import.meta.resolve(specifier))
    return { path: relative(root, path), bytes: statSync(path).size }
  }
  const pkg = JSON.parse(
    readFileSync(
      resolve(here, "../node_modules/onnxruntime-web/package.json"),
      "utf8"
    )
  ) as { version: string }
  return {
    onnxruntime_web: pkg.version,
    entry: resolveFile("onnxruntime-web/wasm"),
    wasm_flavour:
      "ort-wasm-simd-threaded.wasm (CPU/WASM execution provider, SIMD build, no JSEP/WebGPU/WebNN)",
    wasm_binary: resolveFile("onnxruntime-web/ort-wasm-simd-threaded.wasm"),
    num_threads: ort.env.wasm.numThreads,
    simd:
      "simd" in ort.env.wasm
        ? ort.env.wasm.simd
        : "always on in this build (no flag in 1.30)",
    proxy: ort.env.wasm.proxy,
    graph_optimization_level:
      "all requested; logits are identical at disabled/basic/extended/layout/all in this wasm build (no fused DynamicQuantizeMatMul), i.e. the graph runs as written",
    tokenizer_and_crf: `TypeScript (src/runtime), same thread as the ORT session: ${releaseVersionNote}`,
    node: process.version,
  }
}

async function main(): Promise<number> {
  const meta = JSON.parse(
    readFileSync(resolve(parityDir, "meta.json"), "utf8")
  ) as Meta
  const bundle = resolve(root, meta.bundle)
  const tokenizerBytes = readFileSync(resolve(bundle, "tokenizer.json"))
  const configBytes = readFileSync(resolve(bundle, "config.json"))
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
  const tokenizer = new BundleTokenizer(
    JSON.parse(tokenizerBytes.toString("utf8")),
    config.maxLength
  )

  // --- Unicode tables and the code-point sweep (no model) --------------------------------------
  const tableFailures = [
    ...(same(rangesOf(isCombining), meta.unicode_ranges.combining)
      ? []
      : ["COMBINING_RANGES differ from unicodedata.combining"]),
    ...(same(rangesOf(isPySpace), meta.unicode_ranges.space)
      ? []
      : ["SPACE_RANGES differ from str.isspace"]),
    ...(same(rangesOf(isAlpha), meta.unicode_ranges.alpha)
      ? []
      : ["ALPHA_RANGES differ from str.isalpha"]),
    ...(same(rangesOf(isAlnum), meta.unicode_ranges.alnum)
      ? []
      : ["ALNUM_RANGES differ from str.isalnum"]),
    ...(same(rangesOf(isUpper), meta.unicode_ranges.upper)
      ? []
      : ["UPPER_RANGES differ from str.isupper"]),
    ...(same(rangesOf(isPunct), meta.unicode_ranges.punct)
      ? []
      : ["PUNCT_RANGES differ from unicodedata.category P*"]),
    ...(same(foldPairs(), meta.fold_pairs.flat())
      ? []
      : ["FOLD_PAIRS differ from gidi.value_parser.fold"]),
  ]
  const sweep = readJsonl<SweepRecord>(resolve(parityDir, "sweep.jsonl"))
  const sweepFailures = checkSweep(tokenizer, sweep)
  const parser = checkParser(
    readJsonl<ParserRecord>(resolve(parityDir, "parser.jsonl"))
  )

  // --- Model cases ------------------------------------------------------------------------------
  ort.env.wasm.numThreads = 1
  ort.env.wasm.proxy = false
  const inner = await createOrtRunner(ort, modelBytes, config)
  const runner: ModelRunner = inner

  const records = readJsonl<CaseRecord>(resolve(parityDir, "expected.jsonl"))
  const stageCounts: Record<string, { exact: number; total: number }> = {}
  const tally = (name: string, ok: boolean): void => {
    const entry = (stageCounts[name] ??= { exact: 0, total: 0 })
    entry.total++
    if (ok) entry.exact++
  }
  const maxDelta = {
    type_logits: 0,
    tag_logits: 0,
    type_confidence: 0,
    target_confidence: 0,
  }
  const maxDeltaOnReferenceLogits = {
    type_confidence: 0,
    target_confidence: 0,
  }
  let withinTolerance = 0
  let decodeOnReferenceExact = 0
  let predictions = 0
  let expectedErrors = 0
  const bySource: Record<string, Record<Status | "n", number>> = {}
  const coverage: Record<string, Record<string, number>> = {}
  const sourceDeltas: Record<string, { logits: number; confidence: number }> =
    {}
  const nodePredictions: Record<string, Prediction> = {}
  const cases: unknown[] = []
  const incidents: unknown[] = []
  const vsDefault = {
    inputsDiffering: 0,
    fieldFlips: { type: 0, target_span: 0, value_span: 0 } as Record<
      string,
      number
    >,
    maxDelta: { type_confidence: 0, target_confidence: 0 },
    bySource: {} as Record<
      string,
      {
        n: number
        differing: number
        type: number
        target_span: number
        value_span: number
      }
    >,
    cases: [] as unknown[],
  }
  const presetCases: unknown[] = []
  let presetsIdentical = 0
  const presetDelta = { web_vs_default: 0, web_vs_basic: 0 }
  const webLogits: Record<string, { type: number[]; tag: number[] }> = {}

  for (const rec of records) {
    const stats = (bySource[rec.source] ??= {
      n: 0,
      exact: 0,
      float_tolerance: 0,
      semantic: 0,
    })
    stats.n++
    for (const tag of rec.tags) {
      const row = (coverage[tag] ??= {})
      row[rec.source] = (row[rec.source] ?? 0) + 1
      row.total = (row.total ?? 0) + 1
    }

    let analysis: Analysis | null = null
    let error: string | null = null
    let stage: Tokenization | null = null
    try {
      stage = tokenize(tokenizer, rec.text)
      const logits = await runner.run(stage.tokens.ids, stage.attentionMask)
      analysis = decodeLogits(config, stage, logits)
    } catch (e) {
      error = e instanceof Error ? e.name : String(e)
    }

    if (
      rec.error !== undefined ||
      rec.expected === undefined ||
      rec.trace === undefined
    ) {
      expectedErrors++
      const ok = error === rec.error
      tally("expected_error", ok)
      if (ok) stats.exact++
      else {
        stats.semantic++
        incidents.push({
          source: rec.source,
          id: rec.id,
          text: rec.text,
          status: "semantic",
          first_divergent_stage: "empty_input_error",
          python: rec.error ?? "prediction",
          ts: error ?? "prediction",
        })
      }
      cases.push({
        source: rec.source,
        id: rec.id,
        status: ok ? "exact" : "semantic",
        error: rec.error,
      })
      continue
    }
    if (analysis === null || stage === null) {
      stats.semantic++
      incidents.push({
        source: rec.source,
        id: rec.id,
        text: rec.text,
        status: "semantic",
        first_divergent_stage: "ts_pipeline",
        python: "prediction",
        ts: `unexpected error ${error}`,
      })
      continue
    }
    predictions++
    const t = rec.trace
    const e = rec.expected

    const ins = inputStages(analysis, t, e)
    const dec = decodeStages(analysis, t, e)
    for (const s of [...ins, ...dec]) tally(s.name, same(s.python, s.ts))

    const logitDeltas = {
      type_logits: maxAbs(analysis.logits.typeLogits, t.type_logits),
      tag_logits: maxAbs(analysis.logits.tagLogits, flat(t.tag_logits)),
    }
    const conf = confidenceDeltas(analysis, e)
    for (const [k, v] of Object.entries(logitDeltas)) {
      maxDelta[k as keyof typeof maxDelta] = Math.max(
        maxDelta[k as keyof typeof maxDelta],
        v
      )
    }
    for (const f of CONFIDENCE_FIELDS)
      maxDelta[f] = Math.max(maxDelta[f], conf[f])
    const confOk = maxOf(Object.values(conf)) <= CONFIDENCE_TOLERANCE
    tally("confidences_within_1e-4", confOk)
    if (confOk) withinTolerance++
    const sd = (sourceDeltas[rec.source] ??= { logits: 0, confidence: 0 })
    sd.logits = Math.max(sd.logits, ...Object.values(logitDeltas))
    sd.confidence = Math.max(sd.confidence, ...Object.values(conf))
    nodePredictions[`${rec.source}/${rec.id}`] = analysis.prediction
    if (rec.source === "test" || rec.source === "probe") {
      webLogits[`${rec.source}/${rec.id}`] = {
        type: Array.from(analysis.logits.typeLogits),
        tag: Array.from(analysis.logits.tagLogits),
      }
    }

    // Same TS decoder on the Python reference logits: isolates the port from ORT numerics.
    const reference: ModelOutputs = {
      typeLogits: Float32Array.from(t.type_logits),
      tagLogits: Float32Array.from(flat(t.tag_logits)),
    }
    const onReference = decodeLogits(config, stage, reference)
    const decOnReference = decodeStages(onReference, t, e)
    const confOnReference = confidenceDeltas(onReference, e)
    for (const f of CONFIDENCE_FIELDS) {
      maxDeltaOnReferenceLogits[f] = Math.max(
        maxDeltaOnReferenceLogits[f],
        confOnReference[f]
      )
    }
    const refOk =
      firstMismatch(decOnReference) === undefined &&
      maxOf(Object.values(confOnReference)) <= CONFIDENCE_TOLERANCE
    tally("decode_on_reference_logits", refOk)
    if (refOk) decodeOnReferenceExact++

    const inputBad = firstMismatch(ins)
    const decodeBad = firstMismatch(dec)
    const exactOk = inputBad === undefined && decodeBad === undefined
    let status: Status
    if (exactOk && confOk) status = "exact"
    else if (inputBad !== undefined || !refOk) status = "semantic"
    else status = "float_tolerance"
    stats[status]++

    if (rec.default) {
      const d = rec.default.expected
      const p = analysis.prediction
      if (rec.source === "preset") {
        const same3 = (["type", "target_span", "value_span"] as const).every(
          (f) => same(p[f], d[f]) && same(p[f], e[f])
        )
        if (same3) presetsIdentical++
        for (const f of CONFIDENCE_FIELDS) {
          presetDelta.web_vs_default = Math.max(
            presetDelta.web_vs_default,
            Math.abs(p[f] - d[f])
          )
          presetDelta.web_vs_basic = Math.max(
            presetDelta.web_vs_basic,
            Math.abs(p[f] - e[f])
          )
        }
        presetCases.push({
          text: rec.text,
          web: p,
          python_default: d,
          python_basic: e,
          identical: same3,
        })
      }
      const differs = (["type", "target_span", "value_span"] as const).filter(
        (f) => !same(p[f], d[f])
      )
      const row = (vsDefault.bySource[rec.source] ??= {
        n: 0,
        differing: 0,
        type: 0,
        target_span: 0,
        value_span: 0,
      })
      row.n++
      if (differs.length > 0) {
        vsDefault.inputsDiffering++
        row.differing++
        const brief = (x: Prediction) => ({
          type: x.type,
          type_confidence: x.type_confidence,
          target: x.target,
          target_span: x.target_span,
          target_confidence: x.target_confidence,
          value_text: x.value_text,
          value_span: x.value_span,
          value_confidence: x.value_confidence,
        })
        vsDefault.cases.push({
          source: rec.source,
          id: rec.id,
          text: rec.text,
          fields: differs,
          python_default: brief(d),
          python_basic: brief(e),
          web: brief(p),
        })
      }
      for (const f of differs) {
        vsDefault.fieldFlips[f]++
        row[f]++
      }
      for (const f of CONFIDENCE_FIELDS) {
        vsDefault.maxDelta[f] = Math.max(
          vsDefault.maxDelta[f],
          Math.abs(p[f] - d[f])
        )
      }
    }

    cases.push({
      source: rec.source,
      id: rec.id,
      status,
      tags: rec.tags,
      max_logit_delta: logitDeltas,
      max_confidence_delta: maxOf(Object.values(conf)),
    })

    if (
      status !== "exact" &&
      (status === "semantic" || decodeBad !== undefined)
    ) {
      const first = inputBad ?? decodeBad ?? undefined
      incidents.push({
        source: rec.source,
        id: rec.id,
        text: rec.text,
        status,
        first_divergent_stage: first?.name ?? "confidences",
        python: first?.python ?? e,
        ts: first?.ts ?? analysis.prediction,
        classification:
          status === "semantic"
            ? "semantic: ORT-independent stage differs, or TS decode of the Python logits differs"
            : "float-tolerance: TS decode of the Python reference logits is exact; ORT logits differ " +
              `(max |logit delta| type ${logitDeltas.type_logits.toExponential(2)}, tag ` +
              `${logitDeltas.tag_logits.toExponential(2)})`,
        expected: e,
        actual: analysis.prediction,
        logits: {
          type: {
            python: t.type_logits,
            ts: Array.from(analysis.logits.typeLogits),
          },
          tag: {
            python: t.tag_logits,
            ts: Array.from(analysis.logits.tagLogits),
          },
        },
      })
    }
  }

  const statuses = Object.values(bySource).reduce(
    (acc, s) => ({
      exact: acc.exact + s.exact,
      float_tolerance: acc.float_tolerance + s.float_tolerance,
      semantic: acc.semantic + s.semantic,
    }),
    { exact: 0, float_tolerance: 0, semantic: 0 }
  )
  const confidenceOnlyFloat =
    statuses.float_tolerance -
    incidents.filter(
      (i) => (i as { status: string }).status === "float_tolerance"
    ).length

  const environment = ortEnvironment(
    "Node main thread in this runner; dedicated Web Worker in the browser"
  )
  const sidecar = (name: string): unknown => {
    const path = resolve(parityDir, name)
    return existsSync(path) ? JSON.parse(readFileSync(path, "utf8")) : null
  }
  const browser = sidecar("browser.json")

  const report = {
    generated_by:
      "playground/scripts/parity.ts (pnpm parity) from scripts/dump_web_parity.py",
    bundle: { path: meta.bundle, files: meta.files },
    reference: {
      description:
        "Python gidi.inference GidiPredictor on the gidi-finance-v3 INT8 model (whole-word target " +
        "snap, value from gidi.value_parser); ORT session at " +
        "ORT_ENABLE_BASIC (graph as written). The default-level session is compared separately.",
      python: meta.python,
      unicode: meta.unicode,
      versions: meta.versions,
    },
    environment,
    tolerance: {
      confidence_abs: CONFIDENCE_TOLERANCE,
      logits: "reported only, not gated",
    },
    totals: {
      cases: records.length,
      predictions,
      expected_empty_errors: expectedErrors,
      by_status: statuses,
      confidence_only_float_cases: confidenceOnlyFloat,
      confidences_within_1e4: withinTolerance,
    },
    stage_exact_counts: stageCounts,
    max_abs_delta: maxDelta,
    ts_decode_on_python_logits: {
      exact_cases: decodeOnReferenceExact,
      of: predictions,
      max_confidence_delta: maxDeltaOnReferenceLogits,
    },
    by_source: bySource,
    max_delta_by_source: sourceDeltas,
    coverage: {
      required: REQUIRED_COVERAGE,
      counts: coverage,
    },
    code_point_sweep: {
      strings: sweep.length,
      mismatches: sweepFailures.length,
      stages: "nfc_text, nfc_offset_map, input_ids, token_offsets, truncated",
      failures: sweepFailures.slice(0, 50),
    },
    unicode_tables: { failures: tableFailures },
    parser_only: {
      texts: parser.texts,
      with_value: parser.with_value,
      mismatches: parser.failures.length,
      failures: parser.failures.slice(0, 50),
    },
    vs_python_default_session: {
      note:
        "Informational. ORT's default level fuses DynamicQuantizeMatMul/MatMulIntegerToFloat; its " +
        "arm64 native kernel (KleidiAI) quantizes activations differently from the ONNX graph " +
        "semantics that onnxruntime-web executes.",
      ...vsDefault,
    },
    browser_run: browser,
    presets: {
      identical_all_three: presetsIdentical,
      max_confidence_delta_web_vs_default: presetDelta.web_vs_default,
      max_confidence_delta_web_vs_basic: presetDelta.web_vs_basic,
      cases: presetCases,
    },
    incidents,
    cases,
  }
  writeFileSync(
    resolve(parityDir, "web-logits.json"),
    JSON.stringify(webLogits)
  )
  writeFileSync(
    resolve(parityDir, "node-predictions.json"),
    JSON.stringify(nodePredictions)
  )
  writeFileSync(
    resolve(reportDir, "web-parity.json"),
    JSON.stringify(report, null, 1) + "\n"
  )
  writeFileSync(
    resolve(reportDir, "web-parity.md"),
    renderMarkdown(report, meta)
  )

  console.log(
    `inputs ${records.length}: ${predictions} predictions, ${expectedErrors} expected errors`
  )
  console.log(
    `sweep ${sweep.length}: ${sweepFailures.length} mismatches; tables: ${tableFailures.length}`
  )
  console.log(
    `parser-only ${parser.texts} texts (${parser.with_value} with a value): ${parser.failures.length} mismatches`
  )
  for (const [name, c] of Object.entries(stageCounts)) {
    console.log(`  ${name.padEnd(28)} ${c.exact}/${c.total}`)
  }
  console.log(JSON.stringify({ statuses, maxDelta }))
  await runner.dispose()

  const failed =
    statuses.semantic > 0 ||
    sweepFailures.length > 0 ||
    tableFailures.length > 0 ||
    parser.failures.length > 0
  console.log(
    failed ? "\nFAIL (semantic mismatch)" : "\nPARITY: no semantic mismatch"
  )
  console.log(`report: ${relative(root, resolve(reportDir, "web-parity.md"))}`)
  return failed ? 1 : 0
}

type Report = Record<string, any> // eslint-disable-line @typescript-eslint/no-explicit-any

function renderMarkdown(report: Report, meta: Meta): string {
  const fmt = (n: number): string => (n === 0 ? "0" : n.toExponential(2))
  const lines: string[] = []
  const t = report.totals
  lines.push(
    "# gidi-finance-v3 3.0.0: in-browser runtime parity",
    "",
    "Generated by `pnpm -C playground parity` (`playground/scripts/parity.ts`) from",
    "`scripts/dump_web_parity.py`. Machine-readable twin: `web-parity.json`. Regenerate with",
    "`uv run python scripts/dump_web_parity.py && pnpm -C playground parity`.",
    "",
    "## Verdict",
    "",
    `- ${t.cases} cases (${t.predictions} predictions, ${t.expected_empty_errors} expected ` +
      "`EmptyInputError`).",
    `- Semantic mismatches: **${t.by_status.semantic}**. Exact cases: ${t.by_status.exact}. ` +
      `Float-tolerance cases: ${t.by_status.float_tolerance} ` +
      `(${t.confidence_only_float_cases} differ only in a confidence beyond 1e-4, ` +
      `${t.by_status.float_tolerance - t.confidence_only_float_cases} flip a discrete field).`,
    `- TS tokenizer/NFC/decoder/snap/parser fed the Python reference logits: ` +
      `${report.ts_decode_on_python_logits.exact_cases}/${report.ts_decode_on_python_logits.of} ` +
      "cases exact in every decode stage, confidences within 1e-4.",
    `- Code-point sweep: ${report.code_point_sweep.strings} strings, ` +
      `${report.code_point_sweep.mismatches} mismatches (NFC text, offset map, ids, offsets, truncated).`,
    `- Value parser alone (\`gidi.value_parser\` vs the TS port): ${report.parser_only.texts} distinct ` +
      `corpus / dataset / fuzz texts (${report.parser_only.with_value} with a value), ` +
      `**${report.parser_only.mismatches} mismatches**.`,
    "- Reference = Python `GidiPredictor` on the gidi-finance-v3 INT8 bundle with the ORT session at " +
      "`ORT_ENABLE_BASIC`. See *ORT numerics* for why not the default level.",
    "",
    "## Tolerance",
    "",
    `Exact: NFC text, offset map, input_ids, attention_mask, token offsets (code points), special mask, ` +
      "truncated, target BIO tags, type, target text and span (after the word snap), value text and span (rule parser), value_confidence (null), per-token " +
      `original-string offsets. Confidences: |delta| <= ${report.tolerance.confidence_abs}. ` +
      "Logits: max |delta| reported, not gated.",
    "",
    "## Stage results (exact-match counts)",
    "",
    "| stage | exact | total |",
    "|---|---:|---:|"
  )
  for (const [name, c] of Object.entries(report.stage_exact_counts as Report)) {
    lines.push(`| ${name} | ${c.exact} | ${c.total} |`)
  }
  const m = report.max_abs_delta
  lines.push(
    "",
    "## Numeric deltas (max |delta| over all predictions, onnxruntime-web vs Python)",
    "",
    "| tensor / field | max abs delta |",
    "|---|---:|",
    `| type_logits | ${fmt(m.type_logits)} |`,
    `| tag_logits | ${fmt(m.tag_logits)} |`,
    `| type_confidence | ${fmt(m.type_confidence)} |`,
    `| target_confidence | ${fmt(m.target_confidence)} |`,
    "",
    `Cases with both confidences within 1e-4: ${t.confidences_within_1e4}/${t.predictions}. ` +
      "With the Python logits as input, the TS decoder's confidences differ by at most " +
      `${fmt(report.ts_decode_on_python_logits.max_confidence_delta.type_confidence)} (type), ` +
      `${fmt(report.ts_decode_on_python_logits.max_confidence_delta.target_confidence)} (target): ` +
      "the whole confidence gap is in the logits.",
    "",
    "## ORT numerics (why float differences exist)",
    "",
    "The first operator that differs is the first `LayerNormalization` (about 7e-7, float reduction " +
      "order in the wasm build vs native). The dynamic activation quantization scale then differs by " +
      "~1e-8, a few int8 values round differently, and the difference compounds over 36 " +
      "`DynamicQuantizeLinear` layers into logit deltas up to the value above. Python's default " +
      "ORT session on arm64 additionally fuses `DynamicQuantizeMatMul` (KleidiAI, different activation " +
      "quantization): onnxruntime-web and the unfused graph agree far better with each other than " +
      "either does with that fused arm64 path. This noise is a property of INT8 dynamic quantization " +
      "across two ORT builds, not of the port; every decision flip is listed below with its logits.",
    "",
    "The web runtime also differs from Python's stock default-level session; those differences, " +
      "and which reference the deployed Python numbers correspond to " +
      "are in *Web vs the stock default-level Python session* below.",
    "",
    "## Coverage",
    "",
    "Cases per source:",
    "",
    "| source | cases | exact | float-tolerance | semantic |",
    "|---|---:|---:|---:|---:|"
  )
  for (const [source, s] of Object.entries(report.by_source as Report)) {
    lines.push(
      `| ${source} | ${s.n} | ${s.exact} | ${s.float_tolerance} | ${s.semantic} |`
    )
  }
  lines.push(
    "",
    "Max |delta| by source (onnxruntime-web vs Python): ",
    "",
    "| source | max logit delta | max confidence delta |",
    "|---|---:|---:|"
  )
  for (const [source, d] of Object.entries(
    report.max_delta_by_source as Report
  )) {
    lines.push(`| ${source} | ${fmt(d.logits)} | ${fmt(d.confidence)} |`)
  }
  const sources = Object.keys(report.by_source as Report)
  lines.push(
    "",
    "Sources: golden = `experiments/deployment-v2/golden-suite.jsonl`; robustness = " +
      "`robustness-inputs.jsonl`; hardening = `hardening-cases.jsonl`; test / validation / train = " +
      "`datasets/annotation-v2/training-v1/*.jsonl`; probe = `probe-v1-eval-only.jsonl`; human-value-01 / -02 = the review queues of " +
      "`datasets/annotation-v2/human-value-01` and `datasets/annotation-v3/human-value-02`; regression = " +
      "`tests/data/production-regressions.jsonl`; preset = `playground/src/lib/presets.ts`; stress = seeded " +
      "adversarial strings (special-token text, separators, whitespace variants, NFD/reordered " +
      "combining marks, astral, lone surrogates, long inputs).",
    "",
    "Case-kind coverage (simple text/output predicates, computed in `dump_web_parity.py`):",
    "",
    `| kind | total | ${sources.join(" | ")} |`,
    `|---|---:|${sources.map(() => "---:").join("|")}|`
  )
  const counts = report.coverage.counts as Report
  for (const kind of Object.keys(counts).sort()) {
    const required = (report.coverage.required as string[]).includes(kind)
      ? " (required)"
      : ""
    lines.push(
      `| ${kind}${required} | ${counts[kind].total} | ${sources.map((s) => counts[kind][s] ?? 0).join(" | ")} |`
    )
  }
  const missing = (report.coverage.required as string[]).filter(
    (k) => !counts[k]
  )
  lines.push(
    "",
    missing.length === 0
      ? "All required case kinds are present."
      : `MISSING case kinds: ${missing.join(", ")}.`,
    "",
    "Predicates: accented = has a combining mark after NFD or `đ`; unaccented = ASCII-only with " +
      "letters; bare_number = the whole note is one amount (e.g. `200k`, `1.500.000đ`); slang = a " +
      "token like `ck`, `xị`, `củ`, `1tr5`, `200k`; multi_number = two or more digit runs; " +
      "long_32_tokens = filled the 32-token window; truncated = model `truncated` flag; " +
      "null_target / null_value = the Python prediction has no target / value span; gold_null_* use " +
      "the dataset label where one exists.",
    "",
    "## Runtime environment",
    ""
  )
  const env = report.environment
  lines.push(
    `- onnxruntime-web ${env.onnxruntime_web}, entry \`${env.entry.path}\` (${env.entry.bytes} bytes)`,
    `- wasm: ${env.wasm_flavour}; binary \`${env.wasm_binary.path}\`, ${env.wasm_binary.bytes} bytes`,
    `- numThreads ${env.num_threads}, simd: ${env.simd}, proxy: ${env.proxy}`,
    `- graph optimization: ${env.graph_optimization_level}`,
    `- tokenizer, NFC, decoding, word snap and value parser: ${env.tokenizer_and_crf}`,
    `- Node ${env.node}; Python ${meta.python} (Unicode ${meta.unicode}), onnxruntime ` +
      `${meta.versions.onnxruntime}, tokenizers ${meta.versions.tokenizers}`,
    "- Browser build: Vite emits the same wasm as `assets/ort-wasm-simd-threaded-<hash>.wasm` " +
      "(14,239,897 bytes, self-hosted, no CDN); the JS glue is inlined in the worker chunk. " +
      "Inference, tokenizer and parser run in a dedicated Web Worker; `ort.env.wasm.proxy = false`, " +
      "`numThreads = 1` (no cross-origin isolation required).",
    ""
  )
  if (report.browser_run) {
    const b = report.browser_run as Report
    lines.push(
      "## Real-browser run",
      "",
      `Headless Chromium, production Vite build, \`loadPredictor\` (Web Worker) over all ` +
        `${b.cases} predictions: ${b.final_fields_equal_node}/${b.cases} identical to this Node run in ` +
        `type, target, target_span, value_text, value_span, truncated, confidences ` +
        `(max confidence delta browser vs Node ${fmt(b.max_confidence_delta_vs_node)}); ` +
        `${b.discrete_equal_python}/${b.cases} equal to the Python reference in those discrete fields. ` +
        `Chromium ${b.user_agent}.`,
      ""
    )
  }
  lines.push("## Mismatches", "")
  const incidents = report.incidents as Report[]
  if (incidents.length === 0) lines.push("None.", "")
  else {
    lines.push(
      `${incidents.length} case(s) differ in a stage other than a confidence-only delta. ` +
        'Each is decoded again from the Python logits; "float-tolerance" means that decode is exact.',
      ""
    )
    for (const inc of incidents) {
      lines.push(
        `### ${inc.source} / ${inc.id}`,
        "",
        `- input: \`${JSON.stringify(inc.text)}\``,
        `- first divergent stage: **${inc.first_divergent_stage}**`,
        `- python: \`${JSON.stringify(inc.python)}\``,
        `- ts: \`${JSON.stringify(inc.ts)}\``,
        `- classification: ${inc.classification ?? inc.status}`
      )
      if (inc.logits) {
        const rows = (v: number[][]): string =>
          JSON.stringify(v.map((r) => r.map((x) => +x.toFixed(3))))
        lines.push(
          `- type logits python: \`${JSON.stringify(inc.logits.type.python.map((x: number) => +x.toFixed(3)))}\``,
          `- type logits ts: \`${JSON.stringify(inc.logits.type.ts.map((x: number) => +x.toFixed(3)))}\``
        )
        if (
          inc.first_divergent_stage.endsWith("tags") ||
          inc.first_divergent_stage.startsWith("target")
        ) {
          lines.push(
            `- tag logits python: \`${rows(inc.logits.tag.python)}\``,
            `- tag logits ts: \`${rows(chunk3(inc.logits.tag.ts))}\``
          )
        }
      }
      lines.push("")
    }
  }
  lines.push(
    "Confidence-only deltas beyond 1e-4 are not itemised here; every case with its max logit and " +
      "confidence delta is in `web-parity.json` (`cases`).",
    ""
  )
  lines.push(...defaultSessionSection(report))
  return lines.join("\n")
}

function chunk3(values: number[]): number[][] {
  const rows: number[][] = []
  for (let i = 0; i < values.length; i += 3) rows.push(values.slice(i, i + 3))
  return rows
}

type Brief = {
  type: string
  target: string | null
  target_span: number[] | null
  value_text: string | null
  value_span: number[] | null
  type_confidence: number
}

function briefLine(b: Brief): string {
  const span = (s: number[] | null): string => (s ? `[${s}]` : "null")
  return (
    `${b.type} (${b.type_confidence.toFixed(3)}), target ${JSON.stringify(b.target)} ${span(b.target_span)}, ` +
    `value ${JSON.stringify(b.value_text)} ${span(b.value_span)}`
  )
}

/** Web vs the stock default-level Python session: per-source flips, listed cases. */
function defaultSessionSection(report: Report): string[] {
  const fmt = (n: number): string => (n === 0 ? "0" : n.toExponential(2))
  const d = report.vs_python_default_session
  const lines: string[] = [
    "",
    "## Web vs the stock default-level Python session",
    "",
    "### Which Python reference is the deployed one",
    "",
    "`gidi.inference.runner.OnnxRunner` sets only `intra_op_num_threads=1`, " +
      "`inter_op_num_threads=1` and `ORT_SEQUENTIAL`; it never sets `graph_optimization_level`, " +
      "so deployed Python runs ORT's default, `ORT_ENABLE_ALL`. " +
      "`GidiPredictor.from_bundle` (the deployed Python path) therefore runs the **default " +
      "`ORT_ENABLE_ALL` session on arm64, i.e. with ORT's fused " +
      "`DynamicQuantizeMatMul`/`MatMulIntegerToFloat` kernels (KleidiAI on arm64)**, not the " +
      "unfused graph that onnxruntime-web executes. This " +
      "report's *Python basic* column is the same runtime with `ORT_ENABLE_BASIC` " +
      "(`scripts/dump_web_parity.py`, `graph_level_predictor`).",
    "",
    "### Differences in type / target_span / value_span (web vs Python default)",
    "",
    `${d.inputsDiffering} of ${report.totals.predictions} inputs differ ` +
      `(type ${d.fieldFlips.type}, target_span ${d.fieldFlips.target_span}, ` +
      `value_span ${d.fieldFlips.value_span}; one input can differ in several fields).`,
    "",
    "| source | cases | inputs differing | type | target_span | value_span |",
    "|---|---:|---:|---:|---:|---:|",
  ]
  for (const [source, r] of Object.entries(d.bySource as Report)) {
    lines.push(
      `| ${source} | ${r.n} | ${r.differing} | ${r.type} | ${r.target_span} | ${r.value_span} |`
    )
  }
  const listed = ["golden", "test", "probe", "preset", "regression"]
  const cases = d.cases as Report[]
  lines.push(
    "",
    "Every differing golden / test / probe / preset / regression case (all sources are in the JSON):",
    ""
  )
  const shown = cases.filter((c) => listed.includes(c.source))
  if (shown.length === 0) lines.push("None.")
  for (const c of shown) {
    lines.push(
      `- **${c.source} / ${c.id}** \`${JSON.stringify(c.text)}\` (differs: ${c.fields.join(", ")})`,
      `  - python default: ${briefLine(c.python_default)}`,
      `  - python basic:   ${briefLine(c.python_basic)}`,
      `  - web:            ${briefLine(c.web)}`
    )
  }
  const presets = report.presets as Report
  lines.push(
    "",
    "### Playground presets",
    "",
    `Identical in type, target_span and value_span across web, Python default and Python basic: ` +
      `**${presets.identical_all_three}/${presets.cases.length}**. Max confidence difference web vs ` +
      `default ${fmt(presets.max_confidence_delta_web_vs_default)}, web vs basic ` +
      `${fmt(presets.max_confidence_delta_web_vs_basic)}.`,
    "",
    "| preset | web | python default | python basic | identical |",
    "|---|---|---|---|---|"
  )
  for (const p of presets.cases as Report[]) {
    lines.push(
      `| ${JSON.stringify(p.text)} | ${briefLine(p.web)} | ${briefLine(p.python_default)} | ${briefLine(p.python_basic)} | ${p.identical ? "yes" : "NO"} |`
    )
  }
  return lines
}

process.exitCode = await main()
