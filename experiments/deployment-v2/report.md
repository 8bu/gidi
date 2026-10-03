# deployment-v2: gidi-finance-v2

**Status:** `gidi-finance-v2` is the new **testable candidate**, ready for manual testing.
`gidi-finance-v1` is unchanged: `freeze_deployment --check` reports it byte-identical.

Out of scope, so none of this was started:

- compression or distillation;
- v8;
- any new modeling experiment.

Two frozen INT8-vs-FP32 gate items fail as written. Every failing case comes from the deployed v1
INT8 type/target path, which v2 reproduces bit for bit. The user decided to accept this as the
production baseline; see §5.

The protocol is `experiments/deployment-v2/protocol.json`, declared before packaging (0444,
sha256 `0f0db9e9…1cacb`). Machine-readable results:

- `verification.json`
- `mismatches.jsonl`
- `golden-suite.jsonl`
- `hardening-cases.jsonl`
- `flaky-test-investigation.md`

## 1. Model

| | |
|---|---|
| architecture | accepted value-span-v7 dual encoder |
| source experiment | `experiments/value-span-v7-dual-encoder` (verdict A); protocol sha256 `7792e088…f166` |
| checkpoint | `models/value-span-v7-dual-encoder/seed1` (safetensors `cef088a4…e5b0`) |
| seed | value-head seed 1, the `export_seed` that the v7 protocol declared before training; not chosen on held-out results |
| parameters | 57,181,981 total: path A 28,496,395 (v1 encoder + type/target heads, frozen), path B 28,685,586 (value encoder 28,487,936 + MLP 197,635 + CRF 15) |

Structure: both paths read the same `input_ids` and `attention_mask` from a single
normalization and tokenization.

- **Path A** is the frozen gidi-finance-v1 encoder, type head and target BIO head. It is the only
  source of `type` and `target`.
- **Path B** is the value path, the only source of `value_*`:
  - an independently fine-tuned copy of the v1 encoder;
  - Linear(768,256) → GELU → Linear(256,3) emissions;
  - a linear-chain CRF → Viterbi decoding.
- Path B never changes path A's outputs.

## 2. Quality

**Type/target invariance:** passes.

- v2 FP32 type and target logits are bit-identical to v1's FP32 ONNX.
- v2 INT8 type and target logits and confidences are bit-identical to the `gidi-finance-v1`
  bundle.
- This holds on the golden suite (79), test (143), probe (81) and robustness inputs (62), with 0
  mismatches.

**Value metrics:** identical for PyTorch, ONNX FP32 and ONNX INT8. 0 value predictions change
between backends on test or probe. The numbers reproduce v7 seed 1 exactly.

| set | exact | token F1 | present/null | human-only | multi-number |
|---|---|---|---|---|---|
| test (139 with value labels) | 0.9856 | 0.9970 | 1.000 | 0.9167 (n=12) | 0.9545 (n=22) |
| test-v1 (105) | 0.9810 | 0.9958 | 1.000 | 0.8571 | 0.9091 |
| test-targeted (34) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| probe (81) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

- All five frozen v7 criteria pass for INT8, the deployed precision.
- Test slices: explicit unit 0.982, bare number 1.0, multi-number 0.955, slang 1.0, unaccented
  1.0, unseen span 0.973, long/multi-token 0.968. The slice floor is 0.9545.

Known value errors on test:

1. `cho a Nam vay 1 triệu 20/10` → `1 triệu 20/10` (gold `1 triệu`). The date is absorbed into
   the value. This is the tracked limitation, and the exported runtime returns it with confidence
   ≈ 1.0. It is not special-cased anywhere.
2. `cashback thẻ vib 230k` → `30k`: boundary truncation.

Tracked note `Thắng vay 5 củ, hẹn t10 trả` → `5 củ` is exact on every backend.

## 3. Export

| artifact | path | bytes | sha256 |
|---|---|---|---|
| FP32 ONNX (dynamo, opset 17, 1 graph, 2 encoders) | `models/gidi-finance-v2-onnx/model.onnx` | 228,801,198 | `0704688e…849f` |
| INT8 ONNX (dynamic, QInt8 weights, v1 settings) | `models/gidi-finance-v2-onnx/model.int8.onnx` = `models/gidi-finance-v2/model.int8.onnx` | 57,503,558 | `bc7945c3…1994e` |
| tokenizer | `models/gidi-finance-v2/tokenizer.json` | 574,195 | `564679d0…e045` (same as v1) |
| tokenizer config | `models/gidi-finance-v2/tokenizer_config.json` | 340 | `e7d65c11…0d76` |
| vocab map (provenance) | `models/gidi-finance-v2/vocab_map.json` | 49,336 | `da3f4357…3ea3f68` |
| runtime config | `models/gidi-finance-v2/config.json` | 2,199 | `77747c09…c338` |
| manifest | `models/gidi-finance-v2/manifest.json` | 5,911 | `733a31a1…fe91` |

`config.json` holds:

- `types`, the target `tags` and `value_labels` (O / B-VALUE / I-VALUE);
- `value_decoding`: `crf_viterbi`, the float32-exact start/end/transition scores, the span
  selection rule and the confidence rule;
- `architecture.kind = dual_encoder` with `path_a`/`path_b` and `shared_tokenization: true`;
- the ONNX interface (`input_ids`, `attention_mask` → `type_logits`, `tag_logits`,
  `value_logits`), `max_length` 32, NFC normalization and the truncation rule.

`manifest.json` records:

- every source with its sha256 and size;
- parameter counts;
- seed selection;
- runtime requirements.

Size:

- The bundle is **58.14 MB**, 1.99× v1's 29.29 MB.
- The INT8 model is 57.50 MB, 25% of FP32.
- The estimate was 57.4 MB.

## 4. Parity: PyTorch vs ONNX FP32 (passes)

There are 0 semantic mismatches on golden, test, probe and robustness inputs. The two
lone-surrogate inputs have no PyTorch reference because the HF tokenizer rejects them. Checked,
all with agreement 1.0:

- type;
- target BIO tags;
- target span;
- Viterbi tags;
- value span;
- the full runtime dict.

| max abs diff (worst set) | |
|---|---|
| type logits / confidence | 2.4e-5 / 3.3e-6 |
| target logits / confidence | 2.9e-5 / 7.8e-6 |
| value emissions / confidence | 2.1e-4 / 1.4e-7 |

## 5. Parity: ONNX FP32 vs ONNX INT8

| set | type | target exact | value exact | full output | value conf drift max / mean |
|---|---|---|---|---|---|
| golden (79) | 1.000 | 0.987 | 1.000 | 0.987 | 1.2e-4 / 1.6e-6 |
| test (143) | 0.986 | 0.986 | 1.000 | 0.972 | 2.2e-4 / 1.6e-6 |
| probe (81) | 0.975 | 1.000 | 1.000 | 0.975 | 3e-8 / 6e-9 |
| robustness (59) | 0.983 | 0.932 | 0.983 | 0.915 | 0.074 / 0.002 |

- **Value path:** golden, test and probe have 0 value-span differences. The only value
  difference anywhere is the robustness input `mua sách $20 trên amazon`, where FP32 gives `20`
  and INT8 gives `$20`.
- **Type/target path:** every difference is a note where the **deployed v1 INT8 bundle already
  differs from v1 FP32**, and v2 INT8 reproduces it bit for bit. The near-tie types have
  confidence 0.44–0.59. The disagreeing notes:
  - golden-033: `mb bank…` target `mb` → `m`;
  - test: `mua giay the thao 1tr25`, `coopmart đồ dùng tuần 412k` (type);
  - test: `bác Bảy ck…` and `Vân gửi lại…` (target);
  - probe: `ông nội cho 1 triệu mua sách`, `bo me cho 5tr dong hoc` (type).
- Agreement with the v1 bundle is 100% (§2).
- Type/target confidence drift is up to 0.30 on test and 0.23 on probe. It is the same
  inherited v1 INT8 behavior.

**Frozen gate items failing as written:**

- `int8_vs_fp32_golden_zero_…_differences`: 1 target span.
- `int8_vs_fp32_test_and_probe_each_field_ge_99pct`: test type 98.6%, test target 98.6%, probe
  type 97.5%.

The protocol requires both of these:

- v2 type/target identical to the v1 bundle;
- INT8-vs-FP32 type/target agreement of at least 99%.

v1's own INT8 drift makes the two incompatible. That is a protocol design error made before
packaging, not a v2 regression.

**User decision:** treat the deployed v1 INT8 as the production baseline for the v2
type/target path. The historical protocol results stand unchanged as failing against the FP32
reference, with this note.

## 6. Runtime

- **API:** `GidiPredictor.from_bundle("models/gidi-finance-v2").predict(text).to_dict()` returns
  these keys, in order:
  - `type`, `type_confidence`;
  - `target`, `target_span`, `target_confidence`;
  - `value_text`, `value_span`, `value_confidence`;
  - `truncated`, `model_version` = `gidi-finance-v2`.
- **Dependencies and file access:** the runtime imports only numpy, onnxruntime and tokenizers,
  and reads only the bundle directory. Isolation was verified in a child process.
- **Value decoding:** Viterbi with the bundle's CRF over the real tokens. The span is the one
  with the highest confidence. Confidence is the geometric mean of softmax(emissions) for the
  Viterbi tag over the span's tokens; with no span, it is the minimum P(O). This is the same code
  path (`gidi.inference.decode.decode_value_crf`) that the v7 evaluation now delegates to.
- **Preprocessing:**
  - Each `predict()` runs NFC normalization once, tokenization once and the ORT session once,
    verified by instrumented counts.
  - In both ONNX graphs, both encoders' word-embedding Gathers read the single `input_ids` graph
    input, and the position ids are shared.
  - Nothing is normalized or tokenized twice.
- **Offsets:** `value_text` and `target` are sliced from the caller's original string, including
  NFD input. No numeric normalization happens.
- **Truncation:** more than 32 tokens sets `truncated: true`. This was checked against an
  independent token count. A value that starts after the cut, or straddles it, returns `null`;
  no span ever lies past the cut.
- **Hardening:** 553/553 checks pass on the INT8 runtime. They cover:
  - 89 curated cases: accents, unaccented text, Unicode punctuation, empty and whitespace input
    (→ `EmptyInputError`), extra whitespace, punctuation-heavy text, long input, truncation, no
    target, no value, bare/slang/multi-number/multi-token amounts, target + value, and null
    target with a value present;
  - 62 robustness inputs;
  - 2 lone surrogates;
  - 400 random texts.

  On every input the checks verify:
  - spans are in range and slice back exactly;
  - confidences are finite and in [0, 1];
  - `json.dumps(allow_nan=False)` round-trips;
  - Viterbi tags are valid and match an independent implementation.

  A further 4,000 random CRF decode cases pass, including 736 brute-force optimality checks.
- **Performance** (Apple Silicon CPU, 1 thread, batch 1, 300 warm-up calls + 2,000 timed calls;
  the final run had the playground server idle in the background):

| | v1 INT8 | v2 INT8 | v2 FP32 |
|---|---|---|---|
| warm p50 / p95 | 1.57 / 2.20 ms | **3.52 / 4.86 ms** | 6.50 / 8.34 ms |
| cold ready (import + load + first predict) | 0.113 s | 0.165 s | |
| peak RSS after 200 predicts | 130 MB | 180 MB | |

v2 p50 by component:

| component | time |
|---|---|
| NFC normalization + tokenization | 0.011 ms |
| ONNX run | 3.03 ms (about 95%) |
| type/target decode | 0.01 ms |
| CRF Viterbi + span selection | 0.03 ms |

These measured numbers replace the v7 estimate of 7.6 ms/note, which was PyTorch FP32.

## 7. Testing

**Flaky signal:**

- I reran the suite to try to reproduce it: 6 sequential full runs, 2×2 concurrent full runs and
  5 runs of the runtime-critical subset, all clean. It is **unreproduced**.
- [INFERENCE] Of the suite's fixtures, only the old `test_demo_ui` module-scoped HTTP `server`
  fixture is used by exactly 20 tests, so it was most likely a one-off setup failure of that
  fixture, not of the model.
- The model, export, runtime and tokenizer tests were deterministic and clean throughout. That
  server and its tests were rewritten in this phase.
- Details: `flaky-test-investigation.md`.

**Final runs:**

- 3 full suites, 670 passed each.
- 5 runs of the flaky/runtime subset (demo UI, inference, value inference, value deployment,
  CRF), 111 passed each.
- `ruff check` and `ruff format` are clean.
- `pnpm -C playground build` and `lint` are clean.

**Rebuild and integrity:**

- `freeze_deployment --check` for v1 and v2: byte-identical.
- Re-quantizing the FP32 source gives byte-identical INT8.
- All data-build `--check` scripts reproduce.
- The protocol sha256s for value-span v1–v7 and deployment v1/v2 are unchanged.

**Verification run:** `uv run python scripts/verify_deployment.py --protocol
experiments/deployment-v2/protocol.json`. It exits 1 only because of the two inherited INT8
items in §5.

## 8. Playground

- **Stack:** `playground/`, Vite, React 19, TypeScript, Tailwind v4, shadcn/ui (nova preset,
  Radix base, neutral, lucide icons). `scripts/demo_ui.py` serves the built `playground/dist`
  and the JSON API using only the standard library. The old `scripts/demo_ui.html` is removed.
- **shadcn components:** Button, Badge, Separator, Tooltip, Tabs, Alert, Skeleton, Kbd,
  InputGroup + Textarea, Spinner, Empty, ToggleGroup, Card, Field.
  - The registry has no prompt composer, so the chat-style composer is built from Field +
    InputGroup + InputGroupTextarea + addon buttons with Kbd hints.
- **Launch:** `pnpm -C playground install && pnpm -C playground build && uv run python
  scripts/demo_ui.py` → http://127.0.0.1:8765. The default bundle is v2; `--bundle
  models/gidi-finance-v1` still works.
- **Presets:** all 10 were run against the real v2 runtime, and each UI render matched
  `/api/predict` exactly.

| preset | type | target | value |
|---|---|---|---|
| cơm tấm 100 | expense | — | 100 |
| bún bò 50k | expense | — | 50k |
| mượn chú hai 5 xị | borrow | chú hai | 5 xị |
| ăn 2 tô phở 70 | expense | — | 70 |
| tiền điện tháng 10 hết 612 | expense | — | 612 |
| trả góp kỳ 3 1tr5 | repayment_out | — | 1tr5 |
| 20/10 mua quà mẹ 500 | expense | — | 500 |
| mua 3 vé 150k | expense | — | 150k |
| đổ 5 lít xăng 120k | expense | — | 120k |
| cho a Nam vay 1 triệu 20/10 | lend | Nam | 1 triệu 20/10, with a neutral "Known limitation" badge |

The badge appears only while the returned value differs from `1 triệu`. The output itself is
never altered.

**Visual review:**

- 26 screenshots in `playground-screenshots/`, plus my own pass on the live service.
- Covered: desktop 1440 px and mobile 390/360 px, light and dark; no horizontal overflow.
- Long Vietnamese text wraps, raw JSON wraps inside its box, and an emoji before a span
  highlights exactly.
- Also checked:
  - null target, null value, loading and every error state (empty, offline, inference error,
    missing artifact, malformed response, timeout);
  - Enter / Shift+Enter, Clear, preset clicks and Copy JSON.

**UI limitations:**

- Confidences show one decimal, so 0.99999 displays as 100.0%; the raw value is in the tooltip
  and the JSON.
- The raw JSON is re-serialized in the browser, so `1.0` would print as `1`.
- The theme choice is stored in `localStorage`.
- Enter during IME composition is handled in code but wasn't exercised in a real browser.

## 9. Limitations

- **Date absorption:** `cho a Nam vay 1 triệu 20/10` → `1 triệu 20/10`, with confidence about
  1.0. It is not trained against or special-cased.
- **Inherited v1 INT8 type/target drift vs FP32** (§5). The v1 limitations also carry over,
  since path A *is* v1. The main one is lender-first `X cho mượn/vay`, which is predicted
  `lend`/`borrow` unreliably.
- **Hardening observations, not crashes:**
  - NBSP-separated `cho Nam vay 2 triệu` → value `tr`;
  - full-width `５０k` → `０k`;

  NFC folds neither width nor NBSP, and adding that normalization would be a separate decision.
- **Truncation:** a value past the 32-token cut is `null`.
- **Cost:** about 2× v1 in size (58 MB), latency (3.5 ms p50) and memory (+50 MB RSS).
- **Small samples:** test has no notes without an amount; human-only n=12, slang 10, bare
  number 6.
- **Confidences** are uncalibrated, with no thresholds or abstention.
