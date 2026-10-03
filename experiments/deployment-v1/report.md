# deployment-v1: hardening gidi-finance-v1

Scope: packaging and runtime only. Nothing was trained, the architecture and tokenizer are
unchanged, and no data was generated. `protocol.json` (seed rule, suite source, acceptance gate)
was written before packaging. Usage, schema and limitations are in `docs/deployment.md`.

## 1. Deployment seed: seed 1

**Rule (predeclared):** use seed 1, the export/deployment seed fixed in advance in
distillation-v2, compression-v1, v3 and v4.

**Why not pick by held-out metrics:**
- compression-v3's protocol says "selection: none on held-out data; test and probe-v1 are
  evaluation only".
- The frozen validation split is inside the 723 training notes, and every seed fits it at
  1.000, so there is no clean selection set.
- Choosing the best test/probe seed would select on the test sets and bias every number
  reported for v1.

**Sanity check only, not a ranking.** Seed values for compression-v3 seeds 1 / 2 / 3:

| | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| test type acc | 0.952 | 0.933 | 0.914 |
| test span F1 | 0.808 | 0.838 | 0.876 |
| test joint | 0.848 | 0.838 | 0.848 |
| probe joint | 0.605 | 0.642 | 0.630 |
| probe lender-first | 0.56 | 0.56 | 0.67 |
| probe insurance premium | 0.33 | 0.56 | 0.67 |

Seed 1 is not pathological:
- Its test joint is tied for best.
- Its probe joint is the lowest by 2–3 notes of 81, inside the seed spread.
- Lender-first equals seed 2.
- Its weakest pattern is insurance premium, 2 notes below the other seeds.
- INT8 costs it 2 probe notes (0.605 → 0.580), with no cost on test.

No weight averaging or ensembling.

## 2. Artifact `models/gidi-finance-v1/`

`scripts/freeze_deployment.py` builds the bundle by byte copy from the sources and writes
`config.json` and `manifest.json`.

| file | bytes | sha256 |
|---|---|---|
| model.int8.onnx | 28,658,533 | `a31caa57…dc411` (= compression-v3 seed-1 INT8) |
| tokenizer.json | 574,195 | `564679d0…3ae045` (= seed-1 checkpoint) |
| tokenizer_config.json | 340 | `e7d65c11…d76` |
| vocab_map.json | 49,336 | `da3f4357…3ea3f68` (= B-rank-8000 map) |
| config.json | 796 | `f0afe138…638a` |
| manifest.json | 4,583 | all hashes, sources, rationale |

The manifest also records:
- source checkpoint `bfb2195a…`, FP32 ONNX `0366683f…`, FFN map `486084a8…`;
- positions 2050 → 34, vocab rows 20,481 → 8,338, FFN 3072 → 2048;
- seed, protocol sha, runtime requirements.

The script asserts the hashes recorded in the checkpoint, the 34/8338/2048 shapes, the type
order against `configs/annotation-v1.yaml`, and the ONNX I/O names.

Files are read-only, and a differing rebuild is refused. The bundle totals 29.29 MB.

## 3–4. Runtime and confidences

Package `src/gidi/inference/`:

| module | role |
|---|---|
| `bundle` | config, manifest, `verify_bundle` |
| `text` | NFC with an NFC→original offset map |
| `tokenizer` | `tokenizers` wrapper; truncation 32, longest-first, right |
| `runner` | 1-thread CPU ORT session; checks I/O names |
| `decode` | the `spans_from_tags` rule, softmax, confidences |
| `schema` | `Prediction`, `RawOutput`, `EmptyInputError` |
| `predictor` | `GidiPredictor` |

It imports only numpy, onnxruntime and tokenizers. A test asserts that torch, transformers,
onnx and the training modules stay unloaded. `numpy` is now a direct project dependency.

Behaviour:
- empty or whitespace input raises `EmptyInputError`; non-str raises `TypeError`;
- input over 32 tokens is truncated as in training, with `truncated: true`;
- lone surrogates become U+FFFD for tokenization only;
- no override rules.

Confidences:
- `type_confidence` is the softmax probability of the argmax type.
- `target_confidence` is the geometric mean of the member tokens' predicted-tag probabilities.
  With no span it is the minimum P(O) over real tokens.
- They are uncalibrated, and no thresholds are set.

## 5. Golden regression suite (`golden-suite.jsonl`, 77 cases)

Built by `scripts/build_golden_suite.py`, deterministic and byte-identical on rebuild.

Sources:
- approved train/validation records: 69 human-labelled, plus 8 approved AI-proposal records
  where human examples were thin;
- 5 `skipped` records from `labels.jsonl`.

It never reads test or probe-v1.

Coverage:
- every type ×9;
- target present 48, null 29;
- accented 48, unaccented 29; slang/shorthand 24;
- lender-first 8; borrow with `cho vay` 2; borrow direction 6;
- repayment in 9, out 9;
- family gift 4, insurance 2, refund/cashback 7;
- title excluded from target 5, title included 9;
- merchant 6, payment-channel null 4;
- skipped 5.

Gap: there is no standalone `ko`/`lít` in train or validation. They are covered by the
robustness inputs.

| comparison (77 cases) | type | span | exact | max abs logit diff |
|---|---|---|---|---|
| PyTorch vs ONNX FP32 runtime | 1.000 | 1.000 | 1.000 | 1.7e-5 (identical input ids) |
| ONNX FP32 vs INT8 runtime | 1.000 | 1.000 | 1.000 | type 0.56, tag 1.01 |

The largest INT8 confidence shifts are 0.031 on type and 0.135 on target.

Gold agreement is 72/72 for all three paths. These are training notes, so this is a wiring
check, not quality.

The 5 skipped notes, debt-state notes like `còn nợ …`, all come out as `repayment_in` with
confidence 0.83–0.99. The model has no skip class, so out-of-taxonomy handling belongs to the
surrounding pipeline (see Known limitations).

## 6. Runtime robustness (`robustness-inputs.jsonl`, 64 inputs): 0 failures

The inputs cover:
- empty and whitespace (5 → `EmptyInputError`);
- `k`, `ăn`;
- exactly 32 tokens (not truncated), 33 tokens, 300 words and 5,000 characters (truncated);
- punctuation-heavy notes;
- emoji, including ZWJ and astral characters;
- Vietnamese mixed with English;
- `1tr5`, `200k`, `1.500.000đ`, `$20`;
- unknown names and merchants;
- NFC/NFD pairs;
- tabs, newlines and control characters;
- RTL, CJK and Thai text;
- lone surrogates.

Results:
- **Tokenizer:** the bundle tokenizer matches the training HF path (ids, offsets, truncation
  flag) on 745 texts. These are the golden and robustness inputs plus all train/validation
  notes. The unit tests add 300 fuzzed strings.
- **Offsets:** 272 predictions checked; `original[start:end] == target` for every one. NFC/NFD
  pairs give the same type and the same target, each in its own string's offsets.
- **Determinism:** 3 repeat runs plus a fresh predictor give identical outputs and
  bitwise-identical logits.
- **PyTorch vs FP32 on robustness inputs:** identical ids and types, max logit diff 2.9e-5.
- **INT8 vs FP32 on robustness inputs (59 scorable):** type 0.983, span 0.932. The differences
  are Arabic, Thai/Korean, quote-heavy and 300-word inputs, all out-of-distribution.

## 7. Performance (Apple Silicon Mac, CPU, 1 thread, batch 1, INT8)

| | |
|---|---|
| warm p50 / p95 / p99 (2000 calls; tokenize + run + decode) | 1.19 / 1.61 / 1.75 ms |
| cold: import / load / first predict (median of 5 processes) | 0.052 s / 0.041 s / 2.3 ms |
| RSS after load + 200 predicts | 131 MB peak; +87.5 MB over python + runtime imports |
| model file | 28.66 MB |
| tokenizer artifacts | 0.62 MB |
| bundle total | 29.29 MB |

These are from the first verification run. A rerun gave 1.17 / 1.55 ms p50 / p95 and 121 MB
peak RSS; `verification.json` holds the latest run.

The runtime needs no Python-specific components: NFC normalization, a `tokenizer.json` tokenizer
and ONNX Runtime, plus about 40 lines of argmax/BIO decoding. That ports directly to a plain
ONNX Runtime integration; the steps are in `docs/deployment.md`.

## 8. Package boundary

Production code lives in `src/gidi/inference/` and imports nothing from `gidi.modeling`,
`gidi.export`, training or compression.

`scripts/freeze_deployment.py` and `scripts/verify_deployment.py` are tooling: they use torch
for the reference only. The runtime opens only the bundle directory. An audit-hook subprocess
running from an empty cwd on a copied bundle touched no `datasets/`, `corpus/` or
`experiments/` path. Native libraries read `tokenizer.json` and the ONNX file, which the hook
cannot see.

## 9. Acceptance gate (`verification.json`)

| check | result |
|---|---|
| artifact reproducible | yes. `freeze_deployment.py --check` gives a byte-identical rebuild; re-quantizing the source FP32 gives the same INT8 sha256. |
| tokenizer/model compatible | yes, 745/745 identical encodings; ONNX I/O names checked |
| ONNX INT8 parity | yes: golden 77/77 exact vs FP32; FP32 matches PyTorch within 1.7e-5 |
| stable offsets | yes, 272/272, including NFD input |
| regression suite | passes |
| edge cases | 0 crashes in 64 inputs |
| INT8 size | 28.66 MB (28,658,533 bytes) |
| no runtime dependency on test/probe | yes, isolation check passed |

Gates: 298 tests pass, ruff is clean, and the distillation-data check is OK.

## Known limitations

- Lender-first `X cho mượn/vay` → `borrow`, especially unaccented, remains weaker than the full
  FFN-3072 reference. Probe lender-first joint is 0.59 vs 0.78 across seeds, and 0.56 for
  seed 1.
- Probe-v1 quality for seed 1 is 0.605 joint in FP32 and 0.580 in INT8. Test is 0.848.
- There is no skip or out-of-taxonomy class.
- Confidences are uncalibrated.

## Artifacts

- `protocol.json`, `golden-suite.jsonl`, `robustness-inputs.jsonl`, `verification.json`.
- `models/gidi-finance-v1/`, `docs/deployment.md`.
- Code: `src/gidi/inference/`, `tests/test_inference.py`, plus the scripts
  `freeze_deployment.py`, `build_golden_suite.py` and `verify_deployment.py`.

```sh
uv run python scripts/freeze_deployment.py --out models/gidi-finance-v1   # [--check]
uv run python scripts/build_golden_suite.py
uv run python scripts/verify_deployment.py
```

**A. gidi-finance-v1 is ready for application integration.**
