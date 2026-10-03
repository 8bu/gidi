# value-span-v8-deterministic-parser: report

**V8 is an internal experiment name, not a release.** `gidi-finance-v2` stays the current
release. `gidi-finance-v3` / `3.0.0` is only a *candidate* name, planned if V8 is validated.
Nothing in this report confirms or ships v3.

**Verdict by the pre-declared rule: ACCEPT AS V3 CANDIDATE.** This means a follow-up validation
is justified. It does not mean ship.

- All 11 gates and all 3 secondary conditions of `protocol.json` pass (section 3).
- Test value exact is 0.9928 (138/139) for the parser and 0.9856 (137/139) for gidi-finance-v2.
  The 95% Wilson intervals overlap (0.960–0.999 vs 0.949–0.996), so this is non-inferior, not
  better.
- The candidate system is 50% smaller and 2× faster at p50 (section 6).
- **The evidence is weaker than the numbers look** (section 5): the rule proposer that wrote most
  of the labels scores 1.000 on every set, human subsets included. These sets cannot separate a
  correct parser from one that imitates the proposer.
- **V8 does not address target extraction quality.** The production target issue
  `mua sữa vinamilk hết 500k` → `v` was investigated separately. It is an existing
  gidi-finance-v1 target-model error, recorded in `tests/data/production-regressions.jsonl`
  (prod-0001) and in the known limitations of `docs/deployment.md`. No target code and no parser
  rule was changed for it.

## 1. Question and system

Can the second neural value encoder of gidi-finance-v2 (V7, path B) be replaced by a
deterministic value parser, so that value quality is non-inferior while size, memory and latency
drop?

| system | type / target | value |
|---|---|---|
| **v8 (candidate)** | `models/gidi-finance-v1` INT8, unchanged | `gidi.value_parser.parse_value` |
| v2 (baseline) | `models/gidi-finance-v2` INT8 as shipped (V7 seed 1) | path B encoder + CRF |
| proposer (diagnostic only) | v1 INT8 | `gidi.annotation.value_span.propose_value` |

Parser (`src/gidi/value_parser/`, 3 files, 21 KB source, pure Python):

- one linear pass over the NFC text with an accent-folded shadow string of the same length;
- money forms: `k`, `tr/triệu`, `củ`, `xị`, `chai`, `lít` (slang), `nghìn/ngàn`, `tỷ`, currency
  suffixes, grouped numbers (`80.000`, `1 500 000`), `1tr5`, `1 triệu 2`, `2 củ rưỡi`, number
  words (`bốn triệu`, `nửa củ`);
- competing numbers dropped by context: dates, times, percentages, periods and instalments,
  weekdays, years, quantities and measures, phone numbers, ids;
- choice by tier (explicit unit or money-shaped > slang > bare number) and position;
- the span is mapped to the caller's original string with the inference NFC offset maps.

Not changed: `models/**`, `datasets/**`, `corpus/**`, `configs/**`, existing `experiments/*`,
`dist/**`, `playground/**`, any inference decoding. No retraining, new data or threshold change.

## 2. Protocol and freeze order

| step | file | time (UTC, 2026-10-03) | sha256 |
|---|---|---|---|
| protocol frozen, before the parser existed | `protocol.json` (0444) | 16:27:59 | `78dcb4ba…8d31` |
| parser frozen after train development | `parser-freeze.json` (0444) | 16:41:11 | `4fcfd56f…6f6d`; parser source `f2991ca5…af3` |
| held-out scored once | `results.json` (0444) | 16:41:22 | `ff472616…5364`; records both shas |
| cost measured | `cost.json` | 16:41:55 | `69c7b40e…0341` |

- The held-out stage refuses to run if the parser source differs from `parser-freeze.json` or if
  `results.json` exists. It ran once.
- Scoring is the frozen V7 code (`value_metrics.evaluate_value`, `value_gate.summarize_set`,
  `value_quality_criteria`, `value_error_table`, `compare_value_v1.load_sets`, the V7 tokenizer for
  token offsets). No metric was written for this experiment; predictions come from
  `GidiPredictor` and the parser instead of logits. The shas of the scoring files are in
  `results.json`.
- The v2 INT8 numbers reproduce the published deployment numbers (test exact 0.9856, human
  0.9167).
- The parser was developed on `train.jsonl` only (`dev-log.md`). The parser was written from train
  labels; of the proposer only its docstring and API signatures were read, not its rules.

## 3. Results: v8 vs v2

Value metrics on `value_status == complete` rows. Type and target are identical (section 4).

| set | n | v8 exact | v2 exact | v8 human | v2 human | proposer (diag.) |
|---|---|---|---|---|---|---|
| **test** | 139 | **0.9928** (138) | 0.9856 (137) | 12/12 | 11/12 | 1.000 |
| test-v1 | 105 | 1.0000 | 0.9810 | 7/7 | 6/7 | 1.000 |
| test-targeted | 34 | 0.9706 | 1.0000 | 5/5 | 5/5 | 1.000 |
| probe | 81 | 1.0000 | 1.0000 | 5/5 | 5/5 | 1.000 |
| validation (110 of 149 rows are also in train) | 146 | 1.0000 | 1.0000 | 18/18 | 18/18 | 1.000 |
| validation, 39 rows not in train | 36 | 1.0000 | 1.0000 | 7/7 | 7/7 | 1.000 |
| train (**development**, not evidence) | 889 | 1.0000 | n/a | 82/82 | n/a | n/a |

Test detail:

| metric | v8 | v2 |
|---|---|---|
| span precision / recall / F1 | 1.0000 / 0.9928 / 0.9964 | 0.9856 / 0.9856 / 0.9856 |
| token F1 | 0.9985 | 0.9970 |
| present/null accuracy | 0.9928 | 1.0000 |
| exact on rule-provenance labels | 126/127 | 126/127 |
| exact on human-provenance labels | 12/12 | 11/12 |

Test slices (exact):

| slice | n | v8 | v2 |
|---|---|---|---|
| explicit_unit | 113 | 1.000 | 0.982 |
| bare_number | 6 | 0.833 | 1.000 |
| multi_number | 22 | 0.955 | 0.955 |
| slang | 10 | 1.000 | 1.000 |
| unaccented | 50 | 0.980 | 1.000 |
| unseen_span | 37 | 1.000 | 0.973 |
| long_multi_token | 63 | 1.000 | 0.968 |
| no_amount | 0 | n/a | n/a |

Gates (all pass):

| gate | v8 | threshold | margin |
|---|---|---|---|
| G1 test value exact | 0.9928 | ≥ 0.95 | 0.043 |
| G2 human exact | 12/12 | ≥ 0.90 | |
| G3 present/null | 0.9928 | ≥ 0.97 | |
| G4 multi_number | 0.9545 | ≥ 0.90 | |
| G5 slice floor (n ≥ 5) | 0.8333 (bare_number, n = 6) | ≥ 0.80 | **one more miss would fail it** |
| G6 vs v2 exact | 0.9928 vs 0.9856 | ≥ v2 − 0.02 | |
| G7 vs v2 present/null | 0.9928 vs 1.0000 | ≥ v2 − 0.01 | **exactly one miss allowed, used** |
| G8 vs v2 human | 12 vs 11 notes | ≥ v2 − 1 note | |
| G9 vs v2 multi_number | 21 vs 21 notes | ≥ v2 − 1 note | |
| G10 type/target unchanged | 0 diffs in 373 notes | 0 | |
| G11 cost lower | 29.31 MB vs 58.14 MB; p50 1.37 vs 2.78 ms | smaller and faster | |

Secondary: S1 probe exact 1.000 (≥ 0.95). S2 pooled human held-out (test, probe, validation not
in train) 24/24 for v8 and 23/24 for v2 (≥ 0.90). S3 no crash, 0 repeat mismatches on 555 notes
and on adversarial inputs.

## 4. Type and target

Type and target come from gidi-finance-v1 INT8. Across validation, test and probe (373 notes) the
v1 and v2 bundles gave 0 differences in type, target span, type confidence and target
confidence. So type/target metrics are identical by construction and confirmed:

| set | type accuracy | target exact |
|---|---|---|
| test | 0.9371 | 0.8951 |
| probe | 0.7778 | 0.8148 |
| validation | 0.9732 | 0.9933 |

## 5. Error analysis and caveats

**Parser errors on held-out data: 1.**

| set | note | gold | v8 | category |
|---|---|---|---|---|
| test, test-targeted | `tra no chi Mai 300 hom 5/10` (rule label) | `300` | null | missed value |

**Cause, found after the held-out run (reported, not fixed):** the quantity-after word list has
`hom`, a guess meant for `3 hôm`. Here `hom 5/10` means "on 5/10", so the bare number `300` was
dropped. No train note supports the entry. The protocol forbids a second attempt, so the parser
was not changed and no corrected score is reported. This is a lexicon error, not a model of
the data: a different word list could produce more of the same kind.

**v2 errors on test (2):** `cho a Nam vay 1 triệu 20/10` → `1 triệu 20/10` (right overrun into
the date, human label) and `cashback thẻ vib 230k` → `30k` (truncation). The parser gets both
right. The two systems fail on different notes, which is the expected profile of a rule system
vs a learned one.

| error category | v8 test | v8 probe | v8 validation-new | v2 test |
|---|---|---|---|---|
| missed value | 1 | 0 | 0 | 0 |
| boundary truncation | 0 | 0 | 0 | 1 |
| right overrun into date/time | 0 | 0 | 0 | 1 |
| wrong number, left/other overrun, spurious | 0 | 0 | 0 | 0 |

Multi-number errors on test: v8 1 (above), v2 1 (the `1 triệu 20/10` note). Tracked V7 cases:
`cho a Nam vay 1 triệu 20/10` → `1 triệu` (exact); `Thắng vay 5 củ, hẹn t10 trả` → `5 củ` (exact).

**Development errors on train** (before the final run; `dev-log.md`): the first version had 27
errors in 889 notes (0.9696). 19 were `đóng` (to pay) read as the currency `đồng`; 4 were
`sửa`/`trả` read as `sữa`/`trà` goods nouns; 1 phone number swallowed the amount after it; 1
grouped number dropped by a quantity word; 2 `trieu` continuation guards. All fixed with general
rules; train is 889/889 and is a development number only.

**Circularity (the main caveat).**

- Most value labels are rule-provenance: 127 of 139 complete test labels, 807 of 889 in train.
  They were written by the proposer in `src/gidi/annotation/value_span.py`.
- The proposer, run as a diagnostic, scores **1.000 on every complete label of every set,
  including all 24 held-out human labels**. [INFERENCE] The human review used the proposer's
  output as a starting point (journal §17), so the human labels are not independent of it either.
- So a parser that imitates the proposer's conventions gets the same score as one that extracts
  amounts correctly. V8 agreeing with these labels says it follows the contract as the proposer
  applied it. It does not show that the parser matches an independent annotator.
- Mitigations used: human subsets reported on their own; the proposer rules were not read (only
  its docstring and signatures); the parser was written from train labels and general Vietnamese money rules. They reduce the risk;
  they do not remove it.
- Human subsets are small: test 12, probe 5, validation-new 7. Each test note moves exact by
  0.7% and human exact by 8.3%, and the 12/12 interval is 0.76–1.00.
- The test set has 0 complete no-amount labels. Null behaviour is measured only on train
  (10 no-amount notes, all correct, development) and by the 4 uncertain test notes, which are
  masked. The false-amount risk on counted things the word lists do not know (`mua 5 áo`) is
  not measured by any held-out set.

**Other limits.**

- `300 hom 5/10` shows how thin the margins are: G5 and G7 would both have failed with one more
  miss.
- Not supported: `2m` / `1m5` (m = million), `1k5`, `100kđ`, a leading currency sign.
- The tie rule for several explicit amounts (first) and several bare numbers (last) has no
  complete train example; it is a prior.
- The browser runtime (`playground`) is TypeScript. Using the parser there needs a port and a
  parity test. That work and its parity risk are not costed here.

## 6. Cost (measured)

Same machine (macOS 26.5.2, arm64, Python 3.12.13), one onnxruntime thread, batch 1,
`GidiPredictor.predict` on the 224 test + probe notes, warm-up pass, 3 rounds × 5 passes = 3,360
calls per system. The per-round p50 differed by under 8% between rounds. Journal §25 quotes 3.5 ms for v2 from
another run and note set; compare only within this table.

| | v2 INT8 | v1 INT8 | parser | **v8 = v1 INT8 + parser** |
|---|---|---|---|---|
| `model.int8.onnx` | 57,503,558 B | 28,658,533 B | none | 28,658,533 B |
| bundle directory | 58,135,539 B | 29,287,783 B | 21,059 B source | **29,308,842 B** |
| p50 latency | 2.78 ms | 1.35 ms | 0.0099 ms | **1.37 ms** |
| p95 latency | 3.57 ms | 1.80 ms | 0.014 ms | **1.83 ms** |
| RSS after load + 143 notes | 171.8 MB | 122.0 MB | 46.9 MB | **126.6 MB** |

- The runtime baseline (python + numpy + onnxruntime, no model) is 40.9 MB RSS.
- The v8 bundle is 50.4% of v2 in bytes (−28.83 MB), p50 latency is 49% of v2, RSS is 74% of v2.
- The parser adds 0.01 ms; the saving is the second encoder pass.
- RSS is a median of 3 fresh processes; peak RSS equalled current RSS.
- Not measured: browser / WASM runtime.

## 7. What the verdict is and is not

- Accept as a V3 *candidate*: the pre-declared rule is met, the cost saving is real and large,
  and type/target cannot change.
- Not decided: whether to ship. The quality claim rests on labels that the proposer reproduces
  perfectly, 12 human test notes and one test error that came from a guessed word-list entry.
- Useful follow-up if wanted (not started): a small set of freshly written notes, labelled by
  someone who has not seen the proposer, covering amount + date, counted things, classifier
  words, `m` and `1k5` forms and no-amount notes; score the parser and v2 once on it. Port and
  parity-test in TypeScript only after that.

## 8. Artifacts

| path | content |
|---|---|
| `protocol.json` | frozen protocol, gates, verdict rule |
| `parser-freeze.json` | parser sha256, train development summary |
| `dev-log.md` | train iterations, disclosures |
| `results.json`, `eval/` | all summaries, gates, slices, provenance split, v8-vs-v2 note changes; predictions for v8, v2 and proposer |
| `cost.json`, `verdict.json` | measured cost; verdict |
| `value-errors.jsonl`, `multi-number-errors.jsonl`, `train-errors.jsonl` | every error |
| `src/gidi/value_parser/`, `scripts/run_value_parser_v8.py`, `tests/test_value_parser.py` | parser, runner, 78 tests |
