# gidi-en student v3 (seed 1): v2 + a learned value-span head

**Verdict: FAIL the pre-declared protocol** (`protocol.json`, written before any real v3 training or
scoring; thresholds, data, seed and decode unchanged). The value criteria (a) and the scanner
criterion (c) pass by a wide margin; the no-regression criterion (b) fails on 4 of 6 checks. One
seed, one run, last epoch, no selection. Every number is agreement with **silver** labels
(regex + Clef + Claude adjudication for the value; Claude + Clef for type/target), not human truth.

A control (below) shows the failing regressions are of the size a plain re-run of the v2 recipe
produces on its own; it does not change the verdict.

## What was built

* **Value labels** (new, separate files, never human): `datasets/annotation-en-v1/value-labels-{test,dev,seed,probe-slang,probe-genz}.jsonl`,
  `value-silver-train.jsonl`, `value-silver-slang-train.jsonl` (3,285 complete notes; the slang
  silver notes are the 613 consistent ones v2 trains on). Builder `scripts/build_en_value_labels.py`,
  scanner A `src/gidi/value_parser/english.py`, Clef labeler B
  `experiments/gidi-en-student-v3/run_value_teacher.py`, adjudications and agreement under
  `datasets/annotation-en-v1/work/value/`. Convention clarified in `docs/annotation-en-v1.md`
  ("Amount formats": currency marker inside the span, word amounts, what is not an amount).
* **Model**: `GidiValueModel` of value-span-v1 (`Linear(768, 3)` O/B-VALUE/I-VALUE on the shared
  token states, +2,307 parameters), loss `CE(type) + CE(target) + CE(value)` with the value tags of
  non-`complete` notes at `IGNORE_INDEX` (none were uncertain here: all 2,355 training notes
  supervise the value head), decoded as the highest-confidence value span. v2 recipe and
  hyperparameters otherwise unchanged (`run.sh`, `train_en_student.py --value`).
  The learned head that exists in this repo is value-span-v1 (gidi-finance-v2); gidi-finance-v4
  ships no learned value head (rule parser), and value-span-v4..v7 are frozen-encoder / CRF / dual
  encoder variants that do not fit the size budget.
* **Export**: `scripts/export_value_onnx.py`, outputs `type_logits, tag_logits, value_logits` (the
  Vietnamese value bundle's names), opset 17, FP32 parity max |diff| 4.6e-5, int8 dynamic QInt8.

## Value labels: agreement

| | notes | A (regex) = B (Clef) | adjudicated by Claude |
|---|---|---|---|
| test / dev / seed | 290 / 189 / 280 | 280 / 178 / 276 | 10 / 11 / 4 |
| slang probe / Gen Z probe | 357 / 94 | 350 / 92 | 7 / 2 |
| silver-train / silver-slang-train | 1,462 / 613 | 1,421 / 605 | 41 / 8 |
| **total** | **3,285** | **3,202 (97.5%)** | **83** |

3,165 of the notes hold an amount-shaped token (A and B agree on `none` for the other 120 without
a Clef call); agreement on those is 3,082 / 3,165 = 97.4%. Adjudication: A's span 75, B's 5, no
amount 2, own span 1; 0 `uncertain`. Disclosure: the first pass gave 173 disagreements; fixing
scanner gaps removed 90 of them (a glued `150s$`, Indian grouping `1,65,000`, `s&p 500` / `nifty 50` names,
`in 4 weeks` durations, `4`/`2` for *for*/*to*, `at three`). I fixed scanner A and re-ran both labelers (83
left). Clef answers `none` for a clear amount in most of the rest, so agreement is high mostly
because the candidate lists are shared. Consensus notes were never reviewed by a human or by Claude
(a 70-note spot check of the multi-number consensus notes found no error), so the labels favour
rule-like behaviour; see the scanner comparison.

## Snapping

Value spans are scored with whole-word snapping on, as the target is. A check on all 3,160 gold
value spans found snapping changes 2 (a leading `+`: `300` -> `+300`). The unsnapped value exact is
identical to the snapped one on every set (the head never stops inside a word).

## v2 vs v3 (int8, snapped; `results.json`)

| Set | n | metric | v2 | v3 | v3 - v2 | gate (>= v2 - 0.01) |
|---|---|---|---|---|---|---|
| test primary | 214 | type macro-F1 | 0.9561 | 0.9397 | -0.0164 | **FAIL** |
| | | target exact | 0.8879 | 0.8879 | 0.0000 | pass |
| | | joint (type+target) | 0.8505 | 0.8271 | -0.0234 | n/a |
| | | **value exact** | none | **0.9907** | | >= 0.93 **pass** |
| test secondary | 290 | type macro-F1 | 0.9020 | 0.8784 | -0.0236 | **FAIL** |
| | | target exact | 0.8345 | 0.8207 | -0.0138 | **FAIL** |
| | | joint | 0.7586 | 0.7276 | -0.0310 | n/a |
| | | **value exact** | none | **0.9931** | | >= 0.93 **pass** |
| slang probe | 357 | type macro-F1 | 0.8903 | 0.9034 | +0.0131 | pass |
| | | target exact | 0.8347 | 0.8207 | -0.0140 | **FAIL** |
| | | joint | 0.7535 | 0.7675 | +0.0140 | n/a |
| | | **value exact** | none | **0.9832** | | >= 0.88 **pass** |
| Gen Z probe (not gating) | 94 | type F1 / target / value | 0.8967 / 0.8511 / none | 0.8674 / 0.8511 / **1.000** | | |
| dev (not gating) | 189 | type F1 / target / value | 0.8016 / 0.8042 / none | 0.8342 / 0.8095 / **0.9947** | | |

Full joint (type + target + value exact): primary 0.8178, secondary 0.7207, slang 0.7563, Gen Z
0.7766, dev 0.6984. Value present/null accuracy: 1.000 on test and dev, 0.9944 slang, 1.000 Gen Z.
FP32 and int8 value exact are the same except dev (0.9894 fp32, 0.9947 int8).

### Noise control (not part of the verdict)

I re-trained v2 seed 1 with its unchanged command on the same code and machine
(`run_control_v2.sh`, `control-v2/`). It does not reproduce the recorded v2 (MPS is not
bit-reproducible):

| | primary type / target | secondary type / target | slang type / target | dev type / target |
|---|---|---|---|---|
| v2 as recorded | 0.9561 / 0.8879 | 0.9020 / 0.8345 | 0.8903 / 0.8347 | 0.8016 / 0.8042 |
| v2 re-run (control) | 0.9437 / 0.8925 | 0.8845 / 0.8379 | 0.9024 / 0.8123 | 0.8117 / 0.8413 |
| v3 | 0.9397 / 0.8879 | 0.8784 / 0.8207 | 0.9034 / 0.8207 | 0.8342 / 0.8095 |

The v2 re-run against the recorded v2 differs by up to -0.0175 (secondary type) and -0.0224
(slang target), the same order as the v3 gaps, and it would also fail the declared `v2 - 0.01`
gate on 3 of the 6 checks. v3 against the re-run: primary type -0.0040, secondary type -0.0061,
secondary target -0.0172, slang target +0.0084. So a single-seed comparison cannot separate "the
head costs 1-2 points" from "seed noise"; the pre-declared protocol still reads FAIL. Deciding
whether to accept v3 anyway, or to run more seeds, is a separate decision.

### Per-type F1 (v2 -> v3)

| type | primary (n) | secondary (n) | slang probe (n) |
|---|---|---|---|
| expense | 0.944 -> 0.925 (54) | 0.912 -> 0.890 (94) | 0.912 -> 0.930 (109) |
| income | 0.857 -> 0.833 (19) | 0.840 -> 0.824 (28) | 0.916 -> 0.941 (43) |
| borrow | 1.000 -> 1.000 (18) | 0.906 -> 0.830 (27) | 0.984 -> 0.968 (31) |
| lend | 0.983 -> 0.967 (29) | 0.892 -> 0.877 (36) | 0.897 -> 0.880 (41) |
| repayment_in | 0.974 -> 0.974 (20) | 0.875 -> 0.913 (22) | 0.795 -> 0.889 (39) |
| repayment_out | 0.950 -> 0.905 (20) | 0.898 -> 0.815 (24) | 0.750 -> 0.793 (28) |
| transfer | 0.940 -> 0.914 (40) | 0.894 -> 0.879 (44) | 0.921 -> 0.907 (46) |
| refund | 1.000 -> 1.000 (14) | 1.000 -> 1.000 (15) | 0.947 -> 0.919 (20) |

## Value: model vs the playground regex scanner

The scanner is `findEnglishValue` (`playground/src/app/amount.ts`), run unchanged through bun
(`app_scanner.ts`). It returns the number (and its scale) without `$`, `bucks` or other currency
words and cannot read word amounts, while the labels keep the marker, so `strict` is structurally
low for it; `core` strips currency symbols, codes and money words from both spans. The criterion
(c) is on `core`.

| set | n | model strict | model core | scanner strict | scanner core | gate (model core >= scanner core) |
|---|---|---|---|---|---|---|
| test primary | 214 | 0.9907 | 0.9907 | 0.9346 | 0.9673 | pass |
| test secondary | 290 | 0.9931 | 0.9931 | 0.9379 | 0.9724 | pass |
| slang probe | 357 | 0.9832 | 0.9832 | 0.1597 | 0.8487 | pass |
| Gen Z probe | 94 | 1.0000 | 1.0000 | 0.5000 | 0.9043 | |
| dev | 189 | 0.9947 | 0.9947 | 0.9365 | 0.9788 | |

Caveat (see `protocol.json`): scanner A that built the labels and the app scanner are both rule
scans for the same shapes, and 97.5% of the labels were never reviewed, so the scanner is
optimistic on the consensus test notes (core 0.967-0.979 there against 0.849 on the slang probe).
On the notes Claude adjudicated (A and Clef disagreed) the model is right on 6/6 (primary), 10/10
(secondary), 11/11 (dev), 5/7 (slang), 2/2 (Gen Z); the scanner on 6/6, 10/10, 10/11, 4/7, 2/2.

## Value errors by category

Test secondary + dev + slang probe + Gen Z probe pooled (n = 930 notes; a note can have several
categories). `two_numbers` = the note has at least two number-like tokens.

| category | n | model exact | scanner strict exact |
|---|---|---|---|
| plain bare number | 404 | 404 (1.000) | 402 (0.995) |
| two numbers | 139 | 137 (0.986) | 21 (0.151) |
| scaled (k, grand, lakh) | 33 | 32 (0.970) | 17 (0.515) |
| decimals | 72 | 72 (1.000) | 61 (0.847) |
| grouped thousands (`1,200`) | 43 | 43 (1.000) | 42 (0.977) |
| currency marker (`$`, `bucks`, `s$`, `rs`) | 348 | 343 (0.986) | 0 (0.000) |
| word amounts (`fifty dollars`, `tenner`) | 13 | 11 (0.846) | 0 (0.000) |
| no amount | 22 | 20 (0.909) | 20 (0.909) |

(Scanner strict is low on marked amounts by convention; its core scores are in the table above.)

## The value errors (all 9 on test secondary, dev, slang probe and Gen Z probe; primary's 2 are inside secondary)

| set | note | gold | predicted |
|---|---|---|---|
| test | `lent Desmond fifty dollars` | `fifty dollars` | `dollars` |
| test | `bought 2000 in nifty 50` | `2000` | `50` |
| dev | `recharged fastag wallet with one thousand rupees 1,000` | `one thousand rupees` | `1,000` |
| slang | `company pd annual performance bonus 3 grand rn` | `3 grand` | `3 grand rn` |
| slang | `annual renters insurance pymt 140$ rn` | `140$` | `140$ rn` |
| slang | `target store return refund credited 42 dollars eod` | `42 dollars` | `42 dollars eod` |
| slang | `idk why there is a 12 dollar charge` | `12 dollar` | `idk` |
| slang | `lunch on card 4 sam` | none | `4` |
| slang | `dinner on tab 4 priya` | none | `4` |

Only 9 value errors exist across the four sets, so there is no tenth. Four are slang abbreviations
(`rn`, `eod`) swallowed into the span after the amount, two are the `4` = *for* convention that
the labels follow and the model does not, two are word amounts or an index name.

## Size and runtime

int8 29,170,089 bytes (v2 29,166,949: +3,140 B; <= 30 MB). The int8 graph has the same set of op
types as v2 (+1 `MatMulInteger`, +2 `Mul`, +1 `Add`, +1 `Cast` for the head), opset 17, no new op.
FP32 parity 4.6e-5, int8 agreement with fp32 on 50 dev notes: type 0.98, target tags 1.00, value
tags 1.00. No bundle, release, deploy, HF push or journal entry was made.

## Pass/fail vs protocol

| condition | result |
|---|---|
| (a) value exact >= 0.93 on test primary and secondary | 0.9907 / 0.9931 pass |
| (a) value exact >= 0.88 on the slang probe | 0.9832 pass |
| (b) type macro-F1 >= v2 - 0.01: primary / secondary / slang | 0.9397 vs 0.9461 **fail** / 0.8784 vs 0.8920 **fail** / 0.9034 pass |
| (b) target exact >= v2 - 0.01: primary / secondary / slang | 0.8879 pass / 0.8207 vs 0.8245 **fail** / 0.8207 vs 0.8247 **fail** |
| (c) model core exact >= scanner core exact: primary / secondary / slang | pass / pass / pass |
| int8 <= 30 MB | 29.17 MB pass |
| **overall** | **FAIL** (b) |

Limits: one seed; silver labels (value labels 97.5% consensus of two rule-like labelers); slices
are small (word amounts 13, no-amount 22, scaled 33 pooled); the type/target regression and the
re-run control are both single runs.
