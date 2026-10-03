# value-span-v5-crf: report

**Verdict: B. Invariance passes, value quality fails. Stop: no BiLSTM, transformer layers,
adapters or unfreezing.**

The protocol `experiments/value-span-v5-crf/protocol.json` (0444, sha256 `5767eb2b…f199`) was
written before training. The following are unchanged:

- `gidi-finance-v1`;
- value-span-v1, v2-direction-repair, v3-frozen-v1 and v4-nonlinear-head, including their
  protocols.

Numbers come from `comparison.json`, `invariance.json`, `value-errors.jsonl`,
`multi-number-errors.jsonl` and `train.log`.

## 1. Question

Does sequence-aware linear-chain CRF decoding over the frozen gidi-finance-v1 representation fix
value-span continuity (boundary truncation, the dominant v4 error) while keeping type/target
exactly v1?

## 2. Setup

- **Base:** the trained v1 seed-1 checkpoint (safetensors sha256 `bfb2195a…0556`). The encoder,
  type head and target head are frozen.
- **Value path (the only trainable part, head arch `mlp-crf`):**
  - emissions: `Linear(768,256) → GELU → Dropout(0.1) → Linear(256,3)`, identical to v4;
  - a linear-chain CRF over O / B-VALUE / I-VALUE: start [3], end [3] and transitions [3×3],
    learned only, with no hard BIO constraints;
  - 197,650 parameters in total.
- **Initialisation:**
  - Linear layers as in v4;
  - CRF uniform(-0.1, 0.1) from a private generator seeded `1_000_003 + seed + 2_000_000`.
  - Seeds 1–3 vary only the head initialisation and batch order.
- **Objective:** CRF NLL only, mean over notes with complete value labels; uncertain notes are
  excluded. There is no token cross-entropy term.
- **CRF sequence:** a note's real tokens, which are exactly its value-label positions (checked on
  train, validation and test: identical masks, contiguous).
- **Decoding:** Viterbi over the real tokens, then the unchanged value-span rule and best-span
  choice.
- **Recipe:** same as v4.
  - AdamW lr 1e-3; weight decay 0.01 on the two Linear weight matrices only (biases and CRF
    transitions undecayed); batch 8.
  - 40 epochs, warmup 0.1 then linear decay, max grad norm 1.0.
  - Last epoch kept; no sweep.
  - Data: `annotation-v2/training-v1` unchanged (train 895).
- **Freezing checks** (all passed; recorded in each seed's `metrics.json`):
  - the optimizer held exactly the 7 value-path tensors (4 emission + 3 CRF);
  - all 73 frozen parameters had grad None after the first backward pass;
  - the frozen-state sha256 equals the base's before training, after training and in the saved
    checkpoint.
- **Run:** about 65 s per seed (MPS), 4,480 steps. Final training NLL was 0.087–0.092 per note.
  Validation exact at epoch 40 (Viterbi; monitoring only) was 0.952 / 0.938 / 0.952 for seeds
  1 / 2 / 3.

**Evaluation fix (before any verdict was recorded).** The first evaluation run decoded the value
metrics by per-token argmax instead of Viterbi. `value_eval.evaluate_records`, used by
`compare_value_v1.score_model`, did not receive the CRF; the invariance path and the training
validation already used Viterbi. A spot check found the mismatch: the record
`hạt cho chó 420.000đ` decodes to `420.000đ` with Viterbi but was reported as `000đ`.

I fixed only that plumbing (a `crf` argument passed through) and re-ran the evaluation with
`--overwrite`. The protocol, training and checkpoints were unchanged. All numbers below are from
the corrected Viterbi run. For reference, the argmax-decoded emissions of this model scored test
exact 0.839, below v4's 0.863: the CRF-trained emissions alone are worse, and the gain comes from
decoding.

## 3. Invariance (passes)

All three seeds against the base checkpoint (CPU) on test (143, which contains test-v1's 105 and
test-targeted's 38) and on probe (81):

- maximum absolute difference in type and target logits: 0;
- 0 mismatches in type predictions, target BIO predictions and target spans;
- 0 mismatches against the frozen v1 seed-1 prediction files.

## 4. Value quality (fails)

| criterion | mean | seed 1 | threshold | result | v4 mean |
|---|---|---|---|---|---|
| test value exact | 0.9017 | 0.8993 | ≥ 0.95 | fail | 0.8633 |
| human-only value exact | 0.7500 | 0.7500 | ≥ 0.90 | fail | 0.7500 |
| present/null accuracy | 1.0000 | 1.0000 | ≥ 0.97 | pass | 1.0000 |
| multi_number exact | 0.6818 | 0.6818 | ≥ 0.90 | fail | 0.7273 |
| slice floor (n ≥ 5) | 0.5000 (bare number) | 0.5000 | ≥ 0.80 | fail | 0.5000 |

Value metrics per set (mean over seeds):

| set | n | exact | span F1 | token F1 | present/null | v4 exact |
|---|---|---|---|---|---|---|
| test | 139 | 0.9017 | 0.9017 | 0.9681 | 1.0000 | 0.8633 |
| test-v1 | 105 | 0.9333 | 0.9333 | 0.9769 | 1.0000 | 0.9079 |
| test-targeted | 34 | 0.8039 | 0.8039 | 0.9435 | 1.0000 | 0.7255 |
| probe | 81 | 0.9506 | 0.9565 | 0.9859 | 0.9877 | 0.8889 |
| validation (monitoring) | 146 | 0.9475 | 0.9504 | 0.9829 | 0.9932 | 0.9224 |

Test value slices (exact; mean, with seed 1 in brackets):

| slice | n | v5 CRF | v4 MLP | v3 linear | value-span-v1 |
|---|---|---|---|---|---|
| explicit_unit | 113 | 0.920 (0.920) | 0.888 | 0.817 | 0.988 |
| bare_number | 6 | 0.500 (0.500) | 0.500 | 0.667 | 0.944 |
| multi_number | 22 | 0.682 (0.682) | 0.727 | 0.606 | 0.955 |
| slang | 10 | 0.833 (0.800) | 0.667 | 0.100 | 1.000 |
| unaccented | 50 | 0.900 (0.900) | 0.867 | 0.820 | 0.993 |
| unseen_span | 37 | 0.820 (0.811) | 0.730 | 0.703 | 0.982 |
| long_multi_token | 63 | 0.894 (0.889) | 0.820 | 0.640 | 0.979 |

## 5. Error categories vs v4 (main question)

These count every wrong value on complete labels, per seed (1 / 2 / 3):

| category | test v5 | test v4 | probe v5 | probe v4 |
|---|---|---|---|---|
| wrong-number selection | 1 / 1 / 1 | 1 / 1 / 1 | 0 / 0 / 0 | 1 / 1 / 1 |
| **boundary truncation** | **8 / 7 / 7** | **15 / 15 / 15** | **2 / 2 / 2** | **6 / 6 / 7** |
| left overrun | 1 / 1 / 2 | 1 / 1 / 1 | 1 / 1 / 1 | 0 / 0 / 0 |
| right overrun into date/time | 2 / 2 / 2 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| punctuation overrun | 2 / 2 / 2 | 2 / 2 / 2 | 0 / 0 / 0 | 1 / 1 / 1 |
| other right overrun | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 1 / 0 / 1 |
| missed value | 0 / 0 / 0 | 0 / 0 / 0 | 1 / 1 / 1 | 0 / 0 / 0 |
| total | 14 / 13 / 15 | 19 / 19 / 19 | 4 / 4 / 4 | 9 / 8 / 10 |

**The CRF roughly halves boundary truncation on test (15 → 7–8) and cuts it by about two-thirds
on probe (6–7 → 2).**

Fixed in seed 1:

- unit or word kept: `3 triệu`, `3 cu`, `2 củ`, `700 ngàn`, `4.5tr`, probe `3 triệu`, `2tr`,
  `10tr`;
- full number kept: `890k`;
- overruns removed: probe `5 triệu,` and `1tr đóng`.

Remaining truncations in seed 1, all unchanged from v4:

- `150k` → `150`
- `230k` → `2`
- `2 lít` → `2`
- `612` → `12`
- `1 trieu 5` → `5`
- `540k` → `k`
- `170` → `0`
- `250 ngan` → `250`
- probe `9tr5` → `tr5`

Seed 1 also has `1tr2` → `k`, which is classified as a wrong number. The token view shows what
remains. In `540k` and `170` the emission for the leading number piece is strongly O, and the CRF
does not override it: the path starts at the unit piece with an I tag. In `230k` the path is
B B I. The emission for `30` is B, so `2` and `30k` become two spans, and the confidence rule
keeps `2`.

The CRF also introduced new right overruns into dates and times:

- `cho a Nam vay 1 triệu 20/10` → `1 triệu 20/10`;
- `tra no chi Mai 300 hom 5/10` → `300 hom` (in v4 this was a wrong-number error, `5`).

## 6. Multi-number errors (inspected by hand)

`multi-number-errors.jsonl` has 27 rows. Seed 1 has 9 notes:

| note | prediction | gold | kind |
|---|---|---|---|
| `gia han goi 4g 120k` | `g 120k` | `120k` | left overrun (as v4) |
| `cho a Nam vay 1 triệu 20/10` | `1 triệu 20/10` | `1 triệu` | date overrun (new; v4 exact) |
| `Thắng vay 5 củ, hẹn t10 trả` | `5 củ,` | `5 củ` | punctuation overrun (as v4) |
| `tiền điện nhà trọ tháng 10 hết 612` | `12` | `612` | truncation (as v4) |
| `tiền điện nhà số 12 tháng 8 540k` | `k` | `540k` | truncation (seed 3: `8 540k`, left overrun) |
| `2 vé cgv 170` | `0` | `170` | truncation (as v4) |
| `tra no chi Mai 300 hom 5/10` | `300 hom` | `300` | right overrun (v4: wrong number `5`) |
| probe `… trả kỳ tháng 12 9tr5` | `tr5` | `9tr5` | truncation (as v4) |
| probe `khoan vay online tamo dong ky 2 900k` | `2 900k` | `900k` | left overrun (v4: wrong number `2`) |

Multi-number exact drops from 0.727 to 0.682. That is one note, `cho a Nam…`, now overrunning
into the date.

## 7. Tracked cases (held-out; reported only, never used for training or tuning)

- `cho a Nam vay 1 triệu 20/10`: every seed predicts `1 triệu 20/10`, a right overrun into the
  date. This is a regression from v4, which was exact; the date overrun seen in value-span-v1 is
  back.
- `Thắng vay 5 củ, hẹn t10 trả`: every seed predicts `5 củ,`, a punctuation overrun, the same as
  v4.

## 8. Interpretation

Invariance holds exactly. On the main question the CRF helps, and substantially:

- boundary truncation halves;
- test exact 0.863 → 0.902, probe 0.889 → 0.951;
- every long, slang and unseen slice improves.

It is not enough. The gate fails on four of five criteria:

- human-only exact stays at 0.75;
- multi_number falls to 0.68;
- bare_number stays at 0.50.

The learned transitions favour continuing a span (B→I 1.12, I→I 0.29; O→I -1.03), which repairs
unit-dropping truncations. The same pull now extends spans into following date/number tokens.

[INFERENCE] The remaining truncations are emission errors that a first-order CRF over these
frozen features cannot override: a strongly-O leading number piece, or a B restart inside a
number. The frozen v1 token features still under-separate amount pieces from neighbouring
numbers and dates.

## 9. Outcome

- Verdict B: stop. No export, no runtime or playground change. `gidi-finance-v1` is unchanged.
- `models/value-span-v5-crf/seed{1,2,3}` are kept as evidence (PyTorch only).
- No further experiment was started.

## 10. Artifacts

| path | content |
|---|---|
| `protocol.json` | frozen protocol (0444) |
| `train.log` | training log |
| `models/value-span-v5-crf/seed{n}/metrics.json` | freezing checks |
| `invariance.json` | type/target invariance vs base and vs v1 prediction files |
| `comparison.json`, `eval/` | value metrics, slices, criteria, error counts, tracked notes, verdict |
| `value-errors.jsonl`, `multi-number-errors.jsonl` | every value error with its category; multi-number errors |

Code changes (additive; linear and mlp heads unchanged):

- `src/gidi/modeling/crf.py` (new): `LinearChainCRF` (masked NLL, pytorch-crf init),
  `viterbi` and `viterbi_masked`.
- `src/gidi/modeling/value.py`: head arch `mlp-crf` (`CRFValueHead`: MLP emissions + CRF);
  `value_crf(model)` returns the transitions or `None`.
- `src/gidi/modeling/value_decode.py`: `decode_value(..., crf=None)` uses the Viterbi tags when a
  CRF is given.
- `src/gidi/evaluation/value_predict.py`, `value_eval.py` and `scripts/compare_value_v1.py`:
  pass `crf` through decoding.
- `src/gidi/training/train_value.py`: Viterbi in the validation row; CRF NLL as the validation
  value loss.
- `src/gidi/training/train_value_head.py`: CRF NLL loss for `mlp-crf`; CRF transitions are not
  weight-decayed.
- `scripts/evaluate_frozen_value.py`: Viterbi in `run_model`.
- Tests:
  - `tests/test_crf.py`: NLL and Viterbi against brute-force enumeration, plus a
    transition-override case;
  - `tests/test_train_value_head.py`: an mlp-crf freezing case.
