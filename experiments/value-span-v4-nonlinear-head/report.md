# value-span-v4-nonlinear-head: report

**Verdict: B. invariance passes, value quality fails: stop; no deeper head or unfreezing**

Protocol: `experiments/value-span-v4-nonlinear-head/protocol.json` (0444), written before
training. `gidi-finance-v1` and the value-span-v1, v2-direction-repair and v3-frozen-v1
experiments are unchanged. Numbers come from `comparison.json`, `invariance.json`,
`value-errors.jsonl`, `multi-number-errors.jsonl` and `train.log`.

## 1. Question

Is the frozen gidi-finance-v1 representation sufficient for value-span extraction when it is read
out by a small nonlinear head?

## 2. Setup

- **Base:** the trained v1 seed-1 checkpoint (safetensors sha256 `bfb2195a…0556`). The encoder,
  type head and target head are frozen.
- **Value head (the only trainable part):** `Linear(768, 256) → GELU → Dropout(0.1) →
  Linear(256, 3)`, 197,635 parameters.
- **Freezing checks** (all passed; recorded in each seed's `metrics.json`):
  - the optimizer held exactly the 4 value-head tensors;
  - all 73 frozen parameters had grad None after the first backward pass;
  - the frozen-state sha256 equals the base's before training, after training and in the saved
    checkpoint.
- **Data:** `annotation-v2/training-v1` unchanged (train 895). Uncertain value labels masked.
- **Recipe (predeclared, same head-level settings as v3):**
  - loss: value BIO cross-entropy only;
  - optimizer: AdamW lr 1e-3, weight decay 0.01 on weights only, batch 8;
  - schedule: 40 epochs, warmup 0.1 then linear decay, max grad norm 1.0;
  - only the value head is in train mode, so its dropout is active; everything else runs in eval
    mode;
  - last epoch kept.
- **Run:** about 45 s per seed on MPS, 4,480 steps. Final training value loss was 0.030–0.031
  (v3 linear head: 0.19; value-span-v1 with the encoder trainable: about 0.0002).

## 3. Invariance (passes)

All three seeds against the base checkpoint (CPU) on test (143, which contains test-v1's 105) and
on probe (81):

- maximum absolute difference in type and target logits: 0;
- 0 mismatches in type predictions, target BIO predictions and target spans;
- 0 mismatches against the frozen v1 seed-1 prediction files.

## 4. Value quality (fails)

| criterion | mean | seed 1 | threshold | result |
|---|---|---|---|---|
| test value exact | 0.8633 | 0.8633 | ≥ 0.95 | fail |
| human-only value exact | 0.7500 | 0.7500 | ≥ 0.90 | fail |
| present/null accuracy | 1.0000 | 1.0000 | ≥ 0.97 | pass |
| multi_number exact | 0.7273 | 0.7273 | ≥ 0.90 | fail |
| slice floor (n ≥ 5) | 0.5000 (bare number) | 0.5000 | ≥ 0.80 | fail |

Value metrics per set (mean over seeds):

| set | n | exact | span F1 | token F1 | present/null |
|---|---|---|---|---|---|
| test | 139 | 0.8633 | 0.8633 | 0.9609 | 1.0000 |
| test-v1 | 105 | 0.9079 | 0.9079 | 0.9753 | 1.0000 |
| test-targeted | 34 | 0.7255 | 0.7255 | 0.9200 | 1.0000 |
| probe | 81 | 0.8889 | 0.8889 | 0.9658 | 1.0000 |
| validation (monitoring) | 146 | 0.9224 | 0.9250 | 0.9738 | 0.9932 |

Test value slices (exact; mean, with seed 1 in brackets):

| slice | n | v4 | v3 linear | value-span-v1 |
|---|---|---|---|---|
| explicit_unit | 113 | 0.888 (0.894) | 0.817 | 0.988 |
| bare_number | 6 | 0.500 (0.500) | 0.667 | 0.944 |
| multi_number | 22 | 0.727 (0.727) | 0.606 | 0.955 |
| slang | 10 | 0.667 (0.600) | 0.100 | 1.000 |
| unaccented | 50 | 0.867 (0.880) | 0.820 | 0.993 |
| unseen_span | 37 | 0.730 (0.730) | 0.703 | 0.982 |
| long_multi_token | 63 | 0.820 (0.810) | 0.640 | 0.979 |
| no_amount | 0 | n/a | n/a | n/a |

## 5. Error categories

These count every wrong value on complete labels (`value-errors.jsonl`, 84 rows), per seed
(1 / 2 / 3):

| category | test | probe |
|---|---|---|
| wrong-number selection | 1 / 1 / 1 | 1 / 1 / 1 |
| boundary truncation | 15 / 15 / 15 | 6 / 6 / 7 |
| left overrun | 1 / 1 / 1 | 0 / 0 / 0 |
| right overrun into date/time | 0 / 0 / 0 | 0 / 0 / 0 |
| punctuation overrun | 2 / 2 / 2 | 1 / 1 / 1 |
| other right overrun | 0 / 0 / 0 | 1 / 0 / 1 |
| missed value | 0 / 0 / 0 | 0 / 0 / 0 |

The categories are defined by span position in `value_gate.value_error_category`.

Truncation dominates. The head finds the amount but drops a unit or subword piece:

- unit dropped: `3 triệu` → `3`, `150k` → `150`, `700 ngàn` → `700`, `4.5tr` → `4.5`,
  `2 lít` → `2`;
- number dropped: `3 cu` → `cu`, `2tr` → `tr`, `540k` → `k`;
- only a digit piece kept: `230k` → `2`, `170` → `0`, `612` → `12`, `1tr2` → `1tr`.

[INFERENCE] The token F1 of 0.961 against an exact score of 0.863 is consistent with this. The
nonlinear readout finds the amount token, but it does not reliably extend a span consistently
across the token pieces of one amount.

Wrong-number selection fell from 33 (v3) to 6 over three seeds.

## 6. Multi-number errors (inspected by hand)

`multi-number-errors.jsonl` has 24 rows over 8 notes:

| note | prediction | gold | kind |
|---|---|---|---|
| `gia han goi 4g 120k` | `g 120k` | `120k` | left overrun |
| `Thắng vay 5 củ, hẹn t10 trả` | `5 củ,` | `5 củ` | punctuation overrun |
| `tiền điện nhà trọ tháng 10 hết 612` | `12` | `612` | truncation |
| `tiền điện nhà số 12 tháng 8 540k` | `k` | `540k` | truncation |
| `2 vé cgv 170` | `0` | `170` | truncation |
| `tra no chi Mai 300 hom 5/10` | `5` | `300` | wrong number (the date's day) |
| probe: `… trả kỳ tháng 12 9tr5` | `tr5` | `9tr5` | truncation |
| probe: `khoan vay online tamo dong ky 2 900k` | `2` | `900k` | wrong number (instalment index) |

## 7. Tracked cases (held-out; reported only, never used for training or tuning)

- `cho a Nam vay 1 triệu 20/10`: every seed predicts `1 triệu` (exact). The trailing-date overrun
  seen in value-span-v1 and the direction repair does not occur.
- `Thắng vay 5 củ, hẹn t10 trả`: every seed predicts `5 củ,`. v3 predicted `5 củ, hẹn t10`; here
  the head now stops before the date but keeps the comma. It is no longer a date/time overrun,
  but it is still a punctuation overrun and still a wrong value.
- No test or probe note has a right overrun into a date or time in any seed.

## 8. Interpretation

Invariance holds, so v1's type and target behaviour is preserved exactly. The nonlinear readout
improves clearly on the linear head: test exact 0.765 → 0.863, slang 0.10 → 0.67, wrong-number
errors 33 → 6. It is still far below the frozen thresholds and below value-span-v1 (0.988).

The frozen v1 representation therefore carries much of the information about where the amount is,
but not enough for a small token-wise readout to get exact amount boundaries. The remaining gap is
mostly subword boundary consistency.

[INFERENCE] Closing it would plausibly need either context-aware decoding across tokens or an
adapted encoder. Both are outside this experiment's scope and were not started.

## 9. Outcome

- Verdict B: stop. No export, no runtime or playground change. `gidi-finance-v1` is unchanged.
- `models/value-span-v4-nonlinear-head/seed{1,2,3}` are kept as evidence (PyTorch only).
- No further experiment was started.

## 10. Artifacts

| path | content |
|---|---|
| `protocol.json` | frozen protocol (0444) |
| `train.log` | training log |
| `models/value-span-v4-nonlinear-head/seed{n}/metrics.json` | freezing checks |
| `invariance.json` | type/target invariance vs base and vs v1 prediction files |
| `comparison.json`, `eval/` | value metrics, slices, criteria, error counts, tracked notes, verdict |
| `value-errors.jsonl`, `multi-number-errors.jsonl` | every value error with its category; multi-number errors |

Code changes:

- `src/gidi/modeling/value.py`: a `value_head_arch` option (`linear` | `mlp`); linear
  checkpoints are unchanged.
- `src/gidi/training/train_value_head.py` and `scripts/train_value_head.py`: `--head-arch` and
  `--experiment`.
- `src/gidi/evaluation/value_gate.py`: `value_error_category` and `value_error_table`.
- `scripts/evaluate_frozen_value.py`: the experiment is read from the protocol; adds error
  categories and tracked notes.
- Tests: an MLP freezing case in `tests/test_train_value_head.py`, and
  `test_value_error_category` in `tests/test_value_gate.py`.

B. frozen v1 + nonlinear value head preserves v1 exactly but value quality is below the frozen
thresholds
