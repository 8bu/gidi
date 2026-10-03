# value-span-v3-frozen-v1: report

**Verdict: B. type/target identical to gidi-finance-v1, value quality fails the frozen thresholds; stop**

Protocol: `experiments/value-span-v3-frozen-v1/protocol.json` (0444), written before training.
`gidi-finance-v1`, value-span-v1 and value-span-v2-direction-repair are unchanged. Numbers
come from `comparison.json`, `invariance.json` and `train.log`.

## 1. Question

Can the trained gidi-finance-v1 encoder representations support value-span extraction when the
whole v1 model is frozen and only a new value BIO head is trained?

## 2. Setup

- **Base:** the trained v1 seed-1 checkpoint
  (`models/compression-v3/supervised/…-K2048/seed1`, safetensors sha256 `bfb2195a…0556`), the
  source of `gidi-finance-v1`. It was loaded as trained, not re-initialised.
- **Frozen:** all encoder parameters and the type and target heads. The only trainable part is
  the value head, `Linear(768, 3)`, 2,307 parameters.
- **Checks before and during training,** all passed and recorded in each seed's `metrics.json`:
  - the optimizer holds only `value_head.weight` and `value_head.bias`;
  - all 73 frozen parameters had no gradient after the first backward pass;
  - the hash of the frozen weights equals the base's before training and after the last epoch.
- **Data:** `annotation-v2/training-v1`, unchanged (train 895, no direction-repair rows).
  Uncertain value labels are masked.
- **Recipe (predeclared):**
  - loss: value BIO cross-entropy only;
  - optimizer: AdamW, lr 1e-3, weight decay 0.01 (weight only), batch 8;
  - schedule: 40 epochs, warmup 0.1 then linear decay, max grad norm 1.0;
  - the model runs in eval mode throughout (no dropout);
  - last epoch kept;
  - seeds 1–3 vary only the value-head initialisation and the batch order.

  Each seed took about 40 s on MPS, 4,480 steps. Final training value loss was 0.192–0.193 for
  every seed. Value-span-v1 reached about 0.0002 with the encoder trainable.

## 3. Invariance (passes)

All three seeds were compared with the base checkpoint on CPU over test (143, which includes the
105 test-v1 notes) and probe (81):

- maximum absolute difference in type logits and target logits: 0;
- 0 mismatches in type predictions, target BIO predictions and target spans;
- 0 mismatches against the frozen v1 seed-1 prediction files from value-span-v1.

Type and target behaviour is therefore exactly v1's: test-v1 type accuracy 0.9524 and target
exact 0.8857; probe type accuracy 0.7778 and joint 0.6049.

## 4. Value quality (fails)

| criterion | mean | seed 1 | threshold | result |
|---|---|---|---|---|
| test value exact | 0.7650 | 0.7626 | ≥ 0.95 | fail |
| human-only value exact | 0.4167 | 0.4167 | ≥ 0.90 | fail |
| present/null accuracy | 0.9712 | 0.9712 | ≥ 0.97 | pass |
| multi_number exact | 0.6061 | 0.5909 | ≥ 0.90 | fail |
| slice floor (n ≥ 5) | 0.1000 (slang) | 0.1000 | ≥ 0.80 | fail |

Value exact by test slice, mean over seeds:

| slice | n | exact |
|---|---|---|
| explicit_unit | 113 | 0.817 |
| bare number | 6 | 0.667 |
| multi_number | 22 | 0.606 |
| slang | 10 | 0.100 |
| unaccented | 50 | 0.820 |
| unseen span | 37 | 0.703 |
| long or multi-token | 63 | 0.640 |
| no amount | 0 | n/a |

Value exact by set, mean over seeds:

| set | n | value exact |
|---|---|---|
| test-v1 | 105 | 0.829 |
| test-targeted | 38 | 0.569 |
| probe | 81 | 0.840 |
| validation | 149 | 0.797 |

For comparison, value-span-v1 scored 0.988 on test with the encoder trainable.

## 5. Multi-number errors (inspected by hand)

`multi-number-errors.jsonl` has 41 rows. They fall into three kinds:

- **Picks the wrong number, usually a month, period or instalment index:**
  - `tien phong thang 11 3tr2` → `11`
  - `thưởng kpi quý 3 3tr2` → `3`
  - `trả góp kỳ 3 1tr5` → `3`
  - `tra no chi Mai 300 hom 5/10` → `5`
  - probe: `quý 3 1tr8` → `3`; `ky 5 … 7tr6` → `5`; `ky 3 1tr` → `3`
- **Fragments of a value token:**
  - `góp xe tháng 12 1tr9` → `1tr`
  - `tiền điện nhà số 12 tháng 8 540k` → `k`
- **Spans that run past the amount:**
  - `gia han goi 4g 120k` → `4g 120k` (seed 3: `4g`)
  - `Thắng vay 5 củ, hẹn t10 trả` → `5 củ, hẹn t10`
  - probe: `tháng 12 9tr5` → `12 9tr5`; `ky 2 900k` → `2 900k`
- `ck 3tr qua tk 1234 tiết kiệm` → `k`: this `k` comes from `tk`, outside the gold span, so it
  is a wrong-number error rather than a fragment (see §5.1).

### 5.1 Tracked failure: trailing-date overrun (`cho a Nam vay 1 triệu 20/10`)

This note is held-out evaluation data. It was not used to build training data or tune
anything; this section only reports it.

- **Tracked note:** gold `1 triệu`. value-span-v1 predicted `1 triệu 20/10` in seeds 1 and 3,
  and value-span-v2-direction-repair did so in all three seeds. In this experiment every seed
  predicts `1 triệu` exactly, so the overrun does not recur.

Every wrong value on complete labels was classified by span position, using
`eval/seed*-{test,probe}-predictions.jsonl` and summing over 3 seeds:

| error kind | test | probe |
|---|---|---|
| wrong number selected (no overlap with gold) | 24 | 9 |
| truncation (prediction is a fragment inside gold) | 54 | 19 |
| overrun to the left (absorbs a preceding number) | 2 | 6 |
| overrun to the right, absorbing a date or time | 3 | 0 |
| overrun to the right, other (trailing comma) | 3 | 2 |
| missed (no value predicted) | 12 | 3 |

- **Right-side date/time absorption** appears on one note only: `Thắng vay 5 củ, hẹn t10 trả` →
  `5 củ, hẹn t10` in all 3 seeds (gold `5 củ`). This is the same failure family as the tracked
  note.
- **Other right-side overruns** only add trailing punctuation:
  - `cho Lợi 1tr, hẹn cuối tháng trả` → `1tr,`
  - probe: `Tuấn cho vay 5 triệu, hẹn tháng sau trả` → `5 triệu,`
- **Date or period numbers are mostly a selection error here, not an overrun:** the head picks
  the month, quarter or instalment number instead of the amount (`thang 11 3tr2` → `11`,
  `quý 3 3tr2` → `3`, `hom 5/10` → `5`), or absorbs it on the left (`tháng 12 9tr5` →
  `12 9tr5`).

Known limitation: absorption of a trailing date or time is still present on
`Thắng vay 5 củ, hẹn t10 trả`. The tracked note itself is fixed in this model, but this does not
change the verdict, which fails on value quality overall.

## 6. Interpretation

Freezing preserves v1 exactly, so the integration is correct. A single linear value head on the
frozen v1 token representations does not fit the value task: training loss plateaued near 0.19
in every seed. It fails mostly on the cases that need context, such as telling an amount from a
period or index number, recognising slang units, and handling multi-token amounts.
[INFERENCE] The v1 encoder was trained for type and target only. Its token features do not
appear to encode amount boundaries well enough for a linear read-out, and value-span-v1's quality
came from adapting the encoder. This was not tested further.

## 7. Outcome

- Verdict B: stop. No export, no runtime or playground change. `gidi-finance-v1` remains deployed
  and unchanged.
- `models/value-span-v3-frozen-v1/seed{1,2,3}` are kept as evidence (PyTorch only).
- No further experiment was started.

## 8. Artifacts

| path | content |
|---|---|
| `protocol.json` | frozen protocol (0444) |
| `train.log` | training log |
| `models/value-span-v3-frozen-v1/seed{n}/metrics.json` | freezing checks and final-epoch validation |
| `invariance.json` | type/target invariance vs base and vs v1 prediction files |
| `comparison.json`, `eval/`, `multi-number-errors.jsonl` | value metrics, slices, criteria, verdict |
| `src/gidi/training/train_value_head.py`, `scripts/train_value_head.py`, `scripts/evaluate_frozen_value.py`, `tests/test_train_value_head.py` | code |

B. type/target identical to v1; value-span quality is not sufficient with a frozen v1 encoder
