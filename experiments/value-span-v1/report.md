# value-span-v1: v2 training result

**Verdict: B. value-span works but type/target regressions need fixing**

Frozen protocol: `experiments/value-span-v1/protocol.json` (sha256 `a23ac226e08578380dcf2cca7a2aea4d2a5f8c1b93f880cb4af5f276f863ce28`, declared before training). Every number below is copied from `comparison.json`, `train.log`, `datasets/annotation-v2/training-v1/manifest.json`, `data-audit-training.json` or `review-score.json`; nothing is recomputed by hand. Std is the sample standard deviation (n-1) over the three seeds, as written by `scripts/compare_value_v1.py`.

Summary:

- The value BIO head works. Value quality passes every adoption criterion on the mean over seeds and on seed 1 (10 of 10 criteria; test value exact 0.9880 mean, present/null accuracy 1.0000).
- The regression gate fails on one criterion: **seed 1 type accuracy on test-v1 drops 4 notes (v1 seed 1 100/105 → v2 seed 1 96/105) against a limit of 3.** Every other regression criterion passes.
- Per the frozen protocol the outcome is verdict B. No bundle was exported, `gidi-finance-v1` is untouched and the playground is unchanged.

## 1. Final dataset

`datasets/annotation-v2/training-v1/` (`manifest.json`).

| file | rows | role |
| --- | --- | --- |
| train.jsonl | 895 | 723 v1 train rows + 172 targeted-value-01 rows; the 110 frozen v1 validation rows are inside (in-sample, as in v1) |
| validation.jsonl | 149 | 110 frozen v1 validation (also in train) + 39 targeted-value-01 (held out); monitoring only |
| test.jsonl | 143 | 105 frozen v1 test (test-v1) + 38 targeted-value-01 (test-targeted); evaluation only |
| probe-v1-eval-only.jsonl | 81 | diagnostic; evaluation only, never trained on |

targeted-value-01 contributes 249 records (4 of 253 skipped as not-complete; 228 generation groups, split by group with seed `value-span-v1`). Leakage gate: 0 failures over 249 checked records, max eval similarity 0.844.

## 2. Value-label provenance

| split | records | with value span | no_amount (complete, null) | masked (uncertain) | rule provenance | human provenance |
| --- | --- | --- | --- | --- | --- | --- |
| train | 895 | 879 | 10 | 6 | 807 | 88 |
| validation | 149 | 144 | 2 | 3 | 128 | 21 |
| test | 143 | 139 | 0 | 4 | 127 | 16 |
| probe (eval only) | 81 | 81 | 0 | 0 | 76 | 5 |

By source: train: rule 689, human 33, human-uncertain 1, extra-auto 118, extra-human 54; validation: rule 99, human 11, extra-auto 29, extra-human 10; test: rule 98, human 7, extra-auto 29, extra-human 9; probe_eval_only: rule 76, human 5.

Human value labels (`datasets/annotation-v2/value-labels.jsonl`): 46; 0 differ from the rule proposal. Value loss is applied only to `value_status == complete`; uncertain labels are masked (type/target still trained).

Tiered review score (`review-score.json`):

| dataset | must-review | hard samples | clean audit | escalation |
| --- | --- | --- | --- | --- |
| existing-value-review | 14 queued; errors 1 | 17 queued; errors 0 | 15 queued; errors 0 | accept_auto_remainder |
| targeted-value-01 | 36 queued; errors 16 | 21 queued; errors 0 | 20 queued; errors 0 | accept_auto_remainder |

Must-review errors are expected and only reported (the protocol rule); no hard stratum and no clean-audit sample triggered escalation, so the unsampled rule/synthetic-auto labels stayed trusted. Value metrics on rule-provenance labels measure agreement with audited rule labels (see §12).

## 3. Training-data audit

`data-audit-training.{json,md}` (mode `training`, dir `datasets/annotation-v2/training-v1`):

- records: 1158 unique (1077 excluding probe); by split train 785, validation 149, test 143, probe 81. The audit's train count (785) is the 895 train rows less the 110 validation rows that train.jsonl also contains [INFERENCE: arithmetic match, 895 − 110 = 785].
- with value span 1133, null value span 25; states rule 1039, human 106, masked_uncertain 13.
- multi-number notes 167; accented 666 / unaccented 492.
- every required amount form is covered in train; the only absent form is the extra `2 triệu rưỡi` (0 notes in any split).
- surface patterns vs train (spans / pattern unseen in train / exact span unseen in train): validation 144 / 0 / 38; test 139 / 0 / 40; probe 81 / 0 / 6.
- value spans take 2.39 v1 tokens on average (median 2, p95 4, max 7); max_length 32; no note truncated at 32 tokens and no span loses tokens to truncation.

## 4. Frozen protocol summary

- sha256 `a23ac226e08578380dcf2cca7a2aea4d2a5f8c1b93f880cb4af5f276f863ce28` (`experiments/value-span-v1/protocol.json`, read-only).
- Question: does adding a value BIO head to the gidi-finance-v1 architecture give exact value-span extraction without materially regressing type/target quality? Numeric normalization, taxonomy, compression/KD/width/depth/tokenizer changes are out of scope.
- Architecture: BamiBERT-derived 4×768 (layers 2,5,8,11), FFN 2048, 34 position rows, vocabulary B-rank-8000, max_length 32. Heads: type classifier, target BIO, value BIO (O/B-VALUE/I-VALUE; 768×3 = 2,307 parameters). Same `initial_model(seed)` as compression-v3; type and target heads bit-identical to v1 for the same seed.
- Selection: last epoch, no early stopping, no sweep; export seed = seed 1 (predeclared); test and probe never pick the seed; mean and per-seed reported.
- Adoption: value_quality (test value exact ≥ 0.95, human-label exact ≥ 0.90, present/null ≥ 0.97, multi_number exact ≥ 0.90, no test slice with n ≥ 5 under 0.80, seed 1 meets the same) AND no material regression vs v1. Verdict A = both pass, B = value passes and regression fails, C = value quality fails.
- Material regression: mean over seeds drops by more than 2 notes on the 105-note test (type accuracy, target exact or joint: > 0.019) or more than 2 notes on probe joint (> 0.025), OR seed 1 alone drops by more than 3 notes on any of those, OR the lender-first probe slice loses more than 1 note on mean type accuracy.

## 5. Training

Recipe (compression-v3 supervised recipe, unchanged): AdamW, encoder lr 5e-5, head lr 1e-3, batch 8, 40 epochs, warmup 0.1, weight decay 0.01, dropout 0.1, max grad norm 1.0; loss = type CE + target BIO CE + value BIO CE (equal weights); no KD. 4,480 steps per seed, 28,498,702 parameters, device `mps`. Command:

```sh
uv run python scripts/train_value_student.py --train datasets/annotation-v2/training-v1/train.jsonl --validation datasets/annotation-v2/training-v1/validation.jsonl --seed 1 --seed 2 --seed 3 --out-dir models/value-span-v1
```

| seed | final epoch | type loss | target loss | value loss | wall time (s) | train time (s) | checkpoint |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 40 | 4.75e-05 | 2.19e-05 | 4.75e-05 | 306.6 | 302.1 | `models/value-span-v1/seed1` |
| 2 | 40 | 5.85e-05 | 1.53e-04 | 2.86e-05 | 300.0 | 296.2 | `models/value-span-v1/seed2` |
| 3 | 40 | 5.52e-05 | 3.64e-05 | 2.28e-05 | 301.5 | 297.5 | `models/value-span-v1/seed3` |

Final train-epoch losses are near zero on all three heads. MPS kernels are not bit-reproducible (warning in `train.log`), so a repeated run of a seed may differ.

## 6. Results

v2 (`models/value-span-v1/seed{1,2,3}`). Sets: test (143), test-v1 (105 frozen v1 test rows), test-targeted (38 targeted-value-01 test rows), probe (81). `validation` is monitoring only (§6.5). Value metrics are scored on `value_status == complete` records only, so the value n is smaller than the set size where labels are masked.

### 6.1 test (n = 143; value n = 139, human-only value n = 12)

| metric | seed 1 | seed 2 | seed 3 | mean ± std |
| --- | --- | --- | --- | --- |
| type accuracy | 0.9231 | 0.9231 | 0.9231 | 0.9231 ± 0.0000 |
| type macro-F1 | 0.9312 | 0.9389 | 0.9372 | 0.9358 ± 0.0041 |
| target F1 | 0.9130 | 0.8571 | 0.9000 | 0.8901 ± 0.0292 |
| target exact | 0.9441 | 0.9161 | 0.9301 | 0.9301 ± 0.0140 |
| type+target joint | 0.8881 | 0.8531 | 0.8811 | 0.8741 ± 0.0185 |
| value precision | 0.9784 | 0.9928 | 0.9928 | 0.9880 ± 0.0083 |
| value recall | 0.9784 | 0.9928 | 0.9928 | 0.9880 ± 0.0083 |
| value F1 | 0.9784 | 0.9928 | 0.9928 | 0.9880 ± 0.0083 |
| value token F1 | 0.9909 | 0.9969 | 0.9985 | 0.9954 ± 0.0040 |
| value exact | 0.9784 | 0.9928 | 0.9928 | 0.9880 ± 0.0083 |
| value present/null accuracy | 1.0000 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| full joint (type ∧ target ∧ value exact) | 0.8633 | 0.8417 | 0.8705 | 0.8585 ± 0.0150 |
| value exact, human-only (n = 12) | 0.9167 | 1.0000 | 0.9167 | 0.9444 ± 0.0481 |

### 6.2 test-v1 (n = 105; value n = 105, human-only value n = 7)

| metric | seed 1 | seed 2 | seed 3 | mean ± std |
| --- | --- | --- | --- | --- |
| type accuracy | 0.9143 | 0.9143 | 0.9143 | 0.9143 ± 0.0000 |
| type macro-F1 | 0.9191 | 0.9319 | 0.9289 | 0.9266 ± 0.0067 |
| target F1 | 0.8932 | 0.8381 | 0.9126 | 0.8813 ± 0.0387 |
| target exact | 0.9333 | 0.9048 | 0.9333 | 0.9238 ± 0.0165 |
| type+target joint | 0.8667 | 0.8286 | 0.8762 | 0.8571 ± 0.0252 |
| value precision | 0.9810 | 0.9905 | 0.9905 | 0.9873 ± 0.0055 |
| value recall | 0.9810 | 0.9905 | 0.9905 | 0.9873 ± 0.0055 |
| value F1 | 0.9810 | 0.9905 | 0.9905 | 0.9873 ± 0.0055 |
| value token F1 | 0.9937 | 0.9958 | 0.9979 | 0.9958 ± 0.0021 |
| value exact | 0.9810 | 0.9905 | 0.9905 | 0.9873 ± 0.0055 |
| value present/null accuracy | 1.0000 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| full joint (type ∧ target ∧ value exact) | 0.8476 | 0.8190 | 0.8667 | 0.8444 ± 0.0240 |
| value exact, human-only (n = 7) | 0.8571 | 1.0000 | 0.8571 | 0.9048 ± 0.0825 |

### 6.3 test-targeted (n = 38; value n = 34, human-only value n = 5)

| metric | seed 1 | seed 2 | seed 3 | mean ± std |
| --- | --- | --- | --- | --- |
| type accuracy | 0.9474 | 0.9474 | 0.9474 | 0.9474 ± 0.0000 |
| type macro-F1 | 0.9667 | 0.9667 | 0.9667 | 0.9667 ± 0.0000 |
| target F1 | 0.9714 | 0.9143 | 0.8649 | 0.9169 ± 0.0533 |
| target exact | 0.9737 | 0.9474 | 0.9211 | 0.9474 ± 0.0263 |
| type+target joint | 0.9474 | 0.9211 | 0.8947 | 0.9211 ± 0.0263 |
| value precision | 0.9706 | 1.0000 | 1.0000 | 0.9902 ± 0.0170 |
| value recall | 0.9706 | 1.0000 | 1.0000 | 0.9902 ± 0.0170 |
| value F1 | 0.9706 | 1.0000 | 1.0000 | 0.9902 ± 0.0170 |
| value token F1 | 0.9834 | 1.0000 | 1.0000 | 0.9945 ± 0.0096 |
| value exact | 0.9706 | 1.0000 | 1.0000 | 0.9902 ± 0.0170 |
| value present/null accuracy | 1.0000 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| full joint (type ∧ target ∧ value exact) | 0.9118 | 0.9118 | 0.8824 | 0.9020 ± 0.0170 |
| value exact, human-only (n = 5) | 1.0000 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |

### 6.4 probe (n = 81; value n = 81, human-only value n = 5)

| metric | seed 1 | seed 2 | seed 3 | mean ± std |
| --- | --- | --- | --- | --- |
| type accuracy | 0.8272 | 0.8272 | 0.8642 | 0.8395 ± 0.0214 |
| type macro-F1 | 0.7941 | 0.7838 | 0.8259 | 0.8013 ± 0.0219 |
| target F1 | 0.8257 | 0.8991 | 0.8440 | 0.8563 ± 0.0382 |
| target exact | 0.8642 | 0.9136 | 0.8642 | 0.8807 ± 0.0285 |
| type+target joint | 0.7037 | 0.7531 | 0.7531 | 0.7366 ± 0.0285 |
| value precision | 1.0000 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| value recall | 1.0000 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| value F1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| value token F1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| value exact | 1.0000 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| value present/null accuracy | 1.0000 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| full joint (type ∧ target ∧ value exact) | 0.7037 | 0.7531 | 0.7531 | 0.7366 ± 0.0285 |
| value exact, human-only (n = 5) | 1.0000 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |

### 6.5 validation (monitoring only; n = 149, value n = 146)

Frozen v1 validation (110) is inside train.jsonl, so these numbers are in-sample for 110 of 149 rows; only the 39 targeted-value-01 validation rows are held out. Logged per epoch, never used for selection or stopping.

| metric | seed 1 | seed 2 | seed 3 | mean ± std |
| --- | --- | --- | --- | --- |
| type accuracy | 0.9799 | 0.9933 | 0.9799 | 0.9843 ± 0.0077 |
| type macro-F1 | 0.9767 | 0.9896 | 0.9720 | 0.9794 ± 0.0091 |
| target F1 | 0.9870 | 0.9870 | 0.9806 | 0.9849 ± 0.0037 |
| target exact | 0.9933 | 0.9933 | 0.9866 | 0.9911 ± 0.0039 |
| type+target joint | 0.9732 | 0.9933 | 0.9732 | 0.9799 ± 0.0116 |
| value F1 | 0.9965 | 1.0000 | 0.9965 | 0.9977 ± 0.0020 |
| value token F1 | 0.9985 | 1.0000 | 0.9985 | 0.9990 ± 0.0008 |
| value exact | 0.9932 | 1.0000 | 0.9932 | 0.9954 ± 0.0040 |
| value present/null accuracy | 0.9932 | 1.0000 | 0.9932 | 0.9954 ± 0.0040 |
| full joint | 0.9658 | 0.9932 | 0.9658 | 0.9749 ± 0.0158 |

## 7. Value slices (value exact)

Slice definitions: `explicit_unit` = propose_value category; `bare_number` = propose_value category; `multi_number` = propose_value category; `slang` = propose_value category (xi, chai, lit, cu, ty); `unaccented` = record accented == false; `unseen_span` = gold value text (exact string) not among the complete train values; `long_multi_token` = gold value covers > 2 model tokens or contains a space; `no_amount` = complete record with value null. Scope: value_status == complete; *_human_only: value_provenance == human.

### test

| slice | n | mean ± std | seed 1 | seeds 1/2/3 | human-only n | human-only mean | human-only seed 1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| explicit_unit | 113 | 0.9882 ± 0.0051 | 0.9823 | 0.982/0.991/0.991 | 7 | 0.9048 | 0.8571 |
| bare_number | 6 | 0.9444 ± 0.0962 | 0.8333 | 0.833/1.000/1.000 | 1 | 1.0000 | 1.0000 |
| multi_number | 22 | 0.9545 ± 0.0455 | 0.9091 | 0.909/1.000/0.955 | 3 | 0.7778 | 0.6667 |
| slang | 10 | 1.0000 ± 0.0000 | 1.0000 | 1.000/1.000/1.000 | 2 | 1.0000 | 1.0000 |
| unaccented | 50 | 0.9933 ± 0.0115 | 0.9800 | 0.980/1.000/1.000 | 5 | 1.0000 | 1.0000 |
| unseen_span | 37 | 0.9820 ± 0.0156 | 0.9730 | 0.973/0.973/1.000 | 3 | 1.0000 | 1.0000 |
| long_multi_token | 63 | 0.9788 ± 0.0092 | 0.9683 | 0.968/0.984/0.984 | 10 | 0.9333 | 0.9000 |
| no_amount | 0 | not measurable on test (n = 0) | – | – | 0 | – | – |

### probe

| slice | n | mean ± std | seed 1 | seeds 1/2/3 | human-only n | human-only mean | human-only seed 1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| explicit_unit | 78 | 1.0000 ± 0.0000 | 1.0000 | 1.000/1.000/1.000 | 4 | 1.0000 | 1.0000 |
| bare_number | 0 | n = 0 | – | – | 0 | – | – |
| multi_number | 8 | 1.0000 ± 0.0000 | 1.0000 | 1.000/1.000/1.000 | 1 | 1.0000 | 1.0000 |
| slang | 3 | 1.0000 ± 0.0000 | 1.0000 | 1.000/1.000/1.000 | 1 | 1.0000 | 1.0000 |
| unaccented | 32 | 1.0000 ± 0.0000 | 1.0000 | 1.000/1.000/1.000 | 4 | 1.0000 | 1.0000 |
| unseen_span | 6 | 1.0000 ± 0.0000 | 1.0000 | 1.000/1.000/1.000 | 1 | 1.0000 | 1.0000 |
| long_multi_token | 26 | 1.0000 ± 0.0000 | 1.0000 | 1.000/1.000/1.000 | 4 | 1.0000 | 1.0000 |
| no_amount | 0 | n = 0 | – | – | 0 | – | – |

Slices are tiny on test (bare_number 6, slang 10, human-only 3-7 per slice); read them as directional. `no_amount` has n = 0 on test, so null-value accuracy is not measured there (§12).

## 8. Adoption gate: value quality (PASS)

| id | scope | metric | value | cmp | threshold | n | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| test_value_exact | mean | test value exact (complete labels) | 0.9880 | >= | 0.95 | 139 | pass |
| test_value_exact_human | mean | test value exact (human-provenance labels) | 0.9444 | >= | 0.9 | 12 | pass |
| test_value_present_accuracy | mean | test value present/null accuracy | 1.0000 | >= | 0.97 | 139 | pass |
| test_multi_number_exact | mean | multi_number test slice value exact | 0.9545 | >= | 0.9 | 22 | pass |
| test_slice_floor | mean | lowest value exact over test value slices with n >= 5 | 0.9444 | >= | 0.8 |  | pass |
| test_value_exact | seed1 | test value exact (complete labels) | 0.9784 | >= | 0.95 | 139 | pass |
| test_value_exact_human | seed1 | test value exact (human-provenance labels) | 0.9167 | >= | 0.9 | 12 | pass |
| test_value_present_accuracy | seed1 | test value present/null accuracy | 1.0000 | >= | 0.97 | 139 | pass |
| test_multi_number_exact | seed1 | multi_number test slice value exact | 0.9091 | >= | 0.9 | 22 | pass |
| test_slice_floor | seed1 | lowest value exact over test value slices with n >= 5 | 0.8333 | >= | 0.8 |  | pass |

value_quality pass = True; failed criteria = []. The tightest margin is on seed 1: `bare_number` 5/6 = 0.8333 vs the 0.80 floor, human-label exact 11/12 = 0.9167 vs 0.90, and `multi_number` 20/22 = 0.9091 vs 0.90; each is one note from failing.

## 9. Adoption gate: regression vs v1 (FAIL)

Baseline: frozen compression-v3 seeds 1-3 (seed 1 = `gidi-finance-v1`), scored with the same metric code on the same records. "value" is the drop in notes (positive = v2 worse; negative = v2 better).

| criterion | scope | metric | value | cmp | threshold | result |
| --- | --- | --- | --- | --- | --- | --- |
| test-v1_type_mean | mean | type accuracy drop vs v1 on test-v1 (notes of 105) | 2.0 | <= | 2 | pass |
| test-v1_type_seed1 | seed1 | type accuracy drop vs v1 on test-v1 (notes of 105) | 4.0 | <= | 3 | **FAIL** |
| test-v1_target_mean | mean | target exact drop vs v1 on test-v1 (notes of 105) | -3.0 | <= | 2 | pass |
| test-v1_target_seed1 | seed1 | target exact drop vs v1 on test-v1 (notes of 105) | -5.0 | <= | 3 | pass |
| test-v1_joint_mean | mean | type+target joint drop vs v1 on test-v1 (notes of 105) | -1.3333 | <= | 2 | pass |
| test-v1_joint_seed1 | seed1 | type+target joint drop vs v1 on test-v1 (notes of 105) | -2.0 | <= | 3 | pass |
| probe_joint_mean | mean | type+target joint drop vs v1 on probe (notes of 81) | -9.0 | <= | 2 | pass |
| probe_joint_seed1 | seed1 | type+target joint drop vs v1 on probe (notes of 81) | -8.0 | <= | 3 | pass |
| probe_lender_first_type_mean | mean | probe lender-first slice type accuracy loss vs v1 (notes) | -2.3333 | <= | 1 | pass |

material_regression = True; flagged = ['test-v1_type_seed1']; rate-shorthand-only = ['test-v1_type_mean'].

The gating failure:

- **`test-v1_type_seed1`**: seed 1 type accuracy on test-v1 is 100/105 for v1 and 96/105 for v2, a drop of 4 notes against a limit of 3. This alone makes the verdict B.
- `test-v1_type_mean`: the mean type drop is exactly 2.0 notes (v1 100/98/96 → v2 96/96/96 correct per seed, seeds 1/2/3). The wording "more than 2 notes" passes it (2.0 ≤ 2). The parenthetical rate shorthand (> 0.019) would flag it, since the exact drop rate is 0.01905; the verdict is B either way because seed 1 fails.

### 9.1 v1 vs v2 correct counts (notes)

| set | metric | v1 seeds 1/2/3 | v2 seeds 1/2/3 | v1 mean | v2 mean | Δ mean (v2 − v1) |
| --- | --- | --- | --- | --- | --- | --- |
| test-v1 | type | 100/98/96 | 96/96/96 | 98.00 | 96.00 | -2.00 |
| test-v1 | target exact | 93/94/95 | 98/95/98 | 94.00 | 97.00 | +3.00 |
| test-v1 | type+target joint | 89/88/89 | 91/87/92 | 88.67 | 90.00 | +1.33 |
| probe | type | 63/65/66 | 67/67/70 | 64.67 | 68.00 | +3.33 |
| probe | target exact | 66/66/64 | 70/74/70 | 65.33 | 71.33 | +6.00 |
| probe | type+target joint | 49/52/51 | 57/61/61 | 50.67 | 59.67 | +9.00 |

### 9.2 Metric rates, v1 vs v2 (mean over seeds)

| set | metric | v1 mean ± std | v2 mean ± std |
| --- | --- | --- | --- |
| test | type accuracy | 0.9301 ± 0.0140 | 0.9231 ± 0.0000 |
| test | type macro-F1 | 0.9397 ± 0.0132 | 0.9358 ± 0.0041 |
| test | target F1 | 0.8591 ± 0.0256 | 0.8901 ± 0.0292 |
| test | target exact | 0.9091 ± 0.0070 | 0.9301 ± 0.0140 |
| test | type+target joint | 0.8578 ± 0.0040 | 0.8741 ± 0.0185 |
| test-v1 | type accuracy | 0.9333 ± 0.0190 | 0.9143 ± 0.0000 |
| test-v1 | type macro-F1 | 0.9433 ± 0.0181 | 0.9266 ± 0.0067 |
| test-v1 | target F1 | 0.8407 ± 0.0343 | 0.8813 ± 0.0387 |
| test-v1 | target exact | 0.8952 ± 0.0095 | 0.9238 ± 0.0165 |
| test-v1 | type+target joint | 0.8444 ± 0.0055 | 0.8571 ± 0.0252 |
| test-targeted | type accuracy | 0.9211 ± 0.0000 | 0.9474 ± 0.0000 |
| test-targeted | type macro-F1 | 0.9349 ± 0.0000 | 0.9667 ± 0.0000 |
| test-targeted | target F1 | 0.9143 ± 0.0000 | 0.9169 ± 0.0533 |
| test-targeted | target exact | 0.9474 ± 0.0000 | 0.9474 ± 0.0263 |
| test-targeted | type+target joint | 0.8947 ± 0.0000 | 0.9211 ± 0.0263 |
| probe | type accuracy | 0.7984 ± 0.0189 | 0.8395 ± 0.0214 |
| probe | type macro-F1 | 0.7798 ± 0.0203 | 0.8013 ± 0.0219 |
| probe | target F1 | 0.7675 ± 0.0214 | 0.8563 ± 0.0382 |
| probe | target exact | 0.8066 ± 0.0143 | 0.8807 ± 0.0285 |
| probe | type+target joint | 0.6255 ± 0.0189 | 0.7366 ± 0.0285 |

### 9.3 Regression slices (correct notes, v1 → v2 per seed)

Slice definitions (`slice_definitions.regression`): 

- probe: pattern tag of datasets/probe-v1/notes.jsonl: lender_first = ['lender_first_cho_muon']; title_name = ['title_name_span']; insurance = ['insurance_payout', 'insurance_premium']; loan_installment = ['loan_installment']; unaccented = record accented == false
- test-v1: golden-suite rules (scripts/build_golden_suite.py): lender_first = borrow_with_cho_vay_phrase; title_name = title_excluded_from_target or title_included_in_target; insurance = insurance; loan_installment = type repayment_out and /khoản vay|khoan vay|(?<!\w)vay(?!\w)|trả góp|tra gop|(?<!\w)g[óo]p(?!\w)|(?<!\w)k[ỳy](?!\w)|home credit|fe credit/i; unaccented = record accented == false

| set | slice | n | type | target | joint |
| --- | --- | --- | --- | --- | --- |
| test-v1 | lender_first | 1 | 1/1/1 → 1/1/1 (Δ +0.00) | 0/1/1 → 0/0/1 (Δ -0.33) | 0/1/1 → 0/0/1 (Δ -0.33) |
| test-v1 | title_name | 12 | 12/11/11 → 11/12/12 (Δ +0.33) | 10/10/12 → 11/11/12 (Δ +0.67) | 10/9/11 → 10/11/12 (Δ +1.00) |
| test-v1 | insurance | 2 | 2/2/2 → 2/2/2 (Δ +0.00) | 2/2/2 → 2/2/2 (Δ +0.00) | 2/2/2 → 2/2/2 (Δ +0.00) |
| test-v1 | loan_installment | 4 | 4/4/3 → 4/4/4 (Δ +0.33) | 3/3/3 → 4/4/4 (Δ +1.00) | 3/3/3 → 4/4/4 (Δ +1.00) |
| test-v1 | unaccented | 42 | 40/38/37 → 38/38/38 (Δ -0.33) | 36/38/38 → 39/39/39 (Δ +1.67) | 35/35/35 → 37/36/37 (Δ +1.67) |
| probe | lender_first | 9 | 5/6/7 → 8/8/9 (Δ +2.33) | 9/8/8 → 9/9/9 (Δ +0.67) | 5/5/6 → 8/8/9 (Δ +3.00) |
| probe | title_name | 9 | 8/8/8 → 8/8/8 (Δ +0.00) | 7/7/6 → 7/8/7 (Δ +0.67) | 6/6/5 → 7/7/7 (Δ +1.33) |
| probe | insurance | 18 | 14/15/15 → 13/13/14 (Δ -1.33) | 11/13/13 → 15/16/14 (Δ +2.67) | 7/10/10 → 10/11/10 (Δ +1.33) |
| probe | loan_installment | 9 | 7/7/7 → 7/7/7 (Δ +0.00) | 7/8/8 → 8/8/7 (Δ +0.00) | 5/6/6 → 6/6/5 (Δ +0.00) |
| probe | unaccented | 32 | 24/25/25 → 25/25/27 (Δ +1.00) | 26/26/24 → 27/28/27 (Δ +2.00) | 19/20/18 → 20/22/23 (Δ +2.67) |

## 10. Seed 1 changes vs v1 seed 1 (the export seed)

### test-v1: type regressions 4, type improvements 0 (net -4)

Type regressions (v1 correct → v2 wrong):

| text | gold | v1 | v2 |
| --- | --- | --- | --- |
| `mua giay the thao 1tr25` | expense | expense | transfer |
| `anh Long mượn 3 triệu sửa xe` | lend | lend | borrow |
| `spotify refund 59k` | refund | refund | expense |
| `ck cho Vũ mượn 700k` | lend | lend | borrow |

Type improvements (v1 wrong → v2 correct):

none.

Target exact vs v1 seed 1: improvements 6, regressions 1 (regression: `mua hoa qua biếu bà 200k`, gold no target, v2 predicts `bà`).

### probe: type regressions 2, type improvements 6 (net +4)

Type regressions (v1 correct → v2 wrong):

| text | gold | v1 | v2 |
| --- | --- | --- | --- |
| `anh trai cho 1tr5` | income | income | lend |
| `prudential chi trả tiền mổ 12tr` | refund | refund | repayment_in |

Type improvements (v1 wrong → v2 correct):

| text | gold | v1 | v2 |
| --- | --- | --- | --- |
| `Tuấn cho vay 5 triệu, hẹn tháng sau trả` | borrow | lend | borrow |
| `thg Khoa cho muon tam 300k an trua` | borrow | lend | borrow |
| `được cô Hà cho vay 4 củ` | borrow | lend | borrow |
| `bo me cho 5tr dong hoc` | income | transfer | income |
| `được cô Út cho 500k` | income | expense | income |
| `quay số trúng 500k` | income | borrow | income |

Wrong in both with a different prediction: `mẹ gửi 500k tiêu vặt` (gold income: repayment_in → transfer).

Target exact vs v1 seed 1: improvements 6, regressions 2 (regressions: `anh Duc muon 2tr dong hoc phi` gold `Duc` → `anh`; `bác Thành hoàn lại 1tr đã mượn` gold `Thành` → `ác`).

Cross-seed type behaviour of the same notes (v1 correct → v2 wrong, from `changes_vs_v1`):

| set | text | gold | v2 wrong in |
| --- | --- | --- | --- |
| test-v1 | `mua giay the thao 1tr25` | expense | seed 1 → transfer, seed 2 → transfer |
| test-v1 | `anh Long mượn 3 triệu sửa xe` | lend | seed 1 → borrow |
| test-v1 | `spotify refund 59k` | refund | seed 1 → expense |
| test-v1 | `ck cho Vũ mượn 700k` | lend | seed 1 → borrow, seed 2 → borrow, seed 3 → borrow |
| probe | `anh trai cho 1tr5` | income | seed 1 → lend |
| probe | `prudential chi trả tiền mổ 12tr` | refund | seed 1 → repayment_in, seed 2 → repayment_in |
| test-v1 | `hạt cho chó 420.000đ` | expense | seed 2 → transfer |
| test-v1 | `de rieng tien sua xe 800k` | transfer | seed 2 → expense |
| probe | `bhxh tu dong thang nay 1tr5` | expense | seed 2 → refund, seed 3 → repayment_out |
| test-v1 | `bách hoá xanh rau thịt 187k` | expense | seed 3 → income |
| probe | `ông nội cho 1 triệu mua sách` | income | seed 3 → borrow |

### 10.1 Observed pattern [INFERENCE from the lists above, not tested]

- The type regressions on test-v1 include direction errors on user-as-lender `cho X mượn` / `X mượn` notes: `ck cho Vũ mượn 700k` (gold lend → borrow) regresses in all three seeds, and `anh Long mượn 3 triệu sửa xe` (lend → borrow) regresses in seed 1. The other seed 1 test-v1 type regressions (`spotify refund 59k`, `mua giay the thao 1tr25`) are not direction errors.
- In the same family the lender-first probe slice improves: type correct 5/6/7 → 8/8/9 (mean +2.33 notes of 9) and probe joint improves (mean +9.0 notes of 81). Target exact improves on test-v1 (mean +3.0 notes of 105).
- The golden-suite lender-first slice on test-v1 has n = 1, and the probe lender-first slice n = 9, so both the gain and the loss rest on a handful of notes; with these sample sizes the pattern is a hypothesis, not a finding.

## 11. Multi-number value errors (inspected by hand)

Protocol requires every multi-number value error on test and probe to be listed. `multi-number-errors.jsonl`: 3 rows, over 22 multi-number test notes and 8 probe notes (test errors per seed 1/2/3: 2/0/1; probe errors: 0/0/0).

| seed | text | gold value | predicted value | confidence | provenance | categories |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `cho a Nam vay 1 triệu 20/10` | `1 triệu` 14-21 | `1 triệu 20/10` 14-27 | 0.9997 | human | explicit_unit, multi_number, context_excluded |
| 1 | `tra no chi Mai 300 hom 5/10` | `300` 15-18 | `300 hom 5/10` 15-27 | 0.7617 | rule | bare_number, multi_number, context_excluded, unaccented |
| 3 | `cho a Nam vay 1 triệu 20/10` | `1 triệu` 14-21 | `1 triệu 20/10` 14-27 | 0.9989 | human | explicit_unit, multi_number, context_excluded |

Manual analysis (2 distinct notes):

- `cho a Nam vay 1 triệu 20/10` (seeds 1 and 3; seed 2 correct): gold `1 triệu`, predicted `1 triệu 20/10`. The start is right and the head picks the correct amount over the date; the span swallows the trailing date, an end-boundary overrun. The type was correct (`lend`) in both seeds. Confidence is 0.9997 and 0.9989, so the error is confident, not hedged. The label is human-provenance (this note is one of the 12 human value labels on test).
- `tra no chi Mai 300 hom 5/10` (seed 1 only): gold `300`, predicted `300 hom 5/10`: the same overrun, this time through the word `hom` into the date. Type correct (`repayment_out`). Confidence 0.7617 is much lower than in the first case. Rule-provenance label.
- Both are the same failure: the span continues from the amount into a following date. No multi-number error picked the wrong number, a quantity, or a month. Neither error touches the amount start. [INFERENCE] The training data has few date-trailing amounts: the existing-data `multi_number_date_or_time` review stratum has a population of only 4 notes (2 sampled); the date content of the targeted-value-01 rows was not checked. The head may therefore have seen few date-after-amount examples; this was not tested.
- Outside the protocol list: one non-multi-number test error also exists, `cashback thẻ vib 230k` (gold `230k`, predicted `2`, rule label) in seeds 1 and 2 — a truncated span, not a multi-number issue. Test value errors per seed: seed 1 3, seed 2 1, seed 3 1.

## 12. Known limitations

- Most value labels are rule-provenance, accepted by stratum audit (0 errors in sampled strata, one must-review error on existing data). Value metrics on them measure agreement with audited rule labels. Human-only value labels on test: n = 12 of 139 (1 error for seed 1).
- Slices are tiny: test bare_number 6, slang 10, human-only slices 0-7; probe is mostly perfect (value exact 1.0000 in all seeds), so it has no discriminating power for value. Slice results are directional.
- `no_amount` has n = 0 on test; null-value accuracy is only measured on validation (2 complete no_amount) and is not an adoption gate.
- `2 triệu rưỡi` is absent from every split; the model has no evidence for it.
- The frozen v1 validation (110 rows) is inside train, as in v1; validation numbers are monitoring only, and only 39 targeted-value-01 validation rows are held out.
- The regression gate is a note-count test on 105 and 81 notes; one note is ≈ 0.01 on test-v1 and the seed-1 and mean thresholds differ by one note. Seed variance is of the same order as the thresholds (per-seed v2 type correct on test-v1: 96/96/96; v1: 100/98/96).
- MPS training is not bit-reproducible; a rerun can move individual notes.
- Value labels are spans of the original string; no numeric normalization is evaluated.

## 13. Outcome and what was not done

**B. value-span works but type/target regressions need fixing**

Per the frozen protocol:

- no ONNX/INT8 bundle was exported; `models/gidi-finance-v1/` is untouched and remains the deployed model;
- the playground is unchanged: its default bundle remains v1; start command `uv run python scripts/demo_ui.py`;
- `models/value-span-v1/seed{1,2,3}` are PyTorch checkpoints only, kept as evidence.

Fixing the type regression is a **new experiment** and needs a user decision before it starts (test and probe stay evaluation-only; the frozen protocol is not edited). Candidate directions, none started and none ranked:

- inspect and, if warranted, rebalance the direction (lend vs borrow) notes in the targeted-value-01 training rows, in particular the `cho X mượn` / `X mượn` family;
- review the multi-seed export rule: the seed 1 gate fails by one note while the mean passes at the limit, and v1 seeds themselves span 96-100 on test-v1;
- look at the trailing-date boundary overrun (§11) with more date-after-amount training examples;
- keep v2 as a research artifact and leave v1 deployed without further work.

Implementation fixes during this run (no effect on data, labels, training or metrics):

- `scripts/build_targeted_value_queue.py --check` failed after the training set was built, because its duplicate scan counted the files of the downstream `datasets/annotation-v2/training-*` output (54 → 58 files; same 1,382 texts, same queue and groups). The scan now excludes those derived training directories, and `--check` reproduces again.

## 14. Artifacts

| path | content |
|---|---|
| `experiments/value-span-v1/protocol.json` | frozen protocol (read-only) |
| `experiments/value-span-v1/comparison.json` | verdict, per-seed and mean metrics, slices, regression table, changes vs v1 |
| `experiments/value-span-v1/multi-number-errors.jsonl` | multi-number value errors (3 rows) |
| `experiments/value-span-v1/eval/seed{1,2,3}-{test,test-v1,test-targeted,probe,validation}.json` | v2 metrics per seed and set |
| `experiments/value-span-v1/eval/seed{1,2,3}-*-predictions.jsonl` | v2 per-record predictions |
| `experiments/value-span-v1/eval/seed{1,2,3}-v1-*.json(l)` | v1 baseline re-scored on the same records |
| `experiments/value-span-v1/train.log` | training log and per-seed summary |
| `experiments/value-span-v1/data-audit-training.{json,md}` | training-data audit |
| `experiments/value-span-v1/data-audit.{json,md}` | pre-review data audit |
| `experiments/value-span-v1/review-score.json` | tiered review score and escalation decisions |
| `experiments/value-span-v1/existing-value-review-plan.{json,md}` | existing-data review plan |
| `experiments/value-span-v1/targeted-value-01-review-plan.{json,md}` | targeted-value-01 review plan |
| `datasets/annotation-v2/training-v1/` | train/validation/test/probe files and manifest.json |
| `docs/annotation-v2.md` | annotation-v2 contract and pipeline |
| `models/value-span-v1/seed{1,2,3}/` | v2 PyTorch checkpoints (no ONNX export) |
| `scripts/train_value_student.py, scripts/evaluate_value.py, scripts/compare_value_v1.py` | training, evaluation and comparison code |

B. value-span works but type/target regressions need fixing
