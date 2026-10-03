# distillation-v1: 4×256 student from the targeted-v2 BamiBERT teacher

This is the first distillation baseline. One final targeted-v2 teacher produces soft labels. The
same 4×256 student is then trained in two arms, three seeds each, with identical data, init,
batches and budget:

- **supervised-only:** human labels only;
- **distilled:** human labels plus the teacher's soft distributions.

Out of scope this round: architecture changes, temperature or weight sweeps, MiniLM and new data.
All numbers are in `comparison.json`.

## 1. Frozen teacher artifact

The provenance is recorded in `experiments/distillation-v1/manifest.json`:

- the targeted-v2 recipe, data, manifest hashes and per-seed checkpoint sha256;
- the tokenizer;
- the architecture;
- the final teacher;
- the frozen evaluation hashes;
- the exclusions.

| | value |
|---|---|
| teacher recipe | targeted-v2: `datasets/annotation-v1/training-v2/` (frozen train 501 + targeted-02 112), runs `experiments/targeted-v2/runs/bamibert/lr5e-05-seed{1,2,3}` (unchanged) |
| tokenizer | `Qualcomm-AI-Research/BamiBERT` fast tokenizer, snapshot `57bc1340debbe4e348ec549047a763caebe4a977`, vocab 20,481, max_length 32 (input ids verified identical to the hub snapshot on all 938 train/validation/test texts) |
| architecture | RoBERTa 12 layers × 768, 12 heads, FFN 3072; masked-mean type head (8) + token BIO head (3); 102.4 M parameters |
| guard | `gidi.distillation.guard` refuses frozen-test ids, `probe-*` ids and targeted-03 (by id prefix and `source_batch`). It runs in the data builder, the target builder and student training. training-v3 is never read. |

## 2. Final teacher (soft-label source)

- **Data:** `datasets/annotation-v1/distillation-v1/train.jsonl` is the training-v2 train bytes
  followed by the frozen validation bytes: 723 records (501 + 112 + 110).
  - `scripts/build_distillation_data.py --check` verifies training-v2, the frozen split hashes,
    the guard and id disjointness.
  - Frozen test and probe-v1 are excluded.
- **Recipe:** exactly targeted-v2: lr 5e-5 / head 1e-3, batch 8, AdamW wd 0.01, linear schedule,
  10% warmup, dropout 0.1, clip 1.0, max_length 32, MPS.
- **Duration rule, fixed in advance:**
  - stop epoch = median of targeted-v2's best epochs (9, 14, 13 → **13**);
  - LR schedule horizon kept at 20 epochs;
  - seed 1;
  - **last-epoch weights**;
  - no selection on validation, test or probe (`TrainConfig.stop_epoch`).
- **Outputs:**
  - checkpoint `models/distillation-v1/teacher/bamibert/lr5e-05-seed1/` (sha256 `3838de12…`);
  - run `experiments/distillation-v1/teacher-run/…`;
  - final train loss 0.0065.

Validation is in-sample for this teacher (and for the students). Its validation scores (0.991 /
1.000) are **not** held-out results.

## 3. Student

| | |
|---|---|
| encoder | RoBERTa, 4 layers, hidden 256, 4 heads, FFN 1024, GELU, dropout 0.1, 34 positions (max_length 32 + RoBERTa offset) |
| vocabulary | BamiBERT tokenizer unchanged (20,481) |
| heads | the same `GidiMultiTaskModel`: masked-mean type head (8 types) + token BIO head (O/B/I), shared encoder |
| size | 8,414,475 parameters (5.2 M of them the embedding matrix); 33.7 MB fp32 safetensors |
| semantics | note → type distribution + per-token BIO distribution. Spans are decoded exactly as for the teacher, and amounts stay outside the model. |

### Initialization: option C, independent random init

- **A is not available:** no pretrained 4×256 encoder exists for BamiBERT's 20,481-token
  vocabulary.
- **B is not valid as a copy:** teacher layers are 768-wide (12×64 heads). Moving them to 256
  would need a projection or truncation, which approximates the weights. That would be a
  separate design decision, not a copy.
- **What C means here:** standard HF init (std 0.02) under `set_seed(seed)`. For a given seed,
  both arms start from bit-identical weights and see the same batch order.
- **Consequence:** the student has no pretraining. Everything it knows comes from 723 labelled
  notes and the teacher's outputs on them (see §9).

## 4. Distillation targets

- **Cache:** `experiments/distillation-v1/teacher-targets/`, built by
  `scripts/build_teacher_targets.py`. `--check` regenerates the cache and matches it.
- **Contents of `targets.safetensors`:**
  - `input_ids`, `attention_mask`;
  - gold `tag_labels` (−100 off real tokens) and gold `type_ids`;
  - raw teacher `teacher_type_logits [723,8]` and `teacher_tag_logits [723,32,3]`: fp32, not
    argmax, not temperature-scaled.
- **Contents of `index.jsonl`:** id, text, gold labels and per-position offsets.
- **Manifest:** teacher and tokenizer sha256, data sha256, shapes, and file hashes. Student
  training fails if a hash differs or if re-encoding does not reproduce the cached tensors.
- **Teacher vs gold on the transfer set:** type 0.9986, tag tokens 0.9997.

The teacher's distributions on its own training notes are nearly one-hot:

| | T = 1 | T = 2 |
|---|---|---|
| type: mean max probability | 0.999 | 0.973 |
| type: share of notes with max prob < 0.9 | 0.3% | 0.4% |
| type: mean entropy (nats) | 0.003 | 0.169 |
| BIO: mean max probability | 1.000 | 0.988 |

## 5. Loss (`gidi.distillation.loss`, configurable)

$$
L = w_{type}\,[\alpha\,\mathrm{CE}(s, y) + (1-\alpha)\,T^2\,\mathrm{KL}(p_t^{T}\,\|\,p_s^{T})]
  + w_{span}\,[\alpha\,\mathrm{CE}_{tok} + (1-\alpha)\,T^2\,\mathrm{KL}_{tok}]
$$

- **Token terms:** averaged over exactly the real-token positions (gold tag ≠ −100), so padding
  and special tokens are excluded from both the CE and the KL. Type KL is batch-mean.
- **Distilled arm:** T = 2, α (hard weight) = 0.5, w_type = w_span = 1. This is the conventional
  Hinton setting: equal hard/soft weight, mild temperature, and task weights equal to the
  teacher's own equal loss weights.
- **Supervised arm:** α = 1, so no teacher terms. It is the same code path.

**Student training, identical in both arms:**
- AdamW, encoder lr 5e-4, head lr 1e-3, wd 0.01, linear schedule with 10% warmup, clip 1.0,
  batch 8;
- **fixed 40 epochs (3,640 steps), last-epoch weights, no selection**;
- MPS, seeds 1/2/3.

These are standard from-scratch small-transformer settings, fixed before any evaluation. Both
arms fit the training data completely: final train loss 0.0001–0.0005 (supervised) and
0.005–0.007 (distilled), and in-sample validation F1 ≈ 1.0.

## 6. Held-out results: frozen test (105) and probe-v1 (81)

- **Reference columns:** targeted-v2 is the 3-seed mean of the recipe. The final teacher is the
  single model the students learned from.
- **Error counts:** absolute counts per run.
- **What "joint" means:** type correct and span exactly correct (null = null).

**Frozen test**

| | targeted-v2 | final teacher | supervised 4×256 | distilled 4×256 |
|---|---|---|---|---|
| type accuracy | 0.930±0.005 | 0.895 | 0.889±0.038 | 0.886±0.010 |
| type macro-F1 | 0.930±0.003 | 0.886 | 0.887±0.037 | 0.871±0.021 |
| span F1 | 0.877±0.029 | 0.940 | 0.648±0.027 | 0.662±0.020 |
| span exact | 0.911±0.024 | 0.952 | 0.768±0.005 | 0.775±0.011 |
| type + span joint | 0.857±0.000 | 0.857 | 0.702±0.024 | 0.705±0.016 |
| false span (gold null, 54) | 2.7 | 1 | 5.0 | 4.7 |
| missed span (gold target, 51) | 3.7 | 3 | 9.3 | 10.0 |
| wrong span | 3.0 | 1 | 10.0 | 9.0 |

**Probe-v1**

| | targeted-v2 | final teacher | supervised 4×256 | distilled 4×256 |
|---|---|---|---|---|
| type accuracy | 0.860±0.007 | 0.901 | 0.733±0.050 | 0.687±0.026 |
| type macro-F1 | 0.876±0.012 | 0.928 | 0.694±0.051 | 0.649±0.023 |
| span F1 | 0.878±0.025 | 0.920 | 0.565±0.054 | 0.624±0.022 |
| span exact | 0.889±0.025 | 0.926 | 0.630±0.045 | 0.691±0.012 |
| **type + span joint** | 0.786±0.040 | **0.840** | 0.477±0.038 | 0.494±0.033 |
| false span (gold null) | 2.0 | 3 | 7.3 | 4.3 |
| missed span | 2.7 | 0 | 3.0 | 4.3 |
| wrong span | 4.3 | 3 | 19.7 | 16.3 |

**Frozen validation (historical only)**

Both the teacher and the students trained on it.

| | targeted-v2 | final teacher | supervised | distilled |
|---|---|---|---|---|
| type macro-F1 | 0.898 | 0.991 | 1.000 | 0.994 |
| span F1 | 0.861 | 1.000 | 1.000 | 1.000 |

**Test per-class F1**

| class | targeted-v2 | final teacher | supervised | distilled |
|---|---|---|---|---|
| expense | 0.932 | 0.900 | 0.883±0.038 | 0.905±0.030 |
| income | 0.857 | 0.818 | 0.781±0.053 | 0.800±0.029 |
| borrow | 0.926 | 0.941 | 0.845±0.047 | 0.821±0.037 |
| lend | 0.980 | 1.000 | **0.743±0.049** | **0.746±0.114** |
| repayment_in | 0.888 | 0.769 | 0.978±0.038 | 0.860±0.063 |
| repayment_out | 0.938 | 0.870 | 0.947±0.000 | 0.893±0.053 |
| transfer | 0.937 | 0.947 | 0.942±0.061 | 0.964±0.015 |
| refund | 0.980 | 0.842 | 0.978±0.038 | 0.980±0.034 |

**Test confusion, students** (rows = gold, columns = predicted, summed over 3 seeds; order
expense, income, borrow, lend, repayment_in, repayment_out, transfer, refund):

| gold | supervised | distilled |
|---|---|---|
| expense | 83 9 0 0 0 0 4 0 | 85 9 0 0 0 0 1 1 |
| income | 2 27 2 0 0 0 2 0 | 1 28 1 0 2 0 1 0 |
| borrow | 0 0 27 0 0 0 0 0 | 0 0 25 1 0 1 0 0 |
| lend | 3 0 **8** 16 0 0 0 0 | 1 0 **7** 17 2 0 0 0 |
| repayment_in | 0 0 0 0 23 0 1 0 | 2 0 1 0 21 0 0 0 |
| repayment_out | 3 0 0 0 0 27 0 0 | 3 0 0 0 0 25 2 0 |
| transfer | 0 0 0 0 0 0 54 0 | 0 0 0 0 0 0 54 0 |
| refund | 1 0 0 0 0 0 0 23 | 0 0 0 0 0 0 0 24 |

The teacher makes 0 lend → borrow errors on test. Both students bring back the old direction
shortcut: 8 and 7 of 27 lend predictions become borrow.

**Accent slices**

| slice | n | final teacher | supervised | distilled |
|---|---|---|---|---|
| test accented: type acc / span F1 | 63 | 0.937 / 0.937 | 0.884 / 0.620 | 0.868 / 0.659 |
| test unaccented: type acc / span F1 | 42 | 0.833 / 0.944 | 0.897 / 0.700 | 0.913 / 0.666 |
| probe accented: type acc / exact / joint | 49 | 0.898 / 0.980 / 0.878 | 0.769 / 0.653 / 0.503 | 0.701 / 0.701 / 0.497 |
| probe unaccented: type acc / exact / joint | 32 | 0.906 / 0.844 / 0.781 | 0.677 / 0.594 / 0.438 | 0.667 / 0.677 / 0.490 |

The student losses are not specific to accent form. Spans drop on both slices, and probe type
drops on both.

## 7. Probe-v1, all 9 patterns (type + span joint success; type accuracy in brackets)

"All-fail" counts the notes (of 9) that fail on every student seed.

| # | pattern | targeted-v2 | final teacher | supervised | distilled | all-fail sup / KD |
|---|---|---|---|---|---|---|
| 1 | lender-first borrow | 1.00 (1.00) | 1.00 (1.00) | 0.48±0.06 (0.67) | 0.41±0.23 (0.63) | 2 / 3 |
| 2 | subject-first lend | 0.93 (0.93) | 0.89 (1.00) | 0.52±0.06 (0.52) | 0.52±0.13 (0.52) | 3 / 3 |
| 3 | family gift in | 0.74 (0.78) | 0.89 (0.89) | 0.44±0.11 (0.56) | 0.48±0.06 (0.59) | 4 / 4 |
| 4 | outgoing gift | 0.78 (0.85) | 0.89 (0.89) | 0.56±0.11 (0.85) | 0.56±0.11 (0.81) | 2 / 3 |
| 5 | insurance premium | 0.78 (0.85) | 0.67 (0.67) | 0.37±0.13 (0.78) | 0.44±0.11 (0.67) | 4 / 4 |
| 6 | insurance payout | 0.63 (0.74) | 0.89 (1.00) | 0.48±0.06 (0.74) | 0.52±0.13 (0.70) | 3 / 3 |
| 7 | windfall | 0.74 (0.85) | 0.78 (0.89) | 0.67±0.11 (0.89) | 0.67±0.11 (0.78) | 1 / 2 |
| 8 | loan installment | 0.67 (0.74) | 0.56 (0.78) | 0.37±0.06 (0.74) | 0.37±0.06 (0.67) | 4 / 4 |
| 9 | title/name span | 0.81 (1.00) | 1.00 (1.00) | 0.41±0.06 (0.85) | 0.48±0.06 (0.81) | 5 / 4 |

## 8. Failure cases

- **Teacher-correct test notes that every student seed gets wrong:** 14 (supervised) / 15
  (distilled), of which 8 / 9 are span-only. On probe-v1 the counts are 22 / 19. Examples from
  the distilled arm:

| kind | note | gold | distilled seeds |
|---|---|---|---|
| direction | `Phat muon tam 1tr` | lend / `Phat` | borrow ×3 |
| direction | `Thắng vay 5 củ, hẹn t10 trả` | lend / `Thắng` | borrow, borrow, lend (span `Th`) |
| type | `tra lai chi Mai 2tr` | repayment_out / `Mai` | expense ×3 |
| type | `coopmart đồ dùng tuần 412k` | expense / `coopmart` | income ×3 |
| title boundary | `vay chú Hải 20tr mua xe` | `Hải` | `chú` ×3 |
| title boundary | `cho bạn Kiên mượn 1tr2 ck vcb` | `Kiên` | `bạn Kiên`, null, `bạn Kiên` |
| subword fragment | `co Thuy cho muon 1tr` | `Thuy` | null, `co Th`, `Th` |
| subword fragment | `điện máy xanh hoàn tiền nồi cơm lỗi 890k` | `điện máy xanh` | null, `iện`, null |
| unseen merchant | `starbucks 1 ly 89,000` | `starbucks` | null ×3 |
| false span | `mừng thọ bà nội 1 triệu` | `bà nội` | `thọ` ×3 |

## 9. Distillation analysis

1. **Test type macro-F1 retained:** supervised 0.887 and distilled 0.871. Against the final
   teacher (0.886) that is **100% / 98%**. Against the targeted-v2 mean (0.930) it is
   95% / 94%. Test type quality survives. On test, lend is the only class that collapses
   (lend → borrow).
2. **Test span quality retained:** span F1 0.648 / 0.662 against 0.940, i.e. **69% / 70%**. Span
   exact 0.768 / 0.775 against 0.952 (81%). Joint 0.702 / 0.705 against 0.857 (82%). Every span
   error type rises 2–9×.
3. **Probe success retained:** 0.477 / 0.494 against 0.840, i.e. **57% / 59%** (61% / 63% of the
   targeted-v2 mean). Probe type accuracy is 0.733 / 0.687 against 0.901.
4. **Does KD beat supervised-only?** No.
   - Paired by seed (identical init and batches), KD minus supervised is:

     | metric | seed 1 | seed 2 | seed 3 |
     |---|---|---|---|
     | test span F1 | +0.014 | +0.007 | +0.022 |
     | test type macro-F1 | −0.003 | −0.076 | +0.032 |
     | probe type accuracy | 0.000 | −0.062 | −0.074 |
     | probe joint | +0.037 | +0.012 | 0.000 |

   - A small, consistent span gain is offset by a type loss on the probe. Every difference is
     inside the student seed spread.
   - Mechanism: the transfer set is the teacher's own training set, so its soft targets are
     essentially one-hot (§4: max probability 0.973 even at T = 2). At α = 0.5 the KD term adds
     little beyond mild label smoothing. There is almost no dark knowledge to transfer on these
     723 notes.
5. **Teacher behaviours that survive:**
   - type decisions for well-populated, lexically marked classes (transfer, refund,
     repayment_in/out, expense);
   - windfall (probe 0.67, type 0.89 / 0.78);
   - premium/payout and gift-out **type** decisions (0.67–0.85);
6. **Behaviours that are lost:**
   - the **direction contrasts**: lender-first borrow 1.00 → 0.41–0.48; subject-first lend
     0.89–0.93 → 0.52; family gift in; test lend → borrow;
   - **span boundaries**: title/name 1.00 → 0.41–0.48, kinship/title prefixes kept, subword
     fragments (`Th`, `iện`, `co Th`), unseen merchant names not tagged, false spans on
     item/occasion words;
   - loan installment (0.37) and insurance span correctness (premium 0.37–0.44).
7. **Systematic or seed noise?** Systematic.
   - Student seed std (span F1 0.02–0.03, probe joint 0.03–0.04) is an order of magnitude
     smaller than the teacher-student gaps (0.28 span F1, 0.35 probe joint).
   - The same 14–22 teacher-correct notes fail on all three seeds of each arm.
   - The per-pattern all-fail counts are 2–5 of 9 on every targeted pattern except windfall.
8. **Is 4×256 large enough to continue optimizing?** Not as built.
   - Capacity to *fit* is not the limit: both arms reach ~0 training loss and in-sample
     validation 1.0.
   - What is missing is *generalization*: the direction and span-boundary knowledge the
     pretrained teacher brings, and a randomly initialized 8.4 M-parameter student cannot
     recover it from 723 notes with near-one-hot targets.
   - This experiment cannot separate width/depth from "no pretrained student encoder on this
     vocabulary". Both are properties of the student design, not of the KD weighting.

## 10. Deployment measurements (distilled seed 1)

Seed 1 was chosen as the distilled seed with the highest test joint score. That choice is for
measurement only; nothing was trained on it.

- **Report:** `experiments/distillation-v1/export-distilled-seed1.json`, from
  `scripts/export_onnx.py`.
- **Models:** `models/distillation-v1-onnx/student-4x256-distilled-seed1/`.

| | value |
|---|---|
| checkpoint (safetensors, fp32) | 33.67 MB |
| ONNX fp32 (dynamo, opset 17) | **33.70 MB**; parity vs PyTorch on 50 test notes: max abs logit diff 6.7e-6 (pass, atol 1e-4) |
| ONNX INT8 (dynamic, QInt8 weights) | **8.51 MB** (ratio 0.25); agreement with fp32 on 50 notes: type 1.000, tags 1.000 (506 tokens) |
| post-quantization, full frozen test (fp32 ONNX → INT8) | type acc 0.895 → 0.886 (one flip), macro-F1 0.886 → 0.872, span F1 0.674 → 0.674, exact 0.781 → 0.781, joint 0.724 → 0.724 |
| post-quantization, probe-v1 | identical to fp32 ONNX: type acc 0.679, span exact 0.704, joint 0.481 |
| CPU batch-1 latency, 1 thread (Apple Silicon arm64, onnxruntime CPU EP, 200 runs × 50 notes, incl. tokenization) | INT8 p50 **0.40 ms** / p95 0.50 ms; fp32 ONNX p50 0.43 ms; PyTorch fp32 p50 0.73 ms |
| teacher for comparison (baseline-v1 BamiBERT export, same harness) | INT8 102.9 MB, p50 4.5 ms (1 thread) |

INT8 lands inside the 10–30 MB target band from below: 8.5 MB, about 12× smaller and about 11×
faster than the teacher. Size is not the problem; quality is.

## Reproduce

```sh
uv run python scripts/build_distillation_data.py --check
uv run python scripts/train_baseline.py --model Qualcomm-AI-Research/BamiBERT --lr 5e-5 --seed 1 \
  --head-lr 1e-3 --batch-size 8 --epochs 20 --stop-epoch 13 --patience 3 --max-length 32 \
  --splits-dir datasets/annotation-v1/distillation-v1 --out-dir experiments/distillation-v1/teacher-run \
  --weights-dir models/distillation-v1/teacher --device mps
uv run python scripts/build_teacher_targets.py --teacher models/distillation-v1/teacher/bamibert/lr5e-05-seed1 \
  --data datasets/annotation-v1/distillation-v1/train.jsonl --out experiments/distillation-v1/teacher-targets
for arm in supervised distilled; do
  uv run python scripts/train_student.py --arm $arm --seed 1 --seed 2 --seed 3 \
    --targets experiments/distillation-v1/teacher-targets --splits-dir datasets/annotation-v1/distillation-v1 \
    --out-dir experiments/distillation-v1/runs --weights-dir models/distillation-v1 --device mps
done
for w in teacher supervised distilled; do
  uv run python scripts/evaluate_probe.py --weights-dir models/distillation-v1/$w \
    --out experiments/distillation-v1/probe-eval-$w.json
done
uv run python scripts/export_onnx.py --checkpoint models/distillation-v1/distilled/student-4x256/seed1 \
  --out models/distillation-v1-onnx/student-4x256-distilled-seed1 \
  --report experiments/distillation-v1/export-distilled-seed1.json --notes datasets/annotation-v1/splits/test.jsonl
```

## Decision

- **A (KD works):** not met. KD does not clearly beat supervised-only; every difference is within
  seed spread.
- **C (objective is the problem):** not met as defined. Supervised 4×256 is not competitive
  either: span 69% of the teacher, probe 57%.
- **B:** both arms lose the same substantial teacher behaviours, systematically. These are the
  direction contrasts and span boundaries that targeted-v2 was built to fix.
  - The evidence points at the student's starting point rather than its ability to fit. It is a
    randomly initialized 4×256 encoder with no pretraining on the BamiBERT vocabulary, fitting
    723 notes perfectly and generalising poorly.
  - The near-one-hot teacher targets on the training set are a secondary factor. They explain why
    KD adds nothing, but they cannot explain why the supervised arm is equally weak.
  - The student design (capacity and initialization) has to be revisited before KD weighting is
    worth tuning.

**B. Revisit student capacity/architecture before further distillation.**
