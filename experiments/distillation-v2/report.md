# distillation-v2: student design diagnostic (4×768, pretrained vs random)

**Question.** Did student-v1 (4×256) fail mainly because it had no pretrained initialization, or
because a shallow student lacks capacity?

**Answer.** Pretraining. A random 4×768 student is no better than a random 4×256 one. The same
4×768 student initialized from four BamiBERT layers closes most of the gap: test joint 0.844 vs
the teacher's 0.857 (98.5%) and probe joint 0.667 vs 0.840 (79%). KD adds nothing measurable on
top. Conclusion **A**.

## 1. Setup

- **Teacher, data, targets:** unchanged from distillation-v1. The final BamiBERT teacher
  (`models/distillation-v1/teacher/bamibert/lr5e-05-seed1`), the 723-record distillation train
  set (sha256 `fc18d15e…`), and the cached teacher logits in
  `experiments/distillation-v1/teacher-targets` are reused as-is. The cache is
  student-independent (same tokenizer, same `input_ids`). Frozen test (105) and probe-v1 (81) are
  evaluation-only. targeted-03 / training-v3 are excluded (the distillation guard still runs on
  every training record).
- **Student-v2:** `student-4x768`: RoBERTa, 4 layers, hidden 768, 12 heads, FFN 3072, BamiBERT
  tokenizer and vocabulary (20,481), max_length 32, the same mean-pooled type head and BIO tag
  head as the teacher. 45,666,059 parameters. The position table keeps BamiBERT's 2,050 rows so
  it can be copied whole instead of truncated.
- **Pretrained init:** source is the pretrained `Qualcomm-AI-Research/BamiBERT` encoder (commit
  `57bc1340`, weights sha256 `c8c8abe5…`), i.e. the checkpoint the teacher was fine-tuned from,
  not the fine-tuned teacher. That keeps "pretrained linguistic knowledge" separate from "task
  fine-tuning". Copied exactly, all shape-identical, no projection/truncation/averaging:
  - word, position and token-type embeddings, embedding LayerNorm;
  - whole transformer blocks (attention, FFN, both LayerNorms) by the rule *student layer k ←
    teacher layer (k+1)·12/4 − 1*: **layers 2, 5, 8, 11** (0-based; 1-based 3, 6, 9, 12), the top
    layer of each 3-layer block, so the top pretrained layer is kept.
  - 69 tensors, 45,657,600 values: every student encoder tensor is covered (enforced in code).
- **Heads:** fresh seeded initialization in all arms, never the teacher's heads. The model is
  built exactly as the random arm first, then the encoder is overwritten, so for a given seed the
  heads are bit-identical across all three arms. The teacher's heads were not used: they read
  layer-12 features of the fine-tuned teacher, which the student does not reproduce.
- **Arms** (same architecture, 3 seeds each, identical batch order per seed):
  - **A** pretrained supervised (hard labels only);
  - **B** pretrained distilled (T 2, alpha_hard 0.5, type/span weights 1/1, the v1 KD config);
  - **C** random supervised (control).
- **Recipe, shared by all three arms, declared before training** (`protocol.json`): encoder lr
  5e-5, head lr 1e-3, 40 epochs (3,640 steps, the v1 budget), batch 8, warmup 0.1, weight decay
  0.01, dropout 0.1, last epoch kept, no selection. The v1 student lr (5e-4) is 10× the usual BERT
  fine-tuning rate and would overwrite copied weights; 5e-5 is the teacher's own fine-tuning lr.
- **Fit check (declared in advance):** a supplementary random arm at the v1 lr was planned if arm
  C failed to fit. It did fit: all 9 runs end at in-sample validation type F1 1.000 and span F1
  1.000, arm C train loss < 1e-4. No supplementary arm was run, so arm C is not an
  under-trained control.
- **Metrics:** recomputed for all six models from saved predictions with one code path; the
  teacher and v1 numbers reproduce `experiments/distillation-v1/comparison.json` exactly. Mean ±
  sample std over 3 seeds (no ± = identical across seeds, or a single teacher run). Joint = type
  correct and span exactly correct. Subword fragment = predicted span starting or ending inside a
  word.

## 2. Results

Columns: final teacher, v1 random 4×256 supervised / distilled, v2 random 4×768 supervised (C),
v2 pretrained 4×768 supervised (A) / distilled (B).

### Frozen test (105)

| | teacher | v1 256 rand sup | v1 256 rand KD | v2 768 rand sup (C) | v2 768 pre sup (A) | v2 768 pre KD (B) |
|---|---|---|---|---|---|---|
| type accuracy | 0.895 | 0.889±0.038 | 0.886±0.010 | 0.889±0.005 | 0.927±0.005 | 0.933±0.010 |
| type macro-F1 | 0.886 | 0.887±0.037 | 0.871±0.021 | 0.891±0.015 | 0.928±0.007 | 0.939±0.011 |
| span F1 | 0.940 | 0.648±0.027 | 0.662±0.020 | 0.620±0.013 | 0.830±0.053 | 0.831±0.006 |
| span exact | 0.952 | 0.768±0.005 | 0.775±0.011 | 0.749±0.011 | 0.898±0.027 | 0.898±0.005 |
| type+span joint | 0.857 | 0.702±0.024 | 0.705±0.016 | 0.679±0.015 | 0.844±0.031 | 0.844±0.005 |
| false span, gold null (54) | 1 | 5.0±2.6 | 4.7±0.6 | 6.3±0.6 | 2.7±0.6 | 3.0±1.0 |
| missed span (51) | 3 | 9.3±1.2 | 10.0±1.0 | 8.3±2.3 | 1 | 0.7±0.6 |
| wrong span | 1 | 10.0±2.0 | 9.0±1.0 | 11.7±1.5 | 7.0±2.6 | 7.0±1.0 |
| subword-fragment spans | 0 | 8.0±1.7 | 5.3±0.6 | 9.3±2.5 | 4.3±2.5 | 1.3±1.2 |

The teacher is a single last-epoch run; on test type macro-F1 it sits below the 3-seed
targeted-v2 reference (0.930), so students "beating" it on type are on par with targeted-v2.

### Probe-v1 (81)

| | teacher | v1 256 rand sup | v1 256 rand KD | v2 768 rand sup (C) | v2 768 pre sup (A) | v2 768 pre KD (B) |
|---|---|---|---|---|---|---|
| type accuracy | 0.901 | 0.733±0.050 | 0.687±0.026 | 0.691±0.021 | 0.811±0.031 | 0.819±0.019 |
| type macro-F1 | 0.928 | 0.694±0.051 | 0.649±0.023 | 0.660±0.023 | 0.795±0.034 | 0.787±0.035 |
| span F1 | 0.920 | 0.565±0.054 | 0.624±0.022 | 0.593±0.032 | 0.792±0.038 | 0.771±0.013 |
| span exact | 0.926 | 0.630±0.045 | 0.691±0.012 | 0.654±0.025 | 0.831±0.026 | 0.819±0.007 |
| type+span joint | 0.840 | 0.477±0.038 | 0.494±0.033 | 0.436±0.040 | 0.667±0.000 | 0.667±0.012 |
| false span, gold null | 3 | 7.3±1.2 | 4.3±1.2 | 6.7±0.6 | 2.3±1.2 | 2.3±0.6 |
| missed span | 0 | 3.0±1.7 | 4.3±0.6 | 3.0±1.0 | 2.0±1.7 | 1.7±1.2 |
| wrong span | 3 | 19.7±2.3 | 16.3±1.5 | 18.3±2.1 | 9.3±2.5 | 10.7±1.5 |
| subword-fragment spans | 3 | 11.0±1.7 | 5.7±1.2 | 8.7±3.2 | 6.3±2.5 | 5.3±2.1 |

### Retention vs the final teacher (mean)

| | test type F1 | test span F1 | test joint | probe type F1 | probe span F1 | probe joint |
|---|---|---|---|---|---|---|
| v1 256 rand sup | 100% | 69% | 82% | 75% | 61% | 57% |
| v1 256 rand KD | 98% | 70% | 82% | 70% | 68% | 59% |
| v2 768 rand sup (C) | 101% | 66% | 79% | 71% | 64% | 52% |
| v2 768 pre sup (A) | 105% | 88% | 99% | 86% | 86% | 79% |
| v2 768 pre KD (B) | 106% | 88% | 99% | 85% | 84% | 79% |

### Per-class type F1 (test)

| | teacher | v1 256 rand sup | v1 256 rand KD | v2 768 rand sup (C) | v2 768 pre sup (A) | v2 768 pre KD (B) |
|---|---|---|---|---|---|---|
| expense | 0.90 | 0.88 | 0.90 | 0.86 | 0.92 | 0.92 |
| income | 0.82 | 0.78 | 0.80 | 0.79 | 0.83 | 0.82 |
| borrow | 0.94 | 0.85 | 0.82 | 0.90 | 0.93 | 0.92 |
| lend | 1.00 | 0.74 | 0.75 | 0.98 | 0.98 | 0.96 |
| repayment_in | 0.77 | 0.98 | 0.86 | 0.88 | 0.96 | 0.98 |
| repayment_out | 0.87 | 0.95 | 0.89 | 0.88 | 0.93 | 0.96 |
| transfer | 0.95 | 0.94 | 0.96 | 0.96 | 0.96 | 0.95 |
| refund | 0.84 | 0.98 | 0.98 | 0.86 | 0.92 | 1.00 |

### Probe-v1, all 9 patterns (9 notes each)

Joint success mean (type accuracy / span exact; notes failed by all 3 seeds).

| | teacher | v1 256 rand sup | v1 256 rand KD | v2 768 rand sup (C) | v2 768 pre sup (A) | v2 768 pre KD (B) |
|---|---|---|---|---|---|---|
| lender_first_cho_muon | 1.00 (1.00/1.00; 0) | 0.48 (0.67/0.81; 2) | 0.41 (0.63/0.78; 3) | 0.37 (0.56/0.81; 4) | 0.78 (0.85/0.93; 0) | 0.78 (0.89/0.89; 1) |
| subject_first_muon | 0.89 (1.00/0.89; 1) | 0.52 (0.52/0.74; 3) | 0.52 (0.52/0.74; 3) | 0.52 (0.56/0.81; 4) | 0.89 (0.93/0.96; 0) | 0.85 (0.89/0.96; 0) |
| family_gift_in | 0.89 (0.89/1.00; 1) | 0.44 (0.56/0.67; 4) | 0.48 (0.59/0.74; 4) | 0.41 (0.56/0.74; 3) | 0.67 (0.70/0.89; 2) | 0.74 (0.78/0.89; 2) |
| gift_out | 0.89 (0.89/1.00; 1) | 0.56 (0.85/0.59; 2) | 0.56 (0.81/0.67; 3) | 0.52 (0.78/0.67; 4) | 0.56 (0.78/0.74; 4) | 0.56 (0.78/0.74; 4) |
| insurance_premium | 0.67 (0.67/1.00; 3) | 0.37 (0.78/0.56; 4) | 0.44 (0.67/0.67; 4) | 0.19 (0.63/0.41; 7) | 0.52 (0.85/0.67; 3) | 0.63 (0.85/0.78; 2) |
| insurance_payout | 0.89 (1.00/0.89; 1) | 0.48 (0.74/0.63; 3) | 0.52 (0.70/0.74; 3) | 0.52 (0.74/0.74; 4) | 0.63 (0.74/0.89; 2) | 0.63 (0.74/0.89; 2) |
| windfall | 0.78 (0.89/0.89; 2) | 0.67 (0.89/0.78; 1) | 0.67 (0.78/0.89; 2) | 0.59 (0.81/0.74; 2) | 0.74 (0.89/0.85; 2) | 0.70 (0.89/0.81; 1) |
| loan_installment | 0.56 (0.78/0.67; 4) | 0.37 (0.74/0.44; 4) | 0.37 (0.67/0.52; 4) | 0.37 (0.70/0.52; 5) | 0.52 (0.67/0.74; 3) | 0.56 (0.67/0.78; 3) |
| title_name_span | 1.00 (1.00/1.00; 0) | 0.41 (0.85/0.44; 5) | 0.48 (0.81/0.48; 4) | 0.44 (0.89/0.44; 4) | 0.70 (0.89/0.81; 2) | 0.56 (0.89/0.63; 4) |

### Accent slices (type acc / span F1 / span exact)

| | teacher | v1 256 rand sup | v1 256 rand KD | v2 768 rand sup (C) | v2 768 pre sup (A) | v2 768 pre KD (B) |
|---|---|---|---|---|---|---|
| test accented (63) | 0.94 / 0.94 / 0.95 | 0.88 / 0.62 / 0.74 | 0.87 / 0.66 / 0.76 | 0.89 / 0.63 / 0.74 | 0.94 / 0.89 / 0.93 | 0.93 / 0.83 / 0.90 |
| test unaccented (42) | 0.83 / 0.94 / 0.95 | 0.90 / 0.70 / 0.81 | 0.91 / 0.67 / 0.79 | 0.88 / 0.60 / 0.77 | 0.91 / 0.73 / 0.85 | 0.94 / 0.83 / 0.90 |
| probe accented (49) | 0.90 / 0.99 / 0.98 | 0.77 / 0.60 / 0.65 | 0.70 / 0.64 / 0.70 | 0.71 / 0.61 / 0.65 | 0.84 / 0.81 / 0.86 | 0.84 / 0.77 / 0.82 |
| probe unaccented (32) | 0.91 / 0.82 / 0.84 | 0.68 / 0.51 / 0.59 | 0.67 / 0.60 / 0.68 | 0.66 / 0.57 / 0.66 | 0.77 / 0.76 / 0.79 | 0.78 / 0.77 / 0.81 |

### Focus areas

| | teacher | v1 256 rand sup | v1 256 rand KD | v2 768 rand sup (C) | v2 768 pre sup (A) | v2 768 pre KD (B) |
|---|---|---|---|---|---|---|
| test lend→borrow, per seed | 0 | 2.67 | 2.33 | 0.33 | 0.33 | 0.33 |
| test borrow→lend, per seed | 0 | 0 | 0.33 | 0 | 0 | 0.33 |
| test span exact, gold target seen in train (36) | 0.94 | 0.72±0.03 | 0.72±0.00 | 0.71±0.07 | 0.92±0.05 | 0.88±0.04 |
| test span exact, gold target unseen in train (15) | 0.87 | 0.38±0.10 | 0.40±0.07 | 0.36±0.10 | 0.67±0.07 | 0.78±0.10 |
| teacher-correct notes failed by all 3 seeds, test | – | 14 | 15 | 13 | 3 | 4 |
| teacher-correct notes failed by all 3 seeds, probe | – | 22 | 19 | 28 | 10 | 12 |

"Unseen" = the gold target string never occurs in any training note (a proxy for unseen
merchants and names).

Teacher-correct notes still failed by every seed of arm A:

- test: `coopmart đồ dùng tuần 412k` (type → income/repayment_out), `co Thuy cho muon 1tr` (span
  `uy`), `điện máy xanh hoàn tiền …` (span `máy xanh`);
- probe: `bo me cho 5tr dong hoc` (span `me`), `mừng tuổi ông bà` (span `ông` or `bà`), `gửi ba 3
  triệu biếu tết` (→ transfer), `li xi be Su 50k` (no span), two insurance payouts (→ income /
  expense), one premium (→ refund), `minigame shopee trúng 100k` (span `min`), `bác Thành hoàn
  lại 1tr đã mượn` (→ refund), `chú Bình gửi lại 500k` (span keeps `chú`).

## 3. Diagnosis

1. **Does 4×768 random beat 4×256 random substantially? No.** Test joint 0.679 vs 0.702, span
   F1 0.620 vs 0.648, probe joint 0.436 vs 0.477. Paired by seed, the differences change sign
   (test joint −0.038/−0.048/+0.019, probe joint +0.037/−0.099/−0.062). Every span and probe
   pattern problem of v1 is still there (title/name 0.44, lender-first 0.37, 9.3 test fragments).
   The one gain is on test lend→borrow (2.67 → 0.33 per seed, lend F1 0.74 → 0.98), but it does
   not carry to the probe direction patterns. Width at random init is not the missing piece.
2. **Does 4×768 pretrained beat 4×768 random substantially? Yes, on every seed and metric.**
   Paired by seed: test joint +0.14/+0.19/+0.16, test span F1 +0.16/+0.24/+0.23, probe joint
   +0.19/+0.25/+0.26, probe type accuracy +0.10/+0.10/+0.16. The gaps are 5–10× the seed std.
   Unseen-target span exact goes 0.36 → 0.67, and teacher-correct notes lost on all seeds drop
   13 → 3 (test) and 28 → 10 (probe).
3. **Does KD help once the student is pretrained? No measurable effect.** B vs A: test joint
   0.844 vs 0.844, probe joint 0.667 vs 0.667; paired differences change sign (test joint
   +0.038/−0.029/−0.010, probe joint −0.012/+0.012/0). Some indicators move in KD's favour
   (fewer test fragments 1.3 vs 4.3, unseen-target exact 0.78 vs 0.67, much lower test span-F1
   std 0.006 vs 0.053), others against it (title/name 0.56 vs 0.70). With 3 seeds and near-one-hot
   teacher targets on the training set (distillation-v1 §softness), none of these is a reliable
   KD effect.
4. **Does it recover the teacher's span behaviour? Mostly, not fully.** Test span F1 0.83 vs
   0.94 (88%), span exact 0.90 vs 0.95. Missed spans fall below the teacher's (1 vs 3), and false
   spans are close (2.7 vs 1). What remains is **boundaries**: 7 wrong spans vs 1: title/kinship
   prefixes kept (`chú Bình`, `dì`), subword fragments (`uy`, `min`, `máy xanh`), and two-word
   kinship targets cut to one word (`me`, `ông`). Probe title/name recovers 0.41 → 0.70 against
   the teacher's 1.00. Unseen-target exact 0.67–0.78 vs 0.87.
5. **Does it recover the targeted-v2 directional patterns? Largely.** Subject-first mượn 0.89 =
   teacher (0 all-seed fails); lender-first cho mượn 0.78 vs 1.00 (0–1 all-seed fails, against 2–4
   for every random student); test lend→borrow 0.33 per seed vs 2.67 in v1. Family gift in
   recovers partly (0.67–0.74 vs 0.89). Gift-out does not move (0.56, mostly → transfer).
6. **How much of the teacher's probe joint success is retained?** 79% for both pretrained arms
   (0.667 / 0.840), up from 57–59% for v1 and 52% for random 4×768. On test, joint retention is
   98.5%.

## 4. Size (4×768, measured on arm A seed 1)

Seed 1 was fixed in advance; size does not depend on the arm or seed.

| | value |
|---|---|
| parameters | 45,666,059 (teacher 102,369,035) |
| word embeddings | 15,729,408 (34%) |
| position embeddings (2,050 rows; 34 used at max_length 32) | 1,574,400 (3%) |
| 4 transformer layers | 28,351,488 (62%), 7.09M per layer |
| heads + other | 10,763 |
| safetensors fp32 | 182.7 MB |
| ONNX fp32 / INT8 | 182.7 MB / 45.9 MB (teacher INT8 102.9 MB) |
| ONNX parity (fp32 vs torch) | max abs diff 2.8e-5 |
| INT8 vs fp32, 50 notes | type agreement 1.000, tag agreement 0.994 |
| INT8 on full test / probe | 4 / 2 notes change; test joint 0.810 → 0.829, probe joint 0.667 → 0.654 |
| CPU latency p50, 1 thread (this Mac) | torch 3.73 ms, ONNX fp32 2.60 ms, INT8 1.42 ms (teacher INT8 4.5 ms) |

Training cost: about 19.5 s per epoch on MPS with three runs sharing the GPU.

At 45.9 MB INT8 the model is outside the 10–30 MB target. Per the brief, that is not a rejection
criterion here.

## 5. Conclusion and next direction

**A. PRETRAINING IS THE MAIN MISSING PIECE.**

Tripling width at random initialization changes nothing (Q1). Copying four pretrained BamiBERT
layers into the same architecture lifts every metric on every seed, nearly matches the teacher
on test (joint 98.5%), and brings back the direction and boundary behaviours that v1 lost (Q2,
Q4, Q5). It is not D: test is close to the teacher, and the probe gap (79% retention) is
specific (title/name boundaries, gift and insurance types) rather than across the board. It is
not B or C: width without pretraining gave no gain. The cause of the remaining probe gap (4
instead of 12 layers, the choice of layers, or recipe) is not separable with this design. KD with
the current objective is neutral once the student is pretrained (Q3).

**Recommended next student direction (not trained):** keep pretrained BamiBERT initialization as
a hard requirement and compress *without* changing the 768 width, so every weight stays
copyable:

1. truncate the position table to the 34 rows max_length 32 uses (lossless, about −1.6 MB INT8);
2. prune the 20,481-row vocabulary to the tokens the domain needs (embeddings are 34% of the
   model), with an explicit out-of-vocabulary policy, since unseen merchant names are a measured
   weakness;
3. only then trade depth: 3 or 2 pretrained layers (~7 MB INT8 each) against the probe gap.

Narrower random students are not worth further work. A narrower width is worth revisiting only
with a pretrained source of that width on the same vocabulary (e.g. a task-agnostic distilled
BamiBERT).

## Artifacts

- `protocol.json`: the pre-declared design, recipe and fit check.
- `comparison.json`: every number above, per-seed values, confusions, lost-note lists,
  provenance.
- `runs/{supervised,distilled}/student-4x768[-pretrained]/seed{1,2,3}/`: configs (including the
  full copy map in `init_report`), train logs, metrics, test predictions.
- `probe-eval-{supervised,distilled}.json`; `export-4x768-pretrained-supervised-seed1.json`.
- Weights: `models/distillation-v2/…`; ONNX: `models/distillation-v2-onnx/…`.
- Code: `student-4x768` and `copy_pretrained_encoder` / `layer_map` in
  `src/gidi/distillation/student.py`; `--student` / `--init` in `scripts/train_student.py`.

Command (per arm):

```sh
uv run python scripts/train_student.py --arm {supervised,distilled} --student student-4x768 \
  --init {pretrained,random} --lr 5e-5 --head-lr 1e-3 --epochs 40 \
  --out-dir experiments/distillation-v2/runs --weights-dir models/distillation-v2 \
  --seed 1 --seed 2 --seed 3
```
