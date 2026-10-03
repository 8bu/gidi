# compression-v2: pretrained 3×768 student

**Question.** compression-v1 left the pretrained 4×768 student at 34.97 MB INT8 (positions 34,
vocabulary B-rank-8000) with no quality loss. Does removing one transformer layer get it under
30 MB INT8 while keeping held-out quality?

**Answer.** Size: yes, **27.84 MB INT8** (estimate 27.9). Quality: no. Probe joint drops
0.654 → 0.576 and probe span F1 0.783 → 0.663, lower on every seed. Span boundaries degrade
systematically (wrong spans 8.7 → 15.7 on probe, more subword fragments), title/name and
subject-first borrow/lend patterns lose the most, and test span exact drops 0.905 → 0.857.
Conclusion **B**: keep the compressed 4×768.

## 1. Setup

Only depth changed compared to compression-v1 (declared in `protocol.json` before training).

- `student-3x768`: 3 layers, hidden 768, 12 heads, FFN 3072; BamiBERT tokenizer pruned to
  B-rank-8000 (`models/compression-v1/vocab/B-rank-8000`, unchanged), 34-row position table,
  max_length 32, same type head and BIO span head.
- Init: BamiBERT layers copied with `layer_map(12, 3)` = 0-based **3, 7, 11** (1-based 4, 8,
  12); 53 tensors / 38,569,728 values copied exactly, embeddings included, then the same
  position truncation and exact vocab-row copy as compression-v1. Fresh seeded heads.
- Recipe (identical to C): encoder lr 5e-5, head lr 1e-3, batch 8, 40 epochs, warmup 0.1,
  weight decay 0.01, dropout 0.1, max grad norm 1.0, same 723 training notes, same seeded batch
  order, last epoch kept, no KD, no sweep. Seeds 1–3 all fit (in-sample type F1 and span F1
  1.000).
- One RNG difference: heads are initialized after the encoder, so a 3-layer encoder leaves a
  different RNG state and head init differs from C for the same seed.
- Test (105) and probe-v1 (81) are evaluation only.

Model sha256 (seed 1/2/3): `95c3c15f…`, `b75f2f69…`, `68be0cef…`.

## 2. Held-out results (mean ± std over 3 seeds)

A = canonical teacher (12×768, one run), B = pretrained 4×768 full vocabulary,
C = compression-v1 4×768 pos34 + B-rank-8000, D = 3×768 pos34 + B-rank-8000.

| test | A teacher | B 4×768 full | C 4×768 compressed | **D 3×768** |
|---|---|---|---|---|
| type accuracy | 0.895 | 0.927±0.005 | 0.933±0.010 | 0.930±0.020 |
| type macro-F1 | 0.886 | 0.928±0.007 | 0.935±0.009 | 0.940±0.016 |
| span F1 | 0.940 | 0.830±0.053 | 0.842±0.031 | **0.769±0.052** |
| span exact | 0.952 | 0.898±0.027 | 0.905±0.016 | **0.857±0.033** |
| type+span joint | 0.857 | 0.844±0.031 | 0.854±0.031 | **0.816±0.022** |
| false span on null gold | 1 | 2.7 | 2.7 | 4.3 |
| missed span | 3 | 1.0 | 1.0 | 1.3 |
| wrong span | 1 | 7.0 | 6.3 | 9.3 |
| subword-fragment spans | 0 | 4.3 | 4.0 | 7.3 |

| probe | A teacher | B 4×768 full | C 4×768 compressed | **D 3×768** |
|---|---|---|---|---|
| type accuracy | 0.901 | 0.811±0.031 | 0.819±0.038 | 0.778±0.037 |
| type macro-F1 | 0.928 | 0.795±0.034 | 0.813±0.036 | 0.767±0.034 |
| span F1 | 0.920 | 0.792±0.038 | 0.783±0.034 | **0.663±0.047** |
| span exact | 0.926 | 0.831±0.026 | 0.815±0.021 | **0.733±0.040** |
| type+span joint | 0.840 | 0.667±0.000 | 0.654±0.012 | **0.576±0.051** |
| false span on null gold | 3 | 2.3 | 2.7 | 3.3 |
| missed span | 0 | 2.0 | 3.7 | 2.7 |
| wrong span | 3 | 9.3 | 8.7 | **15.7** |
| subword-fragment spans | 3 | 6.3 | 6.3 | 7.7 |

Per seed, joint (C → D): test 0.819 → 0.829, 0.867 → 0.790, 0.876 → 0.829; probe 0.667 →
0.519, 0.642 → 0.593, 0.654 → 0.617. Probe joint is lower on all three seeds (−0.148, −0.049,
−0.037); probe span F1 is lower on all three (−0.168, −0.163, −0.030). Type accuracy is
essentially unchanged on test; the loss is mostly in spans.

**Seed variance** grows: probe joint std 0.012 → 0.051, test span F1 std 0.031 → 0.052.

## 3. Probe patterns (joint success; type acc / span exact in brackets)

| pattern (n=9) | B 4×768 full | C 4×768 compressed | D 3×768 |
|---|---|---|---|
| lender-first cho mượn | 0.78 | 0.78 (0.85/0.93) | 0.78 (0.85/0.93) |
| **subject-first mượn/vay** | 0.89 | 0.89 (0.93/0.96) | **0.59** (0.78/0.78) |
| family gift in | 0.67 | 0.67 (0.67/0.89) | 0.67 (0.67/0.85) |
| gift out | 0.56 | 0.56 (0.78/0.74) | 0.56 (0.78/0.70) |
| insurance premium | 0.52 | 0.52 (0.85/0.67) | 0.48 (0.70/0.67) |
| insurance payout | 0.63 | 0.52 (0.78/0.74) | 0.48 (0.78/0.70) |
| loan installment | 0.52 | 0.52 (0.74/0.74) | 0.48 (0.67/0.63) |
| windfall | 0.74 | 0.74 (0.89/0.85) | 0.63 (0.89/0.74) |
| **title/name span** | 0.70 | 0.70 (0.89/0.81) | **0.52** (0.89/0.59) |

- **Title/name boundaries:** clear regression. Type is unchanged (0.89) but span exact drops
  0.81 → 0.59; D predicts the title alone (`cho dì Hương mượn` → `dì` on all seeds).
- **Borrow/lend direction:** subject-first loses 0.30. `Phúc vay 700k mua điện thoại` flips to
  borrow on all 3 seeds; `chi Nga vay 10 cu` gets span `chi`. Probe lend→borrow errors 0.7 →
  2.0 per seed. Lender-first is unchanged and gains two notes (`thg Khoa cho muon`,
  `me cho muon`), so direction isn't uniformly worse.
- **Gift direction:** unchanged (family gift in 0.67, gift out 0.56); one gift-in note
  (`ông nội cho 1 triệu`) becomes borrow on 2 seeds.
- **Insurance payout / loan installment:** −1 note each in total; payout spans become fragments
  (`bảo việt` → `ảo việt` on all seeds).

## 4. Slices

| type acc / span F1 / span exact | C | D |
|---|---|---|
| test accented (63) | 0.94 / 0.90 / 0.94 | 0.95 / 0.73 / 0.83 |
| test unaccented (42) | 0.92 / 0.75 / 0.86 | 0.90 / 0.85 / 0.90 |
| probe accented (49) | 0.84 / 0.80 / 0.84 | 0.78 / 0.70 / 0.76 |
| probe unaccented (32) | 0.78 / 0.76 / 0.78 | 0.77 / 0.61 / 0.70 |

Spans get worse on three of the four slices. Test unaccented improves (+0.10 span F1), so the
damage is not specific to accents.

**Unseen targets** (target string absent from all training notes), span exact C → D: test
unseen 0.69 → 0.67 (15), seen 0.93 → 0.84 (36); probe unseen 0.64 → 0.55 (14), seen 0.82 →
0.71 (41). Both seen and unseen targets degrade, so the loss isn't limited to new merchants:
3 layers draw span boundaries less precisely in general. Probe unseen merchants show it as
fragments (`vietlott` → `etlott`/`ott`, `bảo việt` → `ảo việt`).

## 5. Note-level regression (C vs D)

| | test | probe |
|---|---|---|
| correct on all 3 C seeds, wrong on all 3 D seeds | 2 | 3 |
| wrong on all 3 C seeds, correct on all 3 D seeds | 2 | 0 |
| C majority correct → D majority wrong | 8 | 10 |
| C majority wrong → D majority correct | 6 | 2 |

All-seed losses:
- test `Lộc trả tiền nợ tuần trước` → span `L` (fragment); `đòi được nợ thằng Lâm` → type wrong.
- probe `Phúc vay 700k` → borrow (direction); `chi Nga vay 10 cu` → `chi` (title only);
  `cho dì Hương mượn 2tr` → `dì` (title only).

All-seed gains (test only): `bố cho tiền tiêu`, `co Thuy cho muon`.

Majority-level losses cluster in **span boundaries**: 12 of 18 are span errors, mostly
truncated names (`Th` for `Thắng`, `rể` for `anh rể`, `L` for `Lộc`), title-only spans, or
spans on the wrong word (`thọ`, `ph`, `ba`). The rest are borrow/lend direction and isolated
type errors. On test this roughly nets out (8 vs 6); on probe it does not (10 vs 2). No single
note drives the conclusion: the drop shows up across patterns, seeds and both splits.

## 6. Size, latency, quantization (seed 1; CPU batch 1, 1 thread, this Mac)

| | C 4×768 compressed | D 3×768 |
|---|---|---|
| parameters | 34,791,947 | **27,704,075** |
| FP32 checkpoint | 139.18 MB | 110.82 MB |
| ONNX FP32 | 139.21 MB | 110.85 MB |
| **ONNX INT8** | **34.97 MB** | **27.84 MB** (estimate 27.9) |
| ONNX FP32 parity (max abs diff) | ≤1e-4 | 2.1e-5 |
| torch FP32 p50 / p95 | 4.12 / 4.44 ms | 2.74 / 3.09 ms |
| ONNX FP32 p50 / p95 | 2.90 / 3.29 ms | 1.94 / 2.17 ms |
| ONNX INT8 p50 / p95 | 1.57 / 1.95 ms | 1.08 / 1.34 ms |
| INT8 vs FP32 agreement, 50 notes (type / tag) | 1.000 / 1.000 | 0.980 / 0.994 |
| INT8 − FP32 test joint | 0 | −0.029 |
| INT8 − FP32 probe joint | 0 | +0.037 |

D is 7.1 MB smaller and about 30% faster, but INT8 is less faithful for D: its predictions move
by ±3–4 points per split, vs exactly zero for C.

## 7. Decision

| criterion (option A) | result |
|---|---|
| INT8 ≤ 30 MB | **met** (27.84 MB) |
| test joint close to C | not met: −0.038 mean, span exact −0.048 |
| probe joint close to C | not met: −0.078, lower on all seeds |
| no systematic span regression | not met: probe span F1 −0.12, wrong spans ×1.8, title/name −0.19 |
| seed stability acceptable | weaker: probe joint std 0.012 → 0.051 |

The loss is concentrated in exactly the behaviour earlier experiments worked to fix (title/name
spans, subject-first direction, span boundaries), and it is larger than the noise between C and
B. Since it is not a small, ambiguous drop, option C (borderline) doesn't fit either.

**Next-step implication (not run):** the 30 MB target and 4-layer quality are in tension under
this recipe. Options are accepting ~35 MB, or a different recipe for the 3×768 (e.g. KD or a
different layer choice). Both are new experiments; 2×768 is not indicated.

## Conclusion

**B. Keep compressed 4x768 despite being above the size target.**

## Artifacts

- `protocol.json`, `comparison.json` (metrics, per-seed values, paired deltas, patterns,
  slices, direction confusions, note-level flips with predictions, size/latency, quantization).
- `runs/supervised/student-3x768-pretrained-pos32-vocab-B-rank-8000/seed{1,2,3}/`,
  `train-seed{1,2,3}.log`, `probe-eval.json`, `export-3x768-seed1.json`, `onnx-eval.json`.
- Models: `models/compression-v2/supervised/student-3x768-pretrained-pos32-vocab-B-rank-8000/
  seed{1,2,3}/`, `models/compression-v2-onnx/pos32-vocabB8000-3x768-seed1/`.

```sh
uv run python scripts/train_student.py --arm supervised --student student-3x768 \
  --init pretrained --truncate-positions --vocab-spec models/compression-v1/vocab/B-rank-8000 \
  --lr 5e-5 --head-lr 1e-3 --epochs 40 --out-dir experiments/compression-v2/runs \
  --weights-dir models/compression-v2 --seed 1 --seed 2 --seed 3
uv run python scripts/evaluate_probe.py --weights-dir models/compression-v2/supervised \
  --out experiments/compression-v2/probe-eval.json
```
