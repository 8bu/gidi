# compression-v3: structured FFN pruning of the compressed 4×768 student

**Question.** compression-v2 showed 3×768 reaches 27.8 MB INT8 but loses span quality. Does
keeping all 4 layers and shrinking only each FFN (3072 → 2048 neurons, pretrained neurons
copied exactly) reach the size target while keeping span boundaries?

**Answer.** Mostly. **28.66 MB INT8** (estimate 28.64). Span quality is close to the
compressed 4×768 and far better than 3×768 (probe span F1 0.767 vs 0.783 vs 0.663; wrong spans
9.3 vs 8.7 vs 15.7). Test joint is unchanged within noise (0.844±0.005 vs 0.854±0.031). Probe
joint drops 0.654 → 0.626, about 2 of 81 notes, almost all of it borrow/lend **type**
direction in the lender-first pattern (0.78 → 0.59). Conclusion **C**: promising, one width
check justified.

## 1. Parameter budget (compression-v1 4×768, 34,791,947 params)

| group | params | share |
|---|---|---|
| embeddings (8,338 words + 34 positions + 1 type) | 6,430,464 | 18.5% |
| attention Q/K/V/O, 4 layers | 9,449,472 | 27.2% |
| **FFN W1/b1/W2/b2, 4 layers** | **18,889,728** | **54.3%** |
| layer norms | 13,824 | 0.04% |
| task heads | 8,459 | 0.02% |

The FFN is now the largest block. One intermediate neuron owns a W1 row, a b1 entry and a W2
column: 1,537 values per layer, 6,148 across 4 layers.

INT8 estimates use the measured 34.97 MB minus removed params × 1.0057 bytes/param (the INT8
cost per transformer parameter measured in compression-v2):

| FFN width | FFN kept | params | est. INT8 |
|---|---|---|---|
| 3072 (B) | 100% | 34,791,947 | 34.97 MB (measured) |
| 2816 | 92% | 33,218,059 | 33.38 MB |
| 2560 | 83% | 31,644,171 | 31.80 MB |
| 2304 | 75% | 30,070,283 | 30.22 MB |
| **2048** | **67%** | **28,496,395** | **28.64 MB** |
| 1792 | 58% | 26,922,507 | 27.05 MB |

## 2. Pruning criteria (training data only)

Scored on the compression-v1 *initial* model (pretrained BamiBERT layers 0-based 2/5/8/11,
pos34, B-rank-8000), before any fine-tuning. That encoder doesn't depend on the seed, so one
neuron map serves all seeds. Activations come from the 723 training notes (7,609 non-pad
tokens); test and probe were not touched.

- **magnitude**: ‖W1[j]‖ · ‖W2[:, j]‖
- **activation**: mean |gelu(W1[j]·x + b1[j])| over training tokens
- **combined**: activation · ‖W2[:, j]‖, the mean size of neuron j's contribution to the FFN output

Train-only proxy: the share of FFN-output energy (‖W2·a‖²) on training tokens carried by the
removed neurons. Lower is better.

| removed energy | 2560 | 2304 | 2048 | 1792 |
|---|---|---|---|---|
| magnitude | 0.215 | 0.302 | 0.376 | 0.451 |
| activation | 0.115 | 0.174 | 0.226 | 0.266 |
| **combined** | **0.089** | **0.144** | **0.193** | 0.252 |

Combined is lowest at every width from 2048 up. Weight magnitude alone is a poor proxy: at
2048 it shares only 67% of its kept neurons with combined; activation shares 86%. Per layer
(combined, 2048): 0.165 / 0.189 / 0.143 / 0.273, so the top layer is the most distributed.

## 3. Chosen policy (declared in `protocol.json` before training)

- **combined criterion, K = 2048 in every layer** (HF stores one `intermediate_size`). It is
  the largest width inside the 28–30 MB band: 2304 is estimated at 30.2 MB, and 1792 would
  remove over 40% of each FFN.
- Neuron map: `models/compression-v3/ffn/combined-K2048/ffn_map.json` (sha256 `486084a8…`),
  with ascending kept indices per layer, their scores, the removed-energy proxy, and the base
  model it is valid for. The trainer refuses a map built for a different student, vocabulary or
  position setup.
- Kept neurons are copied exactly: W1 rows, b1 entries and W2 columns, in original order. b2,
  attention, layer norms, embeddings and heads are untouched; no neuron is reinitialized.
  Keeping all 3072 neurons reproduces the model bitwise (unit-tested).
- Pruning happens after the heads are built, so each seed's heads are bit-identical to
  compression-v1's for that seed.

Code: `src/gidi/compression/ffn.py`, `scripts/prune_ffn.py`, trainer flag `--ffn-map`.

## 4. Training

Compression-v1 supervised recipe unchanged: encoder lr 5e-5, head lr 1e-3, batch 8, 40 epochs,
warmup 0.1, weight decay 0.01, dropout 0.1, max grad norm 1.0, same 723 notes, last epoch kept,
no KD, no sweep, seeds 1–3. All seeds fit (in-sample type F1 and span F1 1.000; final loss
1e-4).

Model sha256 (seed 1/2/3): `bfb2195a…`, `f3c80e0d…`, `ed7971eb…`.

## 5. Held-out results (mean ± std, 3 seeds)

A = canonical teacher (one run), B = compression-v1 4×768, C = compression-v2 3×768,
D = 4×768 FFN 2048.

| test | A | B 4×768 | C 3×768 | **D 4×768 FFN 2048** |
|---|---|---|---|---|
| type accuracy | 0.895 | 0.933±0.010 | 0.930±0.020 | 0.933±0.019 |
| type macro-F1 | 0.886 | 0.935±0.009 | 0.940±0.016 | 0.943±0.018 |
| span F1 | 0.940 | 0.842±0.031 | 0.769±0.052 | **0.841±0.034** |
| span exact | 0.952 | 0.905±0.016 | 0.857±0.033 | **0.895±0.010** |
| type+span joint | 0.857 | 0.854±0.031 | 0.816±0.022 | **0.844±0.005** |
| false span on null gold | 1 | 2.7 | 4.3 | 4.0 |
| missed span | 3 | 1.0 | 1.3 | 1.3 |
| wrong span | 1 | 6.3 | 9.3 | 5.7 |
| subword-fragment spans | 0 | 4.0 | 7.3 | 2.7 |

| probe | A | B 4×768 | C 3×768 | **D 4×768 FFN 2048** |
|---|---|---|---|---|
| type accuracy | 0.901 | 0.819±0.038 | 0.778±0.037 | 0.798±0.019 |
| type macro-F1 | 0.928 | 0.813±0.036 | 0.767±0.034 | 0.780±0.020 |
| span F1 | 0.920 | 0.783±0.034 | 0.663±0.047 | **0.767±0.021** |
| span exact | 0.926 | 0.815±0.021 | 0.733±0.040 | **0.807±0.014** |
| type+span joint | 0.840 | 0.654±0.012 | 0.576±0.051 | **0.626±0.019** |
| false span on null gold | 3 | 2.7 | 3.3 | 2.0 |
| missed span | 0 | 3.7 | 2.7 | 4.3 |
| wrong span | 3 | 8.7 | 15.7 | 9.3 |
| subword-fragment spans | 3 | 6.3 | 7.7 | 4.7 |

Paired by seed (D − B), joint: test +0.029 / −0.029 / −0.029; probe −0.062 / 0 / −0.025.

**Seed variance doesn't grow:** test joint std 0.031 → 0.005, probe 0.012 → 0.019 (C was
0.051).

**Main question, span boundaries:** yes. Keeping 4 layers preserves them. D matches B on span
F1/exact (test 0.841 vs 0.842, probe 0.767 vs 0.783) with fewer fragments. C lost 0.07–0.12
span F1.

## 6. Probe patterns (joint success; type acc / span exact)

| pattern (n=9) | B 4×768 | C 3×768 | D FFN 2048 |
|---|---|---|---|
| **lender-first cho mượn** | 0.78 (0.85/0.93) | 0.78 (0.85/0.93) | **0.59** (0.67/0.93) |
| subject-first mượn/vay | 0.89 (0.93/0.96) | 0.59 (0.78/0.78) | 0.78 (0.85/0.93) |
| family gift in | 0.67 (0.67/0.89) | 0.67 (0.67/0.85) | 0.70 (0.70/0.89) |
| gift out | 0.56 (0.78/0.74) | 0.56 (0.78/0.70) | 0.56 (0.78/0.70) |
| insurance premium | 0.52 (0.85/0.67) | 0.48 (0.70/0.67) | 0.52 (0.85/0.67) |
| insurance payout | 0.52 (0.78/0.74) | 0.48 (0.78/0.70) | 0.48 (0.78/0.70) |
| loan installment | 0.52 (0.74/0.74) | 0.48 (0.67/0.63) | **0.63** (0.78/0.85) |
| windfall | 0.74 (0.89/0.85) | 0.63 (0.89/0.74) | 0.74 (0.89/0.85) |
| title/name span | 0.70 (0.89/0.81) | 0.52 (0.89/0.59) | 0.63 (0.89/0.74) |

- **Title/name:** recovers most of what 3×768 lost (span exact 0.59 → 0.74 vs B's 0.81), about
  one note on 2 of 3 seeds below B.
- **Borrow/lend direction:** the one concentrated loss. Lender-first falls 0.78 → 0.59 on type
  alone (spans unchanged at 0.93); probe borrow→lend errors rise 1.3 → 3.0 per seed.
  Subject-first is −0.11 (vs −0.30 for 3×768). On test, direction errors don't increase (no
  lend/borrow confusions in D).
- **Gift direction:** unchanged or slightly better. **Insurance payout:** one fragment note
  (`bảo việt`) as in C. **Loan installment:** improves (+0.11).

## 7. Slices

| type acc / span F1 / span exact | B | C | D |
|---|---|---|---|
| test accented (63) | 0.94 / 0.90 / 0.94 | 0.95 / 0.73 / 0.83 | 0.95 / 0.85 / 0.90 |
| test unaccented (42) | 0.92 / 0.75 / 0.86 | 0.90 / 0.85 / 0.90 | 0.91 / 0.82 / 0.89 |
| probe accented (49) | 0.84 / 0.80 / 0.84 | 0.78 / 0.70 / 0.76 | 0.82 / 0.77 / 0.82 |
| probe unaccented (32) | 0.78 / 0.76 / 0.78 | 0.77 / 0.61 / 0.70 | 0.77 / 0.76 / 0.79 |

**Seen/unseen targets** (span exact; unseen = target string in no training note):

| | B | C | D |
|---|---|---|---|
| test seen (36) | 0.93 | 0.84 | 0.92 |
| test unseen (15) | 0.69 | 0.67 | **0.73** |
| probe seen (41) | 0.82 | 0.71 | 0.80 |
| probe unseen (14) | 0.64 | 0.55 | 0.62 |

## 8. Note-level comparison, D vs B

| | test | probe |
|---|---|---|
| all-seed regressions (B 3/3 → D 0/3) | 0 | 0 |
| all-seed improvements | 0 | 0 |
| majority-correct → majority-wrong | 3 | 3 |
| majority-wrong → majority-correct | 2 | 2 |

- Regressions (6):
  - type: `tra lai chi Mai 2tr` → expense; `Tuấn cho vay 5 triệu…` → lend (lender-first).
  - span: `nuoc mia 12k` → false span `mia`/`ia`; `anh Duc muon 2tr` → `anh` (title only);
    two `bảo việt` notes → `ảo`/`việt` fragments on some seeds.
- Improvements (4): `spotify refund 59k`, `co Thuy cho muon 1tr` (unseen name), and two
  loan-installment notes (`tra no vay sinh vien`, `tra goc lai khoan vay mb` → `mb` on 2 seeds).
- **No cluster large enough to call systematic at note level.** Changes are spread across
  span (4) and type (2), at most 3 notes per split, and none on all seeds. The lender-first loss
  is visible at pattern level (5 of 27 seed-notes) but only one note flips by majority.
- For contrast, D vs C: probe 9 notes improve by majority and 2 regress; test 7 vs 6.
  Keeping depth recovers most of C's losses.

## 9. Size, latency, quantization (seed 1; CPU, batch 1, 1 thread, this Mac)

| | B 4×768 | C 3×768 | **D 4×768 FFN 2048** |
|---|---|---|---|
| parameters | 34,791,947 | 27,704,075 | 28,496,395 |
| FP32 checkpoint | 139.18 MB | 110.82 MB | 113.99 MB |
| ONNX FP32 | 139.21 MB | 110.85 MB | 114.02 MB |
| **ONNX INT8** | 34.97 MB | 27.84 MB | **28.66 MB** |
| ONNX FP32 parity (max abs) | passed | 2.1e-5 | 2.8e-5 |
| INT8 p50 / p95 | 1.57 / 1.95 ms | 1.08 / 1.34 ms | 1.37 / 1.71 ms |
| INT8 vs FP32, 50 notes (type / tag) | 1.000 / 1.000 | 0.980 / 0.994 | 0.960 / 1.000 |
| INT8 − FP32 test joint | 0 | −0.029 | **0** |
| INT8 − FP32 probe joint | 0 | +0.037 | **−0.025** |

Measured INT8 matches the estimate within 0.02 MB. INT8 on the seed-1 D leaves test unchanged
and costs 2 probe joint notes (type accuracy unchanged, span F1 0.782 → 0.764), so the deployed
probe joint for that seed is 0.580 vs B's 0.667.

## 10. Decision

| criterion | result |
|---|---|
| INT8 ≤ 30 MB | **met**, 28.66 MB |
| test close to compression-v1 | **met**: joint 0.844 vs 0.854 (within B's seed std), span exact 0.895 vs 0.905 |
| probe close to compression-v1 | small drop: 0.626 vs 0.654 (−2.3 notes of 81), type rather than span |
| span quality substantially better than 3×768 | **met**: probe span F1 0.767 vs 0.663, wrong spans 9.3 vs 15.7 |
| no new systematic regression | mostly: no note regresses on all seeds, but lender-first direction drops 0.78 → 0.59 and INT8 adds −0.025 probe on seed 1 |

This is the "meets size with a very small quality drop" case. The span behaviour is preserved,
which was the question. The remaining loss is borrow/lend direction on probe, the behaviour
targeted-v2 was built to fix, and it is not yet clear whether it is width-driven or noise.
Since that's the risk, adopting outright (A) would be premature, and the drop is too small and
too unconcentrated at note level to justify rejecting FFN pruning (B).

**Suggested single check (not run):** the same policy at **K = 2304** (est. 30.2 MB, ~0.2 MB
over target, which is operationally negligible). If it holds probe direction at B's level,
adopt it; if it shows the same lender-first drop, the loss is recipe noise and 2048 can be
adopted.

## Conclusion

**C. FFN pruning is promising but needs one smaller/larger width check.**

## Artifacts

- `protocol.json`, `comparison.json` (budget, estimates, criteria, metrics, per-seed values,
  paired deltas, patterns, slices, direction counts, note flips vs B and vs C, size,
  quantization).
- `ffn/analysis.json`; neuron map `models/compression-v3/ffn/combined-K2048/ffn_map.json`.
- `runs/supervised/student-4x768-pretrained-pos32-vocab-B-rank-8000-ffn-combined-K2048/seed{1,2,3}/`,
  `train-seed{1,2,3}.log`, `probe-eval.json`, `export-ffn2048-seed1.json`, `onnx-eval.json`.
- Models: `models/compression-v3/supervised/…/seed{1,2,3}/`,
  `models/compression-v3-onnx/pos32-vocabB8000-ffn2048-4x768-seed1/`.

```sh
uv run python scripts/prune_ffn.py --criterion combined --k 2048
uv run python scripts/train_student.py --arm supervised --student student-4x768 \
  --init pretrained --truncate-positions --vocab-spec models/compression-v1/vocab/B-rank-8000 \
  --ffn-map models/compression-v3/ffn/combined-K2048 --lr 5e-5 --head-lr 1e-3 --epochs 40 \
  --out-dir experiments/compression-v3/runs --weights-dir models/compression-v3 \
  --seed 1 --seed 2 --seed 3
```
