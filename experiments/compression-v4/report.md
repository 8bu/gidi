# compression-v4: final FFN-width check (K = 2304)

**Question.** compression-v3 (FFN 2048, 28.66 MB INT8) kept span quality but lost some probe
borrow/lend direction, mostly in lender-first `X cho mượn/vay`. Does a slightly wider FFN
(2304) recover it while staying near 30 MB?

**Answer.** No, not materially. 2304 is **30.24 MB INT8**. Probe joint is 0.638 vs 0.626 (2048)
and 0.654 (3072), about one note. Lender-first type accuracy is 0.70 vs 0.67 and 0.85. Test
gets *worse*: joint 0.822 vs 0.844 and 0.854, lower than 2048 on all three seeds, mostly from
span boundaries. The gap between 2048 and 2304 is a trade inside training noise, not a capacity
recovery. Conclusion **B**: adopt FFN 2048.

## 1. Setup (only the FFN width changed; `protocol.json` written before training)

- Same procedure as compression-v3: `scripts/prune_ffn.py --criterion combined --k 2304` on the
  compression-v1 initial model, scored on the 723 training notes. The scores reproduce
  compression-v3's `analysis.json` exactly.
- Neuron map: `models/compression-v4/ffn/combined-K2304/ffn_map.json` (sha256 `f54ac1d4…`).
  The 2048 kept set is a strict subset of the 2304 set in every layer, so 2304 = 2048 plus the
  next 256 neurons by score.
- Removed-energy proxy: **0.144** (per layer 0.125 / 0.131 / 0.105 / 0.214) vs 0.193 at 2048.
- 30,070,283 params, estimated 30.22 MB INT8. Kept neurons are copied exactly; everything else
  is identical to compression-v3.
- Same recipe (encoder lr 5e-5, head lr 1e-3, batch 8, 40 epochs, warmup 0.1, wd 0.01,
  dropout 0.1, grad norm 1.0), seeds 1–3, no KD. All seeds fit (in-sample F1 1.000).
- Model sha256 (seed 1/2/3): `a5217740…`, `ecfb4fd1…`, `16c8903e…`.

## 2. Held-out results (mean ± std, 3 seeds)

A = compression-v1 (FFN 3072), B = compression-v3 (FFN 2048), C = FFN 2304.

| test | A 3072 | B 2048 | **C 2304** |
|---|---|---|---|
| type accuracy | 0.933±0.010 | 0.933±0.019 | 0.924±0.010 |
| type macro-F1 | 0.935±0.009 | 0.943±0.018 | 0.939±0.008 |
| span F1 | 0.842±0.031 | 0.841±0.034 | **0.794±0.022** |
| span exact | 0.905±0.016 | 0.895±0.010 | 0.876±0.010 |
| type+span joint | 0.854±0.031 | 0.844±0.005 | **0.822±0.011** |
| false span on null gold | 2.7 | 4.0 | 3.7 |
| missed span | 1.0 | 1.3 | 0.7 |
| wrong span | 6.3 | 5.7 | 8.7 |
| subword-fragment spans | 4.0 | 2.7 | 3.7 |

| probe | A 3072 | B 2048 | **C 2304** |
|---|---|---|---|
| type accuracy | 0.819±0.038 | 0.798±0.019 | 0.798±0.014 |
| type macro-F1 | 0.813±0.036 | 0.780±0.020 | 0.775±0.013 |
| span F1 | 0.783±0.034 | 0.767±0.021 | 0.773±0.014 |
| span exact | 0.815±0.021 | 0.807±0.014 | 0.807±0.007 |
| type+span joint | 0.654±0.012 | 0.626±0.019 | **0.638±0.007** |
| false span on null gold | 2.7 | 2.0 | 1.7 |
| missed span | 3.7 | 4.3 | 5.7 |
| wrong span | 8.7 | 9.3 | 8.3 |
| subword-fragment spans | 6.3 | 4.7 | 4.0 |

Paired by seed, joint:

| | test | probe |
|---|---|---|
| C − B (2304 − 2048) | −0.019 / −0.029 / −0.019 | +0.025 / 0 / +0.012 |
| C − A (2304 − 3072) | +0.010 / −0.057 / −0.048 | −0.037 / 0 / −0.012 |
| B − A (2048 − 3072) | +0.029 / −0.029 / −0.029 | −0.062 / 0 / −0.025 |

**Seed variance:** both pruned models are at least as stable as 3072 (test joint std 0.005 /
0.011 vs 0.031; probe 0.019 / 0.007 vs 0.012).

## 3. Probe patterns (joint; type acc / span exact)

| pattern (n=9) | A 3072 | B 2048 | C 2304 |
|---|---|---|---|
| **lender-first cho mượn/vay** | 0.78 (0.85/0.93) | 0.59 (0.67/0.93) | 0.67 (0.70/0.96) |
| subject-first mượn/vay | 0.89 (0.93/0.96) | 0.78 (0.85/0.93) | 0.81 (0.85/0.96) |
| family gift in | 0.67 (0.67/0.89) | 0.70 (0.70/0.89) | 0.67 (0.70/0.89) |
| gift out | 0.56 (0.78/0.74) | 0.56 (0.78/0.70) | 0.56 (0.78/0.67) |
| insurance premium | 0.52 (0.85/0.67) | 0.52 (0.85/0.67) | 0.59 (0.89/0.70) |
| insurance payout | 0.52 (0.78/0.74) | 0.48 (0.78/0.70) | 0.56 (0.74/0.78) |
| loan installment | 0.52 (0.74/0.74) | 0.63 (0.78/0.85) | 0.56 (0.74/0.78) |
| windfall | 0.74 (0.89/0.85) | 0.74 (0.89/0.85) | 0.70 (0.89/0.81) |
| title/name span | 0.70 (0.89/0.81) | 0.63 (0.89/0.74) | 0.63 (0.89/0.70) |

**Lender-first per note** (correct seeds out of 3, A / B / C):

| note | A | B | C |
|---|---|---|---|
| chị Mai cho mượn 2tr đóng tiền nhà | 3 | 3 | 3 |
| ông ngoại cho mượn 10tr sửa mái | 3 | 3 | 3 |
| sếp cho mượn 3tr trước | 3 | 3 | 3 |
| Vinh cho minh vay 800k tien nha | 3 | 3 | 3 |
| Tuấn cho vay 5 triệu, hẹn tháng sau trả | 3 | 1 | 2 |
| được cô Hà cho vay 4 củ | 3 | 2 | 2 |
| anh Quan cho vay 15tr lam von | 1 | 1 | 2 (span) |
| thg Khoa cho muon tam 300k an trua | 1 | 0 | 0 |
| me cho muon 1tr5 tra tien dien | 1 | 0 | 0 |

The pruned-vs-3072 gap is 5 (2048) / 3 (2304) seed-notes out of 27, on four notes. Two of
them (`thg Khoa`, `me cho muon`, both unaccented) were already wrong on 2 of 3 seeds at 3072.
2304 wins back one type seed-note (`Tuấn`) and one span seed-note (`anh Quan`). Probe
borrow→lend errors per seed: 1.3 (3072), 3.0 (2048), 2.7 (2304).

## 4. Slices

| type acc / span F1 / span exact | A 3072 | B 2048 | C 2304 |
|---|---|---|---|
| test accented (63) | 0.94 / 0.90 / 0.94 | 0.95 / 0.85 / 0.90 | 0.94 / 0.79 / 0.87 |
| test unaccented (42) | 0.92 / 0.75 / 0.86 | 0.91 / 0.82 / 0.89 | 0.90 / 0.79 / 0.88 |
| probe accented (49) | 0.84 / 0.80 / 0.84 | 0.82 / 0.77 / 0.82 | 0.81 / 0.77 / 0.81 |
| probe unaccented (32) | 0.78 / 0.76 / 0.78 | 0.77 / 0.76 / 0.79 | 0.78 / 0.79 / 0.80 |

| span exact | A 3072 | B 2048 | C 2304 |
|---|---|---|---|
| test seen targets (36) | 0.93 | 0.92 | 0.88 |
| test unseen (15) | 0.69 | 0.73 | 0.67 |
| probe seen (41) | 0.82 | 0.80 | 0.79 |
| probe unseen (14) | 0.64 | 0.62 | 0.62 |

## 5. Note-level diagnosis

| | 2304 vs 3072 test | probe | 2304 vs 2048 test | probe |
|---|---|---|---|---|
| all-seed regressions | 0 | 0 | 0 | 0 |
| all-seed improvements | 0 | 0 | 0 | 0 |
| majority correct → majority wrong | 4 | 2 | 2 | 3 |
| majority wrong → majority correct | 1 | 3 | 0 | 5 |

- 2304 vs 2048 on test: `vay ngân hàng mua nhà giải ngân 800tr` (span `hàng`/`ngân` on all
  seeds) and `co Thuy cho muon` (`uy`, `co Thuy` on 2 seeds). Both are span-boundary errors.
- 2304 vs 2048 on probe: gains are `Tuấn cho vay` (type), `anh Quan`, `anh Duc`, `phí bảo
  hiểm`, `bảo việt` (spans, each 1 seed). Losses are `được cô Út cho 500k` (→ expense on 2
  seeds), `trung vietlott` (`etlott` on 2 seeds), `tra goc lai khoan vay mb` (missed on 2 seeds).
- Nothing flips on all seeds in any comparison. Every majority-level change is a 1-seed
  margin.

**Answers to the four questions**

1. *Does lender-first direction recover?* Only partly, and not materially: type 0.67 → 0.70
   (one seed-note) vs 0.85 at 3072. The two unaccented notes stay wrong on every seed.
2. *Does span quality stay at compression-v1 level?* On probe, yes (span F1 0.773 vs 0.783).
   On test, no: span F1 0.794 vs 0.842, span exact 0.876 vs 0.905, the weakest of the three.
3. *Are differences stable across seeds?* 2304 vs 2048 trades test (lower on all 3 seeds,
   ~2–3 notes) for probe (higher on 2 seeds, ~1–2 notes). No individual note changes on all
   seeds.
4. *Capacity or noise?* The 2048-vs-2304 differences look like training noise. More capacity
   didn't bring a consistent improvement: test spans got worse while the removed-energy proxy
   improved, and probe moved by about one note. A small, consistent-sign probe gap to 3072
   (−0.016 / −0.028 mean joint, mostly a few lender-first type decisions on notes 3072 itself
   only partly gets) is present at both widths. That gap could be a mild capacity effect, but
   it doesn't shrink meaningfully with 256 more neurons.

## 6. Deployment (seed 1; CPU, batch 1, 1 thread, this Mac)

| | A 3072 | B 2048 | C 2304 |
|---|---|---|---|
| parameters | 34,791,947 | 28,496,395 | 30,070,283 |
| FP32 checkpoint | 139.18 MB | 113.99 MB | 120.29 MB |
| ONNX FP32 | 139.21 MB | 114.02 MB | 120.32 MB |
| **ONNX INT8** | 34.97 MB | **28.66 MB** | 30.24 MB (est. 30.22) |
| INT8 p50 / p95 | 1.57 / 1.95 ms | 1.37 / 1.71 ms | 1.39 / 1.70 ms |
| ONNX FP32 parity (max abs) | passed | 2.8e-5 | 2.3e-5 |
| INT8 vs FP32, 50 notes (type / tag) | 1.000 / 1.000 | 0.960 / 1.000 | 1.000 / 0.998 |
| INT8 − FP32 test joint | 0 | 0 | 0 |
| INT8 − FP32 probe joint | 0 | −0.025 | 0 |

2304 quantizes slightly more cleanly on seed 1 (probe span F1 −0.012, joint unchanged).
2048's INT8 costs 2 probe notes on that seed. Both are seed-1 measurements only, and smaller
than the seed-to-seed spread.

## 7. Decision

- **A (adopt 2304)** needs a material recovery of direction/probe behaviour. That didn't
  happen: lender-first +1 seed-note, probe joint +0.012 (about one note), test joint −0.022 and
  test span F1 −0.047 vs 2048. It is also above 30 MB.
- **C (keep 3072)** needs a *repeatable important* regression in both pruned models. Both show
  a small probe gap (−0.016 to −0.028) concentrated in a few lender-first notes, two of which
  3072 also mostly misses. Test and span quality at 2048 match 3072, no note regresses on all
  seeds, and 2048 is 6.3 MB smaller. Not enough to justify staying at 34.97 MB.
- **B (adopt 2048)**: 2304 doesn't materially improve held-out behaviour, so prefer the smaller
  model. 2048 holds compression-v1's test quality and span behaviour at 28.66 MB, with a known
  small probe cost in lender-first direction.

Known limitation carried into deployment: lender-first `X cho mượn/vay` → borrow is weaker than
in the 3072 model (0.59 vs 0.78 joint on probe), especially unaccented.

## Conclusion

**B. Adopt K=2048.**

The deployment student is
`models/compression-v3/supervised/student-4x768-pretrained-pos32-vocab-B-rank-8000-ffn-combined-K2048/`
(4×768, FFN 2048, pos34, B-rank-8000, 28.66 MB INT8). This was the final FFN-width experiment.

## Artifacts

- `protocol.json`, `comparison.json` (metrics, per-seed values, paired deltas, patterns,
  per-note focus-pattern predictions, slices, direction counts, note flips vs 3072 and 2048,
  size, quantization).
- `ffn/analysis.json`; map `models/compression-v4/ffn/combined-K2304/ffn_map.json`.
- `runs/supervised/student-4x768-pretrained-pos32-vocab-B-rank-8000-ffn-combined-K2304/seed{1,2,3}/`,
  `train-seed{1,2,3}.log`, `probe-eval.json`, `export-ffn2304-seed1.json`, `onnx-eval.json`.
- Models: `models/compression-v4/supervised/…`,
  `models/compression-v4-onnx/pos32-vocabB8000-ffn2304-4x768-seed1/`.

```sh
uv run python scripts/prune_ffn.py --criterion combined --k 2304 \
  --map-root models/compression-v4/ffn --out experiments/compression-v4/ffn/analysis.json
uv run python scripts/train_student.py --arm supervised --student student-4x768 \
  --init pretrained --truncate-positions --vocab-spec models/compression-v1/vocab/B-rank-8000 \
  --ffn-map models/compression-v4/ffn/combined-K2304 --lr 5e-5 --head-lr 1e-3 --epochs 40 \
  --out-dir experiments/compression-v4/runs --weights-dir models/compression-v4 \
  --seed 1 --seed 2 --seed 3
```
