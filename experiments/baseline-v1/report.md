# baseline-v1: first encoder baseline on frozen annotation-v1

Goal: find which pretrained encoder gives the most useful teacher signal for Gidi before a tiny
student is designed. Nothing here is tuned for production; no distillation.

## 1. Frozen dataset

- `datasets/annotation-v1/combined/` (queue, labels, provenance, `manifest.json` with sha256 of
  every canonical and source file; verify with `uv run python scripts/build_combined.py --check`).
- 763 records: 716 complete (trainable), 24 uncertain, 23 skipped. Source batches: baseline-01
  600, targeted-annotation-v1-01 163. Human-authored or human-reviewed 206, AI-accepted 557.
- Complete types: expense 216, transfer 131, repayment_out 71, income 69, borrow 61, lend 60,
  refund 55, repayment_in 53. Accented / unaccented (complete): 427 / 289.
- The two targeted re-check records (`cô Út gửi trả 2 triệu`, `Tri gui tra 180k tien xang`) were
  kept as repayment_in by the reviewer; provenance marks them human-reviewed (`recheck-02`).
  Record `…6bba62a2c043` still carries the rejected proposal's note; left as saved (human label).

## 2. Split

`datasets/annotation-v1/splits/`, seed `annotation-v1-split-02`, complete records only. Groups
(kept whole): exact duplicates, mirrors (same token bag), near-duplicates (similarity ≥ 0.9), and
templates (target span and amounts masked). 513 groups, 0 cross-split groups, 0 exact-text or
template overlap. Details: [split-report.md](split-report.md).

| type | train | validation | test |
|---|---|---|---|
| expense | 151 | 33 | 32 |
| income | 47 | 11 | 11 |
| borrow | 42 | 10 | 9 |
| lend | 42 | 9 | 9 |
| repayment_in | 37 | 8 | 8 |
| repayment_out | 50 | 11 | 10 |
| transfer | 93 | 20 | 18 |
| refund | 39 | 8 | 8 |
| **total** | **501** | **110** | **105** |

The largest group (23 `cho X mượn N` lend notes) sits in train, so held-out lend uses other
phrasings.

## 3. Model and recipe

Shared encoder → type head (8 classes) + BIO target head (`O`/`B-TARGET`/`I-TARGET`), loss =
type CE + tag CE (equal weight). Code: `gidi.modeling`, `gidi.training`, `gidi.evaluation`;
run with `scripts/train_baseline.py`.

Final recipe (both encoders): lr 5e-5 encoder, **head lr 1e-3**, **batch 8**, **grad clip 1.0**,
AdamW (wd 0.01), linear schedule with 10% warmup, dropout 0.1, max_length 32, **max 20 epochs**,
early stopping patience 3 on val mean(type macro-F1, span F1), seeds 1–3, MPS.

Deviations from the initial spec, each found necessary during the first sweep:

| change | why |
|---|---|
| type head on the masked mean of tokens, not `<s>` | with `<s>`, type loss stayed at the class prior for 3+ epochs |
| heads at lr 1e-3 | at the encoder lr the BamiBERT tag head stayed all-`O` (span F1 0 in all 6 runs) |
| grad clip 1.0, batch 8 | BamiBERT at 5e-5 collapsed without them |
| 20-epoch cap (was 10) | at 10 epochs 10 of 12 runs peaked at epoch 9–10 |

The 10-epoch sweep (2 lr × 3 seeds × 2 encoders) is kept in `runs/` / [summary.md](summary.md).
Final runs: `runs-e20/`, [summary-e20.md](summary-e20.md). No truncation at max_length 32
(longest note: 16 BamiBERT tokens, 17 MiniLM tokens). MPS is not bit-reproducible.

## 4. Results (3 seeds, mean ± std)

| | BamiBERT | MiniLM |
|---|---|---|
| val type macro-F1 | 0.866 ± 0.018 | 0.873 ± 0.008 |
| val span F1 | **0.860 ± 0.014** | 0.836 ± 0.024 |
| test type accuracy | 0.870 ± 0.020 | 0.863 ± 0.015 |
| test type macro-F1 | **0.862 ± 0.019** | 0.844 ± 0.013 |
| test span F1 | **0.876 ± 0.026** | 0.851 ± 0.022 |
| test span exact match (all) | 0.902 ± 0.031 | 0.883 ± 0.020 |
| exact match, gold target (n=51) | **0.922 ± 0.034** | 0.876 ± 0.023 |
| null accuracy, gold null (n=54) | 0.883 ± 0.077 | 0.889 ± 0.019 |
| best / run epochs | 9/12, 19/20, 10/13 | 15/18, 14/17, 18/20 |

Per-class test F1 (mean over seeds; support in parentheses):

| class | BamiBERT | MiniLM |
|---|---|---|
| expense (32) | 0.882 | 0.916 |
| income (11) | 0.755 | 0.753 |
| borrow (9) | 0.844 | 0.787 |
| lend (9) | 0.842 | 0.825 |
| repayment_in (8) | 0.830 | 0.783 |
| repayment_out (10) | 0.889 | 0.916 |
| transfer (18) | 0.923 | 0.900 |
| refund (8) | 0.933 | 0.872 |

Robustness slices (test, mean over seeds):

| slice | n | BamiBERT type acc / span F1 | MiniLM type acc / span F1 |
|---|---|---|---|
| accented | 63 | 0.899 / 0.863 | 0.873 / 0.812 |
| unaccented | 42 | 0.825 / 0.899 | 0.849 / 0.920 |
| baseline-01 | 80 | 0.871 / 0.853 | 0.879 / 0.786 |
| targeted-01 | 25 | 0.867 / 0.908 | 0.813 / 0.934 |

Per-class classes have 8–11 test examples; one error moves F1 by ~0.05–0.1. Confusion matrices
are in [summary-e20.md](summary-e20.md).

## 5. Failure cases (best seed of each)

Type errors shared by both models (8):

- direction in loan phrasing: `anh Long mượn 3 triệu sửa xe` (lend → borrow),
  `co Thuy cho muon 1tr` (borrow → lend), `vay tiêu dùng trả kỳ này` (repayment_out → borrow);
- gift/family money: `bố cho tiền tiêu 1 triệu` (income), `lì xì cháu 200k` (expense);
- insurance: `dong bao hiem nhan tho` (expense), `bao hiem tra tien kham` (refund);
- rare income: `trúng vé số 200k`.

Span errors:

- title boundaries: both predict `chú Hải` for gold `Hải` in `vay chú Hải 20tr`. The doc keeps
  the title only before birth-order names; the model has not learned that distinction;
- missed multi-word merchants: `bách hoá xanh` (both);
- BamiBERT sometimes stops mid-word: `Th` for `Thao`, `bảo` for `bảo việt`, `Tư` for `dì Tư`.
  Byte-level BPE splits these words into several pieces, and the tag head misses a continuation;
- MiniLM produces more false spans on null-target notes: `grab` in `nạp ví grab`, `agribank` in
  `mở sổ tiết kiệm agribank`, `dong`, `mia`.

## 6. Deployment sanity check

Best checkpoint per encoder (by validation score): BamiBERT seed 2, MiniLM seed 3. Exported with
`scripts/export_onnx.py` (torch dynamo exporter, opset 17, single file, dynamic batch/sequence).
INT8 = onnxruntime dynamic weight quantization. Reports: `export/<model>.json`. Weights are in
`models/` (gitignored).

| | BamiBERT | MiniLM |
|---|---|---|
| parameters | 102.4M | 117.5M (96M of them embeddings) |
| ONNX export | ✅ parity max abs diff 2.8e-5 | ✅ |
| FP32 ONNX size | 410 MB | 470 MB |
| INT8 ONNX size | 103 MB | 118 MB |
| test FP32 ONNX: type macro-F1 / span F1 | 0.879 / 0.865 (= PyTorch) | 0.835 / 0.876 (= PyTorch) |
| test INT8 ONNX: type macro-F1 / span F1 | 0.879 / 0.885 | 0.821 / 0.865 |
| CPU p50 / p95, 1 thread, INT8 | 4.5 / 5.9 ms | 1.8 / 2.3 ms |
| CPU p50 / p95, 1 thread, FP32 ONNX | 8.1 / 8.8 ms | 2.7 / 3.2 ms |
| CPU p50, default threads, INT8 | 2.7 ms | 1.6 ms |

Latency is for batch 1 on Apple Silicon CPU and includes tokenization (≈0.1 ms). Of the models
tested, MiniLM (hidden 384) is about 2.5× faster per note. Both are ~100 MB in INT8, far above
the student target, as expected.

Training cost (MPS, batch 8): BamiBERT ≈ 13.4 s/epoch, 3–4.5 min/run; MiniLM ≈ 15.3 s/epoch,
4.5–5 min/run. The 6-run benchmark took about 25 minutes.

## 7. Comparison and choice of teacher

- **Quality:** BamiBERT leads on target spans (+0.025 test span F1, +0.05 exact match on gold
  targets) and on test type macro-F1 (+0.018). On validation, type F1 is level. It is also better
  on repayment_in (0.83 vs 0.78) and refund (0.93 vs 0.87). These gaps are within about 1–1.5 std
  on 8-example classes, so they are indicative, not decisive.
- **Robustness:** both models are weaker on type for unaccented notes (BamiBERT −0.07, MiniLM
  −0.02) but better on spans there. MiniLM makes more false spans on null notes from baseline-01.
- **Tokenizer and student:** BamiBERT's 20k vocabulary is the right size for a student.
  MiniLM's 250k vocabulary would dominate any student.
- **Licence:** BamiBERT's Qualcomm RAIL terms carry over to derivatives, including a distilled
  student, and need legal review ([encoder-shortlist.md](../encoder-shortlist.md)). MiniLM is MIT.
- **Inference:** as a teacher, speed hardly matters. MiniLM is faster, and its INT8 model loses
  about 0.014 type F1, while BamiBERT's INT8 model loses nothing.

BamiBERT is the stronger teacher signal. MiniLM remains the licence-safe fallback, about 0.02
behind. PhoBERT was not run: its tokenizer is slow and gives no character offsets, so the span
head cannot be aligned without a separate offset implementation.

## 8. Smallest useful next experiment

Error analysis before any modelling change. The remaining errors cluster in three places:
loan direction (`X mượn`, `cho muon`), gift and family money, and the title-boundary rule. First
check each of the 8 shared type errors and 2 shared span errors against the frozen guidelines,
to tell label or rule ambiguity apart from model error. Then run one BamiBERT teacher with the
same recipe, trained on train + validation, to produce soft labels for a first student sized
around BamiBERT's 20k vocabulary.
