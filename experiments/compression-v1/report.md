# compression-v1: position truncation and vocabulary pruning of the pretrained 4×768 student

**Question.** What is the safest way to get the pretrained 4×768 student from 45.9 MB INT8
toward ≤30 MB without losing pretrained knowledge?

**Answer.** Position truncation is exactly lossless but small (−1.6 MB INT8). A conservative
vocabulary pruning (20,481 → 8,338 tokens, byte fallback kept) cuts another 9.3 MB with no
measurable quality loss after fine-tuning. The model is now **35.0 MB INT8**, still above target.
Removing one layer is the next step (est. ~27.9 MB). Conclusion **B**.

## 1. Baseline

distillation-v2 arm A (pretrained supervised 4×768), all 3 seeds:
`models/distillation-v2/supervised/student-4x768-pretrained/seed{1,2,3}`, model sha256 `ecce8554…`
/ `8a6cfec5…` / `ecee6166…`, tokenizer sha256 `746cc4a5…`. Same tokenizer, heads, max_length 32,
frozen test (105) and probe-v1 (81). Test joint 0.844±0.031, probe joint 0.667±0.000.

## 2. Position truncation (no retraining)

- **RoBERTa indexing** (transformers 5.17.0, `create_position_ids_from_input_ids`): real tokens get
  `pad_id + cumsum(mask)`, pads get `pad_id` (1). `<s>`/`</s>` count toward the 32 tokens, so
  the largest index is 33. Rows needed: **34** (`max_length + pad_id + 1`). Row 0 is never
  indexed but kept for alignment; row 1 is the pad row.
- Rows **2,050 → 34**; parameters 45,666,059 → 44,117,771 (−1,548,288).
- Verified on all 909 notes (train + test + probe) for all 3 seeds: tokenization identical,
  type and tag logits **bitwise equal** (max abs diff 0), predictions identical.
- ONNX FP32 and INT8 export work; truncated vs original ONNX logits are bitwise equal and INT8
  predictions identical on test and probe.
- Saved: **6.2 MB FP32, 1.56 MB INT8** (45.86 → 44.29 MB).

Code: `src/gidi/compression/positions.py`, `scripts/compress_positions.py`; reports in
`positions/`.

## 3. Vocabulary analysis

Full numbers in `vocab/analysis.json`.

- **Usage:** train uses 1,071 distinct ids, test 400, probe 313; union 1,153. **94.4% of the
  20,481 tokens are never used.** 387 of train's tokens appear exactly once.
- **Train-only coverage is too narrow:** 43/105 test notes and 27/81 probe notes contain a token
  never seen in train, mostly name/merchant pieces (` Bình`, ` Phương`, `arb`, `combank`) and
  unaccented fragments. So pruning to observed tokens alone would break generalization.
- **Composition:** 7,655 Vietnamese-diacritic tokens, 9,530 ASCII-letter, 1,498 numeric, 1,047
  ASCII+punctuation (LaTeX/code-like, never used), 172 partial-byte, 58 non-Latin, 75 other
  symbols. Non-Latin, partial-byte and symbol tokens have zero uses in all three splits.
- **Byte fallback:** byte-level BPE with 210 single-byte base tokens. 46 byte values (control
  chars, 0xC0/0xC1, 0xF2–0xFF) have no token and already map to `<unk>` in BamiBERT; pruning
  doesn't change that. No orphan tokens: every merged token comes from exactly one merge, and
  token id order equals merge rank (a pretraining-frequency proxy).
- **Accented vs unaccented:** accented notes 1.40 tokens/word, 31% of words split; unaccented
  1.59 tokens/word, 51% split. Unaccented Vietnamese already leans on smaller pieces.
- **Numbers/finance shorthand** is already segmented into small, frequent pieces: `1tr2` →
  ` 1|tr|2`, `89,000` → ` 89|,|000`, `412k` → ` 4|12|k`, `2 triệu` → ` 2| triệu`.
- **Targets:** counterparties average 2.0 pieces (persons 1.5, merchants/brands 2.5); unseen
  merchants such as `shopeefood`, `coopmart` take 3–5 pieces. 23/51 test targets and 22/55 probe
  targets never occur as strings in train.

## 4. Candidate policies

Every policy keeps the 5 specials, all 210 byte tokens (so any text still tokenizes, never newly
`<unk>`), every token seen in the 723 training notes, and is **closed under BPE merge parents**,
so a word whose original segmentation uses only kept tokens tokenizes identically; other words
fall back to smaller kept pieces, ultimately bytes. Built only from BamiBERT's vocab/merges,
token categories and the training notes; test/probe were measured, not used.

"Vietnamese-compatible" = Latin incl. Vietnamese diacritics, digits, ASCII punctuation, and
partial-byte prefixes of Vietnamese characters.

| policy | rule | rows | est. INT8 (with pos34) | test / probe notes unchanged | zero-training joint test / probe |
|---|---|---|---|---|---|
| (baseline) | full | 20,481 | 44.29 | 100 / 100% | 0.844 / 0.667 |
| A-conservative | all Vietnamese-compatible | 20,240 | 44.12 | 100 / 100% | 0.844 / 0.667 |
| B-rank-12000 | + VN-compatible rank < 12,000 | 12,215 | 37.96 | 97.1 / 96.3% | 0.848 / 0.658 |
| B-rank-10000 | rank < 10,000 | 10,279 | 36.47 | 95.2 / 96.3% | 0.848 / 0.658 |
| **B-rank-8000** | rank < 8,000 | **8,338** | **34.98** | 95.2 / 95.1% | 0.848 / 0.658 |
| B-rank-6000 | rank < 6,000 | 6,428 | 33.52 | 90.5 / 90.1% | 0.838 / 0.654 |
| B-rank-4000 | rank < 4,000 | 4,565 | 32.08 | 87.6 / 90.1% | 0.825 / 0.654 |
| C-aggressive | top 2,000 by rank | 2,774 | 30.71 | 79.0 / 82.7% | 0.816 / 0.638 |
| T-only (reference) | train tokens only | 1,545 | 29.77 | 68.6 / 71.6% | 0.775 / 0.617 |

Embedding params = rows × 768. "Zero-training" applies the pruning to the trained baseline seeds
with no retraining: on any note whose tokenization is unchanged the pruned model is bitwise the
original, so all differences come from OOV fallback. No policy increases truncation at 32
(maximum pruned length 21).

**OOV behaviour.** Words outside the kept set are segmented into smaller kept tokens (BPE
re-runs with the surviving merges), down to single bytes. Nothing becomes `<unk>` that was not
already `<unk>`. The cost is that a pretrained whole-word embedding is replaced by pieces the
model must compose. On a 40-string hand-written OOV list (names, merchants, shorthand; not from
test/probe), B-rank-8000 changes 8: `Highlands` → ` H|igh|lands`, `Grab` → ` Gr|ab`, `Lazada`,
`The Coffee House`, `Xanh SM`, `highlands coffee`, `3 triệu rưỡi` (` r|ưỡi`), `1tỷ2` (`t|ỷ`).
Vietnamese person names in the list (`Phúc Long`, …) are unchanged. C-aggressive changes 23/40.

## 5. Chosen candidate: B-rank-8000

- **Why:** the largest cut that still keeps every pretraining-frequent Vietnamese-compatible token
  (rank < 8,000) instead of relying on training coverage. Below rank 6,000 the zero-training loss
  and OOV changes grow; A-conservative is lossless but saves only 0.2 MB.
- **Tokenizer rebuild** (`models/compression-v1/vocab/B-rank-8000/`, deterministic): vocab
  renumbered contiguously in old-id order; 8,123 merges kept (relative rank preserved);
  normalizer, pre-tokenizer, decoder and post-processor unchanged except remapped special ids.
- **Specials:** `<s>`=0, `<pad>`=1, `</s>`=2, `<unk>`=3 keep their ids; `<mask>` 20480 → 8337.
- **Mapping:** `vocab_map.json` (sha256 `da3f4357…`) lists `kept_old_ids` (new id i = old id
  `kept_old_ids[i]`); 12,143 tokens removed, by category: ASCII letters 3,069 kept of 9,530,
  Vietnamese 4,023/7,655, numeric 553/1,498, non-Latin 0/58, partial-byte 67/172.
- **Embedding copy:** all 8,338 kept rows copied exactly from the pretrained BamiBERT rows; none
  reinitialized. Training ids are remapped from the teacher-target cache and re-verified against
  the pruned tokenizer (identical for all 723 notes).

Code: `src/gidi/compression/vocab.py`, `scripts/analyze_vocab.py`, `scripts/prune_vocab.py`.

## 6. Training and results

Same pretrained init as distillation-v2 (BamiBERT layers 3/6/9/12 + fresh seeded heads), then
positions truncated to 34 and vocab pruned to 8,338, then the distillation-v2 supervised recipe
unchanged (lr 5e-5, head lr 1e-3, 40 epochs, batch 8, last epoch kept), seeds 1–3, no KD.
Declared in `protocol.json` before training. All seeds fit (in-sample type F1 and span F1 1.000).
A position-only arm was not retrained: truncation is bitwise lossless and rows ≥ 34 are never
indexed.

| mean ± std, 3 seeds | baseline test | pruned test | baseline probe | pruned probe |
|---|---|---|---|---|
| type accuracy | 0.927±0.005 | 0.933±0.010 | 0.811±0.031 | 0.819±0.038 |
| type macro-F1 | 0.928±0.007 | 0.935±0.009 | 0.795±0.034 | 0.813±0.036 |
| span F1 | 0.830±0.053 | 0.842±0.031 | 0.792±0.038 | 0.783±0.034 |
| span exact | 0.898±0.027 | 0.905±0.016 | 0.831±0.026 | 0.815±0.021 |
| **type+span joint** | **0.844±0.031** | **0.854±0.031** | **0.667±0.000** | **0.654±0.012** |
| false span on null gold | 2.7 | 2.7 | 2.3 | 2.7 |
| missed span | 1.0 | 1.0 | 2.0 | 3.7 |
| wrong span | 7.0 | 6.3 | 9.3 | 8.7 |
| subword-fragment spans | 4.3 | 4.0 | 6.3 | 6.3 |

Paired by seed (pruned − baseline): test joint +0.010 / 0 / +0.019; probe joint 0 / −0.025 /
−0.012 (0–2 notes of 81). No note flips between all-seeds-correct and all-seeds-wrong in
either direction, on test or probe.

**Probe patterns** (joint success, mean of 3 seeds): identical on 8 of 9 patterns
(lender-first 0.78, subject-first 0.89, family gift in 0.67, gift out 0.56, insurance premium
0.52, windfall 0.74, loan installment 0.52, **title/name 0.70**). Insurance payout 0.63 → 0.52
(one note on two seeds).

**Slices** (type acc / span F1 / span exact):

| | baseline | pruned |
|---|---|---|
| test accented (63) | 0.94 / 0.89 / 0.93 | 0.94 / 0.90 / 0.94 |
| test unaccented (42) | 0.91 / 0.73 / 0.85 | 0.92 / 0.75 / 0.86 |
| probe accented (49) | 0.84 / 0.81 / 0.86 | 0.84 / 0.80 / 0.84 |
| probe unaccented (32) | 0.77 / 0.76 / 0.79 | 0.78 / 0.76 / 0.78 |

**Unseen targets** (target string absent from every training note): test span exact 0.67 →
0.69 (15 targets); probe 0.71 → 0.64 (14 targets, = one target on each seed). Seen targets
unchanged (test 0.92 → 0.93, probe 0.82 → 0.82). Title/name boundaries and subword-fragment
counts are unchanged.

## 7. Size and latency (seed 1; CPU, batch 1, 1 thread, this Mac)

| | original 4×768 | pos-truncated | pos + vocab B-8000 |
|---|---|---|---|
| parameters | 45,666,059 | 44,117,771 | 34,791,947 |
| FP32 checkpoint | 182.7 MB | 176.5 MB | 139.2 MB |
| ONNX FP32 | 182.7 MB | 176.5 MB | 139.2 MB |
| **ONNX INT8** | **45.86 MB** | **44.29 MB** | **34.97 MB** |
| INT8 p50 / p95 | 1.50 / 1.85 ms | 1.41 / 1.75 ms | 1.57 / 1.95 ms |
| ONNX FP32 p50 / p95 | 2.73 / 3.08 ms | 2.48 / 2.99 ms | 2.90 / 3.29 ms |
| INT8 − FP32, test joint | +0.019 | +0.019 | 0 |
| INT8 − FP32, probe joint | −0.012 | −0.012 | 0 |

Latency differences are run-to-run noise (~0.1 ms): pruning shrinks only the embedding lookup
table, not compute. INT8 vs FP32 agreement for the pruned model is 1.000 type / 1.000 tag on 50
notes; on full test it changes nothing, on probe type accuracy +1 note and span F1 +0.014 with
joint unchanged.

## 8. Depth

35.0 MB is above 30 MB, so depth is needed. Each 768-wide layer is 7,087,872 parameters
(≈7.1 MB INT8). Same positions and vocabulary:

| | est. INT8 |
|---|---|
| 4×768 (measured) | 35.0 MB |
| 3×768 | ~27.9 MB |
| 2×768 | ~20.8 MB |

Vocabulary alone cannot close the gap safely: B-rank-4000 is still ~32.1 MB, and ≤30 MB needs
~1,850 rows, close to the training-only vocabulary (zero-training test joint 0.775). Full
vocabulary with 3 layers would be ~37 MB, so depth reduction should keep B-rank-8000.

**Recommendation:** test a **3×768** pretrained student next (pos 34, B-rank-8000), copying
evenly spaced BamiBERT layers under the same rule (`layer_map(12, 3)` = 1-based layers 4, 8, 12).
It lands inside the target with ~2 MB margin and removes a quarter of the depth. 2×768 is the
fallback only if 3×768 holds quality and more margin is wanted.

## 9. Success criteria

- Position truncation behaviorally lossless: **yes** (bitwise).
- Vocab pruning meaningful: **yes**, −9.3 MB INT8 (−21%), 59% of rows removed.
- Test joint: **no regression** (0.844 → 0.854, within seed noise).
- Probe joint: **no material regression** (0.667 → 0.654, 1 note of 81 on average; 8/9
  patterns identical).
- Unseen-target/span behaviour: acceptable; test unseen exact slightly up, probe one target
  down, fragments and title/name boundaries unchanged.
- INT8 toward target: 45.9 → 35.0 MB, still **above** 30 MB.
- Tokenizer coverage not broken: byte fallback intact, every training note and 95% of
  test/probe notes tokenize identically, no new `<unk>`. Known cost: some brand names and rarer
  shorthand (`1tỷ2`, `rưỡi`) now take more pieces.

## Conclusion

**B. Compression helps but model remains above target; test a 3x768 pretrained student next.**

## Artifacts

- `protocol.json`, `comparison.json` (all numbers, per-seed values, policies, OOV list).
- `positions/verify-seed{1,2,3}.json`, `positions/export-*.json`, `positions/onnx-eval.json`.
- `vocab/analysis.json`, `vocab/policies.json`.
- `runs/supervised/student-4x768-pretrained-pos32-vocab-B-rank-8000/seed{1,2,3}/`,
  `probe-eval-pruned.json`, `export-pos32-vocabB8000-seed1.json`, `onnx-eval-pruned.json`.
- Models: `models/compression-v1/positions/…`, `models/compression-v1/vocab/<policy>/`,
  `models/compression-v1/supervised/…`, `models/compression-v1-onnx/…`.

Training command:

```sh
uv run python scripts/train_student.py --arm supervised --student student-4x768 \
  --init pretrained --truncate-positions --vocab-spec models/compression-v1/vocab/B-rank-8000 \
  --lr 5e-5 --head-lr 1e-3 --epochs 40 --out-dir experiments/compression-v1/runs \
  --weights-dir models/compression-v1 --seed 1 --seed 2 --seed 3
```
