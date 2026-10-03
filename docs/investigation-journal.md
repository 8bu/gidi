# Gidi Model Investigation Journal

_Last updated: 2026-10-03_

This is a compact investigation journal for the Gidi finance-note model work. It records the decisions, experiments, and what each experiment taught us. It is intentionally not a full lab report.

**Current status:** `gidi-finance-v2` (`models/gidi-finance-v2/`, §25) is the new **testable candidate**, ready for manual testing in the playground (`uv run python scripts/demo_ui.py`, http://127.0.0.1:8765). It packages the accepted value-span-v7 dual encoder: path A is gidi-finance-v1, unchanged, for type/target; path B is a fine-tuned encoder clone + MLP + CRF for the value span. 58.1 MB INT8 bundle, 3.5 ms p50 CPU. Known limitation: `cho a Nam vay 1 triệu 20/10` → `1 triệu 20/10`. `gidi-finance-v1` stays frozen and intact. Model research is stopped: no v8, compression or distillation.

Detailed metrics live in each `experiments/*/report.md`; this journal keeps only the reasoning trail (question → experiment → result → decision → next direction). History is append-only: later findings are added chronologically, earlier conclusions are not rewritten. Update policy: see `AGENTS.md`.

---

## 0. Goal

Build a small local model that can read short Vietnamese personal-finance notes and output:

1. **transaction type**
   - expense
   - income
   - borrow
   - lend
   - repayment_in
   - repayment_out
   - transfer
   - refund

2. **counterparty target span**
   - e.g. `Nam`, `Shopee`, `Hải`
   - or `null` when there is no external counterparty

Amount extraction stays outside the model.

Long-term deployment target:
- fast local inference
- roughly 10–30 MB INT8 if quality remains acceptable

---

## 1. Annotation-v1

Details: `docs/annotation-v1.md`, `configs/annotation-v1.yaml`.

We first froze the annotation semantics before training.

Important rules that emerged:

- Debt-state notes with no money movement are **skipped**.
  - `còn nợ Hùng 300k` → skipped
  - `mượn Hùng 500k` → borrow
- Paying an existing loan / BNPL / card balance → **repayment_out**.
- Investment / store-of-value purchase → **transfer**.
  - `mua ccq`
  - `mua vàng tích trữ`
- Cashback / rebate / reimbursement → **refund**.
- Items, dishes, activities, bill names are not targets.
- Merchants/services/platforms are targets only when they are the actual counterparty.
- Payment channels are not automatically targets.
- `vs` means companion context, not counterparty.
- Title / kinship words may need to be stripped from proper names, depending on the phrase.
- `uncertain` records require an explanation note.
- Only `complete` records are trainable.

Quet became the human-review UI for corrections and exception queues.

---

## 2. Baseline annotation dataset

Details: `experiments/annotation-v1-final-audit.md`.

The initial 600-note corpus was annotated using a mix of:

- human labels
- AI high-confidence labels
- Quet exception review

The final baseline had:

- 600 total labels
- 577 complete
- 23 skipped
- 0 uncertain
- 0 validation errors

A semantic audit found that the largest remaining weakness was not labeling correctness but **coverage**: some important Vietnamese phrasing patterns appeared too rarely in training.

---

## 3. Probe-v1: diagnostic set

Details: `datasets/probe-v1/README.md`.

Instead of immediately adding more training data, we created a separate **evaluation-only probe set**.

Purpose:

> Test whether suspected linguistic blind spots were real and systematic.

Probe-v1 contains 81 fresh notes across 9 patterns:

1. lender-first `X cho mượn/vay` → borrow
2. subject-first `X mượn/vay` → lend
3. incoming family gifts → income
4. outgoing lì xì / gifts → expense
5. insurance premium → expense
6. insurance payout → refund
7. lottery / windfall income
8. loan installment → repayment_out
9. title + proper-name span boundaries

Important rule:

> Probe-v1 is never training data.

Baseline models performed badly on this intentionally difficult set. This confirmed that the suspected blind spots were genuine.

---

## 4. Encoder benchmark: BamiBERT vs MiniLM

Details: `experiments/encoder-shortlist.md`, `experiments/baseline-v1/report.md`.

We fine-tuned both encoders on the same task and dataset.

Why fine-tune?

A pretrained encoder only turns text into contextual vectors. It does not know Gidi's 8 transaction types or target-span rules. Fine-tuning adapts the encoder to those tasks.

The benchmark showed:

- BamiBERT was slightly better overall
- BamiBERT was noticeably better for span extraction
- MiniLM was faster
- BamiBERT handled unaccented Vietnamese more robustly

The final teacher direction became **BamiBERT**.

Training stability work also taught us:

- gradient clipping fixed collapse
- a 20-epoch cap was needed because several runs were still improving near epoch 10
- early stopping prevented wasting the full budget when convergence happened earlier

---

## 5. Targeted-v2: coverage repair

Details: `experiments/targeted-v2/report.md`.

Probe-v1 showed that the model was learning shortcuts such as:

- keyword `cho` deciding lend/borrow
- `vay` deciding borrow regardless of repayment context
- gift/insurance keywords dominating direction

We created **targeted-02**, a small augmentation set built around minimal semantic contrasts.

Example idea:

- `chị Mai cho mình mượn 2tr` → borrow
- `cho chị Mai mượn 2tr` → lend

The vocabulary is almost identical; word order and semantic direction must decide the type.

Targeted-v2 results were a major improvement:

- frozen test type macro-F1: about **0.862 → 0.930**
- probe type accuracy: about **0.41 → 0.86**
- probe type+span joint: about **0.27 → 0.79**

This was the strongest evidence that the main baseline problem was **training coverage / shortcut learning**, not a fundamentally bad encoder.

---

## 6. Targeted-v3: final augmentation check

Details: `experiments/targeted-v3/report.md`.

We tried one small final batch aimed at:

- loan wording
- insurance payout wording

It improved some type decisions but caused span regressions and much higher seed variance.

Key failure:

- the model became too eager to emit a target span in cases where the gold target should be null

Decision:

> Reject targeted-v3. Keep targeted-v2 as the canonical teacher recipe.

No more augmentation loop.

---

## 7. Final teacher

Details: `experiments/distillation-v1/report.md` (teacher section).

The canonical teacher is now:

- BamiBERT
- targeted-v2 data
- final training set includes the old validation set for teacher training
- test and probe-v1 remain held out

The final teacher is strong on held-out data:

- test type macro-F1 ≈ **0.886**
- test span F1 ≈ **0.940**
- test joint ≈ **0.857**
- probe joint ≈ **0.840**

This teacher is good enough to be the reference model for compression experiments.

---

## 8. Distillation-v1: 4×256 random student

Details: `experiments/distillation-v1/report.md`.

First student:

- 4 transformer layers
- hidden size 256
- BamiBERT tokenizer
- ~8.4M parameters
- random initialization

Two arms:

1. supervised-only
2. knowledge-distilled from the teacher

Result:

- type quality survived surprisingly well
- span quality collapsed
- most probe gains disappeared
- KD did not clearly beat supervised-only

Important observation:

The student reached almost zero training loss, so it could **memorize** the training set.

But it generalized poorly.

That means the main problem was not "the student cannot fit the data".

It lacked the linguistic knowledge that a pretrained encoder already has.

The distilled model was tiny and fast:

- ~8.5 MB INT8
- ~0.40 ms CPU p50

So deployment size was excellent; quality was the blocker.

---

## 9. Distillation-v2: 4×768 pretrained diagnostic

Details: `experiments/distillation-v2/report.md`.

We needed to separate two hypotheses:

1. 4×256 failed because it was too small
2. 4×256 failed because it started from random weights

So we tested a 4×768 student.

Why 768?

BamiBERT itself uses hidden size 768. Keeping the same width lets us copy pretrained weights exactly.

Three arms:

- 4×768 random supervised
- 4×768 pretrained supervised
- 4×768 pretrained + KD

The copied BamiBERT layers were:

- layer 3
- layer 6
- layer 9
- layer 12

Conceptually:

```text
BamiBERT teacher/pretrained encoder
12 × 768
      ↓ keep evenly spaced layers
4 × 768
```

### Result

The answer was clear:

> **Pretraining is the main missing piece.**

Random 4×768 was no better than random 4×256.

Pretrained 4×768 jumped dramatically:

- test joint ≈ **0.844** vs teacher **0.857**
- test span F1 ≈ **0.830** vs teacher **0.940**
- probe joint ≈ **0.667** vs teacher **0.840**
- probe retention ≈ **79%**
- test joint retention ≈ **98.5%**

KD again added no measurable gain over pretrained supervised training.

This means widening a randomly initialized student is not useful.

The student needs a pretrained linguistic starting point.

---

## 10. Current model-size problem

The pretrained 4×768 student is good, but still too large:

- ~45.7M parameters
- ~182.7 MB FP32
- ~45.9 MB INT8
- ~1.42 ms CPU p50

Current target is roughly 10–30 MB INT8.

Parameter composition matters:

- word embeddings: ~34%
- 4 transformer layers: ~62%
- position embeddings: ~3%

Each 768-wide transformer layer costs roughly 7 MB INT8.

---

## 11. Current hypothesis

The best current direction is:

> Keep pretrained BamiBERT initialization and preserve hidden width 768 for now.

Compress in the lowest-risk order:

1. **truncate unused position embeddings**
2. **prune vocabulary carefully**
3. only then reduce depth from 4 layers to 3 or 2

Why not shrink width yet?

Shrinking 768 → 256 destroys exact weight compatibility. Without a separately pretrained narrow BamiBERT-like encoder, we would go back toward the random-init problem that already failed.

---

## 12. Next experiment

Next phase: **compression-v1**.

Goal:

> Reduce the pretrained 4×768 student toward the 10–30 MB deployment target without losing the quality recovered by pretraining.

Start with transformations that preserve pretrained weights:

### Step 1 — position-table truncation

Current BamiBERT position table has ~2,050 rows, but Gidi uses max_length 32.

Keep only the rows actually required by the tokenizer/model.

This should be essentially lossless.

### Step 2 — vocabulary investigation / pruning

The embedding matrix is ~34% of the model.

Investigate which BamiBERT tokens are actually needed by:

- the Gidi training corpus
- realistic Vietnamese finance text
- names / merchants / unaccented text
- tokenizer fallback behavior

Vocabulary pruning is risky because unseen merchants and names are already a measured weakness.

Do not aggressively prune before defining an explicit OOV strategy.

### Step 3 — depth

If the model remains above target after safe pruning:

- compare 4×768
- 3×768
- 2×768

using copied pretrained layers.

Do not change width yet.

---

## 13. Compression-v2: 3×768 pretrained student

Details: `experiments/compression-v2/report.md` (compression-v1 steps: `experiments/compression-v1/report.md`).

Why 3×768: compression-v1 truncated positions (lossless) and pruned the vocabulary to 8,338 tokens with no quality loss, but the 4×768 was still **~35.0 MB INT8**. One 768-wide layer is ~7 MB, so 3×768 was the next step toward ≤30 MB.

Same recipe, vocabulary and positions; only depth changed (BamiBERT layers 4, 8, 12).

Result:

- actual size **27.8 MB INT8**, ~30% faster
- vs compressed 4×768: test joint **0.854 → 0.816**, probe joint **0.654 → 0.576** (lower on every seed)
- the loss is mostly span quality: probe span F1 0.783 → 0.663, truncated names, title-only spans (`dì` for `dì Hương`), subject-first borrow/lend direction
- type accuracy on test barely moves

Decision:

> Keep the compressed 4×768 despite being above the size target. Removing a layer costs exactly the span and direction behaviour that earlier experiments worked to fix.

---

## 14. Compression-v3: FFN pruning of the 4×768

Details: `experiments/compression-v3/report.md`.

Why not depth: 3×768 hit the size target, but removing a layer damaged span boundaries (title-only spans, truncated names) and subject-first direction.

Why FFN: the FFN is 54% of the compressed 4×768. Shrinking each layer's FFN keeps all 4 layers, the attention and the 768 width. Pretrained neurons are selected, not re-initialized: each neuron is scored by its mean activation on training notes × its output weight norm, and the top **2048 of 3072** per layer are copied exactly.

Result:

- actual size **28.66 MB INT8** (vs 34.97 MB)
- test joint **0.854 → 0.844** (within seed noise); span quality kept (probe span F1 0.783 → 0.767, vs 0.663 for 3×768)
- probe joint **0.654 → 0.626**, mostly lender-first borrow/lend direction (0.78 → 0.59); no note regresses on all seeds

Decision:

> FFN pruning is promising but needs one width check (2304, ~30.2 MB) before replacing the compressed 4×768 as the deployment model.

---

## 15. Compression-v4: final FFN-width check (2304)

Details: `experiments/compression-v4/report.md`.

Why: FFN 2048 kept span quality, but probe joint dropped 0.654 → 0.626, mostly lender-first borrow/lend direction. A slightly wider FFN (2304, the next 256 neurons by the same score) tests whether that was a capacity loss.

Result:

- actual size **30.24 MB INT8**
- probe joint 0.638 vs 0.626 (2048) vs 0.654 (3072), about one note; lender-first type 0.70 vs 0.67 vs 0.85
- test joint got worse: 0.822 vs 0.844 vs 0.854, mostly span boundaries
- no note changes on all seeds; the 2048 → 2304 differences look like training noise, not recovered capacity

Decision:

> Adopt FFN 2048 (28.66 MB INT8) as the deployment student. This closes the FFN-width search.

---

## 16. Deployment-v1: from model research to deployment hardening

Details: `experiments/deployment-v1/report.md`, usage `docs/deployment.md`.

Final compression decision: FFN 2048 (compression-v3); 2304 rejected; FFN 3072 kept only as a quality reference.

Final architecture: pretrained BamiBERT-derived 4×768 (layers 2, 5, 8, 11), FFN 2048, 34 position rows, vocabulary B-rank-8000 (8,338 tokens), max_length 32, dynamic INT8 ONNX: **28.66 MB** (bundle 29.29 MB).

Seed 1 was taken by the predeclared export-seed rule. Picking by test/probe would violate "test and probe are evaluation only", and validation is inside training.

Hardening, no training:

- immutable versioned bundle with checksums
- numpy/onnxruntime/tokenizers runtime that keeps original-string offsets
- golden suite from approved train/validation notes
- robustness inputs
- parity: PyTorch = FP32 on every case; INT8 = FP32 on the golden suite
- ~1.2 ms p50 on 1 CPU thread

Known limitation carried forward: lender-first `X cho mượn/vay` → borrow, especially unaccented. The model has no skip class, and its confidences are uncalibrated.

Decision:

> gidi-finance-v1 is ready for application integration. Next phase (not started): integration, then confidence calibration.

---

## 17. Value-span extension: annotation-v2 toward gidi-finance-v2

Details: `docs/annotation-v2.md`, `experiments/value-span-v1/`, `experiments/targeted-value-01-generation.md`.

Question: the product contract needs the exact monetary amount of a note, not only type and counterparty. Can the model also return the amount as a span?

Scope:

- `gidi-finance-v1` stays frozen and unchanged. It predicts only the transaction type and the target/counterparty span. The earlier statement that amount extraction stays outside the model was true for v1 and is not revised.
- `gidi-finance-v2` will add a third BIO head that returns `value_text` / `value_span`: the exact substring of the amount, with original-string offsets.
- Numeric normalization stays outside the model. `100` → 100000, `50k` → 50000, `5 xị` → 500000 are jobs for a future backend normalizer, not model outputs. v2 does not predict VND values.

Why a span task: choosing the amount depends on context. Quantities, dates, months and installment numbers must not be taken as money:

- `ăn 2 tô phở 70` → `70`
- `tiền điện tháng 10 hết 612` → `612`
- `trả góp kỳ 3 1tr5` → `1tr5`
- `20/10 mua quà mẹ 500` → `500`

Annotation-v2 setup:

- v1 type/target semantics unchanged, plus `value_text` and `value_span`.
- The value is labelled in a separate Quet pass, because Quet labels one span per pass.
- A rule-based proposer pre-labels value spans. Of the ~909 v1-labelled records, 782 were auto-accepted as rule labels (not human labels) and 167 go to value review: multi-number, slang, bare-number, competing-amount or no-amount notes, plus an audit sample.
- Coverage gaps: no `5 xị`, `5 chai`, `5 lít`, `50 K`, `500 ngàn` or `1500000`, and almost no bare-number amounts. A targeted synthetic batch, `targeted-value-01` (253 notes), was generated to fill them.

Provisional value conventions:

- `gửi góp tháng 3 triệu` → `3 triệu`
- `2 triệu rưỡi` → one value span
- `đổ 5 lít xăng 120k` → `120k`; here `5 lít` is a quantity
- `thưởng tết 2 tháng lương` → no explicit amount, value span null

Review policy: the first targeted-value-01 plan flagged too many notes for review under strict must-review rules. Current policy is tiered, risk-based review:

- review every genuinely ambiguous note
- add small stratified samples of the hard patterns: multi-number, bare numbers, `xị`, `chai`, `lít`, `củ`, `tỷ` and other hard surfaces
- add a small clean audit sample, for about 75 human-reviewed notes instead of 253
- if a sampled stratum shows errors, review that whole stratum; otherwise the remainder stays auto-accepted with its provenance
- synthetic or auto-accepted records are never presented as human-reviewed

Planned v2 architecture: the v1 deployment architecture unless something blocks it (BamiBERT-derived 4×768, FFN 2048, pos34, B-rank-8000, max_length 32):

```
shared pretrained encoder
  ├─ type head
  ├─ target BIO head
  └─ value BIO head
```

Compression research, KD, width/depth sweeps and tokenizer/vocabulary changes stay closed unless a concrete blocker appears.

Status: annotation-v2 review in progress. No v2 model has been trained.

Next direction, after the tiered review:

1. rebuild and merge annotation-v2 training data
2. rerun the data audit
3. predeclare the training/evaluation protocol
4. train v2 seeds 1–3
5. evaluate value-span quality
6. check that type/target do not materially regress vs frozen v1
7. give a verdict: adopt v2; value works but the regression needs fixing; or value quality is not yet sufficient

---

## 18. Value-span-v1: v2 training result (verdict B)

Details: `experiments/value-span-v1/report.md`, protocol `experiments/value-span-v1/protocol.json` (frozen before training, sha256 `a23ac226…ce28`).

Question: does a third (value BIO) head on the gidi-finance-v1 architecture return the exact amount span without materially regressing type/target quality?

Setup:

- annotation-v2 training data: 895 train (723 v1 rows + 172 targeted-value-01), 149 validation (110 v1 validation are also in train, as in v1), 143 test (105 v1 + 38 targeted-value-01), 81 probe (evaluation only). Most value labels are rule-provenance accepted by the tiered review (46 + 77 human-reviewed notes, 0 escalations).
- Same recipe as compression-v3 (40 epochs, no KD, no sweep, last epoch), seeds 1–3, ~300 s per seed on MPS. Adoption criteria and the regression gate were frozen before training.

Result:

- Value quality passes every criterion on the mean and on seed 1: test value exact 0.9880 mean (0.9784 seed 1), present/null accuracy 1.0, multi_number exact 0.9545 mean; probe value exact 1.0.
- Regression gate fails on one criterion: seed 1 type accuracy on test-v1 drops 4 notes (100/105 → 96/105) against a limit of 3. The mean type drop is exactly 2.0 notes (passes the "more than 2 notes" wording; the rate shorthand `> 0.019` would flag it). Target exact on test-v1 improves (+3 notes mean), probe joint improves (+9 notes mean) and the lender-first probe slice improves (+2.33 notes mean).
- Observed, not tested: the type regressions include user-as-lender direction errors (`ck cho Vũ mượn 700k` lend → borrow in all three seeds) while lender-first borrow notes improve.
- The three multi-number value errors (2 notes) are the same boundary overrun into a trailing date (`1 triệu 20/10`, `300 hom 5/10`); none picked the wrong number.

Decision: verdict **B. value-span works but type/target regressions need fixing**. Per the frozen protocol no bundle was exported; `gidi-finance-v1` is untouched and stays the deployed model; the playground is unchanged (default bundle v1, `uv run python scripts/demo_ui.py`). `models/value-span-v1/seed{1,2,3}` are PyTorch checkpoints only.

Limitations: most value labels are rule-provenance; slices are tiny (bare_number 6, slang 10, only 12 human value labels on test); no `no_amount` on test; `2 triệu rưỡi` absent; v1 validation is inside train.

Next direction: fixing the regression is a new experiment and needs a user decision. Candidates, none started: inspect/rebalance the lend/borrow direction notes in targeted-value-01; review the multi-seed export rule; add date-after-amount examples for the boundary overrun; or leave v2 as a research artifact.

---

## 19. Value-span-v2-direction-repair: training-data direction rebalance (verdict B)

Details: `experiments/value-span-v2-direction-repair/report.md`, protocol `protocol.json` in the same dir (frozen before training, sha256 `67dbc5c5…9612`).

Question: is the value-span-v1 seed-1 type regression caused by a lend/borrow direction imbalance that targeted-value-01 added to training?

Setup: a training-only audit found balanced classes but an unbalanced name-first family. targeted-value-01 added 8 `X cho mượn` (borrow) rows and 0 `X mượn` (lend) rows, which moved the name-first difference from 5 to 13. The predeclared repair appended 8 verbatim duplicates of existing approved `X mượn` lend rows (difference back to 5; no labels created). Architecture, recipe, seeds, sets and gate are unchanged from value-span-v1; trained once.

Result: value quality still passes, with test value exact 0.9856 on the mean and on seed 1. The regression gate fails on the same criterion: seed 1 type on test-v1 is 96/105 against v1's 100, a drop of 4 where the limit is 3 (the mean drop of 1.33 passes). `ck cho Vũ mượn 700k` is still borrow in all seeds. Seed 1 fixed three of value-span-v1's four seed-1 regressions and broke three unrelated notes. Probe lender-first is 8/7/8, against 8/8/9 for value-span-v1 and 5/6/7 for v1.

Decision: verdict **B**; the hypothesis is not supported. No export, no runtime or playground change, and no further repair started.

Next: needs a user decision. [INFERENCE] The remaining errors look like seed-level type variance on the 105-note set rather than a count imbalance.

---

## 20. Value-span-v3-frozen-v1: value head on the frozen v1 model (verdict B)

Details: `experiments/value-span-v3-frozen-v1/report.md`, protocol `protocol.json` in the same dir (frozen before training).

Question: can the trained gidi-finance-v1 encoder representations support value-span extraction if the whole v1 model is frozen and only a new value BIO head (2,307 params) is trained?

Setup: the base was the trained v1 seed-1 checkpoint, with its encoder, type head and target head frozen. Checks confirmed the optimizer held only value-head parameters, the frozen parameters received no gradient, and the frozen-weight hash was unchanged. Training used the value BIO loss only (AdamW lr 1e-3, 40 epochs, eval mode) on the unchanged `training-v1` data, with seeds 1–3 for the head only.

Result: type and target outputs are bit-identical to v1 on test and probe (0 mismatches, logit difference 0). Value quality fails badly: test value exact 0.765 (threshold 0.95), human-only 0.417, multi_number 0.606, slang slice 0.10; present/null 0.971 passes. Training loss plateaued near 0.19 in every seed. The errors pick period or index numbers (`quý 3 3tr2` → `3`), split unit tokens (`540k` → `k`) or run past the amount.

Tracked failure (held-out note, reported only, never used for training): the trailing-date overrun on `cho a Nam vay 1 triệu 20/10` (`1 triệu 20/10` in value-span-v1 and the direction repair) does not recur here; all seeds predict `1 triệu`. The same family is still present on `Thắng vay 5 củ, hẹn t10 trả` → `5 củ, hẹn t10` (all seeds); it is recorded as a known limitation. Over 3 seeds, this model's test+probe value errors are mostly wrong-number selection (33, often a month/quarter/instalment number) and truncation (73); only 3 are right-side date/time absorption.

Decision: verdict **B**; no export, no runtime or playground change, and no further experiment started.

Next: needs a user decision. [INFERENCE] The value quality of value-span-v1 depended on adapting the encoder, and the v1 token features are not linearly sufficient for value spans.

---

## 21. Value-span-v4-nonlinear-head: MLP value head on the frozen v1 model (verdict B)

Details: `experiments/value-span-v4-nonlinear-head/report.md`, protocol `protocol.json` in the same dir (frozen before training).

Question: is the frozen v1 representation sufficient for value spans if it is read out by a small nonlinear head?

Setup: the same frozen v1 seed-1 base and data as §20. The only trainable part is a value head `Linear(768,256) → GELU → Dropout(0.1) → Linear(256,3)` (197,635 params). Training used the value BIO loss only, with the same head-level recipe as v3 (AdamW lr 1e-3, 40 epochs) and seeds 1–3 for the head only. The freezing checks passed.

Result:

- Type and target outputs are bit-identical to v1 on test and probe (0 mismatches, logit difference 0).
- Value quality improves over the linear head but still fails: test value exact 0.863 (threshold 0.95; v3 0.765, value-span-v1 0.988), human-only 0.75, multi_number 0.727, bare_number 0.50. Present/null 1.0 passes. Token F1 is 0.961.
- The errors are mostly boundary truncation: 15 per seed on test (`3 triệu` → `3`, `230k` → `2`). Wrong-number selection is 1 per seed; there are no date/time overruns and no missed values.
- Tracked held-out notes: `cho a Nam vay 1 triệu 20/10` → `1 triệu` (exact, all seeds). `Thắng vay 5 củ, hẹn t10 trả` → `5 củ,` in all seeds: the date is no longer absorbed, but a punctuation overrun remains.

Decision: verdict **B**; no export, no runtime or playground change, and no deeper head, sequence decoder or unfreezing started.

Next: needs a user decision. [INFERENCE] The frozen v1 features locate the amount, but a token-wise readout does not produce exact multi-piece boundaries.

---

## 22. Value-span-v5-crf: MLP emissions + linear-chain CRF on the frozen v1 model (verdict B)

Details: `experiments/value-span-v5-crf/report.md`, protocol `protocol.json` in the same dir (frozen before training).

Question: does CRF sequence decoding fix the v4 boundary truncation while type/target stay exactly v1?

Setup: the same frozen base, data and head recipe as §21. The value path is the v4 MLP emissions plus a learned linear-chain CRF (15 params) with no hard BIO constraints. It was trained on CRF NLL only, with uncertain notes excluded, and decoded with Viterbi. Seeds 1–3 vary only the head initialisation and batch order. The freezing checks passed.

An evaluation plumbing bug (value metrics decoded by argmax) was fixed before the verdict, and the evaluation was re-run. The protocol and checkpoints were unchanged.

Result:

- Type and target outputs are bit-identical to v1.
- Boundary truncation roughly halves: test 15 → 7–8 per seed, probe 6–7 → 2. Unit-dropping errors (`3 triệu` → `3`, `700 ngàn` → `700`, `2tr` → `tr`) are fixed.
- Test value exact 0.863 → 0.902 and probe 0.889 → 0.951; slang, long and unseen slices improve.
- The gate still fails: human-only 0.75, multi_number 0.682 (down from 0.727), bare_number 0.50. Present/null 1.0 passes.
- The remaining truncations are emission errors that the CRF does not override (`540k` → `k`, `170` → `0`, `230k` → `2`).
- The span-continuation pull re-introduced date overruns: `cho a Nam vay 1 triệu 20/10` → `1 triệu 20/10` in all seeds (v4 was exact), and `300 hom 5/10` → `300 hom`. `Thắng vay 5 củ, hẹn t10 trả` → `5 củ,` is unchanged.

Decision: verdict **B**; no export, no runtime or playground change, and no BiLSTM, transformer, adapter or unfreezing started.

Next: needs a user decision. [INFERENCE] Sequence decoding fixes about half of the continuity errors, but the frozen v1 token features still under-separate amount pieces from adjacent numbers and dates.

---

## 23. Value-span-v6-value-adapter: value-only Transformer block on the frozen v1 model (verdict B)

Details: `experiments/value-span-v6-value-adapter/report.md`, protocol `protocol.json` in the same dir (frozen before training).

Question: can one small trainable contextual block on the value branch only close most of the gap to value-span-v1 (0.988) while v1 type/target stay exact?

Setup:

- The frozen v1 encoder states feed the frozen type and target heads unchanged.
- The value branch is one post-LN `TransformerEncoderLayer` (768, 12 heads, FFN 1024, GELU, dropout 0.1), then the v5 MLP and CRF. That is 4.14M trainable params, 3.94M of them the adapter.
- Training: CRF NLL only, adapter lr 3e-4, head lr 1e-3, 40 epochs, seeds 1–3 for the value branch only.
- The freezing checks passed: 19 optimizer tensors, no grads on the 73 frozen params, hashes unchanged.

Result:

- Type and target outputs are bit-identical to v1.
- Test value exact 0.887 ± 0.011 (v5 0.902, v1 0.988). Probe 0.975 (v5 0.951).
- Multi-number 0.864 (v5 0.682) and bare number 0.833 (v5 0.50) improve; slang 0.70, unseen 0.79 and long 0.82 get worse. Human-only stays 0.75.
- The adapter fixes v5's leading-digit and partial-number failures (`540k`, `170`, `612`, `9tr5`, `5tr2`), but it truncates unit words mid-subword (`600 ngh`, `5 x`, `45.000` without `vnd`, `700` without `ngàn`, `2tr` → `tr`).
- Amount vs date stays unresolved: `1 triệu 20/10` overruns in all seeds, and `300 hom 5/10` → `5`.
- The training NLL of 0.015–0.022, against 0.09 for v5, suggests overfitting.

Decision: verdict **B**; no export, no playground change, and no further block, BiLSTM, encoder adapter, unfreezing, augmentation or loss weighting started.

Next: needs a user decision. The report lists the options, none of them started. [INFERENCE] Frozen-v1 value extraction plateaus around 0.89–0.90 test exact. Reaching value-span-v1 quality likely needs encoder adaptation under a type/target preservation constraint, or more value-labelled data.

---

## 24. Value-span-v7-dual-encoder: frozen v1 path + isolated trainable value encoder (verdict A)

Details: `experiments/value-span-v7-dual-encoder/report.md`, protocol `protocol.json` in the same dir (frozen before training).

Question: is full encoder adaptation what the value task needs, and can it be isolated from the trusted v1 type/target path?

Setup:

- Path A is the frozen gidi-finance-v1 encoder with its type and target heads.
- Path B is a deep copy of the trained v1 encoder, then the v5 MLP and CRF. It is fully trainable on the same tokenized input and never reads path A's states.
- Training: value-span-v1's recipe (encoder lr 5e-5, head lr 1e-3, batch 8, 40 epochs) with CRF NLL only, and no type or target loss. Seeds 1–3 vary the head initialisation, batch order and dropout.
- Trainable: 28.7M of 57.2M params. The freezing and clone checks passed.

Result:

- Type and target outputs are bit-identical to v1.
- Test value exact 0.988 ± 0.004 (value-span-v1 0.988); human-only 0.917; multi-number 0.955; every slice ≥ 0.955; probe 1.000; test-targeted 1.000. All criteria pass on the mean and on seed 1.
- Errors: `cho a Nam vay 1 triệu 20/10` → `1 triệu 20/10` (all seeds, tracked held-out case) and `230k` → `30k` (seeds 1 and 2). All v5/v6 failure classes are gone.
- Cost: 2.01× params; ≈ 57 MB INT8 (estimated, not exported); CPU latency 7.6 vs 3.8 ms/note (1.98×). Tokenization is shared.

Decision: verdict **A**. Isolated full adaptation recovers value-span-v1 quality without touching type/target, so sharing the encoder was what broke type. Stopped before export; no playground change and no further experiment.

Next: needs a user decision. Candidates, none started: a deployment experiment (export v2 dual INT8, parity and latency gates), compressing the value encoder, or sharing frozen lower layers.

---

## 25. Deployment-v2: gidi-finance-v2 export, runtime, playground (testable candidate)

Details: `experiments/deployment-v2/report.md`, `protocol.json` (frozen before packaging), `verification.json`.

Question: can the accepted v7 dual encoder ship as one versioned runtime without touching v1's type/target behavior?

Setup:

- v7 value-head seed 1, the predeclared export seed, exported as a single ONNX graph containing both encoders on shared inputs, then dynamic INT8 with v1's settings.
- The CRF scores live in `config.json`; Viterbi runs in the numpy runtime.
- Golden suite, held-out parity, 553 hardening checks, performance, and a new shadcn playground.

Result:

- **PyTorch → FP32:** 0 semantic mismatches.
- **INT8 path A:** bit-identical to the v1 bundle.
- **INT8 value:** identical to FP32 on golden/test/probe; test exact 0.9856, and all frozen v7 criteria pass.
- **Size and speed:** 58.1 MB bundle, 3.5 vs 1.6 ms p50. Tokenization runs once.
- **Frozen gate:** two INT8-vs-FP32 items fail as written: golden target 1 note, test/probe type 97.5–98.6%. Every case is inherited v1 INT8 drift, reproduced exactly by v2. The protocol was self-contradictory: it required identity with v1 and also ≥99% agreement with FP32.
- The flaky 20-error suite run did not reproduce in 16 runs. [INFERENCE] It was most likely the old playground server fixture.

Decision: by user decision, the deployed v1 INT8 is the production baseline for the type/target path. The failing items stay recorded as failing against FP32. v2 is marked the testable candidate, and v1 stays intact.

Next: manual testing of v2. No compression, distillation, v8 or new data without a user decision.

---

# Mental model

Think of a transformer like a building.

```text
12 × 768
```

means:

- 12 floors = 12 transformer layers
- width 768 = every token is represented by a 768-dimensional vector on each floor

The first failed student:

```text
4 × 256
```

was:

- much shorter
- much narrower
- and built from scratch

The successful diagnostic:

```text
4 × 768
```

was:

- shorter
- same width
- reused pretrained BamiBERT structure

That last property mattered the most.

---

# Glossary

**Encoder**  
The part of the model that converts text into contextual representations.

**Fine-tuning**  
Continuing training from a pretrained model on Gidi's specific task.

**Pretraining**  
Learning general language structure from a large corpus before Gidi-specific training.

**Teacher**  
The strong model used as a reference or knowledge source.

**Student**  
The smaller model we want to deploy.

**KD / Knowledge Distillation**  
Training a student using both human labels and the teacher's probability distributions.

**Soft labels / soft targets**  
Teacher probabilities such as `borrow 0.8, lend 0.15, ...`, instead of only the winning class.

**Random initialization**  
Starting model weights from random numbers instead of pretrained knowledge.

**Layer / depth**  
One transformer processing block. `4×768` has 4 layers.

**Hidden size / width**  
The size of the vector representing each token inside a layer. `4×768` uses 768 dimensions.

**Span**  
The exact substring chosen as the target, such as `Hải` in `vay chú Hải 20tr`.

**BIO tagging**  
Token-level span labels: Begin, Inside, Outside.

**Joint score**  
A record counts as correct only if both transaction type and exact target span are correct.

**Probe set**  
A deliberately difficult evaluation-only dataset used to test known linguistic patterns.

**INT8**  
8-bit quantization used to reduce model size and speed up inference.

**Vocabulary pruning**  
Removing tokenizer vocabulary entries / embedding rows that are not needed.

**OOV**  
Out-of-vocabulary behavior: what happens when new text contains something outside the retained vocabulary.
