# annotation-v3 retrain v3 of encoder 1: contrast-02 notes for the confused type pairs, 3 seeds + ensemble

**Verdict: REJECT by the declared gate** — no candidate passes it: seed 1 drops frozen-test type
macro-F1 by 0.0288 (limit 0.01) and ends at 0.6923 end-to-end on the debt slice
(limit 0.8); the ensemble drops 0.0152 and ends at 0.7692 on the debt slice.
The type confusions the user listed are largely gone on human-value-01 (type accuracy
0.9252 → 0.9592 seed 1, 0.9592 ensemble; macro-F1 0.8929 → 0.9690 / 0.9709),
but the remaining errors are now mostly **target-span** errors, the debt-slice target exactness fell
(1.0000 → 0.7692 seed 1), and the frozen test did not recover.

Every number below is read from `results.json` (held-out scoring, run once) or from the files
named in each section. Question: does adding LLM-composed notes for the five confused families
(income, shop / app / restaurant expense incl. jewelry as a gift, installment / app-loan
repayment_out, lend on behalf of someone, debt-only notes in both directions) to `training-v2`
fix the type errors that the retrain-v2 encoder still makes on human-value-01, without losing the
rest? Plus: does a 3-model ensemble help?

## 1. contrast-02: composed notes

`datasets/annotation-v3/contrast-02/` (builder `scripts/build_contrast_02.py`, `--check`
reproduces it byte for byte). `source.jsonl` is the hand-written input of the builder (149
candidates composed by the assistant from `configs/annotation-v3.yaml`, `docs/annotation-v1.md`
and `docs/annotation-v3.md`; the five failure families were taken from the user's description of
the confusions only; no frozen-test, probe-v1 or human-value-01 note was read, copied or
paraphrased while writing). Labels are LLM-made with user approval: `provenance.jsonl` marks
every note `annotator: llm` and the manifest says so. Every label is `complete`, type + target +
value under the annotation-v3 rules, exact code-point spans (the builder rejects a span that is not
on a word boundary), and passes `scripts/validate_annotations.py --config
configs/annotation-v3.quet.yaml`: **100 labels, 0 errors**.

| family | type | kept | cues (kept) |
|---|---|---|---|
| income | income 25 | 25 | `allowance` 4, `bonus` 4, `writing-fee` 5, `customer-pays` 4, `salary` 1, `sale` 1, `commission` 1, `gift` 2, `tip` 1, `scholarship` 1, `platform` 1 |
| expense | expense 25 | 25 | `supermarket` 3, `delivery-app` 3, `restaurant` 3, `ecommerce` 3, `electronics` 2, `pharmacy` 2, `clothes` 2, `gift-mother` 1, `gift-wife` 1, `gift-child` 1, `fashion-event` 2, `gift-baby` 1, `gift-sister` 1 |
| repayment_out | repayment_out 20 | 20 | `tra-gop` 6, `vay-app` 5, `bnpl` 2, `finance-co` 3, `bank-loan` 4 |
| lend | lend 15 | 15 | `tra-ho` 4, `mua-ho` 4, `ung-cho` 4, `dong-ho` 3 |
| debt | borrow 6, lend 9 | 15 | `lend-thieu` 3, `lend-chua-tra` 3, `lend-null` 1, `lend-hen` 1, `lend-no` 1, `borrow-thieu` 2, `borrow-chua-tra` 2, `borrow-null` 1, `borrow-no` 1 |

100 kept of 149 candidates. Within a family the quota is filled round-robin over the `cue` tags (source order inside a
cue), so no single phrasing takes it; the jewelry notes (gift, ceremony or fashion; 7 kept) have distinct cues so that they fit. Style: 38% unaccented (38 of 100), 22 null targets, 2 null values, slang
(`ck`, `củ`, `thg`, `t`, `dc`-style), lowercase.

Leakage gate (`build_contrast_02.py`; rules of `scripts/build_human_value_queue.py`: exact,
folded, sequence ≥ 0.9, token Jaccard, char-3-gram ≥ 0.8) against human-value-01, the frozen test,
probe-v1, the existing corpus under `corpus/` and `datasets/`, `training-v2` train and earlier
candidates, **plus** a stricter bound for human-value-01: any candidate with folded char-3-gram
Jaccard ≥ 0.6 to a human-value-01 note is dropped.

| drop reason | candidates |
|---|---|
| near duplicate of the existing corpus | 12 |
| near duplicate of human-value-01 (standard rules) | 0 |
| near duplicate of the frozen test | 1 |
| near duplicate of probe-v1 | 0 |
| near duplicate of `training-v2` train | 0 |
| char-3-gram Jaccard ≥ 0.6 to a human-value-01 note | 2 |
| over the per-family quota (round-robin by cue) | 34 |
| **dropped / candidates** | **49 / 149** |

Largest char-3 Jaccard of a kept note to a human-value-01 note: 0.5278 (bound 0.6). The
15 held-out/corpus drops are listed in `contrast-02/manifest.json` (`leakage.dropped`).

## 2. training-v3

`datasets/annotation-v3/training-v3/` (builder `scripts/build_annotation_v3_training_v3.py`,
`--check` reproduces it): `train.jsonl` = the 905 byte-identical `training-v2` records + the
100 contrast-02 notes (`source_batch: contrast-02`, `provenance.annotator: llm`) = **1005**;
`validation.jsonl` and `test.jsonl` are the same symlinks to the frozen annotation-v1 splits.
Annotators: human 256, ai 546 (distillation-v1 labels), llm 203.

| type | training-v2 | contrast-02 | training-v3 |
|---|---|---|---|
| expense | 217 | 25 | 242 |
| income | 84 | 25 | 109 |
| borrow | 116 | 6 | 122 |
| lend | 111 | 24 | 135 |
| repayment_in | 75 | 0 | 75 |
| repayment_out | 99 | 20 | 119 |
| transfer | 131 | 0 | 131 |
| refund | 72 | 0 | 72 |

Leakage asserted (raises, writes nothing): no train id and no verbatim or NFC-lowercase text in
the frozen test (105), probe-v1 (81) or human-value-01 (150), the 9 debt-01 ids excluded from
human-value-01 absent, no contrast-02 note a near duplicate of any note of those three sets, and
every contrast-02 note below char-3 Jaccard 0.6 to a human-value-01 note (max
0.5278). Hashes of every input and `train.jsonl`: `training-v3/manifest.json`.

## 3. Training and export

Recipe unchanged from retrain-v2 (compression-v3 K2048 supervised student, BamiBERT layers
2/5/8/11, FFN 2048, vocab B-rank-8000, positions 32, 40 epochs, batch 8, encoder lr 5e-5, head lr
1e-3, warmup 0.1, wd 0.01, dropout 0.1, grad-clip 1.0, MPS, last epoch kept, `--retokenize`), no
hyperparameter changed, seeds 1, 2 and 3. Teacher-target cache rebuilt for 1005 notes
(`teacher-targets/`, frozen teacher `lr5e-05-seed1`; the supervised arm does not read the teacher
logits). Runs: `runs/supervised/…K2048/seed{1,2,3}/`, log `train-seeds123.log`. Each run: 5040
steps (126/epoch), final in-sample type/span F1 1.000, train time 339.2 / 333.2 / 325.8 s (about 8.3 s/epoch). Trainer FP32 test type macro-F1 (the
trainer scores the frozen test itself, as in retrain-v2): 0.9276 / 0.9470 / 0.9062; the INT8
scores below are 0.9276 / 0.9542 / 0.9062 (seed 2 differs by one note after quantization).

Export (`scripts/export_onnx.py`, `export-seed{1,2,3}.json`): FP32 parity max abs diff ≤ 2.7e-5; INT8
vs FP32 type agreement on the 50 export notes 1.00 / 0.98 / 1.00, tag agreement 0.994 / 0.996 / 0.996;
INT8 size 28,658,533 B each (identical architecture). INT8 sha256: seed 1 `5068ef2a824d4985…`, seed 2 `e6c046b87b6f6b6a…`, seed 3 `25455205d240eba5…`
(full values in `results.json`).

## 4. Ensemble method and gate (fixed before scoring)

> Fixed before any seed was scored. Soft vote over the three INT8 models of seeds 1, 2, 3 (same tokenizer, so tokens align): type = argmax of the mean of the three softmax type distributions; tag = argmax of the mean of the three per-token softmax tag distributions, and the target span is decoded from those tags by the unchanged decode_first_span; value = the unchanged rule parser gidi.value_parser.parse_value. No weights, no tuning.

The soft vote (mean probabilities) was chosen over a hard majority vote because the logits are
available and a majority vote needs a tie rule and a way to combine spans. A one-member ensemble
was checked equal to the member's own prediction on every in-sample validation note before
scoring, and the evaluator smoke run (in-sample data) ran before the held-out run.

> Fixed before any seed was scored; the rule of retrain-v2 (old = the deployed encoder models/gidi-finance-v1): (a) frozen-test type macro-F1 drop <= 0.01, (b) frozen-test target exact drop <= 0.02, (c) human-value-01 debt-only slice (n=13, encoder + unchanged parser) end-to-end exact >= 0.8. Reported for seed 1 (deployment convention), each seed, the three-seed mean and the ensemble; nothing is selected by it.

The held-out scoring ran once (the script refuses to overwrite `results.json`). The old encoder
(`models/gidi-finance-v1`) and retrain-v2 seed 1 were re-run for per-note comparison; their
test, probe and human-value-01 metrics were asserted equal to the stored `retrain-v2` results to the last
digit before anything was written. The "v2 mean ± sd" columns are the three-seed aggregate stored
in `experiments/annotation-v3-retrain-v2/results.json`.

## 5. Frozen test (105)

| metric | old | v2 seed 1 | v2 mean ± sd | v3 seed 1 | v3 seed 2 | v3 seed 3 | v3 mean ± sd | v3 ensemble |
|---|---|---|---|---|---|---|---|---|
| type accuracy | 0.9429 | 0.9238 | 0.9302 ± 0.0055 | 0.9333 | 0.9524 | 0.9143 | 0.9333 ± 0.0190 | 0.9429 |
| type macro f1 | 0.9563 | 0.9219 | 0.9344 ± 0.0121 | 0.9276 | 0.9542 | 0.9062 | 0.9293 ± 0.0240 | 0.9411 |
| target exact | 0.8857 | 0.8667 | 0.8762 ± 0.0165 | 0.8762 | 0.8571 | 0.9048 | 0.8794 ± 0.0240 | 0.8857 |
| span f1 | 0.8077 | 0.7963 | 0.8076 ± 0.0196 | 0.8000 | 0.8037 | 0.8491 | 0.8176 ± 0.0273 | 0.8113 |

Per-class F1:

| type | old | v2 seed 1 | v2 mean ± sd | v3 seed 1 | v3 seed 2 | v3 seed 3 | v3 mean ± sd | v3 ensemble |
|---|---|---|---|---|---|---|---|---|
| expense | 0.9180 | 0.9355 | 0.9237 ± 0.0205 | 0.9355 | 0.9355 | 0.9180 | 0.9297 ± 0.0101 | 0.9355 |
| income | 0.8571 | 0.8182 | 0.8353 ± 0.0297 | 0.9091 | 0.9091 | 0.9091 | 0.9091 ± 0.0000 | 0.9091 |
| borrow | 1.0000 | 0.8235 | 0.8845 ± 0.0589 | 0.8235 | 0.8889 | 0.7143 | 0.8089 ± 0.0882 | 0.8750 |
| lend | 1.0000 | 0.9412 | 0.9608 ± 0.0340 | 0.8000 | 0.9474 | 0.7826 | 0.8433 ± 0.0905 | 0.8571 |
| repayment_in | 1.0000 | 1.0000 | 1.0000 ± 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 | 1.0000 |
| repayment_out | 0.9524 | 0.9091 | 0.9235 ± 0.0250 | 0.9524 | 0.9524 | 0.9524 | 0.9524 ± 0.0000 | 0.9524 |
| transfer | 0.9231 | 0.9474 | 0.9474 ± 0.0000 | 1.0000 | 1.0000 | 0.9730 | 0.9910 ± 0.0156 | 1.0000 |
| refund | 1.0000 | 1.0000 | 1.0000 ± 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 | 1.0000 |

Where the test errors are (confusion matrices in `results.json`): seed 1 has two expense → lend
notes (the retrain-v2 seed 1 had none, the old encoder none), a borrow ↔ lend pair and a borrow →
repayment_out; the ensemble has the same expense → lend pair and one fewer lend → borrow. income
F1 improves on every seed (0.8571 old → 0.9091), transfer 1.0000 on the ensemble, repayment_in
and refund 1.0000 everywhere; borrow/lend F1 falls (borrow 0.8089 ± 0.0882, lend 0.8433 ± 0.0905
mean) — the old encoder had no borrow/lend error on this test. [INFERENCE] The new lend notes
(`mua hộ`, `trả hộ`, `ứng cho`) and the debt-only notes move the lend/borrow boundary; this was
not isolated by an ablation.

## 6. probe-v1 (81 complete labels)

| metric | old | v2 seed 1 | v2 mean ± sd | v3 seed 1 | v3 seed 2 | v3 seed 3 | v3 mean ± sd | v3 ensemble |
|---|---|---|---|---|---|---|---|---|
| type accuracy | 0.7778 | 0.8025 | 0.8025 ± 0.0123 | 0.8395 | 0.8889 | 0.8642 | 0.8642 ± 0.0247 | 0.8519 |
| type macro f1 | 0.7543 | 0.8136 | 0.8222 ± 0.0215 | 0.8380 | 0.9101 | 0.8925 | 0.8802 ± 0.0376 | 0.8835 |
| target exact | 0.8148 | 0.8025 | 0.8148 ± 0.0123 | 0.8272 | 0.8765 | 0.8395 | 0.8477 ± 0.0257 | 0.8519 |
| span f1 | 0.7818 | 0.7928 | 0.8000 ± 0.0096 | 0.8182 | 0.8519 | 0.8148 | 0.8283 ± 0.0205 | 0.8440 |

Per-class F1:

| type | old | v2 seed 1 | v2 mean ± sd | v3 seed 1 | v3 seed 2 | v3 seed 3 | v3 mean ± sd | v3 ensemble |
|---|---|---|---|---|---|---|---|---|
| expense | 0.8235 | 0.8485 | 0.8573 ± 0.0153 | 0.8485 | 0.9143 | 0.8485 | 0.8704 ± 0.0380 | 0.8485 |
| income | 0.8235 | 0.8889 | 0.8783 ± 0.0183 | 0.8889 | 0.8889 | 0.8571 | 0.8783 ± 0.0183 | 0.8571 |
| borrow | 0.6667 | 0.7692 | 0.7414 ± 0.0275 | 0.8571 | 0.8966 | 0.8571 | 0.8703 ± 0.0228 | 0.8966 |
| lend | 0.8000 | 0.7500 | 0.7609 ± 0.0188 | 0.8182 | 0.9000 | 0.9091 | 0.8758 ± 0.0501 | 0.8571 |
| repayment_in | 0.5000 | 0.8000 | 0.8667 ± 0.1155 | 0.8000 | 1.0000 | 1.0000 | 0.9333 ± 0.1155 | 1.0000 |
| repayment_out | 0.8889 | 0.8889 | 0.8515 ± 0.0337 | 0.9474 | 0.9474 | 0.9524 | 0.9490 ± 0.0029 | 0.9474 |
| transfer | 0.0000 | 0.0000 | 0.0000 ± 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 ± 0.0000 | 0.0000 |
| refund | 0.7778 | 0.7500 | 0.7990 ± 0.0425 | 0.7059 | 0.8235 | 0.8235 | 0.7843 ± 0.0679 | 0.7778 |

probe-v1 is not part of the gate. It improves on every v3 seed over the old encoder (type
macro-F1 0.7543 → mean 0.8802, ensemble 0.8835) and over retrain-v2 (mean 0.8222); `transfer` stays at 0.0 F1
as for every earlier encoder.

## 7. human-value-01 (150 labels; 3 `uncertain` excluded, 147 complete scored)

The old encoder column is `gidi-finance-v1` INT8 + the unchanged rule value parser (called V8-old
in earlier reports; it is the stored value, reproduced exactly). Value spans come from the
unchanged parser for every column.

**All complete notes (n=147)**

| metric | old | v2 seed 1 | v2 mean ± sd | v3 seed 1 | v3 seed 2 | v3 seed 3 | v3 mean ± sd | v3 ensemble |
|---|---|---|---|---|---|---|---|---|
| type accuracy | 0.8435 | 0.9252 | 0.9229 ± 0.0039 | 0.9592 | 0.9524 | 0.9524 | 0.9546 ± 0.0039 | 0.9592 |
| type macro f1 | 0.8049 | 0.8929 | 0.8999 ± 0.0063 | 0.9690 | 0.9521 | 0.9656 | 0.9622 ± 0.0090 | 0.9709 |
| target exact | 0.8367 | 0.8980 | 0.8934 ± 0.0039 | 0.8367 | 0.8639 | 0.8639 | 0.8549 ± 0.0157 | 0.8639 |
| type target exact | 0.7415 | 0.8435 | 0.8413 ± 0.0039 | 0.8231 | 0.8435 | 0.8503 | 0.8390 ± 0.0142 | 0.8571 |
| value exact | 1.0000 | 1.0000 | 1.0000 ± 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 | 1.0000 |
| end to end exact | 0.7415 | 0.8435 | 0.8413 ± 0.0039 | 0.8231 | 0.8435 | 0.8503 | 0.8390 ± 0.0142 | 0.8571 |

Per-class F1 on all complete notes:

| type | old | v2 seed 1 | v3 seed 1 | v3 seed 2 | v3 seed 3 | v3 ensemble |
|---|---|---|---|---|---|---|
| expense | 0.8475 | 0.8772 | 0.9091 | 0.8929 | 0.8679 | 0.8889 |
| income | 0.8500 | 0.9000 | 0.9302 | 0.9091 | 0.9091 | 0.9302 |
| borrow | 0.7500 | 1.0000 | 0.9744 | 1.0000 | 1.0000 | 1.0000 |
| lend | 0.8571 | 0.9474 | 0.9524 | 0.9744 | 0.9756 | 0.9756 |
| repayment_in | 0.4444 | 0.8000 | 1.0000 | 0.8889 | 1.0000 | 1.0000 |
| repayment_out | 0.8750 | 0.8966 | 1.0000 | 0.9655 | 1.0000 | 1.0000 |
| transfer | 0.9577 | 0.9722 | 0.9859 | 0.9859 | 0.9722 | 0.9722 |
| refund | 0.8571 | 0.7500 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Recall by gold type (k/n):

| gold type (n) | old | v2 seed 1 | v3 seed 1 | v3 seed 2 | v3 seed 3 | v3 ensemble |
|---|---|---|---|---|---|---|
| expense (29) | 25/29 | 25/29 | 25/29 | 25/29 | 23/29 | 24/29 |
| income (21) | 17/21 | 18/21 | 20/21 | 20/21 | 20/21 | 20/21 |
| borrow (20) | 12/20 | 20/20 | 19/20 | 20/20 | 20/20 | 20/20 |
| lend (20) | 15/20 | 18/20 | 20/20 | 19/20 | 20/20 | 20/20 |
| repayment_in (4) | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| repayment_out (15) | 14/15 | 13/15 | 15/15 | 14/15 | 15/15 | 15/15 |
| transfer (35) | 34/35 | 35/35 | 35/35 | 35/35 | 35/35 | 35/35 |
| refund (3) | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |

**Debt-only slice (n=13)**

| metric | old | v2 seed 1 | v2 mean ± sd | v3 seed 1 | v3 seed 2 | v3 seed 3 | v3 mean ± sd | v3 ensemble |
|---|---|---|---|---|---|---|---|---|
| type accuracy | 0.0769 | 0.9231 | 0.9231 ± 0.0000 | 0.9231 | 1.0000 | 1.0000 | 0.9744 ± 0.0444 | 1.0000 |
| type macro f1 | 0.1000 | 0.9286 | 0.9241 ± 0.0078 | 0.9150 | 1.0000 | 1.0000 | 0.9717 ± 0.0491 | 1.0000 |
| target exact | 0.5385 | 1.0000 | 1.0000 ± 0.0000 | 0.7692 | 0.7692 | 0.8462 | 0.7949 ± 0.0444 | 0.7692 |
| type target exact | 0.0000 | 0.9231 | 0.9231 ± 0.0000 | 0.6923 | 0.7692 | 0.8462 | 0.7692 ± 0.0769 | 0.7692 |
| value exact | 1.0000 | 1.0000 | 1.0000 ± 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 | 1.0000 |
| end to end exact | 0.0000 | 0.9231 | 0.9231 ± 0.0000 | 0.6923 | 0.7692 | 0.8462 | 0.7692 ± 0.0769 | 0.7692 |

**Debt-only, annotation-v1 `skipped` sub-slice (n=8)**

| metric | old | v2 seed 1 | v2 mean ± sd | v3 seed 1 | v3 seed 2 | v3 seed 3 | v3 mean ± sd | v3 ensemble |
|---|---|---|---|---|---|---|---|---|
| type accuracy | 0.1250 | 0.8750 | 0.8750 ± 0.0000 | 0.8750 | 1.0000 | 1.0000 | 0.9583 ± 0.0722 | 1.0000 |
| type macro f1 | 0.1667 | 0.9000 | 0.8910 ± 0.0156 | 0.8730 | 1.0000 | 1.0000 | 0.9577 ± 0.0733 | 1.0000 |
| target exact | 0.5000 | 1.0000 | 1.0000 ± 0.0000 | 0.7500 | 0.7500 | 0.8750 | 0.7917 ± 0.0722 | 0.7500 |
| type target exact | 0.0000 | 0.8750 | 0.8750 ± 0.0000 | 0.6250 | 0.7500 | 0.8750 | 0.7500 ± 0.1250 | 0.7500 |
| value exact | 1.0000 | 1.0000 | 1.0000 ± 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 | 1.0000 |
| end to end exact | 0.0000 | 0.8750 | 0.8750 ± 0.0000 | 0.6250 | 0.7500 | 0.8750 | 0.7500 ± 0.1250 | 0.7500 |

**Non-debt notes (n=134)**

| metric | old | v2 seed 1 | v2 mean ± sd | v3 seed 1 | v3 seed 2 | v3 seed 3 | v3 mean ± sd | v3 ensemble |
|---|---|---|---|---|---|---|---|---|
| type accuracy | 0.9179 | 0.9254 | 0.9229 ± 0.0043 | 0.9627 | 0.9478 | 0.9478 | 0.9527 ± 0.0086 | 0.9552 |
| type macro f1 | 0.9028 | 0.8974 | 0.9044 ± 0.0061 | 0.9744 | 0.9513 | 0.9649 | 0.9635 ± 0.0116 | 0.9701 |
| target exact | 0.8657 | 0.8881 | 0.8831 ± 0.0043 | 0.8433 | 0.8731 | 0.8657 | 0.8607 ± 0.0155 | 0.8731 |
| type target exact | 0.8134 | 0.8358 | 0.8333 ± 0.0043 | 0.8358 | 0.8507 | 0.8507 | 0.8458 ± 0.0086 | 0.8657 |
| value exact | 1.0000 | 1.0000 | 1.0000 ± 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 | 1.0000 |
| end to end exact | 0.8134 | 0.8358 | 0.8333 ± 0.0043 | 0.8358 | 0.8507 | 0.8507 | 0.8458 ± 0.0086 | 0.8657 |

Reading against the five families the user listed, on seed 1 / ensemble (recall table):

- **income** read as expense/refund: 18/21 (v2 seed 1) → 20/21 on seeds 1, 2, 3 and the ensemble; the miss is `phu cap an trua 730k` (allowance → expense).
- **shop / app / restaurant purchases** (expense): no net gain: 25/29 → 25/29 seed 1, 24/29 ensemble, 23/29 seed 3. On seed 1 no purchase note is read as refund and only the gold-as-a-gift note (`mua nửa chỉ vàng cho con 4tr2`) is read as `transfer` (the ensemble also reads `go! big c siêu thị 1tr1` as transfer); but brand-only notes are now read as **income** (`bach hoa xanh trung sua 87k`, `pizza 4p tối t7 …`) or `lend` (`khoa hoc online udemy 249k`). [INFERENCE] Several contrast-02 income notes name a platform or brand as the paying party (`shopee`, `grab`, `chợ tốt`), which may have taught "brand + amount → income"; not ablated.
- **installment / app-loan repayments**: repayment_out 13/15 → 15/15 (seeds 1, 3, ensemble), 14/15 (seed 2).
- **paying on behalf of someone (lend)** and the other lend notes: lend 18/20 → 20/20 (seed 1, seed 3, ensemble); no lend → repayment_out misread is left on seed 1 or the ensemble.
- **debt-only notes**: type accuracy 12/13 → 12/13 (seed 1), 13/13 (seeds 2, 3, ensemble); but end-to-end falls (below).

What got worse: **target exact** on human-value-01 0.8980 (v2 seed 1) → 0.8367 (seed 1), 0.8639 (ensemble),
and the debt slice target exact 1.0000 → 0.7692 / 0.7692. The wrong targets of the debt notes are span-extent
errors (`no tien nha ba chu` → `ba`, `no tien nha chu tro` → `chu`, `Quang con thieu` → `Quang con`).
[INFERENCE] Some contrast-02 targets are a bare generic noun (`ban`, `dong nghiep`) or follow a
`thg`/`em` prefix, which may teach the model to stop at the first word of a role noun; not
ablated. 7 of the 24 wrong targets of seed 1 (4 of 20 of the ensemble) are sub-word fragments (`hsbc` →
`sbc`, `vpbank` → `pbank`, `pizza 4p` → `piz`); the old encoder gets 3 of those 7 right and
retrain-v2 seed 1 gets 4 right, so fragment spans are not new, but not all of them are old
either.

## 8. Notes still wrong on human-value-01

"Wrong" = type, target or value span not exactly the gold (end-to-end). The last two columns say
whether the old encoder / retrain-v2 seed 1 got the note end-to-end right (✓) or not (✗).

**v3 seed 1: 26 of 147 wrong**

| id | note | gold (type / target / value) | pred (type / target / value) | wrong | debt-only | old ok | v2 s1 ok |
|---|---|---|---|---|---|---|---|
| 37a5dafba00f | `ve may bay vietjet tet 2tr9` | expense / vietjet / 2tr9 | expense / viet / 2tr9 | target |  | ✗ | ✓ |
| c6f1f2cad163 | `phu cap an trua 730k` | income / None / 730k | expense / phu / 730k | type,target |  | ✗ | ✗ |
| 08126fa63696 | `thanh toán thẻ hsbc 7,2tr` | repayment_out / hsbc / 7,2tr | repayment_out / sbc / 7,2tr | target |  | ✓ | ✓ |
| 00e7ccfcb8c2 | `be chở về nhà 42.000` | expense / be / 42.000 | expense / None / 42.000 | target |  | ✗ | ✗ |
| e230c530923c | `taxi xanh sm ra san bay 210k` | expense / xanh sm / 210k | expense / taxi xanh sm / 210k | target |  | ✗ | ✗ |
| 0564b63cc451 | `rút tiền mặt ở vpbank 1tr5` | transfer / None / 1tr5 | transfer / vpbank / 1tr5 | target |  | ✓ | ✗ |
| 94b4a287f26a | `quà sinh nhật Vy 300 nghìn` | expense / None / 300 nghìn | expense / Vy / 300 nghìn | target |  | ✗ | ✗ |
| 8c07c9e24fe1 | `bach hoa xanh trung sua 87k` | expense / bach hoa xanh / 87k | income / None / 87k | type,target |  | ✗ | ✗ |
| 43b46e772c2b | `no tien nha ba chu 1 thang` | borrow / ba chu / None | borrow / ba / None | target | yes | ✗ | ✓ |
| f58264804919 | `rut tien vpbank 1tr` | transfer / None / 1tr | transfer / pbank / 1tr | target |  | ✓ | ✗ |
| 79ef1ca54d34 | `no tien nha chu tro 1tr5 chua tra` | borrow / chu tro / 1tr5 | borrow / chu / 1tr5 | target | yes | ✗ | ✓ |
| d3cbf60ba4d3 | `cho ban Quang muon 2tr` | lend / Quang / 2tr | lend / ban / 2tr | target |  | ✗ | ✓ |
| 5facc59b6ce4 | `no tien an cua Vy 90k chua tra` | borrow / Vy / 90k | lend / Vy / 90k | type | yes | ✗ | ✓ |
| 60a88b884cbb | `go! big c siêu thị 1tr1` | expense / go! big c / 1tr1 | expense / big / 1tr1 | target |  | ✗ | ✗ |
| 103d22749e3e | `taxi ra sân bay 250.000d` | expense / None / 250.000d | expense / xi / 250.000d | target |  | ✓ | ✓ |
| b4843fb2a505 | `mua giày nike 1tr8 hơi đắt` | expense / None / 1tr8 | expense / nike / 1tr8 | target |  | ✗ | ✗ |
| ba582f76355e | `pizza 4p tối t7 389.000đ` | expense / pizza 4p / 389.000đ | income / piz / 389.000đ | type,target |  | ✗ | ✗ |
| 4ddd3440798f | `sua tuoi banh keo bach hoa xanh 96k` | expense / bach hoa xanh / 96k | expense / ch hoa xanh / 96k | target |  | ✗ | ✗ |
| 73d3b3309ede | `mua nửa chỉ vàng cho con 4tr2` | expense / None / 4tr2 | transfer / None / 4tr2 | type |  | ✗ | ✗ |
| 0c55e0ef2432 | `khoa hoc online udemy 249k` | expense / udemy / 249k | lend / khoa / 249k | type,target |  | ✗ | ✗ |
| 04650ca0c139 | `vay chị dâu 5tr sửa nhà` | borrow / chị dâu / 5tr | borrow / chị / 5tr | target |  | ✗ | ✓ |
| afdc5bc92175 | `nạp ví momo trả tiền điện 700k` | transfer / None / 700k | transfer / m / 700k | target |  | ✗ | ✓ |
| 991487b7d067 | `tra bot no ban Duy 500` | repayment_out / Duy / 500 | repayment_out / ban Duy / 500 | target |  | ✗ | ✗ |
| 8cba9d0ef664 | `mượn chị hai 3 triệu` | borrow / chị hai / 3 triệu | borrow / chị / 3 triệu | target |  | ✓ | ✓ |
| 08ba8df8cb73 | `chi Loan muon 2 trieu` | lend / Loan / 2 trieu | lend / chi / 2 trieu | target |  | ✗ | ✓ |
| 6b471f45cc33 | `Quang con thieu 250k tien ve` | lend / Quang / 250k | lend / Quang con / 250k | target | yes | ✗ | ✗ |

**v3 ensemble: 21 of 147 wrong**

| id | note | gold (type / target / value) | pred (type / target / value) | wrong | debt-only | old ok | v2 s1 ok |
|---|---|---|---|---|---|---|---|
| c6f1f2cad163 | `phu cap an trua 730k` | income / None / 730k | expense / phu / 730k | type,target |  | ✗ | ✗ |
| 08126fa63696 | `thanh toán thẻ hsbc 7,2tr` | repayment_out / hsbc / 7,2tr | repayment_out / h / 7,2tr | target |  | ✓ | ✓ |
| 00e7ccfcb8c2 | `be chở về nhà 42.000` | expense / be / 42.000 | expense / None / 42.000 | target |  | ✗ | ✗ |
| e230c530923c | `taxi xanh sm ra san bay 210k` | expense / xanh sm / 210k | expense / taxi xanh sm / 210k | target |  | ✗ | ✗ |
| 0564b63cc451 | `rút tiền mặt ở vpbank 1tr5` | transfer / None / 1tr5 | transfer / vpbank / 1tr5 | target |  | ✓ | ✗ |
| 94b4a287f26a | `quà sinh nhật Vy 300 nghìn` | expense / None / 300 nghìn | expense / Vy / 300 nghìn | target |  | ✗ | ✗ |
| 8c07c9e24fe1 | `bach hoa xanh trung sua 87k` | expense / bach hoa xanh / 87k | income / None / 87k | type,target |  | ✗ | ✗ |
| 43b46e772c2b | `no tien nha ba chu 1 thang` | borrow / ba chu / None | borrow / ba / None | target | yes | ✗ | ✓ |
| 79ef1ca54d34 | `no tien nha chu tro 1tr5 chua tra` | borrow / chu tro / 1tr5 | borrow / chu / 1tr5 | target | yes | ✗ | ✓ |
| d3cbf60ba4d3 | `cho ban Quang muon 2tr` | lend / Quang / 2tr | lend / ban / 2tr | target |  | ✗ | ✓ |
| 60a88b884cbb | `go! big c siêu thị 1tr1` | expense / go! big c / 1tr1 | transfer / big / 1tr1 | type,target |  | ✗ | ✗ |
| 103d22749e3e | `taxi ra sân bay 250.000d` | expense / None / 250.000d | expense / taxi / 250.000d | target |  | ✓ | ✓ |
| b4843fb2a505 | `mua giày nike 1tr8 hơi đắt` | expense / None / 1tr8 | expense / nike / 1tr8 | target |  | ✗ | ✗ |
| ba582f76355e | `pizza 4p tối t7 389.000đ` | expense / pizza 4p / 389.000đ | income / piz / 389.000đ | type,target |  | ✗ | ✗ |
| 4ddd3440798f | `sua tuoi banh keo bach hoa xanh 96k` | expense / bach hoa xanh / 96k | expense / ch hoa / 96k | target |  | ✗ | ✗ |
| 73d3b3309ede | `mua nửa chỉ vàng cho con 4tr2` | expense / None / 4tr2 | transfer / None / 4tr2 | type |  | ✗ | ✗ |
| 0c55e0ef2432 | `khoa hoc online udemy 249k` | expense / udemy / 249k | lend / khoa / 249k | type,target |  | ✗ | ✗ |
| afdc5bc92175 | `nạp ví momo trả tiền điện 700k` | transfer / None / 700k | transfer / m / 700k | target |  | ✗ | ✓ |
| 991487b7d067 | `tra bot no ban Duy 500` | repayment_out / Duy / 500 | repayment_out / ban / 500 | target |  | ✗ | ✗ |
| 08ba8df8cb73 | `chi Loan muon 2 trieu` | lend / Loan / 2 trieu | lend / chi / 2 trieu | target |  | ✗ | ✓ |
| 6b471f45cc33 | `Quang con thieu 250k tien ve` | lend / Quang / 250k | lend / Quang con / 250k | target | yes | ✗ | ✗ |

Of the 26 notes of seed 1, 6 have a wrong type; the other 20 are target-span errors only; the value
parser is exact on all 147 notes. Ensemble: 6 wrong types, 15 target-only.

## 9. Size and latency (INT8, 1 thread, 200 runs, 50 test notes, systems interleaved)

| system | ONNX size (B) | bundle size (B) | p50 ms | p95 ms |
|---|---|---|---|---|
| single model (seed 1) | 28,658,533 | 29,287,783 | 1.34 | 1.68 |
| ensemble of 3 | 85,975,599 | 86,604,849 | 3.92 | 5.03 |

The ensemble runs three ONNX sessions and three tokenizations back to back (no parallelism);
the ensemble bundle is 86.6 MB, outside the 10–30 MB deployment goal of the journal; the row
measures what averaging could add, not a deployable bundle.

## 10. Gate and verdict

| candidate | test type macro-F1 drop (≤ 0.01) | test target exact drop (≤ 0.02) | human-value-01 debt e2e (≥ 0.8) | gate |
|---|---|---|---|---|
| v3 seed 1 | +0.0288 (FAIL) | +0.0095 (pass) | 0.6923 (FAIL) | FAIL |
| v3 seed 2 | +0.0022 (pass) | +0.0286 (FAIL) | 0.7692 (FAIL) | FAIL |
| v3 seed 3 | +0.0502 (FAIL) | -0.0190 (pass) | 0.8462 (pass) | FAIL |
| mean of 3 seeds | +0.0270 (FAIL) | +0.0063 (pass) | 0.7692 (FAIL) | FAIL |
| ensemble | +0.0152 (FAIL) | +0.0000 (pass) | 0.7692 (FAIL) | FAIL |

No candidate passes. Seed 1 (deployment convention) fails two criteria. **Verdict: REJECT.**
`gidi-finance-v1` stays the encoder; nothing is deployed or published.

What the numbers say, beyond the rule (context, not a change of the rule):

- **The listed type confusions are mostly fixed.** human-value-01 type accuracy 0.8435 (old) → 0.9252 (v2 seed 1) → 0.9592 (v3 seed 1, ensemble); 3-seed mean 0.9546 ± 0.0039 against 0.9229 ± 0.0039 for v2; income, repayment_out and lend are 20/21, 15/15, 20/20 on seed 1.
- **End-to-end is not better than v2 on seed 1** (0.8231 vs 0.8435), because target exactness fell more than type accuracy rose. The ensemble is the best row (0.8571); seed 3 is 0.8503, the 3-seed mean 0.8390 ± 0.0142 vs v2 0.8413 ± 0.0039.
- **The ensemble fixes variance, not bias.** Test type macro-F1 0.9411 sits between the seeds (0.9062 / 0.9276 / 0.9542) and above their mean 0.9293; target exact equals the old encoder's (0.8857); debt-slice type accuracy is 13/13, but its target exactness (0.7692) is the same as seed 1's.
- **Seed noise is large on a 105-note test**: the v3 seeds span 0.9062–0.9542 type macro-F1 (sd 0.0240, up from 0.0121 for v2); one note is ±0.0095 accuracy. Seed 2 would pass criterion (a) but fails (b) and (c).
- **Side effects of the added families**: the two frozen-test expense → lend errors, brand-only expense notes read as income, and the shorter debt-note targets. They are consistent with the contrast-02 content but were not isolated by an ablation.

## 11. Deviations and notes

- **`scripts/build_debt_relabel_queue.py`** `DERIVED_DIRS` now also skips `contrast-02` and `training-v3`, so the existing builders' `--check` (debt-01, contrast-01, training-v2) stay valid; the committed files of those batches are unchanged and their `--check` passes.
- `contrast-02` also drops near duplicates of the `training-v2` train set (not a corpus reference, because it is a derived set) and applies the extra char-3 ≥ 0.6 bound to human-value-01; contrast-01 had neither.
- Training ran in one command (about 17 minutes in total); no run was killed or restarted.
- MPS kernels are not bit-reproducible; a rerun of a seed may differ slightly (the trainer warns).
- `experiments/annotation-v3-retrain-v3/.gitignore` ignores the ONNX files and `weights/`; reports, results and the per-seed export JSON are tracked.
- Metric conventions are those of the earlier reports: type macro-F1 averages the classes present in gold (`gidi.evaluation.metrics`); target exact counts null = null as correct; value exact is the (start, end) span; end-to-end exact = type, target and value all right; sd is the sample standard deviation over 3 seeds.
- Test and probe per-note change lists against the old encoder were not stored (they are not in the required scope); the human-value-01 lists above are.
