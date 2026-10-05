# annotation-v3 retrain v4 of encoder 1: word snap + contrast-03 (target-span notes), 3 seeds

**Verdict: the 95% end-to-end target is NOT reached on human-value-01 (a proxy; the real score is
human-value-02, not scored here).** The release candidate (seed 1 + word snap) gets
**0.8844** end-to-end exact on the 147 complete human-value-01 notes (130 of 147; 140
are needed for 95%), against 0.7415 (old encoder) and 0.8435 (run 2 seed 1). The three-seed mean
is 0.8776 ± 0.0118. The value parser is exact on all 147 notes, so every
remaining error is an encoder error: 11 target-only, 4 type-only, 2 type + target (seed 1).
The word snap fixes sub-word fragments (frozen-test target exact +0.029 on run 2 seed 1 and on
run 4 seed 1; never breaks a correct span) but the remaining target errors on human-value-01 are
**whole-word** errors (`bach hoa xanh` -> `hoa xanh`, `go! big c` -> `big`, `chi Loan` -> `chi`),
which a snap cannot fix. Nothing is deployed or published.

Every number is read from `results.json` (held-out scoring, run once), `snap-effect.json` or the
files named in each section. Protocol (candidate = seed 1, fixed before anything was trained or
scored): `protocol.json`.

## 1. Word snap

`gidi.inference.decode.snap_to_words` / `snap_span_to_words` and the option
`GidiPredictor(..., snap_words=False)` / `GidiPredictor.from_bundle(..., snap_words=False)`.
After `decode_first_span`, the target span is extended outward to whole-word boundaries: a word is
a maximal run of non-whitespace characters; punctuation (Unicode category `P*`) at the edge of the
extension is trimmed (`(vpbank),` -> `vpbank`); a multi-word span keeps all its words, only its two ends
move, never inward; no span stays `None`. Off by default (web parity unchanged, no change to
`dump_web_parity` output); the evaluation of this run turns it on. Tests
(`tests/test_snap_words.py`, 18): `pbank` in `rut tien vpbank 1tr` -> `vpbank`, `sbc` -> `hsbc`,
`piz` -> `pizza` (only that word, not `4p`), `ch hoa xanh` -> `bach hoa xanh`, Vietnamese
code points (`ảo` -> `thảo`, `dâ` -> `dâu`, `ẹ` -> `mẹ`), multi-word spans kept
(`chu tro`, `go! big c`, `chị dâu`), edge punctuation, an empty span and `None`, member tokens
unchanged, and a predictor test (default off; with the snap the span only grows and sits on whitespace
boundaries).

Paired effect on the target span (`scripts/compare_snap_effect.py` -> `snap-effect.json`;
"fixed" = exact only with the snap, "broken" = exact only without it):

| model | set | notes changed | fixed | broken | changed, still wrong |
|---|---|---|---|---|---|
| old encoder | frozen test (105) | 3 | 2 | 0 | 1 |
| old encoder | probe-v1 (81) | 5 | 3 | 0 | 2 |
| old encoder | human-value-01 (147) | 5 | 2 | 0 | 3 |
| run 2 seed 1 | frozen test (105) | 3 | 3 | 0 | 0 |
| run 2 seed 1 | probe-v1 (81) | 4 | 0 | 0 | 4 |
| run 2 seed 1 | human-value-01 (147) | 3 | 0 | 0 | 3 |
| run 4 seed 1 | frozen test (105) | 3 | 3 | 0 | 0 |
| run 4 seed 1 | probe-v1 (81) | 5 | 4 | 0 | 1 |
| run 4 seed 1 | human-value-01 (147) | 2 | 0 | 0 | 2 |

The snap never breaks a correct span (0 broken in 33 changed notes) and fixes 17 of them. Whole-set effect: run 2 seed 1 frozen-test
target exact 0.8667 -> 0.8952, old encoder 0.8857 -> 0.9048 (probe 0.8148 -> 0.8519, human-value-01
0.8367 -> 0.8503; end-to-end 0.7415 -> 0.7483). On human-value-01 it changes nothing for run 2 seed 1 and
run 4 seed 1: their target errors there are already whole words.

## 2. contrast-03

`datasets/annotation-v3/contrast-03/` (builder `scripts/build_contrast_03.py`, `--check` reproduces
it byte for byte). `source.jsonl` is the hand-written input (135 candidates composed by the assistant
from `configs/annotation-v3.yaml`, `docs/annotation-v1.md` and `docs/annotation-v3.md`; the five
failure families come from the task description and from the *kinds* of target error in the
retrain-v3 report, no human-value-01/02, frozen-test or probe-v1 note was copied or paraphrased while
writing; the shop / item names are common Vietnamese brands). Labels are LLM-made with user approval:
`provenance.jsonl` marks every note `annotator: llm`. Every label is `complete`, exact code-point spans (the
builder rejects a span that is not on a word boundary), and passes
`scripts/validate_annotations.py --config configs/annotation-v3.quet.yaml --queue
datasets/annotation-v3/contrast-03/queue.jsonl`: **83 labels, 0 errors**. Every value span also
equals the unchanged parser's span (checked once, 0 mismatches).

| family | what it teaches | kept (quota) | types |
|---|---|---|---|
| shop | shop / supermarket / app / chain / ride service is the target, multi-word names kept: `bách hóa xanh` (x2), `circle k` (x2), `pizza 4p` (x2), `pizza hut`, `the gioi di dong`, `nguyen kim`, `xanh sm`; the `go! big c` candidates did not survive the quota (section 12); the two `taxi` -> null candidates were dropped by the quota too | 26 (26) | expense |
| gift | purchase for someone: the receiver is not a target; a shop named in the note is | 12 (12) | expense |
| item | product / brand as the item (`giày adidas`, `iphone`, `macbook`): null target unless a shop is named | 12 (12) | expense |
| bank | a bank as channel (cash withdrawal, own-account transfer, savings): transfer, null; a bank as lender / payer: target | 13 (14) | transfer 10, repayment_out 2, refund 1 |
| kin | kinship+role targets of two words (`chủ trọ`, `bà chủ`, `chị dâu`, `anh rể`, `bà ngoại`, `cô giáo`); `Hùng con` -> `Hùng` | 20 (20) | expense, income, borrow, lend, repayment_in, repayment_out |

83 of 135 candidates kept: type expense 54, transfer 10, lend 6, borrow 5, repayment_out 4,
repayment_in 2, income 1, refund 1; 26 null targets, 34 targets of two or more words, 0 null values; 47% unaccented
(39 of 83); family quotas filled round-robin over the `cue` tags. No income note names a brand
(the suspect pattern of run 3).

Leakage gate (`build_contrast_03.py`; rules of `scripts/build_human_value_queue.py`: exact, folded,
sequence >= 0.9, token Jaccard, char-3-gram >= 0.8) against human-value-01 (150), **human-value-02
(150; queue sha256 `a9ba9627f24796d6…`; its texts are only scored by the gate, never printed or stored)**,
the frozen test split files (143 reference rows), probe-v1 (81), the existing corpus, `training-v2` train
(905), `contrast-02` (100) and earlier candidates, **plus** the stricter bound: folded char-3-gram
Jaccard >= 0.6 to a human-value-01 or human-value-02 note drops the candidate.

| drop reason | candidates |
|---|---|
| near duplicate of the existing corpus | 7 |
| near duplicate of human-value-01 (standard rules) | 1 |
| near duplicate of human-value-02 (standard rules) | 3 |
| near duplicate of `training-v2` train | 1 |
| near duplicate of an earlier candidate | 2 |
| char-3 Jaccard >= 0.6 to a human-value-01 note | 1 |
| char-3 Jaccard >= 0.6 to a human-value-02 note | 1 |
| over the per-family quota (round-robin by cue) | 36 |
| **dropped / candidates** | **52 / 135** |

Largest char-3 Jaccard of a kept note: 0.5909 to a human-value-01 note, 0.5278 to a human-value-02
note (bound 0.6). The dropped candidates (my own texts) are listed in `contrast-03/manifest.json`.

**Convention note.** `quà sinh nhật <name> 350k` (1 kept note, `qua sinh nhat Duc 350k`) and the
`mua quà ... cho X` notes give the receiver no target. That follows the human-value-01 gold and the
task, but `docs/annotation-v1.md` also lists `quà sinh nhật bé Na 300k` -> `Na` (recipient of gift
money). The two readings conflict in the contract; this batch follows the human-labelled
convention for a gift *purchase* and was left out of ceremony money (`mừng cưới`, `lì xì`).

## 3. training-v4

`datasets/annotation-v3/training-v4/` (builder `scripts/build_annotation_v3_training_v4.py`,
`--check` reproduces it): `train.jsonl` = the 905 byte-identical `training-v2` records + 95 of the
100 contrast-02 notes + the 83 contrast-03 notes = **1083**; `validation.jsonl` / `test.jsonl` are the
same symlinks to the frozen annotation-v1 splits. Annotators: human 256, ai 546, llm 281.

Removed from contrast-02 (rules fixed from the training notes only, before training):

| id | note | type / target | rule |
|---|---|---|---|
| contrast-02-3661717489ce | `grab thuong chuyen di 120k` | income / grab | brand-income: an income note whose target is a brand or platform (the `brand + amount -> income` pattern that turned shop expenses into income in run 3) |
| contrast-02-92d41fc5f684 | `tam ung cho dong nghiep 400k` | lend / dong nghiep | generic-peer-noun: the target is a bare generic peer noun (bạn, đồng nghiệp) |
| contrast-02-54fb86f37d32 | `dong ho tien phong cho ban 2tr` | lend / ban | generic-peer-noun |
| contrast-02-79a01d300e6d | `khách ck tiền làm web 7tr` | income / khách | char3 >= 0.6 to a human-value-02 note (score only, the note was not read) |
| contrast-02-0507f1239566 | `mua gium Lan 2 ve xem phim 240k` | lend / Lan | char3 >= 0.6 to a human-value-02 note |

Only the grab note is a brand-named income note (the `shopee` / `chợ tốt` income notes of the
report are not in the 100 kept contrast-02 notes; the contrast-02 income notes name no other brand);
other bare role targets (`khách`, `sếp`, `me`, `ba`, `bố`) were kept because the docs make them
real targets.

| type | training-v2 | contrast-02 kept | contrast-03 | training-v4 |
|---|---|---|---|---|
| expense | 217 | 25 | 54 | 296 |
| income | 84 | 23 | 1 | 108 |
| borrow | 116 | 6 | 5 | 127 |
| lend | 111 | 21 | 6 | 138 |
| repayment_in | 75 | 0 | 2 | 77 |
| repayment_out | 99 | 20 | 4 | 123 |
| transfer | 131 | 0 | 10 | 141 |
| refund | 72 | 0 | 1 | 73 |

Leakage asserted (raises, writes nothing) against the frozen test (105), probe-v1 (81), human-value-01
(150) and **human-value-02 (150)**: no train id and no verbatim or NFC-lowercase text in any of them, the 9 debt-01
ids excluded from human-value-01 absent, **no new (contrast-02 kept + contrast-03) note a near duplicate of any note of the four
sets**, none at char-3 Jaccard >= 0.6 to a human-value-01 or human-value-02 note (max 0.5909 / 0.5667), and
no base note a near duplicate of a human-value-02 note. Disclosed, not changed: the 905 base notes are
the templated corpus text of every earlier run; 7 of them are near duplicates of frozen-test notes and 2 of
human-value-01 notes under the standard rules (as in earlier runs; 0 of probe-v1, 0 of human-value-02), and 42
(human-value-01) / 10 (human-value-02) of them are at char-3 >= 0.6, a bound applied to new notes only.
Hashes of every input and `train.jsonl`: `training-v4/manifest.json`.

## 4. Training and export

Recipe unchanged from retrain-v2 / v3 (compression-v3 K2048 supervised student, BamiBERT layers
2/5/8/11, FFN 2048, vocab B-rank-8000, positions 32, 40 epochs, batch 8, encoder lr 5e-5, head lr
1e-3, warmup 0.1, wd 0.01, dropout 0.1, grad-clip 1.0, MPS, last epoch kept, `--retokenize`), seeds 1, 2, 3. Teacher-target
cache rebuilt for 1083 notes (`teacher-targets/`, frozen teacher `lr5e-05-seed1`; the supervised arm
does not read the teacher logits). 5440 steps per run (136 / epoch), final in-sample type/span F1 1.000,
train time 358.5 / 358.0 / 368.9 s, log `train-seeds123.log`. Trainer FP32 test type macro-F1
0.9339 / 0.9294 / 0.9288 (INT8 scores in section 6 are the same).

Export (`scripts/export_onnx.py`, `export-seed{1,2,3}.json`): FP32 parity max abs diff 2.1e-5 / 2.6e-5 / 4.2e-5 (limit 1e-4); INT8 vs FP32 type
agreement on the 50 export notes 1.00 / 1.00 / 1.00, tag agreement 0.998 / 1.000 / 1.000; INT8 size
28,658,533 B each. INT8 sha256: seed 1 `61c17d27b92e1ac4…`, seed 2
`567f6c1157071b66…`, seed 3 `71fa3e04f43d8691…` (full values in `results.json`).
Latency (INT8, 1 thread, 200 runs, 50 test notes): p50 1.41 ms / p95 1.76 ms without the snap, 1.40 / 1.73 with it (no measurable cost); bundle 29,287,783 B.

## 5. Selection rule and what was scored

`protocol.json` (written before training; sha256 in `results.json`): **release candidate = seed 1 (the
deployment convention) with the word snap**; seeds 1-3 are reported, nothing is selected among them by a
held-out score. Success criterion for the real test: end-to-end exact >= 0.95 on human-value-02.
The held-out scoring (`scripts/evaluate_encoder_retrain_v4.py`) ran once on the frozen test, probe-v1 and
human-value-01 (the script refuses to overwrite `results.json`); it was debugged only with `--smoke` on in-sample
data. The no-snap `old` and `run 2 seed 1` rows were asserted equal to the stored retrain-v2 results to the last digit
before anything was written. Columns: `old` = deployed `models/gidi-finance-v1` (no snap), `old+snap`, `run 2 s1` =
retrain-v2 seed 1 without the snap (the best earlier single model), `run 2 s1+snap`, `run 4 s1 no snap`
(isolates the data effect), `run 4 s1+snap` = **release candidate**, seeds 2 and 3 with the snap,
mean ± sd of the three snapped seeds (sample sd). Run 3 is not re-scored (see the run 3 report).

## 6. Frozen test (105)

| metric | old | old+snap | run 2 s1 | run 2 s1+snap | run 4 s1 no snap | run 4 s1+snap | run 4 s2+snap | run 4 s3+snap | run 4 mean ± sd |
|---|---|---|---|---|---|---|---|---|---|
| type accuracy | 0.9429 | 0.9429 | 0.9238 | 0.9238 | 0.9429 | 0.9429 | 0.9333 | 0.9333 | 0.9365 ± 0.0055 |
| type macro f1 | 0.9563 | 0.9563 | 0.9219 | 0.9219 | 0.9339 | 0.9339 | 0.9294 | 0.9288 | 0.9307 ± 0.0028 |
| target exact | 0.8857 | 0.9048 | 0.8667 | 0.8952 | 0.8952 | 0.9238 | 0.9143 | 0.9048 | 0.9143 ± 0.0095 |
| span f1 | 0.8077 | 0.8462 | 0.7963 | 0.8519 | 0.8269 | 0.8846 | 0.8571 | 0.8598 | 0.8672 ± 0.0151 |

Where it moves: the data change (no snap) lifts run 4 seed 1 over run 2 seed 1 in type macro-F1
(0.9219 -> 0.9339) and target exact (0.8667 -> 0.8952); the snap adds +0.0286 target exact on
top (0.9238, the best row). borrow / lend F1 stay below the old encoder's 1.000 (0.800 / 0.857 seed 1).

Per-class F1:

| type | old | old+snap | run 2 s1 | run 2 s1+snap | run 4 s1 no snap | run 4 s1+snap | run 4 s2+snap | run 4 s3+snap |
|---|---|---|---|---|---|---|---|---|
| expense | 0.9180 | 0.9180 | 0.9355 | 0.9355 | 0.9524 | 0.9524 | 0.9355 | 0.9355 |
| income | 0.8571 | 0.8571 | 0.8182 | 0.8182 | 0.9091 | 0.9091 | 0.9091 | 0.8696 |
| borrow | 1.0000 | 1.0000 | 0.8235 | 0.8235 | 0.8000 | 0.8000 | 0.8235 | 0.8000 |
| lend | 1.0000 | 1.0000 | 0.9412 | 0.9412 | 0.8571 | 0.8571 | 0.8421 | 0.9000 |
| repayment_in | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| repayment_out | 0.9524 | 0.9524 | 0.9091 | 0.9091 | 0.9524 | 0.9524 | 0.9524 | 0.9524 |
| transfer | 0.9231 | 0.9231 | 0.9474 | 0.9474 | 1.0000 | 1.0000 | 0.9730 | 0.9730 |
| refund | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## 7. probe-v1 (81 complete labels)

| metric | old | old+snap | run 2 s1 | run 2 s1+snap | run 4 s1 no snap | run 4 s1+snap | run 4 s2+snap | run 4 s3+snap | run 4 mean ± sd |
|---|---|---|---|---|---|---|---|---|---|
| type accuracy | 0.7778 | 0.7778 | 0.8025 | 0.8025 | 0.8272 | 0.8272 | 0.7901 | 0.9136 | 0.8436 ± 0.0634 |
| type macro f1 | 0.7543 | 0.7543 | 0.8136 | 0.8136 | 0.8264 | 0.8264 | 0.8235 | 0.9268 | 0.8589 ± 0.0588 |
| target exact | 0.8148 | 0.8519 | 0.8025 | 0.8025 | 0.8025 | 0.8519 | 0.8519 | 0.8272 | 0.8436 ± 0.0143 |
| span f1 | 0.7818 | 0.8364 | 0.7928 | 0.7928 | 0.7679 | 0.8393 | 0.8333 | 0.7928 | 0.8218 ± 0.0253 |

## 8. human-value-01 (150 labels; 3 `uncertain` excluded, 147 complete scored)

The old encoder column is `gidi-finance-v1` INT8 + the unchanged rule value parser (V8-old of earlier reports;
reproduces the stored value). Value spans come from the unchanged parser in every column.

**All complete notes (n=147)**

| metric | old | old+snap | run 2 s1 | run 2 s1+snap | run 4 s1 no snap | run 4 s1+snap | run 4 s2+snap | run 4 s3+snap | run 4 mean ± sd |
|---|---|---|---|---|---|---|---|---|---|
| type accuracy | 0.8435 | 0.8435 | 0.9252 | 0.9252 | 0.9592 | 0.9592 | 0.9456 | 0.9320 | 0.9456 ± 0.0136 |
| type macro f1 | 0.8049 | 0.8049 | 0.8929 | 0.8929 | 0.9411 | 0.9411 | 0.9460 | 0.9514 | 0.9462 ± 0.0052 |
| target exact | 0.8367 | 0.8503 | 0.8980 | 0.8980 | 0.9116 | 0.9116 | 0.8912 | 0.9184 | 0.9070 ± 0.0142 |
| type target exact | 0.7415 | 0.7483 | 0.8435 | 0.8435 | 0.8844 | 0.8844 | 0.8639 | 0.8844 | 0.8776 ± 0.0118 |
| value exact | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| end to end exact | 0.7415 | 0.7483 | 0.8435 | 0.8435 | 0.8844 | 0.8844 | 0.8639 | 0.8844 | 0.8776 ± 0.0118 |

Recall by gold type (k/n):

| gold type (n) | old | old+snap | run 2 s1 | run 2 s1+snap | run 4 s1 no snap | run 4 s1+snap | run 4 s2+snap | run 4 s3+snap |
|---|---|---|---|---|---|---|---|---|
| expense (29) | 25/29 | 25/29 | 25/29 | 25/29 | 27/29 | 27/29 | 24/29 | 23/29 |
| income (21) | 17/21 | 17/21 | 18/21 | 18/21 | 19/21 | 19/21 | 19/21 | 19/21 |
| borrow (20) | 12/20 | 12/20 | 20/20 | 20/20 | 19/20 | 19/20 | 20/20 | 19/20 |
| lend (20) | 15/20 | 15/20 | 18/20 | 18/20 | 20/20 | 20/20 | 19/20 | 19/20 |
| repayment_in (4) | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| repayment_out (15) | 14/15 | 14/15 | 13/15 | 13/15 | 14/15 | 14/15 | 15/15 | 15/15 |
| transfer (35) | 34/35 | 34/35 | 35/35 | 35/35 | 35/35 | 35/35 | 35/35 | 35/35 |
| refund (3) | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |

**Debt-only slice (n=13)**

| metric | old | old+snap | run 2 s1 | run 2 s1+snap | run 4 s1 no snap | run 4 s1+snap | run 4 s2+snap | run 4 s3+snap | run 4 mean ± sd |
|---|---|---|---|---|---|---|---|---|---|
| type accuracy | 0.0769 | 0.0769 | 0.9231 | 0.9231 | 0.9231 | 0.9231 | 1.0000 | 0.9231 | 0.9487 ± 0.0444 |
| type macro f1 | 0.1000 | 0.1000 | 0.9286 | 0.9286 | 0.9150 | 0.9150 | 1.0000 | 0.9150 | 0.9434 ± 0.0491 |
| target exact | 0.5385 | 0.6154 | 1.0000 | 1.0000 | 0.9231 | 0.9231 | 0.8462 | 0.9231 | 0.8974 ± 0.0444 |
| type target exact | 0.0000 | 0.0000 | 0.9231 | 0.9231 | 0.8462 | 0.8462 | 0.8462 | 0.8462 | 0.8462 ± 0.0000 |
| value exact | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| end to end exact | 0.0000 | 0.0000 | 0.9231 | 0.9231 | 0.8462 | 0.8462 | 0.8462 | 0.8462 | 0.8462 ± 0.0000 |

**Debt-only, annotation-v1 `skipped` sub-slice (n=8)**

| metric | old | old+snap | run 2 s1 | run 2 s1+snap | run 4 s1 no snap | run 4 s1+snap | run 4 s2+snap | run 4 s3+snap | run 4 mean ± sd |
|---|---|---|---|---|---|---|---|---|---|
| type accuracy | 0.1250 | 0.1250 | 0.8750 | 0.8750 | 0.8750 | 0.8750 | 1.0000 | 0.8750 | 0.9167 ± 0.0722 |
| type macro f1 | 0.1667 | 0.1667 | 0.9000 | 0.9000 | 0.8730 | 0.8730 | 1.0000 | 0.8730 | 0.9153 ± 0.0733 |
| target exact | 0.5000 | 0.6250 | 1.0000 | 1.0000 | 0.8750 | 0.8750 | 0.8750 | 0.8750 | 0.8750 ± 0.0000 |
| type target exact | 0.0000 | 0.0000 | 0.8750 | 0.8750 | 0.7500 | 0.7500 | 0.8750 | 0.7500 | 0.7917 ± 0.0722 |
| value exact | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| end to end exact | 0.0000 | 0.0000 | 0.8750 | 0.8750 | 0.7500 | 0.7500 | 0.8750 | 0.7500 | 0.7917 ± 0.0722 |

**Non-debt notes (n=134)**

| metric | old | old+snap | run 2 s1 | run 2 s1+snap | run 4 s1 no snap | run 4 s1+snap | run 4 s2+snap | run 4 s3+snap | run 4 mean ± sd |
|---|---|---|---|---|---|---|---|---|---|
| type accuracy | 0.9179 | 0.9179 | 0.9254 | 0.9254 | 0.9627 | 0.9627 | 0.9403 | 0.9328 | 0.9453 ± 0.0155 |
| type macro f1 | 0.9028 | 0.9028 | 0.8974 | 0.8974 | 0.9473 | 0.9473 | 0.9452 | 0.9569 | 0.9498 ± 0.0062 |
| target exact | 0.8657 | 0.8731 | 0.8881 | 0.8881 | 0.9104 | 0.9104 | 0.8955 | 0.9179 | 0.9080 ± 0.0114 |
| type target exact | 0.8134 | 0.8209 | 0.8358 | 0.8358 | 0.8881 | 0.8881 | 0.8657 | 0.8881 | 0.8806 ± 0.0129 |
| value exact | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| end to end exact | 0.8134 | 0.8209 | 0.8358 | 0.8358 | 0.8881 | 0.8881 | 0.8657 | 0.8881 | 0.8806 ± 0.0129 |

Reference rows from the run 3 report (stored, not re-scored): seed 1 type accuracy 0.9592, target exact 0.8367, end-to-end 0.8231; debt-slice end-to-end 0.6923.

Reading: the three target-error families of run 3 largely disappear (`vpbank`/`hsbc`/`pizza 4p` fragments, `nike`,
`chu tro`/`chị dâu`/`Quang con` cut to one word, `momo`, the `vpbank` cash withdrawal). Run 4 seed 1 is
the best single model so far (end-to-end 0.8844 vs 0.8435 for run 2 seed 1; type accuracy 0.9592; target exact
0.9116), but it does not match run 2 on the debt slice (11/13 vs 12/13) and the seed spread
is visible: end-to-end 0.8844 / 0.8639 / 0.8844. The debt slice sits at 0.8462 on all three seeds. Against run 2 seed 1 it fixes 10
notes and breaks 4 (two debt notes, `no tien nha ba chu` and `no tien an cua Vy`; two non-debt, `tiền phòng trọ t11` and `chi Loan muon`). Caveat: contrast-03 was written knowing the *kinds* of
human-value-01 errors (the task list), so the human-value-01 gain is not a clean held-out estimate;
human-value-02 is the independent test.

## 9. Notes still wrong on human-value-01 (seed 1 + snap): 17 of 147

"Wrong" = type, target or value span not exactly the gold. By kind: **target only 11, type only 4,
type + target 2, value 0**. The last four columns say whether the system got the note end-to-end right.

| id | note | gold (type / target / value) | pred (type / target / value) | wrong | debt | old ok | run2 s1 ok | run2 s1+snap ok | run4 s1 nosnap ok |
|---|---|---|---|---|---|---|---|---|---|
| c6f1f2cad163 | `phu cap an trua 730k` | income / None / 730k | expense / phu cap / 730k | type+target |  | ✗ | ✗ | ✗ | ✗ |
| bfd895a9e78f | `đóng tiền trả góp laptop 2tr3` | repayment_out / None / 2tr3 | repayment_in / None / 2tr3 | type |  | ✗ | ✗ | ✗ | ✗ |
| 00e7ccfcb8c2 | `be chở về nhà 42.000` | expense / be / 42.000 | expense / None / 42.000 | target |  | ✗ | ✗ | ✗ | ✗ |
| e230c530923c | `taxi xanh sm ra san bay 210k` | expense / xanh sm / 210k | expense / taxi xanh sm / 210k | target |  | ✗ | ✗ | ✗ | ✗ |
| 94b4a287f26a | `quà sinh nhật Vy 300 nghìn` | expense / None / 300 nghìn | expense / Vy / 300 nghìn | target |  | ✗ | ✗ | ✗ | ✗ |
| 8c07c9e24fe1 | `bach hoa xanh trung sua 87k` | expense / bach hoa xanh / 87k | expense / hoa xanh / 87k | target |  | ✗ | ✗ | ✗ | ✗ |
| 43b46e772c2b | `no tien nha ba chu 1 thang` | borrow / ba chu / None | borrow / nha ba / None | target | yes | ✗ | ✓ | ✓ | ✗ |
| e1889be5d444 | `icloud 50gb 19k` | expense / icloud / 19k | expense / icloud 50gb / 19k | target |  | ✓ | ✗ | ✗ | ✗ |
| 36067775309d | `tien thuong tet 12tr` | income / None / 12tr | expense / None / 12tr | type |  | ✗ | ✗ | ✗ | ✗ |
| 5facc59b6ce4 | `no tien an cua Vy 90k chua tra` | borrow / Vy / 90k | lend / Vy / 90k | type | yes | ✗ | ✓ | ✓ | ✗ |
| 60a88b884cbb | `go! big c siêu thị 1tr1` | expense / go! big c / 1tr1 | expense / big / 1tr1 | target |  | ✗ | ✗ | ✗ | ✗ |
| 4ddd3440798f | `sua tuoi banh keo bach hoa xanh 96k` | expense / bach hoa xanh / 96k | expense / tuoi / 96k | target |  | ✗ | ✗ | ✗ | ✗ |
| 73d3b3309ede | `mua nửa chỉ vàng cho con 4tr2` | expense / None / 4tr2 | transfer / None / 4tr2 | type |  | ✗ | ✗ | ✗ | ✗ |
| 0c55e0ef2432 | `khoa hoc online udemy 249k` | expense / udemy / 249k | refund / khoa / 249k | type+target |  | ✗ | ✗ | ✗ | ✗ |
| 844f7b8dd50f | `tiền phòng trọ t11 2tr8` | expense / None / 2tr8 | expense / trọ / 2tr8 | target |  | ✓ | ✓ | ✓ | ✗ |
| 991487b7d067 | `tra bot no ban Duy 500` | repayment_out / Duy / 500 | repayment_out / ban / 500 | target |  | ✗ | ✗ | ✗ | ✗ |
| 08ba8df8cb73 | `chi Loan muon 2 trieu` | lend / Loan / 2 trieu | lend / chi / 2 trieu | target |  | ✗ | ✓ | ✓ | ✗ |

What is left, grouped:

- **Multi-word names cut at a word boundary (3):** `bach hoa xanh trung sua 87k` -> `hoa xanh`,
  `sua tuoi banh keo bach hoa xanh 96k` -> `tuoi`, `go! big c siêu thị 1tr1` -> `big`. [INFERENCE] contrast-03
  never saw the failing spellings: its kept `bách hóa xanh` notes are accented (the unaccented candidate
  was dropped by the char-3 gate against human-value-01) and no `go! big c` / `big c` note survived the
  per-family quota (section 12).
- **Service prefix / suffix kept or missing (4):** `be chở về nhà` (`be` missed), `taxi xanh sm` (`taxi` added), `icloud 50gb`
  (`50gb` added), `tiền phòng trọ t11` (`trọ` instead of null; a run 2 pass).
- **Prefix word taken as the name (3):** `tra bot no ban Duy` -> `ban`, `chi Loan muon` -> `chi`; `no tien nha ba chu`
  -> `nha ba` (debt).
- **Receiver as target (1):** `quà sinh nhật Vy 300 nghìn` -> `Vy`, although contrast-03 has a null-target
  `qua sinh nhat Duc`; one note in the family cannot move the model.
- **Type (6):** `phu cap an trua` and `tien thuong tet` (income -> expense), `đóng tiền trả góp laptop`
  (repayment_out -> repayment_in, new vs run 3), `no tien an cua Vy` (borrow -> lend), `mua nửa chỉ vàng cho con`
  (expense -> transfer), `khoa hoc online udemy` (expense -> refund, plus target `khoa`).

The value parser is exact on all 147 notes.

## 10. Verdict and next direction

Not at 95% on the proxy. The fragment errors are gone; what remains are whole-word boundary errors
(multi-word brand names, prefix words kept or dropped) and four type errors, none of which the snap addresses.
To move from 0.884 to 0.95 on human-value-01 would need about 10 more notes right; the real
answer is the human-value-02 score (command below). If human-value-02 shows the same pattern, candidate next
steps (new experiments, not done here): a word-level span-boundary loss or more multi-word brand / prefix
(`ban X`, `chi X`) contrast notes; a left-extension rule is not justified by these errors (they are not fragments).

## 11. human-value-02 scoring (prepared, NOT run)

`scripts/score_human_value_02.py` refuses to run until `datasets/annotation-v3/human-value-02/labels.jsonl`
exists (checked: it exits with "does not exist yet" now), refuses to overwrite its output, verifies the
candidate ONNX hash equals the model scored in `results.json`, and scores once: `candidate` (run 4 seed 1 + snap),
`candidate_nosnap`, `old` (deployed encoder), `run2` (run 2 seed 1) and `run2_snap`, all with the unchanged
value parser. It writes `experiments/annotation-v3-retrain-v4/results-hv02.json` with type accuracy / macro-F1,
target exact, value exact, end-to-end exact, recall by type, every wrong note per system and the verdict against 0.95.

```sh
uv run python scripts/score_human_value_02.py
```

## 12. Deviations and notes

- `scripts/build_debt_relabel_queue.py` `DERIVED_DIRS` now also skips `contrast-03`, `training-v4` and
  `datasets/annotation-v3/human-value-02`, so the existing builders' `--check` (debt-01, contrast-01, contrast-02, training-v2, training-v3)
  stay valid; the new builders gate human-value-02 as an explicit group. The committed files of those batches are unchanged.
- human-value-02 lives under `datasets/annotation-v3/` (its queue and hash came from the test-set agent); this run only used
  it in the leakage gates and never read its notes for any other purpose.
- **Coverage gap in contrast-03 (found after scoring, not fixed).** The round-robin quota over `cue` tags
  (26 shop notes for 47 candidates) dropped every `go! big c`, `big c`, `lotte mart`, `mm mega market`,
  `aeon mall` and `emart` candidate (cue `supermarket` kept 4: two `bách hóa xanh`, `winmart`, `co.opmart`);
  `go! big c`, named in the task, is therefore not in the training data. Changing the data now would be tuning
  on the held-out numbers above, so run 4 stays as trained; a v5 should raise the shop quota or
  draw per-name instead of per-cue.
- The task's candidate list had 5 families; `kin` includes two `name + con` notes (`Hùng con`, `Nam con`) for the
  `Quang con` error. contrast-03 `source.jsonl` also has two `taxi` -> null candidates (the `taxi ra sân bay` error); neither was kept.
- Training ran in one command (about 18 minutes in total); no run was killed or restarted. MPS kernels are not bit-reproducible.
- `experiments/annotation-v3-retrain-v4/.gitignore` ignores the ONNX files and `weights/`; reports, results, protocol and the
  per-seed export JSON are tracked.
- Metric conventions are those of the earlier reports: type macro-F1 averages the classes present in gold
  (`gidi.evaluation.metrics`); target exact counts null = null as correct; value exact is the (start, end) span;
  end-to-end exact = type, target and value all right; sd is the sample standard deviation over 3 seeds.
- Extra scripts: `scripts/compare_snap_effect.py` (paired snap effect, `snap-effect.json`).
