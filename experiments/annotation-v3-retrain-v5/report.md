# annotation-v3 retrain v5 of encoder 1: gift receiver is the target + per-name shop quota, 3 seeds

Question (`protocol.json`, written before anything was trained or scored): after the user decision that
**a gift receiver is the target** (human-value-01 amendment-01, a relabel map for the LLM contrast notes) and a
per-name quota for the shop names that the contrast-03 quota dropped (`contrast-04`), does `training-v5`
improve encoder 1 (type + target, word snap on) + the unchanged rule value parser over run 4, without a
regression on the frozen test and probe-v1?

**Verdict: no improvement over run 4 on human-value-01.** Seed 1 + snap (the pre-declared candidate)
scores 0.8776 end-to-end exact (129 of 147) on the amended human-value-01 labels; run 4 seed 1 scores 0.8912
(131 of 147) on the same labels. The three-seed mean is 0.8889 ± 0.0142; run 4's mean on the amended labels is 0.8844 (derived, not rescored: the stored run 4 predictions
of all three seeds are `Vy` for the amended note, so each seed gains exactly one note over the reported 0.8776 ± 0.0118). probe-v1 type macro-F1 is clearly better (0.8264 -> 0.8878 on seed 1; mean 0.8695), the
frozen test is flat (type macro-F1 0.9339 -> 0.9372, target exact 0.9238 -> 0.9048 on seed 1; means 0.9406 / 0.9206),
the debt slice type is now 13/13 on every seed but its target spans are worse (seed 1 end-to-end 10/13, run 4 11/13).
`go! big c` is read correctly by seeds 2 and 3 but not by the candidate (`go! big`). Nothing is selected among seeds.

## 1. The rule and human-value-01 amendment-01

User decision (annotation-v3 clarification, `docs/annotation-v3.md` "Gift receivers"; it agrees with
`docs/annotation-v1.md` "Beneficiary is not payee"): **the receiver of a gift or ceremony money is the target**
(`quà sinh nhật bé Na 300k` -> `Na`, `quà sinh nhật Vy 300 nghìn` -> `Vy`, `mừng cưới Hoa 1 triệu` -> `Hoa`; kinship prefix
dropped before a proper name). Buying a concrete item for someone (`mua vở bút cho bé`, `mua hoa tặng mẹ`) keeps no target.
Configs and `datasets/annotation-v3/human-value-02/annotation-guide.md` are unchanged (docs only).

All 150 human-value-01 notes were read for gift / ceremony-money wording. One label changes (user-approved):

| id | note | old target | new target |
|---|---|---|---|
| `baseline-01-94b4a287f26a` | `quà sinh nhật Vy 300 nghìn` | null | `Vy` (14-16) |

`mừng cưới Hoa 1 triệu` -> `Hoa` already followed the rule; `mua nửa chỉ vàng cho con`, `sữa cho con`, `lì xì nhận đc 2tr`
(no receiver) stay null. The change is recorded in `datasets/annotation-v2/human-value-01/amendment-01.json`
(id, text, old / new target, reason, hashes), `freeze.json` has an `amendments` entry (labels hash
`2a79ec30…` -> `e0808f4b…`, `stats.target_null` 75 -> 74), `scripts/amend_human_value_01.py` applied it once and
`--check` verifies it. `scripts/validate_annotations.py --config configs/annotation-v2.quet.yaml --queue …/review-queue.jsonl`:
150 annotations, 0 errors. Every system below is scored on the amended labels, run 4 seed 1 included; its stored wrong-note set
differs from the rescore only by the amended note (asserted before writing results).

## 2. Training data scan: `gift-relabel-01`

`scripts/build_gift_relabel_01.py` scans `training-v4/train.jsonl` (1083 records) for gift wording (folded cue regex, 91
hits: ai 45, human 18, llm 28). Every hit was read.

- **Human annotation-v1 labels that conflict with the rule: none.** The 18 human hits are either gift notes that already follow it
  (`qua sinh nhat bé Na 300k` -> `Na`, `quà 20/10 cho mẹ` -> `mẹ`, `mung dam cuoi Tuan 500k` -> `Tuan`, `lì xì bà cho 500` -> `bà`,
  `đám cưới anh Phong mừng 500k` -> `Phong`, ...) or non-gift uses of `qua` (via / yesterday) with a null target (4, listed in the manifest).
  Human labels are not changed. The `ai` labels also follow the rule (receivers are targets, item purchases such as `mua hoa tặng mẹ` null).
- **LLM contrast notes, map `datasets/annotation-v3/gift-relabel-01/map.jsonl` (5 rows):**

| id | note | old | new | action |
|---|---|---|---|---|
| contrast-03-202ee7b034e9 | `mua qua sinh nhat cho Hanh 450k` | null | `Hanh` | relabel |
| contrast-03-5e8eb16c75e6 | `qua sinh nhat Duc 350k` | null | `Duc` | relabel |
| contrast-03-88c14150f4cf | `mua quà 20/10 cho cô Lan 340k` | null | `Lan` | relabel (prefix `cô` dropped) |
| contrast-03-24eb1714b7a3 | `gói quà tặng sếp mới 600k` | null | `sếp mới` | relabel |
| contrast-03-7813f886774e | `mua quà cho bạn Minh ở shopee 280k` | `shopee` | - | drop: two plausible counterparties (receiver and marketplace), uncertain under the one-target rule |

contrast-02 holds no such note (its gift notes are item purchases: `mua nhẫn vàng tặng mẹ`, `mua 1 chỉ vàng tặng cháu đầy tháng` -> null).
The contrast-03 gift items bought in a named shop (`mua nước hoa tặng vợ ở sephora` -> `sephora`) stay as they are.

## 3. contrast-04

`datasets/annotation-v3/contrast-04/` (builder `scripts/build_contrast_04.py`, `source.jsonl` written by an LLM from the rules, `--check`
reproduces it). The contrast-03 quota took notes round-robin over coarse `cue` tags and dropped every `go! big c`, `big c`, `lotte mart`, `mm mega market`,
`aeon mall` and `emart` note; contrast-04 draws **per shop name** (at most 3 notes per name; the build raises if one of those six is missing). 87 candidates, **43 kept**:
25 shop notes (go! big c 3, big c 3, lotte mart 3, mm mega market 3, aeon mall 3, emart 2, family mart 2, fpt shop 2, nhà sách phương nam 2, the pizza company 2),
15 gift notes with the receiver as target (`mừng cưới Lan 1tr` -> `Lan`, `quà 8/3 cho cô Hạnh 250k` -> `Hạnh`, `biếu bác Tư 500k` -> `bác Tư`, `mừng thọ ông nội 1tr` -> `ông nội`, ...; birthday, wedding,
housewarming, lì xì, holiday, elders, funeral, newborn, farewell) and 3 gift / ceremony notes without a receiver (`tiền mừng cưới 500k` -> null). All `expense`, LLM labels (`annotator: llm`), 26% unaccented.

Leakage gate = contrast-03's, with human-value-02 batches a **and** b (`review-queue.jsonl` + `review-queue-b.jsonl`, 200 notes): exact / folded / sequence / token / char-3 >= 0.8 near duplicates of the frozen test, probe-v1, human-value-01,
human-value-02, the corpus, training-v2, contrast-02 and contrast-03; then folded char-3 Jaccard >= 0.6 to a human-value-01 or -02 note. Dropped: contrast-03 2, existing-corpus 4, human-value-01 1,
probe-v1 1, over-quota 36 (max char-3 of the kept notes 0.4333 to human-value-01, 0.5833 to human-value-02). The test-set agent changed one note of batch b after the first build (`quà thôi nôi bé Gấu` replaced `lì xì cháu Bin`); contrast-04 and training-v5 were rebuilt against the final file before training.

## 4. training-v5

`datasets/annotation-v3/training-v5/` (builder `scripts/build_annotation_v3_training_v5.py`, `--check` reproduces it): the 905 byte-identical training-v2 records + the 178 other training-v4 records with the map applied (4 relabelled, 1 dropped)
and **2 removed** for human-value-02 batch b (`ban laptop cu duoc 5tr5`, `nhà thuốc long châu mua vitamin 280k`: contrast-02 notes at folded char-3 >= 0.6 to a batch-b note; decided by the gate score only, the test notes were not read)
+ the 43 contrast-04 notes = **1123**. `validation.jsonl` / `test.jsonl` are the same symlinks to the frozen annotation-v1 splits. Annotators: human 256, ai 546, llm 321.

| type | training-v2 | training-v4 rest (after map + removal) | contrast-04 | training-v5 |
|---|---|---|---|---|
| expense | 217 | 77 | 43 | 337 |
| income | 84 | 23 | 0 | 107 |
| borrow | 116 | 11 | 0 | 127 |
| lend | 111 | 27 | 0 | 138 |
| repayment_in | 75 | 2 | 0 | 77 |
| repayment_out | 99 | 24 | 0 | 123 |
| transfer | 131 | 10 | 0 | 141 |
| refund | 72 | 1 | 0 | 73 |

Leakage asserted (raises, writes nothing) against the frozen test (105), probe-v1 (81), human-value-01 (150) and human-value-02 (200): no train id and no verbatim or NFC-lowercase text in any of them, the 9 debt-01 ids excluded from human-value-01 absent,
no non-base note a near duplicate of any note of the four sets, none at char-3 >= 0.6 to a human-value-01 or -02 note (max 0.5909 / 0.5909), no base note a near duplicate of a human-value-02 note. Disclosed as in run 4: of the 905 base notes 7 are near duplicates of frozen-test
notes and 2 of human-value-01 notes (0 of probe-v1, 0 of human-value-02), and 42 (human-value-01) / 18 (human-value-02, now 200 notes) are at char-3 >= 0.6, a bound applied to non-base notes only. Hashes: `training-v5/manifest.json`.

## 5. Training and export

Recipe unchanged from runs 2-4 (compression-v3 K2048 supervised student, BamiBERT layers 2/5/8/11, FFN 2048, vocab B-rank-8000, positions 32, 40 epochs, batch 8, encoder lr 5e-5, head lr 1e-3, warmup 0.1, wd 0.01, dropout 0.1, grad-clip 1.0, MPS, last epoch kept,
`--retokenize`), seeds 1, 2, 3. Teacher-target cache rebuilt for 1123 notes (frozen teacher `lr5e-05-seed1`; the supervised arm does not read the teacher logits). 5640 steps per run (141 / epoch), final in-sample type / span F1 1.000,
train time 402.0 / 381.4 / 375.0 s, log `train-seeds123.log`. Trainer FP32 test type macro-F1 0.9372 / 0.9561 / 0.9433 (the INT8 scores below are 0.9372 / 0.9413 / 0.9433: seeds 1 and 3 agree, seed 2 loses 0.015 in INT8).

Export (`scripts/export_onnx.py`, `export-seed{1,2,3}.json`): FP32 parity max abs diff 3.8e-5 / 1.7e-5 / 2.8e-5 (limit 1e-4); INT8 vs FP32 type agreement on the 50 export notes 1.00 / 1.00 / 1.00, tag agreement 0.998 / 0.994 / 0.998; INT8 size 28,658,533 B each.
INT8 sha256: seed 1 `00d9a234d9e52646…`, seed 2 `89e3ab8a2fdc2552…`, seed 3 `610fb7d8a9f26937…` (full values in `results.json`). Latency (INT8, 1 thread, 200 runs, 50 test notes): p50 1.39 ms / p95 1.84 ms without the snap, 1.39 / 1.91 with it; bundle 29,287,783 B.

## 6. Selection rule and what was scored

`protocol.json` (written before training; sha256 in `results.json`): **release candidate = seed 1 + word snap**; seeds 1-3 are reported, nothing is selected among them. Success criterion for the real test: end-to-end exact >= 0.95 on human-value-02.
`scripts/evaluate_encoder_retrain_v5.py` ran once on the frozen test, probe-v1 and the amended human-value-01 (it refuses to overwrite `results.json`; debugged only with `--smoke` on in-sample data). The no-snap `old` and `run 2 seed 1` rows were asserted equal to the stored retrain-v2 results on the frozen test
and probe-v1 (human-value-01 cannot be asserted any more: its labels changed); the `run 4 seed 1 + snap` row was asserted equal to the stored run 4 results on the test and probe and on every human-value-01 note except the amended one.
Columns: `old` = deployed `models/gidi-finance-v1` (no snap), `old+snap`, `run 2 s1` (no snap), `run 4 s1+snap` = the previous candidate rescored on the amended labels, `run 5 s1 no snap` (isolates the data effect), `run 5 s1+snap` = **release candidate**, seeds 2 and 3 with the snap, mean ± sd of the three snapped seeds (sample sd).

## 7. Frozen test (105)

| metric | old | old+snap | run 2 s1 | run 4 s1+snap | run 5 s1 no snap | run 5 s1+snap | run 5 s2+snap | run 5 s3+snap | run 5 mean ± sd |
|---|---|---|---|---|---|---|---|---|---|
| type accuracy | 0.9429 | 0.9429 | 0.9238 | 0.9429 | 0.9429 | 0.9429 | 0.9524 | 0.9524 | 0.9492 ± 0.0055 |
| type macro-F1 | 0.9563 | 0.9563 | 0.9219 | 0.9339 | 0.9372 | 0.9372 | 0.9413 | 0.9433 | 0.9406 ± 0.0031 |
| target exact | 0.8857 | 0.9048 | 0.8667 | 0.9238 | 0.9048 | 0.9048 | 0.9429 | 0.9143 | 0.9206 ± 0.0198 |
| span f1 | 0.8077 | 0.8462 | 0.7963 | 0.8846 | 0.8491 | 0.8491 | 0.9020 | 0.8654 | 0.8721 ± 0.0271 |

## 8. probe-v1 (81 complete labels)

| metric | old | old+snap | run 2 s1 | run 4 s1+snap | run 5 s1 no snap | run 5 s1+snap | run 5 s2+snap | run 5 s3+snap | run 5 mean ± sd |
|---|---|---|---|---|---|---|---|---|---|
| type accuracy | 0.7778 | 0.7778 | 0.8025 | 0.8272 | 0.8642 | 0.8642 | 0.8519 | 0.8765 | 0.8642 ± 0.0123 |
| type macro-F1 | 0.7543 | 0.7543 | 0.8136 | 0.8264 | 0.8878 | 0.8878 | 0.8445 | 0.8761 | 0.8695 ± 0.0224 |
| target exact | 0.8148 | 0.8519 | 0.8025 | 0.8519 | 0.8519 | 0.8642 | 0.8642 | 0.8519 | 0.8601 ± 0.0071 |
| span f1 | 0.7818 | 0.8364 | 0.7928 | 0.8393 | 0.8440 | 0.8624 | 0.8654 | 0.8519 | 0.8599 ± 0.0071 |

## 9. human-value-01 (amended labels; 150 labels, 3 `uncertain` excluded, 147 complete scored)

All complete notes:

| metric | old | old+snap | run 2 s1 | run 4 s1+snap | run 5 s1 no snap | run 5 s1+snap | run 5 s2+snap | run 5 s3+snap | run 5 mean ± sd |
|---|---|---|---|---|---|---|---|---|---|
| type accuracy | 0.8435 | 0.8435 | 0.9252 | 0.9592 | 0.9524 | 0.9524 | 0.9592 | 0.9456 | 0.9524 ± 0.0068 |
| type macro-F1 | 0.8049 | 0.8049 | 0.8929 | 0.9411 | 0.9505 | 0.9505 | 0.9701 | 0.9585 | 0.9597 ± 0.0098 |
| target exact | 0.8435 | 0.8571 | 0.9048 | 0.9184 | 0.8980 | 0.9048 | 0.9252 | 0.9252 | 0.9184 ± 0.0118 |
| span f1 | 0.7619 | 0.7891 | 0.8684 | 0.8609 | 0.8289 | 0.8421 | 0.8859 | 0.8874 | 0.8718 ± 0.0257 |
| value exact | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| end-to-end exact | 0.7483 | 0.7551 | 0.8503 | 0.8912 | 0.8707 | 0.8776 | 0.9048 | 0.8844 | 0.8889 ± 0.0142 |

Debt-only slice (n = 13) and the other 134 notes:

| metric | old | old+snap | run 2 s1 | run 4 s1+snap | run 5 s1 no snap | run 5 s1+snap | run 5 s2+snap | run 5 s3+snap | run 5 mean ± sd |
|---|---|---|---|---|---|---|---|---|---|
| type accuracy | 0.0769 | 0.0769 | 0.9231 | 0.9231 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| type macro-F1 | 0.1000 | 0.1000 | 0.9286 | 0.9150 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| target exact | 0.5385 | 0.6154 | 1.0000 | 0.9231 | 0.7692 | 0.7692 | 0.8462 | 0.8462 | 0.8205 ± 0.0444 |
| span f1 | 0.5714 | 0.6667 | 1.0000 | 0.9167 | 0.7500 | 0.7500 | 0.8333 | 0.8333 | 0.8056 ± 0.0481 |
| value exact | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| end-to-end exact | 0.0000 | 0.0000 | 0.9231 | 0.8462 | 0.7692 | 0.7692 | 0.8462 | 0.8462 | 0.8205 ± 0.0444 |

| metric | old | old+snap | run 2 s1 | run 4 s1+snap | run 5 s1 no snap | run 5 s1+snap | run 5 s2+snap | run 5 s3+snap | run 5 mean ± sd |
|---|---|---|---|---|---|---|---|---|---|
| type accuracy | 0.9179 | 0.9179 | 0.9254 | 0.9627 | 0.9478 | 0.9478 | 0.9552 | 0.9403 | 0.9478 ± 0.0075 |
| type macro-F1 | 0.9028 | 0.9028 | 0.8974 | 0.9473 | 0.9474 | 0.9474 | 0.9701 | 0.9570 | 0.9581 ± 0.0114 |
| target exact | 0.8731 | 0.8806 | 0.8955 | 0.9179 | 0.9104 | 0.9179 | 0.9328 | 0.9328 | 0.9279 ± 0.0086 |
| span f1 | 0.7937 | 0.8095 | 0.8438 | 0.8504 | 0.8438 | 0.8594 | 0.8960 | 0.8976 | 0.8843 ± 0.0216 |
| value exact | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| end-to-end exact | 0.8209 | 0.8284 | 0.8433 | 0.8955 | 0.8806 | 0.8881 | 0.9104 | 0.8881 | 0.8955 ± 0.0129 |

Recall by type (correct of n):

| type | n | old | old+snap | run 2 s1 | run 4 s1+snap | run 5 s1 no snap | run 5 s1+snap | run 5 s2+snap | run 5 s3+snap |
|---|---|---|---|---|---|---|---|---|---|
| expense | 29 | 25 | 25 | 25 | 27 | 26 | 26 | 26 | 25 |
| income | 21 | 17 | 17 | 18 | 19 | 18 | 18 | 18 | 19 |
| borrow | 20 | 12 | 12 | 20 | 19 | 20 | 20 | 20 | 20 |
| lend | 20 | 15 | 15 | 18 | 20 | 20 | 20 | 20 | 19 |
| repayment_in | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| repayment_out | 15 | 14 | 14 | 13 | 14 | 14 | 14 | 15 | 14 |
| transfer | 35 | 34 | 34 | 35 | 35 | 35 | 35 | 35 | 35 |
| refund | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |

## 10. Notes still wrong on human-value-01 (seed 1 + snap): 18 of 147

Run 4 seed 1 is wrong on 16 on the amended labels. Run 5 fixes `no tien an cua Vy 90k chua tra` and `tra bot no ban Duy 500` and breaks `rút tiền mặt ở vpbank 1tr5`, `no tien nha chu tro 1tr5 chua tra`, `Quang con thieu 250k tien ve` and `được mẹ cho 1 củ`.
By kind: 11 target-only, 4 type-only, 3 type + target (the value parser is exact on all 147).

| note | gold (type / target) | run 5 s1+snap | kind | run 4 s1+snap right? | run 5 s2 / s3? |
|---|---|---|---|---|---|
| `phu cap an trua 730k` | income / None | expense / phu cap | type+target | no | wrong / wrong |
| `đóng tiền trả góp laptop 2tr3` | repayment_out / None | repayment_in / None | type | no | ok / ok |
| `be chở về nhà 42.000` | expense / be | expense / None | target | no | wrong / wrong |
| `taxi xanh sm ra san bay 210k` | expense / xanh sm | expense / taxi xanh sm | target | no | wrong / wrong |
| `rút tiền mặt ở vpbank 1tr5` | transfer / None | transfer / vpbank | target | yes | wrong / wrong |
| `bach hoa xanh trung sua 87k` | expense / bach hoa xanh | income / hoa xanh | type+target | no | wrong / wrong |
| `no tien nha ba chu 1 thang` (debt) | borrow / ba chu | borrow / nha ba chu | target | no | wrong / wrong |
| `icloud 50gb 19k` | expense / icloud | expense / icloud 50gb | target | no | ok / ok |
| `tien thuong tet 12tr` | income / None | expense / None | type | no | wrong / wrong |
| `no tien nha chu tro 1tr5 chua tra` (debt) | borrow / chu tro | borrow / nha | target | yes | wrong / wrong |
| `go! big c siêu thị 1tr1` | expense / go! big c | expense / go! big | target | no | ok / ok |
| `sua tuoi banh keo bach hoa xanh 96k` | expense / bach hoa xanh | expense / sua | target | no | wrong / ok |
| `mua nửa chỉ vàng cho con 4tr2` | expense / None | transfer / None | type | no | wrong / wrong |
| `khoa hoc online udemy 249k` | expense / udemy | lend / khoa | type+target | no | wrong / wrong |
| `tiền phòng trọ t11 2tr8` | expense / None | expense / phòng trọ | target | no | ok / ok |
| `chi Loan muon 2 trieu` | lend / Loan | lend / chi | target | no | wrong / ok |
| `Quang con thieu 250k tien ve` (debt) | lend / Quang | lend / Quang con | target | yes | ok / ok |
| `được mẹ cho 1 củ` | income / mẹ | borrow / mẹ | type | yes | ok / wrong |

## 11. Reading

- **Gift notes.** `quà sinh nhật Vy 300 nghìn` (now `Vy`) and `mừng cưới Hoa 1 triệu` are right on all three seeds; the amendment moves run 4 seed 1 from 0.8844 to 0.8912 (every run 4 seed predicted `Vy`, which was counted wrong before). No gift note is among the 18 wrong notes.
- **Shop names.** `go! big c siêu thị 1tr1` is right on seeds 2 and 3 and wrong on the candidate (`go! big`); `bach hoa xanh` spans are still wrong on seeds 1 / 2 / 3 (`hoa xanh`, `sua`) although run 4 and 5 both train on `bách hóa xanh` notes. Per-name quota did not fix the multi-word
  whole-word boundary errors on its own.
- **Debt spans got worse.** Type is now 13/13 on the debt slice (run 4: 12/13) but the role-word targets are cut: `chu tro` -> `nha`, `ba chu` -> `nha ba chu`, `Quang` -> `Quang con` (debt end-to-end 10/13 vs 11/13 for run 4 s1; 11/13 on seeds 2 and 3).
- **Type.** probe-v1 type macro-F1 +0.061 (seed 1), borrow / lend recall 20 / 20 each on seed 1; the remaining type errors are `được mẹ cho 1 củ` -> borrow, `tien thuong tet 12tr` -> expense, `mua nửa chỉ vàng cho con` -> transfer, `đóng tiền trả góp laptop` -> repayment_in.
- Frozen test: type macro-F1 0.9372 (old 0.9563), target exact 0.9048 (old 0.8857, snapped 0.9048). Run 4 s1 had 0.9238 target exact; seed 2 of this run has 0.9429, so the spread between seeds (sd 0.0198) is larger than the seed-1 difference.
- Differences of one or two notes of 147 (0.007 / 0.014) are inside the seed spread (sd 0.014); run 5 (mean 0.8889) is not shown to be better or worse than run 4 (mean 0.8844 on the amended labels) on human-value-01, only that the gift / shop data did not lift it. human-value-01 is not a clean held-out estimate: contrast-03 and contrast-04 were
  written knowing its error kinds.

## 12. Verdict and next direction

No release, no deployment; `gidi-finance-v1` stays the encoder. The single scoring on human-value-02 (below) decides whether the candidate is close to the 95% goal. The remaining errors are whole-word boundaries of role words and shop names (`nha chu tro`, `hoa xanh`, `go! big`),
prefix words (`chi Loan`, `phu cap`, `xanh sm`) and a few type confusions; more contrast notes of the same kind did not move them, so the options are a boundary-aware span loss or a word-level (instead of token-level) tag decode; each is a new experiment.

## 13. human-value-02 scoring (prepared, NOT run)

`scripts/score_human_value_02_v5.py` (a copy of `scripts/score_human_value_02.py` as of this run, candidate switched to run 5) refuses to run until `datasets/annotation-v3/human-value-02/labels.jsonl` exists and refuses to overwrite its output. Systems: `candidate` = run 5 seed 1 + snap (its ONNX hash must equal the one in `results.json`),
`candidate_nosnap`, `run4` = run 4 seed 1 + snap (hash checked against run 4's `results.json`), `old`, `run2`, `run2_snap`. Same metrics, the `agreed` subset and the criterion (candidate end-to-end exact >= 0.95).

```sh
uv run python scripts/score_human_value_02_v5.py
```

## 14. Deviations and notes

- **Training was restarted once.** The first launch ran on a `training-v5` built before the test-set agent replaced one note of human-value-02 batch b; the rebuild changed one contrast-04 note (`li xi cho chau Heo 100k` -> `lì xì cháu Bin 100k`), so the cache and the run were discarded
  (nothing was scored) and everything was retrained on the final file. A tool timeout (300 s) also killed early launches; the final run used a detached process (`train-seeds123.log`).
- human-value-01 is now an amended set. `datasets/annotation-v3/training-v1..v4/manifest.json` record the hash of `human-value-01/labels.jsonl` and `freeze.json`; their `--check` went stale, so the four manifests were regenerated by their own builders
  (only those hash lines changed; every `train.jsonl` is byte-identical). Run 4's teacher-target cache records the old training-v4 manifest hash.
- `scripts/build_debt_relabel_queue.py` `DERIVED_DIRS` also skips `gift-relabel-01`, `contrast-04` and `training-v5` (they hold texts of earlier batches); the committed debt-01, contrast-01 / 02 / 03 files are unchanged and their `--check` passes.
- Added `tests/test_build_contrast_04.py` (reproducibility + leakage of amendment-01, the relabel map, contrast-04, training-v5; the per-name quota; gift targets without kinship prefixes; the relabel guard).
- Metric conventions are those of the earlier reports: type macro-F1 averages the classes present in gold; target exact counts null = null as correct; value exact is the (start, end) span; end-to-end exact = type, target and value all right; sd is the sample standard deviation over 3 seeds.
- `experiments/annotation-v3-retrain-v5/.gitignore` ignores the ONNX files and `weights/`; reports, results, protocol and the per-seed export JSON are tracked.
