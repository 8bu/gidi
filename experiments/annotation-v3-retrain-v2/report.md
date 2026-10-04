# annotation-v3 retrain v2 of encoder 1: contrast notes against debt-word confusion

**Verdict: REJECT** — the release candidate (seed 1) drops frozen-test type macro-F1 by 0.0345
(limit 0.01). The four regression notes of the first retrain are fixed on all three seeds and the
debt slice holds (end-to-end 12/13 on every seed), but the type macro-F1 gap to the old encoder is
only partly closed: mean 0.9344 ± 0.0121 (old 0.9563, v3-retrain-v1 0.9119).

Every number below is read from `results.json` (held-out scoring, run once) or from the files
named in each section. Question: does adding LLM-composed contrast notes (the debt words `nợ / trả
/ hoàn / để riêng` in repayment, refund, transfer and expense notes that are **not** debt-only) to
the `training-v1` set keep the debt-only gain of the first retrain
(`experiments/annotation-v3-retrain`) without the frozen-test regression?

## 1. Training data inspection (before composing anything)

`datasets/annotation-v3/training-v1/train.jsonl`, 802 notes. A note "has a debt word" if one of
the whitespace/punctuation tokens of the lowercased text is `nợ`, `no`, `trả`, `tra`, `mượn`,
`muon` or `vay` (exact token, not substring; `no` and `tra` also match the unaccented forms).
The two right-hand blocks are `training-v2` (section 3), shown for comparison.

| type | training-v1 n | with debt word | nợ | no | trả | tra | mượn | muon | vay | training-v2 n | with debt word |
|---|---|---|---|---|---|---|---|---|---|---|---|
| expense | 207 | 5 | 0 | 0 | 1 | 4 | 0 | 0 | 1 | 217 | 15 |
| income | 84 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 84 | 0 |
| borrow | 116 | 108 | 24 | 7 | 14 | 3 | 24 | 23 | 27 | 116 | 108 |
| lend | 111 | 103 | 25 | 1 | 14 | 4 | 27 | 23 | 18 | 111 | 103 |
| repayment_in | 45 | 41 | 7 | 8 | 19 | 16 | 4 | 4 | 1 | 75 | 68 |
| repayment_out | 69 | 54 | 13 | 6 | 25 | 20 | 2 | 0 | 9 | 99 | 81 |
| transfer | 113 | 2 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 131 | 2 |
| refund | 57 | 9 | 0 | 0 | 5 | 4 | 0 | 0 | 0 | 72 | 12 |

Reading: borrow/lend are 227 of 802 notes (28%) and 211 of them carry a debt word; the classes
the first retrain broke carry far fewer: 45 repayment_in (41 with a debt word, only 15 with
`nợ`/`no`), 57 refund (9), 113 transfer (2), 207 expense (5). The model had 79 more notes in which
`nợ`/`trả` mean "a debt exists" and almost none in which they appear next to a payment event,
a refund or a set-aside, so `trả lại`, `đòi nợ` and `để riêng` had little to separate them.
[INFERENCE] This is the hypothesis the contrast notes test; the inspection does not prove it.

## 2. contrast-01: composed notes

`datasets/annotation-v3/contrast-01/` (builder `scripts/build_contrast_01.py`, `--check`
reproduces it byte for byte). `source.jsonl` is the hand-written input of the builder (146
candidates composed by the assistant from `docs/annotation-v1.md`, `docs/annotation-v3.md` and
`configs/annotation-v3.yaml`; none was copied or paraphrased from a frozen-test, probe-v1 or
human-value-01 note; the notes were written from the rules, and the four regression notes were
used only afterwards, as the similarity bound below). Labels
are LLM-made with user approval: `provenance.jsonl` marks every note `annotator: llm`, and the
manifest says so. Every label is `complete`, type + target + value under the annotation-v3 rules,
exact code-point spans (the builder rejects a span that is not on a word boundary), and passes
`scripts/validate_annotations.py --config configs/annotation-v3.quet.yaml`: **103 labels, 0
errors**.

| type | kept | cue it contrasts with a debt-only reading |
|---|---|---|
| repayment_out | 30 | `trả nợ / trả lại / trả tiền mượn X`, `trả góp`, `thanh toán thẻ`, BNPL / `fe credit` |
| repayment_in | 30 | `X trả nợ / trả lại / chuyển trả tiền mượn`, `đòi được nợ`, `thu nợ`, money the user fronted (`mình ứng hộ`) |
| refund | 15 | `hoàn tiền`, `shop hoàn`, `shop trả lại tiền`, cashback |
| transfer | 18 | `để riêng`, `để dành`, `rút`, `chuyển giữa tài khoản`, `gửi tiết kiệm` |
| expense | 10 | `trả tiền điện / ăn / net / phòng`, `trả tiền grab` (a trả-word, no debt) |
| **total** | **103** | 41% unaccented (42 of 103), 33 null targets, about 15 targets with a kinship prefix dropped, `củ`/`ck`/`dc`/`thg`-style slang |

Transfer has 18 notes, not the planned 15: the set-aside / withdraw cues were the class that
regressed, so the quota was raised before building (nothing was scored). Within a type the quota
is filled round-robin over 24 `cue` tags (`provenance.jsonl`), so no single phrasing takes it.

Leakage gate (`build_contrast_01.py`, rules of `scripts/build_human_value_queue.py`: exact, folded,
sequence ≥ 0.9, token Jaccard, char-3-gram, against human-value-01, the frozen test, probe-v1, the
existing corpus under `corpus/` and `datasets/`, and earlier candidates):

| drop reason | candidates |
|---|---|
| near duplicate of the existing corpus | 12 |
| near duplicate of human-value-01 | 5 |
| near duplicate of probe-v1 | 1 |
| near duplicate of the frozen test | 1 |
| within 0.75 folded-difflib ratio of one of the 4 regression notes | 0 |
| over the per-type quota (round-robin by cue) | 24 |
| **dropped / candidates** | **43 / 146** |

Largest similarity of a kept note to a regression note: 0.70 (`trả nợ chủ nhà 3tr` ~ `tra lai chi
Mai 2tr`), under the 0.75 bound. The 19 held-out/corpus drops are listed in
`contrast-01/manifest.json` (`leakage.dropped`).

## 3. training-v2

`datasets/annotation-v3/training-v2/` (builder `scripts/build_annotation_v3_training_v2.py`,
`--check` reproduces it): `train.jsonl` = the 802 byte-identical `training-v1` records + the 103
contrast notes (`source_batch: contrast-01`, `provenance.annotator: llm`) = **905**;
`validation.jsonl` and `test.jsonl` are the same symlinks to the frozen annotation-v1 splits.
Types: expense 217, income 84, borrow 116, lend 111, repayment_in 75, repayment_out 99,
transfer 131, refund 72. Annotators: human 256, ai 546 (the distillation-v1 labels), llm 103.
Leakage asserted (raises, writes nothing): no train id and no verbatim or NFC-lowercase text in
the frozen test (105), probe-v1 (81) or human-value-01 (150), the 9 debt-01 ids excluded from
human-value-01 absent, and no contrast note is a near duplicate of any note of those three sets.
Hashes of every input and `train.jsonl`: `training-v2/manifest.json`.

## 4. Training and export

Recipe unchanged from the first retrain (compression-v3 K2048 supervised student, BamiBERT layers
2/5/8/11, FFN 2048, vocab B-rank-8000, positions 32, 40 epochs, batch 8, encoder lr 5e-5, head lr
1e-3, warmup 0.1, wd 0.01, dropout 0.1, grad-clip 1.0, MPS, last epoch kept, `--retokenize`), no
hyperparameter changed, seeds 1, 2 and 3. Teacher-target cache rebuilt for 905 notes
(`teacher-targets/`, frozen teacher `lr5e-05-seed1`; the supervised arm does not read the teacher
logits). Runs: `runs/supervised/…K2048/seed{1,2,3}/`, log `train-seeds123.log`. Each run: 4560
steps, final in-sample type/span F1 1.000, train time 300.9 / 299.5 / 299.1 s (7.5 s/epoch); the
whole training command took 915 s. Trainer FP32 test type macro-F1: 0.9219 / 0.9354 / 0.9459
(equal to the INT8 scores below).

Export (`scripts/export_onnx.py`, `export-seed{1,2,3}.json`): FP32 parity max abs diff ≤ 2.3e-5,
INT8 vs FP32 type agreement 1.00 on the 50 export notes for every seed; INT8 size 28,658,533 B
(identical to the old encoder: same architecture, so latency is unchanged and was not re-measured).
INT8 sha256: seed 1 `7e884b8816d42fe5…`, seed 2 `9f747968a31e4fac…`, seed 3 `addf16acf7738b59…`
(full values in `results.json`).

## 5. Selection rule (fixed before scoring)

> Fixed before any seed was scored. Gate (G) on the frozen test and human-value-01, old = the deployed encoder models/gidi-finance-v1: (a) test type macro-F1 drop <= 0.01, (b) test target exact drop <= 0.02, (c) human-value-01 debt-only slice (n=13, encoder + unchanged parser) end-to-end exact >= 0.8. The release candidate is seed 1 (deployment convention) unless seed 1 fails G while the three-seed mean passes G; then it is the seed whose test type macro-F1 is the median of the three. Verdict ACCEPT iff the release candidate passes G, else REJECT (a mean that passes with a failing candidate is reported, not accepted).

The rule was written into `scripts/evaluate_encoder_retrain_v2.py` and the evaluator smoke-tested
on in-sample data before any held-out note was scored; the held-out scoring ran once (the script
refuses to overwrite `results.json`).

## 6. Frozen test (105) — per seed and mean ± sd

Columns: old = `models/gidi-finance-v1` INT8, v3-retrain-v1 = first retrain (one seed, reused
from `experiments/annotation-v3-retrain/results.json`), v2 = this experiment.

| metric | old | v3-retrain-v1 | v2 seed 1 | v2 seed 2 | v2 seed 3 | v2 mean ± sd | mean − old |
|---|---|---|---|---|---|---|---|
| type accuracy | 0.9429 | 0.9048 | 0.9238 | 0.9333 | 0.9333 | 0.9302 ± 0.0055 | -0.0127 |
| type macro f1 | 0.9563 | 0.9119 | 0.9219 | 0.9354 | 0.9459 | 0.9344 ± 0.0121 | -0.0219 |
| target exact | 0.8857 | 0.8952 | 0.8667 | 0.8667 | 0.8952 | 0.8762 ± 0.0165 | -0.0095 |
| span f1 | 0.8077 | 0.8302 | 0.7963 | 0.7963 | 0.8302 | 0.8076 ± 0.0196 | -0.0001 |

Per-class F1 (frozen test):

| type | old | v3-retrain-v1 | seed 1 | seed 2 | seed 3 | mean ± sd |
|---|---|---|---|---|---|---|
| expense | 0.9180 | 0.8889 | 0.9355 | 0.9355 | 0.9000 | 0.9237 ± 0.0205 |
| income | 0.8571 | 0.8182 | 0.8182 | 0.8182 | 0.8696 | 0.8353 ± 0.0297 |
| borrow | 1.0000 | 1.0000 | 0.8235 | 0.8889 | 0.9412 | 0.8845 ± 0.0589 |
| lend | 1.0000 | 0.9474 | 0.9412 | 0.9412 | 1.0000 | 0.9608 ± 0.0340 |
| repayment_in | 1.0000 | 0.8571 | 1.0000 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| repayment_out | 0.9524 | 0.9474 | 0.9091 | 0.9524 | 0.9091 | 0.9235 ± 0.0250 |
| transfer | 0.9231 | 0.8947 | 0.9474 | 0.9474 | 0.9474 | 0.9474 ± 0.0000 |
| refund | 1.0000 | 0.9412 | 1.0000 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |

| note | gold (type / target) | old | v3-retrain-v1 | v2 seed 1 | v2 seed 2 | v2 seed 3 |
|---|---|---|---|---|---|---|
| `tra lai chi Mai 2tr` | repayment_out / Mai | repayment_out / Mai | expense / Mai ✗ | repayment_out / Mai | repayment_out / Mai | repayment_out / Mai |
| `de rieng tien sua xe 800k` | transfer / None | transfer / None | expense / None ✗ | transfer / None | transfer / None | transfer / None |
| `Nhi ck tra lai 120k tien sua` | repayment_in / Nhi | repayment_in / Nhi | refund / Nhi ✗ | repayment_in / Nhi | repayment_in / Nhi | repayment_in / Nhi |
| `đòi được nợ thằng Lâm 400k` | repayment_in / Lâm | repayment_in / Lâm | lend / Lâm ✗ | repayment_in / Lâm | repayment_in / Lâm | repayment_in / Lâm |


Seed 3 fixes the four regression notes and ends 0.0104 below the old encoder, 0.0004 over the
bound; seeds 1 and 2 end 0.0345 and 0.0209 below.

Seed 1: notes whose type or target prediction differs from the old encoder (17 of 105):

| id | note | gold | old | seed 1 |
|---|---|---|---|---|
| dcc75d1e | mua giay the thao 1tr25 | expense / None | transfer / thao ✗ | repayment_out / thao ✗ |
| 4c255202 | mua hoa qua biếu bà 200k | expense / None | expense / None | expense / bà ✗ |
| cce877fb | hủ tiếu tối qua 35k | expense / None | expense / None | expense / tiếu ✗ |
| 5b6adcc4 | qua sinh nhat chi Ha 350k | expense / Ha | expense / Ha | expense / chi Ha ✗ |
| 40164a2e | tien phong thang 11 3tr2 | expense / None | transfer / None ✗ | expense / None |
| bca50f8d | tien chay grab hom nay 450k | income / grab | expense / None ✗ | expense / grab ✗ |
| d560ebae | coopmart đồ dùng tuần 412k | expense / coopmart | repayment_out / coopmart ✗ | income / coopmart ✗ |
| 6245d7af | vay chú Hải 20tr mua xe | borrow / Hải | borrow / chú ✗ | borrow / chú Hải ✗ |
| 808256ca | nhan tien viet bai 600 nghin | income / None | income / None | income / viet ✗ |
| 040c0585 | starbucks 1 ly 89,000 | expense / starbucks | expense / starbuck ✗ | expense / starb ✗ |
| e33f35c6 | dì Tư gửi trả 3 triệu đã mượn | repayment_in / dì Tư | repayment_in / dì Tư | repayment_in / ì Tư ✗ |
| 6b6eb45c | bảo việt bồi thường viện phí 2tr6 | refund / bảo việt | refund / ảo ✗ | refund / bảo việt |
| d00f5a29 | vay home credit giai ngan 15tr | borrow / home credit | borrow / home credit | repayment_out / home credit ✗ |
| 1d910520 | vay ngân hàng mua nhà giải ngân 800tr | borrow / ngân hàng | borrow / hàng ✗ | borrow / ngân hàng |
| 37648087 | roommate tra tien nha minh dong truoc 1tr5 | repayment_in / roommate | repayment_in / ro ✗ | repayment_in / ommate ✗ |
| 105851cc | ứng 2tr | borrow / None | borrow / None | transfer / None ✗ |
| 04d4944f | ck cho Vũ mượn 700k | lend / Vũ | lend / Vũ | borrow / Vũ ✗ |

Type regressions (old right, new wrong): 3; type fixes: 1.

Seed 2: notes whose type or target prediction differs from the old encoder (18 of 105):

| id | note | gold | old | seed 2 |
|---|---|---|---|---|
| dcc75d1e | mua giay the thao 1tr25 | expense / None | transfer / thao ✗ | expense / thao ✗ |
| a362f627 | ban may anh cu 6tr5 | income / None | income / None | income / anh cu ✗ |
| 4c255202 | mua hoa qua biếu bà 200k | expense / None | expense / None | expense / bà ✗ |
| cce877fb | hủ tiếu tối qua 35k | expense / None | expense / None | expense / tiếu ✗ |
| 9c0ccb74 | vay tiêu dùng trả kỳ này 3tr4 | repayment_out / None | repayment_out / tiêu ✗ | repayment_out / None |
| bca50f8d | tien chay grab hom nay 450k | income / grab | expense / None ✗ | expense / grab ✗ |
| 0e980666 | bách hoá xanh rau thịt 187k | expense / bách hoá xanh | expense / bách hoá ✗ | expense / bách ✗ |
| d560ebae | coopmart đồ dùng tuần 412k | expense / coopmart | repayment_out / coopmart ✗ | income / coopmart ✗ |
| 6245d7af | vay chú Hải 20tr mua xe | borrow / Hải | borrow / chú ✗ | borrow / chú Hải ✗ |
| 808256ca | nhan tien viet bai 600 nghin | income / None | income / None | income / et ✗ |
| 040c0585 | starbucks 1 ly 89,000 | expense / starbucks | expense / starbuck ✗ | expense / starb ✗ |
| 90f0205f | mừng thọ bà nội 1 triệu | expense / bà nội | expense / bà nội | expense / thọ ✗ |
| e33f35c6 | dì Tư gửi trả 3 triệu đã mượn | repayment_in / dì Tư | repayment_in / dì Tư | repayment_in / Tư ✗ |
| 6b6eb45c | bảo việt bồi thường viện phí 2tr6 | refund / bảo việt | refund / ảo ✗ | refund / bảo việt |
| d00f5a29 | vay home credit giai ngan 15tr | borrow / home credit | borrow / home credit | repayment_out / home credit ✗ |
| 1d910520 | vay ngân hàng mua nhà giải ngân 800tr | borrow / ngân hàng | borrow / hàng ✗ | borrow / ngân hàng |
| 37648087 | roommate tra tien nha minh dong truoc 1tr5 | repayment_in / roommate | repayment_in / ro ✗ | repayment_in / omm ✗ |
| 04d4944f | ck cho Vũ mượn 700k | lend / Vũ | lend / Vũ | borrow / Vũ ✗ |

Type regressions (old right, new wrong): 2; type fixes: 1.

Seed 3: notes whose type or target prediction differs from the old encoder (14 of 105):

| id | note | gold | old | seed 3 |
|---|---|---|---|---|
| dcc75d1e | mua giay the thao 1tr25 | expense / None | transfer / thao ✗ | repayment_out / thao ✗ |
| dc138256 | bố cho tiền tiêu 1 triệu | income / bố | transfer / bố ✗ | income / bố |
| 4c255202 | mua hoa qua biếu bà 200k | expense / None | expense / None | expense / bà ✗ |
| cce877fb | hủ tiếu tối qua 35k | expense / None | expense / None | transfer / tiếu ✗ |
| 9c0ccb74 | vay tiêu dùng trả kỳ này 3tr4 | repayment_out / None | repayment_out / tiêu ✗ | repayment_out / None |
| bca50f8d | tien chay grab hom nay 450k | income / grab | expense / None ✗ | expense / grab ✗ |
| 0e980666 | bách hoá xanh rau thịt 187k | expense / bách hoá xanh | expense / bách hoá ✗ | expense / ách hoá ✗ |
| d560ebae | coopmart đồ dùng tuần 412k | expense / coopmart | repayment_out / coopmart ✗ | income / coopmart ✗ |
| 6245d7af | vay chú Hải 20tr mua xe | borrow / Hải | borrow / chú ✗ | borrow / chú Hải ✗ |
| 040c0585 | starbucks 1 ly 89,000 | expense / starbucks | expense / starbuck ✗ | expense / starbucks |
| e33f35c6 | dì Tư gửi trả 3 triệu đã mượn | repayment_in / dì Tư | repayment_in / dì Tư | repayment_in / dì ✗ |
| 6b6eb45c | bảo việt bồi thường viện phí 2tr6 | refund / bảo việt | refund / ảo ✗ | refund / bảo ✗ |
| d00f5a29 | vay home credit giai ngan 15tr | borrow / home credit | borrow / home credit | repayment_out / home credit ✗ |
| 1d910520 | vay ngân hàng mua nhà giải ngân 800tr | borrow / ngân hàng | borrow / hàng ✗ | borrow / ngân hàng |

Type regressions (old right, new wrong): 2; type fixes: 1.


## 7. probe-v1 (81 complete labels)

| metric | old | v3-retrain-v1 | v2 seed 1 | v2 seed 2 | v2 seed 3 | v2 mean ± sd | mean − old |
|---|---|---|---|---|---|---|---|
| type accuracy | 0.7778 | 0.8148 | 0.8025 | 0.7901 | 0.8148 | 0.8025 ± 0.0123 | +0.0247 |
| type macro f1 | 0.7543 | 0.7796 | 0.8136 | 0.8062 | 0.8466 | 0.8222 ± 0.0215 | +0.0678 |
| target exact | 0.8148 | 0.8148 | 0.8025 | 0.8272 | 0.8148 | 0.8148 ± 0.0123 | +0.0000 |
| span f1 | 0.7818 | 0.7890 | 0.7928 | 0.8108 | 0.7963 | 0.8000 ± 0.0096 | +0.0181 |

Per-class F1 (probe-v1):

| type | old | v3-retrain-v1 | seed 1 | seed 2 | seed 3 | mean ± sd |
|---|---|---|---|---|---|---|
| expense | 0.8235 | 0.8485 | 0.8485 | 0.8750 | 0.8485 | 0.8573 ± 0.0153 |
| income | 0.8235 | 0.8889 | 0.8889 | 0.8571 | 0.8889 | 0.8783 ± 0.0183 |
| borrow | 0.6667 | 0.8000 | 0.7692 | 0.7143 | 0.7407 | 0.7414 ± 0.0275 |
| lend | 0.8000 | 0.8182 | 0.7500 | 0.7500 | 0.7826 | 0.7609 ± 0.0188 |
| repayment_in | 0.5000 | 0.5000 | 0.8000 | 0.8000 | 1.0000 | 0.8667 ± 0.1155 |
| repayment_out | 0.8889 | 0.8235 | 0.8889 | 0.8235 | 0.8421 | 0.8515 ± 0.0337 |
| transfer | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 ± 0.0000 |
| refund | 0.7778 | 0.7778 | 0.7500 | 0.8235 | 0.8235 | 0.7990 ± 0.0425 |


probe-v1 is not part of the gate. It improves on every seed (type macro-F1 +0.0678 mean over the
old encoder); `transfer` stays at 0.0 F1 as for the old encoder and the first retrain.

## 8. human-value-01 (150 labels; 3 `uncertain` excluded, 147 complete scored)

V7 (`gidi-finance-v2` with its own value head) and V8-old (`gidi-finance-v1` INT8 + the unchanged
rule value parser) are reused from `experiments/annotation-v3-retrain/results.json`, not rerun,
as is the first retrain; the v2 columns are the retrained INT8 encoders + the same parser. The
debt-only slice is the same 13 notes (`results.json: debt_only_ids`).

**All complete notes (n=147)**

| metric | V7 | V8-old | v3-retrain-v1 | v2 seed 1 | v2 seed 2 | v2 seed 3 | v2 mean ± sd |
|---|---|---|---|---|---|---|---|
| type accuracy | 0.8435 | 0.8435 | 0.9320 | 0.9252 | 0.9252 | 0.9184 | 0.9229 ± 0.0039 |
| type macro f1 | 0.8049 | 0.8049 | 0.9097 | 0.8929 | 0.9051 | 0.9015 | 0.8999 ± 0.0063 |
| target exact | 0.8367 | 0.8367 | 0.8639 | 0.8980 | 0.8912 | 0.8912 | 0.8934 ± 0.0039 |
| type target exact | 0.7415 | 0.7415 | 0.8231 | 0.8435 | 0.8435 | 0.8367 | 0.8413 ± 0.0039 |
| value exact | 0.9864 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| end to end exact | 0.7347 | 0.7415 | 0.8231 | 0.8435 | 0.8435 | 0.8367 | 0.8413 ± 0.0039 |

**Debt-only slice (n=13)**

| metric | V7 | V8-old | v3-retrain-v1 | v2 seed 1 | v2 seed 2 | v2 seed 3 | v2 mean ± sd |
|---|---|---|---|---|---|---|---|
| type accuracy | 0.0769 | 0.0769 | 0.9231 | 0.9231 | 0.9231 | 0.9231 | 0.9231 ± 0.0000 |
| type macro f1 | 0.1000 | 0.1000 | 0.9286 | 0.9286 | 0.9286 | 0.9150 | 0.9241 ± 0.0078 |
| target exact | 0.5385 | 0.5385 | 0.8462 | 1.0000 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| type target exact | 0.0000 | 0.0000 | 0.8462 | 0.9231 | 0.9231 | 0.9231 | 0.9231 ± 0.0000 |
| value exact | 0.9231 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| end to end exact | 0.0000 | 0.0000 | 0.8462 | 0.9231 | 0.9231 | 0.9231 | 0.9231 ± 0.0000 |

**Debt-only, annotation-v1 `skipped` sub-slice (n=8)**

| metric | V7 | V8-old | v3-retrain-v1 | v2 seed 1 | v2 seed 2 | v2 seed 3 | v2 mean ± sd |
|---|---|---|---|---|---|---|---|
| type accuracy | 0.1250 | 0.1250 | 0.8750 | 0.8750 | 0.8750 | 0.8750 | 0.8750 ± 0.0000 |
| type macro f1 | 0.1667 | 0.1667 | 0.9000 | 0.9000 | 0.9000 | 0.8730 | 0.8910 ± 0.0156 |
| target exact | 0.5000 | 0.5000 | 0.8750 | 1.0000 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| type target exact | 0.0000 | 0.0000 | 0.8750 | 0.8750 | 0.8750 | 0.8750 | 0.8750 ± 0.0000 |
| value exact | 0.8750 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| end to end exact | 0.0000 | 0.0000 | 0.8750 | 0.8750 | 0.8750 | 0.8750 | 0.8750 ± 0.0000 |

**Non-debt notes (n=134)**

| metric | V7 | V8-old | v3-retrain-v1 | v2 seed 1 | v2 seed 2 | v2 seed 3 | v2 mean ± sd |
|---|---|---|---|---|---|---|---|
| type accuracy | 0.9179 | 0.9179 | 0.9328 | 0.9254 | 0.9254 | 0.9179 | 0.9229 ± 0.0043 |
| type macro f1 | 0.9028 | 0.9028 | 0.9143 | 0.8974 | 0.9088 | 0.9070 | 0.9044 ± 0.0061 |
| target exact | 0.8657 | 0.8657 | 0.8657 | 0.8881 | 0.8806 | 0.8806 | 0.8831 ± 0.0043 |
| type target exact | 0.8134 | 0.8134 | 0.8209 | 0.8358 | 0.8358 | 0.8284 | 0.8333 ± 0.0043 |
| value exact | 0.9925 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 ± 0.0000 |
| end to end exact | 0.8060 | 0.8134 | 0.8209 | 0.8358 | 0.8358 | 0.8284 | 0.8333 ± 0.0043 |


Borrow/lend confusion on all complete notes (gold row → predicted):

seed 1 all-complete gold borrow: {'borrow': 20}, gold lend: {'lend': 18, 'repayment_out': 1, 'expense': 1}

seed 2 all-complete gold borrow: {'borrow': 20}, gold lend: {'lend': 18, 'expense': 2}

seed 3 all-complete gold borrow: {'borrow': 19, 'lend': 1}, gold lend: {'lend': 19, 'expense': 1}


## 9. Gate and verdict

| candidate | test type macro-F1 drop (≤ 0.01) | test target exact drop (≤ 0.02) | human-value-01 debt e2e (≥ 0.8) | gate |
|---|---|---|---|---|
| seed 1 | +0.0345 (FAIL) | +0.0190 (pass) | 0.9231 (pass) | FAIL |
| seed 2 | +0.0209 (FAIL) | +0.0190 (pass) | 0.9231 (pass) | FAIL |
| seed 3 | +0.0104 (FAIL) | -0.0095 (pass) | 0.9231 (pass) | FAIL |
| mean of 3 seeds | +0.0219 (FAIL) | +0.0095 (pass) | 0.9231 (pass) | FAIL |


Release candidate: **seed 1** (deployment convention; it fails the gate and so does the
three-seed mean, so the median-seed fallback of the rule does not apply). Seed 1 fails the first
criterion (0.0345 > 0.01), so **Verdict: REJECT**. `gidi-finance-v1` stays the encoder; nothing
is deployed or published.

What the numbers say, beyond the rule (context, not a change of the rule):

- **The targeted failure is fixed.** All four regression notes (`tra lai chi Mai 2tr`, `de rieng
  tien sua xe 800k`, `Nhi ck tra lai 120k tien sua`, `đòi được nợ thằng Lâm 400k`) have the right
  type on all three seeds (section 6), repayment_in F1 is 1.0000 on every seed, refund 1.0000,
  transfer 0.9474. The debt-only gain is kept: debt slice type accuracy 0.9231 and end-to-end
  0.9231 (12/13) on every seed, against 0.0000 for V7/V8-old and 0.8462 for the first retrain.
- **The remaining test loss moved, it did not vanish.** The notes the new encoders get wrong
  that the old one gets right are 3 / 2 / 2 (seeds 1 / 2 / 3; the first retrain had 4), against 1
  fix each. `vay home credit giai ngan 15tr` (borrow → repayment_out) is wrong on all three seeds;
  `ck cho Vũ mượn 700k` (lend → borrow) on seeds 1 and 2; `ứng 2tr` (borrow → transfer) on seed 1;
  `hủ tiếu tối qua 35k` (expense → transfer) on seed 3. [INFERENCE] The first looks caused by the
  contrast data: three new repayment_out notes pair a lender name with a financing word
  (`góp tủ lạnh fe credit`, `trả vay fe credit`, `tất toán khoản vay home credit`), so
  `vay home credit giải ngân` may now be read as a repayment. This was not isolated by an ablation.
- **Seed noise is of the size of the gap.** The old data has a 3-seed test type macro-F1 of
  0.943 ± 0.018 (`experiments/compression-v3/report.md`, K2048 column); the deployed seed 1
  (0.9563) is above that mean. The v2 three-seed mean 0.9344 ± 0.0121 is 0.009 below the old
  3-seed mean and 0.022 below the old seed 1 that the gate compares against. The test has 105
  notes, so one note is ±0.0095 accuracy. The rule compares to the best-looking old seed, and
  nothing was re-selected to get around it.
- Target exact on the test: 0.8667 / 0.8667 / 0.8952 vs 0.8857 old (drops 0.0190 / 0.0190 /
  −0.0095, all within 0.02). human-value-01 all-complete end-to-end: 0.8435 / 0.8435 / 0.8367 vs
  0.7415 (V8-old) and 0.8231 (first retrain); non-debt notes do not regress.

## 10. Deviations and notes

- **`scripts/build_debt_relabel_queue.py`** treated every other jsonl under `datasets/` as a leakage
  reference. It now skips `training-v1`, `contrast-01` and `training-v2` (`DERIVED_DIRS`), the sets
  built from the batch, so `--check` of `debt-01` stays valid; the committed debt-01 files are
  unchanged and `--check` passes.
- `contrast-01` is judged by the same near-duplicate rules as debt-01; its leakage reference
  groups include the existing corpus, so its 12 corpus drops are notes too close to an existing
  training or corpus note.
- Training was first launched with a 300 s tool deadline and was killed in epoch 38 of seed 1;
  it was relaunched from scratch (the partial run directory was deleted), so every reported run is
  a complete 40-epoch run. No result of the killed run was seen or used.
- MPS kernels are not bit-reproducible; a rerun of a seed may differ slightly (the trainer warns).
- The old-encoder row is recomputed in `results.json` and asserted equal to the stored first-retrain
  result to the last digit before the reused rows were accepted.
- `experiments/annotation-v3-retrain-v2/.gitignore` ignores the ONNX files (140 MB per seed) and
  `weights/`; reports, results and the per-seed export JSON are tracked.
- Metric conventions are those of the first report: type macro-F1 averages the classes present in
  gold (`gidi.evaluation.metrics`); target exact counts null = null as correct; value exact is the
  (start, end) span; end-to-end exact = type, target and value all right; sd is the sample
  standard deviation over 3 seeds.
