# annotation-v3 retrain of encoder 1 (type + target)

**Verdict: REJECT** — frozen-test type macro-F1 drops 0.0445 (> 0.01).

Every number below is read from `results.json` (held-out scoring, run once) or from the run
files named in section 2. Question: does retraining encoder 1 with the annotation-v3 rule
(a debt-only note is `borrow` when the user owes and `lend` when the other party owes, instead
of `skipped`) fix debt-only notes without hurting the rest? Candidate **V8-new** = retrained
encoder INT8 + the unchanged rule value parser (`src/gidi/value_parser`).

## 1. Setup

| item | value |
|---|---|
| recipe | compression-v3 K2048 supervised student, exactly as the deployed encoder: BamiBERT layers 2/5/8/11, FFN 2048 (`combined` map), vocab spec B-rank-8000 (8338 rows), positions truncated to 32, no KD, 40 epochs, batch 8, encoder lr 5e-5, head lr 1e-3, warmup 0.1, wd 0.01, dropout 0.1, grad-clip 1.0, seed 1, MPS, last epoch kept |
| train data | `datasets/annotation-v3/training-v1/train.jsonl`, 802 notes |
| validation / test | the frozen annotation-v1 splits (validation in-sample, test held out), as in distillation-v1 |
| old encoder | `models/gidi-finance-v1/model.int8.onnx`, sha256 `a31caa5764c5ba99…` |
| new encoder | `experiments/annotation-v3-retrain/onnx/model.int8.onnx`, sha256 `aede21a6b95d709e6511bc3b8904887d3880e55816648a05a9ca7d42b67e5a0b` |
| V7 | `models/gidi-finance-v2` INT8 (encoder + its own CRF value head), sha256 `bc7945c3a7ff1d43…` |
| value for V8-old / V8-new | `gidi.value_parser.parse_value`, identical for both |
| inference | `GidiPredictor` (1 thread); old and new share the `gidi-finance-v1` bundle config, tokenizer and decoding, only the ONNX file differs |
| held-out sets (never trained on) | frozen test (105), probe-v1 (81 complete labels), human-value-01 (150 labels, annotation-v3 debt rule) |

V7 and V8-old were trained on the annotation-v1 rule (debt-only = `skipped`, never a trained
class), so their debt-only slice is **expected to fail**. This is not a defect of the parser.

## 2. Data and training

`scripts/build_annotation_v3_training.py` (`--check` reproduces it; the leakage gate raises on
any held-out id or exact / NFC-lowercase text) builds `training-v1`:

| source | notes |
|---|---|
| `distillation-v1/train.jsonl` (byte-identical prefix) | 723 |
| `debt-01` `complete` human labels | 79 (14 old annotation-v1 `skipped` notes + 65 new notes) |
| `debt-01` `uncertain` labels (not trainable) | 4, dropped |
| **train** | **802** |

Types: borrow 79 → 116, lend 69 → 111, the other six types unchanged (expense 207, income 84,
repayment_in 45, repayment_out 69, transfer 113, refund 57). The 9 human-value-01 ids named in
the debt-01 manifest are asserted absent; no train id or text occurs in the frozen test,
probe-v1 or human-value-01. Hashes of every input and output: `datasets/annotation-v3/training-v1/manifest.json`.

Run: `experiments/annotation-v3-retrain/runs/supervised/student-4x768-pretrained-pos32-vocab-B-rank-8000-ffn-combined-K2048/seed1/`
(`config.json`, `train_log.jsonl`, `metrics.json`, `checkpoint.json`), log `train-seed1.log`.
Training completed all 40 epochs (4040 steps, final in-sample type/span F1 1.000):
**wall time 273.4 s** (train 271.6 s, 6.79 s/epoch, 118 notes/s; the old run took 242.7 s).
PyTorch FP32 test metrics from the trainer: type macro-F1 0.9171, type accuracy 0.9143.

Export (`scripts/export_onnx.py`, `export-seed1.json`): FP32 `model.onnx` 114,024,066 B sha256
`c184501e…8b98` (parity max abs diff 2.2e-5), INT8 `model.int8.onnx` 28,658,533 B sha256
`aede21a6b95d709e6511bc3b8904887d3880e55816648a05a9ca7d42b67e5a0b`; INT8 vs FP32 type
agreement 0.98 on the 50 export notes (old export: 0.96).

## 3. Frozen test (105) — non-regression check

| metric | old | new | new - old |
|---|---|---|---|
| type accuracy | 0.9429 | 0.9048 | -0.0381 |
| type macro-F1 | 0.9563 | 0.9119 | -0.0445 |
| target exact | 0.8857 | 0.8952 | +0.0095 |
| span F1 | 0.8077 | 0.8302 | +0.0225 |

Per-class F1:

| type | old F1 | new F1 |
|---|---|---|
| expense | 0.9180 | 0.8889 |
| income | 0.8571 | 0.8182 |
| borrow | 1.0000 | 1.0000 |
| lend | 1.0000 | 0.9474 |
| repayment_in | 1.0000 | 0.8571 |
| repayment_out | 0.9524 | 0.9474 |
| transfer | 0.9231 | 0.8947 |
| refund | 1.0000 | 0.9412 |

Notes whose type or target prediction changed (15 of 105):

| id | note | gold (type / target) | old (type / target) | new (type / target) |
|---|---|---|---|---|
| cce877fb | hủ tiếu tối qua 35k | expense / None | expense / None | expense / tiếu ✗ |
| dd3df660 | tra lai chi Mai 2tr | repayment_out / Mai | repayment_out / Mai | expense / Mai ✗ |
| 9c0ccb74 | vay tiêu dùng trả kỳ này 3tr4 | repayment_out / None | repayment_out / tiêu ✗ | repayment_out / None |
| bca50f8d | tien chay grab hom nay 450k | income / grab | expense / None ✗ | expense / grab ✗ |
| 1f56011d | de rieng tien sua xe 800k | transfer / None | transfer / None | expense / None ✗ |
| 0e980666 | bách hoá xanh rau thịt 187k | expense / bách hoá xanh | expense / bách hoá ✗ | expense / bách ✗ |
| d560ebae | coopmart đồ dùng tuần 412k | expense / coopmart | repayment_out / coopmart ✗ | income / coopmart ✗ |
| 6245d7af | vay chú Hải 20tr mua xe | borrow / Hải | borrow / chú ✗ | borrow / chú Hải ✗ |
| 808256ca | nhan tien viet bai 600 nghin | income / None | income / None | income / et ✗ |
| 6b6eb45c | bảo việt bồi thường viện phí 2tr6 | refund / bảo việt | refund / ảo ✗ | refund / bảo ✗ |
| 2e5a18da | Nhi ck tra lai 120k tien sua | repayment_in / Nhi | repayment_in / Nhi | refund / Nhi ✗ |
| 1d910520 | vay ngân hàng mua nhà giải ngân 800tr | borrow / ngân hàng | borrow / hàng ✗ | borrow / ngân ✗ |
| 37648087 | roommate tra tien nha minh dong truoc 1tr5 | repayment_in / roommate | repayment_in / ro ✗ | repayment_in / ommate ✗ |
| 1e618b99 | co Thuy cho muon 1tr | borrow / Thuy | borrow / co Thuy ✗ | borrow / Thuy |
| b78854de | đòi được nợ thằng Lâm 400k | repayment_in / Lâm | repayment_in / Lâm | lend / Lâm ✗ |

Type regressions (old right, new wrong): 4; type fixes: 0.

- `tra lai chi Mai 2tr`: gold repayment_out, old repayment_out, new expense
- `de rieng tien sua xe 800k`: gold transfer, old transfer, new expense
- `Nhi ck tra lai 120k tien sua`: gold repayment_in, old repayment_in, new refund
- `đòi được nợ thằng Lâm 400k`: gold repayment_in, old repayment_in, new lend

Type fixes:

- none

Target regressions (old exact, new not): 2; target fixes: 3.

- `hủ tiếu tối qua 35k`: gold target None, old None, new 'tiếu'
- `nhan tien viet bai 600 nghin`: gold target None, old None, new 'et'

Target fixes:

- `vay tiêu dùng trả kỳ này 3tr4`: gold target None, old 'tiêu', new None
- `tien chay grab hom nay 450k`: gold target 'grab', old None, new 'grab'
- `co Thuy cho muon 1tr`: gold target 'Thuy', old 'co Thuy', new 'Thuy'

## 4. probe-v1 (81 complete labels)

| metric | old | new | new - old |
|---|---|---|---|
| type accuracy | 0.7778 | 0.8148 | +0.0370 |
| type macro-F1 | 0.7543 | 0.7796 | +0.0252 |
| target exact | 0.8148 | 0.8148 | +0.0000 |
| span F1 | 0.7818 | 0.7890 | +0.0072 |

Per-class F1:

| type | old F1 | new F1 |
|---|---|---|
| expense | 0.8235 | 0.8485 |
| income | 0.8235 | 0.8889 |
| borrow | 0.6667 | 0.8000 |
| lend | 0.8000 | 0.8182 |
| repayment_in | 0.5000 | 0.5000 |
| repayment_out | 0.8889 | 0.8235 |
| transfer | 0.0000 | 0.0000 |
| refund | 0.7778 | 0.7778 |

Notes whose prediction changed (17 of 81):

| id | note | gold (type / target) | old (type / target) | new (type / target) |
|---|---|---|---|---|
| 7bb2d6c9 | thg Khoa cho muon tam 300k an trua | borrow / Khoa | lend / Khoa ✗ | borrow / Khoa |
| ba9bbf3b | me cho muon 1tr5 tra tien dien | borrow / me | lend / me ✗ | borrow / me |
| 837cccab | được cô Hà cho vay 4 củ | borrow / Hà | lend / Hà ✗ | borrow / Hà |
| 9f9383c4 | Phúc vay 700k mua điện thoại | lend / Phúc | lend / Phúc | borrow / Phúc ✗ |
| b4ca6cdd | ông nội cho 1 triệu mua sách | income / ông nội | borrow / ông nội ✗ | income / ông nội |
| 37cfc9f2 | được cô Út cho 500k | income / cô Út | expense / cô Út ✗ | borrow / cô Út ✗ |
| ec3782f5 | dong bh y te cho con 700k | expense / None | expense / dong ✗ | expense / None |
| 1d4ef0b7 | phí bh aia tháng 12 1tr1 | expense / aia | expense / phí ✗ | expense / None ✗ |
| 27901313 | prudential chi trả tiền mổ 12tr | refund / prudential | refund / p ✗ | refund / pr ✗ |
| e0112e34 | bảo việt duyệt bồi thường 2tr4 | refund / bảo việt | refund / ảo việt ✗ | refund / bảo việt |
| ce82632b | trung vietlott 10tr | income / vietlott | income / vietlott | income / etlott ✗ |
| b075334c | quay số trúng 500k | income / None | borrow / None ✗ | income / None |
| 85d8d156 | được giải ba hội thao 800k | income / None | income / None | income / ba ✗ |
| 72c5d076 | khoan vay online tamo dong ky 2 900k | repayment_out / tamo | repayment_out / None ✗ | lend / None ✗ |
| 4a44be47 | chú Bình gửi lại 500k tiền xăng | repayment_in / Bình | repayment_in / chú Bình ✗ | repayment_in / chú ✗ |
| 30e8b087 | muon chu Dung 5 cu | borrow / Dung | borrow / Dung | borrow / chu ✗ |
| ef2aa1e1 | trả dì Loan 1tr tiền mượn | repayment_out / Loan | repayment_out / dì ✗ | repayment_out / Loan |

## 5. human-value-01 (150 labels; 3 `uncertain` excluded, 147 complete scored)

Three systems side by side, each scored once: **V7** = released `gidi-finance-v2` (own value
head), **V8-old** = `gidi-finance-v1` INT8 + parser, **V8-new** = retrained INT8 + parser.
V7 and V8-old share the annotation-v1 encoder, so their type/target agree (path A of V7 is the
v1 encoder); they differ only in the value.

The debt-only slice is the **13** notes whose label was `skipped` before the second review turned
debt-only notes into borrow/lend (ids in `results.json: debt_only_ids`). The task text said 10
(the figure in `docs/annotation-v2.md`); the frozen labels contain 13 such notes (8 were
annotation-v1 `skipped` notes, 5 were never queued), so all 13 are used. The 8-note
v1-skipped sub-slice is also reported. `non-debt` = the other 134 complete notes.

**All complete notes (n=147)**

| metric | V7 (gidi-finance-v2) | V8-old | V8-new |
|---|---|---|---|
| type accuracy | 0.8435 | 0.8435 | 0.9320 |
| type macro-F1 | 0.8049 | 0.8049 | 0.9097 |
| target exact | 0.8367 | 0.8367 | 0.8639 |
| type+target exact | 0.7415 | 0.7415 | 0.8231 |
| value exact | 0.9864 | 1.0000 | 1.0000 |
| end-to-end exact | 0.7347 | 0.7415 | 0.8231 |

**Debt-only slice (n=13)**

| metric | V7 (gidi-finance-v2) | V8-old | V8-new |
|---|---|---|---|
| type accuracy | 0.0769 | 0.0769 | 0.9231 |
| type macro-F1 | 0.1000 | 0.1000 | 0.9286 |
| target exact | 0.5385 | 0.5385 | 0.8462 |
| type+target exact | 0.0000 | 0.0000 | 0.8462 |
| value exact | 0.9231 | 1.0000 | 1.0000 |
| end-to-end exact | 0.0000 | 0.0000 | 0.8462 |

**Debt-only, annotation-v1 `skipped` sub-slice (n=8)**

| metric | V7 (gidi-finance-v2) | V8-old | V8-new |
|---|---|---|---|
| type accuracy | 0.1250 | 0.1250 | 0.8750 |
| type macro-F1 | 0.1667 | 0.1667 | 0.9000 |
| target exact | 0.5000 | 0.5000 | 0.8750 |
| type+target exact | 0.0000 | 0.0000 | 0.8750 |
| value exact | 0.8750 | 1.0000 | 1.0000 |
| end-to-end exact | 0.0000 | 0.0000 | 0.8750 |

**Non-debt notes (n=134)**

| metric | V7 (gidi-finance-v2) | V8-old | V8-new |
|---|---|---|---|
| type accuracy | 0.9179 | 0.9179 | 0.9328 |
| type macro-F1 | 0.9028 | 0.9028 | 0.9143 |
| target exact | 0.8657 | 0.8657 | 0.8657 |
| type+target exact | 0.8134 | 0.8134 | 0.8209 |
| value exact | 0.9925 | 1.0000 | 1.0000 |
| end-to-end exact | 0.8060 | 0.8134 | 0.8209 |

Value exact by stratum (`review-queue.jsonl` `strata`; `null_value` = gold value is null):

| value exact | n | V7 | V8-old | V8-new |
|---|---|---|---|---|
| all complete | 147 | 0.9864 | 1.0000 | 1.0000 |
| multi_number | 19 | 1.0000 (19/19) | 1.0000 (19/19) | 1.0000 (19/19) |
| bare_number | 2 | 1.0000 (2/2) | 1.0000 (2/2) | 1.0000 (2/2) |
| null_value | 1 | 0.0000 (0/1) | 1.0000 (1/1) | 1.0000 (1/1) |
| null_no_number_ambiguous | 4 | 0.7500 (3/4) | 1.0000 (4/4) | 1.0000 (4/4) |
| debt-only slice | 13 | 0.9231 | 1.0000 | 1.0000 |

### borrow / lend confusion (gold rows, predicted columns; zero cells are 0)

**all complete notes**

| system | gold | borrow | lend | repayment_in | repayment_out | expense | income | transfer | refund |
|---|---|---|---|---|---|---|---|---|---|
| V7 | borrow | 12 | 0 | 5 | 0 | 2 | 1 | 0 | 0 |
| V7 | lend | 0 | 15 | 3 | 1 | 1 | 0 | 0 | 0 |
| V8-old | borrow | 12 | 0 | 5 | 0 | 2 | 1 | 0 | 0 |
| V8-old | lend | 0 | 15 | 3 | 1 | 1 | 0 | 0 | 0 |
| V8-new | borrow | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| V8-new | lend | 0 | 18 | 0 | 1 | 1 | 0 | 0 | 0 |

**non-debt notes only**

| system | gold | borrow | lend | repayment_in | repayment_out | expense | income | transfer | refund |
|---|---|---|---|---|---|---|---|---|---|
| V7 | borrow | 11 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| V7 | lend | 0 | 15 | 0 | 1 | 0 | 0 | 0 | 0 |
| V8-old | borrow | 11 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| V8-old | lend | 0 | 15 | 0 | 1 | 0 | 0 | 0 | 0 |
| V8-new | borrow | 11 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| V8-new | lend | 0 | 15 | 0 | 1 | 0 | 0 | 0 | 0 |

### Every note where old and new encoder differ (27)

`✗` marks a wrong type or target. `debt` = in the debt-only slice. Value is not listed: the
parser gives the same value for both.

| id | note | gold (type / target) | old (type / target) | new (type / target) | debt |
|---|---|---|---|---|---|
| dafba00f | ve may bay vietjet tet 2tr9 | expense / vietjet | repayment_out / vietjet ✗ | expense / vietjet |  |
| 152d7abb | Quang còn nợ mình 1tr | lend / Quang | repayment_in / Quang ✗ | lend / Quang | yes |
| f2cad163 | phu cap an trua 730k | income / None | expense / phu cap ✗ | expense / phu ✗ |  |
| b025cf3e | hoàn tiền đơn lazada 189.000đ | refund / lazada | refund / l ✗ | refund / lazada |  |
| 6fa63696 | thanh toán thẻ hsbc 7,2tr | repayment_out / hsbc | repayment_out / hsbc | repayment_out / hsb ✗ |  |
| c530923c | taxi xanh sm ra san bay 210k | expense / xanh sm | expense / ta ✗ | expense / xi xanh sm ✗ |  |
| f5a1a0c5 | con no chi Thao 300k | borrow / Thao | repayment_in / con ✗ | borrow / Thao | yes |
| b63cc451 | rút tiền mặt ở vpbank 1tr5 | transfer / None | transfer / None | transfer / vpbank ✗ |  |
| de5b84d5 | tien viet bai seo 900k | income / None | expense / viet ✗ | income / viet ✗ |  |
| a645bdb7 | con no Duc 300k chua tra | borrow / Duc | repayment_in / con ✗ | borrow / Duc | yes |
| 6e772c2b | no tien nha ba chu 1 thang | borrow / ba chu | expense / None ✗ | borrow / ba chu | yes |
| 2ec5ac38 | cho co Ut muon 3tr | lend / co Ut | lend / Ut ✗ | lend / co ✗ |  |
| 9be5d444 | icloud 50gb 19k | expense / icloud | expense / icloud | expense / loud ✗ |  |
| 32c0b2c6 | no tien dien thang truoc 400k | borrow / None | income / None ✗ | borrow / None | yes |
| 1ca54d34 | no tien nha chu tro 1tr5 chua tra | borrow / chu tro | repayment_in / None ✗ | borrow / chu ✗ | yes |
| c59b6ce4 | no tien an cua Vy 90k chua tra | borrow / Vy | repayment_in / Vy ✗ | borrow / Vy | yes |
| 3e661f54 | nợ tiền ăn Hương 85k chưa trả | borrow / Hương | repayment_in / Hương ✗ | borrow / Hương | yes |
| 8b884cbb | go! big c siêu thị 1tr1 | expense / go! big c | transfer / None ✗ | transfer / b ✗ |  |
| d5387a64 | no tien tro chu nha 1tr5 chua dua | borrow / chu nha | borrow / None ✗ | borrow / chu nha | yes |
| 559fe3c3 | no anh Khoa 300 chua tra | borrow / Khoa | expense / Khoa ✗ | borrow / Khoa | yes |
| 2f76355e | pizza 4p tối t7 389.000đ | expense / pizza 4p | income / piz ✗ | income / iz ✗ |  |
| fae51dfa | Phúc chưa trả 1tr5 tháng trước | lend / Phúc | repayment_in / Phúc ✗ | lend / Phúc | yes |
| 0ca0c139 | vay chị dâu 5tr sửa nhà | borrow / chị dâu | borrow / chị ✗ | borrow / chị dâu |  |
| 5bc92175 | nạp ví momo trả tiền điện 700k | transfer / None | transfer / m ✗ | transfer / momo ✗ |  |
| 7d6f6fac | Nam hứa t6 trả 800k | lend / Nam | repayment_in / Nam ✗ | lend / Nam | yes |
| 8df8cb73 | chi Loan muon 2 trieu | lend / Loan | lend / chi ✗ | lend / Loan |  |
| 1f45cc33 | Quang con thieu 250k tien ve | lend / Quang | expense / Qu ✗ | expense / Quang con ✗ | yes |

## 6. Cost

| | old | new |
|---|---|---|
| INT8 `predict` p50 / p95 (ms, 1 thread, 200 runs, interleaved) | 1.392 / 1.742 | 1.386 / 1.728 |
| INT8 ONNX size (bytes) | 28,658,533 | 28,658,533 |
| bundle size (bytes, with the unchanged tokenizer and config files) | 29,287,783 | 29,287,783 |

Same architecture, so size and latency are identical (difference is noise).

## 7. Verdict

Rule: ACCEPT if the debt slice improves and the frozen-test and probe type macro-F1 each drop
<= 0.01 and target exact each drop <= 0.02; otherwise REJECT.

| criterion | old | new | change | pass |
|---|---|---|---|---|
| debt-only type accuracy (human-value-01, n=13) | 0.0769 | 0.9231 | +0.8462 | yes |
| frozen test type macro-F1 | 0.9563 | 0.9119 | -0.0445 | no |
| frozen test target exact | 0.8857 | 0.8952 | +0.0095 | yes |
| probe type macro-F1 | 0.7543 | 0.7796 | +0.0252 | yes |
| probe target exact | 0.8148 | 0.8148 | +0.0000 | yes |

**Verdict: REJECT** — frozen-test type macro-F1 drops 0.0445 (> 0.01).

What the numbers say, beyond the rule:

- The debt-only fix is real: V8-new gets 85% end-to-end exact on the 13 debt-only notes
  (V7 and V8-old: 0), and whole-set end-to-end exact goes 0.7415 → 0.8231.
  borrow/lend confusion with repayment/expense disappears on human-value-01 (borrow 12/20 → 20/20 correct).
- The non-debt notes of human-value-01 do not regress (type accuracy 0.9179 → 0.9328, target exact unchanged) and probe-v1 improves (type macro-F1 +0.0252).
- The frozen test regresses on type: macro-F1 -0.0445, accuracy -0.0381 (4 more wrong notes of 105). Three of the 4 type regressions are repayment notes that became `expense`, `refund` or `lend` (`tra lai chi Mai`, `Nhi ck tra lai`, `đòi được nợ thằng Lâm`); the fourth is `de rieng tien sua xe` transfer → expense. [INFERENCE] the debt-direction vocabulary added by debt-01 now competes with repayment cues. No type fix on the test offsets them. The target span exact on the test improves slightly.
- The trainer's own FP32 test metrics (macro-F1 0.917) show the same drop, so it is not an INT8 artifact.
- One seed, as the rule requires. The compression-v3 report gives a seed spread of ±0.018 type macro-F1 on this test for the old data (mean 0.943 over 3 seeds; the deployed seed 1 scored 0.956 INT8), so a part of the gap may be seed noise. This run did not measure it, and no run was selected or tuned.

## 8. Deviations and notes

- **Trainer change (`--retokenize`).** The trainer takes inputs from a teacher-target cache of
  original-vocabulary ids and remaps them to the pruned vocabulary; the remap refuses any removed
  id. 11 of the new debt notes (`nợ`, `chưa`, `còn nợ`, `công nợ` forms) use 8 BPE tokens that the
  B-rank-8000 spec dropped (it was built for the old data), so the first launch stopped before
  step 1. `train_student.py` / `scripts/train_student.py` gained an opt-in `--retokenize` flag
  that encodes the notes with the pruned student tokenizer (the deployed one) and re-verifies the
  cache rows. On the 723 old notes it reproduces the remapped cache tensors exactly (checked). The
  vocabulary spec, architecture and the FFN map are unchanged, so the deployed tokenizer works.
- The teacher-target cache was rebuilt for the 802 notes (`experiments/annotation-v3-retrain/teacher-targets`,
  frozen teacher `lr5e-05-seed1`); the supervised arm does not read the teacher logits.
- Layout: the trainer writes `runs/supervised/<variant>/seed1/`; checkpoint weights go to
  `experiments/annotation-v3-retrain/weights/` and the ONNX files to `onnx/`; both are git-ignored
  by `experiments/annotation-v3-retrain/.gitignore` (sizes 114 MB / 140 MB).
- `scripts/export_onnx.py` calibrates and checks parity on 50 frozen-test notes, as it did for the
  deployed encoder; the weights never saw them.
- `scripts/evaluate_encoder_retrain.py` ran once on the held-out sets (it refuses to overwrite
  `results.json`); before that it was debugged only with `--smoke` on in-sample data.
- `scripts/build_debt_relabel_queue.py` treated every other jsonl under `datasets/` as a leakage reference, so the new `training-v1` (which holds the batch's own texts) made `--check` stale; it now skips `datasets/annotation-v3/training-v1` like it skips its own directory. The committed debt-01 files are unchanged and `--check` passes.
- Metric conventions: type macro-F1 averages the classes present in gold (`gidi.evaluation.metrics`);
  target exact counts null = null as correct; value exact is the (start, end) span, null = null correct;
  end-to-end exact = type, target and value all right.
