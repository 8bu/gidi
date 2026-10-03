# annotation-v1: AI-assisted labeling of positions 141–600

Positions 141–600 were labelled automatically, using the frozen annotation-v1 rules
(`configs/annotation-v1.yaml`, `docs/annotation-v1.md`). The 140 corrected human labels served as
calibration. Every AI decision was made by the assistant, one record at a time, against the doc
rules and the matching human examples. No model was trained. Human labels were not modified.

## Files

| file | role |
|---|---|
| `datasets/annotation-v1/labels.jsonl` | canonical labels: 140 human labels (lines kept byte-for-byte) + 409 accepted AI labels, ordered by queue position |
| `datasets/annotation-v1/provenance.jsonl` | sidecar metadata, one line per canonical label: `human`, or `ai` + `confidence` |
| `datasets/annotation-v1/recheck-ai-01.jsonl` | Quet subset queue: the 51 records pending human review, copied verbatim from `queue.jsonl` |
| `datasets/annotation-v1/ai-proposals.jsonl` | advisory proposals for those 51 (`confidence`, `reason`); not canonical |

## Acceptance policy

A record was accepted only when all of these held:
- confidence ≥ 0.90;
- it matched an established human or doc pattern;
- there was no escalation reason.

Anything proposed `uncertain`, any confidence below 0.90, and any escalation reason (new wording,
span boundary, uncovered case, two plausible types) went to review instead.

## Audit

- **Validator:** `549 annotation(s), 0 error(s) [complete 529, skipped 20]`.
- **Human labels:** all 140 original lines are present verbatim and in their original relative
  order. None of their ids were written by AI.
- **Partition:** each of the 600 queue records is exactly one of the following:
  - a human label (140);
  - an accepted AI label (409);
  - pending human review (51).
- **Integrity:** there are no duplicate ids. `provenance.jsonl` covers exactly the canonical ids,
  `ai-proposals.jsonl` covers exactly the review ids, and every review-queue line is identical to
  its `queue.jsonl` line.

## Counts

| | n |
|---|---|
| human labels | 140 |
| new accepted AI labels | 409 (396 complete, 13 skipped) |
| escalated for human review | 51 |
| canonical skipped (human 7 + AI 13) | 20 |
| uncertain proposals | 3 |
| skipped proposals (pending) | 2 |

AI confidence buckets (all 460 AI decisions):

| confidence | accepted | escalated |
|---|---|---|
| ≥ 0.95 | 349 | 0 |
| 0.90–0.94 | 60 | 0 |
| 0.80–0.89 | 0 | 33 |
| 0.70–0.79 | 0 | 14 |
| < 0.70 | 0 | 4 |

## Distributions (canonical labels)

Type distribution (529 complete; the last column shows the proposed types of the 51 pending
records, which are not counted in the totals):

| type | human | AI accepted | total canonical | % of complete | pending review (proposed) |
|---|---|---|---|---|---|
| expense | 50 | 150 | 200 | 37.8 | 13 |
| income | 16 | 42 | 58 | 11.0 | 11 |
| borrow | 12 | 23 | 35 | 6.6 | 5 |
| lend | 8 | 28 | 36 | 6.8 | 4 |
| repayment_in | 3 | 2 | 5 | 0.9 | 3 |
| repayment_out | 13 | 46 | 59 | 11.2 | 4 |
| transfer | 24 | 99 | 123 | 23.3 | 4 |
| refund | 7 | 6 | 13 | 2.5 | 2 |

Target present vs null (complete):

| type | target | null |
|---|---|---|
| expense | 70 | 130 |
| income | 11 | 47 |
| borrow | 35 | 0 |
| lend | 36 | 0 |
| repayment_in | 5 | 0 |
| repayment_out | 33 | 26 |
| transfer | 0 | 123 |
| refund | 10 | 3 |

Accented vs unaccented (all 549 canonical; overall 330 / 219):

| type | accented | unaccented |
|---|---|---|
| expense | 120 | 80 |
| income | 31 | 27 |
| borrow | 21 | 14 |
| lend | 23 | 13 |
| repayment_in | 5 | 0 |
| repayment_out | 37 | 22 |
| transfer | 76 | 47 |
| refund | 6 | 7 |
| (skipped) | 11 | 9 |

## Coverage warnings

- **repayment_in**: 5 canonical examples (0.9%), all accented, plus 3 pending. This is far too
  few for a stratified split with meaningful val/test counts. It is the top priority for targeted
  labelled examples.
- **refund**: 13 examples (2.5%), dominated by shopee refunds, employer reimbursements and card
  cashback.
- **borrow** and **lend**: 35 and 36 examples. That is adequate for training, but per-type
  test metrics will be noisy.
- **uncertain**: 0 canonical and 3 proposed. Ambiguous notes are rare in this corpus.
- **transfer**: heavily represented (23%) because savings, top-up, withdrawal and gold notes are
  common in the corpus. This is corpus imbalance, not a labelling issue.

## Escalated records

### Direction or type not established by the text (proposed uncertain or low confidence) (10)

| pos | id | text | proposed | conf | why human review |
|---|---|---|---|---|---|
| 201 | baseline-01-19b477e99681 | `trả tiền mua chung laptop cho anh Tú 3tr` | uncertain, —, `Tú` | 0.60 | two types plausible: paying Tú a share of a jointly bought laptop — expense vs repayment_out; direction of the debt not stated |
| 246 | baseline-01-658b60f6ae55 | `trả Quân tiền vé 180k` | uncertain, —, `Quân` | 0.60 | direction/type: "trả Quân tiền vé" — paying Quân for a ticket (expense) vs repaying a ticket he fronted (repayment_out); doc row 5 pattern |
| 265 | baseline-01-ba1e717b27a8 | `tiền dạy kèm tuần này 1tr6` | complete, income, null | 0.70 | direction: "tiền dạy kèm" can be tutoring income or paying a tutor; no verb states which |
| 271 | baseline-01-b590dd3df660 | `tra lai chi Mai 2tr` | complete, repayment_out, `Mai` | 0.85 | direction: "tra lai chi Mai" (give back to Mai) without "nợ"/"mượn"; doc keeps bare "trả Nam" uncertain |
| 301 | baseline-01-ebf8bca50f8d | `tien chay grab hom nay 450k` | complete, income, `grab` | 0.70 | direction: "tiền chạy grab" is likely driver earnings (income), but could be ride spending; target unclear |
| 306 | baseline-01-d290c719854d | `day kem them 800k` | complete, income, null | 0.70 | direction: "dạy kèm thêm 800k" — tutoring income vs paying for extra tutoring |
| 364 | baseline-01-3f29fa04ff7a | `trả tiền bác Sáu mượn lần trước 3tr` | uncertain, —, `Sáu` | 0.70 | direction: "trả tiền bác Sáu mượn lần trước" — repaying money the user borrowed from Sáu, or Sáu borrowed? the grammar allows both |
| 448 | baseline-01-ae6b5167768d | `tra lai vay 1tr3` | complete, expense, null | 0.60 | two types plausible: "tra lai vay" read as paying loan interest (trả lãi vay) — expense vs repayment_out; the unaccented form could also mean "trả lại vay" |
| 472 | baseline-01-7b6b0301f7d4 | `tiền dạy kèm tuần này 800k` | complete, income, null | 0.70 | direction: "tiền dạy kèm" — tutoring income vs paying a tutor; same as 265 |
| 561 | baseline-01-a961e5d8132f | `tra bot chi Hanh 1 trieu` | complete, repayment_out, `Hanh` | 0.85 | direction: "tra bot chi Hanh" (partial payment to Hạnh) without "nợ"; pilot 6 had "trả bớt nợ" |

### Case not covered by annotation-v1 (possible taxonomy gap) (10)

| pos | id | text | proposed | conf | why human review |
|---|---|---|---|---|---|
| 177 | baseline-01-87fbf0233b5a | `ứng trước của sếp 2tr` | complete, borrow, `sếp` | 0.70 | salary advance ("ứng trước của sếp"): borrow vs income is not settled by annotation-v1 |
| 199 | baseline-01-e8c066dd0681 | `dong bao hiem nhan tho 1tr2` | complete, expense, null | 0.75 | two types plausible: life-insurance premium as expense vs savings-type transfer; not covered by annotation-v1 |
| 307 | baseline-01-faa233fdac51 | `trả tiền ứng lương cty 1tr5` | complete, repayment_out, `cty` | 0.80 | depends on salary-advance decision (177, 544): repaying an advance is repayment_out only if the advance is borrow |
| 378 | baseline-01-cfed079d807f | `hoan tien bao hiem 670k` | complete, refund, null | 0.75 | two types plausible: insurance payout ("hoàn tiền bảo hiểm") as refund vs income; not covered by annotation-v1 |
| 428 | baseline-01-84749c40fd70 | `hoàn thuế tncn 2tr4` | complete, refund, null | 0.75 | two types plausible: personal income-tax refund ("hoàn thuế tncn") as refund vs income; not covered by annotation-v1 |
| 434 | baseline-01-d1c0bf5f7646 | `chuyen tien the phu 700k` | complete, transfer, null | 0.75 | unclear semantics: "chuyển tiền thẻ phụ" — own supplementary card (transfer) vs paying a credit card |
| 435 | baseline-01-fcdcf1b678eb | `nhan tien ship cod 760k` | complete, income, null | 0.75 | new pattern: COD collection ("nhận tiền ship cod") — sale proceeds vs money collected for someone else |
| 544 | baseline-01-65670c602f08 | `ứng trước lương cty 2tr` | complete, borrow, `cty` | 0.70 | salary advance ("ứng trước lương cty"): borrow vs income is not settled by annotation-v1; same as 177 |
| 596 | baseline-01-fe2a7509cd58 | `dong hui thang nay 2 trieu` | complete, transfer, null | 0.60 | not covered by annotation-v1: "đóng hụi" (rotating savings group) — transfer, expense, or lend? |
| 600 | baseline-01-f0f2013b4395 | `gửi góp tháng 3 triệu` | complete, transfer, null | 0.70 | two types plausible: "gửi góp tháng" — savings deposit (transfer) vs installment payment (repayment_out) |

### Two types plausible under current rules (4)

| pos | id | text | proposed | conf | why human review |
|---|---|---|---|---|---|
| 221 | baseline-01-613ed8f6b8e3 | `nạp đt viettel 100k` | complete, expense, `viettel` | 0.80 | new pattern: phone airtime top-up ("nạp đt") — expense to the carrier vs transfer like wallet top-ups |
| 357 | baseline-01-bdd71c7c21b9 | `chuyen tien vao the tin dung 6tr` | complete, repayment_out, null | 0.85 | two types plausible: "chuyển tiền vào thẻ tín dụng" worded like a transfer but pays the card balance (repayment_out) |
| 410 | baseline-01-bc965ebcb41f | `game nap steam 300k` | complete, expense, `steam` | 0.80 | two types plausible: "nạp steam" — game spending (expense) vs top-up of the user's own Steam wallet (transfer) |
| 488 | baseline-01-c9fe72c3eb50 | `nap the mobi 50k` | complete, expense, `mobi` | 0.80 | new pattern: phone card top-up ("nạp thẻ mobi"); same as 221 |

### New wording not in the human calibration set (8)

| pos | id | text | proposed | conf | why human review |
|---|---|---|---|---|---|
| 160 | baseline-01-ea7e07f54f08 | `pass lại vé concert 1tr5` | complete, income, null | 0.80 | slang: "pass lại vé" read as reselling a ticket (income); slang not in calibration set |
| 169 | baseline-01-ec8912373acc | `Tâm gửi lại tiền taxi 90k` | complete, repayment_in, `Tâm` | 0.85 | new wording: "gửi lại tiền taxi" (sending back fronted taxi money) not in calibration; repayment_in vs refund |
| 190 | baseline-01-4a30fae51dfa | `Phúc chưa trả 1tr5 tháng trước` | skipped, —, null | 0.85 | unusual debt wording: "Phúc chưa trả … tháng trước" read as a debt-state note (skipped), not a movement |
| 286 | baseline-01-f54eac924a73 | `gom tien mua ipad 700k` | complete, transfer, null | 0.85 | slang: "gom tiền mua ipad" read as saving toward a purchase (transfer); not in calibration |
| 319 | baseline-01-4f39386d1d22 | `chị Mai gửi lại tiền ăn 200k` | complete, repayment_in, `Mai` | 0.85 | new wording: "gửi lại tiền ăn" (sending back meal money); same question as 169 |
| 447 | baseline-01-3f8d7d6f6fac | `Nam hứa t6 trả 800k` | skipped, —, null | 0.85 | unusual debt wording: "Nam hứa t6 trả 800k" records a promise, not a movement (skipped?) |
| 455 | baseline-01-d74a957149fd | `lì xì cháu 200k` | complete, expense, `cháu` | 0.85 | new pattern: giving lì xì — the cash goes to cháu (target) by the gift rule; calibration only has lì xì received |
| 549 | baseline-01-f8da0dfdcf96 | `được trả tiền ship hộ 45k` | complete, repayment_in, null | 0.80 | new pattern: "được trả tiền ship hộ" — someone repays shipping the user fronted; payer not named |

### Marketplace or platform: channel or counterparty (4)

| pos | id | text | proposed | conf | why human review |
|---|---|---|---|---|---|
| 207 | baseline-01-28cf001e3f34 | `ban do cu tren cho tot 450k` | complete, income, null | 0.85 | new pattern: sale via marketplace ("chợ tốt") — is the platform a channel (null) or the counterparty? |
| 291 | baseline-01-f225673b6ef5 | `bán đồ cũ chợ tốt 600k` | complete, income, null | 0.85 | new pattern: sale via marketplace ("chợ tốt") — platform as channel (null) or counterparty? |
| 511 | baseline-01-40e581203ef0 | `bán quần áo cũ trên fb 400k` | complete, income, null | 0.85 | new pattern: sale via platform ("trên fb") — channel (null) or counterparty? |
| 548 | baseline-01-169ed58e113c | `bán quần áo cũ trên shopee 450k` | complete, income, null | 0.85 | new pattern: sale via marketplace ("trên shopee") — channel (null) or counterparty? |

### Target-span boundary (15)

| pos | id | text | proposed | conf | why human review |
|---|---|---|---|---|---|
| 163 | baseline-01-e5e787109ed6 | `vay chú Ba 10 triệu sửa nhà` | complete, borrow, `Ba` | 0.80 | span boundary: "chú Ba" — stripping the prefix leaves "Ba", which also reads as "father"; keep "chú Ba" or "Ba"? |
| 173 | baseline-01-a2a26540de33 | `mượn đồng nghiệp Hằng 1tr đến cuối tháng` | complete, borrow, `Hằng` | 0.85 | span boundary: role word + name ("đồng nghiệp Hằng"); prefix-stripping list covers kinship/titles, not role nouns |
| 192 | baseline-01-2540aa4c22f7 | `lãi tiết kiệm vcb 412k` | complete, income, `vcb` | 0.80 | target: bank named as the payer of savings interest; pilot interest notes named no bank |
| 238 | baseline-01-0276b9299437 | `cho dong nghiep Thao muon 200k` | complete, lend, `Thao` | 0.85 | span boundary: role word + name ("dong nghiep Thao"); same question as 173 |
| 252 | baseline-01-5fa094c58941 | `cho anh hai muon 1tr5` | complete, lend, `anh hai` | 0.85 | span boundary: "anh hai" is a kinship term (eldest brother), not prefix + name; keep whole span? |
| 258 | baseline-01-20011b4a0ab6 | `bún chả hàng Mành 55k` | complete, expense, null | 0.80 | target: "hàng Mành" is a street naming the stall — location or merchant identifier? |
| 287 | baseline-01-35edc7c7d6b3 | `cho ban cung phong muon 200k` | complete, lend, `ban cung phong` | 0.85 | span boundary: multi-word role noun "ban cung phong" (roommate) — whole phrase or shorter? |
| 323 | baseline-01-824a9e75ce3c | `be bike ve nha 28k` | complete, expense, `be` | 0.80 | span boundary: ride app "be" ("be bike") — target "be" or "be bike"? |
| 336 | baseline-01-ce97d487edba | `quà 20/10 cho mẹ 500k` | complete, expense, `mẹ` | 0.70 | conflicting examples: "quà … cho mẹ" — gift recipient as target (like 140 "quà sinh nhật bé Na") vs beneficiary null (like 35 "cho bé") |
| 368 | baseline-01-5b281db51839 | `be bike ve nha 27k` | complete, expense, `be` | 0.80 | span boundary: ride app "be" ("be bike"); same as 323 |
| 379 | baseline-01-404651364c5e | `mua giày adidas 1tr9` | complete, expense, `adidas` | 0.80 | target: brand name may be the product brand, not the seller |
| 467 | baseline-01-0191e0eef8be | `vay cậu Ba 10tr` | complete, borrow, `Ba` | 0.80 | span boundary: "cậu Ba" — stripping leaves "Ba", which also reads as "father"; same as 163 |
| 517 | baseline-01-13224bd5cc19 | `bee đi học 18k` | complete, expense, `bee` | 0.80 | span boundary: ride app spelled "bee" (typo of "be") — target "bee"? |
| 528 | baseline-01-13f0e22ece01 | `bee đi sân bay 142.000` | complete, expense, `bee` | 0.80 | span boundary: ride app spelled "bee"; same as 517 |
| 552 | baseline-01-b9ed70a54713 | `cho cậu út vay 3tr` | complete, lend, `cậu út` | 0.85 | span boundary: "cậu út" is a kinship term (youngest uncle), not prefix + name; keep whole span? |

## Notes for the reviewer

- **Recurring themes.** Several escalation groups repeat, and one ruling each would settle them
  and future notes:
  - **Salary advance** (177, 544, 307): is it borrow or income?
  - **Insurance and tax refunds** (378, 428) and **premiums** (199): which type?
  - **Hụi** (596): is it transfer, expense or lend?
  - **Marketplace as channel** (207, 291, 511, 548): is the platform the counterparty?
  - **`be`/`bee` ride app span** (323, 368, 517, 528).
  - **Kinship terms that are ordinals or look like names** (`chú Ba`, `cậu Ba`, `anh hai`,
    `cậu út`).
  - **Role noun before a name** (`đồng nghiệp Hằng`).

  Where one of these needs a new rule, that is an annotation-v1 wording decision, so record it in
  the doc rather than resolving it per record.
- **Accepted AI labels that follow a human precedent the pilot report called debatable:**
  `quà sinh nhật <name>` → the person (156, 237, 267, 479, 560), matching human 140. I did not
  re-open these.

