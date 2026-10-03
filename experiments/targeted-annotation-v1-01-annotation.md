# targeted-annotation-v1-01: QC, annotation, and exception routing

Batch: `corpus/raw/targeted-annotation-v1-01.jsonl` (163 records, unchanged). Human review is
exception-based: automatic QC, a first annotation pass, a blind independent second pass by a
separate agent (never saw the first pass or the generation buckets), and Quet only for
exceptions. Nothing is merged into the baseline labels.

Revision 3 (current): the escalation policy now auto-accepts `uncertain` records that the
frozen rules settle (see section 6). Labels and annotation-v1 semantics are unchanged from
revision 2.

Revision 2: re-run after three annotation-v1 rulings were added to `docs/annotation-v1.md`
(refundable deposits → transfer; generic counterparty nouns as targets; bare `X trả` needs debt
evidence for repayment_in). The first pass was re-applied to all 163 records under the new
rules and a fresh blind second pass was run against the updated doc.

## 1. Corpus QC

| check | result |
|---|---|
| malformed / empty / non-NFC / multi-line text | 0 |
| exact duplicates | 0 |
| normalised duplicates (accents stripped, amounts masked), in batch and vs baseline-01 | 0 |
| template collapse (a frame used more than twice) | 0; 8 frames used twice, the intended direction pairs |
| synthetic / nonsensical language | 0 |
| non-finance content | 0 |
| contradictory or unusable wording | 0 |
| **QC accepted / rejected** | **163 / 0** |

Slang, teencode, typos (`lazda`), missing accents, fragments, and English mixing (`roommate`,
`done`, `refund`) are intended and were not penalised.

## 2. Queue

`datasets/annotation-v1/targeted-01/queue.jsonl`: 163 entries `{id, text, corpus, position}`
(`corpus` = the raw batch, `position` = 1-based line). The baseline queue is untouched.

## 3. Effect of the new rules

| | revision 1 | revision 2 |
|---|---|---|
| escalated | 41 (25.2%) | 37 (22.7%) |
| auto-accepted | 122 | 126 |

Resolved by the new rules:

| rule | resolved and auto-accepted | now |
|---|---|---|
| 1 refundable deposits | 1, 6, 108, 112, 160 | transfer, null |
| 2 generic counterparty nouns | 5, 51, 52, 55, 61, 102, 114 | target kept (`cua hang`, `công ty du lịch`, `nhà thuốc`, `cửa hàng`, `kế toán`, `phòng nhân sự`, `ngân hàng`) |
| 3 bare `X trả` | 31, 70, 98, 106, 130, 149, 156 | proposal settled as uncertain; still escalated because every uncertain record is |

Newly escalated because rule 3 now applies to previously accepted records (8):
- 10, 115, 128 (`X trả tiền ăn/cơm` with no `lại`, `hôm qua`, or debt word): now proposed
  `uncertain`; both passes agree. Human label 108 (`Duy trả tiền ăn hôm qua` → repayment_in)
  differs by `hôm qua`, so the reviewer decides whether a meal share counts as debt context.
- 35, 43, 60, 135 (`X trả nốt/đủ`) and the mirror 103 (`trả nốt Hà`): a remaining balance is
  stated, but not that it is a debt rather than a payment for goods or services.

Baseline check: the 600 baseline labels contain no deposit notes and no bare `X trả`
repayment_in label (`Hoa trả tiền ăn hôm qua` matches human 108), so no baseline label conflicts
with the new rules.

## 4–5. Review and auto-accept

| | n |
|---|---|
| label disagreements between the passes | 4 (10, 115, 128, 134), all escalated |
| auto-accepted `complete` | 126 |
| auto-accepted `uncertain` (rule-settled) | 24 |
| escalated | 13 (8.0%) |

Auto-accepted records have first-pass confidence ≥ 0.90, identical labels in both passes, status
`complete`, a validated span, and no QC concern. The blind reviewer scored 21 of them
below 0.90; each doubt is settled by the doc or a human baseline label as cited below. Both
confidences are kept in provenance (`confidence`, `review_confidence`).

| pos | text | label | review conf | settled by |
|---|---|---|---|---|
| 9 | `cho ban cu muon 1tr5` | complete / lend / `ban cu` | 0.88 | role noun kept (human 86 `bạn thân`, 287 `ban cung phong`) |
| 12 | `trả lại chị Thảo 1tr hôm trước mượn` | complete / repayment_out / `Thảo` | 0.88 | `trả lại X … mượn` → repayment_out (doc edge row 6) |
| 13 | `ứng lương tháng 10 3tr` | complete / borrow / null | 0.85 | doc: salary advance → borrow; target null when employer unwritten |
| 21 | `tra gop cho Vinh ky 3 1tr` | complete / repayment_out / `Vinh` | 0.85 | doc: installment payments → repayment_out |
| 26 | `cho Lợi 1tr, hẹn cuối tháng trả` | complete / lend / `Lợi` | 0.88 | `hẹn … trả` states the expected repayment (lend definition) |
| 42 | `youtube premium refund 79k` | complete / refund / `youtube` | 0.85 | service not tier: human 5 `spotify family` → `spotify` |
| 49 | `cashback thẻ vib 230k` | complete / refund / `vib` | 0.85 | doc: cashback → refund; named issuer is target (`momo cashback` → `momo`) |
| 56 | `anh re tra lai 5tr muon hom truoc` | complete / repayment_in / `anh re` | 0.88 | doc: kinship/role term as only identifier kept |
| 59 | `cô Út gửi trả 2 triệu` | complete / repayment_in / `cô Út` | 0.80 | doc: title before birth-order name keeps phrase |
| 63 | `em họ vay 2 triệu nộp học phí` | complete / lend / `em họ` | 0.88 | doc: kinship term kept |
| 68 | `muon tam ban 300k` | complete / borrow / `ban` | 0.85 | role noun kept (human 86 `bạn thân`) |
| 71 | `ung luong 2tr` | complete / borrow / null | 0.85 | doc: salary advance → borrow |
| 73 | `cho Tâm bên kế toán mượn 500k` | complete / lend / `Tâm` | 0.88 | proper name before a role phrase: the name is the target |
| 87 | `anh Tư cho vay 20tr` | complete / borrow / `anh Tư` | 0.80 | doc edge row 4: `mẹ cho mượn` → borrow |
| 111 | `Tri gui tra 180k tien xang` | complete / repayment_in / `Tri` | 0.85 | `gửi trả` = pays back |
| 112 | `khách sạn hoàn cọc 500k` | complete / transfer / null | 0.85 | doc: returned refundable deposit (`hoàn cọc`) → transfer |
| 113 | `Tien hoi muon 600k da ck` | complete / lend / `Tien` | 0.88 | `X hỏi mượn … đã ck` → lend |
| 125 | `Vinh tra gop ky 3 cho minh 1tr` | complete / repayment_in / `Vinh` | 0.85 | `trả góp cho mình` states an instalment obligation to the user |
| 131 | `gui lai Phong tien grab 65k` | complete / repayment_out / `Phong` | 0.85 | mirror of human 169 |
| 137 | `ứng cho em trai 1tr` | complete / lend / `em trai` | 0.88 | doc: kinship term kept |
| 141 | `Tuấn trả góp lần 2 cho mình 1tr` | complete / repayment_in / `Tuấn` | 0.85 | `trả góp cho mình` states an instalment obligation to the user |

## 6. Rule-settled `uncertain` records

An `uncertain` annotation is auto-accepted only when all hold: an annotation-v1 rule or edge-case
row explicitly makes that wording uncertain; both passes chose `uncertain`; both chose the same
target (or both null); the label carries a non-empty `note`; no corpus-quality, span, or new-rule
question remains. Confidence alone never qualifies. `uncertain` records are never trainable.

24 of the 26 proposed-uncertain records qualify (14 written to be ambiguous, 10 bare `X trả`):

| pos | text | target | settled by | note |
|---|---|---|---|---|
| 2 | `Nam gửi 500k` | `Nam` | edge row 1 | incoming money with no reason: gift, loan, repayment, or refund |
| 3 | `trả Linh 300k` | `Linh` | edge row 5 | trả X with no reason: paying X for something vs repaying a debt |
| 7 | `nhận 700k từ Hạnh` | `Hạnh` | edge row 1 | money received from Hạnh with no reason: gift, loan, repayment, or refund |
| 10 | `dong nghiep tra tien an trua 70k` | `dong nghiep` | edge row 27 / repayment_in evidence rule | X trả tiền ăn/cơm with no debt word: repayment of a fronted meal, reimbursement, or payment |
| 31 | `nhan dc tien Quan tra 1tr5` | `Quan` | edge row 27 / repayment_in evidence rule | bare X trả with no debt context: repayment, payment for goods/services, or reimbursement |
| 33 | `cho Tùng 200k` | `Tùng` | edge row 9 | cho X without vay/mượn: gift vs loan |
| 37 | `ck Quynh 600k` | `Quynh` | edge row 9 | ck X with no reason: gift, payment, or loan |
| 39 | `tra Dung 150k` | `Dung` | edge row 5 | trả X with no reason: paying X for something vs repaying a debt |
| 57 | `chi Lan dua 1tr` | `Lan` | edge row 1 | Lan hands over money with no reason: gift, loan, repayment, or refund |
| 65 | `đưa Khoa 500k` | `Khoa` | edge row 9 | đưa X with no reason: gift, payment, or loan |
| 70 | `Kiet tra minh 500 nghin` | `Kiet` | edge row 27 / repayment_in evidence rule | bare X trả with no debt context: repayment, payment for goods/services, or reimbursement |
| 74 | `Tuan gui 400k` | `Tuan` | edge row 1 | incoming money with no reason: gift, loan, repayment, or refund |
| 78 | `anh Phát chuyển 3tr` | `Phát` | edge row 1 | incoming transfer with no reason: gift, loan, repayment, or refund |
| 91 | `ck cho Mạnh 1tr` | `Mạnh` | edge row 9 | ck cho X without vay/mượn or reason: gift, payment, or loan |
| 93 | `mình trả Quyên 250k` | `Quyên` | edge row 5 | mình trả X with no reason: paying X for something vs repaying a debt |
| 98 | `Quyên trả mình 250k` | `Quyên` | edge row 27 / repayment_in evidence rule | bare X trả with no debt context: repayment, payment for goods/services, or reimbursement |
| 106 | `nhan tien Thanh tra 800k` | `Thanh` | edge row 27 / repayment_in evidence rule | bare X trả with no debt context: repayment, payment for goods/services, or reimbursement |
| 115 | `Vy ck trả 350k tiền ăn` | `Vy` | edge row 27 / repayment_in evidence rule | X trả tiền ăn/cơm with no debt word: repayment of a fronted meal, reimbursement, or payment |
| 118 | `Hương ck hộ Bảo 250k` | null | edge row 14 | X ck hộ Y: who owes whom, and in which direction, is unclear |
| 128 | `đồng nghiệp Nhung trả 150k tiền cơm` | `Nhung` | edge row 27 / repayment_in evidence rule | X trả tiền ăn/cơm with no debt word: repayment of a fronted meal, reimbursement, or payment |
| 130 | `Hieu ck tra 400k` | `Hieu` | edge row 27 / repayment_in evidence rule | bare X trả with no debt context: repayment, payment for goods/services, or reimbursement |
| 139 | `cty ck 1tr1` | `cty` | refund table `cty gửi 780k` | cty sends money with no reason: pay, reimbursement, or advance |
| 149 | `nhận tiền Phúc trả 1tr2` | `Phúc` | edge row 27 / repayment_in evidence rule | bare X trả with no debt context: repayment, payment for goods/services, or reimbursement |
| 156 | `em gai ck tra 500k` | `em gai` | edge row 27 / repayment_in evidence rule | bare X trả with no debt context: repayment, payment for goods/services, or reimbursement |

Note on 10, 115, 128: human label 108 (`Duy trả tiền ăn hôm qua`) is repayment_in; these notes
lack `hôm qua`/`lại`, and the repayment_in evidence rule lists reimbursement as an open reading,
so the rule settles them as uncertain.

## 7. Escalations

13 records (8.0%; revision 2: 37).

### unusual single cases (5)

| pos | text | proposal | conf (pass 1 / review) | reason |
|---|---|---|---|---|
| 11 | `rút thấu chi tcb 2tr` | complete / borrow / `tcb` | 0.85 / 0.70 | overdraft drawdown not covered by the doc; borrow from bank-as-lender assumed |
| 17 | `vay qua app 2tr, nhận tiền rồi` | complete / borrow / null | 0.85 / 0.90 | lender is the generic word `app` after `qua`: target app vs null unclear |
| 19 | `ông anh họ trả dần 500k` | complete / repayment_in / `ông anh họ` | 0.85 / 0.72 | span: `ông anh họ` vs `anh họ` (colloquial `ông`) |
| 34 | `bac Sau tra tien muon 3tr` | complete / repayment_in / `bac Sau` | 0.85 / 0.90 | span: unaccented birth-order name `bac Sau` (keep phrase) vs `Sau` |
| 123 | `cho em hang xom vay 300k` | complete / lend / `em hang xom` | 0.85 / 0.80 | span: `em hang xom` vs `hang xom` (em as prefix before a role noun) |

### `trả nốt/đủ`: balance vs debt (5)

| pos | text | proposal | conf (pass 1 / review) | reason |
|---|---|---|---|---|
| 35 | `Duy tra du 1tr2 roi` | complete / repayment_in / `Duy` | 0.80 / 0.80 | `trả nốt/đủ` implies a remaining balance, but not that it is a debt rather than a payment for goods/services |
| 43 | `Trang trả đủ 1,5tr rồi nha` | complete / repayment_in / `Trang` | 0.80 / 0.80 | `trả nốt/đủ` implies a remaining balance, but not that it is a debt rather than a payment for goods/services |
| 60 | `Linh tra not 300k nha` | complete / repayment_in / `Linh` | 0.80 / 0.95 | `trả nốt/đủ` implies a remaining balance, but not that it is a debt rather than a payment for goods/services |
| 103 | `trả nốt Hà 600k` | complete / repayment_out / `Hà` | 0.80 / 0.80 | `trả nốt/đủ` implies a remaining balance, but not that it is a debt rather than a payment for goods/services |
| 135 | `Hà trả nốt 600k` | complete / repayment_in / `Hà` | 0.80 / 0.80 | `trả nốt/đủ` implies a remaining balance, but not that it is a debt rather than a payment for goods/services |

### span disagreement on an uncertain note (1)

| pos | text | proposal | conf (pass 1 / review) | reason |
|---|---|---|---|---|
| 82 | `cho em 300k` | uncertain / — / `em` | 0.95 / 0.50 | `cho em` is gift vs loan (edge row 9), but the passes disagree on the span: `em` vs null |

### passes disagree (1)

| pos | text | proposal | conf (pass 1 / review) | reason |
|---|---|---|---|---|
| 134 | `vay chi ho 5tr` | complete / borrow / `chi ho` | 0.93 / 0.60 | `vay chi ho`: chị họ (cousin) vs chị Hồ vs `vay hộ`; second pass proposed target `ho` |

### uncertain reason not settled by annotation-v1 (1)

| pos | text | proposal | conf (pass 1 / review) | reason |
|---|---|---|---|---|
| 138 | `ứng 2tr` | uncertain / — / null | 0.95 / 0.50 | bare `ứng 2tr`: annotation-v1 covers `ứng lương` (borrow) and `ứng cho X` (lend) but not `ứng` with no party or direction |

Files: `recheck.jsonl` (13 queue lines, verbatim) and `proposals.jsonl` (Quet `--proposals`:
`id`, `annotation_status`, `type`, `target`, `confidence` = min of both passes, `reason`;
uncertain proposals also carry `note`).

## 8. Augmentation labels

- `labels.jsonl`: 150 labels (126 `complete`, 24 `uncertain` with `note`), validated against
  the targeted queue.
- `provenance.jsonl`: `{"id","annotator":"ai","origin":"synthetic-targeted","source_batch":"targeted-annotation-v1-01","confidence","review_confidence"}`,
  plus `settled_by` (the doc rule) on rule-settled uncertain records.
- Human decisions go to `human-labels.jsonl` (Quet `--out`) and are merged into this batch's
  labels afterwards, never into `datasets/annotation-v1/labels.jsonl` at this stage.

## 9. Distribution

First-pass labels (escalated rows show the proposal):

| type (first pass) | accepted | escalated (proposed) | total | accented | unaccented | target | null |
|---|---|---|---|---|---|---|---|
| repayment_in | 39 | 6 | 45 | 21 | 24 | 45 | 0 |
| repayment_out | 6 | 1 | 7 | 4 | 3 | 7 | 0 |
| refund | 40 | 0 | 40 | 25 | 15 | 33 | 7 |
| transfer | 5 | 0 | 5 | 3 | 2 | 0 | 5 |
| lend | 19 | 1 | 20 | 13 | 7 | 20 | 0 |
| borrow | 17 | 3 | 20 | 13 | 7 | 17 | 3 |
| uncertain | 24 | 2 | 26 | 14 | 12 | 24 | 2 |
| **all** | 150 | 13 | 163 | 93 | 70 | 146 | 17 |

Accepted labels: 86 accented / 64 unaccented.

Intended bucket vs actual label (16 mismatches; bucket intent was not used to label):

| pos | text | intended | actual (first pass) |
|---|---|---|---|
| 1 | `chu nha tra lai tien coc 4tr` | refund | transfer |
| 6 | `hoàn cọc phòng trọ 3tr` | refund | transfer |
| 10 | `dong nghiep tra tien an trua 70k` | repayment_in | uncertain |
| 31 | `nhan dc tien Quan tra 1tr5` | repayment_in | uncertain |
| 70 | `Kiet tra minh 500 nghin` | repayment_in | uncertain |
| 93 | `mình trả Quyên 250k` | contrast_repayment_out | uncertain |
| 98 | `Quyên trả mình 250k` | repayment_in | uncertain |
| 106 | `nhan tien Thanh tra 800k` | repayment_in | uncertain |
| 108 | `chủ nhà trả cọc 5tr.` | refund | transfer |
| 112 | `khách sạn hoàn cọc 500k` | refund | transfer |
| 115 | `Vy ck trả 350k tiền ăn` | repayment_in | uncertain |
| 128 | `đồng nghiệp Nhung trả 150k tiền cơm` | repayment_in | uncertain |
| 130 | `Hieu ck tra 400k` | repayment_in | uncertain |
| 149 | `nhận tiền Phúc trả 1tr2` | repayment_in | uncertain |
| 156 | `em gai ck tra 500k` | repayment_in | uncertain |
| 160 | `nhan lai tien coc xe 1tr` | refund | transfer |
