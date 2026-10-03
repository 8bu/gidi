# annotation-v1: final reconciliation and audit

Inputs: `queue.jsonl`, `labels.jsonl`, `provenance.jsonl`, `ai-proposals.jsonl`,
`configs/annotation-v1.yaml`, `docs/annotation-v1.md`. No labels were modified by this audit.

## 1. Canonical validation

`uv run python scripts/validate_annotations.py datasets/annotation-v1/labels.jsonl`
→ `600 annotation(s), 0 error(s) [complete 577, skipped 23]`

| check | result |
|---|---|
| queue records | 600 |
| canonical annotations | 600 |
| complete / uncertain / skipped | 577 / 0 / 23 |
| missing ids | 0 |
| ids not in queue | 0 |
| duplicate ids | 0 |
| validation errors | 0 |
| original 140 human lines | present byte-for-byte, original relative order |

Quet appended the 51 reviewed records after the first 549 lines, so `labels.jsonl` is no longer
in queue-position order. This is harmless (ids are unique) and was left as is.

## 2. Provenance

`provenance.jsonl` now has one line per queue record (600), ordered by queue position:

| representation | n |
|---|---|
| `{"id","annotator":"human"}`: original human labels | 140 |
| `{"id","annotator":"ai","confidence"}`: accepted AI labels, never reviewed | 409 |
| `{"id","annotator":"human","reviewed":"ai-proposal","proposal_confidence","proposal_kept"}` | 51 |

`proposal_kept` is true when the canonical label equals the AI proposal (status, type, target
span): 39 kept, 12 changed. Every line with `annotator: human` is human-authoritative.

Human changes to AI proposals:

| pos | AI proposal | human decision |
|---|---|---|
| 163 `vay chú Ba 10 triệu sửa nhà` | complete / borrow / `Ba` | complete / borrow / `chú Ba` |
| 201 `trả tiền mua chung laptop cho anh Tú 3tr` | uncertain / — / `Tú` | complete / expense / `Tú` |
| 207 `ban do cu tren cho tot 450k` | complete / income / null | complete / income / `cho tot` |
| 246 `trả Quân tiền vé 180k` | uncertain / — / `Quân` | complete / repayment_out / `Quân` |
| 258 `bún chả hàng Mành 55k` | complete / expense / null | complete / expense / `hàng Mành` |
| 291 `bán đồ cũ chợ tốt 600k` | complete / income / null | complete / income / `chợ tốt` |
| 323 `be bike ve nha 28k` | complete / expense / `be` | complete / expense / `be bike` |
| 364 `trả tiền bác Sáu mượn lần trước 3tr` | uncertain / — / `Sáu` | skipped / — / null |
| 368 `be bike ve nha 27k` | complete / expense / `be` | complete / expense / `be bike` |
| 467 `vay cậu Ba 10tr` | complete / borrow / `Ba` | complete / borrow / `cậu Ba` |
| 548 `bán quần áo cũ trên shopee 450k` | complete / income / null | complete / income / `shopee` |
| 596 `dong hui thang nay 2 trieu` | complete / transfer / null | complete / expense / `hui` |

## 3. Semantic audit

All 600 labels were read against the frozen rules, grouped by type, plus an automatic check for
near-identical notes (accent-, case- and amount-normalised) with differing labels.

| area | finding |
|---|---|
| repayment direction | consistent (`X trả …` → repayment_in; `trả X …`/`trả nợ X` → repayment_out) |
| borrow vs lend | consistent (`mượn/vay X` → borrow; `cho X mượn/vay`, `X mượn` → lend) |
| debt-state notes | all 23 skipped are debt-state or unresolvable; no debt-state note is labelled |
| installment / BNPL / card | all repayment_out; issuer/provider targeted when named |
| refund / reimbursement / cashback | all refund; employer reimbursements → `cty` |
| investment / store of value | ccq, savings, tích lũy, gửi góp, currency → transfer |
| quantified precious assets | all quantified gold → transfer; gift purchases stay expense |
| merchant / platform targets | named merchants targeted; no named counterparty left null |
| companions / beneficiaries | no companion targeted; `mua … tặng/cho X` → null |
| payment channels | no channel targeted; wallet top-ups are null transfers |
| transfer with non-null target | none (126/126 null) |
| generic words in spans | none found |
| near-identical notes | no conflicting labels, except for the title-prefix spans below |
| title / kinship prefixes | **inconsistent: 4 AI labels** |

**Title before a birth-order name.** In review, the human kept the title when the name is a
birth-order word: 163 `chú Ba`, 467 `cậu Ba`, 252 `anh hai`, 552 `cậu út`. Four accepted AI
labels strip the title in the same construction: 145 `Sáu`, 274 `Hai`, 475 `Tư`, 557 `Năm`.
274 (`cho chú Hai vay`) against 163 (`vay chú Ba`) is a direct conflict. 68 and 486 (`cau Tu` →
`Tu`) follow a human pilot label; unaccented `Tu` can be the name Tú, so they are left alone.

Rulings from the review that are consistent with the frozen rules but not yet written in
`docs/annotation-v1.md` (recording them is a doc decision, not a label change):
- salary advance → borrow, repaying it → repayment_out (177, 544, 307);
- insurance payout and tax refund → refund (378, 428); insurance premium → expense (199);
- hụi contribution → expense, target `hui` (596); `gửi góp` → transfer (600);
- a marketplace is the counterparty of a sale (207, 291, 548); `fb` is not (511);
- ride-app span `be bike`/`bee` (323, 368, 517, 528); stall name `hàng Mành` (258);
- title kept before birth-order names (above).

## Manual re-check candidates

| pos | id | text | current label | conflicts with |
|---|---|---|---|---|
| 145 | baseline-01-73065b9f70e5 | `cho dì Sáu mượn 10tr` | complete / lend / `Sáu` (AI) | 163/467 (`chú Ba`, `cậu Ba`), 552 `cậu út` |
| 274 | baseline-01-420d930d589b | `cho chú Hai vay 10 triệu` | complete / lend / `Hai` (AI) | 163 `vay chú Ba` → `chú Ba` |
| 475 | baseline-01-47644bd6095c | `cho bác Tư mượn 1tr2` | complete / lend / `Tư` (AI) | 163/467 (`chú Ba`, `cậu Ba`) |
| 557 | baseline-01-a28a344e745c | `vay tạm dì Năm 2tr` | complete / borrow / `Năm` (AI) | 163/467 (`chú Ba`, `cậu Ba`) |

Queue: `datasets/annotation-v1/recheck-final-04.jsonl`.

## 4. Coverage

Complete labels by type, plus skipped (`~15% slice` = expected records in one of validation or
test under a 70/15/15 split):

| type | total | accented | unaccented | target | null | human | human-reviewed AI | AI | ~15% slice |
|---|---|---|---|---|---|---|---|---|---|
| expense | 215 | 128 | 87 | 83 | 132 | 50 | 15 | 150 | 32.2 |
| income | 69 | 38 | 31 | 16 | 53 | 16 | 11 | 42 | 10.3 |
| borrow | 40 | 26 | 14 | 40 | 0 | 12 | 5 | 23 | 6.0 |
| lend | 40 | 24 | 16 | 40 | 0 | 8 | 4 | 28 | 6.0 |
| repayment_in | 8 | 8 | 0 | 7 | 1 | 3 | 3 | 2 | 1.2 |
| repayment_out | 64 | 39 | 25 | 37 | 27 | 13 | 5 | 46 | 9.6 |
| transfer | 126 | 77 | 49 | 0 | 126 | 24 | 3 | 99 | 18.9 |
| refund | 15 | 7 | 8 | 10 | 5 | 7 | 2 | 6 | 2.2 |
| (skipped) | 23 | 14 | 9 | 0 | 23 | 7 | 3 | 13 | 3.4 |
| **all** | 600 | 361 | 239 | 233 | 367 | 140 | 51 | 409 | 90 |

Uncertain: 0 labels. Skipped: 23 (3.8%), almost all `còn nợ` debt-state notes.

Weak classes:
- **repayment_in**: 8, all accented; 6 of 8 are `X trả (lại) tiền ăn` / `gửi lại` patterns.
  About 1 record per validation or test slice; no unaccented examples at all.
- **refund**: 15. Only three sources: shopee (5), `cty` reimbursements (4), cashback (4), plus
  insurance and tax. About 2 per slice.
- **borrow / lend**: 40 each, about 6 per slice and 2 unaccented per slice. Adequate for training,
  but per-class test metrics will swing by ~15 points per error.
- **uncertain**: none. The model will never see this status unless it is added.

## 5. Augmentation decision

Not ready to split: repayment_in and refund cannot support meaningful validation and test sets,
and uncertain is absent. Proposed targeted annotation-v1 augmentation (to be generated through the
normal raw → Quet review → annotate flow, as a separate batch; the 600-record queue is unchanged):

| type | add | goal total | missing variants |
|---|---|---|---|
| repayment_in | ~55 | ~60 | `X trả nợ`, `X trả lại`, `X ck trả`, `nhận tiền X trả`, `X gửi trả`, `X trả góp/trả dần`, `X hoàn tiền mượn`, `thu nợ X`, `đòi được nợ X`; role/kinship subjects (`em`, `bạn cùng phòng`, `khách nợ`); partial and final repayments |
| refund | ~45 | ~60 | merchants beyond shopee (lazada, tiki, grab, cửa hàng, hãng bay/vé), cancelled orders, `trả hàng hoàn tiền`, deposit refunds (`hoàn cọc`), `bảo hành hoàn`, employer/`cty thanh toán lại`, friends reimbursing a shared bill vs repayment boundary, cashback with named issuer (`vib`, `tcb`) |
| lend | ~20 | ~60 | unaccented and slang (`cho muon`, `ung cho`, `cho vay nong`), role/kinship borrowers, `X mượn` without `cho` |
| borrow | ~20 | ~60 | unaccented and slang (`vay nong`, `muon tam`, `ung`), apps/banks as lenders (`vay app`, `vay tín chấp vpbank`) |
| uncertain | ~15 | ~15 | genuinely direction-ambiguous notes (`Nam ck hộ Linh`, `trả tiền mua chung`, bare `tiền vé X`) with a `note` |

Mix requirements for the batch:
- at least 50% unaccented for repayment_in (currently 0%) and ~40% overall;
- shorthand and slang: `ck`, `tr`, `củ`, `k`, `nghìn`, teencode (`dc`, `ko`, `r`), dropped
  subjects, misspellings (`hòan`, `tra no`), names with and without titles, including
  birth-order names;
- targets both present and null, so the span head does not learn a type-only shortcut;
- near-miss contrasts in the same batch (`X trả` vs `trả X`, `hoàn tiền` vs `trả lại tiền`,
  `cho X mượn` vs `mượn X`) to force direction learning.

## Conclusion

**C. Specific labels still need manual re-check**: 145, 274, 475, 557 (title-prefix spans).
After that pass the labels are clean, and the dataset moves to **B**: targeted augmentation is
needed before splitting.
