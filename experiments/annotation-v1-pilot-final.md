# annotation-v1 pilot: final analysis (positions 1–140)

This analysis covers the canonical labels after both re-check passes (`recheck-pilot-01`,
`recheck-frozen-v1-02`). It judges them against the frozen annotation-v1 rules in
`docs/annotation-v1.md`, including the vi-VN quantified precious-asset heuristic. Inputs:
`datasets/annotation-v1/{queue,labels}.jsonl` and `configs/annotation-v1.yaml`. No labels were
modified and nothing was trained. The first-pass analysis is in `annotation-v1-pilot.md`.

## 1. Validation

`uv run python scripts/validate_annotations.py datasets/annotation-v1/labels.jsonl`
→ `140 annotation(s), 0 error(s) [complete 134, skipped 6]`.

Every structural and contract rule passes:
- span offsets are correct;
- every transfer has a null target;
- every skipped record has a null type and a null target;
- there are no uncertain records, so the rule requiring a note on them does not apply.

## 2. Status

| status | n |
|---|---|
| complete | 134 |
| uncertain | 0 |
| skipped | 6 |

All 6 skipped records are debt-state notes (positions 2, 27, 77, 103, 124, 135).

## 3. Type distribution (134 complete)

| type | n | % of complete |
|---|---|---|
| expense | 50 | 37.3 |
| transfer | 24 | 17.9 |
| income | 16 | 11.9 |
| borrow | 13 | 9.7 |
| repayment_out | 13 | 9.7 |
| lend | 8 | 6.0 |
| refund | 7 | 5.2 |
| repayment_in | 3 | 2.2 |

These changes from the first pilot come from the frozen rules:
- borrow went from 19 to 13 because debt-state notes became skipped;
- repayment_out went from 8 to 13 (card, BNPL and liability payments);
- transfer went from 21 to 24 (ccq and gold);
- refund went from 5 to 7 (employer reimbursements);
- repayment_in went from 4 to 3, after `trả nợ spaylater` was corrected to repayment_out.

## 4. Target present vs null by type

| type | target | null |
|---|---|---|
| expense | 19 | 31 |
| transfer | 0 | 24 |
| income | 5 | 11 |
| borrow | 13 | 0 |
| repayment_out | 8 | 5 |
| lend | 8 | 0 |
| refund | 6 | 1 |
| repayment_in | 3 | 0 |
| (skipped) | 0 | 6 |

Overall, 62 records have a target and 78 are null.
- The null expense targets are notes that name no party: dishes, bills, fuel and clinics.
- The null repayment_out targets are instalment notes that name no lender (11, 12, 43, 57, 102).
- The one null refund is `cashback the tin dung`, which names no card issuer.

## 5. Accented vs unaccented

Overall there are 83 accented (59%) and 57 unaccented (41%) records. The full queue is 60% / 40%.

| type | accented | unaccented |
|---|---|---|
| expense | 32 | 18 |
| transfer | 12 | 12 |
| income | 9 | 7 |
| borrow | 10 | 3 |
| repayment_out | 7 | 6 |
| lend | 4 | 4 |
| refund | 3 | 4 |
| repayment_in | 3 | **0** |
| (skipped) | 3 | 3 |

## 6. Remaining semantic inconsistencies

- **Position 111, `no tien nha ba chu 1 thang`**, is labelled complete, `borrow`, `ba chu`. This is
  the unaccented form of the frozen doc's own debt-state example (`nợ tiền nhà bà chủ 1 tháng` →
  skipped). It was in `recheck-frozen-v1-02` but was not changed. The other five debt-state notes
  are skipped, so this one also conflicts with them (§8).

Everything else is consistent with the frozen rules:
- loan direction;
- implied-user repayments;
- employer reimbursement → refund (41, 119);
- cashback → refund (85);
- card, BNPL and instalment payments → repayment_out, including 102;
- ccq and gold → transfer (9, 93, 137), with 93 following the vi-VN `nửa chỉ` heuristic;
- `đổi đô` → transfer;
- wallet top-up → transfer;
- the companion notes (16, 29, 65, 112) have a null target;
- the beneficiary notes (35, 131) have a null target.

## 7. Remaining suspicious target spans

- **Position 134, `thuoc cam 72k nha thuoc long chau` → `nha thuoc long chau`.** The span includes
  the generic shop word `nhà thuốc`, but the identical merchant at position 69 (`… nhà thuốc long
  châu 126k`) → `long châu`. The minimal-span rule, and the parallel `cf highlands` → `highlands`,
  both favour `long chau`.
- **Low confidence, not flagged: position 140, `qua sinh nhat bé Na 300k` → `Na`.** `quà` (a gift)
  could be cash handed to Na, which gives target `Na`, or an item bought for Na, which gives
  `null`, as with 35 and 131. Both readings fit the rules. The doc's beneficiary rule doesn't
  settle the `quà` case, so this is a wording gap, not a labelling error.

The prefix stripping (`Long`, `Hoa`, `Na`, `Thao`, `Tu`, `Tí`) and the role- and kinship-only
targets are applied consistently.

## 8. Remaining conflicts among near-identical notes

| group | positions | labels | status |
|---|---|---|---|
| `còn nợ X …` / `nợ tiền nhà …` | 2, 27, 77, 103, 124 vs **111** | skipped vs borrow | conflict (see §6) |
| `nhà thuốc long châu` | 69 vs **134** | `long châu` vs `nha thuoc long chau` | span-boundary conflict (see §7) |

Every other group I checked is consistent:
- `circle k` (26, 84, 96);
- `highlands` (23, 107);
- `spotify` (5, 122);
- `hoàn tiền shopee` (34, 74, 80, 101);
- `tiền nhà` (3, 126);
- `đổ xăng` (15, 83, 115);
- `khám răng` (45, 92, 114);
- `trả góp` / `góp` with no lender named (11, 12, 43, 57);
- `rút tiền mặt` (14, 25, 55, 59);
- `lãi tiết kiệm` (97, 109);
- `mừng cưới` (28, 71, 79, 89);
- `cty` refunds (41, 119).

## 9. Coverage weaknesses (class imbalance, not labelling errors)

These are corpus coverage issues. The labels themselves are correct.

| weakness | pilot n | projected over 600 (×4.3) | concern |
|---|---|---|---|
| `repayment_in` | 3 (0 unaccented) | ~13 | about 2 per val/test split; no unaccented examples; only the `X trả (lại) …` template |
| `refund` | 7 (4 are the shopee template) | ~30 | few kinds of refund; no merchant refunds besides shopee |
| `lend` | 8 | ~34 | enough for training, but thin for per-type test metrics |
| `borrow`, unaccented | 3 | ~13 | skewed toward accented text |
| `uncertain` | 0 | — | the ambiguous templates the spec is built around (`X gửi 500k`, `trả X 500k`, `cho X 500k`) do not appear; the path is untested |
| `skipped` reasons | 6, all debt-state | — | non-finance or unusable notes never occur |
| `income` with target | 5 | ~21 | most income names no payer (salary, sales, interest) |

## Final re-check candidates

These two records have concrete, high-confidence conflicts with the frozen rules or with a
near-identical note. Both are built into `datasets/annotation-v1/recheck-final-03.jsonl`.

| pos | id | text | current | reason |
|---|---|---|---|---|
| 111 | baseline-01-43b46e772c2b | `no tien nha ba chu 1 thang` | complete, borrow, `ba chu` | This is the unaccented form of the doc's own debt-state example, which is skipped (type and target null). It conflicts with the 5 other debt-state notes. |
| 134 | baseline-01-2561a90bd911 | `thuoc cam 72k nha thuoc long chau` | complete, expense, `nha thuoc long chau` | The span includes the generic `nhà thuốc`. The same merchant at 69 → `long châu`, and the minimal-span rule gives `long chau`. |

## 10. Is annotation-v1 safe for positions 141–600?

**Yes, the specification is safe.** Neither remaining issue needs a rule change. 111 contradicts a
rule that is already written, and 134 is a boundary slip that the minimal-span rule already
covers. Two wording clarifications would help before continuing. They are not semantic changes:
1. A generic shop-type word before a merchant name (`nhà thuốc`, `quán`, `cf`, `tiệm`) is left
   out of the span.
2. The doc should say how to handle `quà` (a gift) for a named person. Suggested wording: target
   the person only when the note says the money itself was given to them; otherwise use `null`.

## Conclusion

**B. A small number of specific records still need manual re-check:** positions 111 and 134, in
`datasets/annotation-v1/recheck-final-03.jsonl`. After those two are corrected, the pilot is clean
enough to continue labelling positions 141–600 without changing annotation-v1.

Types to cover later with targeted labelled examples, rather than by changing the 600-record queue:
- **repayment_in**: the highest priority. It needs unaccented forms, `trả nợ tao/mình`, and
  `X gửi lại tiền …`.
- **refund**: needs merchants other than shopee, employer reimbursements, and cashback from named
  issuers.
- **lend**: needs forms without `mượn`/`muon`, such as `cho X vay` and `ứng cho X`.
- **borrow**: needs unaccented forms.
- **uncertain** and **skipped**: need genuinely ambiguous notes and non-finance notes, so that
  those paths get exercised and evaluated.
