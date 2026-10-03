# annotation-v1: targeted-01 merge, final audit, and combined dataset

## 1. Merge

`datasets/annotation-v1/targeted-01/human-labels.jsonl` (13 human decisions) was merged into
`targeted-01/labels.jsonl` (150 auto-accepted labels). No id overlap, no duplicates; human lines
are copied verbatim and ordered by queue position with the rest.

- validator: `163 annotation(s), 0 error(s) [complete 139, uncertain 24]`
- provenance: 150 `annotator: ai` (unchanged) + 13
  `{"annotator":"human","reviewed":"ai-proposal","proposal_confidence","proposal_kept","origin","source_batch"}`;
  11 kept the proposal, 2 changed it (82 `cho em 300k` uncertain → expense `em`; 138 `ứng 2tr`
  uncertain → borrow, null).
- baseline: `datasets/annotation-v1/labels.jsonl` unchanged (`600 annotation(s), 0 error(s)`).

## 2. Targeted batch

| type/status | n | accented | unaccented | target | null | human | AI |
|---|---|---|---|---|---|---|---|
| expense | 1 | 0 | 1 | 1 | 0 | 1 | 0 |
| borrow | 21 | 14 | 7 | 17 | 4 | 4 | 17 |
| lend | 20 | 13 | 7 | 20 | 0 | 1 | 19 |
| repayment_in | 45 | 21 | 24 | 45 | 0 | 6 | 39 |
| repayment_out | 7 | 4 | 3 | 7 | 0 | 1 | 6 |
| transfer | 5 | 3 | 2 | 0 | 5 | 0 | 5 |
| refund | 40 | 25 | 15 | 33 | 7 | 0 | 40 |
| uncertain | 24 | 13 | 11 | 23 | 1 | 0 | 24 |

Accented / unaccented: 93 / 70. Human-reviewed / AI-accepted: 13 / 150.

## 3. Semantic audit (targeted batch)

All 163 labels were checked against the frozen rules, the review rulings, and each other
(mirror pairs, title spans, debt evidence, transfer targets, deposits, generic nouns).

Consistent: direction mirrors (`Hùng trả nợ` / `trả nợ Hùng`, `Long`, `Thảo`, `Phong`, `Hà`,
`Vinh`), birth-order title spans (`dì Tư`, `anh Bảy`, `bác Bảy`, `bac Sau`, `chú Hai`, `cô Út`,
`anh Tư`), title stripping before proper names, deposits (5 transfer, null), generic
counterparty nouns, refund targets, transfers (all null), salary advances.

**Inconsistent (2, AI-accepted):**

| pos | text | current | conflicts with |
|---|---|---|---|
| 59 | `cô Út gửi trả 2 triệu` | complete / repayment_in / `cô Út` | 156 `em gai ck tra 500k`, 130 `Hieu ck tra 400k` → uncertain (edge row 27) |
| 111 | `Tri gui tra 180k tien xang` | complete / repayment_in / `Tri` | 115 `Vy ck trả 350k tiền ăn`, 10, 128 → uncertain (edge row 27) |

`gửi trả` is not in the documented debt-evidence list (`gửi lại` is), so under the
repayment_in evidence rule these read as bare `X trả`. Both passes accepted them before rule 3
was tightened; they were not re-escalated then. Re-check queue:
`targeted-01/recheck-02.jsonl` with advisory `targeted-01/proposals-02.jsonl` (proposed
uncertain).

**Human ruling not yet in the doc (not an error):** 82 `cho em 300k` → expense, target `em`.
Edge row 9 makes `cho Nam 500k` uncertain; the reviewer read `cho` + a kinship term as a gift,
matching the logic of edge row 3 (`mẹ cho 2tr` → income). The named-person cases 33, 91 stay
uncertain, so there is no label conflict, but the distinction (kinship vs name) should be written
down if it is meant as a rule.

## 4. Combined dataset (baseline-01 + targeted-01)

`datasets/annotation-v1/combined/`: `queue.jsonl` (763, with `source_batch`), `labels.jsonl`
(baseline lines then targeted lines, verbatim), `provenance.jsonl` (original provenance plus
`source_batch`). Validator: `763 annotation(s), 0 error(s) [complete 716, skipped 23,
uncertain 24]`.

| type/status | n | accented | unaccented | target | null | human | AI |
|---|---|---|---|---|---|---|---|
| expense | 216 | 128 | 88 | 84 | 132 | 66 | 150 |
| income | 69 | 38 | 31 | 16 | 53 | 27 | 42 |
| borrow | 61 | 40 | 21 | 57 | 4 | 21 | 40 |
| lend | 60 | 37 | 23 | 60 | 0 | 13 | 47 |
| repayment_in | 53 | 29 | 24 | 52 | 1 | 12 | 41 |
| repayment_out | 71 | 43 | 28 | 44 | 27 | 19 | 52 |
| transfer | 131 | 80 | 51 | 0 | 131 | 27 | 104 |
| refund | 55 | 32 | 23 | 43 | 12 | 9 | 46 |
| uncertain | 24 | 13 | 11 | 23 | 1 | 0 | 24 |
| skipped | 23 | 14 | 9 | 0 | 23 | 10 | 13 |

Accented / unaccented: 454 / 309. Human-authored or human-reviewed: 204; AI-accepted: 559.

## 5. Split (provisional until 59 and 111 are re-checked)

Coverage is sufficient: every type has ≥ 53 complete examples, and every type has ≥ 8 records
in each of validation and test (previously repayment_in had 8 in total).

`scripts/build_splits.py` (logic in `gidi.annotation.split`, tests in `tests/test_split.py`),
seed `annotation-v1-split-01`, complete records only. Leakage groups: identical token bags after
normalisation (diacritics stripped, amounts masked, direction/filler words removed), so
direction mirrors group together, plus normalised similarity ≥ 0.9. 565 groups;
each group sits wholly in one split, stratified by (type, accented), with a repair step that
guarantees every type in validation and test. Recomputing groups from the written files shows no
cross-split group.

Accented / unaccented per split:

| type | train | validation | test |
|---|---|---|---|
| expense | 90 / 61 | 19 / 14 | 19 / 13 |
| income | 25 / 22 | 7 / 4 | 6 / 5 |
| borrow | 28 / 15 | 6 / 3 | 6 / 3 |
| lend | 27 / 15 | 5 / 4 | 5 / 4 |
| repayment_in | 19 / 16 | 5 / 4 | 5 / 4 |
| repayment_out | 32 / 20 | 6 / 4 | 5 / 4 |
| transfer | 59 / 34 | 11 / 9 | 10 / 8 |
| refund | 22 / 17 | 5 / 3 | 5 / 3 |
| **total** | 502 | 109 | 105 |

Files: `datasets/annotation-v1/combined/splits/{train,validation,test}.jsonl`, `manifest.json`.
After the re-check the combined files and the split are rebuilt with the same seed
(`uv run python scripts/build_splits.py --overwrite`).
