# human-value-01 annotation guide

You label **one thing per note: the value span**, the exact substring that is the monetary amount
of the transaction. This is the annotation-v2 value pass, unchanged
([`docs/annotation-v2.md`](../../../docs/annotation-v2.md), "The span convention" and "Value pass
labels"). This set is the independent test of two models, so label what the note says, nothing
else. You never see model or rule output: there are no proposals, no prefill, no suggestions. Do
not run the models or the value parser to help yourself. The `strata` in the queue only say why a
note was sampled; they are not hints about the answer.

Quet command (no `--proposals`):

```sh
quet annotate datasets/annotation-v2/human-value-01/review-queue.jsonl \
  --schema configs/annotation-v2-value.quet.yaml \
  --out datasets/annotation-v2/human-value-01/labels.jsonl
```

## Fields

| what you decide | Quet field |
|---|---|
| the amount is in the note, or not | type `amount` / `no_amount` |
| the value span | the `target` span (`no_amount`: leave it empty) |
| sure / unsure / not worth labelling | status `complete` / `uncertain` / `skipped` |
| why you are unsure, or anything odd | `note` (required for `uncertain`) |

The span is the **minimal full amount expression**: the number plus the unit, slang or currency
marker attached to it. It is an exact substring of the note with the original offsets (Python
code points; Quet records them), no leading or trailing space. **Never normalise**: `5 xị` stays
`5 xị`, `1tr5` stays `1tr5`; do not rewrite it as 500000 or 1500000, do not fix typos or case.

## Examples

| note | label |
|---|---|
| `cơm tấm 100` | `amount`, span `100` (a bare number is money when nothing else explains it) |
| `mượn chú hai 5 xị` | `amount`, span `5 xị` |
| `tiền điện tháng 10 hết 612` | `amount`, span `612` (`tháng 10` is a month) |
| `20/10 mua quà mẹ 500` | `amount`, span `500` (`20/10` is a date) |
| `họp t6 2025 mua 100k` | `amount`, span `100k` (month and year are not amounts) |
| `mua 3 vé 150k` | `amount`, span `150k` (`3 vé` is a quantity) |
| `ăn 2 tô phở 70` | `amount`, span `70` (item count `2`) |
| `trả góp kỳ 3 1tr5` | `amount`, span `1tr5` (`kỳ 3` is the installment/index) |
| `đóng đợt 2 khoản vay 3tr` | `amount`, span `3tr` |
| `chuyển khoản stk 123456789 200k` | `amount`, span `200k` (the account number is a reference) |
| `mã đơn 884213 hoàn 90k` | `amount`, span `90k` (order/reference numbers are not amounts) |
| `tầm 50k/tháng` | `amount`, span `50k` (no approximator, no `/tháng`) |
| `gói 1 triệu 2` | `amount`, span `1 triệu 2` (one compound expression) |
| `thưởng tết 2 tháng lương` | `no_amount`, `complete` (`2 tháng` is a period) |
| `cho mượn tiền ăn` | `no_amount`, `complete` (no number, no stated amount) |
| `bốn triệu tiền nhà` | `amount`, span `bốn triệu` (spelled-out amounts count) |

## Judgement rules

* **Several numbers**: pick the one that is the transaction amount; dates, months, years,
  quantities, item counts, installment or period indices, percentages, times, identifiers and
  model numbers are never part of the span.
* **Several money amounts and none is clearly the transaction amount** (a bill and a tip, a total
  and a share): status `uncertain` with a `note` saying which candidates compete. Never pick one
  to be safe.
* **Bare numbers** (`100`, `500`, no unit) are money when nothing else explains them; if you
  cannot tell money from a count or a reference, mark `uncertain` with a note.
* **Null**: a note that states no amount at all (no number, or only dates, periods or
  quantities) is `no_amount` with status `complete`.
* **Ambiguous or undelimitable** (the amount is unreadable, truncated, or you cannot say where it
  starts or ends): `uncertain` with a note. Use `skipped` only for text that is not worth
  labelling at all (garbage, not a note).
* When you change your mind or find an oddity (strange spacing, mixed languages), say so in the
  `note`; do not change the span to compensate.
