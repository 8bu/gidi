# human-value-01 annotation guide

You label **three things per note in one pass: the transaction type, the target span (the
counterparty) and the value span (the exact substring that is the monetary amount)**. This is the
annotation-v2 combined pass: type and target are annotation-v1, unchanged
([`docs/annotation-v1.md`](../../../docs/annotation-v1.md)); the value is the annotation-v2 value
span ([`docs/annotation-v2.md`](../../../docs/annotation-v2.md), "The span convention" and
"Combined pass"). This set is the independent test of two models, so label what the note says,
nothing else. You never see model or rule output: there are no proposals, no prefill, no
suggestions. Do not run the models or the value parser to help yourself. The `strata` in the queue
only say why a note was sampled; they are not hints about the answer.

Quet command (no `--proposals`):

```sh
quet annotate datasets/annotation-v2/human-value-01/review-queue.jsonl \
  --schema configs/annotation-v2.quet.yaml \
  --out datasets/annotation-v2/human-value-01/labels.jsonl
```

Check the labels with `uv run python scripts/validate_annotations.py
datasets/annotation-v2/human-value-01/labels.jsonl --config configs/annotation-v2.quet.yaml
--queue datasets/annotation-v2/human-value-01/review-queue.jsonl`.

## How to label a note

1. Pick the **type** (`t` or `1`-`8`): the eight annotation-v1 types.
2. With the **target** span active, select the counterparty and confirm, or press `n` for no
   counterparty. `transfer` has no target (it is saved as null).
3. Press **Tab** to switch to the **value** span, select the amount, or press `n` if the note
   states no amount. Tab switches back.
4. Press `c` while the value span is active to switch the **value status** between `complete`
   (default) and `uncertain`.
5. `enter` saves the note as `complete`, `u` as `uncertain` (type, target or value is unclear),
   `s` skips it (type, target and value are saved as null).

| what you decide | Quet field |
|---|---|
| the transaction type | type (annotation-v1 taxonomy) |
| the counterparty, or none | the `target` span (`n` = null; always null for `transfer`) |
| the amount, or none stated | the `value` span (`n` = null) |
| sure / unsure about the amount alone | value status `complete` / `uncertain` (`c`) |
| sure / unsure / not worth labelling | status `complete` / `uncertain` / `skipped` |
| why you are unsure, or anything odd | `note` (required for status `uncertain` and for value status `uncertain`) |

Both spans are exact substrings of the note with the original offsets (Python code points; Quet
records them), no leading or trailing space. **Never normalise**: `5 xị` stays `5 xị`, `1tr5`
stays `1tr5`; do not rewrite it as 500000 or 1500000, do not fix typos or case.

The value span is the **minimal full amount expression**: the number plus the unit, slang or
currency marker attached to it.

## Value examples

| note | value |
|---|---|
| `cơm tấm 100` | span `100` (a bare number is money when nothing else explains it) |
| `mượn chú hai 5 xị` | span `5 xị` |
| `tiền điện tháng 10 hết 612` | span `612` (`tháng 10` is a month) |
| `20/10 mua quà mẹ 500` | span `500` (`20/10` is a date) |
| `họp t6 2025 mua 100k` | span `100k` (month and year are not amounts) |
| `mua 3 vé 150k` | span `150k` (`3 vé` is a quantity) |
| `ăn 2 tô phở 70` | span `70` (item count `2`) |
| `trả góp kỳ 3 1tr5` | span `1tr5` (`kỳ 3` is the installment/index) |
| `đóng đợt 2 khoản vay 3tr` | span `3tr` |
| `chuyển khoản stk 123456789 200k` | span `200k` (the account number is a reference) |
| `mã đơn 884213 hoàn 90k` | span `90k` (order/reference numbers are not amounts) |
| `tầm 50k/tháng` | span `50k` (no approximator, no `/tháng`) |
| `gói 1 triệu 2` | span `1 triệu 2` (one compound expression) |
| `thưởng tết 2 tháng lương` | value null, `complete` (`2 tháng` is a period) |
| `cho mượn tiền ăn` | value null, `complete` (no number, no stated amount) |
| `bốn triệu tiền nhà` | span `bốn triệu` (spelled-out amounts count) |

## Judgement rules for the value

* **Several numbers**: pick the one that is the transaction amount; dates, months, years,
  quantities, item counts, installment or period indices, percentages, times, identifiers and
  model numbers are never part of the span.
* **Several money amounts and none is clearly the transaction amount** (a bill and a tip, a total
  and a share): value status `uncertain` (`c`) with a `note` saying which candidates compete.
  Never pick one to be safe.
* **Bare numbers** (`100`, `500`, no unit) are money when nothing else explains them; if you
  cannot tell money from a count or a reference, mark the value `uncertain` with a note.
* **Null**: a note that states no amount at all (no number, or only dates, periods or
  quantities) has a null value (`n`) and stays `complete`.
* **Ambiguous or undelimitable** (the amount is unreadable, truncated, or you cannot say where it
  starts or ends): value status `uncertain` with a note. Use `skipped` only for text that is not
  worth labelling at all (garbage, not a note).
* When you change your mind or find an oddity (strange spacing, mixed languages), say so in the
  `note`; do not change the span to compensate.

Type and target rules (what is a counterparty, `uncertain` for an ambiguous direction, `skipped`
for debt-state notes) are in [`docs/annotation-v1.md`](../../../docs/annotation-v1.md); the value
status is independent of them: a note can be `complete` with an `uncertain` value only if the
type and target are settled and only the amount is in doubt.
