# human-value-02 annotation guide

You label **three things per note in one pass: the transaction type, the target span (the
counterparty) and the value span (the exact substring that is the monetary amount)**. This is the
annotation-v3 combined pass: type and target follow annotation-v1
([`docs/annotation-v1.md`](../../../docs/annotation-v1.md)), the value is the annotation-v2 value
span ([`docs/annotation-v2.md`](../../../docs/annotation-v2.md), "The span convention"), and the
one v3 change applies: a **debt-only note** is `borrow` (you owe) or `lend` (the other party owes
you), not skipped ([`docs/annotation-v3.md`](../../../docs/annotation-v3.md)).

This set is the independent **test** of the V8 pipeline (type + target encoder, then the rule
value parser), so label what the note says, nothing else. There are no proposals, no prefill, no
suggestions; the notes were written by an LLM and never seen by any model. Do not run the models
or the value parser to help yourself. The `strata` in the queue only say why a note was sampled;
they are not hints about the answer. The set is never used for training.

Quet command (no `--proposals`):

```sh
quet annotate datasets/annotation-v3/human-value-02/review-queue.jsonl \
  --schema configs/annotation-v3.quet.yaml \
  --out datasets/annotation-v3/human-value-02/labels.jsonl
```

On quet-web the same queue is project `gidi-hv02`. Check the labels with `uv run python
scripts/validate_annotations.py datasets/annotation-v3/human-value-02/labels.jsonl --config
configs/annotation-v3.quet.yaml --queue datasets/annotation-v3/human-value-02/review-queue.jsonl`.

## How to label a note

1. Pick the **type** (`t` or `1`-`8`): the eight types.
2. With the **target** span active, select the counterparty and confirm, or press `n` for no
   counterparty. `transfer` has no target (it is saved as null).
3. Press **Tab** to switch to the **value** span, select the amount, or press `n` if the note
   states no amount. Tab switches back.
4. Press `c` while the value span is active to switch the **value status** between `complete`
   (default) and `uncertain`.
5. `enter` saves the note as `complete`, `u` as `uncertain` (type, target or value is unclear),
   `s` skips it. In this pass **`skipped` means "reject this note as unusable"** (garbage, not a
   finance note): type, target and value are saved as null. A debt-only note is never skipped.

| what you decide | Quet field |
|---|---|
| the transaction type | type |
| the counterparty, or none | the `target` span (`n` = null; always null for `transfer`) |
| the amount, or none stated | the `value` span (`n` = null) |
| sure / unsure about the amount alone | value status `complete` / `uncertain` (`c`) |
| sure / unsure / unusable | status `complete` / `uncertain` / `skipped` |
| why you are unsure, or anything odd | `note` (required for status `uncertain` and for value status `uncertain`) |

Both spans are exact substrings of the note with the original offsets (Python code points; Quet
records them), no leading or trailing space. **Never normalise**: `5 xị` stays `5 xị`, `1tr5`
stays `1tr5`; do not rewrite it as 500000 or 1500000, do not fix typos or case. Strange spacing or
punctuation in a note (`grab   85k`, `cf 25k..`) is part of the test: label it as it is and say so
in the `note` if you like.

## Type and target (annotation-v1 + v3 debt-only)

* **Type.** The user owes and nothing moves now (`còn nợ Hùng 300k`) -> `borrow`; the other party
  owes the user (`Hoa còn nợ mình 350k`) -> `lend`; direction unclear -> `uncertain` with a note.
  Receiving principal (`mượn Hùng 500k`) is `borrow`, repaying is `repayment_out`; a bare
  `X trả 500k` with no sign of a prior debt is `uncertain`.
* **Target.** The external counterparty: a named shop, brand, platform, person, or a kinship/role
  noun that is the only identifier (`mẹ`, `khách`, `chủ trọ`). Drop a kinship/title prefix before a
  proper name (`chị Mai` -> `Mai`), keep birth-order names whole (`chú Tám`, `dì Út`). Never an
  item, dish, activity or bill name, a companion (`với`, `vs`), a beneficiary (`cho bé`, `tặng
  mẹ`), or a bank / wallet used only as the channel (`rút tiền vpbank` is a transfer: null). A gift
  or ceremony money goes to its recipient (`mừng cưới Tuấn` -> `Tuấn`). Several plausible
  counterparties -> `uncertain`.

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
| `chuyển khoản stk 123456789 200k` | span `200k` (the account number is a reference) |
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
  starts or ends): value status `uncertain` with a note.
* When you change your mind or find an oddity (strange spacing, mixed languages), say so in the
  `note`; do not change the span to compensate.

The value status is independent of type and target: a note can be `complete` with an `uncertain`
value only if the type and target are settled and only the amount is in doubt.
