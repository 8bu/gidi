# Annotation v4

annotation-v4 = [annotation-v3](annotation-v3.md) with **four clarified labelling rules**, taken
from the field issues in `experiments/v3-field-issues.md`. Machine-readable:
[`configs/annotation-v4.yaml`](../configs/annotation-v4.yaml) (extends v3; only the changed type
definitions and the rules are restated).

## What changed vs v3

| | annotation-v3 | annotation-v4 |
|---|---|---|
| `Long trả 5 lít` (person pays, nothing named) | undecided (`uncertain`) | R1 `repayment_in`, target `Long` |
| `công ty trả tiền t10 5 lít` | undecided | R1 `income`, target `công ty` |
| `trả giùm / mua hộ X ...` | mostly `lend`, not stated | R2 `lend`, target X |
| `chuyển cho mẹ 2 củ` | model confuses with `transfer` | R3 `expense`, target `mẹ` |
| `mới nhận lương 26cu`, `lãnh lương` | not listed | R4 `income`, no target |

## The four rules

**R1, `X trả ...`: the payer decides.**

* X is a person (name or kinship term) and the note names no goods, service or work ->
  `repayment_in`, target X: `Long tra 5 lit`, `long tra 5 lit`, `chị Linh trả 2tr`.
* X is an employer, client or customer, or the note says what the money pays for (wages, a month,
  an order, goods) -> `income`: `cong ty tra tien t10 5 lit`, `khach tra tien 500k`,
  `anh Minh tra tien cong son nha 4tr`. The target is the employer/client noun when named
  (`cong ty`, `khach`).
* An explicit debt word still wins: `X trả nợ / trả lại ...` -> `repayment_in`.

**R2, paying on someone's behalf.** `trả giùm / trả hộ / ứng giùm / ứng hộ / mua hộ / đóng hộ X`
-> `lend`, target X (`ứng giùm Lan tiền điện 400k`, `mua hộ Tú 200k`). The return side
(`Khoa trả lại tiền mình trả giùm 50k`, `Khoa gửi lại tiền cơm`) -> `repayment_in`, target X.

**R3, sending money to other people.** `chuyển / gửi / biếu cho mẹ, họ hàng, bạn, ba ...` ->
`expense`, target the person or kinship noun (`chuyển cho mẹ 2 củ` -> `mẹ`;
`chuyển cho họ hàng 2 củ` -> `họ hàng`), or `lend` when a loan is stated. `transfer` is only
between the user's own accounts, wallets, cash, savings or investments (`chuyển sang ví momo
1tr`, `gửi tiết kiệm 5tr`).

**R4, salary phrasing.** `nhận lương`, `lãnh lương` (southern), `lương về`, `có lương`,
`mới nhận lương 26cu` (also `moi nhan luong`, `moi lanh luong`) -> `income`, no target.
`ứng lương` / `tạm ứng` -> `borrow`.

## What stays the same

All other types and target rules, the value span, statuses (`complete`, `uncertain`, `skipped`),
null rules, trainable statuses and the Quet schemas of annotation-v3. Existing human labels are
not rewritten: `datasets/annotation-v3/distill-clef-02/rule-conflicts.jsonl` lists the few
training notes whose current label disagrees with R1-R4, for the user to decide.
