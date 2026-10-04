# Annotation v3

annotation-v3 = [annotation-v2](annotation-v2.md) (type, target span, value span) with **one**
taxonomy change: a **debt-only note** is `borrow` or `lend`, not `skipped`.

A debt-only note states an existing debt and no money moves now (`còn nợ Hùng 300k`,
`Hoa còn nợ mình 350k`, `Phúc chưa trả 1tr5`).

## What changed vs v2

| | annotation-v1 / v2 | annotation-v3 |
|---|---|---|
| debt-only, the user owes | `skipped` | `borrow` |
| debt-only, the other party owes the user | `skipped` | `lend` |
| debt-only, direction unclear | `skipped` | `uncertain` + `note` |

Everything else is unchanged: other types, the target rules, the value span convention, null
rules and trainable statuses. The new machine-readable files are
[`configs/annotation-v3.yaml`](../configs/annotation-v3.yaml) and the combined Quet schema
[`configs/annotation-v3.quet.yaml`](../configs/annotation-v3.quet.yaml).

## Why

Both sides of a debt note carry the information the user wants: who is the counterparty, which
way the money is owed and how much. `skipped` threw all of it away and left the model with a
23-note hole in a class (`skipped`) that is otherwise "not a finance note". A debt note is the
same relationship as `borrow` / `lend`, only without the movement of money in the note itself.

## Label rules

* **Type.** The user owes -> `borrow`. The other party owes the user -> `lend`.
* **Target.** The counterparty by the usual v1 rules: a kinship/title prefix before a proper name
  is dropped (`còn nợ chị Mai 700k` -> `Mai`), a role noun that is the only name is kept
  (`nợ chủ trọ tiền điện 280k` -> `chủ trọ`, `no tien nha ba chu` -> `ba chu`), birth-order
  names stay whole (`chú Tám`, `dì Út`). Bill names are not targets: `tiền điện`, `tiền vé`,
  `tiền ăn` -> `null` unless a person or business is named too.
* **Value.** The usual value span; `null` when the note has no amount.
* **Who is the subject.** `nợ X`, `còn nợ X`, `chưa trả X`, `hứa trả X` with an implied subject
  mean the **user** owes X. `X nợ mình`, `X còn thiếu`, `X chưa trả`, `X hứa trả` mean X owes the
  user.
* **Unclear direction -> `uncertain` with a note** (never forced): `còn nợ 500k` (no
  counterparty, no direction), `tiền nợ Lan 1tr5`, `nợ nhau với Bảo 150k`, `Hương 700k chưa trả`,
  `công nợ Minh 2tr`.

## Examples

| note | status | type | target | value |
|---|---|---|---|---|
| `còn nợ Hùng 300k` | complete | `borrow` | `Hùng` | `300k` |
| `nợ chủ trọ tiền điện 280k` | complete | `borrow` | `chủ trọ` | `280k` |
| `chưa trả anh Việt 600k tiền cafe` | complete | `borrow` | `Việt` | `600k` |
| `còn nợ Phương tiền cafe` | complete | `borrow` | `Phương` | null |
| `Hoa còn nợ mình 350k` | complete | `lend` | `Hoa` | `350k` |
| `Phúc chưa trả 1tr5` | complete | `lend` | `Phúc` | `1tr5` |
| `thằng Tuấn nợ t 200k` | complete | `lend` | `Tuấn` | `200k` |
| `Lộc quên trả mình tiền xăng` | complete | `lend` | `Lộc` | null |
| `tiền nợ Lan 1tr5` | uncertain | — | `Lan` | `1tr5` |

Not debt-only (unchanged): `trả nợ chị Hoa 1tr` is `repayment_out`, `Hải trả nợ 500k rồi` is
`repayment_in`, `mượn Hùng 500k` is `borrow` because the principal is received now.
`trả tiền bác Sáu mượn lần trước 3tr` is a repayment, not a debt-only note.

## Re-label batch `debt-01`

`datasets/annotation-v3/debt-01/` (builder: `scripts/build_debt_relabel_queue.py`, `--check`
verifies it) holds the 14 annotation-v1 `skipped` debt notes outside the frozen held-out set
`datasets/annotation-v2/human-value-01` plus hand-written new debt-only notes that passed the
near-duplicate leakage filter. The 9 v1 `skipped` notes inside the held-out set are never
relabelled and never trained on. `proposals.jsonl` is advisory LLM output; a human decides in
Quet. In this pass **`skipped` means "reject this note"** (it turned out not to be a debt-only
note).
