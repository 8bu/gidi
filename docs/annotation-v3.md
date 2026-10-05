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

## Gift receivers

Clarification (user decision, 2026-10-05; no taxonomy change, it makes the v1 rule
"Beneficiary is not payee" explicit): **the receiver of a gift or ceremony money is the
counterparty, so it is the `target`.**

* Gift or ceremony-money notes name the gift itself (`quà`, `mừng`, `lì xì`, `biếu`, `tặng` +
  receiver): the named receiver is the target. A kinship/title prefix before a proper name is
  dropped, a kinship or role term that is the only identifier is kept:
  `quà sinh nhật bé Na 300k` -> `Na`, `quà sinh nhật Vy 300 nghìn` -> `Vy`,
  `mừng cưới Hoa 1 triệu` -> `Hoa`, `quà 20/10 cho mẹ` -> `mẹ`, `lì xì cháu 200k` -> `cháu`,
  `mua quà sinh nhật cho cô Lan 340k` -> `Lan`. No receiver named (`tiền mừng cưới 500k`,
  `mua quà sinh nhật 300k`) -> `null`.
* Buying a concrete item for someone is a purchase, not a gift to a counterparty, so the
  beneficiary stays out of the target: `mua vở bút cho bé 180k`, `mua hoa tặng mẹ`,
  `mua vòng tay cho mẹ`, `mua hoa quả biếu bà` -> `null` (the shop is the target when named:
  `mua nước hoa tặng vợ ở sephora` -> `sephora`).
* A gift note that also names a shop or marketplace (`mua quà cho bạn Minh ở shopee 280k`) has two
  plausible counterparties: `uncertain`, not a training note.

Applied to the data in run 5: `human-value-01` amendment-01 (one label,
`datasets/annotation-v2/human-value-01/amendment-01.json`), the relabel map
`datasets/annotation-v3/gift-relabel-01/`, and `contrast-04`.

## Test set `human-value-02`

`datasets/annotation-v3/human-value-02/` (builder: `scripts/build_human_value_02.py`, `--check`
verifies it) is the **test set** for the V8 pipeline under annotation-v3 (type + target encoder,
then the rule value parser), labelled by the user in one combined Quet pass
(`configs/annotation-v3.quet.yaml`, quet-web project `gidi-hv02`). 150 LLM-written notes
(`candidates.txt`, written from the rules and the reviewed-corpus style only, never from training
data): about 70 target the weak strata of human-value-01 batch 2 (bare number, multi number,
quantity, installment index, date, no amount, unusual whitespace / punctuation), about 80 are
general notes over all eight types (debt-only notes both ways, shops and brands, gifts, banks,
kinship targets). Every candidate is gated against human-value-01, the frozen test splits,
probe-v1, the annotation-v3 batches and training sets up to `training-v3`, the annotation-v1/v2
datasets and the whole `corpus/` tree with the near-duplicate rules of
`scripts/build_human_value_queue.py`; the dropped notes are listed in `manifest.json`. There is
**no proposals file**, no model or rule label, and the set is **never trained on**: later
training sets must gate against `review-queue.jsonl` (`id` = `hv02-<sha256(text)[:12]>`). In this
pass `skipped` means "reject this note as unusable".

### Batch B and LLM labels (200 notes)

The user will not label `human-value-02` and **explicitly approved LLM labels for this test set**.
This is an exception to the repo rule "never auto-generate human labels"; it applies to this set
only, and the labels are LLM labels, never human labels (the file names, `manifest-labels.json`
and the scorer say so). The 150 notes of `review-queue.jsonl` and their quet-web project
`gidi-hv02` are unchanged.

* **Batch B**: 50 new LLM-written notes (`candidates-b.txt`, `scripts/build_human_value_02_b.py`,
  `--check` verifies): 22 weak-stratum notes (bare number, multi number, quantity, installment,
  date, null, spacing/punctuation) and 28 general ones (8 gifts with a named receiver, 9
  debt-only notes both ways, 4 multi-word shop names, 7 other types). They are gated with the
  `human-value-02` leakage rules against every earlier reference group, the 150 batch-A notes and
  **every** `datasets/annotation-v3/*` set on disk (so `contrast-04`, `training-v5`,
  `gift-relabel-01` too). One candidate was swapped after the gate saw `contrast-04/source.jsonl`
  (`manifest-labels.json`, `swapped_note`). Output: `review-queue-b.jsonl` (`hv02-<sha12>` ids)
  and `review-queue-all.jsonl` (the 150 bytes unchanged + the 50 = **200**), `manifest-b.json`.
  Later training sets must gate against `review-queue-all.jsonl`; the set is never trained on.
* **Labellers** (`scripts/label_human_value_02_llm.py`; prompts, notes and raw output in
  `llm-labelling/`): A = Claude (`claude -p`, Opus), B = an independent LLM labeller (B).
  Each saw only the rules (`rules.md`: annotation-v1, v2 span
  convention, v3 debt-only rules, the combined-pass guide and the Quet schema; the gift-receiver
  and kinship-prefix rules are in them) and `{id, text}` of its notes, in an empty working
  directory with no tools; neither saw the other's labels, the strata, training data or any model
  output. Output: combined labels `{id, annotation_status, type, target, value, span_status,
  note}`.
* **Validation**: `a-labels.jsonl`, `b-labels.jsonl` and `labels.jsonl` pass
  `scripts/validate_annotations.py --config configs/annotation-v3.quet.yaml --queue
  datasets/annotation-v3/human-value-02/review-queue-all.jsonl` with 0 errors. Only mechanical
  offset errors were repaired (the span text is re-located in the note; the chosen words are never
  changed): A 2 value spans, B 5 target and 6 value spans.
* **Agreement** (`agreement.json`): status 97.5 %, type 97.5 %, target text 98.5 %, value text
  100 %, value status 99.5 %; **full label 191 / 200 = 95.5 %** (status, type, target, value and
  value status all equal). The 9 disagreements were judgement calls (product brand vs seller,
  bank fee target, bill split, advance direction, bank interest direction, `gửi`).
* **Final labels**: `labels.jsonl` = the 191 agreed labels as is plus the 9 others adjudicated by
  a third Claude pass that saw both labels (anonymised as X / Y in a hashed order) and the rules
  and picked one or `uncertain`. `labels-provenance.jsonl` marks each record `agreed` or
  `adjudicated` (with the pick and the reason). The adjudicator is the same model family as A
  and picked A's label all 9 times, so score the `agreed` subset too.
* **Scoring**: `scripts/score_human_value_02.py` reads `review-queue-all.jsonl` and `labels.jsonl`
  (`complete` labels only) and reports every system on all complete notes and on the `agreed`
  subset (`systems_agreed`, `verdict_agreed`). It is run once, after training run 5.
