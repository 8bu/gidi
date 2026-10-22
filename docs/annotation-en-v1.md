# Annotation en-v1

annotation-en-v1 is the English contract for **gidi-en**. It keeps the taxonomy and semantic rules
of [annotation-v4](annotation-v4.md) (R1-R4) and re-expresses them with English notes.
Machine-readable: [`configs/annotation-en-v1.yaml`](../configs/annotation-en-v1.yaml). It is a new
annotation version; the Vietnamese configs are untouched.

> **Clarification 2026-10-21.** The example row `uber home $18` was `null`, which contradicted the
> inherited rule that named merchants, services and platforms are targets (config comment:
> "named merchants/services when they are the counterparty"). The row now reads `uber`. The rule
> did not change; no existing label was relabelled (labels were produced with this rule applied).

## Provenance: silver labels only

There are **no human labels** for English. Every gidi-en label is produced by Claude and Cloudflare
Clef (few-shot kNN teacher, see `scripts/clef_teacher.py`). These labels are *silver*:

* they live only under `datasets/annotation-en-v1/` and are never mixed with the human Vietnamese
  labels or presented as human;
* any eval number is **agreement with Claude+Clef labels**, not accuracy against ground truth.

## Inherited vs re-stated

| Part | Status |
|---|---|
| Eight types (`expense`, `income`, `borrow`, `lend`, `repayment_in`, `repayment_out`, `transfer`, `refund`), no `other` | inherited from v4 |
| Statuses `complete` / `uncertain` / `skipped`, note and null rules, trainable = `complete` | inherited |
| Debt-only notes are `borrow` / `lend` (v3) | inherited |
| Target = exact substring, null when none; `transfer` target always null | inherited |
| Value span convention (amount substring, no normalisation) | inherited |
| R1-R4 and every type text | **re-stated** with English examples |
| Target rules (names, titles, channels) | **re-stated** for English |
| Payment apps, bill splitting, IOU, tip, paycheck vocabulary, English amount formats | **new** |

The config is self-contained because the validator does not follow `extends`.

## The four rules in English

**R1, `X paid ...`: the payer decides.**

* X is a person and nothing is named that the money pays for -> `repayment_in`, target X:
  `Mike paid 15`, `sam sent 20`, `sis paid 30`.
* X is an employer/client/customer, or the note says what it pays for -> `income`:
  `company paid oct 5k` (target `company`), `client paid 500` (`client`),
  `Dan paid for the logo work 400` (target `Dan`).
* An explicit debt word wins: `Mike paid me back 15`, `Sam repaid 20`, `Kim settled up 40` ->
  `repayment_in`.
* `paid X ...` (the user is the payer) is not R1: `paid Mike back 15` -> `repayment_out`;
  `paid Mike 15` for a stated service -> `expense`.

**R2, paying on someone's behalf -> `lend`, target the person.**
`covered lunch for Sam 20`, `fronted Sam 20`, `paid for Lena's ticket 60 she owes me`,
`put the Uber on my card for Joe 18`. Return side -> `repayment_in`: `Sam paid me back for lunch`,
`Joe sent back the 15`. Treating someone as a gift (`treated mom to dinner 50`) is `expense`;
`lend` needs a stated or clearly implied expectation to be repaid.

**R3, sending money to other people -> `expense`.** `venmo'd mom 50`, `zelle to dad 200`,
`sent my brother 100`, `gave mom 200` -> `expense`, target `mom` / `dad` / `brother`.
A stated loan -> `lend` (`venmo'd Sam 20 he'll pay me back`). `transfer` only for the user's own
holdings: `moved 500 to savings`, `cash app to chase 200`, `withdrew 100 cash`,
`bought 0.01 btc 300` (store of value).

**R4, salary phrasing.** `got paid`, `payday`, `paycheck`, `salary came in`, `direct deposit 2400`
-> `income`, no target. `advance on paycheck 300`, `salary advance`, `payday loan 400` ->
`borrow`.

## English-specific cases

| Note | type | target |
|---|---|---|
| `paid rent 800` | expense | null |
| `tip for the driver 5` | expense | null |
| `venmo'd mom 50` | expense | `mom` |
| `zelle dad 200 for groceries` | expense | `dad` |
| `lent Sam $20` | lend | `Sam` |
| `Sam owes me 20` | lend | `Sam` |
| `IOU Mike 15` | borrow | `Mike` (the user owes; see note) |
| `Mike IOU 15` | lend | `Mike` |
| `owe landlord 300` | borrow | `landlord` |
| `borrowed 50 from Tom` | borrow | `Tom` |
| `paid Sam back 20` | repayment_out | `Sam` |
| `Mike paid me back 15` | repayment_in | `Mike` |
| `Sam sent me my 20` | repayment_in | `Sam` |
| `covered lunch for Sam 20` | lend | `Sam` |
| `fronted Priya 40 for tickets` | lend | `Priya` |
| `split dinner, my half 32` | expense | null |
| `split the bill w Jo, she owes me 25` | lend | `Jo` |
| `salary came in 3200` | income | null |
| `got paid` | income | null |
| `advance on paycheck 300` | borrow | null |
| `boss paid bonus 500` | income | `boss` |
| `amazon refund 24` | refund | `amazon` |
| `work reimbursed flight 300` | refund | null |
| `cash app to chase 200` | transfer | null |
| `moved 500 to savings` | transfer | null |

Conventions behind the table:

* **Tips** are `expense`. A tip's recipient is a target only when named as a counterparty
  (`tipped Maria 10` -> `Maria`); a role noun used only to describe what the tip was for
  (`tip for the driver 5`, `tip 5`) is not a counterparty: target null.
* **`IOU X`** is ambiguous in direction. `IOU Mike 15` and `owe Mike 15` (the user owes) ->
  `borrow`; `Mike IOU 15`, `Mike owes me 15` (Mike owes) -> `lend`. A bare `IOU 15` or
  `iou mike` with no direction cue -> `uncertain` with a note.
* **Payment apps** (`venmo`, `zelle`, `paypal`, `cash app`, `gpay`, `apple pay`, `paynow`,
  `upi`, `revolut`, bank names) are channels, never targets. `venmo'd mom 50` -> `mom`.
* **Own share of a split** is `expense`; money others owe the user from a split is `lend` with
  that person; if several people owe, uncertain (several counterparties).
* **Titles**: `uncle Raj` -> `Raj`, `aunty Meena` -> `Meena`, `Mr Tan` -> `Tan`; `mom`, `bro`
  stay whole. `Mom` keeps its case.
* **Notes with no direction** (`Sam 20`, `lunch w sam 20`) follow the v4 rules: bare names with no
  verb -> `uncertain`; a companion is not a target.

## Amount formats (value span)

The span is the amount exactly as written: `$20`, `20 bucks`, `20$`, `20 usd`, `1.5k`, `2k`,
`3 grand`, `£12`, `rs 500`, `₹500`, `s$30`, `1,200`, `1200.50`, `twenty`, and bare numbers
(`paid rent 800`). No normalisation (`3 grand` is never turned into 3000). A note with no amount
has a null value span. Two unrelated amounts and no clear transaction amount -> value uncertain.

## Examples across the types

| Note | type | target |
|---|---|---|
| `coffee 4.50` | expense | null |
| `uber home $18` | expense | `uber` |
| `freelance gig paid 1.5k` | income | null |
| `client paid 500` | income | `client` |
| `sold my old bike 120` | income | null |
| `borrowed 200 from dad` | borrow | `dad` |
| `lent my roommate 30` | lend | `roommate` |
| `repaid dad 500` | repayment_out | `dad` |
| `paid off the credit card 300` | repayment_out | null |
| `Mike paid me back 15` | repayment_in | `Mike` |
| `ATM withdrawal 200` | transfer | null |
| `refund from zara 40` | refund | `zara` |
| `lol` | skipped | null |

## Inference: word-boundary snapping

The tag head works on pruned-BPE pieces and sometimes stops inside a word (`ader` for `adery`,
`pap` for `papa`). The English decode therefore supports **word snapping**
(`snap_words`): the predicted span is expanded outward to the whole whitespace-delimited word,
edge punctuation trimmed, never shrunk (`gidi.inference.decode.snap_to_words`, the same rule as
the Vietnamese v3 bundles and `playground/src/runtime/decode.ts`'s `snapToWords`).

* Evaluation: `scripts/evaluate_onnx.decode(..., snap_words=True)`; `evaluate_en_student.py`
  reports `int8` (raw BIO span) and `int8_snap` side by side, `evaluate_en_probe.py --snap-words`
  scores a probe with it. gidi-en v2 is scored with snapping on (`gidi-en-student-v2/protocol.json`).
* ONNX / web: snapping is **not** in the ONNX graph (the graph returns type logits and tag
  logits). A runtime must apply it after BIO decoding. `GidiPredictor(snap_words=True)` (Python)
  and the playground `predictor.ts` (always on for `target_snap: "words"`) already do; a gidi-en
  bundle config must set `"target_snap": "words"` to get it. Without it a runtime returns the raw
  BIO span, i.e. the `int8` numbers.
