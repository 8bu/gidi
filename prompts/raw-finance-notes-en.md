# Prompt: raw English personal-finance notes

English analogue of `prompts/raw-finance-notes.md`, for gidi-en (contract:
`docs/annotation-en-v1.md`). The model supplies unlabeled note text only: ids and provenance are
attached locally by the ingest script and labels come later from Claude+Clef (silver, no human
review). Replace `{{BATCH_SIZE}}`, `{{SEED_TOPIC}}` and `{{VARIANT}}` (one of `us`, `uk`,
`india`, `singapore`, `mixed`) before sending.

---

## Instructions

You are generating realistic English personal-finance notes: the short lines a person types into
a phone app or notes app for themselves right after spending, earning, owing, or moving money.
They are private jottings, not bookkeeping entries. Nobody edits them for grammar.

Write **{{BATCH_SIZE}}** notes in the `{{VARIANT}}` English variant (`mixed` = blend them). If
`{{SEED_TOPIC}}` is not empty, bias roughly a third of the batch toward it.

### Coverage (per 100 notes; keep rare types well represented)

| Type the note will be labelled as | Share | What it looks like |
|---|---|---|
| expense | 28 | food, groceries, transport, rent, bills, subscriptions, tips, gifts, medical, **money sent to family or friends** (`venmo'd mom 50`) |
| income | 12 | salary/paycheck/payday, freelance, client paid, bonus, sold something, gift received |
| lend | 11 | lent / fronted / covered for someone, `Sam owes me 20`, paid on someone's behalf |
| borrow | 9 | borrowed from someone, `IOU Mike 15`, `owe landlord 300`, paycheck/salary advance |
| repayment_in | 10 | `Mike paid me back 15`, `Sam sent my 20`, bare `Mike paid 15` |
| repayment_out | 8 | `paid Sam back 20`, credit card / loan / BNPL payment |
| transfer | 12 | own accounts only: to savings, between banks/wallets, ATM withdrawal, top up cash app, invest |
| refund | 5 | `amazon refund 24`, reimbursed by work, cashback |
| unclear / not a note | 5 | a name and a number, a vague line, a non-finance line (kept rare) |

Do not output the type. This table is only for you to balance the batch.

### Hard cases (write several of each per batch, in varied wording)

* **R1, who paid.** Bare person payer, nothing named: `mike paid 15`, `sis paid 30`, `Sam sent
  20` (someone repays). Employer/client or a stated purpose: `company paid oct 5k`, `client
  paid 500`, `dan paid for the logo work 400`. Explicit debt word: `kim settled up 40`.
  Reverse direction: `paid mike back 15`, `paid mike 15 for the haircut`.
* **R2, on behalf.** `covered lunch for sam 20`, `fronted priya 40 for tickets`, `paid for
  lena's ticket, she owes me 60`, `put the uber on my card for joe`. Return side: `sam paid me
  back for lunch`, `joe sent back the 15`. Contrast: `treated mom to dinner 50` (a gift).
* **R3, sending money.** `venmo'd mom 50`, `zelle to dad 200`, `sent my brother 100`, `paypal
  sis 40 for the gift`, `gave mom 200`, `paynow to aunty 50`, `upi to bhai 500`, plus loans
  `venmo'd sam 20 he'll pay back`. Contrast own accounts: `moved 500 to savings`, `cash app to
  chase 200`, `withdrew 100 cash`, `put 300 in index fund`.
* **R4, salary.** `got paid`, `payday!`, `paycheck 2400`, `salary came in`, `direct deposit
  3200`, `pay hit`. Contrast: `advance on paycheck 300`, `salary advance`, `payday loan 400`.
* **Debt-only**: `owe sam 20`, `iou mike 15`, `mike iou 15`, `sam owes me 20`, `still owe
  landlord 300`. Include a few with unclear direction (`iou 15`).
* **Splitting bills**: `split dinner my half 32`, `split the bill w jo she owes me 25`.
* **Tips / refunds**: `tip 5`, `tipped maria 10`, `amazon refund 24`, `work reimbursed flight
  300`, `got security deposit back 1200`.

### How the notes are written

Vary these deliberately; do not settle into one style.

- **Length**: 2-5 words about 60%; 6-9 words about 30%; 10-16 words about 10% (a short clause
  with a date, place, or reason). Never a paragraph.
- **Register**: mostly lowercase and unpunctuated; some Capitalised first words; a few with a
  trailing `.` or `!`; occasional ALL CAPS or emoji (at most 1 in 20, a single emoji).
- **Typos and noise**: one plausible slip per note at most, in roughly 1 note in 5: dropped or
  doubled letters (`lunhc`, `recieved`), missing space, `u` for `you`, `w/` for `with`,
  missing apostrophe (`venmod`, `venmo'd`, `venmoed`). No gibberish.
- **Slang and shorthand**: `bucks`, `quid`, `grand`, `k`, `lakh`, `bro`, `sis`, `mom`/`mum`/
  `mama`, `ma`, `dad`/`papa`, `bhai`, `lunch run`, `uber`, `doordash`, `deliveroo`, `grab`,
  `swiggy`, `zomato`, `tesco`, `trader joes`, `ntuc`, `hawker`, `starbucks`, `venmo`, `zelle`,
  `paypal`, `cash app`, `gpay`, `paynow`, `upi`, `revolut`, `monzo`.
- **Amount formats** (mix; keep each variant consistent in itself): `$20`, `20$`, `20 bucks`,
  `20 usd`, `£12`, `12 quid`, `rs 500`, `₹500`, `500 rupees`, `s$30`, `1.5k`, `2k`, `$2k`, `3
  grand`, `1,200`, `1200.50`, `4.50`, `a hundred`, `twenty`, and **no-currency bare numbers**
  (`paid rent 800`). About 8% of notes carry no amount at all (`got paid`, `payday`).
- **Names**: invented, common first names across backgrounds (Sam, Priya, Mike, Lena, Tunde,
  Mei, Raj, Joe, Kim, Dan, Maria, Ahmed), nicknames, kinship words (`mom`, `dad`, `bro`,
  `sis`, `grandma`, `uncle raj`, `aunty meena`), role nouns (`landlord`, `boss`, `client`,
  `company`), merchant names. Sometimes the name leads (`sam 20 lunch`), sometimes trails.
  **Never** a real public figure, celebrity, politician, or a real private person's full name.
- **Variants**: `uk` = quid, mum, `£`, tesco, monzo, `sorted`, `mate`; `us` = bucks, `$`, mom,
  venmo, zelle, `paycheck`; `india` = `rs`/`₹`, lakh, bhai/didi/papa, upi/gpay, swiggy,
  `salary credited`; `singapore` = `s$`, paynow, grab, hawker, ntuc, `lah`/`liao` sparingly.
- **Dates/context** appear occasionally (`today`, `yesterday`, `fri`, `9/20`, `oct rent`).

### Hard constraints

- English only. Each note stands alone; no dialogue, no translation, no explanation.
- **Text only.** No ids, sources, labels, types, tags, amount fields, confidences.
- `text` is a single line: no newline, no tab, no leading or trailing whitespace, Unicode NFC,
  non-empty.
- `text` is the **only** key on every object.
- Notes must be **distinct**: no near-duplicates, no re-ordering of the same words, no copying
  an example below, and do not reuse the same name+amount pair.
- Do not make the type obvious by always using the same verb: vary verbs and word order across
  the batch.

### Output format

One JSON object per line (JSONL), UTF-8, **no Markdown fence, no commentary, no numbering**:

```json
{"text": "lunch w sam 14.50"}
```

Emit exactly {{BATCH_SIZE}} lines and stop.

---

## Examples (do not reuse these; they only fix the style)

```json
{"text": "paid rent 800"}
{"text": "venmo'd mom 50"}
{"text": "lent Sam $20"}
{"text": "mike paid me back 15"}
{"text": "salary came in"}
{"text": "fronted priya 40 for tickets"}
{"text": "split dinner my half 32"}
{"text": "advance on paycheck 300"}
{"text": "moved 500 to savings"}
{"text": "amazon refund 24"}
{"text": "owe landlord 3 grand"}
{"text": "upi to bhai rs 500"}
```

## After generation

The model output is a scratch file, **not** a corpus file. Keep it outside `corpus/raw/`
(e.g. `/tmp/gidi-en-baseline-01.jsonl`). Ingest it, then validate:

```bash
uv run python scripts/ingest_generated_corpus.py /tmp/gidi-en-baseline-01.jsonl \
    --source claude --prompt-id en-baseline-01 --out corpus/raw/en-baseline-01.jsonl
uv run python scripts/validate_corpus.py corpus/raw/en-baseline-01.jsonl
```

Labels for these notes are silver (Claude + Clef, `configs/annotation-en-v1.yaml`) and go only
under `datasets/annotation-en-v1/`.
