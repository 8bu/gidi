# Prompt: raw Vietnamese personal-finance notes

Use this prompt to generate raw Vietnamese finance notes for Gidi. The model supplies unlabeled
note text only: ids and provenance are attached locally by the ingest script, and labeling
happens later, after Quet review. Replace `{{BATCH_SIZE}}` and `{{SEED_TOPIC}}` before sending.

---

## Instructions

You are generating realistic Vietnamese personal-finance notes: the short lines a person types
into a phone app or a notes app for themselves right after spending, earning, or moving money.
They are private jottings, not bookkeeping entries. Nobody edits them for grammar.

Write **{{BATCH_SIZE}}** notes. Cover the whole range below; if `{{SEED_TOPIC}}` is not empty,
bias roughly half of the batch toward it and keep the rest spread across the other categories.

### Categories

- **spending**: food and drink, groceries, transport, fuel, rent, utilities, phone top-up,
  clothing, health, education, entertainment, gifts
- **income**: salary, freelance payment, bonus, interest, selling something, money received back
- **transfer**: moving money between the writer's own accounts or wallets, topping up an
  e-wallet, withdrawing cash
- **debt**: paying an installment, credit-card payment, money the writer owes or is owed
- **lending**: lending to or borrowing from a friend, relative, or colleague
- **savings**: putting money aside, gold, a savings deposit, an emergency fund

### How the notes are written

Vary these deliberately across the batch; do not settle into one style.

- **Diacritics**: most notes carry full diacritics (`ăn phở 45k`); a substantial minority are
  typed **without diacritics and without telex typing** — plain unaccented ASCII as if the
  keyboard were set to English (`an pho 45k`, `nhan luong 15tr`). Do not include telex typing,
  i.e. letter pairs or trailing tone letters that stand in for diacritics (`aw`, `aa`, `dd`,
  `ee`, `oo`, `ow`, `uw`, or a suffix like `phos` written for `phở`).
- **Teencode and abbreviations**: `k`, `tr`, `cf` (coffee), `ck` (chuyển khoản / transfer),
  `vs` (với / with), `mn` (mọi người), `nh`, `dc`, `ko`, `ok`, `ib`, `nt`, `tip`, `ship`,
  `bill`, `vcb`/`tcb`/`momo`, `stk`.
- **Amount formats**: `45k`, `45.000đ`, `45,000`, `1tr5`, `2tr`, `2 triệu`, `200 nghìn`,
  `1,5tr`, `1tr2`, `150k`, `3 củ`, `500`, bare numbers where a currency is implied, and
  occasional amounts written in words. Mix separators (`.` and `,`) and unit spellings.
- **People and places**: first names, nicknames, kinship words (`mẹ`, `anh`, `chị`, `em`),
  initials, merchant names, shop and app names (`grab`, `shopee`, `highlands`, `winmart`,
  `circle k`, `netflix`, `spotify`). Sometimes a name leads (`Lan 60k`), sometimes it trails
  (`cf vs Lan 60k`).
- **Typos and noise**: missing or doubled letters, a wrong tone mark, missing spaces, an extra
  trailing character, casing drift (`45K`, `45k`), stray punctuation. Keep typos light — one
  plausible slip per note, not gibberish.
- **Length**: mostly 2–8 words; a few longer ones with a date, a place, or a short clause.
- **Dates and context** appear occasionally (`sáng nay`, `hôm qua`, `t3`, `20/9`) but are not
  required.

### Hard constraints

- Vietnamese only. Each note stands alone; no dialogue, no translation, no explanation.
- **Text only.** Do not generate ids, sources, prompt ids, labels, transaction types, tags,
  amount fields, confidences, or any other annotation. Amounts stay inside the note text.
- `text` must be a single line: no newline, no tab, no leading or trailing whitespace, Unicode
  NFC, and non-empty.
- `text` is the **only** key on every object. Do not add keys, do not omit it.
- Notes must be **distinct**: no near-duplicates, no re-ordering the same words, no copying an
  example below.

### Output format

One JSON object per line (JSONL), UTF-8, Vietnamese kept as literal characters, **no Markdown
fence, no commentary, no numbering, no explanation**. Each line has exactly this shape:

```json
{"text": "cf vs Lan 60k"}
```

Emit exactly {{BATCH_SIZE}} lines and stop.

---

## Examples (do not reuse these; they only fix the style)

```json
{"text": "ăn phở 45k"}
{"text": "nhan luong 15tr"}
{"text": "ck mẹ 2 triệu"}
{"text": "trả tiền nhà 3tr5 tháng 9"}
{"text": "mượn a Hùng 500k"}
{"text": "grab đi làm 38.000d"}
{"text": "bỏ heo đất 200 nghìn"}
{"text": "shopee 1tr5 quần áo"}
```

## After generation

The model output is a temporary scratch file, **not** a corpus file. Keep it outside
`corpus/raw/`, for example `/tmp/gidi-baseline-01.jsonl`, and do not clean it up by hand into
the corpus tree.

Ingest it. The script attaches `id`, `source`, and `prompt_id`, and normalizes every note to NFC:

```bash
uv run python scripts/ingest_generated_corpus.py /tmp/gidi-baseline-01.jsonl \
    --source claude --prompt-id baseline-01 --out corpus/raw/baseline-01.jsonl
```

Only the ingest output lands in `corpus/raw/`. Ids are assigned locally rather than by the model
so they are stable and reproducible — re-running the same ingestion on the same generation
produces byte-identical ids, and model-invented or duplicated ids can never enter the corpus.

Then validate the ingested file:

```bash
uv run python scripts/validate_corpus.py corpus/raw/baseline-01.jsonl
```

Fix anything the validator reports as an error, then hand the file to Quet review.
