# Prompt: targeted annotation-v1 augmentation (targeted-annotation-v1-01)

Use this prompt to generate a small raw batch that fills weak annotation-v1 coverage found in
`experiments/annotation-v1-final-audit.md`. It follows every style rule and hard constraint of
[`raw-finance-notes.md`](raw-finance-notes.md); only the content mix differs. The model supplies
unlabeled note text only. The coverage buckets below steer generation; they are **not labels**.
Every note still goes through Quet corpus review and is annotated later against
`docs/annotation-v1.md`, where it may land in a different type, `uncertain`, or `skipped`.

---

## Instructions

You are generating realistic Vietnamese personal-finance notes: the short lines a person types
into a phone app or a notes app for themselves right after money moves. They are private
jottings, not bookkeeping entries. Nobody edits them for grammar. The writer is always the user.

Write about **155** notes, in these coverage buckets:

| bucket | notes | at least unaccented | what the note describes |
|---|---|---|---|
| money paid back to the writer | ~55 | 50% | another party repays money they owed the writer |
| refunds | ~45 | ~35% | money returned to the writer for earlier spending or tax |
| lending | ~20 | ~35% | the writer lends or advances money to someone |
| borrowing | ~20 | ~35% | the writer borrows or receives an advance |
| genuinely unclear | ~15 | ~35% | a money movement whose reason or direction the text does not state |

About 40% of the whole batch is unaccented. A few extra contrast notes (below) are allowed.

### Bucket details

**Money paid back to the writer.** Vary the wording: `X trả nợ`, `X trả lại`, `X ck trả`,
`thu nợ X`, `đòi được nợ X`, `X gửi lại tiền`, `nhận tiền X trả`, `X trả góp/trả dần` (an
instalment of what X owes the writer), partial and final repayments (`trả nốt`, `trả đủ`).
Subjects: first names, nicknames, initials, kinship/role words (`mẹ`, `em trai`, `bạn cùng
phòng`, `đồng nghiệp`), titles before names, and titles before birth-order names (`anh Bảy`,
`cô Út`). Include a few near-opposite contrast notes that describe the writer paying back
instead (`trả nợ Hùng 500k` next to `Hùng trả nợ 500k`), so direction, not keywords, decides.

**Refunds.** Go well beyond one marketplace: cancelled orders, returned goods, merchant and
service-cancellation refunds (airlines, cinemas, ride-hailing, subscriptions), deposits returned
(`hoàn cọc`, landlord returning a deposit), employer reimbursement of work spending, tax refunds,
insurance claim payouts, cashback from a named card issuer or wallet, marketplace refunds. Vary
whether a counterparty is named.

**Lending.** `cho X vay/mượn`, `X mượn`, `ứng cho X`, `X hỏi mượn … đã ck`, informal and
unaccented forms, family / friend / colleague / role nouns. Always make the loan explicit
(`vay`, `mượn`, `ứng`, a promise to pay back); a bare `cho X 500k` belongs in the unclear bucket.

**Borrowing.** `vay X`, `mượn X`, `X cho mượn`, salary advance (`ứng lương`), borrowing from a
bank or lending app (`vay tín chấp`, `vay app`), overdraft drawn on a bank account, slang and
unaccented forms. The note must describe receiving the money, not only owing it.

**Genuinely unclear.** Real notes a person might write where the text does not say why money
moved or in which direction: `X gửi 500k` with no reason, `trả X 300k` with no reason, `cho X
200k`, `ck cho X`, `X ck hộ Y`, `ứng 2tr` with no party, `cty gửi 780k`. Each must be a
plausible jotting, not a riddle, nonsense, or a note that is merely hard to read.

### How the notes are written

Same as `raw-finance-notes.md`: full diacritics or plain unaccented ASCII (no telex), teencode and
abbreviations (`k`, `tr`, `củ`, `ck`, `dc`, `ko`, `r`, `nha`, `stk`), mixed amount formats, light
typos (one plausible slip per note), casing drift, occasional dates, mostly 2–8 words. Natural
Vietnamese/English mixing where people really do it (`refund`, `cashback`, `done`, `ok`,
`deposit`). Vary names and amounts; do not repeat a sentence frame with only the name or amount
swapped more than twice.

### Hard constraints

Identical to `raw-finance-notes.md`: Vietnamese only, one standalone note per line, text only
(no ids, labels, types, buckets, or amounts as fields), single line, NFC, `text` is the only key,
no near-duplicates, no copying examples. Do not emit the bucket name.

### Output format

One JSON object per line, UTF-8, no Markdown fence, no commentary:

```json
{"text": "Hùng ck trả 500k"}
```

---

## After generation

Write the output to a scratch file outside `corpus/raw/` (e.g.
`generated/targeted-annotation-v1-01.jsonl`), then ingest and validate:

```bash
uv run python scripts/ingest_generated_corpus.py generated/targeted-annotation-v1-01.jsonl \
    --source claude --prompt-id targeted-annotation-v1-01 \
    --out corpus/raw/targeted-annotation-v1-01.jsonl
uv run python scripts/validate_corpus.py corpus/raw/targeted-annotation-v1-01.jsonl
```

Then review it in Quet like any raw batch (`quet corpus/raw/targeted-annotation-v1-01.jsonl`).
Bucket intent is never copied into labels.
