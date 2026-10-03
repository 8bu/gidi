# targeted-annotation-v1-02: generation and annotation report

targeted-02 is a training augmentation for the nine gaps confirmed by
`experiments/baseline-v1/probe-report.md`. Nothing was trained.

- It is **not merged** into annotation-v1. `datasets/annotation-v1/combined/` and `splits/` are
  unchanged.
- probe-v1, validation and test were never a source. The generators were forbidden to open them
  and worked only from the pattern descriptions.

| file (`datasets/annotation-v1/targeted-02/`) | content |
|---|---|
| `notes.jsonl` | `{id, text, block, patterns, group, pair_role, intended_type, intended_target}`; generation intent is weak metadata, never a label |
| `queue.jsonl` | `{id, text, corpus, position}`, shuffled with seed `targeted-annotation-v1-02` |
| `labels.jsonl` | 106 auto-accepted annotation-v1 labels |
| `provenance.jsonl` | 112 rows: origin `synthetic-targeted`, batch, block, patterns, group, pair role, generation intent, `confidence` (pass 1), `review_confidence` (pass 2), `settled_by`, `review_state`, `human_reviewed` |
| `recheck.jsonl`, `proposals.jsonl` | the 6 escalated exceptions |
| `manifest.json` | counts and sha256 hashes |

## Pipeline

1. **Generation.** Six independent generators wrote blocks A–F from the probe pattern
   descriptions and the frozen doc. Each was given counts, the direction contrasts, a 50/50
   accent target and a minimal-pair structure. Each generator self-checked with
   `scripts/check_note_similarity.py`, which hides evaluation neighbour texts so that a rewrite
   cannot copy them. At least 4 drafts were rewritten at this stage (blocks B, C, D, E); the
   exact in-loop count was not retained.
2. **Central QC** over the 112 drafts, followed by every rewrite:
   - the similarity/template check against the full batch;
   - an LLM judge for naturalness, finance content and consistency with the intended pattern;
   - an LLM paraphrase judge comparing each note with its three nearest evaluation notes;
   - a manual read of every note.
3. **Two blind annotation passes.** The annotators saw only `queue.jsonl` and the frozen doc and
   config, not the intent. Pass 2 worked in reverse order. Both validate with 0 errors.
4. **Auto-accept or escalate.** Rules are below.

## QC

| | count |
|---|---|
| candidates through central QC | 136 (112 first drafts + 24 rewrites) |
| rejected at central QC | 24 |
| · semantic: not clearly a loan (`chủ trọ cho ứng tạm tiền điện`) | 1 |
| · too close to an evaluation note's frame (entity/amount/verb swap), incl. pair partners | 23 in 16 groups; nearest evaluation set: probe-v1 10, test 8, validation 5 |
| accepted into targeted-02 | 112 |
| malformed text (non-NFC, padding, target not a substring) | 0 |
| exact or normalised duplicates (batch, train, eval) | 0 |
| template collapse (a skeleton reused across > 2 groups) | 0; the most reused skeleton appears twice |
| judge: natural / finance (min P(yes)) | 0.73 / 0.63; none rejected (the 0.63 is `chi gui 1tr2 qua sinh nhat`, read manually as fine) |
| judge: intent consistency < 0.5 | 11, all reviewed manually and kept. They are exactly the taught contrasts (subject-first lend, `lì xì <kin>` → expense, `<kin> cho` → income, named-insurer payout); the small judge shares the models' direction bias |
| paraphrase judge ≥ 0.5 after rewrites | 2, kept: `e cho chu Tin vay 5tr sua xe` shares only the generic `cho X mượn/vay N <reason>` lend frame (35 train notes, per the error audit); `500k tiền trúng thưởng bốc thăm khai trương` shares one phrase with a different structure |

### Similarity diagnostics (final 112)

Normalised as in `gidi.annotation.split.normalize`. The skeleton also masks the counterparty
and capitalised names, so entity and amount swaps collide. Similarity is SequenceMatcher on
normalised text. Paraphrase was checked semantically by the judge above.

| reference | notes | exact | same bag | same skeleton | median | p90 | max | ≥ 0.80 |
|---|---|---|---|---|---|---|---|---|
| train (split) | 501 | 0 | 0 | 0 | 0.60 | 0.68 | 0.78 | 0 |
| validation | 110 | 0 | 0 | 0 | 0.56 | 0.67 | 0.82 | 1 |
| test | 105 | 0 | 0 | 0 | 0.56 | 0.67 | 0.75 | 0 |
| probe-v1 | 81 | 0 | 0 | 0 | 0.62 | 0.73 | 0.85 | 3 |
| targeted-01 (all) | 163 | 0 | 0 | 0 | 0.57 | 0.67 | 0.78 | 0 |

Within the batch, the highest similarity between notes of different groups is
0.77. Near-identical notes occur only inside minimal-pair groups, by design.

## Annotation

- **Agreement:** the two passes disagree on 0 notes in status, type or span (all 112 complete).
- **Generation intent vs the agreed labels:**
  - **type mismatches:** 0;
  - **span mismatches:** 4.
    - `vietlott` ×2: intended null, annotated `vietlott`. Both are escalated.
    - `bảo việt nhân thọ` ×2: intended `bảo việt`, annotated as the full company name. Accepted,
      because that is the named insurer.
- **Auto-accept rule:** both passes agree exactly, every rule-sensitive decision cites a frozen
  rule or edge row, the target span is clear, no QC concern remains, and the lower of the two
  confidences is ≥ 0.70.
  - Below 0.70, a note is accepted only when a specific frozen rule or human ruling settles it.
    4 notes qualified, listed below.
  - Confidence of the accepted 106: 68 at ≥ 0.85, 34 at
    0.70–0.84, 4 below 0.70 (rule-settled).

| note | label | basis |
|---|---|---|
| `biếu ba 2 triệu mua thuốc bổ` | expense / `ba` | gift/ceremony money → recipient target (doc: `lì xì cháu 200k` → `cháu`); cash given, so row 21 (goods bought as a gift) does not apply |
| `sáng mùng 1 lì xì bà nội 200k lấy hên` | expense / `bà nội` | `lì xì <kin>` = user gives gift money → expense, recipient target (doc: `lì xì cháu 200k` → `cháu`) |
| `duoc chu Nam cho 2 cu mung do dai hoc` | income / `Nam` | unaccented form that reads as a given name follows the proper-name rule (doc: `cau Tu` → `Tu`) → `Nam`; `<kin> cho` gift → income (edge rows 3/12) |
| `cho anh hai 700k đi chơi nhân dịp ra trường` | expense / `anh hai` | `cho <kin> N` with an explicit gift occasion → expense (human ruling on targeted-01 record 82 `cho em 300k`); `anh hai` kept (doc: `cho anh hai muon 1tr5` → `anh hai`) |

**Escalated to Quet (6).** Each is a genuine rule gap that the frozen doc does not settle:

| note | proposal | conf | open question |
|---|---|---|---|
| `chu tiem tap hoa cho muon 1tr5 xoay tien hang` | borrow / `chu tiem tap hoa` | 0.60 | role-noun lender `chu tiem tap hoa`: the whole owner phrase vs the shop noun; the doc has no `chủ <shop>` lender example |
| `do ve vietlott trung giai ba 300k` | income / `vietlott` | 0.55 | lottery operator named with the ticket: `vietlott` as the paying counterparty vs product name (null); no doc rule, and the pair partner depends on it |
| `zalopay boc tham trung 80k` | income / `zalopay` | 0.60 | a platform running a lucky draw pays the prize: counterparty target (cf. `momo hoàn` → `momo`) vs channel (null) |
| `mua ve vietlott ky nay 30k` | expense / `vietlott` | 0.55 | ticket purchase naming the operator: `vietlott` as the seller (target) vs product name (null); pair partner of the prize note |
| `trúng thưởng bốc số ở momo 70k` | income / `momo` | 0.60 | `ở momo` names where the draw ran: momo as the paying counterparty vs a venue/channel (null) |
| `cau Hai vay 3 trieu, thang sau tra` | lend / `cau Hai` | 0.60 | birth-order rule (`anh hai` kept whole) vs unaccented given-name rule (`cau Tu` → `Tu`): unaccented `Hai` can be `Hải`, which would give target `Hai` |

## Distribution

Counts include the 6 proposals; their types are not in question, only their spans.

| block | notes | accented / unaccented | types | pair groups | auto-accepted | escalated |
|---|---|---|---|---|---|---|
| A lender-first + title/name | 28 | 14 / 14 | borrow 20, lend 8 | 8 | 27 | 1 |
| B subject-first direction | 14 | 7 / 7 | borrow 4, lend 10 | 4 | 13 | 1 |
| C gift direction | 28 | 14 / 14 | expense 14, income 14 | 10 | 28 | 0 |
| D insurance | 16 | 8 / 8 | expense 6, refund 10 | 6 | 16 | 0 |
| E windfall / lottery | 15 | 7 / 8 | expense 3, income 12 | 3 | 11 | 4 |
| F loan installment | 11 | 5 / 6 | borrow 3, repayment_out 8 | 3 | 11 | 0 |
| **total** | 112 | 55 / 57 | expense 23, income 26, borrow 27, lend 18, repayment_out 8, refund 10 | 34 | 106 | 6 |

Across the whole batch, 55 notes are accented and 57 unaccented. Every block is within one
note of 50/50.

| # | probe pattern | probe class | notes | accented | types |
|---|---|---|---|---|---|
| 1 | `lender_first_cho_muon` | A | 28 | 14 | borrow 20, lend 8 |
| 2 | `subject_first_muon` | B | 14 | 7 | borrow 4, lend 10 |
| 3 | `family_gift_in` | A | 14 | 7 | income 14 |
| 4 | `gift_out` | A | 14 | 7 | expense 14 |
| 5 | `insurance_premium` | B | 6 | 3 | expense 6 |
| 6 | `insurance_payout` | A | 10 | 5 | refund 10 |
| 7 | `windfall` | A | 15 | 7 | expense 3, income 12 |
| 8 | `loan_installment` | B | 11 | 5 | borrow 3, repayment_out 8 |
| 9 | `title_name_span` | A | 36 | 18 | expense 5, income 6, borrow 14, lend 11 |

- **Title boundary:** 36 notes carry `title_name_span`.
  - 17 strip the title before a proper name (`Hạnh`, `Khôi`, `Tin`, `Lộc`, `Xuân`, `Bao`, `Binh`,
    `Hùng`, `Tuan`, `Lan`, `Nam` …).
  - 19 keep a title + birth-order phrase (`bác Hai`, `dì Út`, `chú Năm`, `cô Bảy`, `chi ba`,
    `anh ba`, `chu Ba`, `mợ Hai`, `di Sau` …).
  - The same titles (`chú`, `bác`, `cô`/`chi`, `dì`) appear in both forms.

## Minimal pairs and groups

- **Groups:** 78 in total: 34 minimal-pair groups (68 notes) and 44 singles.
  - All 34 pairs flip the type under the agreed labels, as intended.
  - The pairs cover `X cho mượn` ↔ `cho X mượn`, `X mượn` ↔ `mượn X`,
    `<kin> lì xì/cho/gửi/mừng tuổi/tặng/ck` ↔ the outgoing form, premium ↔ payout with the same
    insurer wording, `trúng` ↔ `mua`, and installment ↔ drawdown with the same provider.
  - The cue verbs `lì xì`, `cho`, `gửi`, `mừng tuổi`, `mượn` and `vay` each occur in both
    directions.
- **Where the group lives:** `group` is stored in `notes.jsonl` and `provenance.jsonl`.
  - `build_splits` does not read it yet. Mirror pairs are still kept together by its existing
    bag-of-words rule, but not every pair is a word-for-word mirror (for example
    `trung ve so chieu nay 2tr` / `mua ve so chieu nay 50k`).
  - If targeted-02 is ever placed into a split, the split must treat `group` as one unit.

## Human review

Six notes need review in Quet:

```sh
quet annotate datasets/annotation-v1/targeted-02/recheck.jsonl \
  --schema configs/annotation-v1.yaml \
  --out datasets/annotation-v1/targeted-02/human-labels.jsonl \
  --proposals datasets/annotation-v1/targeted-02/proposals.jsonl
```

After review, merge `human-labels.jsonl` into `labels.jsonl`, mark those provenance rows
`human_reviewed: true`, re-validate, and refresh `manifest.json`. targeted-02 then becomes
eligible for merging and retraining. Re-score both the test split and probe-v1 after
retraining.
