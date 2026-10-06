# gidi-finance-v3 field issues

Issues found while using the notes app on gidi.8bu.dev (model `gidi-finance-v3` 3.0.0). Collected
here to fix in one batch (data round, retrain, v3.1). Add new cases at the end of the open list.

## Open

| # | note | model | expected | cause |
|---|---|---|---|---|
| 1 | `Chuyển cho họ hàng 2 củ` | `transfer` | `expense` (`lend` only if a loan is written) | `chuyển` is learned as a transfer cue; `transfer` is only between the user's own accounts, wallets, savings |
| 2 | `trả giùm tiền cơm Khoa 50k` | `expense` | `lend`, target `Khoa` | paying on someone's behalf: one `trả hộ` example in training (`trả hộ Nam tiền điện tháng này 650k` -> `lend`), none with `giùm` |
| 3 | `moi nhan luong 26cu` | `expense` | `income` | no-diacritic salary note; training-v5 has `luong ve …` / `nhan tien …` as `income` but no `nhan luong`; the glued slang `26cu` is rare |
| 4 | `moi lanh luong 26tr` | `borrow` | `income` | `lãnh`/`lanh` (receive, southern) never appears in training; `luong` co-occurs with `ung luong` -> `borrow` |
| 5 | `long tra 5 lit` | `income`, no target | `repayment_in`, target `Long` | lowercase, no-diacritic name `long` is missed as the subject; type needs the `X trả …` rule below |

### Proposed rule: `X trả …` (needs a new annotation version)

The payer decides. Today annotation-v1 marks a bare `X trả …` as `uncertain` (row 17).

- `X` is a person (name, kinship term) and the note gives no goods, service or work:
  `repayment_in`, target `X` (`Long tra 5 lit`, `chị Linh trả 2tr`).
- `X` is an employer, client or customer, or the note says what the money pays for (wages, a
  month, an order, goods): `income` (`cong ty tra tien t10 5 lit`, `khach tra tien 500k`).
- An explicit debt word still wins: `X trả nợ / trả lại …` -> `repayment_in`.

### Planned fix (one batch)

- Contrast notes for `X trả …`: person payer (`Long tra 5 lit`, `chị Linh trả 2tr`) ->
  `repayment_in`, against employer/customer or stated purpose (`cong ty tra tien t10 5 lit`,
  `khach tra tien 500k`) -> `income`; include lowercase, no-diacritic names.

- Contrast notes for paying on behalf: `trả giùm / trả hộ / ứng giùm / mua hộ X …` -> `lend`,
  target X; and the return side `X trả lại tiền mình trả giùm` -> `repayment_in`.
- Contrast notes for sending money to other people: `chuyển cho mẹ / họ hàng / bạn …` ->
  `expense` (or `lend` when a loan is stated), against `chuyển sang ví / tài khoản tiết kiệm` ->
  `transfer`.
- Salary notes with and without diacritics: `mới nhận lương / lãnh lương / lương về / có lương`
  -> `income` (also `moi nhan luong`, `lanh luong`, glued slang `26cu`), against `ứng lương`
  -> `borrow`. Training-v5 has 439 of 1,123 notes without diacritics but no `lãnh` at all.
- Annotate in Quet, validate, retrain, release v3.1.

### Schedule: v3.1 batch, 2026-10-07

Human labels gate the retrain, so the day runs in this order:

1. Agent: write the new annotation version (config + doc) with the `X trả …` rule above; the
   other cases already fit the current rules and only need data.
2. Agent: draft contrast notes for every open case (raw text only, no labels), ingest and
   validate them, and build a Quet annotation queue.
3. User: review and label the queue in Quet.
4. Agent: validate labels, build the training split, retrain with the frozen run-4 recipe
   (seeds 1–3), score on human-value-02 and the field cases, compare with v3 under a rule written
   before the run.
5. Agent: move `tỏi` / `cành` / `chai` = 1M into `gidi.value_parser` (Python + TS port, version
   bump, parity check).
6. User decides on release v3.1; then export, publish, deploy.

Also pull the PostHog "Sửa loại: từ → sang" table (dashboard 2175888) for new field cases before
step 2.

## Fixed in the app (no model change)

| note | was | now | fix |
|---|---|---|---|
| `trúng 1 tỏi` | 1.000 ₫ | 1.000.000.000 ₫ | `playground/src/app/amount.ts` widens the span over `tỏi` (commit `247c700`) |
| `cho em 1 chai` | 100.000 ₫ | 1.000.000 ₫ | app reads `chai` as 1 million (commit `247c700`) |
| `tiêu 50 cành` | 50.000 ₫ | 5.000.000 ₫ | app widens the span over `cành` = 100k (commit `247c700`) |
| `moi mua xe 5 toi` | 5.000 ₫ | 5.000.000.000 ₫ | unaccented `toi` / `canh` read too (commit `84ba7ea`) |
| multi-line note | one record | one record per line | `sticky-composer.tsx` (commit `3b8ba60`) |

The value parser (`gidi.value_parser`, version "1") is pinned by the v3 bundle, so `tỏi`, `cành`
and `chai` = 1 million are app-side only. Move them into the parser (Python and the TS port, with
a parser version bump) in the v3.1 release.
