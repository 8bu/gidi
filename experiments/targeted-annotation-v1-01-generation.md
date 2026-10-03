# targeted-annotation-v1-01: generation report

Targeted raw batch for weak annotation-v1 coverage (`experiments/annotation-v1-final-audit.md`).
Prompt: `prompts/targeted-annotation-v1-01.md`, generator Claude, `prompt_id`
`targeted-annotation-v1-01`. Nothing here is a label: the buckets steer generation only, and
every note goes through Quet corpus review and later annotation-v1 annotation.

| file | role |
|---|---|
| `generated/targeted-annotation-v1-01.jsonl` | scratch model output, shuffled (seeded) |
| `generated/targeted-annotation-v1-01-buckets/*.jsonl` | scratch: generation intent per bucket, for this report only; never a label source |
| `corpus/raw/targeted-annotation-v1-01.jsonl` | canonical raw batch (ingested) |

## Intended distribution

| bucket | notes | accented | unaccented | unaccented % |
|---|---|---|---|---|
| repayment_in | 55 | 25 | 30 | 55% |
| contrast: repayment_out mirrors | 8 | 5 | 3 | 38% |
| refund | 45 | 28 | 17 | 38% |
| lend | 20 | 13 | 7 | 35% |
| borrow | 20 | 13 | 7 | 35% |
| uncertain | 15 | 9 | 6 | 40% |
| **total** | 163 | 93 | 70 | 42.9% |

The 8 contrast notes are direction mirrors of repayment_in notes (`Hùng trả nợ 500k` /
`trả nợ Hùng 500k`, `Quyên trả mình 250k` / `mình trả Quyên 250k`), added on top of the ~155
requested. Some notes were written as boundary cases on purpose (`X trả mình`, `mình trả X`,
`cho Lợi 1tr, hẹn cuối tháng trả`, a salary advance from `chủ quán`); their final status is
the annotator's call.

Variation covered: names and nicknames, kinship and role subjects, titles before names
and before birth-order names (`anh Bảy`, `cô Út`, `dì Tư`, `chú Hai`, `bác Sáu`, `anh Tư`,
`bác Bảy`), instalment repayment to the user, partial/final repayment, refunds from 20+ distinct
merchants/issuers/services plus unnamed ones, deposits, tax, insurance (named and unnamed),
salary advances, bank/app/overdraft borrowing, teencode (`ck`, `dc`, `r`, `nha`, `ok`, `củ`),
mixed English (`done`, `refund`, `cashback`, `roommate`), typos (`lazda`), casing drift, stray
punctuation.

## Validation

- ingest: `163 line(s), 163 record(s), 0 error(s), 0 warning(s)`
- `validate_corpus.py`: `163 record(s), 0 error(s), 0 warning(s)` (baseline-01 re-validated:
  `1000 record(s), 0 error(s), 0 warning(s)`)

## Duplicate and template diagnostics

Normalisation: lowercase, strip diacritics, `đ`→`d`, amounts masked. Skeleton: normalised text
with capitalised names masked as `X`.

| check | result |
|---|---|
| exact duplicates (batch) | 0 |
| normalised duplicates within batch | 0 |
| normalised duplicates against baseline-01 | 0 |
| skeleton frames used more than twice | 0 (max 2; 8 frames used twice, mostly the intended direction pairs) |
| batch pairs with similarity ≥ 0.85 | 5 |
| notes with a baseline-01 neighbour ≥ 0.90 | 0 |
| notes with a baseline-01 neighbour ≥ 0.85 | 13 |
| median nearest-baseline similarity | 0.71 |

Four notes were rephrased during generation after crossing 0.90 (one within the batch, three
against baseline-01). The remaining close pairs are short fixed frames with different
subjects, amounts, and accent style:

| sim | batch note | other batch note |
|---|---|---|
| 0.88 | `lazda hoàn tiền đơn hủy 259k` | `lazada tra tien don huy 99K` |
| 0.88 | `anh Bảy trả nợ 5tr` | `anh Tai tra no 3 cu` |
| 0.85 | `trả nợ Hùng 500k` | `tra no Long 200k` |
| 0.85 | `cho bạn cùng lớp mượn 150k` | `cho ban cu muon 1tr5` |
| 0.85 | `Hùng trả nợ 500k` | `Long tra no 200k` |

| sim | batch note | nearest baseline-01 note |
|---|---|---|
| 0.89 | `trả nợ Hùng 500k` | `tra no Khang 400k` |
| 0.88 | `ứng lương tháng 10 3tr` | `lương tháng 13 12 củ` |
| 0.88 | `cho bạn cùng lớp mượn 150k` | `cho bạn cùng phòng mượn 250k` |
| 0.88 | `cho ban cu muon 1tr5` | `cho bác Tư mượn 1tr2` |
| 0.88 | `cho Hiếu vay 2tr` | `cho chị Diệu vay 3tr` |
| 0.87 | `tra no Long 200k` | `trả nợ anh Long 1tr` |
| 0.87 | `muon tam ban 300k` | `muon tam Quang 300k` |
| 0.87 | `cho Dung muon 400k` | `cho Long mượn 1tr` |
| 0.86 | `nhận hoàn thuế tncn 1tr8` | `hoàn thuế tncn 2tr4` |
| 0.86 | `mượn tạm đồng nghiệp 150k gửi xe` | `muon tam dong nghiep 300k` |
| 0.85 | `trả nốt Hà 600k` | `tra no Khang 400k` |
| 0.85 | `mượn chị Loan 500k tiền trọ` | `muon chi Lan 2tr tien nha` |
| 0.85 | `me tra lai tien muon 1tr` | `Trang trả lại tiền mượn 400k` |

## Next steps

Full Quet corpus review was superseded by exception-based review: automatic QC, two annotation
passes, and Quet only for exceptions. See `experiments/targeted-annotation-v1-01-annotation.md`.
