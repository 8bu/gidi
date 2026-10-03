# targeted-value-01: generation report

Targeted raw batch for **value-span coverage gaps** in the labelled data (848 records had zero
`xị`, `chai`, `lít`, `tỷ` and long plain-digit (`1500000`) amounts, one bare-number amount, and few
`50 K` / `500 ngàn`). Generator Claude, `source` `claude`, `prompt_id` `targeted-value-01`.
Nothing here is a label: `pattern`, `group`, and the intended type/target/value are generation
metadata and advisory proposals; the batch is reviewed by risk (see Next steps), not note by note,
before any label reaches training. The value is span selection only (no numeric normalisation
anywhere).

| file | role |
|---|---|
| `generated/targeted-value-01.jsonl` | scratch model output, `{"text"}` only, shuffled (seeded) |
| `generated/targeted-value-01-notes.jsonl` | scratch sidecar (text-keyed): pattern, group, generation intent |
| `corpus/raw/targeted-value-01.jsonl` | canonical raw batch (ingested, 253 records) |
| `datasets/annotation-v2/targeted-value-01/generation-notes.jsonl` | sidecar keyed by corpus id, with offsets (below) |
| `datasets/annotation-v2/targeted-value-01/type-proposals.jsonl` | advisory type/target proposals, Quet format (never labels) |
| `scripts/build_targeted_value_queue.py` | risk-based review plan: small Quet queue, proposals, auto labels, provenance, manifest (see Next steps) |
| `scripts/score_value_review.py` | human labels vs proposals per group and stratum; escalation decision and stratum verdicts (see Next steps) |

`generation-notes.jsonl` fields per id: `text`, `pattern`, `group` (minimal pairs share one;
keep a group in one split), `accented`, `amount_form`, `intended_type`, `intended_target`
(`{text,start,end}`|null), `intended_value` (`{text,start,end}`|null; code-point offsets,
`text[start:end] == value`), `type_status` (`complete`|`uncertain`), `type_note`, `value_status`
(`complete`|`no_amount`|`uncertain`), `value_note`. These are the "value notes for later": the
intended value span is what the generator meant, to compare against rule proposals and human
labels, never a label source.

## Distribution

253 notes. Accented 136, unaccented 117 (46%). Unaccented is plain ASCII, no telex typing.
Light noise: 12 first-letter capitalised, 8 with a trailing `.`; `K`/`k` casing drift, `ck`,
`tk`, `cf`, `thg`, `bh`, `cty` abbreviations.

| type | notes | accented | unaccented | unaccented % |
|---|---|---|---|---|
| expense | 48 | 23 | 25 | 52% |
| income | 29 | 17 | 12 | 41% |
| borrow | 32 | 18 | 14 | 44% |
| lend | 27 | 15 | 12 | 44% |
| repayment_in | 30 | 19 | 11 | 37% |
| repayment_out | 30 | 14 | 16 | 53% |
| transfer | 27 | 14 | 13 | 48% |
| refund | 30 | 16 | 14 | 47% |
| **total** | 253 | 136 | 117 | 46% |

Types are the generator's intent (and the proposal), not labels. 149 notes carry a counterparty
target in the intent, 104 are null. Accent state is chosen per group (hash-seeded), so both
sides of a minimal pair share it.

### Amount forms (intended value span; 228 notes with a single amount)

| form | notes | example |
|---|---|---|
| bare number (value is only digits) | 36 | `cơm tấm 100`, `ăn 2 tô phở 70` |
| `1tr5` | 21 | `mua tai nghe 1tr5` |
| dotted `1.500.000` | 18 | `chuyển 5.000.000 qua tài khoản tiết kiệm` |
| `củ` (incl. `4 củ rưỡi`, `nửa củ`) | 17 | `cty thưởng 5 củ` |
| currency suffix `đ` / `vnd` / `vnđ` / `đồng` | 15 | `200.000đ`, `45.000 vnd` |
| `nghìn` / `ngàn` | 15 | `500 ngàn` |
| `50 K` (spaced K) | 14 | `rút 500 K ở cây ATM` |
| plain long digits (`1500000`) | 14 | `đóng học phí 1500000` |
| `chai` (incl. `1 chai rưỡi`) | 12 | `cho Quân vay 2 chai` |
| `1.5tr` / `1,5tr` | 11 | `lương 12,5tr` |
| `xị` | 11 | `mượn chú hai 5 xị` |
| `triệu` (incl. `1 triệu 5`) | 10 | `bán laptop cũ được 7 triệu` |
| `150k` | 10 | `mua 3 vé 150k` |
| `lít` | 9 | `cho Bình mượn 3 lít` |
| `2tr` | 9 | `đóng kỳ 2 khoản vay 3tr` |
| `tỷ` (incl. `1 tỷ 2`) | 6 | `vay vcb mua nhà 2 tỷ` |

Note on the half-unit forms (`1 chai rưỡi`, `4 củ rưỡi`, `2 củ rưỡi`, `nửa củ`, `1 triệu 5`,
`3 củ 5`, `1 tỷ 2`, `4 củ 2`): the convention lists `1tr5` as one expression but is silent on
`rưỡi`/`nửa`; the intended span is the whole expression. Reviewers decide.

### Patterns

| pattern | notes | what it exercises |
|---|---|---|
| `multi_number` | 28 | quantities (`10kg`, `3 vé`, `2 lần`), months (`tháng 10`, `t10`), installments (`kỳ 3`, `đợt 7`), dates (`20/10`, `15/11`), times (`7h`), percent (`7%`, `20%`), `3 lít xăng` (litre quantity), account/house/room numbers; exactly one amount |
| `bare_amount` | 19 | the only number is the amount, with no unit |
| `bare_nonmoney` | 15 | bare amount next to non-money numbers: phone `0912345678`, `lớp`, `tuần`, `lần 2`, `5.5%`, `20/11 200` |
| `lender_first_cho_muon` | 15 | `X cho mượn` (borrow) / `cho X mượn` (lend) minimal pairs in new amount forms |
| `ambiguous_multi` | 15 | several money amounts where none is clearly the transaction amount (`grab 89k ship 15k`, `lương 12tr thưởng 3tr`); 3 of them are also type-uncertain |
| `plain_digits` | 12 | `1500000`-style |
| `spaced_K` | 12 | `50 K`, `500 K` |
| `nghin_ngan` | 12 | `ngàn`, `ngan`, `nghìn` |
| `slang_cu` | 11 | `củ`, `củ rưỡi`, `nửa củ` |
| `currency_suffix` | 11 | `đ`, `d`, `vnd`, `VND`, `vnđ`, `đồng` |
| `dotted_full` | 11 | `1.500.000` |
| `no_amount` | 10 | finance notes that state no amount (`nạp momo`, `mua vàng 1 chỉ`) |
| `slang_xi` | 10 | `5 xị`, `4 xi` |
| `slang_chai` | 9 | `2 chai`, `1 chai rưỡi` |
| `loan_installment` | 9 | loan/BNPL installments and disbursements in new forms |
| `tr_mid` | 9 | `1tr5`, `2tr8` |
| `trieu_word` | 8 | `2 triệu`, `1 triệu 5` |
| `decimal_tr` | 8 | `1.5tr`, `1,5tr`, `2,8tr` |
| `title_name_span` | 7 | title + name / birth-order targets with new amount forms |
| `slang_lit` | 6 | `3 lít` money slang |
| `slang_ty` | 6 | `1 tỷ`, `1 tỷ 2` |
| `insurance_payout` | 6 | insurer payouts, claim reimbursements |
| `insurance_premium` | 3 | premiums (expense) |
| `tax_refund` | 1 | tax refund |

Minimal pairs (21 groups of 2) flip the type with the same amount: `cho Phát mượn 3 xị` /
`Phát trả nợ 3 xị`, `dì Út cho mượn 2 chai` / `cho dì Út mượn 2 chai`, `Bảo trả hết nợ 2 triệu`
/ `trả hết nợ Bảo 2 triệu`, `mua vàng 1 chỉ` / `mua vàng 1 chỉ 7tr8`. 232 groups in total.

Boundary cases on purpose (final call is the annotator's): `bác Bảy ck 2 củ mừng nhà mới`,
`mừng sinh nhật Hà 20/11 200`, `thầy Phúc dạy thêm 2 buổi 600k`, `bạn cùng phòng trả lại 450 K tiền điện`,
`gửi lại anh Đạt 1tr2 tiền ứng hôm trước`, `đặt 3 bàn tiệc 12/11 trả cọc 3tr` (deposit refundability
unstated, proposed uncertain), unaccented title + ambiguous name (`vay chu Chin 3 lit`).

## Validation

- ingest: `253 line(s), 253 record(s), 0 error(s), 0 warning(s)`
- `validate_corpus.py corpus/raw/targeted-value-01.jsonl`: `253 record(s), 0 error(s), 0 warning(s)`
- The proposals validate under the annotation-v1 validator (confidence/reason stripped):
  `253 annotation(s), 0 error(s) [complete 249, uncertain 4]`.

## Dedupe

`scripts/check_note_similarity.py generated/targeted-value-01-notes.jsonl` (normalised text,
amounts masked; eval = validation, test, `datasets/probe-*`; train = annotation-v1 combined
records outside validation/test):

| check | result |
|---|---|
| first draft | 56 of 253 rejected (eval-near/exact, train duplicate/near, intra-batch duplicate); all 56 rewritten with different frames, none dropped silently |
| final run | `253 candidate(s): 0 rejected, 14 warned; eval refs 296, train refs 548` |
| warnings | 14 × `train_skeleton` (short `X trả nợ N` / `mượn X N` / `X hoàn N` frames), 0 `template_collapse` |
| normalised duplicates in batch | 0 |
| skeleton frames used by 2 groups / by more than 2 | 5 / 0 |
| batch pairs with similarity ≥ 0.85 across groups | 4 (`gửi 2 củ vào sổ tiết kiệm` ~ `ck 6,5tr vao so tiet kiem` 0.88, and similar savings frames, `Lan trả lại tiền mượn rồi` ~ `Hạnh trả lại tiền mượn 300` 0.86) |

Broader check against **every** other text in `corpus/raw/*`, `corpus/reviewed/*`,
`datasets/**/*.jsonl`, and `generated/**` (1,382 unique texts, including baseline-01 notes
that were never annotated):

| check | result |
|---|---|
| normalised exact matches | 0 (3 were found in the first draft and rewritten) |
| nearest neighbour ≥ 0.90 | 0 (2 more found in the first draft and rewritten) |
| nearest neighbour ≥ 0.85 | 27 |
| median nearest-neighbour similarity | 0.745 |

Short bare-amount frames overlap structurally with existing `X trả nợ 500k`-style notes because
amounts are masked; that is the point of the batch (new amount forms in familiar frames), and
the notes that were exact or ≥ 0.90 duplicates were reworded.

## Next steps (human): risk-based review

Decision: the 253 notes get **no full manual review** and no Quet corpus-review/export step.
`scripts/build_targeted_value_queue.py` works from `corpus/raw/targeted-value-01.jsonl`, the
sidecar and the proposals and sorts every record into a group (seed
`targeted-value-01:review`, deterministic, `--check` reproduces). Full plan, reason counts,
coverage and escalation rule: `experiments/value-span-v1/targeted-value-01-review-plan.{md,json}`.

| group | records | what a human does |
|---|---|---|
| `must_review` | 36 | label all: proposer `multiple_money_candidates` (15), `no_candidate` (10), `compound_amount` (8), `unusual_punctuation` (1); type proposal uncertain/disagreeing with intent (4); a number dropped only by elimination (3). No duplicate, validation or length signal fired |
| `hard_sample` | 21 | label 3 per hard stratum (`multi_number`, `bare_number`, `xị`, `chai`, `lít`, `củ`, `tỷ`) |
| `clean_audit` | 20 | label a stratified sample of the clean records (8 types, accent, form, target/null, short/long) |
| `auto_accept` | 176 | nothing: `synthetic-auto` labels from the proposals (never `human`) |

Review queue: 77 notes (must 36 + hard 21 + audit 20). Nothing in the inputs flags unnatural
wording objectively, so none is invented. Existing-text duplicate scan: 1,382 texts, 0 normalised
matches, 0 with similarity ≥ 0.90 (max 0.898); 31 notes sit at 0.85–0.90 and are not forced to
review.

1. Build (already done; refuses to overwrite without `--force`):

   ```sh
   uv run python scripts/build_targeted_value_queue.py [--check]
   ```

   Writes `review-queue.jsonl` (Quet queue, must-review first), `review-type-proposals.jsonl`,
   `review-value-proposals.jsonl` (generation intent in `reason`), `auto-labels.jsonl`,
   `value-auto-labels.jsonl`, `review-provenance.jsonl` and `review-manifest.json` in
   `datasets/annotation-v2/targeted-value-01/`.

2. Two Quet passes over the same small queue (Quet writes `labels.jsonl` and
   `value-labels.jsonl`; nothing else creates them):

   ```sh
   quet annotate datasets/annotation-v2/targeted-value-01/review-queue.jsonl \
     --schema configs/annotation-v1.yaml \
     --out datasets/annotation-v2/targeted-value-01/labels.jsonl \
     --proposals datasets/annotation-v2/targeted-value-01/review-type-proposals.jsonl
   quet annotate datasets/annotation-v2/targeted-value-01/review-queue.jsonl \
     --schema configs/annotation-v2-value.quet.yaml \
     --out datasets/annotation-v2/targeted-value-01/value-labels.jsonl \
     --proposals datasets/annotation-v2/targeted-value-01/review-value-proposals.jsonl
   ```

3. Score the review and apply the predeclared escalation rule (exits 2 until both label files
   exist; merges the dataset's entry into `experiments/value-span-v1/review-score.json`):

   ```sh
   uv run python scripts/score_value_review.py --dataset targeted-value-01
   ```

   An error is a human label that differs from the proposal on type, target span or value span,
   or whose status is not `complete`. (a) Any error in a hard stratum → review that whole
   stratum. (b) Clean audit: 0–1 errors → accept the auto remainder; ≥ 2 → review every
   auto-accepted record. (c) Must-review errors are expected: reported, no escalation. When
   escalation is required the scorer writes `escalation-queue.jsonl` (+ proposals) for the
   auto-accepted records to label with `quet annotate … --labels <labels file>` (merge mode).

4. Training data: `scripts/build_training_v2.py --extra datasets/annotation-v2/targeted-value-01`
   (dry run first). Human labels override auto labels; auto-accepted records train as
   `synthetic-auto`; queued records without both human labels are excluded (counted), not masked.

The 15 `ambiguous_multi` notes should land as `uncertain` with a note in the value pass; the 10
`no_amount` notes as `no_amount`. `type-proposals.jsonl` marks 4 notes `uncertain`
(two deposit/booking, two multi-counterparty); everything else is `complete` with confidence
0.4–0.9 and a reason, and low-confidence ones (≤ 0.7) are the boundary cases above.
