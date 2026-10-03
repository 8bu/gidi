# Annotation v2

annotation-v2 = [annotation-v1](annotation-v1.md) (transaction type and target span, semantics
**unchanged**) + one new output: the **value span**, the exact substring of the note that is the
monetary amount.

| note | value span |
|---|---|
| `cơm tấm 100` | `100` |
| `mượn chú hai 5 xị` | `5 xị` |
| `ăn 2 tô phở 70` | `70` (not `2`) |
| `tiền điện tháng 10 hết 612` | `612` |
| `trả góp kỳ 3 1tr5` | `1tr5` |
| `20/10 mua quà mẹ 500` | `500` |
| `mua 3 vé 150k` | `150k` |
| `đóng kỳ 2 khoản vay 3tr` | `3tr` |

**This is span selection only. Nothing normalizes the span into a number.** `5 xị` is not turned
into 500000, `1tr5` is not turned into 1500000, and no numeric value is stored anywhere (labels,
training records, model output). Turning a span into a number stays deterministic parsing outside
the model and outside this contract.

Machine-readable contract: [`configs/annotation-v2.yaml`](../configs/annotation-v2.yaml)
(type/target sections copied verbatim from v1, plus a `value:` section). Quet schema for the value
pass: [`configs/annotation-v2-value.quet.yaml`](../configs/annotation-v2-value.quet.yaml).
Validators and the rule proposer: `src/gidi/annotation/value_span.py`.

## The span convention

The value span is the **minimal full amount expression**: the number plus the unit, slang or
currency marker attached to it.

Include:

| form | examples |
|---|---|
| `k` / `K` (optionally spaced) | `50k`, `45K`, `50 K` |
| `tr` / `triệu`, compact and decimal | `1tr`, `1tr5`, `1tr150`, `4tr664`, `1.5tr`, `1,5tr`, `2 triệu` |
| `nghìn` / `ngàn`, `tỷ` | `500 nghìn`, `500 ngàn`, `2 tỷ` |
| slang | `2 củ`, `5 xị`, `5 chai`, `5 lít`, `2 triệu rưỡi` |
| separators and long digits | `1.500.000`, `612,000`, `745.000`, `1500000` |
| a currency marker that directly follows | `80.000đ`, `80.000 đồng`, `200 vnd`, `200 vnđ` |
| a bare number | `100`, `500` |
| spelled-out amounts | `bốn triệu`, `hai trăm nghìn` |

Exclude (these are never part of the span, and a number that is only one of these is not an
amount):

| excluded | examples |
|---|---|
| per-unit qualifier | `/tháng` in `50k/tháng` (the span is `50k`) |
| approximator | `tầm`, `khoảng`, `~` (`tầm 50k` → `50k`) |
| quantity | `2 tô`, `3 vé`, `2 ly`, `10kg`, `2 chỉ` of gold, `4 người` |
| month / period / ordinal | `tháng 10`, `t11`, `kỳ 3`, `quý 3`, `lần 2`, `đợt 1`, `học kỳ 2`, `giải 8` |
| date or time | `20/10`, `8h`, `8:30`, `mùng 1`, `thứ 6` |
| percentage | `2%`, `10%` |
| identifiers and model numbers | `stk 123456789`, `iphone 15`, `4g` |

If the note states several money amounts and **no single one is the transaction amount**
(`grabfood tối 89k ship 15k`), the status is `uncertain` with a note; never pick one. A note that
states no amount (`thưởng tết 2 tháng lương`) is `no_amount`.

Span mechanics are the same as v1 targets: Python code-point offsets into the NFC note,
`text[start:end] == span.text`, no leading or trailing whitespace.

## Value pass labels (Quet)

The value is labelled in a **separate** Quet pass; Quet's single span field `target` *is* the
value span in that pass. Schema: `configs/annotation-v2-value.quet.yaml`.

| type | meaning |
|---|---|
| `amount` | the note states an amount; `target` is the value span |
| `no_amount` | the note states none; `target` must be null |

| status | meaning |
|---|---|
| `complete` | the span (or the absence of any amount) is determined |
| `uncertain` | several amounts and none is clearly the transaction amount, or the amount cannot be delimited; Gidi's validator requires a `note` |
| `skipped` | not worth labelling; saves a null type and target |

Gidi's validator (`validate_value_label`) adds to Quet's own checks: `uncertain` needs a note,
`no_amount` has a null span, and a `complete` `amount` must have a span. Human labels are written
only by Quet to `datasets/annotation-v2/value-labels.jsonl`; nothing in Gidi creates, edits or
overwrites that file.

## Rule proposer and provenance

`propose_value(text)` (`gidi.annotation.value_span`) finds money-looking expressions, discards
numbers that context explains as something else, and picks the single remaining amount, or
refuses to choose (`chosen=None`) when more than one plausible amount remains. It never assumes
every number is money. It returns the candidates, the chosen span or `None`, a confidence,
category tags (`explicit_unit`, `decimal_unit`, `compact_unit`, `compound_amount`,
`currency_suffix`, `separator_format`, `long_digits`, `bare_number`, `slang`, `number_words`,
`multi_number`, `multiple_money_candidates`, `no_candidate`, `unusual_punctuation`,
`context_excluded`, `unaccented`) and `needs_review` with its reasons.

A proposal is **sent to a human** (`needs_review`) when it has any of: `multi_number`,
`multiple_money_candidates`, `compound_amount`, `bare_number`, `slang`, `unusual_punctuation`,
`no_candidate`.

`multi_number` means the note holds **two or more distinct numeric expressions**, whether or not
the extras were excluded by context: `tiền điện tháng 10 620k`, `ăn 2 tô phở 70`,
`mua 3 vé 150k`, `trả góp kỳ 3 1tr5`, `họp 20/10 mua 100k`. One amount expression counts once
(`1tr5`, `1.500.000`, `2 triệu rưỡi`, `12,5tr`, `1 triệu 2`), a date or time (`20/10`, `8:30`,
`8h30`) counts as one numeric expression, and spelled-out number words do not count. It is
independent of `multiple_money_candidates` (several plausible *amounts*) and of
`context_excluded` (a number was explained away). A `multi_number` proposal keeps its chosen
span and capped confidence, so the reviewer can accept it with one key. `compound_amount`
(`1 triệu 2`) is a single expression but is still reviewed, because the tail is ambiguous.
Clean notes (a single explicit amount, nothing flagged) get a label with `provenance: "rule"` in
`value-auto-labels.jsonl`. Rule output never goes into a human labels file, and a human label
always beats a rule label.

Unaccented notes with a single explicit-unit amount (`tra no cho Hung 2 trieu`) are **not forced
to review**: the accent does not change which span is the amount, and the unit patterns are
matched accent-insensitively. A small clean audit (below) measures that this is safe
(`scripts/audit_value_spans.py` reports the proposer's precision once `value-labels.jsonl`
exists).

Evaluation records (frozen validation, test, probe) that are only rule-labelled measure agreement
with the proposer, not truth; they are covered by the strata and the clean audit like every other
record, and a failed stratum or audit expands the review (see the escalation rule below).

## Reduced, tiered review of the existing notes

A human does not read every flagged note. `scripts/build_value_queue.py` groups the 909 in-scope
records with the same policy as `targeted-value-01` (rules in `gidi.annotation.value_review`,
seed `annotation-v2:existing-value-review`; `--check` reproduces; plan and counts in
`experiments/value-span-v1/existing-value-review-plan.{md,json}`). The previous layout queued
all 127 flagged notes plus a 40-note audit sample (167); the reduced queue is 46:

| group | rule | notes |
|---|---|---|
| `must_review` | flagged notes a rule cannot settle: proposer `multiple_money_candidates`, `no_candidate`, `bare_number`, `compound_amount`, `unusual_punctuation`, spelled-out `number_words`; a split/share amount (token `chia`: total or per-person); a `củ` amount where `cu`/`cũ`/`củ` occurs again outside the span | all (14) |
| `hard_sample` | the other flagged notes, by stratum: the proposer's exclusion reason of the extra number (`period_or_ordinal`, `quantity`, `date_or_time`, `attached_unit`, `weekday`, `percent`) or the slang `củ` unit; a record counts once, slang first | 3 per stratum of 6 or more, 2 of 3-5, all of 1-2 (17) |
| `clean_audit` | the clean rule-accepted notes | 15, spread over v1 type, accents, span unit form, target/null, split |
| `auto_accept` | everything else: clean notes with a rule label, unsampled flagged notes with their rule *proposal* (status `pending-stratum-check`) | rest |

Every draw is a pure function of `(seed, id)` (greedy least-represented-feature pick, ties by
sha256). Nothing in the unsampled groups is ever a human label.

**Escalation rule (predeclared, in `value-manifest.json` and the plan).** An *error* is a human
value label that differs from the proposal (span, or present versus null) or whose status is not
`complete`. Any error in a sampled stratum expands that stratum to full review; 0-1 errors in the
15-note clean audit let the clean set stand, 2 or more stop it and expand the review of the clean
population in stages of 50; must-review errors are reported, not escalated. Unsampled flagged
records train only if their stratum passed.

## Files

`datasets/annotation-v2/` (built by `scripts/build_value_queue.py`, deterministic, `--check`
verifies):

| file | content |
|---|---|
| `value-queue.jsonl` | `{id, text, split, source, source_batch}` for every in-scope record |
| `value-proposals.jsonl` | one advisory Quet proposal per record (`--proposals`) |
| `value-auto-labels.jsonl` | the clean rule-accepted labels, `provenance: "rule"`; not Quet input |
| `value-review-queue.jsonl` | the reduced Quet queue: `{id, text, review_group, stratum, reasons, split, source}`, must-review first |
| `value-review-provenance.jsonl` | per in-scope record: group, stratum, provenance (`queued-for-human` or `rule`), status |
| `value-manifest.json` | counts, the review plan and escalation rule, input and output sha256, seeds |
| `value-labels.jsonl` | **human output of Quet; written only by Quet** |
| `value-escalation-queue.jsonl`, `value-escalation-proposals.jsonl` | written by the scorer when the rule escalates |

In-scope records are every note that feeds gidi-finance-v1: the 723 notes of
`distillation-v1/train.jsonl` (compression-v3 trained on these), the frozen validation and test
splits and `probe-v1`.

## Running the value pass

```sh
uv run python scripts/build_value_queue.py          # once; --force replaces, never over human labels
quet annotate datasets/annotation-v2/value-review-queue.jsonl \
  --schema configs/annotation-v2-value.quet.yaml \
  --out datasets/annotation-v2/value-labels.jsonl \
  --proposals datasets/annotation-v2/value-proposals.jsonl
uv run python scripts/score_value_review.py --dataset existing-value-review
```

`--out` is strict: labels must be ids of the review queue. The proposals file covers all
in-scope records; Quet reports those outside the queue as ignored, which is expected. Press `p`
to accept a proposal, `P` to edit it. Proposals are advisory: only what you confirm is saved.

`scripts/score_value_review.py` (one command for both reduced reviews; `--dataset NAME` scores
one) compares the human labels with the proposals per group and stratum, prints the error rates,
and merges the result into `experiments/value-span-v1/review-score.json`: the escalation decision
and a verdict per stratum (`passed`, `failed`, `pending`) with the sha256 of every input. It exits
2 while the human labels are absent and never writes a label. When the rule escalates it writes
`value-escalation-queue.jsonl` (+ proposals) for the records still lacking a label, with its Quet
command in the score file (`quet annotate ... --labels datasets/annotation-v2/value-labels.jsonl`,
merge mode); re-run the scorer after that pass. Only then run `scripts/build_training_v2.py`.

## Training records

`scripts/build_training_v2.py` merges the v1 training records and the value labels into
`datasets/annotation-v2/training-v1/{train,validation,test}.jsonl` (the same ids per split and the
same order as `datasets/annotation-v1/distillation-v1`) and `probe-v1-eval-only.jsonl`
(evaluation only, never train on it). Each record keeps every v1 field and gains

| field | value |
|---|---|
| `value` | `{"text", "start", "end"}` or `null` |
| `value_status` | `complete` or `uncertain` |
| `value_provenance` | `human` or `rule` |

`value` comes from the human label when there is one, otherwise from the rule label **if the
record's stratum passed the score** (clean notes follow the clean audit; unsampled flagged notes
use their rule proposal and follow their pattern stratum). A human `no_amount` is `value: null`
with `value_status: complete` (every token is `O`). A record that awaits review, one whose stratum
has not passed ("unverified"), a human `uncertain` and a human `skipped` are `value: null`,
`value_status: uncertain`: they still train type and target, **only the value loss is masked**
(`ignore_index`). The builder refuses to run if the human labels fail validation or if
`review-score.json` is missing or stale (an input changed since scoring, e.g. a new human label),
and counts the masked records in its manifest.

An optional extra batch (`--extra DIR`, e.g. `targeted-value-01`, see below) is split by its
generation `group` with the seed `value-span-v1` (over every group of the batch, so the
assignment does not move while review labels arrive), checked for leakage with
`gidi.annotation.leakage` against the frozen validation/test splits and probes, and appended after
the v1 rows (see the script docstring). Its rows carry `provenance.annotator` `human` (both Quet
passes labelled the record; human labels override auto labels) or `synthetic-auto` (auto-accepted
by the review plan, `value_provenance` `rule`). Records in the review queue that still lack a
human label, and auto-accepted records of a stratum the score has not passed, are **excluded**
from training (not masked) and counted in `extra.review`. The batch needs its own fresh score
entry (`scripts/score_value_review.py --dataset targeted-value-01`). Filter by
`provenance.annotator` to evaluate on human-labelled rows only.

### Risk-based review of a synthetic batch (targeted-value-01)

The 253-note `targeted-value-01` batch is **not** reviewed note by note. There is no Quet
corpus-review/export step: `scripts/build_targeted_value_queue.py` works from
`corpus/raw/targeted-value-01.jsonl`, `generation-notes.jsonl` and `type-proposals.jsonl`, and
assigns every record to one group (rules in `gidi.annotation.risk_review`, seed
`targeted-value-01:review`, `--check` reproduces):

| group | rule |
|---|---|
| `must_review` | every record with an objective risk signal: proposer `multiple_money_candidates`, `no_candidate`, `compound_amount`, `unusual_punctuation`; type proposal not complete or disagreeing with the generation intent; proposed value disagreeing with the intended value; a failing validator; a (near) duplicate of existing text (normalised match or similarity ≥ 0.90); length outside 3-120 chars or > 32 v1 tokens; a number dropped only by elimination |
| `hard_sample` | 3 per stratum from the rest: `multi_number`, `bare_number`, each slang unit (`xị`, `chai`, `lít`, `củ`, `tỷ`); a record counts once, in that order |
| `clean_audit` | ~20 stratified over the remaining clean records (type, accent, amount form, target/null, short/long) |
| `auto_accept` | the rest, `synthetic-auto`; never a human label |

Files in `datasets/annotation-v2/targeted-value-01/`: `review-queue.jsonl` (Quet queue with
`review_group`, `stratum`, `reasons`; must-review first), `review-type-proposals.jsonl`,
`review-value-proposals.jsonl` (generation intent in `reason`), `auto-labels.jsonl` and
`value-auto-labels.jsonl` (auto-accepted), `review-provenance.jsonl`, `review-manifest.json`.
Quet writes the human `labels.jsonl` and `value-labels.jsonl`: two passes over the same small
queue (commands in `experiments/value-span-v1/targeted-value-01-review-plan.md`).

`scripts/score_value_review.py --dataset targeted-value-01` then compares the human labels with
the proposals per group and stratum and applies the escalation rule declared up front: an
**error** is a human label that differs from the proposal on type, target span or value span, or
whose status is not `complete`. Any error in a hard stratum reviews that whole stratum; 0-1 errors
in the clean audit accept the auto remainder, 2 or more review every auto-accepted record;
must-review errors are expected and only reported. It exits 2 while the human labels are absent
and merges its entry into `experiments/value-span-v1/review-score.json` (and writes an
`escalation-queue.jsonl` with proposals when escalation is needed).

## Data audit

`scripts/audit_value_spans.py` writes `experiments/value-span-v1/data-audit.{json,md}`: counts,
surface-pattern frequencies (accented vs unaccented, seen vs unseen in train), span lengths in
characters and v1 tokens, and an explicit coverage table of the amount forms above. Without
built training data it audits the proposals and labels itself `PRE-REVIEW`; with
`--mode training` it audits the built training records.
