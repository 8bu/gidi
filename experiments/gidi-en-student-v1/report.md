# gidi-en student v1 (seed 1)

**Verdict: PASS the pre-declared protocol** (`experiments/gidi-en-data/protocol.json`, thresholds
unchanged). Primary and secondary both pass, every per-type F1 on primary >= 0.75.

Caveat that applies to every number: labels are **silver** (Gemini + Clef consensus, Claude
adjudication on the hard test notes). These are agreement rates with those labels, not accuracy
against human ground truth. One seed, one run, no model selection (last epoch kept).

## Recipe

- Init: `scripts/build_en_student.py` on `splits/train.jsonl` only: roberta-base layers 2/5/8/11,
  34 positions, 9,000 vocab rows (rank limit 7,499 + 2,412 train token ids), FFN 3072 -> 2048,
  29.0M params. FFN calibration used train notes.
- Data: `silver-train.jsonl` (1,462 notes) + `labels-seed.jsonl` complete (280) = 1,742 notes
  (`gidi.training.english`). Dev (189 complete) logged per epoch for monitoring only; test never
  loaded during training. No Vietnamese data, no probes. 0 truncated notes.
- Training: `scripts/train_en_student.py`, 40 epochs, batch 8, encoder lr 5e-5, head lr 1e-3, AdamW
  (wd 0.01), linear warmup 0.1, MPS, loss CE(type) + CE(BIO tags), seed 1. About 15 min.
- Export: `scripts/export_onnx.py` (dynamic int8). Scoring: `scripts/evaluate_en_student.py`,
  1-thread ORT sessions, on the **int8** model (fp32 in `results.json`).

Model: `models/gidi-en-v1-onnx/seed1/model.int8.onnx` = **29,166,948 bytes (29.17 MB)**
(fp32 116.06 MB). Checkpoint `models/gidi-en-v1/seed1`, init `models/gidi-en-v1-init` (gitignored).

## Results (int8)

| Set | n | type acc | type macro-F1 | target exact | joint |
|---|---|---|---|---|---|
| **primary** (consensus test) | 214 | 0.935 | **0.934** (>= 0.90) | **0.869** (>= 0.85) | 0.818 (>= 0.80) |
| **secondary** (all complete test) | 290 | 0.890 | **0.883** (>= 0.82) | **0.834** (>= 0.78) | 0.752 |
| dev (sanity, complete) | 189 | 0.868 | 0.824 | 0.788 | 0.693 |

fp32 for comparison: primary 0.934 / 0.860 / 0.808, secondary 0.886 / 0.821 / 0.741, dev 0.821 /
0.794 / 0.688 (type F1 / target exact / joint). int8 does not hurt.

Teacher reference: the Clef teacher was never scored on test (the data step kept test unscored),
so there is no test-set teacher number. On dev (pool = seed, 189 complete notes) the teacher got
type 0.915 / target exact 0.894 / joint 0.825 (`gidi-en-data/report.md`); the student on the same
dev notes gets accuracy 0.868 / 0.788 / 0.693, i.e. below the teacher on dev. The student was
trained on teacher labels, so it should not be expected to beat it.

## Per-type F1 (int8)

| Type | primary n | primary F1 (>= 0.75) | secondary n | secondary F1 |
|---|---|---|---|---|
| expense | 54 | 0.933 | 94 | 0.911 |
| income | 19 | 0.895 | 28 | 0.893 |
| borrow | 18 | 1.000 | 27 | 0.906 |
| lend | 29 | 0.982 | 36 | 0.873 |
| repayment_in | 20 | 0.895 | 22 | 0.837 |
| repayment_out | 20 | 0.905 | 24 | 0.846 |
| transfer | 40 | 0.927 | 44 | 0.891 |
| refund | 14 | 0.933 | 15 | 0.909 |

## 10 worst errors (secondary, int8; both type and target wrong first, then by confidence)

| Note | gold type / target | predicted |
|---|---|---|
| `transferred money to papa for his eye checkup 4,500` | expense / `papa` | transfer / `pap` |
| `lunch w daryl` | expense / null | lend / `daryl` |
| `lunch with harry 15` | expense / null | lend / `harry` |
| `paid the vet bill for my sister, 150 she'll repay` | lend / `sister` | repayment_out / `vet` |
| `paid council tax monthly instalment through direct debit 135` | expense / null | repayment_out / `council` |
| `dbs shares dividend 140` | income / `dbs` | repayment_out / null |
| `dentist root canal treatment second installment paid 3,500` | expense / null | repayment_in / `ist` |
| `drinks for callum's birthday 28` | expense / null | lend / `um` |
| `tom spotted me 20` | borrow / `tom` | repayment_in / `tom` |
| `cdc vouchers claimed 300` | income / null | income / `cdc` |

Patterns: companion notes (`lunch w X`) read as lend; sub-word target spans (`pap`, `ist`, `um`)
where the pruned BPE splits a name and the tag head stops mid-word; instalment/repay wording
confuses repayment_out/in; `spotted me` (borrow direction) is the known weak silver region.

## Notes and limits

- Dev (type F1 0.824, target 0.788) is weaker than the test sets and sits at the secondary floor
  (0.82 / 0.78). Dev was not used for any decision; the pass is on the declared test sets, but dev
  suggests the secondary margin is thin.
- Single seed: no variance estimate. Margins on primary are 0.034 (type) and 0.019 (target) over
  the thresholds, so a different seed could land closer to the line.
- Sub-word span fragments suggest a cheap decode fix (extend tagged spans to word boundaries); not
  applied, since it would change the evaluation after seeing results.
- Browser/ORT Web behaviour not tested; op set equals the Vietnamese model's (backbone report).
- Housekeeping done for the task: guide row `uber home $18 -> uber` amended with a 2026-10-21
  clarification in `docs/annotation-en-v1.md` (config already stated the rule; no labels changed);
  `scripts/build_debt_relabel_queue.py` now skips English data when collecting Vietnamese leakage
  references (the new `datasets/annotation-en-v1/` and `corpus/raw/en-baseline-01.jsonl` had made
  five existing reproducibility tests fail).

## Files

`run.sh`, `train-seed1.log`, `runs/seed1/{train_log.jsonl,config.json}`, `export-seed1.json`,
`results.json` (all metrics, per-type, protocol check, worst errors). Code:
`src/gidi/training/english.py`, `scripts/train_en_student.py`, `scripts/evaluate_en_student.py`.
