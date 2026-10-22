# gidi-en student v1: slang probe

Eval-only probe `datasets/probe-en-slang/` (never trained on, follows the `datasets/probe-*` rule).
Question: the protocol test set has little slang; does the int8 student
(`models/gidi-en-v1-onnx/seed1/model.int8.onnx`, protocol-passing) hold up on texting slang,
abbreviations (pls/rn/tmrw/bc/thx/gf/u/ur/pd/cvrd/tf/4/2), `5k`/`grand`/`quid`/`lakh`/`s$`, emoji?

Same caveat as every gidi-en number: labels are **silver** (agreement with Gemini + Clef + Claude
labels, not human ground truth). One model, one run, nothing tuned on the probe.

## Probe

- Source: `generated/en-slang-01.jsonl`, 400 Gemini-written notes. Deduped against
  `corpus/raw/en-baseline-01.jsonl` and the en splits (normalised exact match, or `SequenceMatcher`
  >= 0.85 via `gidi.annotation.leakage`) and within the batch (>= 0.90): **26 dropped** (3 exact,
  22 near corpus, 1 near batch; `work/dropped.jsonl`), **374 kept**.
- Labels (`scripts/build_en_slang_probe.py`, same method as `scripts/build_en_labels.py`, guide
  `docs/annotation-en-v1.md` incl. the 2026-10-21 named-merchants clarification): Gemini through
  `gemgen` (`work/gem-prompt.md`, same brief) + Clef zero-shot (`scripts/clef_teacher.py`
  `lang="en"`, no shots). Agreement on type and target text = `consensus` (**274**); the other
  **100** adjudicated by Claude against the guide (`work/adjudicated.jsonl`).
- Result: 374 notes, 357 `complete` (scored), 13 `uncertain`, 4 `skipped`
  (`manifest.json`; validator: 0 errors with `configs/annotation-en-v1.yaml`).
- Complete notes per type: expense 109, transfer 46, income 43, lend 41, repayment_in 39,
  borrow 31, repayment_out 28, refund 20 (counts include uncertain/skipped notes that carry a type
  in `manifest.json`; scored n below is complete only).

## Results (int8)

| Set | n | type acc | type macro-F1 | target exact | joint |
|---|---|---|---|---|---|
| **probe, all complete** | 357 | 0.854 | **0.845** | **0.754** | **0.664** |
| probe, consensus | 274 | 0.883 | 0.871 | 0.807 | 0.734 |
| probe, claude-adjudicated | 83 | 0.759 | 0.737 | 0.578 | 0.434 |
| *ref: en test primary (consensus)* | 214 | 0.935 | 0.934 | 0.869 | 0.818 |
| *ref: en test secondary (all)* | 290 | 0.890 | 0.883 | 0.834 | 0.752 |

The probe is harder than the test set: type macro-F1 drops 0.883 -> 0.845 on the all-complete
comparison (0.934 -> 0.871 on consensus), joint 0.752 -> 0.664 (0.818 -> 0.734). The probe's
consensus subset would fail the primary thresholds (type 0.90, target 0.85, joint 0.80). The
adjudicated subset is the hard tail (as on test), but it is only 23% of the probe.

Per-type F1 (all complete): expense 0.906, borrow 0.918, refund 0.923, income 0.860,
transfer 0.864, lend 0.779, repayment_in 0.785, **repayment_out 0.721**.
Per-type F1 on consensus: expense 0.926, borrow 0.931, transfer 0.904, refund 0.880,
income 0.868, lend 0.821, repayment_in 0.814, repayment_out 0.821.

## Where it breaks

Type errors (52/357): direction/debt types dominate: lend->expense 5, repayment_in<->repayment_out
7, plus transfer<->income 6 (`metro card reload`, `ezlink topup` read as expense/repayment;
`sold ... on craigslist` style income read as transfer), `cvrd ur cab chloe 14`-style abbreviated
verbs.

Target errors (88/357 = 0.246): 48 spurious spans where gold is null (generic nouns and
companions: `gym`, `elec`, `dentist`, `barber`, `bro`, `fam`, `gf`, `hotel`, `didi`), 28 wrong
spans, 12 missed. About half the wrong spans are the right word plus an extra token
(`tesco groc`, `ntuc groc`, `boots pharmacy`, `donki sushi`, `etsy shop`). Sixteen predictions are
**sub-word fragments** that start or end inside a word (`ader` for `trader joes`, `ate` in
`late`, `osa` in `dosa`, `n` in `rn`, `e` in `eod`, `g` in `gf`, `c` in `bc`); none of them is
correct. Missed counterparties are mostly after `w/` and `bc`-style tokens (`nandos w/ lads`,
`toast box breakfast`, `boss pd bonus`).

Slices (type acc / target exact / joint; slices overlap):

| Slice | n | type | target | joint |
|---|---|---|---|---|
| emoji | 11 | 0.818 | 0.455 | 0.364 |
| `k` / `grand` amounts | 23 | 0.783 | 0.696 | 0.609 |
| `quid` | 70 | 0.914 | 0.729 | 0.657 |
| `rs` / `lakh` / `₹` | 54 | 0.852 | 0.852 | 0.741 |
| `s$` | 51 | 0.824 | 0.667 | 0.569 |
| shorthand tokens (pd/cvrd/tf/4/2/pls/rn...) | 182 | 0.819 | 0.720 | 0.621 |

## Caveats

- The probe's null-target convention (generic business nouns, items, bills and companions are not
  targets) is the dominant source of target errors; part of the 48 spurious spans is the student
  copying the teacher's habit of picking nouns, not a slang failure. Count them as a labelling-
  convention gap rather than slang noise when deciding what to fix.
- 100 of 374 labels were adjudicated by Claude on my reading of the guide; direction notes like
  `diwali cash dad` (uncertain), `lunch on card 4 sam` (lend), `mate cleared debt 30 quid`
  (repayment_in) are judgment calls.
- Slice overlap and small n (emoji n = 11) make the slice rows indicative only.

## Reproduce

```sh
uv run python scripts/build_en_slang_probe.py dedupe      # then Gemini + Clef, then compare/build
uv run python scripts/validate_annotations.py --config configs/annotation-en-v1.yaml \
    datasets/probe-en-slang/labels.jsonl --queue datasets/probe-en-slang/queue.jsonl
uv run python scripts/evaluate_en_probe.py --probe datasets/probe-en-slang \
    --onnx-dir models/gidi-en-v1-onnx/seed1 --checkpoint models/gidi-en-v1/seed1 \
    --out experiments/gidi-en-student-v1/slang-probe.json
```

Per-note predictions, worst errors and per-type tables: `slang-probe.json`.
