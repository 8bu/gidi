# value-span-v8-deterministic-parser: development log (train only)

Everything here uses `datasets/annotation-v2/training-v1/train.jsonl` (895 notes, 889 complete) or
hand-written synthetic notes. Validation, test and probe labels and note texts were not read.
Only aggregate facts were read before the freeze (see `protocol.json`).

## Iterations

0. **Design.** Read the distinct train value surface patterns (digit runs masked) and every train
   note that has two or more numbers, a bare-number value or no value. The design follows:
   tiers (explicit unit / money-shaped > slang unit > bare number), context exclusion of
   dates, times, percentages, periods and quantities, `1tr5` and `1 triệu 2` continuations.
   Honest reading: the rules were written after looking at all of train, so the train number
   below is a development number, not evidence of generalisation.
1. **First run: 862 / 889 exact (0.9696), 5 / 82 human wrong.** The 27 errors, by cause:
   - 19 notes: `đóng` (to pay) folded to `dong` and was read as the currency `đồng`
     (`3tr đóng học` -> `3tr đóng`). Fix: `dong` is a currency only with the accent `ồ`, or
     unaccented at the end of the note.
   - 4 notes: `3 lit sua xe` (2), `8 xi tra tien muon`, `2 củ trả nợ`: the goods-noun list made `sửa` (repair) and `trả`
     (pay) look like `sữa` / `trà`. Fix: ambiguous folded nouns need the accented spelling.
   - 1 note: a phone number followed by an amount (`0912345678 100`) was chained into one
     13-digit "phone" and rejected. Fix: longest 9-11 digit prefix.
   - 1 note: `5.000.000 qua tài khoản`: a money-shaped number was dropped because `qua` was in the
     quantity list. Fix: grouped and >= 5-digit numbers skip the quantity context.
   - 2 notes: `1 trieu 5` / `1 trieu 2`: the "spaced digit after bare tr" guard also caught the
     word `trieu`. Fix: the guard applies to the abbreviation `tr` only.
   - the remaining human errors were the same causes.
2. **Second run: 889 / 889 exact, human 82 / 82, present/null 1.000, token F1 1.000.**

## Changes made without a train error (general robustness, hand-written notes only)

- Space-grouped thousands (`1 500 000`) were read as `000` on a synthetic note. Added: two or
  more trailing three-digit groups joined by single spaces.
- Two quadratic scans on adversarial text (`hai hai hai ...` took 141 s, repeated phone groups
  49 s). Fixed by resuming after a failed numeral run and capping phone chains at 11 digits.
  All adversarial cases now run in about 0.1 s for 100k characters.
- Quantity-after list pruned of words that are also common nouns or names (`ban`, `do`, `ca`,
  `qua`, `tin`, ...). No train error changed.
- Tie rule (`choose`): several explicit amounts -> the first; several bare numbers -> the last.
  **No complete train note exercises this**: the only notes with two explicit amounts are the 6
  `uncertain` ones, which are masked. The rule is a prior, not a tuned rule.

## Known gaps left on purpose (no train evidence)

- `2m` / `1m5` (m = million) is not a money form, a glued unknown letter run blocks the number.
- Unknown classifier nouns (`mua 5 áo`) are read as bare amounts when nothing else is explicit.
- `1k5`, `100kđ`, amounts written with a leading currency sign are not parsed.

## Disclosure

A smoke run of the cost harness before the freeze fed the test and probe note **texts** (no
labels, no parser output shown) through the timing loop. It also ran the held-out code path on
train rows to check the script. No held-out parser output was inspected.
