# value-span-v2-direction-repair: report

**Verdict: B. value-span works but type/target regressions need fixing**

Protocol: `experiments/value-span-v2-direction-repair/protocol.json` (0444, sha256
`67dbc5c528741a967a550d3ac2fa7166244b7101de0e81386086102cb5549612`), written before training.
value-span-v1 (verdict B, protocol sha256 `a23ac226…ce28`) and `gidi-finance-v1` are unchanged.
Numbers come from `comparison.json` (frozen gate code, `scripts/compare_value_v1.py`) and
`vs-value-span-v1.{md,json}` (`scripts/compare_direction_repair.py`).

## 1. Question

Is the value-span-v1 type regression caused by a lend/borrow direction imbalance that
targeted-value-01 added to training? Does restoring that balance pass the frozen regression gate
and keep value quality?

## 2. Training-data diagnosis (training rows only)

Full tables and examples: `data-audit.md`. Test and probe were not read for the audit.

- Class level is balanced. targeted-value-01 adds 23 borrow and 21 lend; the other batches are
  similar.
- Wording families are not. All 21 targeted-value-01 lend rows are `cho X mượn`. Its borrow rows
  are 14 `mượn X` and 8 name-first `X cho mượn`, and those 8 form minimal pairs with its
  `cho X mượn` rows (`dì Út cho mượn 2 chai` ↔ `cho dì Út mượn 2 chai`). It adds no name-first
  lend rows (`X mượn ...`, where X borrows from the user).
- Name-first balance (lender-first borrow vs borrower-first lend): 22 vs 17 in the v1 rows
  (difference 5), 30 vs 17 after merging (difference 13).
- Accent mix of the 8 added lender-first rows: 7 accented, 1 unaccented.
- There were no prefixed `ck/chuyển cho X mượn` rows in any training batch. This is a coverage
  gap rather than an imbalance, and it was deliberately not repaired, because doing so would
  copy the held-out error.

## 3. Repair (predeclared, deterministic)

The repair appends 8 verbatim duplicates of existing approved name-first lend rows (family
`B_X_muon`, non-targeted-value-01, complete value label). This restores the name-first
difference to 5 (30 vs 25). No label was created or edited. Validation, test and probe are
byte-identical. The rows are listed in `protocol.json` and `data-audit.md`. Train has 903 rows
(895 + 8). Builder: `uv run python scripts/build_direction_repair.py --check`.

## 4. Training

The architecture and recipe are identical to value-span-v1: 4×768 BamiBERT-derived, FFN 2048,
same tokenizer and max_length 32, equal-weight type + target BIO + value BIO cross-entropy, 40
epochs, last epoch, seeds 1–3. Each seed ran 4,520 steps (4,480 in value-span-v1, because train
has 8 more rows) and took about 300 s on MPS. Log: `train.log`. Output:
`models/value-span-v2-direction-repair/seed{1,2,3}`.

Run notes (no results were seen before any of them):

- The first launch stopped at startup because `--weights-dir` defaults to `models/value-span-v1`
  (`FileExistsError`, nothing written). The flag was added to the protocol's command before any
  training step; the protocol records this.
- The second launch was killed by the agent tool's 300 s command deadline during seed 1, before
  any checkpoint or metric was written (`train-interrupted.log`). The identical command was then
  run to completion. This was an infrastructure restart, not a reseed.

## 5. Gate (frozen criteria)

Value quality passes 10/10:

| criterion | mean | seed 1 | threshold |
|---|---|---|---|
| test value exact | 0.9856 | 0.9856 | ≥ 0.95 |
| human-only value exact | 0.9167 | 0.9167 | ≥ 0.90 |
| present/null accuracy | 1.0000 | 1.0000 | ≥ 0.97 |
| multi_number exact | 0.9545 | 0.9545 | ≥ 0.90 |
| slice floor (n ≥ 5) | 0.9545 | 0.9545 | ≥ 0.80 |

Regression vs v1 fails on the same single criterion as value-span-v1. A negative drop means the
repair model got more notes right than v1.

| criterion | value | limit | result |
|---|---|---|---|
| test-v1 type, mean drop (notes) | 1.33 | ≤ 2 | pass |
| **test-v1 type, seed 1 drop** | **4** (100 → 96) | ≤ 3 | **fail** |
| test-v1 target exact mean / seed 1 | −1.67 / −5 | ≤ 2 / ≤ 3 | pass |
| test-v1 joint mean / seed 1 | −0.33 / −2 | ≤ 2 / ≤ 3 | pass |
| probe joint mean / seed 1 | −6.67 / −11 | ≤ 2 / ≤ 3 | pass |
| probe lender-first type mean | −1.67 | ≤ 1 | pass |

## 6. Headline metrics (seeds 1 / 2 / 3)

| set | metric | v1 | value-span-v1 | repair |
|---|---|---|---|---|
| test-v1 | type correct /105 | 100 / 98 / 96 | 96 / 96 / 96 | 96 / 97 / 97 |
| test-v1 | target exact /105 | 93 / 94 / 95 | 98 / 95 / 98 | 98 / 94 / 95 |
| test-v1 | joint /105 | 89 / 88 / 89 | 91 / 87 / 92 | 91 / 88 / 88 |
| probe | joint /81 | 49 / 52 / 51 | 57 / 61 / 61 | 60 / 54 / 58 |
| probe | lender-first type /9 | 5 / 6 / 7 | 8 / 8 / 9 | 8 / 7 / 8 |

Repair, mean over seeds:

| set | type macro-F1 | target F1 | target exact |
|---|---|---|---|
| test-v1 | 0.9293 | 0.8479 | 0.9111 |
| test (143) | 0.9403 | 0.8702 | 0.9254 |
| probe | 0.8093 | 0.8258 | 0.8560 |

Full test (143), repair, mean over seeds: type accuracy 0.9301 and joint 0.8718.

## 7. Required questions

- **Is the all-seed `ck cho Vũ mượn 700k` regression repaired?** No. All three seeds still
  predict borrow (v1: lend in all seeds).
- **Are the lender-first gains retained?** Partly. Probe lender-first type correct is 8 / 7 / 8,
  against 8 / 8 / 9 for value-span-v1 and 5 / 6 / 7 for v1. This is still +1.67 notes over v1 on
  the mean, but 0.67 below value-span-v1.
- **Do unrelated categories regress?** Seed 1 on test-v1 fixed three of value-span-v1's four
  seed-1 regressions (`anh Long mượn 3 triệu sửa xe`, `mua giay the thao 1tr25`,
  `spotify refund 59k`). It broke three other notes:
  - `tra lai chi Mai 2tr` (repayment_out → expense);
  - `de rieng tien sua xe 800k` (transfer → expense);
  - `bách hoá xanh rau thịt 187k` (expense → income).

  So seed 1 stays at 96/105. The per-gold-type table in `vs-value-span-v1.md` shows changes of
  one note in either direction across expense, transfer, repayment and refund, with no
  systematic shift. On probe, seeds 2 and 3 newly mis-direct one lend/borrow note each
  (`thg Khoa cho muon tam 300k` borrow → lend; `me cho muon 1tr5` borrow → lend;
  `ban cung lop muon 400k` lend → borrow).
- **Subject-first lend slice:** none is defined in the frozen evaluation data, so none was added.

## 8. Multi-number value errors (inspected by hand)

The only error is `cho a Nam vay 1 triệu 20/10` → `1 triệu 20/10` in all three seeds: the same
overrun into a trailing date as in value-span-v1. The `tra no chi Mai 300 hom 5/10` error from
value-span-v1 seed 1 does not recur. Date-after-amount examples were out of scope here.

## 9. Interpretation

Restoring the name-first balance in training did not fix the targeted direction error. It moved
the seed-1 type errors around without reducing their number. The hypothesis, that a
targeted-value-01 direction imbalance alone causes the gate failure, is **not supported**.
[INFERENCE] The remaining `ck cho X mượn` error is more consistent with the absent prefixed
wording (§2) and with seed-level variance in type decisions on a 105-note set than with the
name-first count. Neither was tested.

## 10. Outcome

- Verdict B: stop. No export, no runtime or playground change. `gidi-finance-v1` remains
  deployed and byte-identical to its rebuild.
- `models/value-span-v2-direction-repair/seed{1,2,3}` are kept as evidence (PyTorch only).
- No further repair experiment was started.

## 11. Artifacts

| path | content |
|---|---|
| `protocol.json` | frozen protocol (0444) |
| `data-audit.{md,json}` | training-data direction audit and repair rows |
| `datasets/annotation-v2/training-v2-direction-repair/` | repaired train + unchanged val/test/probe, manifest |
| `train.log`, `train-interrupted.log` | training logs |
| `comparison.json`, `eval/`, `multi-number-errors.jsonl` | frozen gate output |
| `vs-value-span-v1.{md,json}` | per-note and per-type comparison with value-span-v1 and v1 |
| `scripts/build_direction_repair.py`, `scripts/compare_direction_repair.py` | builder and comparison |

B. value-span works but type/target regressions need fixing
