# clef-distill-v2: report

Protocol: `protocol.json` (written before any model of this run existed, incl. `amendment_1`).
Scoring: `scripts/evaluate_clef_distill_v2.py` (once, word snap on, rule value parser unchanged);
raw numbers in `results.json`. Data: `scripts/build_distill_clef_v2_training.py`
(`--check` passes), datasets `datasets/annotation-v3/training-v7-d1|d2|d3`.

## Question

Do (1) deterministic augmentation, (2) soft-label training that also uses the notes Clef was unsure
about, and (3) the annotation-v4 rules (R1-R4) given to the Clef teacher improve encoder 1 over
clef-distill-v1 arm A seed 1 and the released gidi-finance-v3?

## Verdict: the success criterion FAILS

Evaluated verbatim: "D3 seed 1 type+target exact on probe-rules-v1 >= 0.10 above both arm A seed 1
and v3, AND no drop larger than 0.02 against arm A seed 1 on each of: frozen test type macro-F1,
frozen test target exact, probe-v1 type macro-F1, human-value-01 end-to-end exact, human-value-02
end-to-end exact."

| check | D3 seed 1 | arm A seed 1 / v3 | need | result |
|---|---|---|---|---|
| probe-rules-v1 type+target vs arm A | 0.7656 | 0.7188 (gain +0.0469) | gain >= 0.10 | **fail** |
| probe-rules-v1 type+target vs v3 | 0.7656 | 0.5000 (gain +0.2656) | gain >= 0.10 | pass |
| frozen test type macro-F1 | 0.9613 | 0.9493 (drop -0.0120) | drop <= 0.02 | pass |
| frozen test target exact | 0.8476 | 0.9429 (drop 0.0952) | drop <= 0.02 | **fail** |
| probe-v1 type macro-F1 | 0.9505 | 0.8673 (drop -0.0832) | drop <= 0.02 | pass |
| human-value-01 end-to-end | 0.8980 | 0.9184 (drop 0.0204) | drop <= 0.02 | **fail** (by one note; 0.0204 > 0.02) |
| human-value-02 end-to-end | 0.7895 | 0.8632 (drop 0.0737) | drop <= 0.02 | **fail** |

Four of seven checks fail. Nothing was tuned afterwards; the next step is a user decision.

## Data

All three arms are `training-v5` train (1,123: 256 human, 546 ai, 321 llm; D1 uses `training-v6a`'s
silver-A tail instead of nothing, see below) plus the rows listed. Counts from the manifests.

| arm | base | augment-01 | soft silver (Clef) | train | silver share | soft / span-trusted |
|---|---|---|---|---|---|---|
| D1 | 2,010 (v5 1,123 + v6a silver-A 887) | 786 | 0 | 2,796 | 0.317 (hard silver-A only) | 0 / 0 |
| D2 | 1,123 | 786 | 1,661 (old pool) | 3,570 | 0.465 | 1,661 / 706 |
| D3 | 1,123 | 786 | 2,257 (1,661 pool + 596 rules) | 4,166 | 0.542 | 2,257 / 1,018 |

Per type (expense / income / borrow / lend / repayment_in / repayment_out / transfer / refund):

| arm | expense | income | borrow | lend | rep_in | rep_out | transfer | refund |
|---|---|---|---|---|---|---|---|---|
| D1 | 961 | 330 | 271 | 300 | 196 | 281 | 279 | 178 |
| D2 | 1,347 | 443 | 310 | 401 | 225 | 328 | 303 | 213 |
| D3 | 1,509 | 550 | 341 | 539 | 323 | 346 | 344 | 214 |

Overrides (`amendment_1`, regexes in the script over NFC-lowercase diacritic-stripped text):
R2 `\b(?:tra|ung|mua|dong|nop)\s+(?:\S+\s+)?(?:ho|gium|dum|giup)\b` -> lend; R4-ung
`\b(?:ung luong|tam ung)\b` -> borrow. R2 wins if both match. One implementation choice not in the
amendment: R2 skips notes containing the accented `đồng hồ` (a watch), which the folded regex would
otherwise read as `dong ho`. Rows matched in all 2,395 labels: R2 190 (argmax type changed in 170),
R4-ung 42 (changed in 25). In the final training sets: D2 53 R2 + 5 R4-ung rows, D3 181 + 36.

Dropped by the leakage gate (listed with reasons in the manifests): 13 of v6a's silver-A rows
(`base_dropped`, D1 only: they hit probe-rules-v1, which did not exist when v6a was gated; v6a
itself is untouched), 320 of 1,106 augment rows (228 equal a training-v5 text because stripping
accents of an unaccented note is a no-op, 46 duplicate another augment/silver note, the
rest near-duplicate / char-3 hits on the human sets and probe-rules-v1, some with several reasons) and
138 of 2,395 silver notes (near-duplicates and char-3 >= 0.6 to human-value-01/02 and
probe-rules-v1, 4 verbatim hits, 1 equal to a v5 text). The final gate over the result passes in
every arm (max folded char-3 to human-value-01/02 and probe-rules-v1 of any new note 0.59).

## Results

Mean is over seeds 1-3. Type F1 = macro-F1; e2e = type + target + value (human sets only); on
probe-rules-v1 there are no value labels, so "type+target" is the headline. Controls: v3 (run 4
seed 1, `models/gidi-finance-v3`) and A1 (clef-distill-v1 arm A seed 1); both rescored and asserted
equal to their stored results on the four old sets.

| set (n) / metric | v3 | A1 | D1 s1 | D1 mean | D2 s1 | D2 mean | D3 s1 | D3 mean |
|---|---|---|---|---|---|---|---|---|
| frozen test (105) type F1 | 0.9339 | 0.9493 | 0.9640 | 0.9687 | 0.9703 | 0.9608 | 0.9613 | 0.9562 |
| frozen test target exact | 0.9238 | 0.9429 | 0.9429 | 0.9397 | 0.8952 | 0.8889 | 0.8476 | 0.8603 |
| probe-v1 (81) type F1 | 0.8264 | 0.8673 | 0.9025 | 0.9135 | 0.9530 | 0.9380 | 0.9505 | 0.9650 |
| probe-v1 target exact | 0.8519 | 0.8642 | 0.8889 | 0.8930 | 0.8765 | 0.8724 | 0.8642 | 0.8683 |
| human-value-01 (147) type F1 | 0.9411 | 0.9856 | 0.9563 | 0.9642 | 0.9870 | 0.9870 | 0.9846 | 0.9875 |
| human-value-01 e2e | 0.8912 | 0.9184 | 0.9116 | 0.9252 | 0.9048 | 0.9002 | 0.8980 | 0.9025 |
| human-value-02 (190) type F1 | 0.8973 | 0.9590 | 0.9818 | 0.9515 | 0.9309 | 0.9440 | 0.9225 | 0.9362 |
| human-value-02 e2e | 0.7895 | 0.8632 | 0.8842 | 0.8684 | 0.7789 | 0.7772 | 0.7895 | 0.7982 |
| probe-rules-v1 (64) type acc | 0.7031 | 0.9062 | 0.8750 | 0.8698 | 0.9219 | 0.9219 | 0.9219 | 0.9167 |
| probe-rules-v1 type F1 | 0.7161 | 0.8990 | 0.8683 | 0.8675 | 0.9290 | 0.9309 | 0.9295 | 0.9257 |
| probe-rules-v1 target exact | 0.7344 | 0.8125 | 0.7969 | 0.7812 | 0.7969 | 0.7917 | 0.8125 | 0.8229 |
| **probe-rules-v1 type+target** | 0.5000 | 0.7188 | 0.7031 | 0.6875 | 0.7500 | 0.7552 | 0.7656 | 0.7708 |

Per seed (1 / 2 / 3), selected metrics:

| arm | test target exact | hv01 e2e | hv02 e2e | probe-rules type+target |
|---|---|---|---|---|
| D1 | 0.943 / 0.924 / 0.952 | 0.912 / 0.925 / 0.939 | 0.884 / 0.853 / 0.868 | 0.703 / 0.688 / 0.672 |
| D2 | 0.895 / 0.876 / 0.895 | 0.905 / 0.891 / 0.905 | 0.779 / 0.779 / 0.774 | 0.750 / 0.766 / 0.750 |
| D3 | 0.848 / 0.867 / 0.867 | 0.898 / 0.912 / 0.898 | 0.789 / 0.789 / 0.816 | 0.766 / 0.766 / 0.781 |

Full per-seed, all-metric tables are in `results.json`.

### probe-rules-v1 per rule (type+target exact, 16 notes each; type acc in brackets)

| system | R1 `X tra ...` | R2 on-behalf | R3 sending money | R4 salary |
|---|---|---|---|---|
| v3 | 0.312 (0.56) | 0.375 (0.94) | 0.750 (0.75) | 0.562 (0.56) |
| A1 | 0.812 (1.00) | 0.500 (0.94) | 0.750 (0.88) | 0.812 (0.81) |
| D1 s1 | 0.625 (0.81) | 0.562 (1.00) | 0.812 (0.88) | 0.812 (0.81) |
| D2 s1 | 0.750 (0.88) | 0.500 (0.88) | 0.812 (0.94) | 0.938 (1.00) |
| D3 s1 | 0.812 (1.00) | 0.375 (0.69) | 0.875 (1.00) | 1.000 (1.00) |
| D3 s2 | 0.812 (0.94) | 0.375 (0.69) | 0.875 (1.00) | 1.000 (1.00) |
| D3 s3 | 0.812 (1.00) | 0.375 (0.69) | 0.938 (1.00) | 1.000 (1.00) |

R2 is the weak rule for D3: its type accuracy is 0.69 on all three seeds (it misses the lend type
for about five of 16 on-behalf notes), and its target exact stays at 0.56 for every system but v3.

### 17 field cases (dev only, reported, not gated)

Correct: v3 9, A1 13, D1 13/13/13, D2 14/14/15, D3 14/14/14 (seeds 1/2/3). D3 seed 1 vs A1:
newly right `long tra 5 lit`; newly wrong `Khoa trả lại tiền mình trả giùm 50k` (D2 and D3 predict
lend/Khoa instead of repayment_in, all six seeds). Wrong for every system (v3 included):
`Chuyển cho họ hàng 2 củ` (D2/D3 give expense/`họ hàng`, the expected target is none) and
`cong ty tra tien t10 5 lit` (income/`cong ty`, expected none). Everyone else matches
(`trả giùm ... Khoa`, `ứng lương`, `mua hộ Tú`, `lãnh/nhận lương` etc. are right in D1-D3).

## Reading

- **Noise.** probe-rules-v1 has 64 notes: one note = 0.0156, so D3 s1 vs A1 (+0.047) is three
  notes. The seed range inside an arm on it is 0.03-0.05 (D1 0.672-0.703, D2 0.750-0.766, D3
  0.766-0.781). On hv01 (147) one note = 0.0068; D3 s1's 0.0204 drop is three notes. On the frozen
  test (105) one note = 0.0095; the 0.095 target-exact drop is ten notes and is not noise (all
  six D2/D3 runs sit at 0.85-0.90, all D1 runs at 0.92-0.95).
- **Idea 1, augmentation (D1 vs A1).** Mixed and small. D1 mean vs A1: frozen-test type F1 +0.02,
  probe-v1 type F1 +0.05, hv02 e2e +0.005, hv01 e2e +0.007, test target exact -0.003, but
  probe-rules type+target -0.03 (A1 is the one seed 1 point; D1 seed 1 is 0.703 vs 0.719).
  A1 is a single seed, so differences at this size are not separated from seed noise. D1 is the
  only arm that keeps A1's frozen-test target exact and human-set end-to-end numbers.
- **Idea 2, soft silver (D2 vs D1; confounded with the rule-aware prompt).** Type improves
  (probe-v1 type F1 +0.02 mean, probe-rules type+target +0.07, hv01 type F1 +0.023), but spans
  get worse: frozen-test target exact -0.05, hv02 target exact -0.09 and hv02 e2e -0.09 (all three
  seeds). Because the span loss runs on Clef's span choice whenever it is >= 0.6 probable, and
  the old pool relabelled with the rule-aware prompt changed some span choices, the span loss of
  the human sets looks like the thing that moved; this experiment cannot separate "soft type loss"
  from "new prompt labels" ([INFERENCE]; not tested).
- **Idea 3, rule notes (D3 vs D2).** Helps what it targets and nothing else: probe-rules type+target
  +0.015 mean (0.755 -> 0.771; under two notes), R4 (salary) 0.94-1.00, R3 0.81 -> 0.88-0.94, but
  frozen-test target exact falls again (0.889 -> 0.860), hv02 e2e +0.021 and hv01 e2e +0.002 are
  within noise, and R2 gets worse (0.50 -> 0.375 type+target; type accuracy 0.75-1.00 -> 0.69),
  despite 181 R2-override rows in the D3 training set.
- **Against the controls.** Every arm beats v3 on probe-rules-v1 by 0.19-0.27, so the larger
  gain of the criterion's v3 half is met. Against A1, D3 gains only 0.047 on the new probe, and
  loses 0.095 on frozen-test target exact and 0.074 on hv02 e2e, so the release candidate (D3 seed 1,
  fixed in advance) does not replace A1. The only consistent win of D2/D3 is type quality
  (probe-v1 type F1 0.95 vs 0.87); the price is span quality on held-out human sets.
- No held-out number was used to choose a seed, arm or setting; D3 seed 1 is the pre-declared
  candidate. Seeds 2 and 3 of D3 do not change the verdict (probe-rules 0.766/0.781, frozen-test target
  exact 0.867 each).

## Deviations

- **Interrupted first launch:** the first training launch (`run.sh`, `run.log`) was killed by the
  agent shell's 1 h timeout during D2 seed 1 (D1 had finished and was exported). The partial D2 runs
  were deleted and D3 then D2 were retrained from scratch with the same command and flags
  (`run2.sh`, `run2.log`). No seed was re-run after scoring.
- **v6a base for D1:** 13 silver-A rows of training-v6a hit probe-rules-v1 (a set that post-dates
  v6a) and were dropped from D1 (`base_dropped`); D1's base is therefore 2,010, not 2,023.
- The `đồng hồ` exclusion in R2 above (accent-insensitive regex alone would match it).
- A duplicate-of-earlier-augment/silver rule was added to the dedupe (46 notes), next to the
  specified v5-text duplicate rule.

---

# D4 follow-up (user-approved after the D1-D3 verdict; protocol under `protocol.json` key `d4`)

Pre-declared before any D4 model existed. **Question:** do the hard-label route (arm A style) plus
only the new R1-R4 rule notes keep arm A's span quality on the human sets while gaining on
probe-rules-v1? D1-D3 pointed at the soft silver / relabelled old pool as the source of the span loss.

**Data** (`training-v7-d4`, `scripts/build_distill_clef_v2_training.py`, `--check` passes; D1-D3
datasets byte-identical to before): D1 (2,010 base + 786 augment) + the `rules` shard of
`distill-clef-02`, hard labels (no `soft` field), amendment_1 overrides applied first, kept only
where type prob >= 0.6 and target prob >= 0.6 (the silver-A rule). Of 638 rule notes, 42 were dropped
by the leakage gate (listed in the manifest), 216 of the remaining 596 were below the threshold and
unused, 380 added. Total 3,176 records; type counts: expense 1,061, income 397, borrow 291, lend 387,
repayment_in 265, repayment_out 283, transfer 313, refund 179; Clef-labelled share 0.399 (887
v6a silver-A + 380 rule notes); override rows in the train set: R2 84, R4-ung 20. Old-pool relabels
are not used. Recipe: identical to D1 (`run3.sh`, same flags, seeds 1-3). Scored once with
`scripts/evaluate_clef_distill_v2.py --d4` into `results-d4.json` (`results.json` is unchanged);
controls v3 and arm A seed 1 were rescored and asserted equal to their stored results.

## D4 verdict: the success criterion FAILS (two of seven checks)

| check | D4 seed 1 | arm A seed 1 / v3 | need | result |
|---|---|---|---|---|
| probe-rules-v1 type+target vs arm A | 0.7656 | 0.7188 (gain +0.0469) | gain >= 0.10 | **fail** |
| probe-rules-v1 type+target vs v3 | 0.7656 | 0.5000 (gain +0.2656) | gain >= 0.10 | pass |
| frozen test type macro-F1 | 0.9772 | 0.9493 (drop -0.0279) | drop <= 0.02 | pass |
| frozen test target exact | 0.9333 | 0.9429 (drop 0.0095) | drop <= 0.02 | pass |
| probe-v1 type macro-F1 | 0.9152 | 0.8673 (drop -0.0478) | drop <= 0.02 | pass |
| human-value-01 end-to-end | 0.9388 | 0.9184 (drop -0.0204) | drop <= 0.02 | pass |
| human-value-02 end-to-end | 0.8421 | 0.8632 (drop 0.0211) | drop <= 0.02 | **fail** (by 0.0011; four notes) |

## D4 results

| system | test type F1 | test target exact | probe-v1 type F1 | hv01 e2e | hv02 e2e | probe-rules type+target |
|---|---|---|---|---|---|---|
| D4 seed 1 (candidate) | 0.9772 | 0.9333 | 0.9152 | 0.9388 | 0.8421 | 0.7656 |
| D4 seed 2 | 0.9704 | 0.9238 | 0.9034 | 0.9388 | 0.8789 | 0.7812 |
| D4 seed 3 | 0.9651 | 0.9143 | 0.9243 | 0.9524 | 0.8684 | 0.7812 |
| **D4 mean** | 0.9709 | 0.9238 | 0.9143 | 0.9433 | 0.8631 | 0.7760 |
| arm A seed 1 | 0.9493 | 0.9429 | 0.8673 | 0.9184 | 0.8632 | 0.7188 |
| D1 mean | 0.9687 | 0.9397 | 0.9135 | 0.9252 | 0.8684 | 0.6875 |
| D2 mean | 0.9608 | 0.8889 | 0.9380 | 0.9002 | 0.7772 | 0.7552 |
| D3 mean | 0.9562 | 0.8603 | 0.9650 | 0.9025 | 0.7982 | 0.7708 |
| v3 | 0.9339 | 0.9238 | 0.8264 | 0.8912 | 0.7895 | 0.5000 |

probe-rules-v1 per rule, type+target (type acc in brackets), D4 seed 1: R1 0.750 (0.94), R2 0.500
(0.81), R3 0.938 (1.00), R4 0.875 (0.88); seeds 2 / 3: R1 0.750 / 0.688, R2 0.500 / 0.500, R3
0.938 / 0.938, R4 0.938 / 1.000. Field cases (dev only): D4 13 / 14 / 14 of 17 (seeds 1/2/3), arm A
13; the same two cases fail everywhere (`Chuyển cho họ hàng 2 củ`, `cong ty tra tien t10 5 lit`)
and D4 predicts lend/Khoa for `Khoa trả lại tiền mình trả giùm 50k` in all seeds (arm A gets it).

## D4 reading

- **Noise.** hv02 (190): one note = 0.0053, so seed 1's 0.0211 miss is four notes and seeds 2/3
  (0.879, 0.868) would pass; on probe-rules-v1 (64) one note = 0.0156, so +0.047 is three notes.
  The criterion is on the pre-declared seed 1 and is reported as scored: fail.
- **Soft silver was the span problem (D4 vs D3, D2).** Dropping the soft labels / old-pool relabels
  restores frozen-test target exact (0.860 -> 0.924 mean) and hv02 e2e (0.798 -> 0.863 mean) to the
  arm A / D1 level, while keeping most of the type gain (probe-v1 type F1 0.914 vs arm A 0.867).
  This is consistent with, though not a controlled test of, the D2/D3 reading ([INFERENCE]: D4
  also differs from D3 by having no old-pool silver at all).
- **Rule notes on hard labels (D4 vs D1).** probe-rules-v1 type+target 0.688 -> 0.776 mean (+0.09,
  about six notes of 64, same-size seed spread 0.766-0.781), driven by R3 (0.69-0.81 -> 0.94) and
  R4, with R1 slightly up (0.63-0.69 -> 0.69-0.75); R2 does not improve (stays 0.50). hv01 e2e +0.018 mean and the other sets are
  within noise of D1. This is the only arm that moves probe-rules-v1 without a visible cost on the
  other sets, but not by the pre-declared 0.10 over arm A (seed 1: +0.047; mean: +0.057).
- R2 on-behalf (`tra/ung/mua ho X`) is unresolved in every arm including D4 (0.375-0.56 type+target),
  although the override gave 84 R2 rows to the D4 train set.

## D4 deviations

- None from the protocol. Training was launched detached (`run3.sh`, `run3.log`, no timeout) as
  instructed; no seed was re-run. The 13 `base_dropped` v6a rows (as in D1) apply here too.
- `evaluate_clef_distill_v2.py` gained a `--d4` mode (writes `results-d4.json`); its default mode
  and `results.json` are unchanged.
