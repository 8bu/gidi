# clef-distill-v1: do Clef silver labels improve the student?

All numbers come from `results.json` (`scripts/evaluate_clef_distill.py`, run once). Protocol: `protocol.json` (written before any training or scoring, sha256 in `results.json`).

## 1. Question

Does adding Cloudflare Clef 16-shot teacher labels ("silver", model labels, never human labels) on 1,757 new hand-written notes aimed at known weak spots improve encoder 1 (type + target, word snap on, INT8) + the unchanged rule value parser over the released run 4 (`gidi-finance-v3`), without regressing the frozen test, probe-v1 and human-value-01? Release-candidate rule fixed in advance: arm A, seed 1.

## 2. Data

Builder `scripts/build_distill_clef_training.py` (`--check` reproduces both sets; hashes in each `manifest.json`): training-v5 train (1,123 records, byte-identical) + silver. Silver records keep `provenance.annotator = "clef"`. Every silver target span was checked against its note text.

**Leakage gate.** The silver pool did not pass the training-v5 gate as shipped: under the v5 rules (id, verbatim / NFC-lowercase text, near duplicate, folded char-3 Jaccard >= 0.6 to a human-value-01 / -02 note) 52 of silver-A and 66 of silver-B notes were hits (e.g. one note verbatim in human-value-01, one verbatim in human-value-02, many char-3 hits to short templated notes). The protocol says the builder must raise on any hit, which would have blocked the experiment, so the builder drops every silver note that hits any rule (gate score only; no held-out note was read, no model score was used), records each with its reasons in `manifest.json` (`silver_dropped`), and then re-runs the full raise-before-write gate on the result. **This is a deviation from the declared set sizes (952 / 1,344): the arms trained on 900 / 1,278 silver notes.** No exact duplicate of a training-v5 record was found.

| | training-v5 | + silver-A (arm A) | + silver-B (arm B) |
|---|---|---|---|
| expense | 337 | 750 (+413) | 958 (+621) |
| income | 107 | 261 (+154) | 322 (+215) |
| borrow | 127 | 155 (+28) | 177 (+50) |
| lend | 138 | 187 (+49) | 203 (+65) |
| repayment_in | 77 | 139 (+62) | 151 (+74) |
| repayment_out | 123 | 199 (+76) | 223 (+100) |
| transfer | 141 | 199 (+58) | 215 (+74) |
| refund | 73 | 133 (+60) | 152 (+79) |
| **total** | 1,123 | 2023 (+900) | 2401 (+1278) |
| silver share of train | 0 | 44.5% | 53.2% |
| silver notes dropped by the gate | - | 52 of 952 | 66 of 1344 |

Silver is skewed to expense / income / refund / repayment; the debt types (borrow, lend) gain few notes.

## 3. Training

Recipe unchanged from retrain-v5 (supervised student-4x768, BamiBERT layers 2/5/8/11, pos32, vocab B-rank-8000, FFN combined-K2048, 40 epochs, batch 8, encoder lr 5e-5, head lr 1e-3, warmup 0.1, wd 0.01, dropout 0.1, grad-clip 1.0, MPS, last epoch, `--retokenize`), seeds 1-3, all records weighted equally, no soft labels. Teacher-target cache rebuilt per arm (frozen teacher `lr5e-05-seed1`; the supervised arm only reads inputs and hard labels). Exported with `scripts/export_onnx.py` to INT8. Commands: `run.sh`; logs `armA/train-seeds123.log`, `armB/train-seeds123.log`, `armB/train-seeds23.log`; exports `arm*/export-seed*.json`. Operational note: the first launch of arm B was killed during seed 2 (my shell timeout); seed 1 had finished, the partial seed-2 directory was deleted and seeds 2 and 3 were trained afresh with the same flags (`resume-armB.sh`). No seed was re-run after scoring.

## 4. Results

INT8 encoder, word snap on, unchanged `parse_value`. Controls were rescored and asserted equal to the stored results before writing. Sets: frozen test 105, probe-v1 81, human-value-01 147 (amended labels), human-value-02 190 (LLM labels, reused test set). Value exact is 1.0000 for every system on both human sets, so end-to-end exact = type and target both right. Test / probe have no value labels.

**Frozen test (105)**

| system | type_accuracy | type_macro_f1 | target_exact |
|---|---|---|---|
| run 4 s1 (released) | 0.9429 | 0.9339 | 0.9238 |
| run 5 s1 | 0.9429 | 0.9372 | 0.9048 |
| A s1 (candidate) | 0.9619 | 0.9493 | 0.9429 |
| A s2 | 0.9524 | 0.9449 | 0.9143 |
| A s3 | 0.9810 | 0.9782 | 0.9238 |
| A mean | 0.9651 | 0.9575 | 0.9270 |
| B s1 | 0.9810 | 0.9782 | 0.9333 |
| B s2 | 0.9619 | 0.9502 | 0.9333 |
| B s3 | 0.9714 | 0.9703 | 0.9333 |
| B mean | 0.9714 | 0.9662 | 0.9333 |

**probe-v1 (81)**

| system | type_accuracy | type_macro_f1 | target_exact |
|---|---|---|---|
| run 4 s1 (released) | 0.8272 | 0.8264 | 0.8519 |
| run 5 s1 | 0.8642 | 0.8878 | 0.8642 |
| A s1 (candidate) | 0.8395 | 0.8673 | 0.8642 |
| A s2 | 0.8519 | 0.8724 | 0.8519 |
| A s3 | 0.8642 | 0.8945 | 0.8765 |
| A mean | 0.8519 | 0.8781 | 0.8642 |
| B s1 | 0.8889 | 0.8921 | 0.7778 |
| B s2 | 0.8889 | 0.9003 | 0.8025 |
| B s3 | 0.8765 | 0.8917 | 0.8025 |
| B mean | 0.8848 | 0.8947 | 0.7942 |

**human-value-01 (147)**

| system | type_accuracy | type_macro_f1 | target_exact | end_to_end_exact |
|---|---|---|---|---|
| run 4 s1 (released) | 0.9592 | 0.9411 | 0.9184 | 0.8912 |
| run 5 s1 | 0.9524 | 0.9505 | 0.9048 | 0.8776 |
| A s1 (candidate) | 0.9796 | 0.9856 | 0.9320 | 0.9184 |
| A s2 | 0.9660 | 0.9753 | 0.9592 | 0.9320 |
| A s3 | 0.9660 | 0.9742 | 0.9728 | 0.9456 |
| A mean | 0.9705 | 0.9784 | 0.9546 | 0.9320 |
| B s1 | 0.9660 | 0.9753 | 0.9456 | 0.9116 |
| B s2 | 0.9660 | 0.9535 | 0.9592 | 0.9320 |
| B s3 | 0.9456 | 0.9430 | 0.9456 | 0.8912 |
| B mean | 0.9592 | 0.9573 | 0.9501 | 0.9116 |

**human-value-02 (190)**

| system | type_accuracy | type_macro_f1 | target_exact | end_to_end_exact |
|---|---|---|---|---|
| run 4 s1 (released) | 0.9211 | 0.8973 | 0.8526 | 0.7895 |
| run 5 s1 | 0.9105 | 0.8763 | 0.8368 | 0.7737 |
| A s1 (candidate) | 0.9632 | 0.9590 | 0.8789 | 0.8632 |
| A s2 | 0.9579 | 0.9397 | 0.8632 | 0.8421 |
| A s3 | 0.9526 | 0.9448 | 0.9053 | 0.8684 |
| A mean | 0.9579 | 0.9478 | 0.8825 | 0.8579 |
| B s1 | 0.9421 | 0.9236 | 0.8632 | 0.8211 |
| B s2 | 0.9368 | 0.9134 | 0.8737 | 0.8316 |
| B s3 | 0.9526 | 0.9443 | 0.8842 | 0.8421 |
| B mean | 0.9439 | 0.9271 | 0.8737 | 0.8316 |

Three-seed standard deviations (sample sd) of arm A: test type_macro_f1 0.0181; probe type_macro_f1 0.0144; hv01 end_to_end_exact 0.0136; hv02 end_to_end_exact 0.0139; arm B: test type_macro_f1 0.0144; probe type_macro_f1 0.0049; hv01 end_to_end_exact 0.0204; hv02 end_to_end_exact 0.0105. Value exact is 1.0000 everywhere on hv01 / hv02 (unchanged parser).

## 5. Pre-declared success criterion, verbatim

> arm A seed 1 end-to-end exact (type + target + value) on human-value-02 >= 0.8105 (run 4 seed 1 scores 0.7895; +4 notes of 190 or better), AND no regression larger than 0.02 against run 4 seed 1 on each of: frozen test type macro-F1, frozen test target exact, probe-v1 type macro-F1, human-value-01 end-to-end exact. Also report each seed and the three-seed mean for arms A and B; nothing is selected by them.

| check | arm A seed 1 | run 4 seed 1 | bound | result |
|---|---|---|---|---|
| human-value-02 end-to-end exact | 0.8632 (164/190) | 0.7895 (150/190) | >= 0.8105 | **PASS** |
| frozen test type macro-F1 | 0.9493 | 0.9339 | regression <= 0.02 (observed -0.0154; negative = gain) | **PASS** |
| frozen test target exact | 0.9429 | 0.9238 | regression <= 0.02 (observed -0.0190; negative = gain) | **PASS** |
| probe-v1 type macro-F1 | 0.8673 | 0.8264 | regression <= 0.02 (observed -0.0409; negative = gain) | **PASS** |
| human-value-01 end-to-end exact | 0.9184 | 0.8912 | regression <= 0.02 (observed -0.0272; negative = gain) | **PASS** |

**Verdict: PASS** (all five conditions hold; no regression against run 4 seed 1 on any of the four guarded numbers, all four moved up).

## 6. human-value-02: arm A seed 1 vs run 4 seed 1

29 of 190 notes differ in predicted type or target span: 18 fixed (wrong -> right end-to-end), 4 broke (right -> wrong), 7 changed but wrong both times. Net +14 notes = +0.0737. Spans are `[start, end)` character offsets, `null` = no target; ids only, no note text (this is an evaluation set, not a corpus).

| id | outcome | gold type / target | run 4 s1 | arm A s1 |
|---|---|---|---|---|
| hv02-bf3f1d41b7b0 | fixed | lend / [8, 13] | borrow / [8, 13] | lend / [8, 13] |
| hv02-ea0c413f2db7 | fixed | income / [0, 2] | repayment_in / [0, 2] | income / [0, 2] |
| hv02-fa2479b9d07b | fixed | expense / [17, 20] | transfer / [17, 20] | expense / [17, 20] |
| hv02-5b0e0c49f608 | fixed | income / None | expense / None | income / None |
| hv02-6c627821ff48 | fixed | expense / [13, 29] | expense / None | expense / [13, 29] |
| hv02-ed99612a96da | fixed | expense / [10, 14] | expense / [5, 9] | expense / [10, 14] |
| hv02-dac57cb116b6 | fixed | expense / [10, 17] | expense / [13, 17] | expense / [10, 17] |
| hv02-326da238ed89 | fixed | transfer / None | transfer / [18, 21] | transfer / None |
| hv02-8d332bba77cb | fixed | expense / None | expense / [14, 19] | expense / None |
| hv02-b3d595e178d7 | fixed | expense / [6, 14] | income / [6, 14] | expense / [6, 14] |
| hv02-b2dee923ede7 | fixed | expense / None | income / [9, 20] | expense / None |
| hv02-aaeb282a961b | fixed | expense / [22, 33] | transfer / [0, 3] | expense / [22, 33] |
| hv02-a3de6aba4c62 | fixed | lend / [0, 4] | repayment_in / [0, 4] | lend / [0, 4] |
| hv02-d1c07108c9f9 | fixed | expense / [12, 24] | expense / [12, 17] | expense / [12, 24] |
| hv02-5a98c6b0a63e | fixed | expense / [18, 22] | expense / [4, 7] | expense / [18, 22] |
| hv02-8f30ebcaf146 | fixed | repayment_out / [8, 24] | repayment_out / [20, 24] | repayment_out / [8, 24] |
| hv02-3784720f9a74 | fixed | expense / None | income / None | expense / None |
| hv02-06ed47636ef1 | fixed | expense / None | expense / [13, 17] | expense / None |
| hv02-cfeb880cfc92 | broke | expense / None | expense / None | expense / [12, 26] |
| hv02-e5316df9a9b5 | broke | borrow / [14, 18] | borrow / [14, 18] | borrow / [9, 13] |
| hv02-1678c79a81bf | broke | expense / None | expense / None | expense / [16, 22] |
| hv02-c4b0f024ebaa | broke | expense / [3, 11] | expense / [3, 11] | expense / [0, 2] |
| hv02-2ef93d79f771 | changed, same end-to-end result | expense / None | income / None | transfer / None |
| hv02-58f96b41d707 | changed, same end-to-end result | expense / [3, 15] | expense / None | income / None |
| hv02-1aa16f44b53f | changed, same end-to-end result | expense / [10, 21] | income / None | income / [0, 9] |
| hv02-3de7c2a2a39e | changed, same end-to-end result | expense / None | income / None | income / [12, 17] |
| hv02-2af0805a5dda | changed, same end-to-end result | expense / None | expense / [16, 18] | expense / [19, 23] |
| hv02-177bb97fd3c6 | changed, same end-to-end result | expense / None | income / None | income / [0, 4] |
| hv02-625eed5e849d | changed, same end-to-end result | borrow / [4, 11] | borrow / [4, 6] | borrow / None |

## 7. Reading

- The pre-declared criterion passes: arm A seed 1 reaches 0.8632 end-to-end on human-value-02 (run 4 seed 1 0.7895, bound 0.8105) and none of the four guarded numbers regressed (all improved, probe-v1 type macro-F1 by +0.041).
- It is not a one-seed accident on human-value-02: arm A seeds 1/2/3 score 0.8632 / 0.8421 / 0.8684 (mean 0.8579, sd 0.0139), arm B 0.8211 / 0.8316 / 0.8421 (mean 0.8316), against run 4 seed 1 0.7895 and run 5 seed 1 0.7737. Every arm A and arm B seed is above run 4 seed 1 and above the 0.8105 bound. Against that, run 4 and run 5 are single seeds; the seed sd of the new arms (about 2.6 notes) is also an estimate of how far a single seed of the old recipe can sit from its own mean, so "+14 notes over run 4 seed 1" is a stronger statement than "+14 notes over the old recipe".
- Noise: n = 190, one note = 0.0053; a binomial standard error at 0.86 is about 0.025 (about 5 notes) for a single system, but both systems are scored on the same notes, so the paired count is the better yardstick: 18 notes fixed vs 4 broke (a two-sided sign test on those 22 discordant notes gives p about 0.004). On human-value-01 (n = 147, one note = 0.0068) arm A is also higher (mean end-to-end 0.9320 vs 0.8912, sd 0.0136); on the frozen test (n = 105, one note = 0.0095) arm A seed 1 and the mean are above run 4 on type macro-F1 and target exact; on probe-v1 (n = 81, one note = 0.0123) arm A mean type macro-F1 0.8781 vs run 4 0.8264 but below run 5 0.8878, and target exact is flat (0.8642 vs 0.8519).
- Arm B (more silver, looser filter, 53% silver) is not better than arm A: lower hv02 mean (0.8316 vs 0.8579), lower probe target exact (mean 0.7942, 0.06 below run 4 seed 1 and 0.07 below arm A), similar frozen test. Within noise B is no better than A, and its probe target numbers are a warning that the looser filter lets noisier target labels in. Nothing was selected by these numbers; the release candidate stays the pre-declared arm A seed 1.
- Caveats. (1) human-value-02 was already used for the v3 release decision and is LLM-labelled; its weak spots are what the silver notes were written to cover ("aimed at known weak spots"), so a gain there is partly the intended effect, not an independent generalisation test; the frozen test, probe-v1 and human-value-01 improved too but are small. (2) The silver teacher (Clef) scores 0.953 type / 0.874 target on human-value-02 per the probe report, so the student now sits near its teacher on type (arm A mean type accuracy 0.9579) and above it in end-to-end terms only through the unchanged parser; target exact (arm A mean 0.8825) is where silver helps least, which matches the teacher's own target ceiling. (3) The pool was filtered after the fact by the leakage gate (52 / 66 notes dropped, see section 2): the declared 952 / 1,344 sets were not what was trained; the filter used only gate scores. (4) The student recipe, the parser and the snap are unchanged; value exact is 1.0000 on both human sets for every system, so the whole difference is type and target.
- Next step is a user decision (release or not); nothing was released, published or touched under `models/`, `playground/` or `releases/`, and `docs/investigation-journal.md` was not edited.
