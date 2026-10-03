# targeted-v3: final targeted round (loan and insurance contrasts)

This is a controlled experiment. targeted-v3 is the targeted-v2 recipe with exactly one change:
20 more training notes (targeted-03).

- **Model:** BamiBERT only, seeds 1–3.
- **Recipe:** the same as targeted-v2. Encoder lr 5e-5, head lr 1e-3, batch 8, clip 1.0; AdamW
  with weight decay 0.01; linear schedule, 10% warmup; dropout 0.1; max_length 32; ≤ 20 epochs
  with patience 3; MPS.
  - Per seed, the `config.json` diff against targeted-v2 is only out_dir, weights_dir,
    splits_dir, train n (613 → 633), steps_per_epoch (77 → 80), total_steps (1540 → 1600) and
    the `overwrite` flag.
- **Evaluation sets:** frozen validation, frozen test and probe-v1, byte-identical. The sha256
  matches `splits/manifest.json`, and the hashes are recorded in `training-v3/manifest.json`.
- **Not done:** no distillation and no MiniLM.

| | path |
|---|---|
| targeted-03 | `datasets/annotation-v1/targeted-03/` (notes, queue, labels, provenance, manifest) |
| training data | `datasets/annotation-v1/training-v3/`, built by `scripts/build_training.py --version v3` |
| runs / checkpoints | `experiments/targeted-v3/runs/bamibert/lr5e-05-seed{1,2,3}/`, `models/targeted-v3/bamibert/…` |
| probe predictions | `experiments/targeted-v3/probe-eval.json` |
| all numbers | `experiments/targeted-v3/comparison.json` |

## 1. targeted-03

- **Size:** 20 notes in 10 minimal-pair groups. Every pair flips the type, and each pair uses the
  same provider or insurer in both notes.
- **Accents:** 10 accented, 10 unaccented.
- **Types:** borrow 6, repayment_out 6, refund 4, expense 4.

| block | notes | pairs | contrast | providers / insurers |
|---|---|---|---|---|
| L loan | 12 | 6 | drawdown (`đã nhận`, `nhan tien ve tk`, `mới vay`, `tien vay ve tk`, `vay … duoc`, `giải ngân`) vs installment (`trả kỳ`, `thanh toan ky`, `trả khoản vay`, `tra gop ky`, `tra ky cuoi`, `đóng kỳ`) | fe credit, hd saison, shinhan finance, bidv, mcredit, vietinbank |
| I insurance | 8 | 4 | payout (`đã nhận … tiền bảo hiểm`, `duoc boi thuong bh`, `chi tra quyen loi`, `thanh toán bồi thường`) vs premium (`đã nộp … tiền bảo hiểm`, `dong them goi bh`, `gia han hop dong`, `thanh toán phí`) | fwd, generic `bh` (null target), manulife, liberty |

In the loan block, `vay` appears in both labels at the start of the note (`vay fe credit …`,
`vay bidv …`, `vay mcredit …`, `vay thế chấp …`) and in the middle (`khoan vay`, `mới vay`,
`trả khoản vay`). It is followed by a provider in some notes and by a product in others
(`vay thế chấp sổ đỏ`). `thanh toán`/`trả` occurs on both sides of the insurance contrast.

### QC and leakage

- **Check:** `gidi.annotation.leakage` against frozen validation, frozen test and probe-v1 (296
  references). The training references are the annotation-v1 train split, which includes
  targeted-01, plus all 112 targeted-02 notes.
- **First draft:** 0 rejects. Even so, I rewrote 7 notes (all over two rounds) because they
  reused the frame of an evaluation note:
  - the prudential pair: too close to `nhận tiền bảo hiểm tai nạn 6tr` and
    `đóng phí bảo hiểm prudential 2tr`; moved to `fwd` and reworded twice;
  - the generic health-insurance pair: too close to `tiền bảo hiểm nằm viện về 2tr8` and
    `phí bảo hiểm sức khỏe quý 3 1tr8`; reworded to a dental cover (`bh rang mieng`);
  - the `bảo việt` pair, which a probe payout note uses; moved to `liberty`;
  - 1 changed from `vừa giải ngân` so it is no longer a near rewrite of the regressed test note
    `vay home credit giai ngan 15tr`.
- **Final 20:** 0 rejects and 0 warnings (no exact, bag or skeleton matches, no template
  collapse). Max similarity is 0.69 to an evaluation note and 0.754 to a training note.
- **Text checks:** every note is NFC, and every intended target is a substring of its note.

### Annotation

- **Process:** two blind passes, the same as targeted-02. The annotators saw only `queue.jsonl`,
  the frozen doc and the config. Pass 2 worked in reverse order. Both validate with 0 errors.
- **Agreement:** the passes disagree on 0 of 20 in status, type or span, and 0 labels differ
  from the generation intent.
- **Confidence:** the minimum across both passes is 0.70, on `gia han hop dong manulife 3tr`.
  It is settled by the frozen premium = expense rule and by the targeted-02 precedent
  `gia hạn bảo việt nhân thọ 12tr`.
- **Concerns raised:**
  - named-insurer targets: settled by the named-counterparty rule and the targeted-02
    precedents (`aia`, `pvi`, `generali`, `bảo việt nhân thọ`);
  - premiums worded without `bảo hiểm`: settled in the same way;
  - `tra ky cuoi`: read as an installment on the loan.
- **Result:** 20 notes auto-accepted and **0 sent to human review**, so the run went straight to
  training.

## 2. training-v3

| source | records |
|---|---|
| frozen train (byte-identical prefix) | 501 |
| targeted-02 (complete) | 112 |
| targeted-03 (complete) | 20 |
| **total** | **633** |

- **Types:** expense 178, income 73, borrow 75, lend 60, repayment_in 37, repayment_out 64,
  transfer 93, refund 53.
- **Accents:** 364 accented, 269 unaccented.
- **validation.jsonl, test.jsonl:** symlinks to the frozen splits.
- **Leakage gate:** 132 augmented records checked, 0 failures. The gate checks eval ids,
  normalised text, bag of words, skeleton, similarity ≥ 0.85, and exact duplicates of frozen
  train or of an earlier batch. Max eval similarity is 0.848 (a targeted-02 note).
- **Rebuild checks:** `build_training.py --version v3 --check` and `--version v2 --check` both
  pass. training-v2 still rebuilds byte-identically.

## 3. Frozen test and validation (mean ± std, 3 seeds)

| test | baseline-v1 | targeted-v2 | targeted-v3 | v3 − v2 |
|---|---|---|---|---|
| type accuracy | 0.870±0.020 | 0.930±0.005 | 0.930±0.015 | +0.000 |
| type macro-F1 | 0.862±0.019 | 0.930±0.003 | 0.934±0.016 | +0.004 |
| span F1 | 0.876±0.026 | 0.877±0.029 | 0.856±0.080 | −0.021 |
| span exact | 0.902±0.031 | 0.911±0.024 | 0.914±0.058 | +0.003 |

| validation | baseline-v1 | targeted-v2 | targeted-v3 | v3 − v2 |
|---|---|---|---|---|
| type accuracy | 0.870±0.023 | 0.888±0.010 | 0.882±0.045 | −0.006 |
| type macro-F1 | 0.866±0.018 | 0.898±0.001 | 0.886±0.043 | −0.011 |
| span F1 | 0.860±0.014 | 0.861±0.029 | 0.850±0.017 | −0.010 |
| span exact | 0.885±0.014 | 0.879±0.029 | 0.876±0.026 | −0.003 |

| test per-class F1 | support | baseline-v1 | targeted-v2 | targeted-v3 |
|---|---|---|---|---|
| expense | 32 | 0.882 | 0.932 | 0.922 |
| income | 11 | 0.755 | 0.857 | 0.842 |
| borrow | 9 | 0.844 | 0.926 | 0.961 |
| lend | 9 | 0.842 | 0.980 | 0.982 |
| repayment_in | 8 | 0.830 | 0.888 | 0.956 |
| repayment_out | 10 | 0.889 | 0.938 | 0.910 |
| transfer | 18 | 0.923 | 0.937 | 0.943 |
| refund | 8 | 0.933 | 0.980 | 0.958 |

The mean test type score is flat. Seed variance is much higher than targeted-v2 on the type
metrics and on test spans: test macro-F1 std 0.003 → 0.016, validation macro-F1 std 0.001 →
0.043, test span F1 std 0.029 → 0.080. Only validation span F1 is steadier (0.029 → 0.017).

## 4. Probe-v1 overall (81)

| | baseline-v1 | targeted-v2 | targeted-v3 | v3 − v2 |
|---|---|---|---|---|
| type accuracy | 0.407±0.025 | 0.860±0.007 | **0.901±0.000** | +0.041 |
| type macro-F1 | 0.438±0.006 | 0.876±0.012 | 0.882±0.005 | +0.006 |
| span F1 | 0.733±0.031 | 0.878±0.025 | 0.835±0.038 | −0.043 |
| span exact | 0.737±0.019 | 0.889±0.025 | 0.844±0.038 | −0.045 |
| type + span success | 0.272±0.012 | 0.786±0.040 | 0.765±0.037 | −0.021 |

Span errors over 81 notes × 3 seeds, by kind:

| | baseline-v1 | targeted-v2 | targeted-v3 |
|---|---|---|---|
| span predicted where gold is null | 20 | 6 | **17** |
| wrong span | 24 | 13 | 19 |
| missed span | 20 | 8 | 2 |

targeted-03 improves type decisions but makes span prediction more eager: a span is now emitted
where the frozen rules say null. Examples are generic `bảo hiểm` / `baohiem` / `bh`, `can ho`,
`vien`, `lịch`. 18 of the 20 new notes have a named provider or insurer target (only the two
generic `bh` notes are null), so the batch tilts span supervision towards non-null.

## 5. All 9 probe patterns (success = type and span both right)

The probe-v1 class is from `probe-report.md`. The 3-run class uses the same rule as the
targeted-v2 report: C means mean ≥ 0.70 and every seed ≥ 0.60; A means mean < 0.40 and ≥ 5
notes fail on all seeds; B is everything else.

| # | pattern | probe class | baseline | v2 | v3 | v3 seeds | v2 type acc → v3 | v2 span exact → v3 | v3 class |
|---|---|---|---|---|---|---|---|---|---|
| 1 | lender_first_cho_muon | A | 0.15 | 1.00 | 1.00 | 1 / 1 / 1 | 1.00 → 1.00 | 1.00 → 1.00 | C |
| 2 | subject_first_muon | B | 0.63 | 0.93 | 0.81 | .67 / .89 / .89 | 0.93 → 0.93 | 0.96 → 0.85 | C |
| 3 | family_gift_in | A | 0.00 | 0.74 | 0.78 | .78 ×3 | 0.78 → 0.85 | 0.93 → 0.93 | C |
| 4 | gift_out | A | 0.22 | 0.78 | 0.96 | 1 / .89 / 1 | 0.85 → 0.96 | 0.89 → 1.00 | C |
| **5** | **insurance_premium** | B | 0.37 | 0.78 | **0.78** | .67 / .78 / .89 | 0.85 → 0.85 | 0.93 → 0.93 | C |
| **6** | **insurance_payout** | A | 0.07 | 0.63 | **0.52** | .44 / .56 / .56 | 0.74 → 0.78 | 0.85 → 0.67 | B |
| 7 | windfall | A | 0.22 | 0.74 | 0.78 | 1 / .56 / .78 | 0.85 → 0.89 | 0.89 → 0.85 | B |
| **8** | **loan_installment** | B | 0.52 | 0.67 | **0.52** | .56 / .56 / .44 | 0.74 → **0.85** | 0.74 → **0.63** | B |
| 9 | title_name_span | A | 0.26 | 0.81 | 0.74 | .78 / .56 / .89 | 1.00 → 1.00 | 0.81 → 0.74 | B |

- **Previously fixed patterns (1–4, 7, 9):** they stay fixed on type, and the type accuracy of
  every one is equal or higher.
  - Patterns 7 and 9 fall from C to B only because one seed dips to 0.56. Both are span or seed
    noise, not type errors.
- **Pattern 5 (premium):** unchanged.
- **Patterns 6 and 8 (the two targets):** type improves, but success drops because the spans got
  worse.

## 6. Loan regression analysis

**Held-out new-loan vs repayment notes** (frozen test; correct-type runs out of 3):

| note | gold | baseline-v1 | targeted-v2 | targeted-v3 |
|---|---|---|---|---|
| `vay home credit giai ngan 15tr` | borrow | 3 | 1 | **2** |
| `vay ngân hàng mua nhà giải ngân 800tr` | borrow | 3 | 3 | 3 |
| `vay tiêu dùng trả kỳ này 3tr4` | repayment_out | 0 | 3 | 3 (span right 3/3; v2 1/3) |
| `vay chú Hải 20tr mua xe` | borrow | 3 | 3 | 3 |
| `vay đồng nghiệp 400k cuối tháng trả` | borrow | 3 | 3 | 3 |

Test borrow/repayment_out confusions, summed over 3 seeds:

| gold → predicted | baseline-v1 | targeted-v2 | targeted-v3 |
|---|---|---|---|
| borrow → repayment_out | 0 | **2** | **1** |
| repayment_out → borrow | 3 | 0 | 0 |
| borrow → lend | 3 | 0 | 1 (`ứng 2tr`) |
| repayment_in → repayment_out | 2 | 2 | 2 |
| other → repayment_out | 2 | 0 | **3** (`mua giay the thao`, `chuyen tien the phu`, `bao hiem tra tien kham`) |
| other → borrow | 2 | 2 | 0 |

**Probe loan_installment → borrow:** 8 baseline, 7 targeted-v2, **4** targeted-v3.
`khoan vay online tamo dong ky` goes 0/3 → 2/3 on type, and `vay ngan hang tra thang nay` goes
2/3 → 3/3. `dong tien vay tin chap thang 11` is still borrow on every seed.

**Verdict:**
- The regression is reduced, not removed. The drawdown note is now right in 2 of 3 seeds
  instead of 1, and borrow → repayment_out falls from 2 to 1. This rests on a single held-out
  note.
- The installment direction improves clearly on the probe (type 0.74 → 0.85).
- New cost: repayment_out now attracts three seed-level false positives from other classes on
  test (0 in targeted-v2).
- The probe success for pattern 8 drops anyway, because the spans got worse:
  `can ho` ×3 (gold null), `vien` ×2, `tamo dong` / `dong` / `o`, `nhà`.

## 7. Insurance analysis

| held-out check | baseline-v1 | targeted-v2 | targeted-v3 |
|---|---|---|---|
| probe premium → refund (of 27) | 1 | 2 | 3 |
| probe payout → expense (of 27) | 4 | 3 | 3 |
| probe payout → income (of 27) | 3 | 3 | 1 |
| probe `nhận tiền bảo hiểm tai nạn 6tr` (correct of 3) | 1 | 0 | **2** |
| probe `tiền bảo hiểm nằm viện về 2tr8` | 0 | 0 | 0 (expense ×3) |
| test `dong bao hiem nhan tho 1tr2` (expense) | 0 | 2 | 2 |
| test `bao hiem tra tien kham 850k` (refund) | 0 | 3 | 2 (1 repayment_out) |

- **Payout type:** improves slightly (0.74 → 0.78). The receipt wording `nhận tiền bảo hiểm`
  now generalises in 2 of 3 seeds.
- **Premium type:** flat. The premium → refund confusion grows by one (2 → 3: `bao hiem xe hoi
  nam nay 4tr9` ×2 and `bhxh tu dong thang nay 1tr5` ×1).
- **Payout span:** regresses. Generic `bảo hiểm` is tagged as the target on
  `bảo hiểm thanh toán tiền viện 4tr` (2/3 → 3/3), and new wrong spans appear: `baohiem` ×2,
  `bh`, `xe`. This breaks the frozen rule that generic `bảo hiểm` gets a null target, even
  though targeted-03 itself labels generic `bh` as null.

## 8. Remaining failure cases (targeted-v3, wrong on ≥ 2 seeds)

| set | note | gold | v3 predictions |
|---|---|---|---|
| probe | `dong tien vay tin chap thang 11 2tr3` | repayment_out / null | borrow ×3 |
| probe | `tiền bảo hiểm nằm viện về 2tr8` | refund / null | expense ×3 |
| probe | `bảo hiểm thanh toán tiền viện 4tr` | refund / null | refund, span `bảo hiểm` ×3 |
| probe | `thanh toan ky 5 khoan vay mua can ho 7tr6` | repayment_out / null | span `can ho` ×3 |
| probe | `tra no vay sinh vien ky 3 1tr` | repayment_out / null | span `vien` ×2 |
| probe | `bao hiem xe hoi nam nay 4tr9` | expense / null | refund ×2 |
| test | `tien chay grab hom nay 450k` | income / `grab` | expense ×3 |
| test | `ban may anh cu 6tr5` | income | expense ×2 |
| test | `de rieng tien sua xe 800k` | transfer | expense ×2 (v2: 3/3 right) |
| test | `chuyen tien the phu 700k` | transfer | repayment_out, income (v2: 2/3 right) |
| test | `roommate tra tien nha minh dong truoc 1tr5` | repayment_in | repayment_out ×2 |

On test, v3 vs v2 by majority vote:
- **Fixed:** 2 type (`vay home credit giai ngan`, `lì xì cháu`) and 3 span.
- **Regressed:** 2 type (`de rieng tien sua xe`, `chuyen tien the phu`) and 2 span (`grab` →
  `ab`/null, `dì Tư` → `ì Tư`).

## 9. Seed variance

| version | seed | best epoch | val type macro-F1 | test type macro-F1 | test span F1 | test span exact | probe success |
|---|---|---|---|---|---|---|---|
| baseline-v1 | 1 | 9 | 0.864 | 0.866 | 0.857 | 0.867 | 0.272 |
| baseline-v1 | 2 | 19 | 0.885 | 0.879 | 0.865 | 0.914 | 0.284 |
| baseline-v1 | 3 | 10 | 0.849 | 0.842 | 0.906 | 0.924 | 0.259 |
| targeted-v2 | 1 | 9 | 0.896 | 0.932 | 0.845 | 0.886 | 0.802 |
| targeted-v2 | 2 | 14 | 0.898 | 0.932 | 0.885 | 0.914 | 0.815 |
| targeted-v2 | 3 | 13 | 0.898 | 0.926 | 0.902 | 0.933 | 0.741 |
| targeted-v3 | 1 | 9 | 0.883 | 0.936 | 0.913 | 0.952 | 0.765 |
| targeted-v3 | 2 | 8 | 0.931 | 0.918 | **0.765** | **0.848** | 0.728 |
| targeted-v3 | 3 | 9 | 0.845 | 0.950 | 0.891 | 0.943 | 0.802 |

- **targeted-v3 seed 2** stops early (epoch 8) with the worst test span F1 of all nine runs.
- **targeted-v3 seed 3** has the lowest validation macro-F1 of all nine runs.
- **targeted-v2** is more stable than targeted-v3 on every type metric and on test spans;
  targeted-v3 is steadier only on validation span F1 (std 0.017 vs 0.029).

## 10. Decision

**Against the freeze criteria:**

- **Loan regression:** materially reduced (borrow → repayment_out 2 → 1; the drawdown note 1/3 →
  2/3), but not removed. The evidence is one held-out note.
- **Insurance/loan probe:** type improves (patterns 6 and 8), but **success falls**: payout
  0.63 → 0.52, installment 0.67 → 0.52. Span exactness on those patterns drops by 0.11–0.18, and
  null-gold spans across the probe nearly triple (6 → 17). This criterion is **not met**.
- **Previously fixed patterns:** they stay fixed on type. Two drop from C to B on single-seed span
  dips.
- **Frozen test:** type is stable, span F1 −0.021, and seed variance on type and test spans is
  much higher. Validation type macro-F1 is −0.011.

**Conclusion:** targeted-03 buys better probe type accuracy (0.860 → 0.901) and a partial fix
on one held-out loan note. In exchange, spans get systematically worse (probe span exact
0.889 → 0.844, null-gold spans 6 → 17) and the teacher is less stable. The two targeted
patterns end with lower success than before. That is no useful net improvement. targeted-v2 is
the better teacher for distillation: it has better spans, equal test type quality, and the
most consistent seeds on type.

**No reopening of augmentation.**
- The over-eager spans are a side effect of this batch's span mix, not a teacher-breaking
  systematic failure in targeted-v2.
- targeted-v2's residual errors are accepted as they are: the loan drawdown note, and loan /
  insurance receipt wording.

**B. Keep targeted-v2 as the BamiBERT teacher and proceed to distillation.**
