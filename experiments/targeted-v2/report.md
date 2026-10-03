# targeted-v2: BamiBERT teacher with targeted-02 augmentation

This is a controlled experiment.

- **Training:** the frozen annotation-v1 train split plus all trainable targeted-02 records.
- **Recipe:** exactly the final baseline-v1 recipe. There was no hyperparameter search and no
  other encoder.
- **Evaluation sets:** byte-for-byte unchanged (frozen validation, frozen test, probe-v1). Their
  sha256 matches `splits/manifest.json` and `probe-v1/manifest.json`, and
  `scripts/build_training.py --version v2 --check` passes.
- **Out of scope:** nothing was distilled.

| | path |
|---|---|
| training data | `datasets/annotation-v1/training-v2/` (`train.jsonl`, `manifest.json`; `validation.jsonl`/`test.jsonl` are symlinks to the frozen splits) |
| runs | `experiments/targeted-v2/runs/bamibert/lr5e-05-seed{1,2,3}/` |
| checkpoints | `models/targeted-v2/bamibert/lr5e-05-seed{1,2,3}/` |
| probe predictions | `experiments/targeted-v2/probe-eval.json` |
| all numbers below | `experiments/targeted-v2/comparison.json` |

## 1. targeted-02, finalised

- **Records:** 112, all `complete` (0 uncertain, 0 skipped).
- **Labelling:** 106 AI-accepted after two blind passes agreed; 6 human-reviewed in Quet. All
  six human labels equal their proposals.
- **Types:** borrow 27, income 26, expense 23, lend 18, refund 10, repayment_out 8.
- **Accents:** 55 accented / 57 unaccented.
- **Groups:** 78 in total, 34 of them minimal pairs. The `group` and `patterns` metadata are
  kept in `notes.jsonl`, `provenance.jsonl` and `training-v2/train.jsonl`.
- **Validation:** `validate_annotations.py --queue`: 112 annotations, 0 errors. The manifest
  hashes were refreshed.

## 2. training-v2

- **Size:** 501 frozen train + 112 targeted-02 = **613 records**. The frozen `train.jsonl` is a
  byte-identical prefix.
- **Types:** expense 174, income 73, borrow 69, lend 60, repayment_in 37, repayment_out 58,
  transfer 93, refund 49.
- **Accents:** 354 accented / 259 unaccented.
- **Leakage gate (fails loudly):** against frozen validation, frozen test and probe-v1 it checks
  ids, normalised text, bag of words, target-masked skeleton and similarity ≥ 0.85. It also
  rejects duplicates of frozen train notes. Result: 0 failures. The closest evaluation
  similarity is 0.848, median 0.65.
- **Recipe:** identical to baseline-v1, read from the run `config.json`:
  - BamiBERT, encoder lr 5e-5, head lr 1e-3, batch 8, clip 1.0;
  - AdamW, weight decay 0.01, linear schedule with 10% warmup;
  - dropout 0.1, max_length 32, ≤ 20 epochs, patience 3;
  - masked-mean type head, equal loss weights, MPS.
  - The only differences are data-derived: 613 train records and 1540 total steps (1260 before).

## 3. Frozen test (105) and validation (110), mean ± std over 3 seeds

**Test**

| metric | baseline-v1 | targeted-v2 | Δ |
|---|---|---|---|
| type accuracy | 0.870±0.020 | 0.930±0.005 | +0.060 |
| type macro-F1 | 0.862±0.019 | 0.930±0.003 | +0.068 |
| span F1 | 0.876±0.026 | 0.877±0.029 | +0.001 |
| span exact | 0.902±0.031 | 0.911±0.024 | +0.010 |

**Validation** (model selection only)

| metric | baseline-v1 | targeted-v2 | Δ |
|---|---|---|---|
| type accuracy | 0.870±0.023 | 0.888±0.010 | +0.018 |
| type macro-F1 | 0.866±0.018 | 0.898±0.001 | +0.032 |
| span F1 | 0.860±0.014 | 0.861±0.029 | +0.001 |
| span exact | 0.885±0.014 | 0.879±0.029 | -0.006 |

**Test per-class F1**

| class | support | baseline-v1 F1 | targeted-v2 F1 | Δ |
|---|---|---|---|---|
| expense | 32 | 0.882±0.034 | 0.932±0.010 | +0.050 |
| income | 11 | 0.755±0.040 | 0.857±0.000 | +0.102 |
| borrow | 9 | 0.844±0.044 | 0.926±0.032 | +0.082 |
| lend | 9 | 0.842±0.000 | 0.980±0.034 | +0.138 |
| repayment_in | 8 | 0.830±0.011 | 0.888±0.040 | +0.059 |
| repayment_out | 10 | 0.889±0.065 | 0.938±0.025 | +0.049 |
| transfer | 18 | 0.923±0.024 | 0.937±0.012 | +0.014 |
| refund | 8 | 0.933±0.000 | 0.980±0.034 | +0.047 |

Every class improves on test, and no class drops. Validation type accuracy is +0.018, span
exact −0.006; span F1 is flat on both sets. The main test benchmark does not regress; it
improves.

## 4. Probe-v1 (81), overall

| metric | baseline-v1 | targeted-v2 | Δ |
|---|---|---|---|
| type accuracy | 0.407±0.025 | 0.860±0.007 | +0.453 |
| type macro-F1 | 0.438±0.006 | 0.876±0.012 | +0.438 |
| span F1 | 0.733±0.031 | 0.878±0.025 | +0.145 |
| span exact | 0.737±0.019 | 0.889±0.025 | +0.152 |
| type+span success | 0.272±0.012 | 0.786±0.040 | +0.514 |

## 5. Probe-v1 by pattern

The class criteria match the probe report, adapted to 3 runs: **A** means mean success < 0.40
and ≥ 5 notes fail on all seeds; **C** means mean success ≥ 0.70 and every seed ≥ 0.60;
**B** is everything else.

| # | pattern | probe class | n (acc/unacc) | type acc | span exact | success baseline → v2 | Δ success | systematic fails (0/3 seeds) | reliable (3/3) | new class |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | lender-first `X cho mượn/vay` → borrow | A | 9 (5/4) | 0.15 → 1.00 | 1.00 → 1.00 | 0.148±0.128 → 1.000±0.000 | **+0.85** | 7 → 0 | 0 → 9 | **C** |
| 2 | subject-first `X mượn/vay` → lend | B | 9 (5/4) | 0.67 → 0.93 | 0.93 → 0.96 | 0.630±0.128 → 0.926±0.064 | **+0.30** | 1 → 0 | 3 → 8 | **C** |
| 3 | incoming family gift → income | A | 9 (5/4) | 0.00 → 0.78 | 0.74 → 0.93 | 0.000±0.000 → 0.741±0.064 | **+0.74** | 9 → 1 | 0 → 5 | **C** |
| 4 | outgoing lì xì / gift → expense | A | 9 (6/3) | 0.33 → 0.85 | 0.70 → 0.89 | 0.222±0.111 → 0.778±0.000 | **+0.56** | 5 → 0 | 0 → 5 | **C** |
| 5 | insurance premium → expense | B | 9 (6/3) | 0.59 → 0.85 | 0.70 → 0.93 | 0.370±0.064 → 0.778±0.111 | **+0.41** | 5 → 0 | 2 → 5 | **C** |
| 6 | insurance payout → refund | A | 9 (6/3) | 0.37 → 0.74 | 0.44 → 0.85 | 0.074±0.128 → 0.630±0.128 | **+0.56** | 7 → 2 | 0 → 5 | **B** |
| 7 | windfall / lottery → income | A | 9 (6/3) | 0.22 → 0.85 | 0.78 → 0.89 | 0.222±0.000 → 0.741±0.064 | **+0.52** | 6 → 1 | 1 → 5 | **C** |
| 8 | named loan installment → repayment_out | B | 9 (3/6) | 0.59 → 0.74 | 0.85 → 0.74 | 0.519±0.064 → 0.667±0.111 | **+0.15** | 4 → 2 | 4 → 5 | **B** |
| 9 | title + proper-name boundary | A | 9 (7/2) | 0.74 → 1.00 | 0.48 → 0.81 | 0.259±0.128 → 0.815±0.128 | **+0.56** | 4 → 1 | 0 → 6 | **C** |

- **Systematic gaps (1, 3, 4, 6, 7, 9):** all improve by +0.52 to +0.85 in success. Five are now
  C.
  - Insurance payout (6) is B. Its spans are fixed (exact match 0.44 → 0.85), but some type
    errors remain on receipt wording.
- **Partial gaps (2, 5, 8):** subject-first lend and insurance premium are now C.
  - Loan installment (8) improves least (+0.15) and stays B. Its span exact match drops 0.85 →
    0.74, because the remaining type errors now come with fragment spans (`dong k`, `tien`) or
    a null span.

## 6. Minimal-pair and opposite-direction regression checks

**Frozen test, directional confusions** (summed over 3 seeds × 105 notes):

| gold → predicted | baseline-v1 | targeted-v2 |
|---|---|---|
| borrow → lend | 3 | 0 |
| lend → borrow | 1 | 0 |
| income → expense | 4 | 5 |
| expense → income | 5 | 3 |
| borrow → repayment_out | 0 | 2 |
| repayment_out → borrow | 3 | 0 |
| income → transfer | 2 | 1 |
| expense → transfer | 7 | 4 |
| income → lend | 2 | 0 |
| expense → lend | 1 | 0 |

**Frozen test, construction slices** (gold-based regex slices; n is small):

| construction (frozen test) | n | baseline-v1 type acc | targeted-v2 type acc |
|---|---|---|---|
| X cho mượn/vay → borrow | 1 | 0.00 | 1.00 |
| cho X mượn/vay → lend | 4 | 1.00 | 1.00 |
| gift/bonus-like income | 4 | 0.75 | 0.92 |
| insurance → expense | 1 | 0.00 | 0.67 |
| insurance → refund | 1 | 0.00 | 1.00 |
| loan repayment_out (vay) | 1 | 0.00 | 1.00 |
| mượn/vay X → borrow | 6 | 1.00 | 0.89 |
| other borrow | 2 | 1.00 | 1.00 |
| other lend | 5 | 0.80 | 0.93 |
| outgoing gift → expense | 6 | 0.78 | 0.89 |

**Probe, the opposite type a pattern could collapse into:**

| probe pattern | wrong opposite-direction type | baseline-v1 (of 27) | targeted-v2 (of 27) |
|---|---|---|---|
| insurance premium → expense | refund | 1 | 2 |
| insurance payout → refund | expense | 4 | 3 |
| incoming family gift → income | expense | 0 | 0 |
| outgoing lì xì / gift → expense | income | 3 | 2 |
| lender-first `X cho mượn/vay` → borrow | lend | 22 | 0 |
| subject-first `X mượn/vay` → lend | borrow | 2 | 2 |
| named loan installment → repayment_out | borrow | 8 | 7 |
| windfall / lottery → income | expense | 8 | 0 |

- **Lending direction:** fixing `X cho mượn` did **not** break `cho X mượn`.
  - Test `cho X mượn/vay` → lend stays at 1.00, and lend F1 rises 0.842 → 0.980.
  - In the probe, the three lenders that appear in both orders (`Mai`, `Tuấn`, `Khoa`) are
    correct in both directions on 3/3 seeds, against 0/3 for the baseline.
  - borrow↔lend confusions on test fall from 4 to 0.
- **Gift direction:** fixing incoming gifts did not break outgoing gifts.
  - Outgoing gifts on test go 0.78 → 0.89, and probe `gift_out` → income goes 3 → 2.
  - income→expense on test is 4 → 5: one extra seed-level error on the `ban may anh cu 6tr5`
    sale note. The rest are the unchanged `tien chay grab` errors; neither note is a gift.
- **Insurance:** premiums did not drift to refund (probe 1 → 2 of 27; test premium 0.00 → 0.67,
  n = 1).
- **Windfall:** lottery wins → expense fell 8 → 0. Ticket purchases cannot be checked on held-out
  data, because neither the test set nor the probe contains one.
- **Ordinary borrowing → repayment_out: a regression.**
  - Test `vay home credit giai ngan 15tr` (a loan drawdown, borrow) is now repayment_out in 2/3
    seeds. borrow→repayment_out on test goes 0 → 2, and the `mượn/vay X` → borrow slice drops
    1.00 → 0.89.
  - **Cause, from the data:** in targeted-02 block F, every repayment note starts with
    `vay <product/provider>`. The three borrow contrasts were rewritten away from that opening
    to avoid a template match with an evaluation note (`acb giải ngân …`,
    `nhan tien vay …`, `mirae asset duyet …`). So `vay <provider>` at the start of a note now
    leans repayment_out.
- **targeted-02 minimal pairs:**
  - The baseline gets both partners right in only 0.06–0.18 of the
    34 pairs, and gives both partners the same type in 0.35–0.47.
    These notes are unseen by the baseline, so this independently confirms the
    direction-shortcut diagnosis.
  - targeted-v2 gets 34/34 pairs right with 0 collapses on every seed. It was trained on these
    notes, so this shows only that both directions are learnable together without
    interference. It is not held-out evidence.

## 7. Accented vs unaccented

**Frozen sets**

| set | form | n | metric | baseline-v1 | targeted-v2 | Δ |
|---|---|---|---|---|---|---|
| test | accented | 63 | type macro-F1 | 0.880±0.029 | 0.967±0.015 | +0.087 |
| test | accented | 63 | span F1 | 0.863±0.019 | 0.867±0.024 | +0.004 |
| test | unaccented | 42 | type macro-F1 | 0.829±0.012 | 0.878±0.011 | +0.049 |
| test | unaccented | 42 | span F1 | 0.899±0.042 | 0.895±0.041 | -0.004 |
| validation | accented | 65 | type macro-F1 | 0.903±0.021 | 0.952±0.023 | +0.049 |
| validation | accented | 65 | span F1 | 0.883±0.029 | 0.890±0.019 | +0.007 |
| validation | unaccented | 45 | type macro-F1 | 0.823±0.020 | 0.820±0.049 | -0.003 |
| validation | unaccented | 45 | span F1 | 0.815±0.067 | 0.802±0.046 | -0.013 |

**Probe**

| form | metric | baseline-v1 | targeted-v2 | Δ |
|---|---|---|---|---|
| accented (49) | type acc | 0.401±0.047 | 0.857±0.020 | +0.456 |
| accented (49) | span F1 | 0.784±0.039 | 0.898±0.051 | +0.114 |
| accented (49) | span exact | 0.803±0.031 | 0.905±0.042 | +0.102 |
| accented (49) | success | 0.279±0.031 | 0.776±0.061 | +0.497 |
| unaccented (32) | type acc | 0.417±0.036 | 0.865±0.018 | +0.448 |
| unaccented (32) | span F1 | 0.649±0.054 | 0.845±0.056 | +0.197 |
| unaccented (32) | span exact | 0.635±0.018 | 0.865±0.048 | +0.229 |
| unaccented (32) | success | 0.260±0.036 | 0.802±0.065 | +0.542 |

- **Unaccented probe notes gain at least as much as accented ones** (success +0.54 vs +0.50;
  span exact +0.23 vs +0.10).
- **On frozen test** both forms improve on type.
- **On unaccented validation** type is flat (0.800 → 0.800), and span F1 is −0.013, within
  seed noise (std about 0.05).

## 8. Failure cases

**Probe notes still failing in ≥ 2 of 3 targeted-v2 seeds (17):**

| pattern | note | gold | baseline seeds ok | v2 seeds ok | v2 predictions (s1; s2; s3) |
|---|---|---|---|---|---|
| family_gift_in | `bo me cho 5tr dong hoc` | income/bo me | 0/3 | 1/3 | income/bo me; transfer/me; income/me |
| family_gift_in | `mẹ gửi 500k tiêu vặt` | income/mẹ | 0/3 | 0/3 | transfer/mẹ; repayment_in/mẹ; transfer/mẹ |
| gift_out | `gửi ba 3 triệu biếu tết` | expense/ba | 0/3 | 1/3 | transfer/∅; transfer/ba; expense/ba |
| gift_out | `lì xì bé Bin 100k` | expense/Bin | 0/3 | 1/3 | expense/Bin; income/Bin; income/Bin |
| insurance_payout | `bảo hiểm thanh toán tiền viện 4tr` | refund/∅ | 0/3 | 1/3 | refund/∅; refund/bảo hiểm; refund/bảo hiểm |
| insurance_payout | `bảo hiểm xe trả tiền sửa đèn 700k` | refund/∅ | 0/3 | 1/3 | refund/∅; refund/bảo hiểm; repayment_in/bảo hiểm xe |
| insurance_payout | `nhận tiền bảo hiểm tai nạn 6tr` | refund/∅ | 1/3 | 0/3 | income/∅; income/∅; income/∅ |
| insurance_payout | `tiền bảo hiểm nằm viện về 2tr8` | refund/∅ | 0/3 | 0/3 | expense/∅; expense/∅; expense/∅ |
| insurance_premium | `bao hiem xe hoi nam nay 4tr9` | expense/∅ | 0/3 | 1/3 | refund/∅; refund/∅; expense/∅ |
| insurance_premium | `phí bh aia tháng 12 1tr1` | expense/aia | 0/3 | 1/3 | expense/∅; expense/aia; expense/∅ |
| loan_installment | `dong tien vay tin chap thang 11 2tr3` | repayment_out/∅ | 0/3 | 0/3 | borrow/∅; borrow/tien; borrow/∅ |
| loan_installment | `khoan vay online tamo dong ky 2 900k` | repayment_out/tamo | 0/3 | 0/3 | borrow/∅; borrow/dong k; borrow/∅ |
| loan_installment | `vay ngan hang tra thang nay 3tr2` | repayment_out/ngan hang | 0/3 | 1/3 | borrow/hang; repayment_out/ngan hang; repayment_out/∅ |
| subject_first_muon | `ban cung lop muon 400k mua sach` | lend/ban cung lop | 1/3 | 1/3 | borrow/ban cung; lend/ban cung lop; borrow/ban cung lop |
| title_name_span | `chú Bình gửi lại 500k tiền xăng` | repayment_in/Bình | 1/3 | 0/3 | repayment_in/chú Bình; repayment_in/ch; repayment_in/chú Bình |
| windfall | `quay số trúng 500k` | income/∅ | 0/3 | 0/3 | borrow/∅; transfer/∅; borrow/∅ |
| windfall | `trung vietlott 10tr` | income/vietlott | 0/3 | 1/3 | income/∅; income/vietlott; income/∅ |

The recurring residuals:
- **Loan installment:** `dong tien vay …`, `khoan vay online … dong ky` → borrow. When `vay`
  appears without a leading payment verb, the old shortcut still wins.
- **Insurance payout:** receipt wording without an insurer as subject (`nhận tiền bảo hiểm …`,
  `tiền bảo hiểm … về`) → income or expense.
- **Incoming `gửi`:** `mẹ gửi 500k tiêu vặt` → transfer.
- **Other leftovers:** `quay số trúng 500k` → borrow or transfer, and the generic-insurer span
  `bảo hiểm` still being tagged now and then.

**Frozen test notes correct in the baseline majority but wrong in the targeted-v2 majority
(6; 13 went the other way):**

| kind | note | gold | baseline ok | v2 ok | v2 predictions |
|---|---|---|---|---|---|
| type | `ban may anh cu 6tr5` | income/∅ | 2/3 | 1/3 | expense/anh cu; expense/anh cu; income/anh cu |
| span | `bố cho tiền tiêu 1 triệu` | income/bố | 3/3 | 1/3 | income/∅; income/bố; income/∅ |
| type | `vay home credit giai ngan 15tr` | borrow/home credit | 3/3 | 1/3 | borrow/home credit; repayment_out/home credit; repayment_out/home credit |
| span | `Thao tra lai 350k minh chi ho` | repayment_in/Thao | 2/3 | 0/3 | repayment_out/∅; repayment_in/∅; repayment_in/Th |
| type | `roommate tra tien nha minh dong truoc 1tr5` | repayment_in/roommate | 3/3 | 1/3 | repayment_out/∅; repayment_in/roommate; refund/roommate |
| span | `điện máy xanh hoàn tiền nồi cơm lỗi 890k` | refund/điện máy xanh | 3/3 | 1/3 | refund/máy xanh; refund/xanh; refund/điện máy xanh |

Only `vay home credit giai ngan 15tr` is linked to a targeted pattern. The others are span
drops (`bố` → null, `Th`, `máy xanh`), repayment_in notes worded with `chi hộ` / `đóng trước`,
and a sale note. None of them belongs to a targeted-02 construction.

## 9. Per seed

| model | seed | best epoch | val type macro-F1 | test type macro-F1 | test span F1 | probe success |
|---|---|---|---|---|---|---|
| baseline-v1 | 1 | 9 | 0.864 | 0.866 | 0.857 | 0.272 |
| baseline-v1 | 2 | 19 | 0.885 | 0.879 | 0.865 | 0.284 |
| baseline-v1 | 3 | 10 | 0.849 | 0.842 | 0.906 | 0.259 |
| targeted-v2 | 1 | 9 | 0.896 | 0.932 | 0.845 | 0.802 |
| targeted-v2 | 2 | 14 | 0.898 | 0.932 | 0.885 | 0.815 |
| targeted-v2 | 3 | 13 | 0.898 | 0.926 | 0.902 | 0.741 |

## 10. Recommendation

**Gains:**
- Probe success 0.27 → 0.79 (+0.51); type macro-F1 0.44 → 0.88.
- 7 of 9 patterns are now C.
- Frozen test type macro-F1 0.862 → 0.930 with spans stable; validation is stable.
- Seed variance on test shrinks (type macro-F1 std 0.019 → 0.003).

**What remains:**
- **Pattern 8 (loan installment):** stays B, with one genuine opposite-direction regression
  (loan drawdown → repayment_out). It traces to a one-sided frame in block F, not to noise.
- **Pattern 6 (insurance payout):** stays B on receipt wording.

**One small follow-up is justified: targeted-03, about 20 notes.**
- **About 12 balanced `vay <product/provider> …` pairs:** drawdown (`giải ngân`, `nhận tiền`,
  `được duyệt`) → borrow, against installment (`trả kỳ`, `đóng kỳ`, `tất toán`) → repayment_out,
  with the **same opening in both directions**. Include notes where `vay` is not the first word
  (`đóng tiền vay …`, `khoản vay … đóng kỳ`).
- **About 8 insurance-receipt notes:** `nhận tiền bảo hiểm …`, `tiền bảo hiểm … về` → refund,
  each paired with a premium using the same wording.
- **Process:** the same QC, leakage gate, blind passes and recipe; then re-run this exact
  comparison.
- **Guardrail:** the borrow side must not reuse the frozen test note's frame.

## Conclusion

**B. targeted-v2 helped but specific gaps/regressions remain; one small follow-up is justified.**
