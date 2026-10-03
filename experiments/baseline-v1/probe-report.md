# baseline-v1 probe report (probe-v1)

This report is evaluation only. Nothing was trained or fine-tuned, annotation-v1 and its frozen
split are unchanged, and probe-v1 is not training data.

Sources:
- `datasets/probe-v1/`, frozen by `manifest.json` with `evaluation_only: true`;
- the six final checkpoints in `models/baseline-v1-e20/<encoder>/lr5e-05-seed{1,2,3}`;
- `scripts/evaluate_probe.py`, which writes per-run metrics and every prediction to
  `probe-eval.json`.

## 1. Probe set

| | |
|---|---|
| records | 81, all `complete`, and all 81 score |
| validation | `validate_annotations.py --queue`: 81 annotations, 0 errors |
| labels | 74 AI-accepted (two blind passes agreed) · 7 human-reviewed in Quet |
| human review | the human accepted all 7 escalated proposals unchanged; provenance is `annotator: human, review: quet` |
| accented / unaccented | 49 / 32 |
| overlap with annotation-v1 | 0 exact and 0 same-word matches against all 763 texts; max SequenceMatcher ratio 0.88 |
| guard | `load_split` refuses ids prefixed `probe-`; `AGENTS.md` forbids training on or paraphrasing probes |

| # | pattern | n | acc / unacc | gold types |
|---|---|---|---|---|
| 1 | lender-first `X cho mượn/vay` -> borrow | 9 | 5 / 4 | borrow 9 |
| 2 | subject-first `X mượn/vay` -> lend | 9 | 5 / 4 | lend 9 |
| 3 | incoming family gift -> income | 9 | 5 / 4 | income 9 |
| 4 | outgoing lì xì / gift money -> expense | 9 | 6 / 3 | expense 9 |
| 5 | insurance premium -> expense | 9 | 6 / 3 | expense 9 |
| 6 | insurance payout / reimbursement -> refund | 9 | 6 / 3 | refund 9 |
| 7 | windfall / lottery / prize -> income | 9 | 6 / 3 | income 9 |
| 8 | installment of a named loan product -> repayment_out | 9 | 3 / 6 | repayment_out 9 |
| 9 | title + proper name target boundary | 9 | 7 / 2 | borrow 4, lend 2, repayment_in 2, repayment_out 1 |

The probe is adversarial by design: every note targets a gap that train covers with 0–3
examples. Its numbers measure those gaps; they are not comparable to test-set scores (test type
macro-F1 is 0.862 for BamiBERT and 0.844 for MiniLM).

## 2. Metrics

- **success:** the type is correct and the target span exactly matches the gold span, including
  null = null. It is the per-record pass/fail used for classification.
- **span F1:** counts only non-null spans. For null-heavy patterns (5, 6, 7) **span exact** is
  the more informative column. Pattern 7 has only 2 non-null gold targets, so its span F1 is 0.
- **type macro-F1:** averages over the gold classes present in the slice. Only pattern 9 has more
  than one gold class; everywhere else macro-F1 would just repeat the single class's F1, so it
  is shown as n/a.
- **mean ± std:** taken across the three seeds of each encoder (sample std).

### BamiBERT (3 seeds)

| # | type acc | type macro-F1 | span F1 | span exact | success |
|---|---|---|---|---|---|
| 1 | 0.15±0.13 | n/a (1 class) | 1.00±0.00 | 1.00±0.00 | 0.15±0.13 |
| 2 | 0.67±0.11 | n/a (1 class) | 0.93±0.13 | 0.93±0.13 | 0.63±0.13 |
| 3 | 0.00±0.00 | n/a (1 class) | 0.82±0.06 | 0.74±0.06 | 0.00±0.00 |
| 4 | 0.33±0.11 | n/a (1 class) | 0.83±0.04 | 0.70±0.06 | 0.22±0.11 |
| 5 | 0.59±0.13 | n/a (1 class) | 0.44±0.10 | 0.70±0.13 | 0.37±0.06 |
| 6 | 0.37±0.17 | n/a (1 class) | 0.40±0.13 | 0.44±0.11 | 0.07±0.13 |
| 7 | 0.22±0.00 | n/a (1 class) | 0.00±0.00 | 0.78±0.00 | 0.22±0.00 |
| 8 | 0.59±0.06 | n/a (1 class) | 0.75±0.22 | 0.85±0.06 | 0.52±0.06 |
| 9 | 0.74±0.06 | 0.80±0.05 | 0.48±0.13 | 0.48±0.13 | 0.26±0.13 |
| all 81 | 0.41±0.02 | 0.44±0.01 | 0.73±0.03 | 0.74±0.02 | 0.27±0.01 |

### MiniLM (3 seeds)

| # | type acc | type macro-F1 | span F1 | span exact | success |
|---|---|---|---|---|---|
| 1 | 0.11±0.19 | n/a (1 class) | 0.81±0.06 | 0.81±0.06 | 0.07±0.13 |
| 2 | 0.44±0.19 | n/a (1 class) | 0.78±0.11 | 0.78±0.11 | 0.33±0.19 |
| 3 | 0.00±0.00 | n/a (1 class) | 0.55±0.14 | 0.44±0.11 | 0.00±0.00 |
| 4 | 0.33±0.11 | n/a (1 class) | 0.67±0.00 | 0.56±0.00 | 0.22±0.11 |
| 5 | 0.56±0.11 | n/a (1 class) | 0.36±0.34 | 0.78±0.11 | 0.44±0.11 |
| 6 | 0.59±0.06 | n/a (1 class) | 0.44±0.17 | 0.56±0.11 | 0.19±0.06 |
| 7 | 0.15±0.17 | n/a (1 class) | 0.00±0.00 | 0.74±0.06 | 0.11±0.11 |
| 8 | 0.56±0.00 | n/a (1 class) | 0.36±0.12 | 0.59±0.06 | 0.44±0.00 |
| 9 | 0.78±0.00 | 0.83±0.00 | 0.41±0.06 | 0.41±0.06 | 0.19±0.06 |
| all 81 | 0.39±0.03 | 0.46±0.04 | 0.58±0.03 | 0.63±0.01 | 0.22±0.02 |

### Success rate per run

| # | B s1 | B s2 | B s3 | M s1 | M s2 | M s3 |
|---|---|---|---|---|---|---|
| 1 | 0.00 | 0.22 | 0.22 | 0.00 | 0.00 | 0.22 |
| 2 | 0.56 | 0.78 | 0.56 | 0.44 | 0.44 | 0.11 |
| 3 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 4 | 0.33 | 0.22 | 0.11 | 0.33 | 0.22 | 0.11 |
| 5 | 0.33 | 0.44 | 0.33 | 0.44 | 0.33 | 0.56 |
| 6 | 0.22 | 0.00 | 0.00 | 0.22 | 0.11 | 0.22 |
| 7 | 0.22 | 0.22 | 0.22 | 0.11 | 0.00 | 0.22 |
| 8 | 0.44 | 0.56 | 0.56 | 0.44 | 0.44 | 0.44 |
| 9 | 0.33 | 0.11 | 0.33 | 0.22 | 0.22 | 0.11 |
| all | 0.27 | 0.28 | 0.26 | 0.25 | 0.20 | 0.22 |

### Accented vs unaccented

| encoder | form | n | type acc | type macro-F1 | span F1 | span exact | success |
|---|---|---|---|---|---|---|---|
| BamiBERT | accented | 49 | 0.40±0.05 | 0.43±0.05 | 0.78±0.04 | 0.80±0.03 | 0.28±0.03 |
| BamiBERT | unaccented | 32 | 0.42±0.04 | 0.47±0.04 | 0.65±0.05 | 0.64±0.02 | 0.26±0.04 |
| MiniLM | accented | 49 | 0.47±0.04 | 0.53±0.03 | 0.67±0.03 | 0.72±0.02 | 0.28±0.04 |
| MiniLM | unaccented | 32 | 0.27±0.02 | 0.34±0.01 | 0.42±0.03 | 0.49±0.02 | 0.14±0.02 |

Success by pattern and form (mean over the 3 seeds; each cell has only 2–7 notes):

| # | B acc / unacc | M acc / unacc |
|---|---|---|
| 1 | 0.27 / 0.00 | 0.13 / 0.00 |
| 2 | 0.67 / 0.58 | 0.60 / 0.00 |
| 3 | 0.00 / 0.00 | 0.00 / 0.00 |
| 4 | 0.28 / 0.11 | 0.17 / 0.33 |
| 5 | 0.44 / 0.22 | 0.61 / 0.11 |
| 6 | 0.11 / 0.00 | 0.28 / 0.00 |
| 7 | 0.06 / 0.56 | 0.17 / 0.00 |
| 8 | 0.67 / 0.44 | 0.67 / 0.33 |
| 9 | 0.24 / 0.33 | 0.10 / 0.50 |

- **BamiBERT:** type accuracy does not depend on accents (0.40 vs 0.42). Accents only affect its
  spans (exact match 0.80 vs 0.64).
- **MiniLM:** loses about 0.2 on type, 0.25 on span F1 and half its success rate on unaccented
  notes. On unaccented notes it scores 0.00 success in patterns 1, 2, 6 and 7.
- **Single cells such as BamiBERT 7 unaccented (0.56)** come from 3 notes. They are noise, not a
  signal.

## 3. Classification

Criteria are fixed in advance and use all six runs, never a single checkpoint.

- **reliable record:** succeeds in ≥5 of 6 runs.
- **systematic failure:** succeeds in ≤1 of 6 runs. Such a record necessarily fails in both
  encoder families.
- **A (systematic gap):** mean success across the 6 runs < 0.40 **and** ≥5 of 9 records are
  systematic failures.
- **C (mostly solved):** mean success ≥ 0.70 **and** both encoder means ≥ 0.60.
- **B (partial generalization):** everything else.

| # | mean success (6 runs) | B / M success | reliable records (≥5/6) | systematic failures (≤1/6) | of which type / span | class |
|---|---|---|---|---|---|---|
| 1 | 0.11 | 0.15 / 0.07 | 0/9 | 7/9 | 7 / 0 | **A** |
| 2 | 0.48 | 0.63 / 0.33 | 1/9 | 2/9 | 2 / 0 | **B** |
| 3 | 0.00 | 0.00 / 0.00 | 0/9 | 9/9 | 9 / 2 | **A** |
| 4 | 0.22 | 0.22 / 0.22 | 0/9 | 5/9 | 5 / 2 | **A** |
| 5 | 0.41 | 0.37 / 0.44 | 3/9 | 4/9 | 2 / 1 | **B** |
| 6 | 0.13 | 0.07 / 0.19 | 0/9 | 7/9 | 3 / 4 | **A** |
| 7 | 0.17 | 0.22 / 0.11 | 0/9 | 6/9 | 6 / 2 | **A** |
| 8 | 0.48 | 0.52 / 0.44 | 4/9 | 4/9 | 4 / 0 | **B** |
| 9 | 0.22 | 0.26 / 0.19 | 1/9 | 6/9 | 2 / 3 | **A** |

- **A:** 1, 3, 4, 6, 7, 9.
- **B:** 2, 5, 8.
- **C:** none.

No pattern is a one-off. Even the B patterns have 2–4 records that all six runs get wrong.

## 4. Error analysis (all patterns are A or B)

### 1. Lender-first `X cho mượn/vay` → borrow — A, type

- **Mistake:** the models predict **lend** in 44 of 54 predictions. Spans are fine (BamiBERT
  1.00; MiniLM 0.81, misreading slang `thg` as part of the name).
- **Both encoders:** fail the same way, including `ông ngoại`, `sếp`, `Vinh cho minh vay`, and
  `được cô Hà cho vay`, where even the explicit beneficiary (`minh`, `được`) is ignored.
- **Cause:** a lexical shortcut, `cho` + `mượn/vay` → lend, regardless of word order. Train
  has 1 lender-first note against 35 `cho X mượn` → lend.
- **Accents:** unaccented notes are 0/4 in every run. The 4 correct BamiBERT predictions are all
  accented.

### 2. Subject-first `X mượn/vay` → lend — B, type

- **BamiBERT:** mostly correct (0.67).
- **MiniLM:** flips to **borrow** (10 of 27), so it reads `mượn/vay` as "user borrows" whatever
  the subject. It succeeds on 0/4 unaccented notes.
- **Both encoders:** notes with a repayment promise (`hẹn tháng sau trả`, `mai trả`,
  `cuối năm trả`) pull towards **repayment_in** (11 predictions).
- **Spans:** errors are the title boundary from pattern 9 (`anh Duc`, `thg`).
- **Cause:** direction is learned only from the verb-first form (`mượn X` → borrow, 36 train
  notes).

### 3. Incoming family gift → income — A, type

- **Mistake:** 0 of 54 predictions are income. BamiBERT says **lend** (16) or **transfer** (9).
  MiniLM says lend (9), **expense** (9) or transfer (8).
- **Cause:** the same `cho` shortcut as pattern 1. `<kin> cho N` looks like lending, and
  `gửi`/`dúi` looks like transfer. Train has 0 incoming kin gifts.
- **Spans:** fine for accented kin terms. They fail on unaccented ones (`bo me` → `me`,
  `chi hai` → `hai`/null, `cau` → null), and MiniLM is worse at this.

### 4. Outgoing lì xì / gift → expense — A, type

- **Mistakes:** the same in both encoders:
  - `lì xì bé Bin` → **income** in 6/6 (the one train lì xì is incoming);
  - `cho mẹ 2tr` → **lend** in 6/6;
  - `gửi ba … biếu tết` → **transfer** with a null span in 6/6;
  - `li xi ong ngoai` → transfer and null.
- **What works:** `mừng tuổi`, `biếu` and accented `tặng` partly work.
- **Cause:** the cue verbs (`lì xì`, `cho`, `gửi`) are tied to one direction or to transfer.
  Gift direction is not learned in either direction (see pattern 3).

### 5. Insurance premium → expense — B, type and span

- **What works:** explicit purchase wording is reliable (`phí bảo hiểm …`, `mua bảo hiểm …`,
  `đóng bảo hiểm …`).
- **Type failures:**
  - wording without a verb (`bao hiem xe hoi nam nay`) → income, refund or transfer in 6/6;
  - `gia hạn` (renewal) → repayment_out;
  - a named insurer with `đóng phí` (`prudential`) → scattered types.
- **Span failure:** named insurers after `phí bh` (`aia`) are missed (null) in 6/6.
- **Cause:** `bảo hiểm` alone does not tell the models the direction. Both encoders behave the
  same; MiniLM is worse on unaccented notes (0.11).

### 6. Insurance payout → refund — A, mostly span

- **Spans:**
  - Both encoders tag the generic insurer word (`bảo hiểm`, `bh`, `baohiem`) as the target,
    although gold is null. 4 records fail on span in ≥5/6 runs.
  - Named insurers (`prudential`, `bảo việt`) are mostly found.
- **Type:**
  - BamiBERT reads `X trả/chi trả` as **repayment_in** (8 predictions);
  - both encoders read `nhận tiền bảo hiểm` / `tiền bảo hiểm … về` as income or expense;
  - MiniLM's type accuracy is higher (0.59 vs 0.37), but its success rate is still 0.19.
- **Cause:** `X trả tiền` → repayment_in is a direction shortcut, and nothing teaches that a
  generic insurer word is not a counterparty.

### 7. Windfall / lottery / prize → income — A, type

- **Mistake:** both encoders predict **transfer** (23) or **expense** (15). Income appears in
  only 10 of 54 predictions.
- **What works:** only notes with a prize noun (`giải`) are sometimes right.
- **Spans:** the two named sources fail. `minigame shopee` is tagged `minigame`, and `vietlott`
  is mostly null or fragmented.
- **Cause:** a coverage gap. Train has 0 windfalls, and `trúng` has no learned meaning.
- **Found money** (`nhat duoc 200k ngoai duong`, human-ruled income) is split 3/6.

### 8. Named loan-product installment → repayment_out — B, type

- **What works:** notes that start with the action (`thanh toán/góp/trả … khoản vay …`) are
  solved 6/6.
- **What fails:** notes that **start with `vay`** (`vay mua nhà techcombank trả kỳ…`,
  `vay ngan hang tra thang nay`) → **borrow** in 6/6. `dong tien vay tin chap` → borrow or
  transfer in 5/6.
- **Spans:** `ngan hang` is truncated to `hang` (4 runs, both encoders), and MiniLM drops `mb`
  in all 3 runs.
- **Cause:** the leading `vay` is a shortcut for borrow, and the payment verb that follows is
  ignored. Unaccented notes are weaker (0.33–0.44).

### 9. Title + proper-name target boundary — A, span

- **Type:** correct except on the two `X cho vay/mượn` notes, which carry over pattern 1.
- **Spans:** both encoders often keep the title (`cô Lan`, `dì Hương`, `bác Thành`,
  `dì Loan`) or tag only the title (`cô`, `dì`, `bác`, `chú`). The birth-order control
  `chú Tư` is always spanned correctly.
- **Consistency:** the boundary changes from seed to seed for the same note, so the rule
  "strip the title before a proper name" is learned for some titles (27 train exclusions) but
  not after `chú/cô/dì/bác`.
- **Accents:** do not explain it (unaccented cells are 2 notes).

### Cross-cutting findings

- **Shortcuts:** every type gap comes from a lexical direction shortcut on a frequent cue word:
  `cho` (patterns 1, 3, 4), `vay` (patterns 2 and 8), `X trả` (pattern 6) and `gửi` (patterns 3
  and 4).
- **Shared errors:** BamiBERT and MiniLM share the dominant wrong type in every A pattern. This
  is a data effect, not an encoder effect.
- **Encoder differences:** MiniLM differs mainly in being fragile on unaccented text and in its
  span boundaries.

## 5. Targeted-02 recommendation (not generated)

Sizing:
- A patterns get 12–20 new notes, and B patterns 6–12.
- Mirror notes give each minimal pair the opposite label in the same surface form.
- Train is 60% accented. MiniLM's main failure is unaccented text, so targeted-02 uses **50/50**
  for the direction families and **60/40** elsewhere.
- Every target uses annotation-v1 span rules: titles stripped before proper names, birth-order
  names and kin terms kept, generic institutions null.

| block | patterns | new | ratio | minimal-pair structure | span coverage |
|---|---|---|---|---|---|
| D1 lender-first | 1 (A), 9 | 20 borrow + 8 lend mirrors | 50/50 | `X cho mượn/vay N` ↔ `cho X mượn/vay N`, same X and amount. Variants balanced across `cho mượn`, `cho vay`, `cho mình/tao/em vay`, `được X cho vay`. | yes: half the X are title + proper name (`cô/chú/dì/bác` + name), the rest kin, bare names, slang (`thg`) |
| D2 subject-first | 2 (B) | 10 lend + 4 borrow mirrors | 50/50 | `X mượn/vay N` ↔ `mượn/vay X N`. Half carry a promise tail (`hẹn mai trả`) so the tail stops implying repayment_in. | yes: titles stripped (`anh Duc` → `Duc`) |
| G gift direction | 3 (A), 4 (A) | 14 income + 14 expense | 50/50 | 7 strict pairs `<kin> cho/gửi/lì xì N` ↔ `cho/gửi/lì xì/biếu <kin> N`. Plus `dúi`, `mừng tuổi`, `tặng`, collective kin (`bố mẹ`, `ông bà`). Contrast with `<kin> cho mượn` comes from D1. | yes: unaccented kin terms (`bo me`, `chi hai`, `cau`) |
| I insurance | 6 (A), 5 (B) | 10 refund + 6 expense | 60/40 | the same insurer phrase in both directions (`bảo hiểm xe trả 700k` ↔ `đóng bảo hiểm xe 700k`). Verbs: `trả/chi trả/bồi thường/hoàn/thanh toán`, `nhận tiền bảo hiểm`; premiums without a verb and `gia hạn`. | yes: generic `bảo hiểm/bh` → null (≥6) vs named insurer → span (≥4: `prudential`, `aia`, `bảo việt`, `manulife`) |
| W windfall | 7 (A) | 12 income + 3 expense contrasts | 60/40 | `trúng` + (`xổ số`, `vé số`, `vietlott`, `lô`, `quay số`, `bốc thăm`, `minigame`, prize `giải`) → income. Contrast: `mua vé số 20k` → expense. | small: 3–4 named platforms (`shopee`, `vietlott`); the rest null |
| L loan installment | 8 (B) | 8 repayment_out + 3 borrow mirrors | 50/50 | `vay <product/bank> … trả/đóng/góp kỳ N` ↔ `vay <bank> N` (borrow). Half of the repayments **start with `vay`**. | yes: bank names (`mb`, `techcombank`, `ngân hàng`) |
| T title boundary | 9 (A) | 8 | 60/40 | non-loan types (repayment_in/out, expense) with `chú/cô/dì/bác` + name → name only, paired with birth-order names kept (`chú Tư`, `cô Út`). | this block is span-only |

**Total: about 120 notes** (about 24% of the 501 train notes). D1, G and T are the most
important. If the budget must be smaller, drop the mirrors in D2 and L first (−7). No pattern is
C, so none is excluded on that ground.

Guardrails:
- Write targeted-02 from these pattern descriptions only. Do not use probe or test notes.
- Check similarity against annotation-v1 **and probe-v1**, so the probe stays an honest
  held-out diagnostic.
- After retraining, re-score all six runs on both the test split and probe-v1, and watch the
  mirror directions for regressions (`cho X mượn` → lend, `mượn X` → borrow).

## 6. Teacher choice

BamiBERT remains the preferred teacher, on a narrow margin:
- **Where BamiBERT wins:** the success rate is 0.27 vs 0.22, and span F1 is 0.73 vs 0.58 with
  exact match 0.74 vs 0.63. It is also much more robust on unaccented text (type 0.42 vs 0.27,
  span F1 0.65 vs 0.42).
- **Where they tie:** type accuracy is a tie (0.41 vs 0.39), and both share every type gap.
- **Caveat:** neither checkpoint should pseudo-label notes from the nine probed patterns until
  targeted-02 is trained. On those patterns the teacher is confidently wrong in the same
  direction as the student.
