# baseline-v1 error audit: failures shared by BamiBERT and MiniLM

This is analysis only. Nothing was trained, no label was edited, and the split is unchanged.
Sources:

- the frozen `docs/annotation-v1.md`;
- `datasets/annotation-v1/combined/{labels,provenance}.jsonl`;
- the six final runs in `runs-e20/` (3 seeds × 2 encoders);
- `datasets/annotation-v1/splits/train.jsonl`, used for the coverage counts only. No
  validation or test label was used beyond the audited records.

Classes: **A** genuine model error · **B** annotation error · **C** annotation-v1 rule ambiguity
· **D** insufficient training coverage / linguistic pattern gap.

## Predictions, all six runs

The request quoted some predictions that do not match the prediction files. Below are the
actual test predictions, `type / span`. Bami s2 and MiniLM s3 are the best runs used in
`report.md`.

| note | gold | Bami s1 | Bami s2 | Bami s3 | MiniLM s1 | MiniLM s2 | MiniLM s3 |
|---|---|---|---|---|---|---|---|
| `vay tiêu dùng trả kỳ này 3tr4` | repayment_out / ∅ | borrow / `tiêu dùng` | borrow / ∅ | borrow / `tiêu dùng` | borrow / `tiêu dùng` | borrow / `tiêu dùng` | borrow / `tiêu dùng` |
| `lì xì cháu 200k` | expense / `cháu` | income / ✓ | income / ✓ | income / ✓ | income / ✓ | income / ✓ | income / ✓ |
| `trúng vé số 200k` | income / ∅ | transfer / ✓ | transfer / ✓ | repayment_out / ✓ | **income** / ✓ | transfer / ✓ | repayment_out / ✓ |
| `dong bao hiem nhan tho 1tr2` | expense / ∅ | transfer / ✓ | repayment_in / ✓ | income / ✓ | borrow / `dong bao` | income / `do` | income / `dong` |
| `bố cho tiền tiêu 1 triệu` | income / `bố` | lend / ✓ | lend / ✓ | borrow / ✓ | expense / ✓ | lend / ✓ | repayment_in / ✓ |
| `co Thuy cho muon 1tr` | borrow / `Thuy` | lend / ✓ | lend / ✓ | lend / ✓ | lend / ✓ | lend / ✓ | lend / ✓ |
| `bao hiem tra tien kham 850k` | refund / ∅ | repayment_in / `bao` | repayment_in / `bao` | repayment_in / `bao hiem` | repayment_out / ✓ | **refund** / `em` | transfer / ✓ |
| `anh Long mượn 3 triệu sửa xe` | lend / `Long` | ✓ / ✓ | borrow / ✓ | ✓ / ✓ | ✓ / ✓ | borrow / ✓ | borrow / `anh` |
| `bách hoá xanh rau thịt 187k` | expense / `bách hoá xanh` | ✓ / ∅ | ✓ / ∅ | ✓ / ∅ | ✓ / ∅ | ✓ / ∅ | ✓ / ∅ |
| `vay chú Hải 20tr mua xe` | borrow / `Hải` | ✓ / `chú` | ✓ / `chú Hải` | ✓ / `chú Hải` | ✓ / `chú Hải` | ✓ / `chú Hải` | ✓ / `chú Hải` |

Corrections to the request:

- **`anh Long mượn` span:** BamiBERT never predicted `anh Long`. It found `Long` in all three
  seeds. Only one MiniLM seed predicted `anh`. So this is not a shared span failure. Its type
  error is shared by the best runs (3 of 6 runs predict borrow).
- **Best-run type predictions:** `trúng vé số` is BamiBERT transfer and MiniLM repayment_out;
  `dong bao hiem` is BamiBERT repayment_in and MiniLM income; `bố cho tiền tiêu` is BamiBERT
  lend and MiniLM repayment_in. The wrong answers vary by seed, which is itself a sign of
  missing coverage: no seed learned a stable decision.

## Per-record audit

### T1. `vay tiêu dùng trả kỳ này 3tr4` (type)

- **Gold:** complete / repayment_out / null (AI-accepted, baseline-01).
- **Rule:** "Financed balances": paying an existing financed balance or liability, including
  installments, is `repayment_out`; the target is null unless a lender is named.
- **Is the gold supported?** Yes. `vay tiêu dùng` names the product (a consumer loan), and
  `trả kỳ này` means paying this period's installment.
- **Another reading?** Not genuinely. Nothing in the note describes receiving principal, and
  `borrow` requires that.
- **Class: D.**
  - In train, `vay` appears in 20 notes (10 borrow, 7 lend, 2 repayment_out, 1 repayment_in).
  - Installment wording appears in 19 repayment_out notes, but only as `trả góp` / `góp` /
    `kỳ N` with the purchased item (`trả góp iphone kỳ 4`).
  - Only 3 train notes name a loan product: `tra kỳ vay mua nhà 8tr` and `tất toán khoản vay`
    → repayment_out, and `vay tín chấp vpbank` → borrow. (The first was missed by the original
    regex scan and found during the probe build.)
  - The models key on the leading `vay`.
- **Action:** add future training coverage: paying an installment of a named loan product
  (`vay …` + `trả kỳ`, `đóng kỳ`, `tất toán`) with no lender named.

### T2. `lì xì cháu 200k` (type)

- **Gold:** complete / expense / `cháu` (human-reviewed).
- **Rule:** "Beneficiary is not payee": when money itself goes to a person, that person is the
  target. `lì xì cháu 200k` → `cháu` is a verbatim example in the doc. Expense is the only type
  consistent with money leaving the user as a gift.
- **Is the gold supported?** Yes, explicitly.
- **Another reading?** No. A younger relative (`cháu`) giving lì xì to the user is not the
  natural reading, and the verb-first form means the user is giving.
- **Class: D.** Train has one lì xì note, `lì xì bà cho 500` → income. It has no outgoing
  lì xì, and only one outgoing gift with a kinship target (`quà 20/10 cho mẹ`). Every model
  copied the one incoming example.
- **Action:** add future training coverage: outgoing gift money to relatives
  (`lì xì <kin>`, `mừng tuổi <kin>`, `cho <kin> tiền …`) contrasted with incoming
  (`được lì xì`, `<kin> lì xì`).

### T3. `trúng vé số 200k` (type)

- **Gold:** complete / income / null (AI-accepted).
- **Rule:** the definition of `income` includes "rewards, or other ordinary incoming money that
  is neither borrowing, debt repayment, nor refund".
- **Is the gold supported?** Yes. `trúng` means the user won.
- **Another reading?** Weak. `vé số 200k` alone could be buying a ticket (expense), but `trúng`
  rules that out.
- **Class: D.** Train has 0 lottery or prize notes (`trúng`, `xổ số`, `vé số`, `vietlott`).
  Windfall-type income appears only as `thưởng` (8 notes, all bonuses). The predictions are
  scattered: transfer, repayment_out, and income in one run.
- **Action:** add future training coverage: windfalls and prizes (`trúng số`, `trúng thưởng`,
  `quay số trúng`, cashback lottery).

### T4. `dong bao hiem nhan tho 1tr2` (type; MiniLM also spans)

- **Gold:** complete / expense / null (human-reviewed).
- **Rule:** "Insurance payout / tax refund": paying a premium is spending. This exact note is an
  edge-case row in the doc.
- **Is the gold supported?** Yes, explicitly.
- **Another reading?** No.
- **Class: D.** Train has 2 insurance notes: `bảo hiểm y tế năm 1tr1` → expense and
  `hoan tien bao hiem 670k` → refund. None is an unaccented premium with `dong` / `đóng`. The
  predictions scatter across four types, and MiniLM even tags `dong` as a counterparty.
- **Action:** add future training coverage: insurance premiums (`đóng/dong bảo hiểm …`,
  `phí bảo hiểm`, `gia hạn bảo hiểm`), accented and unaccented.

### T5. `bố cho tiền tiêu 1 triệu` (type)

- **Gold:** complete / income / `bố` (human).
- **Rule:** edge row 3 (`mẹ cho 2tr` → income; `cho` without `mượn` / `vay` is a gift), and the
  kinship-only target rule.
- **Is the gold supported?** Yes. `tiền tiêu` means pocket money.
- **Another reading?** No. The kinship subject before `cho` makes the user the recipient.
- **Class: D.** Train has 0 notes of the form `<kin> cho …` without a loan word. Its only
  kinship-first `cho` note is `bố cho mượn 5tr mua xe` → borrow. On the other side, 35 notes
  are `cho X mượn` → lend and 8 are `cho em X mượn/vay` → lend. The models learned that `cho`
  means lending out.
- **Action:** add future training coverage: incoming family gifts (`<kin> cho (tiền) …`,
  `<kin> gửi tiền tiêu`), balanced against `<kin> cho mượn` (borrow) and `cho <kin> …`
  (outgoing).

### T6. `co Thuy cho muon 1tr` (type)

- **Gold:** complete / borrow / `Thuy` (AI-accepted, targeted-01).
- **Rule:** edge row 4, `mẹ cho mượn 2tr` → borrow (`X cho mượn` means X lends to the user). The
  title before a proper name is excluded, so the target is `Thuy`.
- **Is the gold supported?** Yes. Read unaccented as `cô Thủy cho mượn`, or even as
  `có Thủy cho mượn`, the lender is Thủy.
- **Another reading?** Not genuinely. The lend form needs `cho` before the borrower
  (`cho cô Thủy mượn`).
- **Class: D.** It is the clearest imbalance in the data. Train has 35 notes of the form
  `cho X mượn/vay` → lend (including one 23-note template group), but only 1 of the form
  `X cho mượn` → borrow (`bố cho mượn 5tr mua xe`). The models reduced the pattern to
  "`cho` + `mượn` means lend", whatever the word order. All 6 runs fail.
- **Action:** add future training coverage: `<person> cho mượn/vay` (borrow), with names,
  titles and kinship terms, accented and unaccented, as minimal pairs against `cho <person> mượn`.

### T7. `bao hiem tra tien kham 850k` (type; BamiBERT also spans)

- **Gold:** complete / refund / null (AI-accepted, targeted-01).
- **Rule:** "Insurance payout": claim reimbursements are refund, and the doc gives
  `bảo hiểm chi trả viện phí 3tr` → refund / null. Generic `bảo hiểm` is not a named insurer,
  so the target is null.
- **Is the gold supported?** Yes.
- **Another reading?** Only in form. The debt-evidence rule says a bare `X trả …` does not
  establish repayment_in, and the insurance rule does not list the bare verb `trả` (it shows
  `chi trả`, `hoàn tiền`, `bồi thường`). Read with both rules, the note is still refund, not
  repayment_in. So the wording is slightly incomplete, not contradictory.
- **Class: D, with a minor C.** Train has 1 insurance payout (`hoan tien bao hiem`) and none
  with `trả`. The models map `X trả` to repayment_in, and BamiBERT also tags `bao` / `bao hiem`
  as the payer.
- **Actions:**
  - Add future training coverage: insurance payouts (`bảo hiểm trả/chi trả/thanh toán/bồi
    thường …`) with generic and named insurers.
  - Clarify the doc for a future version (see wording below).

### T8. `anh Long mượn 3 triệu sửa xe` (type; the span is not a shared failure)

- **Gold:** complete / lend / `Long` (human).
- **Rules:** "User perspective" (`X mượn` means X borrows from the user) and title exclusion
  before a proper name.
- **Is the gold supported?** Yes.
- **Another reading?** No.
- **Class: D, partly A.** Train has only 3 subject-first `X mượn` → lend notes, against 36
  verb-first `mượn/vay X` → borrow. 3 of 6 runs get it right, so the models partly generalize.
  The one `anh` span prediction (MiniLM s3) is a plain model error.
- **Action:** add future training coverage: subject-first `<person> mượn/vay …` (lend).

### S1. `bách hoá xanh rau thịt 187k` (span)

- **Gold:** complete / expense / `bách hoá xanh` (AI-accepted).
- **Rule:** "Named merchants … are targets"; "Do not leave a named counterparty null".
- **Is the gold supported?** Yes. Bách Hóa Xanh is a supermarket chain.
- **Another reading?** Weak. `bách hoá` alone is a generic noun ("general store"), which would
  be null under the generic-noun ruling. With `xanh`, it is the chain name.
- **Class: A, with some D.** Train has 18 expense notes with multi-word merchant targets and 13
  grocery-chain notes in the same shape (`winmart rau thịt cá 342,000`, `circle k …`). The
  span shape (merchant first, then items) is well covered. What fails is a merchant whose name
  is made of ordinary Vietnamese words: every train merchant is a brand-like token (`winmart`,
  `circle k`, `long châu`, `phúc long`). Train has 0 notes containing `bách hoá`.
- **Action:** mostly no action; genuine model mistake. Optionally add future coverage of chains
  whose names are common words (`điện máy xanh`, `thế giới di động`, `bách hoá xanh` itself
  only in new, non-test phrasings).

### S2. `vay tiêu dùng trả kỳ này 3tr4` (span)

- **Gold:** null. The loan product is not a party, and the "items and bill names are never
  targets" rule applies.
- **Is the gold supported?** Yes.
- **Another reading?** No. `tiêu dùng` means "consumer" and is not a lender.
- **Class: D**, the same gap as T1. After `vay`, train almost always has the lender
  (`vay ông ngoại`, `vay chú Ba`, `vay tín chấp vpbank`), so the models tag whatever follows.
- **Action:** the same coverage as T1.

### S3. `anh Long mượn 3 triệu sửa xe` (span)

This is not a shared failure; see T8. BamiBERT found `Long` in 3 of 3 runs, and MiniLM in 2 of 3.
- **Class: A**, for MiniLM s3 only.
- **Action:** none.

### S4. `vay chú Hải 20tr mua xe` (span)

- **Gold:** complete / borrow / `Hải` (AI-accepted).
- **Rule:** exclude a kinship title before a proper name. `Hải` is a name, not a birth-order
  word; the keep-the-phrase rule covers `Hai` (the second-born), not `Hải`.
- **Is the gold supported?** Yes. It hinges on one diacritic (`Hải` vs `Hai`).
- **Another reading?** Not for this accented note. For an unaccented `chu Hai`, both readings
  would be plausible. The doc resolves the analogous `cau Tu` → `Tu` case but does not state
  the general rule.
- **Class: D, with C for the unaccented variant only.** Every `chú` target in train keeps the
  title: `vay chú Ba`, `cho chú Hai vay`, `chú Hai trả tiền mượn`. There are 0 notes of the
  form `chú <proper name>` with the title excluded. Title exclusion is otherwise well covered
  (27 notes), just never after `chú`. The models learned "`chú` + next word".
- **Actions:**
  - Add future training coverage: `chú/cô/dì/bác` + proper name (title excluded), as minimal
    pairs with birth-order names.
  - Clarify the doc for a future version (see wording below).

## Pattern analysis (train only)

Counts come from regex scans of `train.jsonl` (501 notes). No test note was used to write
patterns or examples.

| pattern | failing notes | relevant train examples | verdict |
|---|---|---|---|
| lender-first `X cho mượn/vay` → borrow | T6 | **1** (vs 35 `cho X mượn` → lend) | coverage gap + strong opposing bias |
| subject-first `X mượn` → lend | T8 | 3 (vs 36 verb-first `mượn/vay X` → borrow) | coverage gap; partial generalization |
| incoming family gift `<kin> cho …` → income | T5 | **0** (only `lì xì bà cho`) | coverage gap |
| outgoing gift money `lì xì <kin>` → expense | T2 | **0** outgoing lì xì; 1 incoming; 1 outgoing gift to kin | coverage gap |
| insurance premium → expense | T4 | 1 (`bảo hiểm y tế năm`) | coverage gap |
| insurance payout / claim → refund | T7 | 1 (`hoan tien bao hiem`) | coverage gap |
| windfall / lottery → income | T3 | **0** (8 `thưởng` bonuses) | coverage gap |
| installment of a named loan product → repayment_out, null | T1, S2 | 3 loan-product notes (2 repayment_out, 1 borrow); 19 item installments | coverage gap (`vay` bias) |
| title + proper name after `chú`/`cô`/`dì`/`bác` | S4 | **0** after `chú` (3 kept birth-order `chú` phrases); 27 title exclusions overall | coverage gap, specific to the title word |
| multi-word merchant spans | S1 | 18 multi-word merchant targets; 13 grocery-chain notes | covered; generalization failure on a common-word name |

The recurring cause is a data effect. The baseline corpus and targeted-01 teach the frequent
direction of each construction (`cho X mượn`, `mượn X`, incoming `lì xì`), and almost never its
mirror. Both encoders learn the same lexical shortcut, which is why they fail on the same notes.

## Proposed wording for a future annotation version (not applied to v1)

1. **Insurance verbs.** "An insurer paying the user for a claim is `refund` whatever the verb
   (`trả`, `chi trả`, `thanh toán`, `bồi thường`, `hoàn`). This overrides the bare-`X trả`
   rule for repayment_in. The target is null unless an insurer is named (`bảo việt`,
   `prudential`)."
2. **Birth-order vs proper name without diacritics.** "When an unaccented word after a title
   could be a birth-order word or a name (`chu Hai`, `co Nam`, `cau Tu`), prefer the proper-name
   reading and exclude the title, unless the note's other words are accented and the
   birth-order form is written. Record the alternative in `note` if in doubt." This generalizes
   the existing `cau Tu` → `Tu` ruling.
3. **`cho` + kinship, outgoing.** This is not a failure here, but it borders T2 and T5. The
   human ruling on targeted record 82 (`cho em 300k` → expense / `em`) is not written in the
   doc, while edge row 9 keeps `cho Nam 500k` uncertain. Future wording: "`cho <kinship term>
   N` with no loan word is an outgoing gift (`expense`, target = the kinship term); `cho
   <proper name> N` remains uncertain."

## Conclusions

1. **Confirmed model errors:** only two, both minor. S1 (`bách hoá xanh` missed, although
   merchant spans are well covered) and S3 (`anh` span in one MiniLM run). Everything else is
   driven by coverage.
2. **Suspected annotation errors:** none. All 10 audited gold labels follow the frozen rules,
   and two of them (T2, T4) are verbatim doc examples. No label re-check is required.
3. **Rule ambiguities:** none that changes any audited gold label. Two wording gaps are worth
   fixing in a future version: the insurance-payout verb list (T7) and birth-order vs name
   without diacritics (S4, unaccented variant). The undocumented `cho <kin>` outgoing-gift
   ruling (record 82) should also be written down.
4. **Training-coverage gaps:**
   - lender-first `X cho mượn`;
   - subject-first `X mượn`;
   - incoming family gifts;
   - outgoing lì xì and gift money;
   - insurance premiums;
   - insurance payouts;
   - windfalls;
   - loan-product installments;
   - `chú`/`cô`/`dì`/`bác` + proper name.
5. **Frozen dataset and test split:** keep both unchanged. No annotation error was found, so
   there is nothing to correct. The current test set remains the held-out benchmark. Any
   future data for these patterns must go into a new dataset version as training data only,
   written from the pattern descriptions above, never from the test notes, and evaluated
   against the same fixed test split.
6. **Smallest safe next experiment before distillation:** a diagnostic probe set with no
   training.
   - Write about 8–10 fresh notes per pattern above (about 80 notes). Include minimal pairs,
     both directions, and accented and unaccented forms, and avoid every train, validation and
     test text and template.
   - Label them in Quet under annotation-v1 and keep them as a separate `probe` set that is
     never trained on.
   - Score the existing 6 checkpoints on it.
   - This shows whether each gap is systematic, which needs augmentation, or a one-off, and it
     turns 1–2 test notes per pattern into a measurable number. Only then generate the
     targeted-02 training augmentation for the confirmed patterns.
