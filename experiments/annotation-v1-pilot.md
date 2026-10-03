# annotation-v1 pilot analysis (first 140 labels)

Inputs: `datasets/annotation-v1/{queue,labels}.jsonl`, `configs/annotation-v1.yaml`,
`docs/annotation-v1.md`. Labels were not modified. Nothing was trained.

Validator: `uv run python scripts/validate_annotations.py datasets/annotation-v1/labels.jsonl`
→ `140 annotation(s), 0 error(s) [complete 139, uncertain 1]`. The labels are structurally
valid and every span matches its offsets. All problems below are about meaning, which the
validator does not check.

The labels cover queue positions 1–140 contiguously (140 of 600).

## 1. Total annotated

140.

## 2. Status

| status | n |
|---|---|
| complete | 139 |
| uncertain | 1 |
| skipped | 0 |

With 0.7% uncertain, annotators almost never used `uncertain`. The spec expects it for
notes like `trả Nam 500k` / `cho Nam 500k`. The next 460 labels should show whether the pilot
simply had no such notes or whether annotators are forcing a type.

## 3. Transaction types (all 140, including the one uncertain record, which carries `expense`)

| type | n | % |
|---|---|---|
| expense | 57 | 40.7 |
| transfer | 21 | 15.0 |
| borrow | 19 | 13.6 |
| income | 18 | 12.9 |
| repayment_out | 8 | 5.7 |
| lend | 8 | 5.7 |
| refund | 5 | 3.6 |
| repayment_in | 4 | 2.9 |

## 4. Target present vs null

Overall: present 59 (42%), null 81 (58%).

| type | target | null |
|---|---|---|
| expense | 16 | 41 |
| transfer | 0 | 21 |
| borrow | 19 | 0 |
| income | 6 | 12 |
| lend | 8 | 0 |
| repayment_out | 6 | 2 |
| refund | 0 | 5 |
| repayment_in | 4 | 0 |

`refund` has 0 of 5 targets, even though every refund note names `shopee`. The spec's own
example is `shopee hoàn 299k` → `shopee` (see re-check candidates).

## 5. Accented vs unaccented

Pilot: 83 accented (59%), 57 unaccented (41%). Queue: 361 / 239 (60% / 40%), so the pilot
matches the queue.

| type | accented | unaccented |
|---|---|---|
| expense | 35 | 22 |
| transfer | 10 | 11 |
| borrow | 13 | 6 |
| income | 9 | 9 |
| repayment_out | 5 | 3 |
| lend | 4 | 4 |
| refund | 3 | 2 |
| repayment_in | 4 | **0** |

## 6. Uncertain records by ambiguity reason

| reason | pos | id | text | annotation |
|---|---|---|---|---|
| none recorded (no `note`) | 4 | baseline-01-1e11850c97d6 | `ship đồ ăn baemin 67k` | uncertain, type `expense`, target null |

This is the only uncertain record, and it has no reason. The note reads as an ordinary
food-delivery expense, so it is also listed as a re-check candidate. None of the spec's
ambiguity classes (type undetermined, direction unclear, several counterparties) appear in
the pilot.

## 7. Types with very few examples

- `repayment_in`: 4, all accented and all in the `X trả (lại) …` form. There are no unaccented
  examples and none phrased as `trả nợ tao`.
- `refund`: 5. Four are shopee refunds and one is card cashback. Employer reimbursements were
  labelled `income` (see below), so there are effectively no non-merchant refunds.
- `lend` and `repayment_out`: 8 each.
- `uncertain`: 1. `skipped`: 0.

At the 70/15/15 split, `repayment_in` would leave about 1 example each for val and test. To
cover it, the rest of the queue has to deliver roughly 10× more, or the corpus needs targeted
generation.

## 8. Likely confusion boundaries

- **income vs repayment_in.** The pilot separates these correctly when the wording is explicit:
  `Minh trả lại tiền ăn` / `Duy trả tiền ăn` → repayment_in, and `bố cho tiền tiêu` →
  income. There are no `X gửi 500k`-style notes yet, which is the spec's canonical ambiguous
  case.
- **income vs borrow.** `bố cho tiền tiêu` → income and `vay bố 2tr` → borrow are consistent. The
  real ambiguity is **debt-state notes**: `còn nợ X 300k (tiền ăn / tiền vé)` → borrow at positions
  2, 27, 77, 103, 124, and `nợ tiền nhà bà chủ 1 tháng` → borrow at 111. These notes record an
  unpaid obligation (a shared bill, rent arrears), not money the user received. The `borrow`
  definition ("the user receives loan principal") does not clearly cover them.
- **expense vs repayment_out.** This is the largest inconsistency cluster. Installments are split:
  - `góp laptop fe credit` → repayment_out (125), but `home credit ky nay` → expense (138);
  - `trả góp xe` / `góp xe` → repayment_out (11, 12, 43, 57);
  - credit-card payoff `thanh toán dư nợ thẻ` → expense (90, 106);
  - `tra no tien dien may hom truoc` → expense (102).

  The spec does not say whether paying a financed purchase or a card balance is `repayment_out`
  or `expense`.
- **refund vs income.** `cty tra tien cong tac phi` (41) and `cty hoan tien cong tac` (119) →
  income contradict the resolved employer-reimbursement rule. `cashback the tin dung` → refund
  (85) is not a reversal of a specific expense.
- **lend vs expense.** No confusions. All 8 lend notes contain `mượn`/`muon`.

## 9. Target-span patterns that may be inconsistent

- **The item or activity is used as the target instead of a counterparty:**
  - `cơm gà xối mỡ` (10), `đổ xăng` (15), `nhà` in `tiền nhà` (3), `ccq quỹ đầu tư` (9);
  - `tra gop xe sh` / `trả góp xe máy` (11, 12), which include the verb.
- **Companions used as the target:** `Linh` in `tra sua vs Linh` (65), `mn` in `lẩu vs mn` (29),
  and `lau vs team` (16), which spans the dish, `vs`, and the companion.
- **Kinship/title prefix not stripped:** `anh Long` (110), `chi Hoa` (127), and `bé Na` (140). In
  other records the prefix was stripped: `Hạnh`, `Thảo`, `Mai`, `Long`, `Tu`, `Tí`.
- **A merchant named in the text is sometimes left null:**
  - `spotify 59k` → null (122) vs `spotify family 59k` → `spotify` (5);
  - `netflix` → target (94);
  - `grabfood` → target (20) vs `baemin` → null (4);
  - `highlands` (23, 107), `circle k` (26, 84, 96), `fahasa` (62), `long châu` (69, 134) → null;
  - `shopee` in all four refunds → null.
- **A beneficiary used as the target:** `bé` in `mua vở bút cho bé` (35) is who the purchase was
  for, not who was paid. Among gift recipients, `mừng cưới Tuấn` → `Tuấn` and `qua sinh nhat bé Na`
  → `bé Na` have targets, but `mua hoa qua biếu bà` (131) → null.

## Manual re-check candidates

These are high-precision flags only. Each one contradicts a written rule or example, or
conflicts with a near-identical note.

| pos | id | text | current | reason |
|---|---|---|---|---|
| 135 | baseline-01-ba950489d8ac | `Hoa còn nợ mình 350k` | complete, borrow, `Hoa` | Direction is reversed. Hoa owes the user, so the user is the lender. `borrow` means the user owes. |
| 136 | baseline-01-a86c7f2f4515 | `trả nợ spaylater 560k` | complete, repayment_in, `spaylater` | Direction is reversed. The user is paying the BNPL provider, so this is repayment_out. |
| 119 | baseline-01-5c806d85f402 | `cty hoan tien cong tac 1,2tr` | complete, income, null | This is almost the spec's own example (`cty hoàn tiền công tác 780k` → refund, `cty`). The type and the null target both contradict it. |
| 41 | baseline-01-855e5c80ccc5 | `cty tra tien cong tac phi 2tr3` | complete, income, `cty` | The company paying back travel expenses (`công tác phí`) is an explicit reimbursement, so refund under resolved decision 3. |
| 34 | baseline-01-743813ab08f1 | `hoan tien shopee 89k` | complete, refund, null | Shopee is named as the refunding party, and the spec example `shopee hoàn 299k` → `shopee`. |
| 74 | baseline-01-9c3bfce2c702 | `hoàn tiền shopee 99k` | complete, refund, null | Same as 34. |
| 80 | baseline-01-dfe6bce7e68a | `hoàn tiền shopee 189k` | complete, refund, null | Same as 34. |
| 101 | baseline-01-2fb11315af24 | `shopee hoàn tiền 89k` | complete, refund, null | This is the spec example's exact form, but the target is null. |
| 65 | baseline-01-d24f56fc52e8 | `tra sua vs Linh 55k` | complete, expense, `Linh` | `vs` introduces a companion, and companions are never targets. |
| 29 | baseline-01-7d34d85d1388 | `lẩu vs mn chia đầu người 175k` | complete, expense, `mn` | Companions introduced by `vs`. |
| 16 | baseline-01-84912823d92c | `lau vs team 20/10 chia moi nguoi 320k` | complete, expense, `lau vs team` | The span is dish + `vs` + companion. It is neither minimal nor a counterparty. |
| 10 | baseline-01-c8dbeabd78a9 | `cơm gà xối mỡ 45K` | complete, expense, `cơm gà xối mỡ` | The dish is the purchased item, not a party. Compare `cơm tấm trưa` → null (91). |
| 15 | baseline-01-1251fb90bba6 | `đổ xăng 80.000đ` | complete, expense, `đổ xăng` | An activity, not a party. The same note at 83 has null. |
| 3 | baseline-01-df2d44a6501e | `tiền nhà t11 4tr2` | complete, expense, `nhà` | `tiền nhà` means rent. The same note at 126 has null. |
| 11 | baseline-01-132c720c3030 | `tra gop xe sh thang 12 3tr1` | complete, repayment_out, `tra gop xe sh` | The span includes the verb and the item. Positions 43 and 57, with the same meaning, are null. |
| 12 | baseline-01-ced10836d15e | `trả góp xe máy tháng 11 2,3tr` | complete, repayment_out, `trả góp xe máy` | Same as 11. |
| 110 | baseline-01-90a34dbd7ef8 | `anh Long mượn 3 triệu sửa xe` | complete, lend, `anh Long` | The prefix should be stripped (`Long`), and the doc gives this exact example (`anh Hùng` → `Hùng`). |
| 127 | baseline-01-0cc1a9946095 | `tra no chi Hoa 1tr` | complete, repayment_out, `chi Hoa` | The prefix should be stripped (`Hoa`). |
| 140 | baseline-01-6efa6394bb11 | `qua sinh nhat bé Na 300k` | complete, expense, `bé Na` | The prefix should be stripped (`Na`). |
| 35 | baseline-01-e00954b39593 | `mua vở bút cho bé 180k` | complete, expense, `bé` | `bé` receives the goods and did not receive the money. The seller is not written, so the target should be null. |
| 122 | baseline-01-3ea4b050d7f7 | `spotify 59k` | complete, expense, null | This conflicts with `spotify family 59k` → `spotify` (5) and with `netflix` → `netflix` (94). |
| 138 | baseline-01-09b29f24dc67 | `home credit ky nay 1.250.000` | complete, expense, null | This conflicts with `góp laptop fe credit` → repayment_out, `fe credit` (125). It is the same kind of consumer-finance installment. |
| 4 | baseline-01-1e11850c97d6 | `ship đồ ăn baemin 67k` | uncertain, expense, null | There is no reason given, and the type is clear. Compare `grabfood tối 89k` → complete, `grabfood` (20). If it is resolved as complete, the target is `baemin`. |

Deliberately **not** flagged, because these follow from open taxonomy questions rather than
annotator error:
- the `còn nợ X` → borrow notes;
- credit-card payoff → expense;
- `mua ccq` / gold → expense;
- `đổi đô` → transfer;
- the remaining merchant-null notes (highlands, circle k, fahasa, long châu). These are resolved
  by the wording clarification below.

## Classification of findings

**Spec working correctly**
- Loans are separated cleanly: lend and borrow are identified correctly whenever `mượn`/`vay`
  appears (except 135).
- Transfers are consistent: 21/21 have a null target, and saving, withdrawing, topping up a
  wallet (`nap shopeepay`) and moving to another own account all resolve as intended.
- `X trả (lại) …` → repayment_in with the implied user works as designed.
- Wedding gifts consistently take the person as the target (`mừng cưới Tuấn` → `Tuấn`).
- Role and kinship targets are used correctly (`khách`, `cty`, `mẹ`, `bố`, `đồng nghiệp`,
  `bạn thân`).
- 0 structural or offset errors.

**Wording clarification needed**
- The doc should state plainly that **a named merchant, service or platform is the target**
  (`highlands`, `circle k`, `spotify`, `shopee hoàn`). The pilot shows annotators treat it as
  optional.
- The doc should state that **items, dishes, activities and bill names are never targets**
  (`cơm gà`, `đổ xăng`, `tiền nhà`, `ccq`), and should add rows for them to the edge-case table.
- The companion rule needs an example using the short form: `vs` = `với`.
- The prefix-stripping rule needs to include `anh`/`chị` in the unaccented form (`chi`) and
  `bé` in the list.
- The doc should separate the beneficiary (`cho bé`, `biếu bà`) from the payee, and the gift
  recipient (`mừng cưới X`) from the beneficiary of a purchase.
- The doc should require a `note` on every `uncertain` record, so that the ambiguity reasons can
  be audited.

**Taxonomy decision needed**
- **Debt-state notes** (`còn nợ X 300k tiền ăn`, `nợ tiền nhà bà chủ`): are they `borrow`,
  `expense`, or `uncertain`? They record an obligation, not received principal.
- **Installments, BNPL and credit-card payoff** (`trả góp`, `fe credit`, `home credit`,
  `spaylater`, `thanh toán dư nợ thẻ`): `repayment_out` or `expense`? The target would be the
  lender or card issuer.
- **Investments** (`mua ccq`, gold): `expense` or `transfer`? Currency exchange (`đổi đô`) →
  transfer.
- **Cashback** (`cashback the tin dung`): `refund` or `income`? It is not a reversal of a
  specific expense.

**Corpus coverage issue**
- `repayment_in` has 4 examples and 0 unaccented; `refund` has 5, 4 of them the shopee
  template. Neither is enough for a stratified split. Plan targeted generation or a Quet pass
  aimed at these types.
- The ambiguous templates the spec is built around (`X gửi 500k`, `cho X 500k`, `trả X 500k`,
  `X ck hộ Y`) do not appear in the pilot, so `uncertain` is barely exercised.
- There are 0 skipped notes, so the ability to reject non-finance notes is untested.

**Probable human labeling mistake**
- Direction reversals: 135 and 136.
- Employer reimbursement labelled as income: 41 and 119.
- Refund with a null target despite a named platform: 34, 74, 80, 101.
- Companion targets: 16, 29, 65.
- Item or activity targets: 3, 10, 11, 12, 15.
- Unstripped prefixes: 110, 127, 140.
- Beneficiary as target: 35.
- Conflicts with near-identical notes: 122 vs 5, 138 vs 125, and 4 (uncertain without a
  reason).

## Frozen-v1 re-check candidates

Second pass, after `recheck-pilot-01` was completed and annotation-v1 was frozen. Every current
label (140) was compared against the frozen rules; only records whose **current** label conflicts
with a frozen rule are listed. Built into `datasets/annotation-v1/recheck-frozen-v1-02.jsonl`.

Checked and excluded: 85 `cashback the tin dung` is already `refund` with a null target (issuer
not named); the other installment/BNPL notes (11, 12, 43, 57, 125, 136, 138) are already
`repayment_out`; the remaining `recheck-pilot-01` records now satisfy the frozen rules except 16,
41 and 119, which are listed again.

| pos | id | text | current | category / reason |
|---|---|---|---|---|
| 2 | baseline-01-efb9ecbae1cf | `con no Nam 300k tien an` | complete, borrow, `Nam` | debt state: obligation only, no principal received → skipped |
| 9 | baseline-01-7b7ee93c03a7 | `mua ccq quỹ đầu tư 2tr` | complete, expense, `ccq quỹ đầu tư` | investment → transfer; the span is the asset, and transfers require a null target |
| 16 | baseline-01-84912823d92c | `lau vs team 20/10 chia moi nguoi 320k` | complete, expense, `team` | companion: `vs` introduces a companion, never a target (still wrong after pass 01) |
| 23 | baseline-01-9db03b1ecb83 | `cf highlands 59K` | complete, expense, null | named merchant left null; the doc's own example is `highlands` |
| 26 | baseline-01-b8ee455c160a | `circle k mì ly vs nước 42k` | complete, expense, null | named merchant left null (`circle k`) |
| 27 | baseline-01-8c7ad02a53e9 | `con no a Long 300k` | complete, borrow, `Long` | debt state → skipped |
| 41 | baseline-01-855e5c80ccc5 | `cty tra tien cong tac phi 2tr3` | complete, income, null | reimbursement: the frozen doc lists this form as refund with target `cty` (still wrong after pass 01) |
| 52 | baseline-01-aa11c83a0041 | `grab về nhà khuya 112k` | complete, expense, null | named service left null (`grab`; doc example `đi grab …` → `grab`) |
| 62 | baseline-01-3ba213024208 | `mua sách fahasa 215k` | complete, expense, null | named merchant left null (`fahasa`) |
| 69 | baseline-01-f28ba4be2239 | `mua thuốc cảm nhà thuốc long châu 126k` | complete, expense, null | named merchant left null (`long châu` / `nhà thuốc long châu`) |
| 77 | baseline-01-240955c6a910 | `còn nợ chị Mai 700k` | complete, borrow, `Mai` | debt state → skipped |
| 84 | baseline-01-dcdd29b8128f | `circle k mì ly + nước 43k` | complete, expense, null | named merchant left null (`circle k`) |
| 90 | baseline-01-a4aa852db61d | `thanh toán dư nợ thẻ vcb 2tr` | complete, expense, null | card liability → repayment_out, issuer `vcb` named |
| 93 | baseline-01-0f13fd308b63 | `mua vàng nhẫn pnj nửa chỉ 3tr95` | complete, expense, null | quantified precious asset (`nửa chỉ`), no gift/use stated → complete, transfer, target null under the vi-VN heuristic |
| 96 | baseline-01-1b0a0829f88e | `circle k nước + snack 47.000` | complete, expense, null | named merchant left null (`circle k`) |
| 102 | baseline-01-caafbd67cb0a | `tra no tien dien may hom truoc 700k` | complete, expense, null | liability: paying off a debt for an appliance bought earlier → repayment_out; paying a liability is never another expense |
| 103 | baseline-01-8b1e2e4839da | `còn nợ Nam 200k tiền vé` | complete, borrow, `Nam` | debt state → skipped |
| 106 | baseline-01-6c63ed69250f | `thanh toan du no the vpbank 6tr3` | complete, expense, null | card liability → repayment_out, issuer `vpbank` named |
| 107 | baseline-01-0cb124371317 | `highlands 69K` | complete, expense, null | named merchant left null (`highlands`) |
| 111 | baseline-01-43b46e772c2b | `no tien nha ba chu 1 thang` | complete, borrow, `ba chu` | debt state (rent arrears) → skipped |
| 119 | baseline-01-5c806d85f402 | `cty hoan tien cong tac 1,2tr` | complete, refund, null | the type was fixed in pass 01, but `cty` is named; the doc example gives target `cty` |
| 124 | baseline-01-790ca645bdb7 | `con no Duc 300k chua tra` | complete, borrow, `Duc` | debt state → skipped |
| 134 | baseline-01-2561a90bd911 | `thuoc cam 72k nha thuoc long chau` | complete, expense, null | named merchant left null (`long chau`) |
| 137 | baseline-01-6e0df0256a81 | `mua ccq 2 trieu` | complete, expense, null | investment → transfer (frozen doc example `mua ccq 5tr`) |
