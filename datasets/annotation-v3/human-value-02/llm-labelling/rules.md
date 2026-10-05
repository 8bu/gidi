# Annotation rules (the only source of truth)


## Part 1: annotation-v1 (transaction types and target spans)

# Annotation v1

The first supervised label contract for Gidi. It covers exactly two things:

1. **transaction type** — one of the canonical types below;
2. **target** — the explicit external counterparty span, or `null`.

Out of scope for v1: tags, amounts (amount parsing stays deterministic, outside the model),
dates, and train/validation/test splits (made after labeling, stratified by type and by
accented/unaccented text).

Machine-readable contract: [`configs/annotation-v1.yaml`](../configs/annotation-v1.yaml). The
validator reads the taxonomy from that file; this document explains it. Changing either is a new
annotation version.

## Record format

One JSON object per line, one per queued corpus record:

```json
{"id": "baseline-01-efb9ecbae1cf", "annotation_status": "complete", "type": "lend", "target": {"text": "Nam", "start": 4, "end": 7}}
```

| field | required | value |
|---|---|---|
| `id` | yes | id of an entry in `datasets/annotation-v1/queue.jsonl` (the original corpus id) |
| `annotation_status` | yes | `complete`, `uncertain`, or `skipped` |
| `type` | yes (must be set when `complete`, must be `null` when `skipped`) | a canonical type |
| `target` | yes (must be `null` when `skipped`) | `{"text", "start", "end"}` or `null` |
| `note` | required when `uncertain`, otherwise optional | free-text annotator comment; for `uncertain` a non-empty reason (validator-enforced) |

No other keys are allowed. Annotations never copy or edit the corpus text; they reference it by id.

## Transaction types

| type | definition | example |
|---|---|---|
| `expense` | Money leaves the user with no expectation of principal repayment. | `đóng tiền điện 500k` |
| `income` | Money received as salary, freelance/client payment, gift, sale proceeds, rewards, or other ordinary incoming money that is neither borrowing, debt repayment, nor refund. | `lương 25tr` |
| `borrow` | The user receives loan principal and now owes another party. | `mượn mẹ 2tr` |
| `lend` | The user gives loan principal to another party and expects repayment. | `cho Nam vay 2tr` |
| `repayment_in` | Another party repays money they previously owed the user. | `Nam trả nợ tao 500k` |
| `repayment_out` | The user repays money they previously owed another party, including payments on an existing financed balance or liability (installments, BNPL, credit-card balance). | `trả Nam 500k tiền mượn`, `trả spaylater 560k` |
| `transfer` | Movement between the user's own accounts, wallets, cash holdings, or savings, including conversion into investment/store-of-value assets the user still owns. No economic income or expense occurs. | `ck 5tr qua tk tiết kiệm`, `rút ATM 2tr`, `mua ccq 5tr` |
| `refund` | Money returned to the user as reversal, reimbursement, rebate, or spending-linked cashback of an earlier expense. | `shopee hoàn 299k`, `cashback thẻ tín dụng 120k` |

There is no `other` type. If the text does not establish which type applies, do not guess:
set `annotation_status: "uncertain"`. Example: `Nam gửi tao 500k` is uncertain — it could be
income, repayment, a loan, or a refund — unless the note itself says why the money was sent.

A valid finance note that is not a money-movement event (see [Debt-state notes](#debt-state-notes))
is outside this taxonomy: set `annotation_status: "skipped"`, not a forced type.

Judge only from the note text. Do not use outside knowledge of the user, and do not let the
generation prompt's categories (`spending`, `debt`, `savings`, …) stand in for these types.

## Target span

`target` is the **minimal explicit expression that identifies the external financial
counterparty**: the party money flows from or to (person, merchant, platform, employer, client,
bank-as-lender). A person mentioned in the note is **not** automatically the counterparty.

| note | target |
|---|---|
| `cho Nam vay 500k` | `Nam` |
| `Nam trả tao 500k` | `Nam` |
| `ăn ở Pizza 4P 300k` | `Pizza 4P` |
| `ăn với Nam ở Pizza 4P 300k` | `Pizza 4P` — Nam is a companion, not the counterparty. |
| `đi grab với Linh 100k` | `grab` — Grab is the service being paid; Linh is a companion. |
| `ăn trưa 80k` | `null` |
| `ck 5tr qua tk tiết kiệm` | `null` (transfer) |

Rules:

- **Never infer.** The target must literally occur in the text. If the counterparty is implied
  but not written, `target` is `null`.
- **Transfers have no target.** For `type: "transfer"` the validator requires `target: null`,
  even when an account, wallet, or bank is named.
- **Kinship/title prefix before a proper name: exclude the prefix.** `chị Linh gửi 500k` →
  `Linh`; `anh Hùng trả nợ` → `Hùng`; `thg Nam mượn 2tr` → `Nam`; `bé Na` → `Na`. Same for
  `anh`/`a`, `chị`/`chi`/`c`, `em`/`e`, `bé`, `bác`, `cô`, `chú`, `cậu`/`cau`, `thằng`/`thg`,
  `con`, `bạn`, … (accented or not) when a proper name follows.
- **Kinship/role term as the only identifier: keep it.** `mẹ gửi 2tr` → `mẹ`; `ba cho 500k` →
  `ba`; `vợ chuyển 3tr` → `vợ`. Role nouns follow the same rule: `khách trả 5tr` → `khách`;
  `cty hoàn 780k` → `cty`; `trả chủ nhà 6tr` → `chủ nhà`; `sếp gửi tiền ăn 500k` → `sếp`.
  Do not blindly strip kinship or role terms: strip only a prefix that sits in front of a
  proper name.
- **Generic counterparty nouns.** A generic person, business, or organisation noun is the target
  when it is the only explicit expression identifying the external counterparty: `nhà thuốc hoàn
  120k` → `nhà thuốc`; `cửa hàng trả lại 300k` → `cửa hàng`; `kế toán chuyển lương 20tr` →
  `kế toán`; `phòng nhân sự hoàn công tác phí` → `phòng nhân sự`; `ngân hàng hoàn phí 50k` →
  `ngân hàng`. When the same noun only names an item, category, fee, or activity, it is not a
  target: `tiền thuốc 120k` → `null`; `phí ngân hàng 20k` → `null` (the bank is named only as
  the kind of fee, not as the party paid).
- **Title before a birth-order name: keep the whole phrase.** When the word after the kinship
  title is a birth-order word (`Hai`, `Ba`, `Tư`, `Năm`, `Sáu`, `Bảy`, `Tám`, `Chín`, `Mười`,
  `Út`), the phrase is the identifier, whatever the casing: `vay chú Ba 10tr` → `chú Ba`;
  `cho dì Sáu mượn` → `dì Sáu`; `cho anh hai muon 1tr5` → `anh hai`; `cho cậu út vay` →
  `cậu út`. An unaccented form that also reads as an ordinary given name follows the
  proper-name rule: `cho cau Tu muon 3tr` → `Tu` (Tú).
- **Payment channels are not targets.** A payment app, wallet, or bank mentioned only as the
  channel is not the counterparty: `ck momo cho Nam 200k` → `Nam`; `trả KFC qua momo 200k` →
  `KFC`; `chuyển cho Linh qua VCB` → `Linh`. The platform is the target only when it is itself
  the counterparty: `momo hoàn 50k` → `momo`. Moving money into or out of the user's own
  wallet/account is a transfer with a null target: `nạp momo 200k` → `transfer`, `null`.
- **Named merchants, services, and platforms are targets** when they are the external
  counterparty: `cf highlands 59k` → `highlands`; `circle k mì ly 42k` → `circle k`;
  `spotify 59k` → `spotify`; `shopee hoàn tiền 89k` → `shopee`; `home credit kỳ này` →
  `home credit`. Do not leave a named counterparty `null`.
- **Marketplaces are targets only as the counterparty.** A marketplace is the target when the
  note makes it the seller, the buyer/sale venue, or the party paying the user: `mua giày ở
  shopee` → `shopee`; `shopee hoàn tiền` → `shopee`; `bán đồ cũ chợ tốt 600k` → `chợ tốt`;
  `bán quần áo cũ trên shopee` → `shopee`. A social network or chat app where the user only
  found the buyer or seller is not: `bán quần áo cũ trên fb 400k` → `null`.
- **Ride-hailing and delivery: the service name only.** Keep the app or service as written and
  leave out the trip or order words: `be bike ve nha 28k` → `be bike`; `bee đi học 18k` →
  `bee`; `grab về nhà 112k` → `grab`; `xanh sm đi họp` → `xanh sm`; `grabfood bún đậu` →
  `grabfood`.
- **Items, dishes, activities, and bill names are never targets.** `cơm gà xối mỡ 45k`,
  `đổ xăng 80k`, `tiền nhà 4tr`, `tiền điện`, `trả góp xe máy`, `mua ccq` → `null` unless a
  party is also named.
- **Minimal span, as typed.** Keep the user's spelling, casing, and typos (`shope`, `Vcb`);
  exclude surrounding words such as `với`, `ở`, `cho`, `của`, `tiền`, and verbs (`trả`, `góp`).
- **Companions are not targets.** People introduced by `với`, its shorthand `vs`, `cùng`, or
  otherwise shown sharing the spend are companions; the merchant/service is the target, or
  `null` if none is named: `tra sua vs Linh 55k` → `null`; `lẩu vs mn 175k` → `null`.
- **Beneficiary is not payee.** The person a purchase is *for* is not the counterparty:
  `mua vở bút cho bé 180k` → `null`; `mua hoa quả biếu bà 200k` → `null`; `20/10 mua hoa tặng
  mẹ` → `null`. When money or a gift itself goes to a person, that person is the counterparty:
  wedding and ceremony money (`mừng cưới Tuấn 500k`, `đám cưới anh Phong mừng 500k`, `mừng thọ
  bà nội`) → the recipient (`Tuấn`, `Phong`, `bà nội`); `lì xì cháu 200k` → `cháu`; `quà sinh
  nhật bé Na 300k` → `Na`; `quà 20/10 cho mẹ` → `mẹ`. Ceremony money with no recipient named
  (`tiền mừng cưới 500k`) → `null`.
- **One target.** If several entities are plausible counterparties and the note does not
  establish which one is on the other side of the transaction, set
  `annotation_status: "uncertain"`. Do not guess.
- An empty or whitespace-padded span is invalid.

Offsets:

- The NFC-normalized corpus text (as preserved in the queue) is authoritative.
- `start`/`end` are **Unicode code-point** offsets (Python `str` indices), start inclusive, end
  exclusive.
- `text[start:end]` must equal `target.text` exactly; the validator enforces this.

## User perspective

Notes are written by the user about the user's own money. When the grammar leaves the
recipient or owner unstated, the user may be taken as that party if it is the natural reading;
`tao`, `mình`, or `cho tôi` are not required. `chị Linh trả nợ 1tr` → `complete`,
`repayment_in`, `Linh`.

Do not over-apply this. If the direction itself is still ambiguous, the note is `uncertain`:
`trả Nam 500k` could be paying Nam for something or paying back a debt.

**`repayment_in` needs evidence of a prior debt or obligation.** A party paying the user is
`repayment_in` only when the note shows that the money was owed: `trả nợ`, `trả lại`, `gửi
lại`, `trả tiền mượn`, `hoàn lại khoản đã ứng`, money the user fronted (`tiền điện mình ứng
hộ`, `vé mình mua hộ`), or other clear debt context. A bare `X trả …` does not establish it;
it could be debt repayment, payment for goods or services, or a reimbursement.

| note | status | type | target |
|---|---|---|---|
| `Nam trả nợ 500k` | complete | `repayment_in` | `Nam` |
| `Nam trả lại 500k tiền hôm trước` | complete | `repayment_in` | `Nam` |
| `chị Linh trả tiền mượn` | complete | `repayment_in` | `Linh` |
| `Nam trả 500k` | uncertain | — | (`Nam` may be recorded) |
| `khách trả 5tr tiền web` | complete | `income` | `khách` |
| `khách trả invoice tháng 9` | complete | `income` | `khách` |

## Refunds, reimbursements, and cashback

`refund` covers purchase refunds, reimbursements, rebates, and cashback tied to spending,
whoever pays it: merchant, platform, card issuer, employer, or another party. Ordinary employer
pay and rewards unrelated to prior spending stay `income`.

| note | status | type | target |
|---|---|---|---|
| `cty hoàn tiền công tác 780k` | complete | `refund` | `cty` |
| `cty trả tiền công tác phí 2tr3` | complete | `refund` | `cty` |
| `cashback thẻ tín dụng 120k` | complete | `refund` | `null` (issuer not named) |
| `momo cashback 50k` | complete | `refund` | `momo` |
| `cty trả lương 20tr` | complete | `income` | `cty` |
| `cty thưởng 2tr` | complete | `income` | `cty` |
| `cty gửi 780k` | uncertain | — | (`cty` may be recorded) |
| `hoan tien bao hiem 670k` | complete | `refund` | `null` |
| `bảo hiểm chi trả viện phí 3tr` | complete | `refund` | `null` (`bảo hiểm` is not a named insurer) |
| `hoàn thuế tncn 2tr4` | complete | `refund` | `null` |
| `dong bao hiem nhan tho 1tr2` | complete | `expense` | `null` (a premium is spending) |

Insurance payouts and claim reimbursements, and tax refunds, return money tied to the user's
own earlier spending or tax paid, so they are `refund`. Paying an insurance premium is
`expense`.

## Financed balances: installments, BNPL, credit cards

Paying an existing financed balance or liability is `repayment_out`, never another `expense`.
The target is the lender, provider, or card issuer if explicitly named, otherwise `null`. A note
describing the original purchase itself is classified by its economic purpose (usually `expense`).

| note | type | target |
|---|---|---|
| `trả spaylater 560k` | `repayment_out` | `spaylater` |
| `home credit kỳ này 1tr250` | `repayment_out` | `home credit` |
| `thanh toán dư nợ thẻ 4tr` | `repayment_out` | `null` |
| `thanh toán dư nợ thẻ vpbank 6tr3` | `repayment_out` | `vpbank` |
| `trả góp xe tháng này 2tr3` | `repayment_out` | `null` |
| `góp laptop fe credit 1.450.000` | `repayment_out` | `fe credit` |

## Salary advances

An advance on the user's pay is money the user must pay back out of future salary, so receiving
it is `borrow` and paying it back is `repayment_out`. The employer or boss is the target when
written.

| note | type | target |
|---|---|---|
| `ứng trước của sếp 2tr` | `borrow` | `sếp` |
| `ứng trước lương cty 2tr` | `borrow` | `cty` |
| `trả tiền ứng lương cty 1tr5` | `repayment_out` | `cty` |

Advancing money *to* someone else (`ứng cho Nam 500k`) is `lend`.

## Investment and store-of-value purchases

Converting the user's money into another asset the user still owns is `transfer` (target
`null`). Buying the same kind of asset as goods to consume or give is `expense`; the wording
decides.

| note | type | target |
|---|---|---|
| `mua ccq 5tr` | `transfer` | `null` |
| `mua vàng tích trữ 2 chỉ` | `transfer` | `null` |
| `đổi 10tr sang usd` | `transfer` | `null` |
| `mua nhẫn vàng tặng vợ` | `expense` | `null` (beneficiary, not payee) |
| `mua dây chuyền đeo` | `expense` | `null` |
| `gửi góp tháng 3 triệu` | `transfer` | `null` (periodic savings deposit) |
| `dong hui thang nay 2 trieu` | `expense` | `hui` (a rotating-credit contribution paid into the hụi) |

### Refundable deposits

A refundable deposit stays the user's asset (a receivable). Getting it back converts that asset
into cash, so a returned deposit is `transfer` with a `null` target, whoever returns it.
Paying a deposit is `transfer` only when the wording clearly establishes that it remains
refundable; a non-refundable booking or service fee is not a deposit in this sense and keeps its
ordinary type (paying it is `expense`, getting it back is `refund`). If the note does not say
whether a deposit payment is refundable or a booking fee, it is `uncertain`.

| note | status | type | target |
|---|---|---|---|
| `hoàn cọc nhà 5tr` | complete | `transfer` | `null` |
| `chủ nhà trả cọc 5tr` | complete | `transfer` | `null` |
| `lấy lại tiền cọc 2tr` | complete | `transfer` | `null` |
| `đặt cọc thuê nhà 10tr, trả phòng được hoàn` | complete | `transfer` | `null` |
| `hoàn tiền đặt bàn 200k` | complete | `refund` | `null` (a booking payment, not a deposit) |

### vi-VN heuristic: quantified precious assets

This is an annotation heuristic for Vietnamese notes, not a universal economic rule. In vi-VN,
gold and gems bought by weight or purity are commonly held as savings, so the wording that
matters is the **quantity/purity**, not the item word.

**Quantified precious assets → `transfer`, target `null`.** A precious-metal or gemstone purchase
described with commodity-like quantity or purity is a store-of-value acquisition unless the note
states a consumption or gift purpose. Strong signals: `chỉ`, `lượng`, `phân`, `gram`/`g`,
`carat`, `9999`, `24K`, and other weight/purity descriptions. Words such as `nhẫn`, `vòng`,
`dây chuyền`, or `kim cương` do **not** by themselves make the purchase an `expense`. As with
every transfer, a named shop (`PNJ`, `SJC`) is not recorded as a target.

**Consumption intent overrides.** If the note explicitly says the item is for personal use,
a gift, a ceremony, fashion, or another consumptive purpose, it is `expense`. When both signals
appear, the explicit intent wins.

| note | type | target | why |
|---|---|---|---|
| `mua vàng 1 chỉ` | `transfer` | `null` | quantity (`chỉ`) |
| `mua nhẫn PNJ nửa chỉ` | `transfer` | `null` | quantity; `nhẫn` alone is not consumption |
| `mua nhẫn kim cương 5 carat` | `transfer` | `null` | quantity (`carat`) |
| `mua 2 lượng vàng SJC` | `transfer` | `null` | quantity (`lượng`) |
| `mua nhẫn cưới 12tr` | `expense` | `null` | ceremony (`cưới`) |
| `mua dây chuyền tặng vợ` | `expense` | `null` | gift (`tặng`) |
| `mua bông tai đi tiệc` | `expense` | `null` | personal use / fashion |
| `mua vòng tay cho mẹ` | `expense` | `null` | gift for mẹ (beneficiary) |
| `mua nhẫn vàng 1 chỉ tặng vợ` | `expense` | `null` | gift intent overrides quantity |

An unquantified purchase with no stated purpose (`mua nhẫn vàng 3tr`) is not covered by this
heuristic; judge it from the rest of the note, and use `uncertain` if the purpose is unclear.

## Debt-state notes

A note that only records an outstanding obligation, without describing money moving, is outside
the transaction taxonomy: `annotation_status: "skipped"`, `type: null`, `target: null`.

| note | status | type |
|---|---|---|
| `còn nợ Hùng 300k tiền ăn` | skipped | — |
| `nợ tiền nhà bà chủ 1 tháng` | skipped | — |
| `còn thiếu Nam 500k` | skipped | — |
| `mượn Hùng 500k` | complete | `borrow` (principal received) |
| `vay mẹ 2tr` | complete | `borrow` |

`borrow` requires the note to describe receiving borrowed principal.

## Edge cases

Resolved from the note text alone. "uncertain" rows must not be forced into a type.

| # | case | note | status | type | target | why |
|---|---|---|---|---|---|---|
| 1 | income vs repayment_in | `Nam gửi tao 500k` | uncertain | — | (`Nam` may be recorded) | could be gift, loan, repayment, or refund; the note does not say why |
| 2 | income vs repayment_in | `Nam trả tao 500k tiền hôm trước` | complete | `repayment_in` | `Nam` | `trả … tiền hôm trước` = paying back money owed from earlier |
| 3 | income vs borrow | `mẹ cho 2tr` | complete | `income` | `mẹ` | `cho` without `mượn`/`vay` is a gift |
| 4 | income vs borrow | `mẹ cho mượn 2tr` | complete | `borrow` | `mẹ` | `cho mượn` = mẹ lends to the user |
| 5 | expense vs repayment_out | `trả Nam 500k` | uncertain | — | (`Nam` may be recorded) | paying Nam for something vs paying back a debt; the note does not say |
| 6 | expense vs repayment_out | `trả Nam 500k tiền mượn` | complete | `repayment_out` | `Nam` | `tiền mượn` establishes a prior debt |
| 7 | refund vs income | `shopee hoàn 299k` | complete | `refund` | `shopee` | `hoàn` reverses an earlier purchase |
| 8 | refund vs income | `khách chuyển 5tr tiền web` | complete | `income` | `khách` | client payment for work; `khách` is the only identifier, so it is kept |
| 9 | lend vs ordinary payment | `cho Nam 500k` | uncertain | — | (`Nam` may be recorded) | a gift (expense) or a loan (lend); no `vay`/`mượn` |
| 10 | lend vs ordinary payment | `cho Nam vay 500k` | complete | `lend` | `Nam` | `vay` establishes a loan |
| 11 | person mentioned, merchant is target | `ăn với Hùng ở KFC 200k` | complete | `expense` | `KFC` | Hùng is a companion, not the counterparty |
| 12 | kinship-only target | `mẹ cho 2tr` | complete | `income` | `mẹ` | kinship word is the only identifier |
| 13 | kinship prefix + name | `chị Linh trả nợ 1tr` | complete | `repayment_in` | `Linh` | Linh is the subject repaying; `chị` is a prefix |
| 14 | multiple possible counterparties | `Nam ck hộ Linh 300k` | uncertain | — | — | Nam paid on Linh's behalf: who owes whom, and in which direction, is unclear |
| 15 | missing target, clear type | `mua thuốc 120k` | complete | `expense` | `null` | no counterparty written; never infer a pharmacy |
| 16 | missing target, clear type (transfer) | `ck qua tài khoản tiết kiệm 5tr` | complete | `transfer` | `null` | own savings account; transfers always have a null target |
| 17 | clear target, ambiguous type | `Nam gửi tao 500k`, `trả Nam 500k` | uncertain | — | (`Nam` may be recorded) | the counterparty is clear, the type is not; see rows 1 and 5 |
| 18 | debt state, no movement | `còn nợ Hùng 300k tiền ăn` | skipped | — | — | records an obligation, not money moving; outside the taxonomy |
| 19 | BNPL payment | `trả spaylater 560k` | complete | `repayment_out` | `spaylater` | paying a financed balance, never `repayment_in` or `expense` |
| 20 | investment purchase | `mua ccq 5tr` | complete | `transfer` | `null` | money becomes an asset the user still owns |
| 21 | consumption vs store of value | `mua nhẫn vàng tặng vợ` | complete | `expense` | `null` | a gift, consumed as goods; `vợ` is a beneficiary |
| 22 | companion via `vs` | `tra sua vs Linh 55k` | complete | `expense` | `null` | `vs` = `với`; Linh is a companion, no merchant named |
| 23 | item is not a target | `cơm gà xối mỡ 45k` | complete | `expense` | `null` | a dish, not a party |
| 24 | named merchant | `cf highlands 59k` | complete | `expense` | `highlands` | a named merchant is the counterparty |
| 25 | refundable deposit returned | `chủ nhà trả cọc 5tr` | complete | `transfer` | `null` | the deposit was the user's asset; nothing is earned or refunded |
| 26 | generic counterparty noun | `nhà thuốc hoàn 120k` | complete | `refund` | `nhà thuốc` | the only expression naming the party that pays |
| 27 | bare `X trả` | `Nam trả 500k` | uncertain | — | (`Nam` may be recorded) | no debt context: repayment, payment for goods/services, or reimbursement |

For uncertain rows, recording an obvious target is allowed and is validated like any other span,
but the record still never enters training. Every uncertain record must carry a non-empty `note`
stating the ambiguity (e.g. `"direction unclear: paying Nam vs repaying Nam"`).

## Annotation status

| status | meaning | training |
|---|---|---|
| `complete` | type confidently determined; target set or confidently `null` | eligible |
| `uncertain` | type or target not determinable from the text; a non-empty `note` must say why | excluded |
| `skipped` | not a usable finance note, a valid finance note outside the transaction-event taxonomy (debt-state notes), or annotator chose not to label; `type` and `target` must be `null` | excluded |

Never force a label. `uncertain` records may carry `type`/`target` as `null`; any value they do
carry is still validated. `skipped` records must have both `type: null` and `target: null`.
Neither status enters the initial training split. The validator rejects an `uncertain` record
without a non-empty `note`, and a `skipped` record that keeps a type or target.

## Queue

`datasets/annotation-v1/queue.jsonl` holds 600 approved records of
`corpus/reviewed/baseline-01.jsonl`. Each entry is `{"id", "text", "corpus", "position"}` with the
corpus text preserved verbatim. Order is `sha256("annotation-v1:<id>")`, so it is deterministic
and independent of file order; annotate in `position` order.

```bash
uv run python scripts/build_annotation_queue.py          # reproduce the queue (refuses to overwrite)
uv run python scripts/validate_annotations.py labels.jsonl   # validate against the queue texts
```

No labels are generated automatically.


## Part 2: annotation-v2 (value span convention, combined pass)

# Annotation v2

annotation-v2 = [annotation-v1](annotation-v1.md) (transaction type and target span, semantics
**unchanged**) + one new output: the **value span**, the exact substring of the note that is the
monetary amount.

| note | value span |
|---|---|
| `cơm tấm 100` | `100` |
| `mượn chú hai 5 xị` | `5 xị` |
| `ăn 2 tô phở 70` | `70` (not `2`) |
| `tiền điện tháng 10 hết 612` | `612` |
| `trả góp kỳ 3 1tr5` | `1tr5` |
| `20/10 mua quà mẹ 500` | `500` |
| `mua 3 vé 150k` | `150k` |
| `đóng kỳ 2 khoản vay 3tr` | `3tr` |

**This is span selection only. Nothing normalizes the span into a number.** `5 xị` is not turned
into 500000, `1tr5` is not turned into 1500000, and no numeric value is stored anywhere (labels,
training records, model output). Turning a span into a number stays deterministic parsing outside
the model and outside this contract.

Machine-readable contract: [`configs/annotation-v2.yaml`](../configs/annotation-v2.yaml)
(type/target sections copied verbatim from v1, plus a `value:` section). Quet schemas: the
combined pass (type + target + value, Quet multi-span)
[`configs/annotation-v2.quet.yaml`](../configs/annotation-v2.quet.yaml) and the value-only pass
[`configs/annotation-v2-value.quet.yaml`](../configs/annotation-v2-value.quet.yaml).
Validators and the rule proposer: `src/gidi/annotation/value_span.py`,
`src/gidi/annotation/combined.py`.

## The span convention

The value span is the **minimal full amount expression**: the number plus the unit, slang or
currency marker attached to it.

Include:

| form | examples |
|---|---|
| `k` / `K` (optionally spaced) | `50k`, `45K`, `50 K` |
| `tr` / `triệu`, compact and decimal | `1tr`, `1tr5`, `1tr150`, `4tr664`, `1.5tr`, `1,5tr`, `2 triệu` |
| `nghìn` / `ngàn`, `tỷ` | `500 nghìn`, `500 ngàn`, `2 tỷ` |
| slang | `2 củ`, `5 xị`, `5 chai`, `5 lít`, `2 triệu rưỡi` |
| separators and long digits | `1.500.000`, `612,000`, `745.000`, `1500000` |
| a currency marker that directly follows | `80.000đ`, `80.000 đồng`, `200 vnd`, `200 vnđ` |
| a bare number | `100`, `500` |
| spelled-out amounts | `bốn triệu`, `hai trăm nghìn` |

Exclude (these are never part of the span, and a number that is only one of these is not an
amount):

| excluded | examples |
|---|---|
| per-unit qualifier | `/tháng` in `50k/tháng` (the span is `50k`) |
| approximator | `tầm`, `khoảng`, `~` (`tầm 50k` → `50k`) |
| quantity | `2 tô`, `3 vé`, `2 ly`, `10kg`, `2 chỉ` of gold, `4 người` |
| month / period / ordinal | `tháng 10`, `t11`, `kỳ 3`, `quý 3`, `lần 2`, `đợt 1`, `học kỳ 2`, `giải 8` |
| date or time | `20/10`, `8h`, `8:30`, `mùng 1`, `thứ 6` |
| percentage | `2%`, `10%` |
| identifiers and model numbers | `stk 123456789`, `iphone 15`, `4g` |

If the note states several money amounts and **no single one is the transaction amount**
(`grabfood tối 89k ship 15k`), the status is `uncertain` with a note; never pick one. A note that
states no amount (`thưởng tết 2 tháng lương`) is `no_amount`.

Span mechanics are the same as v1 targets: Python code-point offsets into the NFC note,
`text[start:end] == span.text`, no leading or trailing whitespace.

## Combined pass (Quet multi-span)

Quet 0.2.8 can label several named span fields in one pass. The combined schema
[`configs/annotation-v2.quet.yaml`](../configs/annotation-v2.quet.yaml) has the eight v1 types, the
three statuses (`uncertain` may concern the type, the target or the value; it needs a note) and two
spans in UI order: `target` (the v1 counterparty rule, null for `transfer`) and `value` (the span
convention above, null when the note states no amount, with its own `complete` / `uncertain`
status). Tab switches the active span, `n` nulls it, `c` cycles the value status. A label line:

```json
{"id": "baseline-01-…", "annotation_status": "complete", "type": "expense",
 "target": {"text": "highlands", "start": 8, "end": 17},
 "value": {"text": "45k", "start": 18, "end": 21},
 "span_status": {"value": "complete"}, "note": "optional"}
```

`skipped` saves a null type, target and value and no `span_status`. Gidi validates a file with
`gidi.annotation.combined` (type and target through the v1 validator, the value through the same
span mechanics as above; `span_status.value: uncertain` needs a `note`; unknown fields are
errors); the script picks it when `--config` is a multi-span schema:

```sh
uv run python scripts/validate_annotations.py LABELS.jsonl \
  --config configs/annotation-v2.quet.yaml --queue QUEUE.jsonl
```


## Part 3: annotation-v3 (debt-only notes; overrides parts 1 and 2 where they differ)

# Annotation v3

annotation-v3 = [annotation-v2](annotation-v2.md) (type, target span, value span) with **one**
taxonomy change: a **debt-only note** is `borrow` or `lend`, not `skipped`.

A debt-only note states an existing debt and no money moves now (`còn nợ Hùng 300k`,
`Hoa còn nợ mình 350k`, `Phúc chưa trả 1tr5`).

## What changed vs v2

| | annotation-v1 / v2 | annotation-v3 |
|---|---|---|
| debt-only, the user owes | `skipped` | `borrow` |
| debt-only, the other party owes the user | `skipped` | `lend` |
| debt-only, direction unclear | `skipped` | `uncertain` + `note` |

Everything else is unchanged: other types, the target rules, the value span convention, null
rules and trainable statuses. The new machine-readable files are
[`configs/annotation-v3.yaml`](../configs/annotation-v3.yaml) and the combined Quet schema
[`configs/annotation-v3.quet.yaml`](../configs/annotation-v3.quet.yaml).

## Why

Both sides of a debt note carry the information the user wants: who is the counterparty, which
way the money is owed and how much. `skipped` threw all of it away and left the model with a
23-note hole in a class (`skipped`) that is otherwise "not a finance note". A debt note is the
same relationship as `borrow` / `lend`, only without the movement of money in the note itself.

## Label rules

* **Type.** The user owes -> `borrow`. The other party owes the user -> `lend`.
* **Target.** The counterparty by the usual v1 rules: a kinship/title prefix before a proper name
  is dropped (`còn nợ chị Mai 700k` -> `Mai`), a role noun that is the only name is kept
  (`nợ chủ trọ tiền điện 280k` -> `chủ trọ`, `no tien nha ba chu` -> `ba chu`), birth-order
  names stay whole (`chú Tám`, `dì Út`). Bill names are not targets: `tiền điện`, `tiền vé`,
  `tiền ăn` -> `null` unless a person or business is named too.
* **Value.** The usual value span; `null` when the note has no amount.
* **Who is the subject.** `nợ X`, `còn nợ X`, `chưa trả X`, `hứa trả X` with an implied subject
  mean the **user** owes X. `X nợ mình`, `X còn thiếu`, `X chưa trả`, `X hứa trả` mean X owes the
  user.
* **Unclear direction -> `uncertain` with a note** (never forced): `còn nợ 500k` (no
  counterparty, no direction), `tiền nợ Lan 1tr5`, `nợ nhau với Bảo 150k`, `Hương 700k chưa trả`,
  `công nợ Minh 2tr`.

## Examples

| note | status | type | target | value |
|---|---|---|---|---|
| `còn nợ Hùng 300k` | complete | `borrow` | `Hùng` | `300k` |
| `nợ chủ trọ tiền điện 280k` | complete | `borrow` | `chủ trọ` | `280k` |
| `chưa trả anh Việt 600k tiền cafe` | complete | `borrow` | `Việt` | `600k` |
| `còn nợ Phương tiền cafe` | complete | `borrow` | `Phương` | null |
| `Hoa còn nợ mình 350k` | complete | `lend` | `Hoa` | `350k` |
| `Phúc chưa trả 1tr5` | complete | `lend` | `Phúc` | `1tr5` |
| `thằng Tuấn nợ t 200k` | complete | `lend` | `Tuấn` | `200k` |
| `Lộc quên trả mình tiền xăng` | complete | `lend` | `Lộc` | null |
| `tiền nợ Lan 1tr5` | uncertain | — | `Lan` | `1tr5` |

Not debt-only (unchanged): `trả nợ chị Hoa 1tr` is `repayment_out`, `Hải trả nợ 500k rồi` is
`repayment_in`, `mượn Hùng 500k` is `borrow` because the principal is received now.
`trả tiền bác Sáu mượn lần trước 3tr` is a repayment, not a debt-only note.


## Part 4: combined-pass guide

## How to label a note

1. Pick the **type** (`t` or `1`-`8`): the eight types.
2. With the **target** span active, select the counterparty and confirm, or press `n` for no
   counterparty. `transfer` has no target (it is saved as null).
3. Press **Tab** to switch to the **value** span, select the amount, or press `n` if the note
   states no amount. Tab switches back.
4. Press `c` while the value span is active to switch the **value status** between `complete`
   (default) and `uncertain`.
5. `enter` saves the note as `complete`, `u` as `uncertain` (type, target or value is unclear),
   `s` skips it. In this pass **`skipped` means "reject this note as unusable"** (garbage, not a
   finance note): type, target and value are saved as null. A debt-only note is never skipped.

| what you decide | Quet field |
|---|---|
| the transaction type | type |
| the counterparty, or none | the `target` span (`n` = null; always null for `transfer`) |
| the amount, or none stated | the `value` span (`n` = null) |
| sure / unsure about the amount alone | value status `complete` / `uncertain` (`c`) |
| sure / unsure / unusable | status `complete` / `uncertain` / `skipped` |
| why you are unsure, or anything odd | `note` (required for status `uncertain` and for value status `uncertain`) |

Both spans are exact substrings of the note with the original offsets (Python code points; Quet
records them), no leading or trailing space. **Never normalise**: `5 xị` stays `5 xị`, `1tr5`
stays `1tr5`; do not rewrite it as 500000 or 1500000, do not fix typos or case. Strange spacing or
punctuation in a note (`grab   85k`, `cf 25k..`) is part of the test: label it as it is and say so
in the `note` if you like.

## Type and target (annotation-v1 + v3 debt-only)

* **Type.** The user owes and nothing moves now (`còn nợ Hùng 300k`) -> `borrow`; the other party
  owes the user (`Hoa còn nợ mình 350k`) -> `lend`; direction unclear -> `uncertain` with a note.
  Receiving principal (`mượn Hùng 500k`) is `borrow`, repaying is `repayment_out`; a bare
  `X trả 500k` with no sign of a prior debt is `uncertain`.
* **Target.** The external counterparty: a named shop, brand, platform, person, or a kinship/role
  noun that is the only identifier (`mẹ`, `khách`, `chủ trọ`). Drop a kinship/title prefix before a
  proper name (`chị Mai` -> `Mai`), keep birth-order names whole (`chú Tám`, `dì Út`). Never an
  item, dish, activity or bill name, a companion (`với`, `vs`), a beneficiary (`cho bé`, `tặng
  mẹ`), or a bank / wallet used only as the channel (`rút tiền vpbank` is a transfer: null). A gift
  or ceremony money goes to its recipient (`mừng cưới Tuấn` -> `Tuấn`). Several plausible
  counterparties -> `uncertain`.

## Value examples

| note | value |
|---|---|
| `cơm tấm 100` | span `100` (a bare number is money when nothing else explains it) |
| `mượn chú hai 5 xị` | span `5 xị` |
| `tiền điện tháng 10 hết 612` | span `612` (`tháng 10` is a month) |
| `20/10 mua quà mẹ 500` | span `500` (`20/10` is a date) |
| `họp t6 2025 mua 100k` | span `100k` (month and year are not amounts) |
| `mua 3 vé 150k` | span `150k` (`3 vé` is a quantity) |
| `ăn 2 tô phở 70` | span `70` (item count `2`) |
| `trả góp kỳ 3 1tr5` | span `1tr5` (`kỳ 3` is the installment/index) |
| `chuyển khoản stk 123456789 200k` | span `200k` (the account number is a reference) |
| `tầm 50k/tháng` | span `50k` (no approximator, no `/tháng`) |
| `gói 1 triệu 2` | span `1 triệu 2` (one compound expression) |
| `thưởng tết 2 tháng lương` | value null, `complete` (`2 tháng` is a period) |
| `cho mượn tiền ăn` | value null, `complete` (no number, no stated amount) |
| `bốn triệu tiền nhà` | span `bốn triệu` (spelled-out amounts count) |

## Judgement rules for the value

* **Several numbers**: pick the one that is the transaction amount; dates, months, years,
  quantities, item counts, installment or period indices, percentages, times, identifiers and
  model numbers are never part of the span.
* **Several money amounts and none is clearly the transaction amount** (a bill and a tip, a total
  and a share): value status `uncertain` (`c`) with a `note` saying which candidates compete.
  Never pick one to be safe.
* **Bare numbers** (`100`, `500`, no unit) are money when nothing else explains them; if you
  cannot tell money from a count or a reference, mark the value `uncertain` with a note.
* **Null**: a note that states no amount at all (no number, or only dates, periods or
  quantities) has a null value (`n`) and stays `complete`.
* **Ambiguous or undelimitable** (the amount is unreadable, truncated, or you cannot say where it
  starts or ends): value status `uncertain` with a note.
* When you change your mind or find an oddity (strange spacing, mixed languages), say so in the
  `note`; do not change the span to compensate.

The value status is independent of type and target: a note can be `complete` with an `uncertain`
value only if the type and target are settled and only the amount is in doubt.


## Part 5: the combined schema (types, statuses, spans)

```yaml
# Quet schema for the COMBINED annotation-v3 pass (Quet >= 0.2.8, several span fields): one pass
# labels the transaction type, the target (counterparty) span and the value (amount) span of each
# note. Semantics are annotation-v2 (type + target: docs/annotation-v1.md, configs/annotation-v1.yaml;
# value span: docs/annotation-v2.md "The span convention") with ONE change, annotation-v3
# (docs/annotation-v3.md): a DEBT-ONLY note (an existing debt, no money moves now) is borrow when
# the user owes and lend when the other party owes the user, not skipped. Gidi validates the
# labels with gidi.annotation.combined (scripts/validate_annotations.py --config this file).
#
# Keys in the annotator: Tab / Shift+Tab switch the active span (target, value), `x` selects text
# for it, `n` nulls it, `c` cycles the status of the value span (complete / uncertain).
#
#   quet annotate datasets/annotation-v3/debt-01/queue.jsonl \
#     --schema configs/annotation-v3.quet.yaml \
#     --out datasets/annotation-v3/debt-01/labels.jsonl
version: annotation-v3

types:
  expense: Money leaves the user with no expectation of principal repayment.
  income: >-
    Money received as salary, freelance/client payment, gift, sale proceeds, rewards, or other
    ordinary incoming money that is neither borrowing, debt repayment, nor refund.
  borrow: >-
    The user receives loan principal and now owes another party. annotation-v3: also a
    DEBT-ONLY note where the user owes the other party and no money moves now ("còn nợ Hùng
    300k", "nợ chủ trọ tiền điện 280k", "chưa trả anh Việt 600k"). Target = the creditor
    (kinship prefix dropped before a name, role noun kept when it is the only name, bill names
    are not targets); value = the amount, null if none.
  lend: >-
    The user gives loan principal to another party and expects repayment. annotation-v3: also a
    DEBT-ONLY note where the other party owes the user and no money moves now ("Hoa còn nợ mình
    350k", "Phúc chưa trả 1tr5", "Nam hứa t6 trả 800k"). Target = the debtor, value = the
    amount, null if none. A debt-only note whose direction is unclear is uncertain with a note.
  repayment_in: >-
    Another party repays money they previously owed the user (needs evidence of a prior debt;
    a bare "Nam trả 500k" is uncertain).
  repayment_out: >-
    The user repays money they previously owed another party, including payments on an existing
    financed balance or liability (installments, BNPL, credit-card balance).
  transfer: >-
    Movement between the user's own accounts, wallets, cash holdings, or savings, including
    conversion into investment/store-of-value assets the user still owns (fund certificates,
    gold held as savings, foreign currency). No economic income or expense occurs.
    vi-VN heuristic: precious metals/gems bought by quantity or purity (chỉ, lượng, phân, g,
    carat, 9999, 24K) are transfer ("mua nhẫn PNJ nửa chỉ") unless the note states a use, gift,
    ceremony, or fashion purpose, which overrides ("mua nhẫn vàng 1 chỉ tặng vợ" -> expense).
    A returned refundable deposit is transfer ("chủ nhà trả cọc 5tr", "lấy lại tiền cọc 2tr");
    paying a deposit is transfer only when the note establishes it stays refundable.
  refund: >-
    Money returned to the user as reversal, reimbursement, rebate, or spending-linked cashback of
    an earlier expense.

statuses:
  complete: Type, target (or explicit null target) and value are confidently determined.
  uncertain: >-
    The text does not establish the type, the target or the value; never forced. Gidi requires a
    note saying what is unclear.
  skipped: >-
    Not a usable finance note, a valid finance note outside the transaction-event taxonomy, or
    not worth labelling. Type, target and value are all saved as null. annotation-v3: a
    debt-only note is NOT skipped, it is borrow or lend. In the debt-01 relabel pass, skipped
    means "reject this note" (it is not a debt-only note after all).

# Marking one of these saves a null type and null for every span.
null_label_statuses: [skipped]

# YAML order is the order in the annotator. Offsets are code points into the NFC note.
spans:
  target:
    description: >-
      Minimal explicit expression for the external financial counterparty: a named
      merchant/service/platform ("highlands", "shopee"), a person, or a kinship/role noun that is
      the only identifier (mẹ, khách, cty, chủ nhà, sếp), without a kinship/title prefix before a
      proper name (anh/chị/bé). Never items, dishes, activities, bill names ("cơm gà", "tiền
      nhà"), companions (với/cùng), beneficiaries (cho bé) or a payment app/bank used only as the
      channel. No counterparty named -> null (n). Several plausible counterparties -> status
      uncertain. Always null for transfer.
    null_for_types: [transfer]
  value:
    description: >-
      The monetary amount: the minimal full amount expression, the number plus the unit, slang or
      currency marker attached to it ("50k", "1tr5", "2 triệu", "5 xị", "80.000đ", "100",
      "bốn triệu"). Exclude quantities, item counts, months/periods/ordinals, dates, times,
      percentages, identifiers, approximators (tầm, khoảng) and /tháng. Never normalise the text.
      The note states no amount -> null (n). Several amounts and none clearly the transaction
      amount -> status uncertain on this span, with a note.
    statuses: [complete, uncertain]
```
