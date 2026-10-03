# value-span-v2-direction-repair: training-data direction audit

Training rows only (`datasets/annotation-v2/training-v1/train.jsonl`). Test and probe were not read. Families are defined in `scripts/build_direction_repair.py`.

## Class counts per batch (train)

| batch | borrow | lend | all |
|---|---|---|---|
| baseline-01 | 35 | 36 | 497 |
| targeted-annotation-v1-01 | 17 | 15 | 114 |
| targeted-annotation-v1-02 | 27 | 18 | 112 |
| targeted-value-01 | 23 | 21 | 172 |

## Lend/borrow wording families (train)

| group | family | borrow | lend | accented | unaccented |
|---|---|---|---|---|---|
| baseline-01 | B_cho_X_muon | 0 | 33 | 18 | 15 |
| baseline-01 | B_X_muon | 0 | 3 | 3 | 0 |
| baseline-01 | U_muon_X | 34 | 0 | 21 | 13 |
| baseline-01 | C_other | 1 | 0 | 1 | 0 |
| targeted-annotation-v1-01 | A_X_cho_muon | 2 | 0 | 2 | 0 |
| targeted-annotation-v1-01 | B_cho_X_muon | 0 | 9 | 5 | 4 |
| targeted-annotation-v1-01 | B_X_muon | 0 | 4 | 3 | 1 |
| targeted-annotation-v1-01 | U_muon_X | 11 | 0 | 7 | 4 |
| targeted-annotation-v1-01 | C_other | 4 | 2 | 4 | 2 |
| targeted-annotation-v1-02 | A_X_cho_muon | 19 | 0 | 9 | 10 |
| targeted-annotation-v1-02 | A_duoc_X_cho | 1 | 0 | 1 | 0 |
| targeted-annotation-v1-02 | B_cho_X_muon | 0 | 3 | 2 | 1 |
| targeted-annotation-v1-02 | B_self_cho_X_muon | 0 | 5 | 2 | 3 |
| targeted-annotation-v1-02 | B_X_muon | 5 | 10 | 7 | 8 |
| targeted-annotation-v1-02 | U_muon_X | 2 | 0 | 1 | 1 |
| targeted-value-01 | A_X_cho_muon | 8 | 0 | 7 | 1 |
| targeted-value-01 | B_cho_X_muon | 0 | 21 | 11 | 10 |
| targeted-value-01 | U_muon_X | 14 | 0 | 5 | 9 |
| targeted-value-01 | C_other | 1 | 0 | 0 | 1 |
| v1_rows (all but targeted-value-01) | A_X_cho_muon | 21 | 0 | 11 | 10 |
| v1_rows (all but targeted-value-01) | A_duoc_X_cho | 1 | 0 | 1 | 0 |
| v1_rows (all but targeted-value-01) | B_cho_X_muon | 0 | 45 | 25 | 20 |
| v1_rows (all but targeted-value-01) | B_self_cho_X_muon | 0 | 5 | 2 | 3 |
| v1_rows (all but targeted-value-01) | B_X_muon | 5 | 17 | 13 | 9 |
| v1_rows (all but targeted-value-01) | U_muon_X | 47 | 0 | 29 | 18 |
| v1_rows (all but targeted-value-01) | C_other | 5 | 2 | 5 | 2 |
| merged_train | A_X_cho_muon | 29 | 0 | 18 | 11 |
| merged_train | A_duoc_X_cho | 1 | 0 | 1 | 0 |
| merged_train | B_cho_X_muon | 0 | 66 | 36 | 30 |
| merged_train | B_self_cho_X_muon | 0 | 5 | 2 | 3 |
| merged_train | B_X_muon | 5 | 17 | 13 | 9 |
| merged_train | U_muon_X | 61 | 0 | 34 | 27 |
| merged_train | C_other | 6 | 2 | 5 | 3 |

## Family examples (first 6 by id)

### baseline-01

- **B_cho_X_muon**: `cho em Ngọc mượn 400k`; `cho dong nghiep Thao muon 200k`; `cho thằng Bin mượn 150k`; `cho em Tí mượn 200 nghìn`; `cho em Tú vay 2tr`; `cho Tú mượn 1tr`
- **B_X_muon**: `Hằng mượn 250k ăn trưa`; `Tùng mượn 2tr hẹn cuối tháng`; `Phương mượn 150k đổ xăng`
- **U_muon_X**: `vay cậu Ba 10tr`; `muon tam dong nghiep 300k`; `vay ông ngoại 20tr làm ăn`; `vay bạn Quang 5tr, hẹn tháng sau trả`; `mượn đồng nghiệp 200k ăn trưa`; `mượn a Sơn 700 nghìn`
- **C_other**: `ứng trước lương cty 2tr`

### targeted-annotation-v1-01

- **A_X_cho_muon**: `anh Tư cho vay 20tr`; `bố cho mượn 5tr mua xe`
- **B_cho_X_muon**: `cho em hang xom vay 300k`; `cho vay Nghia 3tr lai 2%`; `cho bạn cùng lớp mượn 150k`; `cho ban cu muon 1tr5`; `cho chị Hoa mượn tạm 800k`; `cho Dung muon 400k`
- **B_X_muon**: `bạn thân hỏi mượn 3tr, đã ck`; `Ngân mượn 300k đóng tiền điện`; `em họ vay 2 triệu nộp học phí`; `Tien hoi muon 600k da ck`
- **U_muon_X**: `vay Hải 3tr`; `mượn em gái 400k đổ xăng`; `muon tam ban 300k`; `vay qua app 2tr, nhận tiền rồi`; `muon Nhan 200k`; `mượn chị Loan 500k tiền trọ`
- **C_other**: `ứng trước lương chủ quán 1tr`; `ung luong 2tr`; `rút thấu chi tcb 2tr`; `ứng lương tháng 10 3tr`; `ứng cho em trai 1tr`; `ung cho Khai 500k`

### targeted-annotation-v1-02

- **A_X_cho_muon**: `anh ba cho t muon 700k sua may tinh`; `cô Hạnh cho mượn 3tr đóng tiền trọ`; `chú Năm cho mượn tạm 1tr5 trả tiền điện`; `ban cung phong cho muon 800k dong tien dien`; `Phuc cho minh muon 500k do xang`; `chi gai cho muon 6tr dat coc phong tro`
- **A_duoc_X_cho**: `được đồng nghiệp cho mượn 2tr đi khám`
- **B_cho_X_muon**: `cho bác Hai vay 10tr mua laptop`; `cho cô Hạnh mượn 3tr đóng tiền trọ`; `cho ban cung phong muon 800k dong tien dien`
- **B_self_cho_X_muon**: `minh cho Phuc muon 500k do xang`; `e cho chu Tin vay 5tr sua xe`; `em cho dì Út vay 4 củ đi viện`; `mk cho chi ba muon 1tr5 dong tien net`; `t cho anh Khôi mượn 2tr, lương về ảnh gửi lại`
- **B_X_muon**: `Thảo vay mình 1tr5, tuần sau trả`; `mirae asset duyet khoan vay tieu dung, nhan 12tr`; `anh Tuan muon cua t 800k`; `thg Binh vay t 2tr hen luong ve tra`; `mình mượn Huy 300k nạp game, thứ 6 trả`; `em gái vay mình 1tr2 đóng học phí`
- **U_muon_X**: `vay Thảo 1tr5, tuần sau trả`; `muon tam chu Ba 5 cu, cuoi thang gui lai`

### targeted-value-01

- **A_X_cho_muon**: `thg Bảo cho mượn 500 K`; `chị Loan cho mượn 1.500.000`; `cậu Năm cho vay 2 củ rưỡi`; `anh Tu cho vay 3 lit sua xe`; `chị Loan cho mượn 800 ngàn tiền trọ`; `mẹ cho mượn 5000000 đóng tiền trường.`
- **B_cho_X_muon**: `cho cậu Năm vay 2 củ rưỡi`; `Cho Tin muon 100`; `cho Bình mượn 3 lít`; `cho cau ut vay 5 xi mua ga`; `cho anh Tu vay 3 lit sua xe`; `cho em muon 200 K tieu vat`
- **U_muon_X**: `muon chi Hai nua cu`; `vay ba Sau 1,5tr`; `muon me 3000000 tra tien hoc them cho be`; `vay vcb mua nhà 2 tỷ`; `muon Tin 100 an sang.`; `Muon Hoa 800.000d`
- **C_other**: `mb bank giai ngan 50 cu mua xe`

## Name-first direction balance

`X cho mượn` (lender-first, borrow) vs `X mượn` (borrower-first, lend):

| rows | lender-first borrow | borrower-first lend | difference |
|---|---|---|---|
| v1_rows | 22 | 17 | 5 |
| targeted_value_01 | 8 | 0 | 8 |
| merged | 30 | 17 | 13 |
| repaired | 30 | 25 | 5 |

## Generator patterns

targeted-value-01 minimal pairs `X cho mượn` / `cho X mượn`: 8

- `lend: cho anh Tu vay 3 lit sua xe` ↔ `borrow: anh Tu cho vay 3 lit sua xe`
- `borrow: dì Út cho mượn 2 chai` ↔ `lend: cho dì Út mượn 2 chai`
- `lend: cho cậu Năm vay 2 củ rưỡi` ↔ `borrow: cậu Năm cho vay 2 củ rưỡi`
- `lend: cho chị Loan mượn 800 ngàn tiền trọ.` ↔ `borrow: chị Loan cho mượn 800 ngàn tiền trọ`
- `borrow: chị Loan cho mượn 1.500.000` ↔ `lend: cho chị Loan mượn 1.500.000`
- `borrow: mẹ cho mượn 5000000 đóng tiền trường.` ↔ `lend: cho mẹ mượn 5000000 đóng tiền trường`
- `lend: cho thg Bảo mượn 500 K` ↔ `borrow: thg Bảo cho mượn 500 K`
- `lend: cho bác Sáu mượn 1.500.000 tiền thuốc` ↔ `borrow: bác Sáu cho mượn 1.500.000 tiền thuốc`

Repeated digit-masked lend/borrow texts within a batch:

- baseline-01: `cho cau tu muon Ntr` × 2
- baseline-01: `cho linh muon Nk` × 2
- baseline-01: `cho nam muon Ntr` × 2
- baseline-01: `muon dong nghiep Nk an trua` × 3
- baseline-01: `muon me Ntr dong hoc` × 2

## Repair (deterministic)

Rule: append verbatim duplicates of existing non-targeted-value-01 `B_X_muon` lend rows with a complete value label until the name-first difference equals its pre-targeted-value-01 value. Needed: 8 (pool 17). Order: sha256(`value-span-v2-direction-repair:<id>`), distinct value surfaces first.

| duplicate of | text | value | accented |
|---|---|---|---|
| targeted-annotation-v1-01-b32a35e194dc | `Ngân mượn 300k đóng tiền điện` | `300k` | True |
| targeted-annotation-v1-02-74c083c8a1ec | `Khoa muon tam 250k tien xang` | `250k` | False |
| targeted-annotation-v1-01-94faabf61933 | `bạn thân hỏi mượn 3tr, đã ck` | `3tr` | True |
| targeted-annotation-v1-01-f56bb0a25e64 | `Tien hoi muon 600k da ck` | `600k` | False |
| targeted-annotation-v1-02-7b51148566ed | `cau Hai vay 3 trieu, thang sau tra` | `3 trieu` | False |
| targeted-annotation-v1-02-0088fe92eea4 | `Thảo vay mình 1tr5, tuần sau trả` | `1tr5` | True |
| targeted-annotation-v1-02-38d93582ba50 | `thg Binh vay t 2tr hen luong ve tra` | `2tr` | False |
| targeted-annotation-v1-02-20ec75c320c8 | `anh Tuan muon cua t 800k` | `800k` | False |
