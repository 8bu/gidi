# baseline-v1 summary

Runs dir `experiments/baseline-v1/runs-e20`: **6 complete** run(s), **0 incomplete/unreadable** (excluded). Selection score = mean of validation type macro-F1 and validation span F1. Spread cells: mean ± sample std [min, max] across seeds; `n` = seeds.

| model | lr | complete seeds | incomplete |
|---|---|---|---|
| bamibert | 5e-05 | s1, s2, s3 | - |
| multilingual-minilm-l12-h384 | 5e-05 | s1, s2, s3 | - |

## 1. Configs (model x lr)

### Validation (n=110, gold-target 54, gold-null 56)

| model | lr | n | type acc | type macro-F1 | span F1 | span EM | EM (gold target) | null acc (gold null) |
|---|---|---|---|---|---|---|---|---|
| bamibert | 5e-05 | 3 | 0.870 ± 0.023 [0.845, 0.891] | 0.866 ± 0.018 [0.849, 0.885] | 0.860 ± 0.014 [0.845, 0.873] | 0.885 ± 0.014 [0.873, 0.900] | 0.889 ± 0.019 [0.870, 0.907] | 0.881 ± 0.037 [0.839, 0.911] |
| multilingual-minilm-l12-h384 | 5e-05 | 3 | 0.873 ± 0.009 [0.864, 0.882] | 0.873 ± 0.008 [0.864, 0.879] | 0.836 ± 0.024 [0.811, 0.860] | 0.870 ± 0.019 [0.855, 0.891] | 0.821 ± 0.028 [0.796, 0.852] | 0.917 ± 0.010 [0.911, 0.929] |

### Test (n=105, gold-target 51, gold-null 54)

| model | lr | n | type acc | type macro-F1 | span F1 | span EM | EM (gold target) | null acc (gold null) |
|---|---|---|---|---|---|---|---|---|
| bamibert | 5e-05 | 3 | 0.870 ± 0.020 [0.848, 0.886] | 0.862 ± 0.019 [0.842, 0.879] | 0.876 ± 0.026 [0.857, 0.906] | 0.902 ± 0.031 [0.867, 0.924] | 0.922 ± 0.034 [0.882, 0.941] | 0.883 ± 0.077 [0.796, 0.944] |
| multilingual-minilm-l12-h384 | 5e-05 | 3 | 0.863 ± 0.015 [0.848, 0.876] | 0.844 ± 0.013 [0.835, 0.859] | 0.851 ± 0.022 [0.838, 0.876] | 0.883 ± 0.020 [0.867, 0.905] | 0.876 ± 0.023 [0.863, 0.902] | 0.889 ± 0.019 [0.870, 0.907] |

### Epochs, time, size

| model | lr | n | val score | best/run epoch per seed | wall s/run | s/epoch | params |
|---|---|---|---|---|---|---|---|
| bamibert | 5e-05 | 3 | 0.863 ± 0.010 [0.854, 0.874] | s1: 9/12, s2: 19/20, s3: 10/13 | 202.5 ± 55.5 | 13.36 ± 0.27 | 102,369,035 |
| multilingual-minilm-l12-h384 | 5e-05 | 3 | 0.855 ± 0.013 [0.844, 0.870] | s1: 15/18, s2: 14/17, s3: 18/20 | 283.3 ± 22.0 | 15.34 ± 0.11 | 117,510,155 |

## 2. Best config per model

| model | lr | n | mean val score | best seed | seed val score | seed test type F1 | seed test span F1 | checkpoint |
|---|---|---|---|---|---|---|---|---|
| bamibert | 5e-05 | 3 | 0.863 ± 0.010 [0.854, 0.874] | s2 | 0.8737 | 0.879 | 0.865 | models/baseline-v1-e20/bamibert/lr5e-05-seed2 |
| multilingual-minilm-l12-h384 | 5e-05 | 3 | 0.855 ± 0.013 [0.844, 0.870] | s3 | 0.8696 | 0.835 | 0.876 | models/baseline-v1-e20/multilingual-minilm-l12-h384/lr5e-05-seed3 |

## 3. Best-config detail (test, across seeds)

### bamibert (lr 5e-05, 3 seed(s))

Per class (mean over seeds):

| class | support | P | R | F1 | mean n_pred |
|---|---|---|---|---|---|
| expense | 32 | 0.953 | 0.823 | 0.882 | 27.7 |
| income | 11 | 0.830 | 0.697 | 0.755 | 9.3 |
| borrow | 9 | 0.805 | 0.889 | 0.844 | 10.0 |
| lend | 9 | 0.800 | 0.889 | 0.842 | 10.0 |
| repayment_in | 8 | 0.761 | 0.917 | 0.830 | 9.7 |
| repayment_out | 10 | 0.883 | 0.900 | 0.889 | 10.3 |
| transfer | 18 | 0.858 | 1.000 | 0.923 | 21.0 |
| refund | 8 | 1.000 | 0.875 | 0.933 | 7.0 |

Confusion matrix summed over seeds (rows gold, columns predicted):

| gold \ pred | exp | inc | bor | len | rep_in | rep_out | xfer | ref |
|---|---|---|---|---|---|---|---|---|
| expense | 79 | 5 | 1 | 1 | 2 | 1 | 7 | 0 |
| income | 4 | 23 | 1 | 2 | 0 | 1 | 2 | 0 |
| borrow | 0 | 0 | 24 | 3 | 0 | 0 | 0 | 0 |
| lend | 0 | 0 | 1 | 24 | 2 | 0 | 0 | 0 |
| repayment_in | 0 | 0 | 0 | 0 | 22 | 2 | 0 | 0 |
| repayment_out | 0 | 0 | 3 | 0 | 0 | 27 | 0 | 0 |
| transfer | 0 | 0 | 0 | 0 | 0 | 0 | 54 | 0 |
| refund | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 21 |

Slices (mean over seeds):

| slice | value | n | type acc | type macro-F1 | span F1 | span EM |
|---|---|---|---|---|---|---|
| accented | false | 42 | 0.825 | 0.829 | 0.899 | 0.913 |
| accented | true | 63 | 0.899 | 0.880 | 0.863 | 0.894 |
| source_batch | baseline-01 | 80 | 0.871 | 0.860 | 0.853 | 0.904 |
| source_batch | targeted-annotation-v1-01 | 25 | 0.867 | 0.872 | 0.908 | 0.893 |

### multilingual-minilm-l12-h384 (lr 5e-05, 3 seed(s))

Per class (mean over seeds):

| class | support | P | R | F1 | mean n_pred |
|---|---|---|---|---|---|
| expense | 32 | 0.928 | 0.906 | 0.916 | 31.3 |
| income | 11 | 0.785 | 0.727 | 0.753 | 10.3 |
| borrow | 9 | 0.707 | 0.889 | 0.787 | 11.3 |
| lend | 9 | 0.845 | 0.815 | 0.825 | 8.7 |
| repayment_in | 8 | 0.821 | 0.750 | 0.783 | 7.3 |
| repayment_out | 10 | 0.933 | 0.900 | 0.916 | 9.7 |
| transfer | 18 | 0.895 | 0.907 | 0.900 | 18.3 |
| refund | 8 | 0.874 | 0.875 | 0.872 | 8.0 |

Confusion matrix summed over seeds (rows gold, columns predicted):

| gold \ pred | exp | inc | bor | len | rep_in | rep_out | xfer | ref |
|---|---|---|---|---|---|---|---|---|
| expense | 87 | 6 | 1 | 0 | 0 | 0 | 2 | 0 |
| income | 3 | 24 | 0 | 1 | 1 | 1 | 3 | 0 |
| borrow | 0 | 0 | 24 | 3 | 0 | 0 | 0 | 0 |
| lend | 0 | 0 | 3 | 22 | 2 | 0 | 0 | 0 |
| repayment_in | 0 | 0 | 3 | 0 | 18 | 0 | 0 | 3 |
| repayment_out | 0 | 0 | 3 | 0 | 0 | 27 | 0 | 0 |
| transfer | 4 | 1 | 0 | 0 | 0 | 0 | 49 | 0 |
| refund | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 21 |

Slices (mean over seeds):

| slice | value | n | type acc | type macro-F1 | span F1 | span EM |
|---|---|---|---|---|---|---|
| accented | false | 42 | 0.849 | 0.820 | 0.920 | 0.937 |
| accented | true | 63 | 0.873 | 0.848 | 0.812 | 0.847 |
| source_batch | baseline-01 | 80 | 0.879 | 0.854 | 0.786 | 0.871 |
| source_batch | targeted-annotation-v1-01 | 25 | 0.813 | 0.815 | 0.934 | 0.920 |

## 4. Failure cases (best run of each model, test)

### bamibert — `experiments/baseline-v1/runs-e20/bamibert/lr5e-05-seed2`: 12 type error(s), 9 span error(s) of 105

| id | text | gold | pred |
|---|---|---|---|
| baseline-01-dab0dcc75d1e | mua giay the thao 1tr25 | expense | transfer |
| baseline-01-90a34dbd7ef8 | anh Long mượn 3 triệu sửa xe | lend | borrow |
| baseline-01-ee03dc138256 | bố cho tiền tiêu 1 triệu | income | lend |
| baseline-01-e8c066dd0681 | dong bao hiem nhan tho 1tr2 | expense | repayment_in |
| baseline-01-d7d840164a2e | tien phong thang 11 3tr2 | expense | transfer |
| baseline-01-82f29c0ccb74 | vay tiêu dùng trả kỳ này 3tr4 | repayment_out | borrow |
| baseline-01-ebf8bca50f8d | tien chay grab hom nay 450k | income | expense |
| baseline-01-dbf5b4023040 | trúng vé số 200k | income | transfer |
| baseline-01-56a7ee75454e | gia han goi 4g 120k | expense | repayment_in |
| baseline-01-d74a957149fd | lì xì cháu 200k | expense | income |
| targeted-annotation-v1-01-6bd81e618b99 | co Thuy cho muon 1tr | borrow | lend |
| targeted-annotation-v1-01-81f1c9cec4b3 | bao hiem tra tien kham 850k | refund | repayment_in |

| id | text | gold span | pred span |
|---|---|---|---|
| baseline-01-2123a362f627 | ban may anh cu 6tr5 | ∅ | `anh cu` [8:14] |
| baseline-01-56a7ee75454e | gia han goi 4g 120k | ∅ | `han` [4:7] |
| baseline-01-30cd0e980666 | bách hoá xanh rau thịt 187k | `bách hoá xanh` [0:13] | ∅ |
| baseline-01-304bfe5637ab | cho a Nam vay 1 triệu 20/10 | `Nam` [6:9] | `a` [4:5] |
| baseline-01-e2f16245d7af | vay chú Hải 20tr mua xe | `Hải` [8:11] | `chú Hải` [4:11] |
| targeted-annotation-v1-01-6ddde33f35c6 | dì Tư gửi trả 3 triệu đã mượn | `dì Tư` [0:5] | `Tư` [3:5] |
| targeted-annotation-v1-01-a07b6b6eb45c | bảo việt bồi thường viện phí 2tr6 | `bảo việt` [0:8] | `bảo` [0:3] |
| targeted-annotation-v1-01-8b9aea00438d | Thao tra lai 350k minh chi ho | `Thao` [0:4] | `Th` [0:2] |
| targeted-annotation-v1-01-81f1c9cec4b3 | bao hiem tra tien kham 850k | ∅ | `bao` [0:3] |

### multilingual-minilm-l12-h384 — `experiments/baseline-v1/runs-e20/multilingual-minilm-l12-h384/lr5e-05-seed3`: 14 type error(s), 10 span error(s) of 105

| id | text | gold | pred |
|---|---|---|---|
| baseline-01-2123a362f627 | ban may anh cu 6tr5 | income | expense |
| baseline-01-90a34dbd7ef8 | anh Long mượn 3 triệu sửa xe | lend | borrow |
| baseline-01-ee03dc138256 | bố cho tiền tiêu 1 triệu | income | repayment_in |
| baseline-01-cd356c06d417 | nạp ví grab 150k | transfer | expense |
| baseline-01-e8c066dd0681 | dong bao hiem nhan tho 1tr2 | expense | income |
| baseline-01-2d74ae01e1f0 | Nam muon 150k tien grab | lend | borrow |
| baseline-01-82f29c0ccb74 | vay tiêu dùng trả kỳ này 3tr4 | repayment_out | borrow |
| baseline-01-dbf5b4023040 | trúng vé số 200k | income | repayment_out |
| baseline-01-d74a957149fd | lì xì cháu 200k | expense | income |
| targeted-annotation-v1-01-6b71966f3f06 | Thắng vay 5 củ, hẹn t10 trả | lend | repayment_in |
| targeted-annotation-v1-01-7d1037648087 | roommate tra tien nha minh dong truoc 1tr5 | repayment_in | refund |
| targeted-annotation-v1-01-6bd81e618b99 | co Thuy cho muon 1tr | borrow | lend |
| targeted-annotation-v1-01-81f1c9cec4b3 | bao hiem tra tien kham 850k | refund | transfer |
| targeted-annotation-v1-01-9d98b78854de | đòi được nợ thằng Lâm 400k | repayment_in | borrow |

| id | text | gold span | pred span |
|---|---|---|---|
| baseline-01-90a34dbd7ef8 | anh Long mượn 3 triệu sửa xe | `Long` [4:8] | `anh` [0:3] |
| baseline-01-cd356c06d417 | nạp ví grab 150k | ∅ | `grab` [7:11] |
| baseline-01-e8c066dd0681 | dong bao hiem nhan tho 1tr2 | ∅ | `dong` [0:4] |
| baseline-01-82f29c0ccb74 | vay tiêu dùng trả kỳ này 3tr4 | ∅ | `tiêu dùng` [4:13] |
| baseline-01-ba4d9f58a5ff | nuoc mia 12k | ∅ | `mia` [5:8] |
| baseline-01-30cd0e980666 | bách hoá xanh rau thịt 187k | `bách hoá xanh` [0:13] | ∅ |
| baseline-01-e2f16245d7af | vay chú Hải 20tr mua xe | `Hải` [8:11] | `chú Hải` [4:11] |
| baseline-01-30d790f0205f | mừng thọ bà nội 1 triệu | `bà nội` [9:15] | ∅ |
| baseline-01-83a8de1156c4 | mở sổ tiết kiệm agribank 15tr | ∅ | `agribank` [16:24] |
| targeted-annotation-v1-01-4d06a6abdd9b | điện máy xanh hoàn tiền nồi cơm lỗi 890k | `điện máy xanh` [0:13] | `xanh` [9:13] |

### Both/all models err (bamibert, multilingual-minilm-l12-h384; 105 ids in common, 0 unmatched)

Type errors in all models: 8

| id | text | gold | pred bamibert | pred multilingual-minilm-l12-h384 |
|---|---|---|---|---|
| baseline-01-82f29c0ccb74 | vay tiêu dùng trả kỳ này 3tr4 | repayment_out | borrow | borrow |
| baseline-01-90a34dbd7ef8 | anh Long mượn 3 triệu sửa xe | lend | borrow | borrow |
| baseline-01-d74a957149fd | lì xì cháu 200k | expense | income | income |
| baseline-01-dbf5b4023040 | trúng vé số 200k | income | transfer | repayment_out |
| baseline-01-e8c066dd0681 | dong bao hiem nhan tho 1tr2 | expense | repayment_in | income |
| baseline-01-ee03dc138256 | bố cho tiền tiêu 1 triệu | income | lend | repayment_in |
| targeted-annotation-v1-01-6bd81e618b99 | co Thuy cho muon 1tr | borrow | lend | lend |
| targeted-annotation-v1-01-81f1c9cec4b3 | bao hiem tra tien kham 850k | refund | repayment_in | transfer |

Span errors in all models: 2

| id | text | gold span | pred bamibert | pred multilingual-minilm-l12-h384 |
|---|---|---|---|---|
| baseline-01-30cd0e980666 | bách hoá xanh rau thịt 187k | `bách hoá xanh` [0:13] | ∅ | ∅ |
| baseline-01-e2f16245d7af | vay chú Hải 20tr mua xe | `Hải` [8:11] | `chú Hải` [4:11] | `chú Hải` [4:11] |

