# baseline-v1 summary

Runs dir `experiments/baseline-v1/runs`: **12 complete** run(s), **0 incomplete/unreadable** (excluded). Selection score = mean of validation type macro-F1 and validation span F1. Spread cells: mean ± sample std [min, max] across seeds; `n` = seeds.

| model | lr | complete seeds | incomplete |
|---|---|---|---|
| bamibert | 2e-05 | s1, s2, s3 | - |
| bamibert | 5e-05 | s1, s2, s3 | - |
| multilingual-minilm-l12-h384 | 2e-05 | s1, s2, s3 | - |
| multilingual-minilm-l12-h384 | 5e-05 | s1, s2, s3 | - |

## 1. Configs (model x lr)

### Validation (n=110, gold-target 54, gold-null 56)

| model | lr | n | type acc | type macro-F1 | span F1 | span EM | EM (gold target) | null acc (gold null) |
|---|---|---|---|---|---|---|---|---|
| bamibert | 2e-05 | 3 | 0.848 ± 0.019 [0.827, 0.864] | 0.848 ± 0.029 [0.815, 0.868] | 0.793 ± 0.105 [0.673, 0.865] | 0.848 ± 0.050 [0.791, 0.882] | 0.790 ± 0.126 [0.648, 0.889] | 0.905 ± 0.027 [0.875, 0.929] |
| bamibert | 5e-05 | 3 | 0.855 ± 0.018 [0.836, 0.873] | 0.856 ± 0.022 [0.833, 0.878] | 0.843 ± 0.004 [0.839, 0.847] | 0.873 ± 0.000 [0.873, 0.873] | 0.864 ± 0.011 [0.852, 0.870] | 0.881 ± 0.010 [0.875, 0.893] |
| multilingual-minilm-l12-h384 | 2e-05 | 3 | 0.761 ± 0.034 [0.736, 0.800] | 0.706 ± 0.058 [0.653, 0.768] | 0.731 ± 0.005 [0.725, 0.736] | 0.809 ± 0.009 [0.800, 0.818] | 0.704 ± 0.019 [0.685, 0.722] | 0.911 ± 0.018 [0.893, 0.929] |
| multilingual-minilm-l12-h384 | 5e-05 | 3 | 0.852 ± 0.005 [0.845, 0.855] | 0.843 ± 0.011 [0.830, 0.851] | 0.787 ± 0.046 [0.743, 0.835] | 0.842 ± 0.037 [0.809, 0.882] | 0.772 ± 0.043 [0.722, 0.796] | 0.911 ± 0.047 [0.875, 0.964] |

### Test (n=105, gold-target 51, gold-null 54)

| model | lr | n | type acc | type macro-F1 | span F1 | span EM | EM (gold target) | null acc (gold null) |
|---|---|---|---|---|---|---|---|---|
| bamibert | 2e-05 | 3 | 0.860 ± 0.015 [0.848, 0.876] | 0.851 ± 0.025 [0.835, 0.880] | 0.759 ± 0.091 [0.654, 0.819] | 0.835 ± 0.056 [0.771, 0.876] | 0.784 ± 0.102 [0.667, 0.843] | 0.883 ± 0.021 [0.870, 0.907] |
| bamibert | 5e-05 | 3 | 0.883 ± 0.011 [0.876, 0.895] | 0.876 ± 0.010 [0.867, 0.887] | 0.873 ± 0.034 [0.838, 0.906] | 0.911 ± 0.022 [0.886, 0.924] | 0.895 ± 0.041 [0.863, 0.941] | 0.926 ± 0.032 [0.907, 0.963] |
| multilingual-minilm-l12-h384 | 2e-05 | 3 | 0.721 ± 0.029 [0.695, 0.752] | 0.638 ± 0.050 [0.582, 0.679] | 0.796 ± 0.012 [0.784, 0.808] | 0.863 ± 0.005 [0.857, 0.867] | 0.765 ± 0.020 [0.745, 0.784] | 0.957 ± 0.011 [0.944, 0.963] |
| multilingual-minilm-l12-h384 | 5e-05 | 3 | 0.851 ± 0.011 [0.838, 0.857] | 0.837 ± 0.003 [0.835, 0.840] | 0.801 ± 0.025 [0.777, 0.827] | 0.860 ± 0.015 [0.848, 0.876] | 0.817 ± 0.030 [0.784, 0.843] | 0.901 ± 0.011 [0.889, 0.907] |

### Epochs, time, size

| model | lr | n | val score | best/run epoch per seed | wall s/run | s/epoch | params |
|---|---|---|---|---|---|---|---|
| bamibert | 2e-05 | 3 | 0.821 ± 0.066 [0.744, 0.863] | s1: 10/10, s2: 10/10, s3: 9/10 | 134.3 ± 1.0 | 13.23 ± 0.08 | 102,369,035 |
| bamibert | 5e-05 | 3 | 0.850 ± 0.013 [0.836, 0.861] | s1: 8/10, s2: 10/10, s3: 10/10 | 132.1 ± 0.2 | 13.04 ± 0.02 | 102,369,035 |
| multilingual-minilm-l12-h384 | 2e-05 | 3 | 0.718 ± 0.028 [0.692, 0.747] | s1: 8/10, s2: 10/10, s3: 9/10 | 149.2 ± 0.2 | 14.81 ± 0.03 | 117,510,155 |
| multilingual-minilm-l12-h384 | 5e-05 | 3 | 0.815 ± 0.028 [0.787, 0.843] | s1: 10/10, s2: 9/10, s3: 9/10 | 148.9 ± 0.1 | 14.79 ± 0.00 | 117,510,155 |

## 2. Best config per model

| model | lr | n | mean val score | best seed | seed val score | seed test type F1 | seed test span F1 | checkpoint |
|---|---|---|---|---|---|---|---|---|
| bamibert | 5e-05 | 3 | 0.850 ± 0.013 [0.836, 0.861] | s2 | 0.8608 | 0.887 | 0.838 | models/baseline-v1/bamibert/lr5e-05-seed2 |
| multilingual-minilm-l12-h384 | 5e-05 | 3 | 0.815 ± 0.028 [0.787, 0.843] | s3 | 0.8430 | 0.840 | 0.827 | models/baseline-v1/multilingual-minilm-l12-h384/lr5e-05-seed3 |

## 3. Best-config detail (test, across seeds)

### bamibert (lr 5e-05, 3 seed(s))

Per class (mean over seeds):

| class | support | P | R | F1 | mean n_pred |
|---|---|---|---|---|---|
| expense | 32 | 0.964 | 0.833 | 0.893 | 27.7 |
| income | 11 | 0.859 | 0.727 | 0.787 | 9.3 |
| borrow | 9 | 0.738 | 0.889 | 0.804 | 11.0 |
| lend | 9 | 0.896 | 0.963 | 0.928 | 9.7 |
| repayment_in | 8 | 0.859 | 1.000 | 0.924 | 9.3 |
| repayment_out | 10 | 0.845 | 0.900 | 0.871 | 10.7 |
| transfer | 18 | 0.899 | 0.963 | 0.929 | 19.3 |
| refund | 8 | 0.875 | 0.875 | 0.875 | 8.0 |

Confusion matrix summed over seeds (rows gold, columns predicted):

| gold \ pred | exp | inc | bor | len | rep_in | rep_out | xfer | ref |
|---|---|---|---|---|---|---|---|---|
| expense | 80 | 4 | 3 | 0 | 1 | 2 | 5 | 1 |
| income | 3 | 24 | 2 | 0 | 0 | 3 | 1 | 0 |
| borrow | 0 | 0 | 24 | 3 | 0 | 0 | 0 | 0 |
| lend | 0 | 0 | 1 | 26 | 0 | 0 | 0 | 0 |
| repayment_in | 0 | 0 | 0 | 0 | 24 | 0 | 0 | 0 |
| repayment_out | 0 | 0 | 3 | 0 | 0 | 27 | 0 | 0 |
| transfer | 0 | 0 | 0 | 0 | 0 | 0 | 52 | 2 |
| refund | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 21 |

Slices (mean over seeds):

| slice | value | n | type acc | type macro-F1 | span F1 | span EM |
|---|---|---|---|---|---|---|
| accented | false | 42 | 0.825 | 0.805 | 0.912 | 0.929 |
| accented | true | 63 | 0.921 | 0.913 | 0.850 | 0.899 |
| source_batch | baseline-01 | 80 | 0.875 | 0.879 | 0.846 | 0.917 |
| source_batch | targeted-annotation-v1-01 | 25 | 0.907 | 0.892 | 0.906 | 0.893 |

### multilingual-minilm-l12-h384 (lr 5e-05, 3 seed(s))

Per class (mean over seeds):

| class | support | P | R | F1 | mean n_pred |
|---|---|---|---|---|---|
| expense | 32 | 0.933 | 0.865 | 0.897 | 29.7 |
| income | 11 | 0.718 | 0.697 | 0.707 | 10.7 |
| borrow | 9 | 0.756 | 0.889 | 0.815 | 10.7 |
| lend | 9 | 0.826 | 0.852 | 0.838 | 9.3 |
| repayment_in | 8 | 0.800 | 0.792 | 0.794 | 8.0 |
| repayment_out | 10 | 0.906 | 0.900 | 0.902 | 10.0 |
| transfer | 18 | 0.845 | 0.926 | 0.881 | 20.0 |
| refund | 8 | 0.952 | 0.792 | 0.863 | 6.7 |

Confusion matrix summed over seeds (rows gold, columns predicted):

| gold \ pred | exp | inc | bor | len | rep_in | rep_out | xfer | ref |
|---|---|---|---|---|---|---|---|---|
| expense | 83 | 7 | 0 | 1 | 0 | 0 | 5 | 0 |
| income | 4 | 23 | 1 | 1 | 0 | 0 | 4 | 0 |
| borrow | 0 | 0 | 24 | 3 | 0 | 0 | 0 | 0 |
| lend | 0 | 0 | 2 | 23 | 2 | 0 | 0 | 0 |
| repayment_in | 0 | 0 | 2 | 0 | 19 | 1 | 1 | 1 |
| repayment_out | 0 | 0 | 3 | 0 | 0 | 27 | 0 | 0 |
| transfer | 2 | 2 | 0 | 0 | 0 | 0 | 50 | 0 |
| refund | 0 | 0 | 0 | 0 | 3 | 2 | 0 | 19 |

Slices (mean over seeds):

| slice | value | n | type acc | type macro-F1 | span F1 | span EM |
|---|---|---|---|---|---|---|
| accented | false | 42 | 0.817 | 0.809 | 0.847 | 0.897 |
| accented | true | 63 | 0.873 | 0.850 | 0.779 | 0.836 |
| source_batch | baseline-01 | 80 | 0.867 | 0.853 | 0.725 | 0.850 |
| source_batch | targeted-annotation-v1-01 | 25 | 0.800 | 0.820 | 0.899 | 0.893 |

## 4. Failure cases (best run of each model, test)

### bamibert — `experiments/baseline-v1/runs/bamibert/lr5e-05-seed2`: 11 type error(s), 12 span error(s) of 105

| id | text | gold | pred |
|---|---|---|---|
| baseline-01-dab0dcc75d1e | mua giay the thao 1tr25 | expense | repayment_out |
| baseline-01-ee03dc138256 | bố cho tiền tiêu 1 triệu | income | borrow |
| baseline-01-e8c066dd0681 | dong bao hiem nhan tho 1tr2 | expense | transfer |
| baseline-01-d7d840164a2e | tien phong thang 11 3tr2 | expense | transfer |
| baseline-01-82f29c0ccb74 | vay tiêu dùng trả kỳ này 3tr4 | repayment_out | borrow |
| baseline-01-ebf8bca50f8d | tien chay grab hom nay 450k | income | expense |
| baseline-01-dbf5b4023040 | trúng vé số 200k | income | repayment_out |
| baseline-01-d1c0bf5f7646 | chuyen tien the phu 700k | transfer | refund |
| baseline-01-d74a957149fd | lì xì cháu 200k | expense | income |
| targeted-annotation-v1-01-6bd81e618b99 | co Thuy cho muon 1tr | borrow | lend |
| targeted-annotation-v1-01-81f1c9cec4b3 | bao hiem tra tien kham 850k | refund | repayment_in |

| id | text | gold span | pred span |
|---|---|---|---|
| baseline-01-2123a362f627 | ban may anh cu 6tr5 | ∅ | `anh cu` [8:14] |
| baseline-01-90a34dbd7ef8 | anh Long mượn 3 triệu sửa xe | `Long` [4:8] | `anh Long` [0:8] |
| baseline-01-f6e34c255202 | mua hoa qua biếu bà 200k | ∅ | `bà` [17:19] |
| baseline-01-82f29c0ccb74 | vay tiêu dùng trả kỳ này 3tr4 | ∅ | `tiêu dùng` [4:13] |
| baseline-01-56a7ee75454e | gia han goi 4g 120k | ∅ | `han goi` [4:11] |
| baseline-01-30cd0e980666 | bách hoá xanh rau thịt 187k | `bách hoá xanh` [0:13] | ∅ |
| baseline-01-304bfe5637ab | cho a Nam vay 1 triệu 20/10 | `Nam` [6:9] | `a Nam` [4:9] |
| baseline-01-e2f16245d7af | vay chú Hải 20tr mua xe | `Hải` [8:11] | `chú Hải` [4:11] |
| baseline-01-00093031738d | trả dần nợ cô Hạnh 500k | `Hạnh` [14:18] | `cô` [11:13] |
| targeted-annotation-v1-01-8b9aea00438d | Thao tra lai 350k minh chi ho | `Thao` [0:4] | ∅ |
| targeted-annotation-v1-01-81f1c9cec4b3 | bao hiem tra tien kham 850k | ∅ | `bao hi` [0:6] |
| targeted-annotation-v1-01-4d06a6abdd9b | điện máy xanh hoàn tiền nồi cơm lỗi 890k | `điện máy xanh` [0:13] | `iện máy xanh` [1:13] |

### multilingual-minilm-l12-h384 — `experiments/baseline-v1/runs/multilingual-minilm-l12-h384/lr5e-05-seed3`: 15 type error(s), 13 span error(s) of 105

| id | text | gold | pred |
|---|---|---|---|
| baseline-01-2123a362f627 | ban may anh cu 6tr5 | income | expense |
| baseline-01-ee03dc138256 | bố cho tiền tiêu 1 triệu | income | expense |
| baseline-01-237a04da0464 | gui xe benh vien 5k | expense | transfer |
| baseline-01-cd356c06d417 | nạp ví grab 150k | transfer | expense |
| baseline-01-e8c066dd0681 | dong bao hiem nhan tho 1tr2 | expense | income |
| baseline-01-82f29c0ccb74 | vay tiêu dùng trả kỳ này 3tr4 | repayment_out | borrow |
| baseline-01-dbf5b4023040 | trúng vé số 200k | income | transfer |
| baseline-01-80e41e000438 | phí gym tháng 500k | expense | income |
| baseline-01-d74a957149fd | lì xì cháu 200k | expense | income |
| targeted-annotation-v1-01-6b71966f3f06 | Thắng vay 5 củ, hẹn t10 trả | lend | repayment_in |
| targeted-annotation-v1-01-a07b6b6eb45c | bảo việt bồi thường viện phí 2tr6 | refund | repayment_in |
| targeted-annotation-v1-01-7d1037648087 | roommate tra tien nha minh dong truoc 1tr5 | repayment_in | refund |
| targeted-annotation-v1-01-6bd81e618b99 | co Thuy cho muon 1tr | borrow | lend |
| targeted-annotation-v1-01-81f1c9cec4b3 | bao hiem tra tien kham 850k | refund | repayment_in |
| targeted-annotation-v1-01-9d98b78854de | đòi được nợ thằng Lâm 400k | repayment_in | borrow |

| id | text | gold span | pred span |
|---|---|---|---|
| baseline-01-90a34dbd7ef8 | anh Long mượn 3 triệu sửa xe | `Long` [4:8] | `anh` [0:3] |
| baseline-01-87fbf0233b5a | ứng trước của sếp 2tr | `sếp` [14:17] | `của` [10:13] |
| baseline-01-cd356c06d417 | nạp ví grab 150k | ∅ | `grab` [7:11] |
| baseline-01-e8c066dd0681 | dong bao hiem nhan tho 1tr2 | ∅ | `dong bao` [0:8] |
| baseline-01-82f29c0ccb74 | vay tiêu dùng trả kỳ này 3tr4 | ∅ | `tiêu dùng` [4:13] |
| baseline-01-ba4d9f58a5ff | nuoc mia 12k | ∅ | `mia` [5:8] |
| baseline-01-ebf8bca50f8d | tien chay grab hom nay 450k | `grab` [10:14] | ∅ |
| baseline-01-8333d3354e18 | cuoc internet viettel 180k | `viettel` [14:21] | ∅ |
| baseline-01-30cd0e980666 | bách hoá xanh rau thịt 187k | `bách hoá xanh` [0:13] | ∅ |
| baseline-01-e2f16245d7af | vay chú Hải 20tr mua xe | `Hải` [8:11] | `chú Hải` [4:11] |
| baseline-01-30d790f0205f | mừng thọ bà nội 1 triệu | `bà nội` [9:15] | `nội` [12:15] |
| baseline-01-83a8de1156c4 | mở sổ tiết kiệm agribank 15tr | ∅ | `agribank` [16:24] |
| targeted-annotation-v1-01-2cda1d910520 | vay ngân hàng mua nhà giải ngân 800tr | `ngân hàng` [4:13] | `hàng` [9:13] |

### Both/all models err (bamibert, multilingual-minilm-l12-h384; 105 ids in common, 0 unmatched)

Type errors in all models: 7

| id | text | gold | pred bamibert | pred multilingual-minilm-l12-h384 |
|---|---|---|---|---|
| baseline-01-82f29c0ccb74 | vay tiêu dùng trả kỳ này 3tr4 | repayment_out | borrow | borrow |
| baseline-01-d74a957149fd | lì xì cháu 200k | expense | income | income |
| baseline-01-dbf5b4023040 | trúng vé số 200k | income | repayment_out | transfer |
| baseline-01-e8c066dd0681 | dong bao hiem nhan tho 1tr2 | expense | transfer | income |
| baseline-01-ee03dc138256 | bố cho tiền tiêu 1 triệu | income | borrow | expense |
| targeted-annotation-v1-01-6bd81e618b99 | co Thuy cho muon 1tr | borrow | lend | lend |
| targeted-annotation-v1-01-81f1c9cec4b3 | bao hiem tra tien kham 850k | refund | repayment_in | repayment_in |

Span errors in all models: 4

| id | text | gold span | pred bamibert | pred multilingual-minilm-l12-h384 |
|---|---|---|---|---|
| baseline-01-30cd0e980666 | bách hoá xanh rau thịt 187k | `bách hoá xanh` [0:13] | ∅ | ∅ |
| baseline-01-82f29c0ccb74 | vay tiêu dùng trả kỳ này 3tr4 | ∅ | `tiêu dùng` [4:13] | `tiêu dùng` [4:13] |
| baseline-01-90a34dbd7ef8 | anh Long mượn 3 triệu sửa xe | `Long` [4:8] | `anh Long` [0:8] | `anh` [0:3] |
| baseline-01-e2f16245d7af | vay chú Hải 20tr mua xe | `Hải` [8:11] | `chú Hải` [4:11] | `chú Hải` [4:11] |

