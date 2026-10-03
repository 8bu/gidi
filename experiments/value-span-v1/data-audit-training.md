# Value-span data audit (training)

## Totals

- records: 1158 (excluding probe: 1077); by split: {'train': 785, 'validation': 149, 'test': 143, 'probe': 81}
- with a value span: 1133; null value span: 25
- states: {'rule': 1039, 'human': 106, 'masked_uncertain': 13}
- auto-proposed (rule) spans: 1039; human spans: 106; needing review / masked: 13
- multi-number notes (2+ numeric expressions): 167
- notes accented / unaccented: 666 / 492

Notes by category: explicit_unit 951, unaccented 492, compact_unit 169, multi_number 167, context_excluded 157, slang 74, separator_format 73, bare_number 37, currency_suffix 30, decimal_unit 28, long_digits 16, multiple_money_candidates 13, no_candidate 12, compound_amount 8, number_words 5, unusual_punctuation 1

## Coverage of required amount forms

| form | req | total | train | val | test | probe | accented notes | unaccented notes | span has diacritics | status |
|---|---|---|---|---|---|---|---|---|---|---|
| `100` | yes | 37 | 26 | 5 | 6 | 0 | 19 | 18 | 0 | covered in train |
| `50k` | yes | 415 | 283 | 53 | 54 | 25 | 224 | 191 | 0 | covered in train |
| `50K` | extra | 7 | 4 | 2 | 1 | 0 | 3 | 4 | 0 | covered in train |
| `50 K` | yes | 14 | 12 | 2 | 0 | 0 | 7 | 7 | 0 | covered in train |
| `1tr` | yes | 211 | 131 | 29 | 21 | 30 | 129 | 82 | 0 | covered in train |
| `1tr5` | yes | 168 | 111 | 20 | 20 | 17 | 94 | 74 | 0 | covered in train |
| `1.5tr` | yes | 5 | 4 | 0 | 1 | 0 | 3 | 2 | 0 | covered in train |
| `1,5tr` | yes | 23 | 16 | 5 | 2 | 0 | 11 | 12 | 0 | covered in train |
| `2 triệu` | yes | 49 | 29 | 4 | 10 | 6 | 43 | 6 | 43 | covered in train |
| `2 củ` | yes | 32 | 23 | 2 | 4 | 3 | 20 | 12 | 20 | covered in train |
| `5 xị` | yes | 11 | 9 | 0 | 2 | 0 | 4 | 7 | 4 | covered in train |
| `5 chai` | yes | 11 | 8 | 0 | 3 | 0 | 9 | 2 | 0 | covered in train |
| `5 lít` | yes | 9 | 8 | 0 | 1 | 0 | 4 | 5 | 4 | covered in train |
| `500 nghìn` | yes | 18 | 15 | 2 | 1 | 0 | 13 | 5 | 13 | covered in train |
| `500 ngàn` | yes | 14 | 7 | 5 | 2 | 0 | 6 | 8 | 6 | covered in train |
| `1.500.000` | yes | 33 | 25 | 6 | 2 | 0 | 24 | 9 | 0 | covered in train |
| `1500000` | yes | 15 | 11 | 2 | 2 | 0 | 7 | 8 | 0 | covered in train |
| `80.000đ` | extra | 25 | 17 | 4 | 4 | 0 | 17 | 8 | 13 | covered in train |
| `12,500,000` | extra | 11 | 9 | 1 | 1 | 0 | 6 | 5 | 0 | covered in train |
| `2 tỷ` | extra | 3 | 2 | 1 | 0 | 0 | 2 | 1 | 2 | covered in train |
| `2 triệu rưỡi` | extra | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | ABSENT |
| `1 triệu 2` | extra | 3 | 2 | 0 | 1 | 0 | 0 | 3 | 0 | covered in train |

## Surface patterns (digits masked)

| pattern | all | accented notes | unaccented notes | train | val | test | probe |
|---|---|---|---|---|---|---|---|
| `<num>k` | 415 | 224 | 191 | 283 | 53 | 54 | 25 |
| `<num>tr` | 211 | 129 | 82 | 131 | 29 | 21 | 30 |
| `<num>tr<num>` | 168 | 94 | 74 | 111 | 20 | 20 | 17 |
| `<num> trieu` | 49 | 43 | 6 | 29 | 4 | 10 | 6 |
| `<num>` | 37 | 19 | 18 | 26 | 5 | 6 | 0 |
| `<dot3>` | 33 | 24 | 9 | 25 | 6 | 2 | 0 |
| `<num> cu` | 32 | 20 | 12 | 23 | 2 | 4 | 3 |
| `<dot3>đ` | 25 | 17 | 8 | 17 | 4 | 4 | 0 |
| `<dec,>tr` | 23 | 11 | 12 | 16 | 5 | 2 | 0 |
| `<num> nghin` | 18 | 13 | 5 | 15 | 2 | 1 | 0 |
| `<long>` | 15 | 7 | 8 | 11 | 2 | 2 | 0 |
| `<num> K` | 14 | 7 | 7 | 12 | 2 | 0 | 0 |
| `<num> ngan` | 14 | 6 | 8 | 7 | 5 | 2 | 0 |
| `<com3>` | 11 | 6 | 5 | 9 | 1 | 1 | 0 |
| `<num> chai` | 11 | 9 | 2 | 8 | 0 | 3 | 0 |
| `<num> xi` | 11 | 4 | 7 | 9 | 0 | 2 | 0 |
| `<num> lit` | 9 | 4 | 5 | 8 | 0 | 1 | 0 |
| `<num>K` | 7 | 3 | 4 | 4 | 2 | 1 | 0 |
| `<dec.>tr` | 5 | 3 | 2 | 4 | 0 | 1 | 0 |
| `<num> trieu <num>` | 3 | 0 | 3 | 2 | 0 | 1 | 0 |
| `<num> ty` | 3 | 2 | 1 | 2 | 1 | 0 | 0 |
| `<dot3> đ` | 3 | 2 | 1 | 2 | 0 | 1 | 0 |
| `<num> cu ruoi` | 3 | 2 | 1 | 3 | 0 | 0 | 0 |
| `<num> ty <num>` | 3 | 0 | 3 | 3 | 0 | 0 | 0 |
| `<words> trieu` | 2 | 2 | 0 | 1 | 1 | 0 | 0 |
| `<num> cu <num>` | 2 | 1 | 1 | 2 | 0 | 0 | 0 |
| `nua cu` | 2 | 0 | 2 | 2 | 0 | 0 | 0 |
| `<words> nghin` | 1 | 0 | 1 | 1 | 0 | 0 | 0 |
| `<long>đ` | 1 | 1 | 0 | 1 | 0 | 0 | 0 |
| `<num> chai ruoi` | 1 | 1 | 0 | 1 | 0 | 0 | 0 |
| `<dot3> VND` | 1 | 0 | 1 | 1 | 0 | 0 | 0 |

## Seen / unseen patterns versus train

- **validation**: 144 spans; pattern seen in train 144, unseen 0; exact span seen in train 106, unseen 38
- **test**: 139 spans; pattern seen in train 139, unseen 0; exact span seen in train 99, unseen 40
- **probe**: 81 spans; pattern seen in train 81, unseen 0; exact span seen in train 75, unseen 6

## Span length

- characters: {"n": 1133, "mean": 4.44, "median": 4, "p95": 8, "max": 14, "histogram": {"2": 8, "3": 306, "4": 546, "5": 65, "6": 35, "7": 81, "8": 35, "9": 45, "10": 7, "11": 2, "12": 1, "13": 1, "14": 1}}
- v1 tokens per span: {"n": 1133, "mean": 2.39, "median": 2, "p95": 4, "max": 7, "histogram": {"1": 35, "2": 749, "3": 260, "4": 56, "5": 28, "6": 4, "7": 1}}
- mean tokens by pattern: {'<com3>': 3.73, '<dec,>tr': 4.0, '<dec.>tr': 4.0, '<dot3>': 4.45, '<dot3> VND': 5.0, '<dot3> đ': 5.33, '<dot3>đ': 3.92, '<long>': 3.0, '<long>đ': 3.0, '<num>': 1.05, '<num> K': 2.0, '<num> chai': 2.0, '<num> chai ruoi': 4.0, '<num> cu': 2.0, '<num> cu <num>': 3.0, '<num> cu ruoi': 4.0, '<num> lit': 2.56, '<num> ngan': 2.57, '<num> nghin': 2.28, '<num> trieu': 2.12, '<num> trieu <num>': 4.0, '<num> ty': 2.0, '<num> ty <num>': 3.0, '<num> xi': 2.36, '<num>K': 2.0, '<num>k': 2.05, '<num>tr': 2.0, '<num>tr<num>': 3.01, '<words> nghin': 5.0, '<words> trieu': 2.0, 'nua cu': 3.0}
- notes truncated at 32 tokens: 0; spans losing tokens to truncation: 0; spans whose tokens straddle an edge: 0
