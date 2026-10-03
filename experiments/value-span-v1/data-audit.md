# Value-span data audit (pre-review)

> **PRE-REVIEW.** These numbers describe the rule proposer's output (`gidi.annotation.value_span`), not human-verified labels. Re-run on built training data (`--mode training`) after the Quet value pass.

## Totals

- records: 909 (excluding probe: 828); by split: {'train': 613, 'validation': 110, 'test': 105, 'probe': 81}
- with a value span: 906; null value span: 3
- states: {'auto_proposed': 782, 'proposed_needs_review': 124, 'multiple_candidates_needs_review': 1, 'no_candidate_needs_review': 2}
- auto-proposed (rule) spans: 782; human spans: 0; needing review / masked: 127
- multi-number notes (2+ numeric expressions): 99
- notes accented / unaccented: 531 / 378
- review queue: 46 records {'must_review': 14, 'hard_sample': 17, 'clean_audit': 15}; flagged by the proposer, primary reasons: {'multi_number': 98, 'bare_number': 2, 'multiple_money_candidates': 1, 'slang': 24, 'no_candidate': 2}

Notes by category: explicit_unit 842, unaccented 378, compact_unit 147, context_excluded 102, multi_number 99, separator_format 41, slang 25, decimal_unit 17, currency_suffix 15, number_words 3, bare_number 2, no_candidate 2, multiple_money_candidates 1

## Coverage of required amount forms

| form | req | total | train | val | test | probe | accented notes | unaccented notes | span has diacritics | status |
|---|---|---|---|---|---|---|---|---|---|---|
| `100` | yes | 2 | 2 | 0 | 0 | 0 | 1 | 1 | 0 | covered in train |
| `50k` | yes | 405 | 277 | 50 | 53 | 25 | 216 | 189 | 0 | covered in train |
| `50K` | extra | 7 | 4 | 2 | 1 | 0 | 3 | 4 | 0 | covered in train |
| `50 K` | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | ABSENT |
| `1tr` | yes | 203 | 126 | 28 | 19 | 30 | 123 | 80 | 0 | covered in train |
| `1tr5` | yes | 147 | 100 | 15 | 15 | 17 | 84 | 63 | 0 | covered in train |
| `1.5tr` | yes | 2 | 2 | 0 | 0 | 0 | 2 | 0 | 0 | covered in train |
| `1,5tr` | yes | 15 | 10 | 3 | 2 | 0 | 9 | 6 | 0 | covered in train |
| `2 triệu` | yes | 42 | 26 | 2 | 8 | 6 | 36 | 6 | 36 | covered in train |
| `2 củ` | yes | 22 | 15 | 1 | 3 | 3 | 11 | 11 | 11 | covered in train |
| `5 xị` | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | ABSENT |
| `5 chai` | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | ABSENT |
| `5 lít` | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | ABSENT |
| `500 nghìn` | yes | 17 | 15 | 1 | 1 | 0 | 12 | 5 | 12 | covered in train |
| `500 ngàn` | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | ABSENT |
| `1.500.000` | yes | 15 | 11 | 4 | 0 | 0 | 11 | 4 | 0 | covered in train |
| `1500000` | yes | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | ABSENT |
| `80.000đ` | extra | 15 | 11 | 2 | 2 | 0 | 14 | 1 | 10 | covered in train |
| `12,500,000` | extra | 11 | 9 | 1 | 1 | 0 | 6 | 5 | 0 | covered in train |
| `2 tỷ` | extra | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | ABSENT |
| `2 triệu rưỡi` | extra | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | ABSENT |
| `1 triệu 2` | extra | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | ABSENT |

## Surface patterns (digits masked)

| pattern | all | accented notes | unaccented notes | train | val | test | probe |
|---|---|---|---|---|---|---|---|
| `<num>k` | 405 | 216 | 189 | 277 | 50 | 53 | 25 |
| `<num>tr` | 203 | 123 | 80 | 126 | 28 | 19 | 30 |
| `<num>tr<num>` | 147 | 84 | 63 | 100 | 15 | 15 | 17 |
| `<num> trieu` | 42 | 36 | 6 | 26 | 2 | 8 | 6 |
| `<num> cu` | 22 | 11 | 11 | 15 | 1 | 3 | 3 |
| `<num> nghin` | 17 | 12 | 5 | 15 | 1 | 1 | 0 |
| `<dec,>tr` | 15 | 9 | 6 | 10 | 3 | 2 | 0 |
| `<dot3>đ` | 15 | 14 | 1 | 11 | 2 | 2 | 0 |
| `<dot3>` | 15 | 11 | 4 | 11 | 4 | 0 | 0 |
| `<com3>` | 11 | 6 | 5 | 9 | 1 | 1 | 0 |
| `<num>K` | 7 | 3 | 4 | 4 | 2 | 1 | 0 |
| `<num>` | 2 | 1 | 1 | 2 | 0 | 0 | 0 |
| `<words> trieu` | 2 | 2 | 0 | 1 | 1 | 0 | 0 |
| `<dec.>tr` | 2 | 2 | 0 | 2 | 0 | 0 | 0 |
| `<words> nghin` | 1 | 0 | 1 | 1 | 0 | 0 | 0 |

## Seen / unseen patterns versus train

- **validation**: 110 spans; pattern seen in train 110, unseen 0; exact span seen in train 90, unseen 20
- **test**: 105 spans; pattern seen in train 105, unseen 0; exact span seen in train 82, unseen 23
- **probe**: 81 spans; pattern seen in train 81, unseen 0; exact span seen in train 75, unseen 6

## Span length

- characters: {"n": 906, "mean": 4.17, "median": 4.0, "p95": 7, "max": 14, "histogram": {"2": 1, "3": 270, "4": 493, "5": 30, "6": 14, "7": 58, "8": 14, "9": 22, "10": 3, "14": 1}}
- v1 tokens per span: {"n": 906, "mean": 2.31, "median": 2.0, "p95": 3, "max": 5, "histogram": {"1": 2, "2": 667, "3": 197, "4": 30, "5": 10}}
- mean tokens by pattern: {'<com3>': 3.73, '<dec,>tr': 4.0, '<dec.>tr': 4.0, '<dot3>': 3.73, '<dot3>đ': 3.67, '<num>': 1.0, '<num> cu': 2.0, '<num> nghin': 2.29, '<num> trieu': 2.14, '<num>K': 2.0, '<num>k': 2.05, '<num>tr': 2.0, '<num>tr<num>': 3.01, '<words> nghin': 5.0, '<words> trieu': 2.0}
- notes truncated at 32 tokens: 0; spans losing tokens to truncation: 0; spans whose tokens straddle an edge: 0
