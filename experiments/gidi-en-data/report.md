# gidi-en data: corpus, splits, silver labels

All labels are **silver** (`label_provenance: silver-claude-clef`): no human labels exist for
English. Every accuracy below is agreement with Claude+Clef labels, not ground truth. Protocol
(written before any student): `protocol.json`. Contract: `configs/annotation-en-v1.yaml`,
`docs/annotation-en-v1.md`.

## Corpus

2,700 generated notes (`generated/en-baseline-01-{us,uk,india,singapore,mixed}.jsonl`) ->
`scripts/build_en_splits.py dedupe` (normalised exact match with amounts masked, then
SequenceMatcher >= 0.90, `gidi.annotation.leakage.NEAR_DUP_THRESHOLD`): 153 exact + 121 near
duplicates dropped -> **2,426 survive**, ingested with `scripts/ingest_generated_corpus.py` into
`corpus/raw/en-baseline-01.jsonl` (`validate_corpus.py`: 0 errors).

## Splits (`datasets/annotation-en-v1/splits/`, `manifest.json`)

Deterministic: order by `sha256("annotation-en-v1-split-01:" + id)`; test 300, dev 200, seed 300,
then train. A note is dropped when it matches a note already in another split by skeleton or
SequenceMatcher >= 0.85 (164 dropped), so train never leaks into test/dev/seed. Placed: test 300,
dev 200, seed 300, **train 1,462** (2,262 of 2,426).

| status / type | test | dev | seed | train (silver) |
|---|---|---|---|---|
| expense | 94 | 63 | 90 | 477 |
| income | 28 | 28 | 42 | 181 |
| borrow | 27 | 13 | 17 | 76 |
| lend | 36 | 18 | 34 | 185 |
| repayment_in | 22 | 20 | 28 | 173 |
| repayment_out | 24 | 12 | 20 | 110 |
| transfer | 44 | 27 | 34 | 168 |
| refund | 15 | 8 | 15 | 92 |
| uncertain | 6 | 6 | 13 | 0 |
| skipped | 4 | 5 | 7 | 0 |
| **notes** | **300** | **200** | **300** | **1,462** |

Every type appears in test and dev. Test `consensus` subset (primary acceptance set): 214
complete notes (expense 54, transfer 40, lend 29, income 19, repayment_in 20, repayment_out 20,
borrow 18, refund 14).

## Labels for test / dev / seed (800 notes)

Two independent labelers on the same notes: (a) Gemini 3.8 flash via `gemgen` (two 400-note
files, brief `work/gem-prompt.md`), (b) Clef zero-shot (`scripts/clef_teacher.py`, `lang="en"`,
type definitions and R1-R4 from the config, no examples; `work/clef-zero.jsonl`). Agreement on
type and target text -> `consensus`; anything else (including Gemini `uncertain`/`skipped`) was
adjudicated by Claude against the guide -> `claude-adjudicated` (`work/adjudicated.jsonl`, 239
notes: 198 complete, 25 uncertain, 16 skipped). Gemini matched the adjudicated label in 199 of
239, Clef zero-shot in 10, neither in 28 (mostly uncertain/skipped).

First Gemini pass read the guide's `uber home $18 -> null` row literally and left every merchant
without a target (consensus only 460). The inherited v3/v4 rule (named merchants/services/
platforms are targets; the config's own target comment) contradicts that one row, so a
clarification was added to the brief and Gemini re-labelled from scratch (first run kept in
`work/gem-v1/`). **The guide itself should be amended** (see `protocol.json`, `contract`).

| Gemini type | n | type agree | target agree | consensus |
|---|---|---|---|---|
| expense | 245 | 230 (0.94) | 166 (0.68) | 155 |
| income | 102 | 89 (0.87) | 82 (0.80) | 73 |
| borrow | 58 | 39 (0.67) | 54 (0.93) | 36 |
| lend | 86 | 78 (0.91) | 80 (0.93) | 73 |
| repayment_in | 71 | 57 (0.80) | 67 (0.94) | 55 |
| repayment_out | 55 | 52 (0.95) | 46 (0.84) | 44 |
| transfer | 106 | 102 (0.96) | 90 (0.85) | 90 |
| refund | 37 | 36 (0.97) | 36 (0.97) | 35 |
| uncertain/skipped (type null) | 40 | - | 34 | 0 |
| **all** | **800** | **0.854** (kappa 0.824) | **0.819** (0.741 on the 557 notes where either gave a target) | **561 (70%)** |

Disagreements concentrate on borrow/lend/repayment direction (R1, R2: `X paid for my ...`,
`fronted me`, `IOU`), on merchant/role-noun targets for expenses, and on bare notes (`sam 20`).
Clef's target on `transfer` notes is often non-null; the final transfer label is always null.

Final files: `labels-{test,dev,seed}.jsonl` (validator-clean), `labels-*.full.jsonl` (text, label,
both raw labeler answers, source), `provenance.jsonl` (`source` per id), `queue.jsonl`.
`validate_annotations.py --config configs/annotation-en-v1.yaml`: 0 errors on all three.

## Silver labels for train (`silver-train.jsonl`, 1,462 notes)

`scripts/clef_teacher.py` (`lang="en"`; Vietnamese path unchanged), run by
`experiments/gidi-en-data/run_teacher.py` in omp `eval`: pass A = 16 nearest seed shots (char
n-gram TF-IDF kNN, pool = labelled seed notes only), pass B = 12 nearest, order reversed.
Fields: `type`, `target` (exact span), `type_prob`, `target_prob`, `consistent` (A and B agree on
type and target), `type_b`/`target_b`, `flags`, `provenance`. No judge errors; 0 spans failed the
substring check; 5 teacher targets on `transfer` notes dropped (`flags: transfer_target_dropped`).
Self-consistent 88.7%; type_prob >= 0.9 on 425, >= 0.7 on 908.

## Teacher accuracy on dev (pool = seed, 189 complete dev notes)

| subset | n | type | target exact | type+target |
|---|---|---|---|---|
| all complete | 189 | 0.915 | 0.894 | 0.825 |
| consensus-labelled | 132 | 0.970 | 0.977 | 0.947 |
| Claude-adjudicated | 57 | 0.789 | 0.702 | 0.544 |
| teacher self-consistent | 172 | 0.948 | 0.901 | 0.849 |
| teacher inconsistent | 17 | 0.588 | 0.824 | 0.588 |
| type_prob >= 0.9 | 57 | 1.000 | 0.930 | 0.930 |
| type_prob >= 0.7 | 124 | 1.000 | 0.935 | 0.935 |

Per type (type / joint): expense 0.937 / 0.778, income 0.929 / 0.786, borrow 0.538 / 0.538,
lend 1.0 / 1.0, repayment_in 0.95 / 0.90, repayment_out 0.833 / 0.833, transfer 0.963 / 0.889
(0.889 target is the null-transfer postprocess not applied here), refund 1.0 / 1.0
(`work/dev-teacher-accuracy.json`). Borrow is the weak type (borrow vs lend direction on
`fronted me`, `spotted me`). Dev was not used to tune the prompt; the test split was not scored.
Silver-train caveats: borrow labels are the least trustworthy; use `consistent` and `type_prob` to
filter or soft-weight.

## Acceptance criterion for the student (`protocol.json`)

Primary (214 consensus test notes): type macro-F1 >= 0.90, target exact match >= 0.85, joint >=
0.80, per-type F1 >= 0.75 for types with n >= 10. Secondary (290 complete test notes): type
macro-F1 >= 0.82, target exact >= 0.78. Both must pass.
