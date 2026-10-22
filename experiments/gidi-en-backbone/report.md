# gidi-en backbone and tokenizer

Question: which encoder, tokenizer and compression recipe gives gidi-en an int8 ONNX model of about
30 MB or less that runs in ONNX Runtime Web, the way BamiBERT gave `gidi-finance-v4` 28.7 MB?

No model was trained. Nothing here measures quality. Quality statements are priors, marked as such.
All inputs are my own 63-note audit sample (`audit-notes.jsonl`, tokenizer audit only) and, for
the vocabulary check, a snapshot of 700 raw generated notes (`proxy-train-notes.jsonl`, no labels).

## Recommendation

**`roberta-base` (MIT), cased byte-level BPE, compressed exactly like the Vietnamese student:**
4 layers initialised from layers 2/5/8/11, position table 34 rows, vocabulary pruned to 9,000
rows, FFN 3072 -> 2048 (`combined` criterion). Measured int8 ONNX: **29,166,948 bytes (29.17 MB)**,
FP32 116.1 MB, 29.0M parameters. Compare `gidi-finance-v3` (same shape, 8,000 vocab rows):
28,658,533 bytes.

Why this one:

- It is the same architecture as BamiBERT (RoBERTa). `student-4x768`, `copy_pretrained_encoder`,
  `truncate_positions`, `prune_vocab`, `prune_ffn`, `export_onnx.py` and the release flow work as is.
  Only the vocabulary builder and the tokenizer file needed new code (`scripts/build_en_student.py`).
- The int8 graph has the **identical op set** as the shipped Vietnamese model
  (`models/compression-v3-onnx/.../model.int8.onnx`: 33 op types, set difference empty), so ORT Web
  support is the same as for the model that already runs in the browser. (I did not run a browser.)
- Cased: the counterparty span ("lent Sam $20", "Dave repaid 250") carries a capitalisation cue
  that uncased WordPiece models throw away. Byte-level BPE has no `[UNK]`, so typos, `$`, `£`,
  emoji and mixed case always round-trip (100% in the audit; uncased BERT: 38%, it lowercases and
  re-spaces). Weakness: ALL-CAPS and unusual case fragment more ("PAID BACK MARCUS"); train with a
  case augmentation.
- The tokenizer needs **no web change**. I swapped roberta's GPT-2 regex pre-tokenizer for the
  Vietnamese bundle's `Split` + byte-level pre-tokenizer (already ported in
  `playground/src/runtime/tokenizer.ts`). On 763 notes (audit + proxy) the ids are identical to the
  GPT-2 regex ones, 0 differences (`verify.json`).

Runner-up: `microsoft/MiniLM-L12-H384-uncased` (MIT). Full 12 layers, no layer skipping, 24.5 MB
int8 with an 8,000-row vocab at stock FFN, compute per token about equal to the recommendation
(12 x 1.8M vs 4 x 5.5M MACs). It loses on cased input, needs WordPiece vocab pruning and a WordPiece
runtime port in TypeScript (neither exists), and the Vietnamese compression code does not apply.
If silver labels exist and the budget allows, it is the only challenger worth an A/B on dev agreement.

## Candidates

Tokenizer columns: `scripts/audit_tokenizers.py` on `audit-notes.jsonl` (63 notes, 298 words,
special tokens not counted; `tokenizer-audit.txt/json`). Round trip = decode(encode(text)) equals
the text. The audit's amount/slang columns are Vietnamese-specific and are ignored.
Size columns: `measure_candidates.py` (`candidates.json`), RANDOM weights of the hub architecture,
dynamo ONNX export + dynamic int8: **full** = stock config; **pruned** = vocab 8,000, 34 position
rows, all layers, stock FFN; per layer = int8 MB per transformer layer.

| Candidate | License | Params | Tok/note (p95) | Words split | Round trip | int8 full | int8 pruned, all layers | MB / layer | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| **roberta-base** | MIT | 124M | 6.87 (11.9) | 39.9% | 100% | 124.2 MB | 91.4 MB | 7.1 | **Chosen.** 4 layers + FFN 2048 + vocab 9,000 = 29.17 MB |
| distilroberta-base | Apache-2.0 | 82M | 6.87 (11.9) | 39.9% | 100% | 81.6 MB | 48.8 MB | 7.1 | Viable, same tokenizer and code path. 6 layers still need cutting to 4, so no size win; keep as an init ablation only |
| bert-base-uncased | Apache-2.0 | 109M | 6.46 (10.0) | 20.5% | 38% | 109.1 MB | 91.4 MB | 7.1 | No: uncased, WordPiece prune and runtime port missing |
| distilbert-base-uncased | Apache-2.0 | 66M | 6.46 (10.0) | 20.5% | 38% | 66.5 MB | 48.8 MB | 7.1 | No: same reasons, same 6-layer problem |
| microsoft/MiniLM-L12-H384-uncased | MIT | 33M | 6.46 (10.0) | 20.5% | 38% | 33.4 MB | 24.5 MB | 1.79 | Runner-up (see above) |
| google/electra-small-discriminator | Apache-2.0 | 14M | 6.46 (10.0) | 20.5% | 38% | 13.6 MB | 10.6 MB | 0.80 | No: 256 wide, uncased; far below the budget, so capacity is wasted |
| google/bert_uncased_L-4_H-512_A-8 | Apache-2.0 | 29M | 6.46 (10.0) | 20.5% | 38% | 28.6 MB | 16.8 MB | n/a | No: uncased, narrower than the chosen model |
| microsoft/deberta-v3-small | MIT | 141M | 6.14 (10.9) | 16.8% | 100% | 163.8 MB | 71.6 MB | 10.6 | No: disentangled attention costs 10.6 MB/layer, 4 layers would be about 50 MB; extra ops (`GatherElements`, `Tile`, `Log`, `Sign`, `Ceil`, `Less`, ...); SentencePiece pruning missing |

Notes:
- Token counts are close (6.1-6.9). Tokenizer fragmentation does not separate the candidates;
  cased-ness, UNK safety, size and code reuse do. roberta-base splits 40% of words against 20% for
  WordPiece, mostly names, `venmo`-style brands and slang; the pruned vocabulary below fixes most
  of that for words that appear in training.
- Licenses are the hub `license:` tags. All are permissive.
- `bert-base-uncased`, `distilbert`, `electra-small`, `MiniLM` and `bert_uncased_L-4` share one
  WordPiece vocabulary (30,522), hence identical tokenizer numbers.
- ORT Web ops: BERT-family graphs use only ops in the Vietnamese model's set. roberta adds
  `CumSum`, `Equal`, `Not` for position ids, which the shipped Vietnamese model already contains.
- Expected quality for token classification (prior, not measured): MiniLM-L12 and roberta-base
  are both strong; layer skipping to 4 layers costs some quality, which the Vietnamese pipeline
  accepted. Whether it holds for English is only knowable after training.

## Vocabulary-pruning plan

Keep **9,000 rows** out of 50,265 (`--max-rows 9000`). The Vietnamese B-rank-8,000 kept 8,000;
English needs a little more because roberta's frequency order comes from web text, not from
finance notes. Size budget: each 1,000 rows cost 0.77 MB int8 (768 wide); 9,000 rows give 29.17 MB,
10,000 rows would be about 29.9 MB, so 9,000 keeps 0.8 MB of margin under 30 MB.

Selection uses only the tokenizer and the gidi-en **train** split:

1. Mandatory: special tokens and all 256 byte tokens (no input can hit `<unk>`).
2. Every token used when tokenizing the **gidi-en train notes** (`--train-notes`).
3. Every English-compatible token (printable ASCII, plus `' " " - - ... £ € ° •`; partial UTF-8
   tokens dropped) with merge rank below a limit. The limit is the largest one that keeps the
   closed set within 9,000 rows (binary search).
4. Close the set under BPE merge parents (existing `closure`), so pruned segmentation stays
   exact for kept words.

No dev, test or probe note is read. The same pruned tokenizer and `vocab_map.json` ship in the
bundle.

Check of the selection rule (`vocab_policy_probe.py`, `vocab_policy_probe.json`). Selection used
only roberta ranks and the 700 proxy notes; the 63 audit notes were written independently and are
only measured:

| Policy (9,000 rows) | Audit notes tokenized identically to full vocab | Tokens / note (full vocab 6.87) |
|---|---|---|
| plain rank only, no train notes | 9 / 63 | 8.68 |
| **plain rank + train closure (chosen)** | **43 / 63** | **7.33** |
| rank over space/case variants, no train notes | 2 / 63 | 9.40 |
| rank over space/case variants + train closure | 38 / 63 | 7.51 |

Findings: rank alone is a poor proxy for finance notes (`paid`, `lent`, `uber` and amounts like
` 800` have rank 10,000-30,000). The train closure carries the domain, and 700 unlabeled notes
already move 9/63 to 43/63. Counting space/case variants was tried to help sentence-initial
words and was worse at equal rows, so it is off (`--variants` remains for reproduction only).
Words missing from the pruned vocabulary fall back to smaller pieces and bytes, never to `<unk>`.
With the real train split (thousands of notes) the closure will be larger; if it ever exceeds
the budget the script fails instead of dropping train tokens.

## Feasibility build (proxy data, random heads)

`scripts/build_en_student.py` on the proxy notes (stand-in for the train split; heads are fresh
and untrained, so any output value is meaningless):

- pretrained roberta-base layers 2/5/8/11 copied by `copy_pretrained_encoder`; position rows 34;
  vocab 9,000 (rank limit 8,249, 1,282 distinct token ids from the proxy notes); FFN K=2048 chosen
  by the `combined` score on the proxy notes.
- `scripts/export_onnx.py`: dynamo exporter, FP32 vs PyTorch max abs diff 2e-6. FP32 116.1 MB ->
  **int8 29,166,948 bytes (29.17 MB)**.
- Plain `onnxruntime` 1.30 load test (`verify_student.py`, `verify.json`): loads, batch of 63 and
  batch of 1 at natural length both run, outputs finite, shapes `[B, types]` and `[B, T, 3]`.
- Latency, batch 1, CPU of this Mac: int8 p50 1.2 ms with 1 thread, 0.8 ms default threads.
  Native CPU only, not a browser number.
- int8 vs FP32 agreement (type 0.90, tag 0.96) is on random heads and says nothing about trained
  models; it must be re-measured on the trained student (the Vietnamese pipeline does this in
  `evaluate_onnx.py`).

## Things this report does not show

- Quality. No labels exist, nothing was trained. The case for roberta-base over MiniLM-L12 is
  reuse, cased-ness and zero web change, not a measured accuracy gap.
- Browser behavior. Support rests on op-set equality with the shipped Vietnamese model.
- The vocabulary and FFN map for the real model: re-run the recipe with the train split.
- Head sizes: the saved checkpoint uses the `annotation-v1` heads (8 types, 3 tags); they cost a
  few KB and change with the English taxonomy.

## Reproduce

```bash
# 1. tokenizer audit
uv run python scripts/audit_tokenizers.py --corpus experiments/gidi-en-backbone/audit-notes.jsonl \
  --tokenizer roberta-base bert-base-uncased google/electra-small-discriminator \
  microsoft/MiniLM-L12-H384-uncased microsoft/deberta-v3-small google/bert_uncased_L-4_H-512_A-8 \
  --out experiments/gidi-en-backbone/tokenizer-audit.json --overwrite

# 2. candidate sizes / op sets (random weights, ~10 min)
uv run python experiments/gidi-en-backbone/measure_candidates.py

# 3. vocabulary policy check
uv run python experiments/gidi-en-backbone/vocab_policy_probe.py

# 4. build the student (feasibility run; for the real one pass the gidi-en TRAIN split to both flags)
uv run python scripts/build_en_student.py \
  --out models/gidi-en-backbone/feasibility-roberta-4x768 \
  --train-notes experiments/gidi-en-backbone/proxy-train-notes.jsonl \
  --max-rows 9000 --ffn-k 2048

# 5. int8 ONNX + load test
uv run python scripts/export_onnx.py --checkpoint models/gidi-en-backbone/feasibility-roberta-4x768 \
  --out models/gidi-en-backbone/feasibility-roberta-4x768-onnx \
  --report experiments/gidi-en-backbone/export-feasibility.json \
  --notes experiments/gidi-en-backbone/audit-notes.jsonl --num-notes 60 --overwrite
uv run python experiments/gidi-en-backbone/verify_student.py
```

For the real model: `scripts/build_en_student.py --train-notes <gidi-en train.jsonl>`
(also used for FFN calibration), then train with the existing student training code on the
resulting checkpoint (`student-4x768` shape, supervised recipe: encoder lr 5e-5, head lr 1e-3,
40 epochs as in compression-v3), then `export_onnx.py` and the release flow in `docs/releasing.md`.
`StudentTrainConfig` already takes `tokenizer_name`, `vocab_spec` and `ffn_map`, but its data path
(teacher-target cache, `annotation-v1` splits) is Vietnamese; wiring gidi-en training to it was not
checked or done here.

## Files

`audit-notes.jsonl`, `tokenizer-audit.{txt,json}`, `measure_candidates.py`, `candidates.json`,
`measure_candidates.log`, `proxy-train-notes.jsonl` (snapshot of `generated/en-baseline-01-us.jsonl`,
700 lines), `vocab_policy_probe.{py,json}`, `verify_student.py`, `verify.json`,
`export-feasibility.json`, `build-feasibility.log`; recipe code in `scripts/build_en_student.py`.
Checkpoint and ONNX files are under `models/gidi-en-backbone/` (untracked, feasibility only).
