# gidi-en backbone bake-off: is another English encoder better than the roberta-base student?

**Decision by the pre-declared rule (`protocol.json`, committed before any build or training):
`microsoft/MiniLM-L12-H384-uncased` meets every condition. It is a `WIN-WITH-PORT`: the model side
passes, but it needs a new WordPiece tokenizer in the web runtime. `roberta-base` stays shipped
until that port exists and the owner accepts its cost.** Nothing was released, deployed, pushed to
HF or entered in the journal.

The owner's premise needs one correction: roberta-base is already an English-only encoder (BamiBERT
is the Vietnamese one). The measured question is whether a *different* English encoder gives a
better type / target / value student under the same ~30 MB int8 ORT-Web budget. For MiniLM the
answer, on silver labels, is yes. The gain comes from the whole student shape (12 layers x 384 wide
instead of 4 layers x 768 wide, layers 2/5/8/11 skipped from a 12-layer model), not from the
language specialisation.

All numbers are agreement with **silver** labels (regex + Clef + Claude), not human truth. Test
primary n=214, secondary n=290, dev n=189, slang probe n=357, Gen Z probe n=94, so 0.01 is 2-3
notes on the test sets.

## Setup (what was held fixed)

Unchanged from v3 (`experiments/gidi-en-student-v3/protocol.json`): data (2,355 training notes with
silver type/target/value labels), the `GidiValueModel` three-head model and loss, 40 epochs, batch
8, encoder lr 5e-5, head lr 1e-3, AdamW, warmup 0.1, weight decay 0.01, MPS, last epoch, the v3
export (`export_value_onnx.py`, opset 17, dynamic int8) and the v3 evaluation script
(`evaluate_en_value.py`, int8, snapped). Only the backbone changes, and its shape comes from a fixed
rule in `protocol.json` (`width_rule`): evenly spaced source layers, parameter budget
<= 29.2 M (the shipped student has 29.0 M and 29.17 MB int8), vocabulary rows and FFN width from
that budget, vocabulary and FFN calibration from the **train split only**. No width was chosen by
looking at a score.

| candidate | license | tokenizer | layers x width | FFN | vocab rows | params | int8 bytes |
|---|---|---|---|---|---|---|---|
| roberta-base (control, v3) | MIT | byte-level BPE | 4 (2/5/8/11) x 768 | 2048 | 9,000 | 29.00 M | 29,170,089 |
| distilroberta-base | Apache-2.0 | same BPE | 6 (all) x 768 | 832 | 9,000 | 28.82 M | 29,036,406 |
| electra-base-discriminator | Apache-2.0 | uncased WordPiece | 4 (2/5/8/11) x 768 | 2048 | 9,000 | 29.00 M | 29,169,805 |
| MiniLM-L12-H384-uncased | MIT | uncased WordPiece | 12 (all) x 384 | 1536 (stock) | 20,500 | 29.18 M | 29,499,600 |

Candidates tried and why others were dropped before training (details in `protocol.json`,
evidence in `feasibility.json`):

* **ModernBERT-base: infeasible, not forced.** A 4-layer student exports only at opset 18 (opset 17
  gives an invalid graph: `Split` with `num_outputs`) and adds `Abs, Cos, LessOrEqual, Neg, Sin,
  Split, Sub` (RoPE, GeGLU) that the shipped graph does not contain. It also has no
  `encoder.layer.N` layout for `copy_pretrained_encoder`.
* **bge-small-en-v1.5:** same 12x384 shape and WordPiece as MiniLM, but an embedding model tuned for
  retrieval; MiniLM is its pretrained base. No new information.
* **FinBERT (`ProsusAI`): cc-by-nc-sa-4.0**, non-commercial, not allowed. `finbert-tone` has no
  license tag. Both are sentiment fine-tunes of bert-base-uncased on 10-K style text, not personal
  notes.
* Plain BERT/DistilBERT/ELECTRA-small/bert_uncased_L-4 were rejected earlier
  (`experiments/gidi-en-backbone/report.md`).

Op set: the int8 graph of every trained run contains only op types of the shipped v3 int8 graph
(33 types) and is opset 17 (`aggregate.json` `ops`, checked on all 8 new runs plus the v3 run).
I did not run a browser; ORT Web support rests on that op-set equality, as for the shipped model.

## Results: mean over seeds (min-max), int8, snapped

Composite = mean of the five bold-faced columns (protocol `composite`). roberta-base has seeds
1-3 (seed 1 is the existing v3 run), each challenger seeds 1-2.

| candidate | seeds | **composite** | **sec. type F1** | **sec. target** | **slang joint** | **slang value** | **Gen Z joint** | prim. type F1 | prim. target |
|---|---|---|---|---|---|---|---|---|---|
| roberta-base (control) | 3 | 0.8471 (0.8453-0.8483) | 0.8826 (0.8765-0.8928) | 0.8379 (0.8207-0.8552) | 0.7563 (0.7311-0.7703) | 0.9823 (0.9776-0.9860) | 0.7766 (0.7766-0.7766) | 0.9389 (0.9316-0.9455) | 0.8925 (0.8879-0.8972) |
| distilroberta-base | 2 | 0.8537 (0.8437-0.8637) | 0.8976 (0.8917-0.9035) | 0.8517 (0.8483-0.8552) | 0.7353 (0.7283-0.7423) | 0.9860 (0.9804-0.9916) | 0.7979 (0.7447-0.8511) | 0.9407 (0.9315-0.9499) | 0.9019 (0.9019-0.9019) |
| electra-base | 2 | 0.8630 (0.8581-0.8678) | 0.8761 (0.8665-0.8857) | 0.8655 (0.8621-0.8690) | 0.7969 (0.7815-0.8123) | 0.9678 (0.9636-0.9720) | 0.8085 (0.8085-0.8085) | 0.9287 (0.9176-0.9398) | 0.9159 (0.9112-0.9206) |
| **MiniLM-L12-H384** | 2 | **0.8786 (0.8757-0.8814)** | 0.9066 (0.8904-0.9228) | 0.8948 (0.8897-0.9000) | 0.8095 (0.8039-0.8151) | 0.9734 (0.9608-0.9860) | 0.8085 (0.8085-0.8085) | 0.9499 (0.9423-0.9575) | 0.9393 (0.9393-0.9393) |

Several of these "min = max" cells are real coincidences (same score from different notes; for
example the Gen Z joint of all three roberta seeds is 73/94, with 8-10 notes differing between
seeds), not a copied result. Per run: `aggregate.json`, `tables.md`.

Other means (not gating): dev type F1 / target exact: roberta 0.837 / 0.818, distilroberta 0.842 /
0.836, electra 0.865 / 0.852, MiniLM 0.883 / 0.865. Secondary joint: 0.744 / 0.779 / 0.772 / 0.812.
Value exact on test secondary: 0.994 / 0.998 / 0.997 / 0.997. Gen Z target: 0.858 / 0.862 / 0.867 /
0.851.

## Decision per the pre-declared rule

| condition | distilroberta | electra-base | MiniLM-L12-H384 |
|---|---|---|---|
| composite mean >= roberta mean + 0.01 (0.8471 -> 0.8571) | 0.8537, delta +0.0066: **fail** | 0.8630, delta +0.0158: pass | 0.8786, delta +0.0314: pass |
| no mean regression > 0.01, primary type F1 | +0.0018 pass | **-0.0102: fail** | +0.0110 pass |
| no mean regression > 0.01, primary target exact | +0.0093 pass | +0.0234 pass | +0.0467 pass |
| int8 <= 30 MB (all runs) | 29.04 MB pass | 29.17 MB pass | 29.50 MB pass |
| op types within shipped set, opset 17 | pass | pass | pass |
| web tokenizer | existing | port | port |
| **verdict** | no | **no** (fails the primary-type guard by 0.0002 of the 0.01 allowance) | **WIN-WITH-PORT** |

Noise check (protocol `noise_flag_not_gating`): MiniLM is `separated`, its lowest seed composite
(0.8757) is above roberta's highest (0.8483); electra is separated too (0.8581 > 0.8483) but is
excluded by the primary-type guard. distilroberta's seeds overlap roberta's, so its +0.0066 is not
distinguishable from noise.

How strong is the MiniLM result? Reasons to trust it: the composite gap (+0.031) is about ten
times roberta's seed range on the composite (0.003), both MiniLM seeds beat every roberta seed on
secondary target (0.890-0.900 vs 0.821-0.855), slang joint (0.804-0.815 vs 0.731-0.770) and
primary target (both 0.939 vs 0.888-0.897), and the gain appears on the held-out dev set too
(type 0.883 vs 0.837). Reasons for care: two seeds, silver labels, small evaluation sets (0.0467 on
primary target is 10 notes), and MiniLM is the weakest of the three on slang value exact (0.973 vs
0.982, still far above the 0.88 bar). Slang-probe *type* F1 is about equal for every candidate
(0.90-0.92); the gains are in target spans and in the harder test notes.

## Web tokenizer cost

* **roberta / distilroberta**: none. distilroberta's `tokenizer.json` vocabulary and merges are
  byte-identical to roberta-base's (asserted in `build_candidate.py`), so the shipped byte-level BPE
  TypeScript path and the Python `tokenizers` bundle path work unchanged.
* **MiniLM / ELECTRA (WordPiece, uncased)**: Python needs nothing (`gidi.inference.tokenizer` is the
  Rust `tokenizers` library, which loads the saved `BertTokenizerFast` file). The web port
  (`playground/src/runtime/tokenizer.ts` is BPE only and throws on any other `model.type`) needs a
  new `WordPieceTokenizer` with the same `encode(cps)` contract. Components, from the HF
  `tokenizer.json`: BertNormalizer (clean control/whitespace, spaces around CJK, lowercase, NFD +
  drop combining marks) with offsets mapped back to code points of the NFC input; BertPreTokenizer
  (split on whitespace, every punctuation/symbol char its own word); greedy longest-match WordPiece
  with the `##` prefix and `[UNK]` for a whole word that does not decompose (100-char word cap);
  `[CLS] A [SEP]` and right truncation. My estimate, not measured: about 150-250 lines of TS,
  reusing `text.ts` and `unicode-tables.ts` (White_Space already there; punctuation and Mn can use
  JS `\p{P}` / `\p{Mn}` property escapes), plus a parity fixture of a few hundred notes dumped from
  `tokenizers` like `english.fixture.json` / `dump_web_parity_en.py`, plus bundle/manifest wiring
  and a decision about the loader's tokenizer class. Roughly one to two days including the parity
  tests, mostly the offset mapping through the normaliser. Offsets matter because the target and
  value spans are cut from character offsets.
* **What uncased costs**: the case cue ("lent Sam $20") is lost to the encoder, yet the MiniLM target
  exact is higher, not lower, so on this data the cue is not what limits roberta. `[UNK]` exists:
  the pruned vocabulary produces 2 `[UNK]` pieces on the 290 test notes and 11 on the 374 slang-probe
  notes (on the 1,462 train notes the pruned and the full 30,522-row vocabulary give identical
  pieces, so those `[UNK]`s come from characters BERT's vocabulary lacks, not from pruning; I did not
  inspect the test ones), against 0 for byte-level BPE. Pieces per note (test / slang): roberta and
  distilroberta 8.28 / 9.17, ELECTRA 7.68 / 8.63, MiniLM 7.48 / 8.54 (`tokenizer_report.json`).
  Reading eval notes here was a report-only count; no selection used it.

## Runtime and size

Native CPU of this Mac, int8, one thread, batch 1, 300 calls on dev notes (not a browser number):
p50 / p95 ms: roberta 1.23 / 2.19, distilroberta 1.52 / 2.80, electra 1.16 / 2.07, MiniLM 1.56 /
2.53. MiniLM's 12 sequential layers cost about 0.3 ms more; browser latency was not measured.
MiniLM is 0.33 MB larger than the shipped model (29.50 MB vs 29.17 MB, still 0.5 MB under 30 MB).
int8 vs fp32 agreement on 50 dev notes (export report): MiniLM seed 1 type 1.00, tags 0.998, value
1.00; ELECTRA type 0.98.

## roberta-base seed noise: is v3 worse than v2?

v3 = roberta-base + value head, seeds 1-3 (this experiment; seed 1 is the v3 report). v2 = same
recipe without the head: only two runs exist (the recorded run and the control re-run of seed 1);
no further v2 seeds were trained. Int8, snapped.

| metric | v2 recorded | v2 re-run | v3 seed 1 / 2 / 3 | v3 mean (min-max) | v3 mean - v2 mean |
|---|---|---|---|---|---|
| primary type F1 | 0.9561 | 0.9437 | 0.9397 / 0.9316 / 0.9455 | 0.9389 (0.9316-0.9455) | -0.0110 |
| primary target | 0.8879 | 0.8925 | 0.8879 / 0.8972 / 0.8925 | 0.8925 (0.8879-0.8972) | +0.0023 |
| secondary type F1 | 0.9020 | 0.8845 | 0.8784 / 0.8765 / 0.8928 | 0.8826 (0.8765-0.8928) | -0.0107 |
| secondary target | 0.8345 | 0.8379 | 0.8207 / 0.8379 / 0.8552 | 0.8379 (0.8207-0.8552) | +0.0017 |
| slang type F1 | 0.8903 | 0.9024 | 0.9034 / 0.9047 / 0.9164 | 0.9082 (0.9034-0.9164) | +0.0118 |
| slang target | 0.8347 | 0.8123 | 0.8207 / 0.8319 / 0.7871 | 0.8133 (0.7871-0.8319) | -0.0103 |
| slang joint | 0.7535 | 0.7451 | 0.7675 / 0.7703 / 0.7311 | 0.7563 (0.7311-0.7703) | +0.0070 |
| dev type F1 | 0.8016 | 0.8117 | 0.8342 / 0.8256 / 0.8509 | 0.8369 (0.8256-0.8509) | +0.0303 |
| dev target | 0.8042 | 0.8413 | 0.8095 / 0.8095 / 0.8360 | 0.8183 (0.8095-0.8360) | -0.0044 |

Answer: **a single seed cannot separate them, and three seeds do not show a clear cost of the value
head.** The two v2 runs of one seed differ from each other by up to 0.0124 (primary type), 0.0175
(secondary type), 0.0224 (slang target) and 0.0371 (dev target); the three v3 seeds differ by
0.0139, 0.0163, 0.0448 and 0.0265 on the same metrics. The v3-minus-v2 mean gaps (-0.011 to
+0.012 on test and slang, +0.030 on dev type) are the size of that spread, in both directions. The
one consistent sign is type F1 on the two test sets (-0.011 each, below the single-run 0.0164 and
0.0236 the v3 report saw), which is at the edge of the 0.01 gate and cannot be called a cost or
noise with n=2 for v2. Seed-to-seed noise on the roberta student is about 0.01-0.02 on a single
metric and 0.003 on the five-metric composite, which is why the bake-off gate is on the composite
mean and why a 0.01 margin on a single metric would not have been decidable.

## Limits

* Two seeds per challenger. The rule was declared that way; a third MiniLM seed would tighten the
  mean but cannot change the verdict unless it drops the composite by more than 0.02.
* Silver labels throughout; no human labels exist for English. Probes are diagnostics, not truth.
* MiniLM is wider in vocabulary (20,500 rows against 9,000) by the fixed budget rule; the budget
  puts the parameters into rows because its transformer is only 21 M. A different split of the same
  budget was not tried (the protocol forbids per-candidate tuning).
* ELECTRA misses the primary-type guard by 0.0002 of the 0.01 allowance on a 214-note set (about a
  half-note-equivalent shift); the rule says no, and I report it as no.
* ORT Web support is by op-set equality, not by a browser run. Latency is native CPU.
* roberta-base's recipe was the one tuned in v1-v3; the challengers received the identical, untuned
  recipe. A tuned challenger could be better, a tuned roberta could narrow the gap; neither was
  tried.

## Reproduce

```bash
uv run python experiments/gidi-en-backbone-bakeoff/feasibility.py       # sizes, op sets (random weights)
bash experiments/gidi-en-backbone-bakeoff/run_all.sh                    # build, train, export, score (about 3.5 h)
uv run python experiments/gidi-en-backbone-bakeoff/aggregate.py         # tables.md, aggregate.json, decision
uv run python experiments/gidi-en-backbone-bakeoff/tokenizer_report.py  # pieces per note, [UNK]
```

Files: `protocol.json` (declared first), `feasibility.{py,json}`, `build_candidate.py`,
`run_all.sh` and `run_all.log`, `runs/<candidate>/seed<N>/{train.log,train_log.jsonl,export.json,results.json}`,
`aggregate.{py,json}`, `tables.md`, `tokenizer_report.{py,json}`. Checkpoints and ONNX files are under
`models/gidi-en-bakeoff*` (untracked).
