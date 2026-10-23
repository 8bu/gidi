# gidi-en-v4: the MiniLM student with the value head

`gidi-en-v4` (release `4.0.0`) is seed 1 of the MiniLM-L12-H384 student that won the pre-declared
backbone bake-off (`experiments/gidi-en-backbone-bakeoff/report.md`: `WIN-WITH-PORT`). Nothing was
trained for this release. The work is the choice of seed, the web port that the bake-off named as
the cost (a WordPiece tokenizer and the value head in the browser runtime), the bundle and the
release. Every score is agreement with **silver** labels (regex + Clef + Claude), not human truth.

## Seed choice (`protocol.json`, written before the number was computed)

The bake-off trained two MiniLM seeds. The candidate is the one with the higher **dev-only**
composite: the mean of int8, snapped type macro-F1, target exact, joint (type and target) and value
exact on the dev split (n=189); within 0.002 the tie goes to seed 1. Test and probe scores did not
take part.

| | type macro-F1 | target exact | joint | value exact | dev composite |
|---|---|---|---|---|---|
| seed 1 | 0.8868 | 0.8730 | 0.7989 | 0.9841 | **0.8857** |
| seed 2 | 0.8798 | 0.8571 | 0.7831 | 0.9788 | 0.8747 |

Seed 1 wins by 0.0110, so the tie rule did not apply. Shipped: seed 1, last epoch, no checkpoint
selection, no averaging. INT8 file 29,499,600 B, sha256 `faf46f35…9ba6`, opset 17, 33 op types of
the shipped graph.

## Scores of the shipped model

From `experiments/gidi-en-backbone-bakeoff/runs/minilm/seed1/results.json` (int8, whole-word snap
on target and value, the bake-off's batched evaluation; no re-evaluation selected anything). `gidi-en-v2` in brackets
(type F1 / target / joint; v2 has no amount).

| set | n | type macro-F1 | target exact | joint | value exact | type + target + value |
|---|---|---|---|---|---|---|
| test primary | 214 | 0.9575 (0.9561) | 0.9393 (0.8879) | 0.9019 (0.8505) | 0.9953 | 0.8972 |
| test secondary | 290 | 0.9228 (0.9020) | 0.9000 (0.8345) | 0.8276 (0.7586) | 0.9966 | 0.8241 |
| dev | 189 | 0.8868 (0.8016) | 0.8730 (0.8042) | 0.7989 (0.6772) | 0.9841 | 0.7831 |
| slang probe | 357 | 0.9192 (0.8903) | 0.8627 (0.8347) | 0.8151 (0.7535) | 0.9608 | 0.7899 |
| Gen Z probe | 94 | 0.9289 (0.8967) | 0.8404 (0.8511) | 0.8085 (0.7872) | 0.9894 | 0.7979 |

Bake-off composite (mean of secondary type F1, secondary target, slang joint, slang value, Gen Z
joint): MiniLM 0.8786 (seeds 0.8757 / 0.8814) against roberta-base 0.8471 (0.8453-0.8483). The seed
shipped has composite 0.8757.

Value: 0.995 exact on test primary, 0.997 on secondary, 0.961 on the slang probe. By category on
the slang probe: currency marker 0.978 (273/279), decimals 0.944, scaled `k`/`grand` 0.917, two
numbers in the note 0.926 (113/122), no amount 1/3. The slang probe errors are mostly a span that
runs one word too far (`30 4`, `5000 rs 4`, `1.80 rn`) and amounts in notes the model gives no
amount to. The playground regex scanner (`findEnglishValue`, what gidi-en-v2 used in the app) on the same
sets, strict span: 0.935 / 0.938 / 0.160 (test primary / secondary / slang probe); with the
currency marker and money word stripped from both sides ("core"): 0.967 / 0.972 / 0.849. The model's
core scores are 0.995 / 0.997 / 0.961.

## The web port

* **WordPiece tokenizer** (`playground/src/runtime/wordpiece.ts`): the `BertNormalizer`, the
  `BertPreTokenizer` and WordPiece of the Hugging Face `tokenizers` crate, with offsets. Which code
  points are dropped, white space, CJK or punctuation is **read off the crate**
  (`scripts/gen_runtime_wordpiece_tables.py` -> `wordpiece-tables.ts`): the crate's Unicode data is
  older than JavaScript's (it keeps unassigned code points and newer `Cf`/`Mn` marks), and a first
  version built on `\p{...}` failed 41,789 of 132,941 sweep strings for that reason. NFD and
  lower-casing use the JavaScript built-ins. `config.json` declares `tokenizer_type`
  (`bpe`/`wordpiece`) and the manifest repeats it; the loader refuses a mismatch with `tokenizer.json`.
* **Value head** (`decode.ts`, `predictor.ts`): per-token argmax over the real tokens, the span of
  highest geometric-mean tag probability, whole-word snap: a port of `decode_value_argmax`
  (Python runtime and `scripts/evaluate_en_value.py` decode the same way, no CRF).
* **App** (`app/amount.ts`): the span is read as US dollars (`$`, `bucks`, `5k`, `2 grand`,
  `twenty bucks`, `50 cents`); a span that runs on into the next word is cut back to its amount; the
  regex scanner answers only for a model without a value output or a span that holds no amount.
  No span from the model means no amount.

Parity (`pnpm -C playground parity:en`, 4,291 notes plus 132,941 sweep strings, the Python
reference at `ORT_ENABLE_BASIC`): tokenizer ids, offsets and masks identical on **all** of them (the
sweep covers every BMP and SMP code point and a sample of the others, the rest are every test, dev,
probe, seed and train note, the app notes, 914 seeded adversarial strings and 612 WordPiece fuzz
strings); the decoder fed the Python reference logits is exact on type, target, value, tags and
confidences. The committed fixture `wordpiece.fixture.json` (every test/dev/probe note and the fuzz
strings) checks the tokenizer in `pnpm -C playground test` without a model.

ORT numerics (reported, not gated, as for the Vietnamese bundle): onnxruntime-web and the Python
session compute different INT8 activation quantization, so 34 of 4,261 notes decide differently
(type, target or value tags) and the max logit difference is 4.8 on adversarial strings (1.7 on test
notes). Gold-label quality of the three backends on the same sets (`scripts/score_web_parity_en.py`,
composite = the bake-off composite):

| backend | composite | secondary type F1 | secondary target | slang type | slang target | slang value | Gen Z type |
|---|---|---|---|---|---|---|---|
| bake-off harness (batched) | 0.8757 | 0.9228 | 0.9000 | 0.9192 | 0.8627 | 0.9608 | 0.9289 |
| Python, default session, one note at a time | 0.8766 | 0.9212 | 0.8966 | 0.9174 | 0.8571 | 0.9580 | 0.8801 |
| Python, `ORT_ENABLE_BASIC` | 0.8792 | 0.9309 | 0.9000 | 0.9121 | 0.8543 | 0.9636 | 0.8740 |
| browser (onnxruntime-web) | 0.8792 | 0.9309 | 0.9000 | 0.9211 | 0.8515 | 0.9636 | 0.8740 |

The browser composite equals the Python-basic one and is not below the bake-off's. The Gen Z type
F1 moves by 0.05 between runtimes (94 notes, three or four notes): the INT8 model is numerically
fragile on borderline notes, which is a property of dynamic quantization and of this recipe.

## Caveats

* Silver labels only: no human label exists for English; every number above is agreement with a
  model's or a scanner's label.
* Two seeds trained, one shipped, picked on dev. There is no variance estimate for the shipped
  seed, and the bake-off's means rest on two (MiniLM) and three (roberta) seeds.
* The evaluation sets are small (0.01 is 2-3 notes on the test sets, one on the Gen Z probe) and the
  probes were written by language models.
* Uncased: the case cue is gone; the pruned vocabulary maps characters outside it to `[UNK]` for
  the whole word.
* The numbers of the bake-off come from a batched, padded evaluation; the runtime scores one note at a
  time. They differ by a few notes (table above).
* The web app reads the value span as US dollars; other currencies keep their number.

## Reproduce

```bash
uv run python scripts/freeze_en_bundle.py --out models/gidi-en-v4-bundle \
  --protocol experiments/gidi-en-deployment-v4/protocol.json
uv run python scripts/gen_runtime_wordpiece_tables.py
uv run python scripts/dump_web_parity_en.py --fixture
pnpm -C playground parity:en
uv run python scripts/score_web_parity_en.py
```
