---
language: en
library_name: onnx
license: apache-2.0
base_model: microsoft/MiniLM-L12-H384-uncased
pipeline_tag: text-classification
tags:
  - onnx
  - english
  - personal-finance
  - token-classification
  - on-device
---

# gidi-en

A tiny classifier for short English personal-finance notes such as `covered lunch for Sam 20`.
It runs on a CPU in about two milliseconds, and the INT8 file is 29.5 MB.

This is release `{release_version}` of `{model_version}`. It is still an early, experimental
model: it has been scored only against labels written by language models, from two training runs
(seeds), and one of them is shipped. Please read [Results](#results) and
[Limitations](#limitations) before relying on it.

For one note the model predicts three things:

- the transaction **type**, one of {types};
- the **target**: the counterparty span (`Sam`, `mom`, `company`), or `null` when the note has no
  counterparty;
- the **value**: the span of the amount, exactly as written (`50`, `$20`, `5k`, `2 grand`,
  `twenty bucks`, `s$30`), or `null` when the note has no amount. It is text, not a number: turning
  `5k` into 5000 and choosing a currency is up to the app (see [The value span](#the-value-span)).

`{model_version}` replaces `gidi-en-v2`, which had no amount. The backbone changed too: the
student is now a 12-layer MiniLM instead of a 4-layer roberta cut, chosen by a pre-declared
comparison (see [How it was built](#how-it-was-built)).

Gidi is a small open project. This model is the English companion to the Vietnamese
[`x8bu/gidi-finance`](https://huggingface.co/x8bu/gidi-finance) family. Code, annotation guide and
experiment reports: <https://github.com/8bu/gidi>.

## Intended use

Notes people write to themselves while tracking money: `coffee 4.50`, `Mike paid 15`,
`moved 500 to savings`. The notes are short (the encoder reads at most {max_length} tokens), informal, often
lower case, and may use slang, abbreviations or emoji. The model is meant to run inside a note app,
on the device, with no server. The [Gidi web app](https://gidi.8bu.dev) runs it in the browser.

It is not a bookkeeping tool, and it does not decide anything about money. Use it to suggest a
type, a counterparty and an amount that the person can confirm or fix. The confidence values are
not calibrated, so do not use them as thresholds.

## The types

| type | meaning |
|---|---|
| `expense` | money out with no expectation of repayment: purchases, bills, tips, gifts, money sent to family |
| `income` | pay, freelance or client payment, sale proceeds, other ordinary money in |
| `borrow` | the user receives a loan, or owes someone (`owe Sam 20`), or takes a salary advance |
| `lend` | the user lends money, or someone owes the user (`Sam owes me 20`), or the user paid on someone's behalf and expects it back |
| `repayment_in` | someone pays the user back |
| `repayment_out` | the user pays back a debt, or pays down a loan, instalment plan or card balance |
| `transfer` | money moved between the user's own accounts, cash, savings or investments |
| `refund` | money returned by a merchant |

There is no "other" class. A note that does not fit still gets one of these eight. The full
guide, with the rules below, is in the GitHub repository (`docs/annotation-en-v1.md`).

Four rules decide most of the hard cases:

- **R1, the payer decides.** `Mike paid 15` is `repayment_in` with target `Mike` (a person, no goods
  named). `company paid oct 5k` is `income` with target `company` (an employer or client).
- **R2, paying on someone's behalf.** `covered lunch for Sam 20` is `lend` with target `Sam`.
- **R3, sending money to people.** `venmo'd mom 50` is `expense` with target `mom`.
  `transfer` is only for the user's own accounts.
- **R4, salary wording.** `got paid 2400`, `payday`, `paycheck hit` are `income` with no target.
  A salary advance is `borrow`.

## The value span

The value is the amount **as written**, with its currency marker or money word when it has one:
`$20`, `20 bucks`, `120$`, `s$30`, `150s$`, `rs 500`, `5k`, `1.2m`, `1,200`, `2 grand`,
`twenty bucks`, `a hundred`. It is not normalised and not converted. A `4` or `2` that stands for
`for` or `to` (`paynow sis 30 4 bday`) is not an amount. A note with several numbers has one
transaction amount (`2 coffees $9` is `$9`); the model picks it. When a note has no amount the value
is `null`.

Reading the span as a number is easy and deliberately left out, because the right answer depends on
the app: a currency, a locale for `1,200` and `1.200`, whether `5k` is 5000. The Gidi web app reads
the span as US dollars.

## Examples

These are real outputs of the INT8 model with the word snap applied (see [Usage](#usage)).

| note | type | target | value |
|---|---|---|---|
| `Mike paid 15` | `repayment_in` | `Mike` | `15` |
| `company paid oct 5k` | `income` | `company` | `5k` |
| `covered lunch for Sam 20` | `lend` | `Sam` | `20` |
| `venmo'd mom 50` | `expense` | `mom` | `50` |
| `got paid 2400` | `income` | null | `2400` |
| `owe Sam 20` | `borrow` | `Sam` | `20` |
| `Sam owes me 20` | `lend` | `Sam` | `20` |
| `moved 500 to savings` | `transfer` | null | `500` |
| `amazon refund 25` | `refund` | `amazon` | `25` |
| `2 grand for the car` | `expense` | null | `2 grand` |
| `twenty bucks to sam` | `lend` | `sam` | `twenty bucks` |
| `paid rent` | `expense` | null | null |

And some notes it gets wrong, by my reading of the annotation guide:

| note | model output | expected |
|---|---|---|
| `paid rent 1450` | `borrow`, null, `1450` | `expense` |
| `Sam iou 20` | `borrow`, `Sam`, `20` | `lend` (R2 direction) |
| `barber fade 20` | `income`, null, `20` | `expense` |
| `cvrd lunch 4 mate 12` | `expense`, `mate`, `12` | `lend` |
| `paynow sis 30 4 bday` | `expense`, `sis`, `30 4` | value `30` |
| `lunch w daryl` | `lend`, null, null | `expense` |

Bare nouns with no verb (`barber fade 20`), the direction of `iou`, and abbreviated verbs (`cvrd`,
`pd`) are known weak spots. The value span is usually right but can run on over a following
number-like word (`30 4`, `5000 rs 4`).

## Usage

The release is a runtime bundle: the model, `config.json` (labels, ONNX interface, `target_snap`,
`value_decoding`, `value_snap`, `tokenizer_type`), `tokenizer.json`, `tokenizer_config.json`,
`vocab_map.json`, `LICENSE`, the MiniLM license in `licenses/` and this card. Files and hashes are
listed in `manifest.json` and `checksums.txt`.

| variant | archive | model file | size | sha256 |
|---|---|---|---|---|
| int8 | `{int8_archive}` | `model.int8.onnx` | {int8_size_mb} | `{int8_sha256}` |
| fp32 | `{fp32_archive}` | `model.onnx` | {fp32_size_mb} | `{fp32_sha256}` |

`int8` is the recommended file. This repository holds only the INT8 files, in the release layout:
`model/int8/`, `tokenizer/`, `runtime/`. The FP32 reference export and the two archives are built
with the release but are not uploaded here. You can check the files with
`sha256sum -c --ignore-missing checksums.txt`.

Download it with `hf download x8bu/gidi-en --local-dir gidi-en`.

### With the `gidi` package

Requirements: Python {python_requirement}, onnxruntime {onnxruntime_requirement}, tokenizers
{tokenizers_requirement}, numpy {numpy_requirement}, and the `gidi` package from the GitHub
repository (`gidi.inference`). `GidiPredictor` wants one flat directory, so copy the runtime files
together first:

```bash
mkdir bundle
cp gidi-en/model/int8/model.int8.onnx gidi-en/runtime/config.json \
   gidi-en/tokenizer/tokenizer.json gidi-en/tokenizer/tokenizer_config.json bundle/
```

Then:

```python
from gidi.inference.predictor import GidiPredictor

predictor = GidiPredictor.from_bundle("bundle")
print(predictor.predict("covered lunch for Sam 20").to_dict())
# {'type': 'lend', 'type_confidence': ..., 'target': 'Sam', 'target_span': [18, 21],
#  'target_confidence': ..., 'value_text': '20', 'value_span': [22, 24],
#  'value_confidence': ..., 'truncated': False, 'model_version': '{model_version}'}
```

`GidiPredictor` reads `target_snap`, `value_decoding` and `value_snap` from `config.json` and does
the decoding below itself. Spans are in Unicode code points of the string you passed in.

### With plain ONNX Runtime

The graph takes `input_ids` and `attention_mask` (both `int64`, shape `[batch, tokens]`) and
returns `type_logits` (`[batch, 8]`), `tag_logits` (`[batch, tokens, 3]`) and `value_logits`
(`[batch, tokens, 3]`). The tag order is `O`, `B-TARGET`, `I-TARGET`; the value order is `O`,
`B-VALUE`, `I-VALUE`; the type order is the one in `config.json`.

```python
import json
import unicodedata

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

config = json.load(open("runtime/config.json"))
tok = Tokenizer.from_file("tokenizer/tokenizer.json")  # uncased WordPiece, [CLS] ... [SEP]
tok.no_padding()
tok.enable_truncation(max_length=config["max_length"])
session = ort.InferenceSession("model/int8/model.int8.onnx", providers=["CPUExecutionProvider"])

text = unicodedata.normalize("NFC", "covered lunch for Sam 20")
enc = tok.encode(text)
ids = np.array([enc.ids], dtype=np.int64)
out = session.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)})
type_logits, tag_logits, value_logits = out
print(config["types"][int(type_logits[0].argmax())])

target_tags = tag_logits[0].argmax(-1)  # 0 = O, 1 = B-TARGET, 2 = I-TARGET
real = [not special for special in enc.special_tokens_mask]  # skip [CLS] and [SEP]
value_tags = [int(t) if r else 0 for t, r in zip(value_logits[0].argmax(-1), real)]
```

The tokenizer lower-cases and strips accents, so `enc.offsets` point into the NFC text but the
case of the note is not seen by the model. Decode the **target** as the first span that starts at
a `B-TARGET` token (and continues over `I-TARGET` tokens), using `enc.offsets` to cut it out of
`text`, then apply the word snap.

Decode the **value** from `value_tags` the same way, except that a note can have several spans
(`2 coffees $9`): collect every span (`B-VALUE` starts one, `I-VALUE` extends it, `O` ends it), score
each by the geometric mean of the softmax probability of its tokens' predicted tags, and keep the
span with the highest score (the earlier one on a tie). Then apply the word snap. No span means no
amount. The reference code is `decode_value_argmax` in `gidi.inference.decode`.

### The word snap

The word snap is a post-processing step. It is **not** part of the ONNX graph, so a plain ONNX
Runtime integration must do it, for the target and for the value. The tag heads work on word pieces
and sometimes stop inside a word (`pap` for `papa`, `5` for `5k`). Snapping widens the predicted
span outward to whole words: take the maximal run of non-whitespace characters at each end of the
span, then trim punctuation from the edges of the part you added. The span is only ever widened,
never shortened (`$20,` stays `$20`). The reference code is `snap_to_words` in
`gidi.inference.decode`; the numbers below are scored with the snap on.

### In the browser

The graph uses the same operators as the Vietnamese model that runs in a browser with ONNX Runtime
Web, and the Gidi web app runs this release that way (`gidi.8bu.dev`). The app ports the WordPiece
tokenizer, the decoding and the snap to TypeScript; their tokenizer matches the Python one on every
test, dev and probe note and on fuzz strings (ids and character offsets). Under ONNX Runtime Web the
INT8 numerics differ slightly from native ONNX Runtime, so a few borderline notes decide
differently (see Limitations).

Speed on one CPU thread, ONNX Runtime, an Apple Silicon laptop: about 1.6 ms median per note
(2.5 ms at the 95th percentile), measured with 300 dev notes, 0.3 ms slower than the roberta-based
v2 because 12 layers run one after another.

## How it was built

There are no human labels for English. Every label came from a language model, a regex scan, or
both.

1. **Notes.** About 2,700 short English notes were generated by a language model in several regional
   styles (US, UK, India, Singapore, mixed), then de-duplicated. A second round added slang,
   abbreviation and Gen Z notes written with Gemini.
2. **Type and target labels.** Held-out and seed notes were labelled independently by Gemini and by
   Clef (a Cloudflare-hosted model), and where they disagreed Claude adjudicated against the
   annotation guide. Training labels came from a few-shot Clef teacher that used the seed notes as
   examples.
3. **Value labels.** A deterministic scan for amounts (en-US, UK, India and Singapore formats) and a
   Clef zero-shot choice among its candidates; where they disagreed Claude adjudicated.
4. **Backbone bake-off.** The question was whether another English encoder would beat the
   roberta-base student of `gidi-en-v2` under the same ~30 MB INT8 budget. The rule, the candidates
   and the win condition were written before any run (`experiments/gidi-en-backbone-bakeoff/`):
   roberta-base (control, 3 seeds), distilroberta-base, ELECTRA-base and MiniLM-L12-H384 (2 seeds
   each), the same data, recipe and export, only the backbone changes. MiniLM won by the declared
   rule; ELECTRA missed a guard by 0.0002; distilroberta did not clear the margin.
5. **Student.** [`microsoft/MiniLM-L12-H384-uncased`](https://huggingface.co/microsoft/MiniLM-L12-H384-uncased)
   (MIT) with all 12 layers kept at 384 wide, the position table cut to 33 rows and the vocabulary
   pruned to 20,500 uncased WordPiece rows (the ones the training notes use plus the lowest ids of
   the BERT uncased vocabulary). It has 29.2 million parameters. A type head, a BIO target head and
   a BIO value head sit on top. It was trained for 40 epochs on 2,355 notes, and the last epoch was
   kept without checkpoint selection.
6. **Seed.** Two seeds were trained. The shipped one (seed 1) was picked by a rule written before the
   release: the better of the two by the bake-off's composite computed on the **dev** split only
   (0.8857 against 0.8747). Test and probe scores did not take part.
7. **Export.** ONNX (opset 17), then INT8 dynamic quantization.

The training notes never include the test split or the two probe sets below. The annotation guide,
the data report and the experiment reports are in the GitHub repository (`docs/annotation-en-v1.md`,
`experiments/gidi-en-data/report.md`, `experiments/gidi-en-backbone-bakeoff/report.md`,
`experiments/gidi-en-student-v4/report.md`).

## Results

All scores are agreement with silver labels from Claude, Gemini, Clef and the amount scan. **They
are not accuracy against human labels.** Scored once, with the word snaps on, INT8 model, seed 1,
with the evaluation script that scored the bake-off. The pass thresholds were written before the
model was trained.

| set | notes | type macro-F1 | target exact | type and target both right | value exact |
|---|---|---|---|---|---|
| slang probe | 357 | 0.919 | 0.863 | 0.815 | 0.961 |
| Gen Z probe | 94 | 0.929 | 0.840 | 0.809 | 0.989 |
| test, primary (Gemini and Clef agreed) | 214 | 0.958 | 0.939 | 0.902 | 0.995 |
| test, secondary (all complete notes) | 290 | 0.923 | 0.900 | 0.828 | 0.997 |
| dev | 189 | 0.887 | 0.873 | 0.799 | 0.984 |

`gidi-en-v2` on the same sets (no amount): slang 0.890 / 0.835 / 0.754, Gen Z 0.897 / 0.851 / 0.787,
test primary 0.956 / 0.888 / 0.850, test secondary 0.902 / 0.834 / 0.759, dev 0.802 / 0.804 / 0.677.

Per-type F1 on the slang probe: expense 0.929, income 0.933, borrow 0.937, lend 0.842,
repayment_in 0.919, repayment_out 0.889, transfer 0.957, refund 0.947.

The bake-off in one table (INT8, snapped, mean over seeds; composite = mean of test-secondary type
F1, test-secondary target, slang joint, slang value, Gen Z joint):

| backbone | seeds | composite | test-secondary type F1 | test-secondary target | slang joint |
|---|---|---|---|---|---|
| roberta-base 4 layers x 768 (`gidi-en-v3` recipe) | 3 | 0.847 | 0.883 | 0.838 | 0.756 |
| distilroberta-base | 2 | 0.854 | 0.898 | 0.852 | 0.735 |
| ELECTRA-base | 2 | 0.863 | 0.876 | 0.866 | 0.797 |
| **MiniLM-L12-H384 (this model)** | 2 | **0.879** | 0.907 | 0.895 | 0.810 |

How to read these numbers:

- The slang probe and the Gen Z probe were written by the same model family (Gemini) as part of
  the training notes, and labelled by the same Clef family. Part of the score may come from
  familiarity with that style, and not from general skill with slang.
- Two seeds per backbone, no confidence interval. The MiniLM seeds are far above every roberta seed
  on the composite (0.876-0.881 against 0.845-0.848) but the sets are small: on the primary test set
  0.01 is about two notes, on the Gen Z probe about one.
- These numbers come from the bake-off's batched evaluation. The runtime that ships scores one note at
  a time; on the same sets that gives a composite of 0.877 against the bake-off's 0.876. INT8
  dynamic quantization is numerically fragile (see Limitations), so single scores move by a few
  notes between runtimes.
- Value exact is the exact span including its currency marker, so `40 bucks` against `40` counts
  as wrong. The slang probe's errors are mostly a span that runs one word too far (`30 4`,
  `5000 rs 4`) or stops short.
- The Gen Z notes were written by a language model, not collected from real people.

## Limitations

- **No human labels.** The model learned from, and was scored against, language-model and regex
  labels. Where those labels are consistently wrong, the model copies the mistake and the score
  does not show it.
- **Two seeds, one shipped.** A different seed would give somewhat different decisions on borderline
  notes; no run-to-run variance is available beyond the two bake-off runs.
- **Uncased.** The model never sees capital letters, so a cue like `lent Sam $20` against `lent sam
  $20` is lost. (On this data the cased roberta student was not better.) Characters outside the
  pruned WordPiece vocabulary (some symbols and scripts) become `[UNK]` for the whole word.
- **Borrow and lend are shaky.** The `borrow` and `lend` direction is the hardest part of the
  guide, and the labellers agreed least there. `iou` and abbreviated verbs (`cvrd`, `pd`, `rfd`) are
  often read the wrong way.
- **Generic nouns without a verb** (`gym 30`, `barber fade 20`, `paid rent 1450`) can get a made-up
  target or the wrong type. Spurious targets on notes that have none are still common.
- **The amount is text.** There is no number, no currency code and no conversion. Several numbers
  in one note (`2 coffees $9 and 4 tips`) can pick the wrong one; a note without an amount can still
  get one (3 of the 357 slang notes have none, the model found 2 of them wrong).
- **INT8 numerics.** Dynamic INT8 quantization means another ONNX Runtime build (browser, mobile,
  another CPU) computes slightly different logits. In the browser runtime, 34 of 4,261 checked
  inputs (every split, the probes, adversarial and fuzz strings) decide differently from native
  ONNX Runtime with the graph run as written, and logits differ by up to a few units on odd
  inputs. On the evaluation sets above the browser and native scores differ by at most about two
  notes per metric.
- **Confidences are not calibrated** and there is no abstention. Every note gets a type.
- Notes are cut at {max_length} tokens; a counterparty or amount after the cut cannot be found
  (`truncated` is `true` in `GidiPredictor`).
- English only, and only the styles it was generated in. Real user notes will differ.

## License

The Gidi code, labels and model weights are under the Apache License 2.0 (`LICENSE`).

The student is derived from [`microsoft/MiniLM-L12-H384-uncased`](https://huggingface.co/microsoft/MiniLM-L12-H384-uncased)
(Wang et al., 2020), released under the MIT license. Its copyright and permission notice is kept in
`licenses/MiniLM-L12-H384-uncased-MIT.txt`. The WordPiece vocabulary is a subset of the BERT
uncased vocabulary (Google, Apache License 2.0).

## Related

- Vietnamese models: [`x8bu/gidi-finance`](https://huggingface.co/x8bu/gidi-finance).
- Code and docs: <https://github.com/8bu/gidi>.
