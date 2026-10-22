---
language: en
library_name: onnx
license: apache-2.0
base_model: FacebookAI/roberta-base
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
It runs on a CPU in about a millisecond, and the INT8 file is 29 MB.

This is release `{release_version}` of `{model_version}`. It is an early, experimental release: the
model has been scored only against labels written by language models, and trained with one seed.
Please read [Results](#results) and [Limitations](#limitations) before relying on it.

For one note the model predicts two things:

- the transaction **type**, one of {types};
- the **target**: the counterparty span (`Sam`, `mom`, `company`), or `null` when the note has no
  counterparty.

It does not read the amount. Gidi's Vietnamese models have a rule-based amount parser; there is
none for English yet.

Gidi is a small open project. This model is the English companion to the Vietnamese
[`x8bu/gidi-finance`](https://huggingface.co/x8bu/gidi-finance) family. Code, annotation guide and
experiment reports: <https://github.com/8bu/gidi>.

## Intended use

Notes people write to themselves while tracking money: `coffee 4.50`, `Mike paid 15`,
`moved 500 to savings`. The notes are short (the encoder reads at most {max_length} tokens), informal, often
lower case, and may use slang, abbreviations or emoji. The model is meant to run inside a note app,
on the device, with no server.

It is not a bookkeeping tool, and it does not decide anything about money. Use it to suggest a
type and a counterparty that the person can confirm or fix. The confidence values are not
calibrated, so do not use them as thresholds.

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

## Examples

These are real outputs of the INT8 model with the word snap applied (see [Usage](#usage)).

| note | type | target |
|---|---|---|
| `Mike paid 15` | `repayment_in` | `Mike` |
| `company paid oct 5k` | `income` | `company` |
| `covered lunch for Sam 20` | `lend` | `Sam` |
| `venmo'd mom 50` | `expense` | `mom` |
| `got paid 2400` | `income` | null |
| `owe Sam 20` | `borrow` | `Sam` |
| `Sam owes me 20` | `lend` | `Sam` |
| `moved 500 to savings` | `transfer` | null |
| `amazon refund 25` | `refund` | `amazon` |
| `fronted jay 30 for the uber` | `lend` | `jay` |
| `bro paid me back 20` | `repayment_in` | `bro` |
| `paycheck hit 2400` | `income` | null |

And some notes it gets wrong, by my reading of the annotation guide:

| note | model output | expected |
|---|---|---|
| `salary advance 300` | `income`, null | `borrow` (R4) |
| `gym 30` | `repayment_in`, `gym` | `expense`, null |
| `bruh i fronted jay 30 for the uber` | `lend`, `bruh` | `lend`, `jay` |
| `cvrd drinks 4 crew 60` | `expense`, null | `lend`, `crew` |

Bare nouns with no verb (`gym 30`, `barber cut 15 quid`) are a known weak spot: the model sometimes
invents a counterparty and a repayment direction.

## Usage

The release is a runtime bundle: the model, `config.json` (labels, ONNX interface, `target_snap`),
`tokenizer.json`, `tokenizer_config.json`, `vocab_map.json`, `LICENSE`, the roberta-base license in
`licenses/` and this card. Files and hashes are listed in `manifest.json` and `checksums.txt`.

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
#  'target_confidence': ..., 'truncated': False, 'model_version': '{model_version}'}
```

`GidiPredictor` reads `target_snap: "words"` from `config.json` and applies the word snap itself.

### With plain ONNX Runtime

The graph takes `input_ids` and `attention_mask` (both `int64`, shape `[batch, tokens]`) and
returns `type_logits` (`[batch, 8]`) and `tag_logits` (`[batch, tokens, 3]`). The tag order is
`O`, `B-TARGET`, `I-TARGET`; the type order is the one in `config.json`.

```python
import json
import unicodedata

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

config = json.load(open("runtime/config.json"))
tok = Tokenizer.from_file("tokenizer/tokenizer.json")
tok.no_padding()
tok.enable_truncation(max_length=config["max_length"])
session = ort.InferenceSession("model/int8/model.int8.onnx", providers=["CPUExecutionProvider"])

text = unicodedata.normalize("NFC", "covered lunch for Sam 20")
enc = tok.encode(text)
ids = np.array([enc.ids], dtype=np.int64)
out = session.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)})
type_logits, tag_logits = out
print(config["types"][int(type_logits[0].argmax())])

tags = tag_logits[0].argmax(-1)  # 0 = O, 1 = B-TARGET, 2 = I-TARGET
```

Decode the target as the first span that starts at a `B-TARGET` token (and continues over
`I-TARGET` tokens), using `enc.offsets` to cut it out of `text`. Skip the `<s>` and `</s>` tokens
(`enc.special_tokens_mask`). A token includes the space before it (` Sam` is offsets 17 to 21 in
the example), so move the span start past leading whitespace. Then apply the word snap.

### The word snap

The word snap is a post-processing step. It is **not** part of the ONNX graph, so a plain ONNX
Runtime integration must do it. The tag head works on word pieces and sometimes stops inside a word
(`pap` for `papa`). Snapping widens the predicted span outward to whole words: take the maximal
run of non-whitespace characters at each end of the span, then trim punctuation from the edges of
the part you added. The span is only ever widened, never shortened. The reference code is
`snap_to_words` in `gidi.inference.decode`; the numbers below are scored with the snap on. Without
it you get the raw spans, which scored a little lower (target exact 0.824 instead of 0.835 on the
slang probe).

### In the browser

The graph uses the same operators as the Vietnamese model that already runs in a browser with
ONNX Runtime Web, and the tokenizer is a byte-level BPE that `tokenizers` and the web port both
handle. I have not run this English model in a browser yet, so treat web use as expected to work, not tested.

Speed on one CPU thread, ONNX Runtime, an Apple Silicon laptop: about 1.2 ms median per note
(2.2 ms at the 95th percentile), measured with 50 sample notes.

## How it was built

There are no human labels for English. Every label came from a language model.

1. **Notes.** About 2,700 short English notes were generated by a language model in several regional
   styles (US, UK, India, Singapore, mixed), then de-duplicated. A second round added slang,
   abbreviation and Gen Z notes written with Gemini.
2. **Labels.** Held-out and seed notes were labelled independently by Gemini and by Clef (a
   Cloudflare-hosted model), and where they disagreed Claude adjudicated against the annotation
   guide. Training labels came from a few-shot Clef teacher that used the seed notes as examples.
3. **Student.** The model is a four-layer student cut out of `roberta-base` (layers 2, 5, 8 and
   11), with the position table cut to 34 rows, the vocabulary pruned to 9,000 byte-level BPE
   tokens and the feed-forward width reduced from 3072 to 2048. It has about 29 million
   parameters. A type head and a BIO target head sit on top. It was trained for 40 epochs on
   2,355 notes, from one seed, and the last epoch was kept without checkpoint selection.
4. **Export.** ONNX, then INT8 dynamic quantization.

The training notes never include the test split or the two probe sets below. The annotation guide,
the data report and the experiment reports are in the GitHub repository (`docs/annotation-en-v1.md`,
`experiments/gidi-en-data/report.md`, `experiments/gidi-en-student-v2/report.md`).

## Results

All scores are agreement with silver labels from Claude, Gemini and Clef. **They are not accuracy
against human labels.** Scored once, with the word snap on, INT8 model, one seed. The pass
thresholds were written before the model was trained.

| set | notes | type macro-F1 | target exact | type and target both right |
|---|---|---|---|---|
| slang probe | 357 | 0.890 | 0.835 | 0.754 |
| Gen Z probe | 94 | 0.897 | 0.851 | 0.787 |
| test, primary (Gemini and Clef agreed) | 214 | 0.956 | 0.888 | 0.850 |
| test, secondary (all complete notes) | 290 | 0.902 | 0.834 | 0.759 |
| dev | 189 | 0.802 | 0.804 | 0.677 |

Per-type F1 on the slang probe: expense 0.912, income 0.916, borrow 0.984, lend 0.897,
repayment_in 0.795, repayment_out 0.750, transfer 0.921, refund 0.947.

How to read these numbers:

- The slang probe and the Gen Z probe were written by the same model family (Gemini) as part of
  the training notes, and labelled by the same Clef family. Part of the score may come from
  familiarity with that style, and not from general skill with slang.
- With one seed there is no estimate of run-to-run variance. The margins over the pre-set
  thresholds are small (+0.034 joint, +0.035 target, +0.020 type on the slang probe).
- The Gen Z probe has 94 notes; one note is about one percentage point.
- Dev type F1 (0.802) is lower than the test sets and was not used for any decision.
- The Gen Z notes were written by a language model, not collected from real people.

## Limitations

- **No human labels.** The model learned from, and was scored against, language-model labels.
  Where those labels are consistently wrong, the model copies the mistake and the score does not show it.
- **Borrow labels are shaky.** The `borrow` and `lend` direction is the hardest part of the guide,
  and the labellers agreed least there (`borrow` was the lowest-agreement type).
- **`repayment_out` is the weakest type** (F1 0.750 on the slang probe). Abbreviated verbs such as
  `pd` and `rfd` are not understood.
- **Generic nouns without a verb** (`gym 30`, `kopi c peng 2 s$`) can get a made-up target and the
  wrong direction. Spurious targets on notes that have none are still common: 32 notes in the
  slang probe whose labelled target is null got a target.
- **Salary advances** (`salary advance 300`) can come out as `income`, against rule R4.
- **No amount.** The model does not extract or normalize amounts, dates or currencies.
- **Confidences are not calibrated** and there is no abstention. Every note gets a type.
- Notes are cut at {max_length} tokens; a counterparty after the cut cannot be found (`truncated` is
  `true` in `GidiPredictor`).
- English only, and only the styles it was generated in. Real user notes will differ.

## License

The Gidi code, labels and model weights are under the Apache License 2.0 (`LICENSE`).

The student is derived from [`FacebookAI/roberta-base`](https://huggingface.co/FacebookAI/roberta-base)
(Liu et al., 2019), released under the MIT license. Its copyright and permission notice is kept in
`licenses/roberta-base-MIT.txt`.

## Related

- Vietnamese models: [`x8bu/gidi-finance`](https://huggingface.co/x8bu/gidi-finance).
- Code and docs: <https://github.com/8bu/gidi>.
