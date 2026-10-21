# Gidi — GiaoDịch

[Tiếng Việt](README.md) · **English**

This is my experiment to build a tiny classifier.

Demo: <https://gidi.8bu.dev>. The model runs entirely in your browser (onnxruntime-web, INT8).

Model: `{model_version}`, release `{release_version}`.

A tiny on-device model for short Vietnamese personal-finance notes such as `mượn chú hai 5 xị`.
For one note the system predicts:

- the transaction **type**: one of {types};
- the **target**: the counterparty span (for example `chú hai`), or `null`;
- the **value**: the span that holds the amount as the user wrote it (for example `5 xị`), or
  `null`. It is a substring of the note; no number is derived from it.

`{model_version}` has three parts:

1. One INT8 encoder (4 layers, 768 wide, vocabulary pruned to 8,338 tokens) in one ONNX graph.
   It predicts type and target.
2. A *word snap*: the target span is extended to whole-word boundaries, so `pbank` in
   `rut tien vpbank 1tr` becomes `vpbank`.
3. A rule-based parser (`gidi.value_parser`, version `{value_parser_version}`) that reads the
   value from the original note. There is no neural value head. `config.json` records the parser
   version, and the runtime refuses to load a bundle whose parser version does not match.

Notes are NFC-normalized and cut at {max_length} tokens for the encoder. The parser reads the whole
note, including the part after the cut.

## What changed since `gidi-finance-v3`

Same architecture, ONNX interface, tokenizer, value parser and word snap as `gidi-finance-v3`;
only the encoder weights are new. The training set added deterministic augmentation and notes
labelled by a stronger teacher model under four clarified rules (`docs/annotation-v4.md`):
R1 who pays (`Long trả 5 lít`), R2 paying on someone's behalf (`ứng giùm Lan tiền điện 400k`),
R3 sending money to other people (`chuyển cho mẹ 2 củ`) and R4 salary phrasing
(`mới nhận lương 26cu`). The INT8 file has the same size as v3 (28.66 MB).

## Results

Scored once on the same sets as `gidi-finance-v3`, with the encoder from seed 1 (fixed before
training). `human-value-02` labels come from two independent LLM labellers with adjudication; they
are LLM labels, not human labels. End-to-end means type, target and value are all exact.

| `human-value-02` (n = 190) | `gidi-finance-v4` | `gidi-finance-v3` |
|---|---|---|
| type accuracy | 93.68 % | 92.11 % |
| target exact | 88.95 % | 85.26 % |
| end-to-end | 84.21 % | 78.95 % |

| set | type v4 | type v3 | target v4 | target v3 |
|---|---|---|---|---|
| frozen test (n = 105) | 98.10 % | 94.29 % | 93.33 % | 92.38 % |
| probe-v1 (n = 81) | 90.12 % | 82.72 % | 85.19 % | 85.19 % |

On the new rule probe `probe-rules-v1` (64 notes), type and target are both right for 76.6 % of
notes (v3: 50.0 %). Full numbers and the experiment record: `experiments/clef-distill-v2/report.md`.
On sets this small, a difference of a few points may not be meaningful.

## Variants

| variant | archive | model file | size | sha256 |
|---|---|---|---|---|
| int8 | `{int8_archive}` | `model.int8.onnx` | {int8_size_mb} | `{int8_sha256}` |
| fp32 | `{fp32_archive}` | `model.onnx` | {fp32_size_mb} | `{fp32_sha256}` |

`int8` is the recommended deployment model. `fp32` is the reference export, used for parity
checks; it is published only as the `-fp32` archive on the GitHub release. Both archives are
complete runtime bundles: the model, `config.json` (labels, ONNX interface, value rule and
`target_snap`), `tokenizer.json`, `tokenizer_config.json`, `vocab_map.json`, `LICENSE`, this
README and `release-manifest.json`. Every file of the release is listed with its sha256 in
`manifest.json` and `checksums.txt`; a repository that hosts only the INT8 files can check them with
`sha256sum -c --ignore-missing checksums.txt`.

## Usage

Requirements: Python {python_requirement}, onnxruntime {onnxruntime_requirement}, tokenizers
{tokenizers_requirement}, numpy {numpy_requirement}, plus the `gidi` package (`gidi.inference` and
`gidi.value_parser`). The parser is part of the `gidi` package, not of the ONNX file.

Extract an archive, then load the directory with `GidiPredictor`. It reads `config.json` and turns
the word snap and the parser on by itself.

```python
from pathlib import Path

from gidi.inference.predictor import GidiPredictor

# INT8 archive: the bundle contains model.int8.onnx, which is the default model file.
predictor = GidiPredictor.from_bundle("{model_version}-{release_version}-int8")
print(predictor.predict("mượn chú hai 5 xị").to_dict())

# FP32 archive: point to its model.onnx.
root = Path("{model_version}-{release_version}-fp32")
predictor = GidiPredictor.from_bundle(root, model_path=root / "model.onnx")
```

`predict(...).to_dict()` returns these keys:

| key | meaning |
|---|---|
| `type` | transaction type |
| `type_confidence` | softmax probability of `type` |
| `target` | counterparty text (after the word snap), or `null` |
| `target_span` | `[start, end]` code-point offsets into the input, or `null` |
| `target_confidence` | confidence of the target span |
| `value_text` | value text (`text[start:end]` of the input), or `null` |
| `value_span` | `[start, end]` offsets of the value, or `null` |
| `value_confidence` | always `null`: the rule-based parser has no score |
| `truncated` | `true` if the encoder cut the note at the token limit |
| `model_version` | `{model_version}` |

For a plain ONNX Runtime integration, the graph takes `input_ids` and `attention_mask` and
returns `type_logits` and `tag_logits`. Take the argmax of `tag_logits`, decode the target span,
then extend it to word boundaries as `target_snap` in `config.json` says. The graph has no
`value_logits`: the value needs the `gidi.value_parser` parser (or an equivalent port of version
`{value_parser_version}`).

## Known limitations

- **Paying on someone's behalf (R2)** is still weak: on `probe-rules-v1` type and target are right
  for about half of the `trả/ứng/mua hộ X` notes.
- On a 17-note field check, 13 notes are right. Still wrong: `Chuyển cho họ hàng 2 củ`,
  `long tra 5 lit` (lowercase name), `cong ty tra tien t10 5 lit` and
  `Khoa trả lại tiền mình trả giùm 50k`.
- Multi-word brand or shop names can be cut short, and some counterparties are missed.
- **The test sets are LLM-labelled.** `human-value-02` labels come from two LLM labellers with
  adjudication, not from humans. The value labels equal the parser's output, so the 100 % value
  score does not show that the parser is right on other notes. Treat all numbers as rough
  estimates.
- Type and target confidences are uncalibrated. There are no thresholds or abstention policy.
- NBSP and full-width digits are not folded before the encoder.
- A target after the {max_length}-token cut cannot be found by the encoder (`truncated` is
  `true`). The parser still reads the whole note.
- The system does not extract or normalize amounts or dates, and it has no "out of scope" class:
  every note gets one of the types above.

## License

Apache License 2.0. The full text is in `LICENSE`.
