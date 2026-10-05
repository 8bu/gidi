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

## What changed since `gidi-finance-v2`

- **Debt rule.** A note that only states an existing debt (`còn nợ Hùng 300k`,
  `Hoa còn nợ mình 350k`) is `borrow` (you owe) or `lend` (the other party owes you). It used to
  be skipped. See `docs/annotation-v3.md`.
- **Gift rule.** The receiver of a gift or ceremony money is the counterparty, so it is the
  `target` (`mừng cưới Hoa 1 triệu` has the target `Hoa`). The test set uses this rule. This
  model's training data did not apply it yet, so gift notes are still a weak spot (see
  Limitations).
- **Rule-based amount.** The value comes from a deterministic parser, not from a CRF value
  head. The parser has no score, so `value_confidence` is `null`.
- **Word snap** is on by default (`target_snap: words` in `config.json`).
- **Half the size.** One encoder instead of two: the INT8 bundle is 29.3 MB against 58.1 MB for
  v2. Latency is about 1.4 ms p50 (v2: 3.5 ms), CPU, one thread, Apple Silicon, batch 1.

## Results

New test set `human-value-02`: 200 notes, 190 of them scored. The labels come from two independent
LLM labellers, with adjudication where they disagreed. They are LLM labels, not human labels.
End-to-end means type, target and value are all exact. The v2 column is the shipped
`gidi-finance-v2` bundle run unchanged on the same 190 notes.

| `human-value-02` (n = 190) | `gidi-finance-v3` | `gidi-finance-v2` |
|---|---|---|
| type accuracy | 92.11 % | 75.79 % |
| target exact | 85.26 % | 74.74 % |
| value exact | 100.00 % | 98.42 % |
| end-to-end | 78.95 % | 59.47 % |

Two older sets, scored on type and target only. v2's type and target are identical to the
`gidi-finance-v1` INT8 encoder, so the v2 column is that encoder.

| set | type v3 | type v2 | target v3 | target v2 |
|---|---|---|---|---|
| frozen test (n = 105) | 94.29 % | 94.29 % | 92.38 % | 88.57 % |
| probe-v1 (n = 81) | 82.72 % | 77.78 % | 85.19 % | 81.48 % |

Every number is from one scoring run, with seed 1 fixed before training and no selection by
score. On sets this small, a difference of a few points may not be meaningful.

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

- **The counterparty (target) is right about 85 % of the time.** On `human-value-02`, 25 notes
  have a wrong target, 12 a wrong type and 3 both (40 wrong notes out of 190). Typical target
  errors are multi-word brand or shop names cut short (`đi xem phim lotte cinema 130k` gives
  `lotte`, `trả góp thế giới di động 1tr8` gives `động`) and missed counterparties
  (`cf trung nguyen 45k`).
- **Gift and lì xì notes.** The model has not learned the gift rule. Known wrong notes:
  `lì xì bà ngoại 500k` (predicted `income`, the right type is `expense`) and
  `quà tốt nghiệp em Khải 500k` (target `tốt`, the right target is `Khải`).
- **Some debt notes still get the wrong direction**, for example `Hung no minh 2tr chua tra`
  (predicted `repayment_in`, right is `lend`) and `nhớ đòi Trang 450k` (predicted `borrow`,
  right is `lend`).
- **The test set is LLM-labelled.** `human-value-02` has 190 scorable notes, labelled by two
  independent LLM labellers and adjudicated, not by humans. The value labels equal the parser's
  output on all 190 notes, so the 100 % value score does not show that the parser is right on
  other notes. Treat all numbers as rough estimates.
- Type and target confidences are uncalibrated. There are no thresholds or abstention policy.
- NBSP and full-width digits are not folded before the encoder.
- A target after the {max_length}-token cut cannot be found by the encoder (`truncated` is
  `true`). The parser still reads the whole note.
- The system does not extract or normalize amounts or dates, and it has no "out of scope" class:
  every note gets one of the types above.

## License

Apache License 2.0. The full text is in `LICENSE`.
