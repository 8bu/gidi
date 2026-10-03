# {model_version} {release_version}

[Tiếng Việt](README.md) · **English**

A tiny on-device model for short Vietnamese personal-finance notes such as `mượn chú hai 5 xị`.
For one note it predicts:

- the transaction **type**: one of {types};
- the **target**: the counterparty span (for example `chú hai`), or `null`;
- the **value**: the span that holds the amount as the user wrote it (for example `5 xị`), or
  `null`. It is a substring of the note; no number is derived from it.

The model is `{experiment}` (seed {seed}): a dual encoder in one ONNX graph. One path (the frozen
gidi-finance-v1 encoder) gives type and target; a second fine-tuned path with a CRF gives the
value span. Notes are NFC-normalized and cut at {max_length} tokens.

## Variants

| variant | archive | model file | size | sha256 |
|---|---|---|---|---|
| int8 | `{int8_archive}` | `model.int8.onnx` | {int8_size_mb} | `{int8_sha256}` |
| fp32 | `{fp32_archive}` | `model.onnx` | {fp32_size_mb} | `{fp32_sha256}` |

`int8` is the recommended deployment model. `fp32` is the reference export, used for parity
checks; it is published only as the `-fp32` archive on the GitHub release. Both archives are
complete runtime bundles: the model, `config.json` (labels, ONNX interface and CRF scores),
`tokenizer.json`, `tokenizer_config.json`, `vocab_map.json`, `LICENSE`, this README and
`release-manifest.json`. Every file of the release is listed with its sha256 in `manifest.json`
and `checksums.txt`; a repository that hosts only the INT8 files can check them with
`sha256sum -c --ignore-missing checksums.txt`.

## Usage

Requirements: Python {python_requirement}, onnxruntime {onnxruntime_requirement}, tokenizers
{tokenizers_requirement}, numpy {numpy_requirement}, plus the `gidi` package (`gidi.inference`).

Extract an archive, then load the directory with `GidiPredictor`.

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
| `target` | counterparty text, or `null` |
| `target_span` | `[start, end]` code-point offsets into the input, or `null` |
| `target_confidence` | confidence of the target span |
| `value_text` | value text (`text[start:end]` of the input), or `null` |
| `value_span` | `[start, end]` offsets of the value, or `null` |
| `value_confidence` | geometric mean of the span tokens' tag probabilities |
| `truncated` | `true` if the note was cut at the token limit |
| `model_version` | `{model_version}` |

For a plain ONNX Runtime integration, the graph takes `input_ids` and `attention_mask` and
returns `type_logits`, `tag_logits` and `value_logits`. The value tags need CRF Viterbi decoding
with the scores in `config.json`; argmax is not equivalent.

## Known limitations

- `cho a Nam vay 1 triệu 20/10` gives the value `1 triệu 20/10`: the date is absorbed into the
  value span (the right value is `1 triệu`).
- INT8 type/target differs from FP32 on a few near-tie notes, exactly like the v1 INT8 model.
- NBSP and full-width digits are not folded; they reach the model as they are.
- Values past the {max_length}-token cut are `null` (`truncated` is `true`).
- The model does not extract or normalize amounts or dates, and it has no "out of scope" class:
  every note gets one of the types above.

## License

Apache License 2.0. The full text is in `LICENSE`.
