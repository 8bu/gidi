# Deployment: gidi-finance-v1 and gidi-finance-v2

## gidi-finance-v2 (testable candidate)

`models/gidi-finance-v2/` is immutable, like v1. Details: `experiments/deployment-v2/report.md`.

Packaging and publishing `gidi-finance-v2` as a release: [releasing.md](releasing.md).

v2 adds a **value span** to v1. Its `type` and `target` come from v1's model unchanged: v2's
INT8 type/target logits are bit-identical to the v1 bundle's. It is a dual encoder in one ONNX
graph that reads one tokenization:

- path A is the frozen v1 encoder with the type and target heads;
- path B is a fine-tuned copy of the encoder, then an MLP, then the CRF emissions in
  `value_logits`.

```python
GidiPredictor.from_bundle("models/gidi-finance-v2").predict("mượn chú hai 5 xị").to_dict()
# {"type": "borrow", ..., "target": "chú hai", "target_span": [5, 12], ...,
#  "value_text": "5 xị", "value_span": [13, 17], "value_confidence": 0.99999...,
#  "truncated": false, "model_version": "gidi-finance-v2"}
```

- `value_text` is `text[start:end]` of the caller's string, or `null`. No number is ever
  derived from it.
- **Value decoding** uses the CRF scores in `config.json` (`value_decoding`):
  1. Viterbi runs over the real tokens; `<s>`/`</s>` are tagged O.
  2. BIO spans are built as for the target.
  3. The span with the highest confidence is kept.
- **`value_confidence`** is the geometric mean of the softmax probability of each span token's
  Viterbi tag. With no span, it is the minimum P(O).
- **Plain ONNX Runtime integration:** same as v1 below, with a third output, `value_logits`
  `[1, T, 3]` (O, B-VALUE, I-VALUE), decoded with Viterbi instead of argmax.

| file | bytes | sha256 |
|---|---|---|
| `model.int8.onnx` | 57,503,558 | `bc7945c3a7ff1d43648b65fe4fcae1ca3ca3c21dc23dccd61aa750f19881994e` |
| `config.json` | 2,199 | `77747c09b21692863a1776545433a6950a8ebe8cff311b95a834db51e23cc338` |
| `tokenizer.json`, `tokenizer_config.json`, `vocab_map.json` | as v1 | identical to v1 |

- The bundle is 58.14 MB. The FP32 reference is `models/gidi-finance-v2-onnx/model.onnx`
  (228.8 MB).
- CPU, 1 thread, warm: p50 3.52 ms, p95 4.86 ms (v1: 1.57 / 2.20 ms).
- Known limitations:
  - `cho a Nam vay 1 triệu 20/10` → value `1 triệu 20/10` (the date is absorbed);
  - INT8 type/target differs from FP32 on a few near-tie notes, exactly as v1 INT8 does;
  - all v1 limitations below apply to type/target.

```sh
uv run python scripts/freeze_deployment.py --out models/gidi-finance-v2 \
  --protocol experiments/deployment-v2/protocol.json --check
uv run python scripts/verify_deployment.py --protocol experiments/deployment-v2/protocol.json
```

The verifier exits 1 because of the two inherited INT8-vs-FP32 gate items described in the
report.

The rest of this document describes `gidi-finance-v1`.

## Purpose

`gidi-finance-v1` classifies one short Vietnamese personal-finance note. It predicts:

- the **transaction type**, one of `expense`, `income`, `borrow`, `lend`, `repayment_in`,
  `repayment_out`, `transfer`, `refund` (annotation-v1 semantics, `docs/annotation-v1.md`);
- the **counterparty/target span**: one contiguous substring of the note, or null.

It does not extract amounts or dates, and it has no "skipped"/out-of-taxonomy class. A
debt-state note such as `còn nợ Hùng 300k` still gets one of the 8 types. Filtering such notes
is the surrounding application's job.

## Artifact

`models/gidi-finance-v1/` is immutable: files are read-only, and the freeze script refuses to
overwrite a bundle whose bytes differ.

| file | bytes | sha256 |
|---|---|---|
| `model.int8.onnx` | 28,658,533 | `a31caa5764c5ba99c44d4575da4695ce44232b836edb6d99b775558869dfc411` |
| `tokenizer.json` | 574,195 | `564679d03ffb980406a70829b99b5e969df3112e240c819b6534f843fd3ae045` |
| `tokenizer_config.json` | 340 | `e7d65c11cae92e4995e5a41f7c78593ab90096cbc35f3e0e7f96120bb1080d76` |
| `vocab_map.json` | 49,336 | `da3f4357262e0379a6167107effd7c1ff5e5b8f8a1f99e87f2a44bb736ea3f68` |
| `config.json` | 796 | `f0afe138b5eea63dafb6b255ded7191c7b9cd6fd38f4b7f47822fb15a1b3638a` |
| `manifest.json` | 4,583 | lists every checksum above plus the sources |

Sources, recorded in `manifest.json`:

- checkpoint `model.safetensors`: `bfb2195a…0556`;
- FP32 ONNX: `0366683f…ae97`;
- FFN map: `486084a8…b00a`;
- vocab map: `da3f4357…3ea3f68`.

Architecture:

- BamiBERT-derived 4×768 encoder (BamiBERT layers 2, 5, 8 and 11), 12 heads;
- FFN 2048;
- 34 position rows (2050 → 34);
- vocabulary B-rank-8000 with 8,338 tokens (from 20,481);
- `max_length` 32 tokens including `<s>` and `</s>`;
- dynamic INT8 quantization (QInt8 weights);
- compression-v3, seed 1.

The bundle totals 29.29 MB.

## Running inference

The runtime is `src/gidi/inference/`. It depends only on `numpy`, `onnxruntime` and `tokenizers`,
not on torch, transformers or the training code.

```python
from gidi.inference import GidiPredictor, EmptyInputError

predictor = GidiPredictor.from_bundle("models/gidi-finance-v1")  # 1 CPU thread
predictor.predict("trả nợ chị Mai 500k").to_dict()
# {"type": "repayment_out", "type_confidence": 0.99999, "target": "Mai",
#  "target_span": [11, 14], "target_confidence": 0.99999, "truncated": false,
#  "model_version": "gidi-finance-v1"}
```

### Input

- One `str` per call, at batch size 1.
- `""` or whitespace-only input raises `EmptyInputError` (a `ValueError`). Non-`str` input raises
  `TypeError`.
- The text is NFC-normalized, exactly as the training data was. Accents and case are never
  removed or changed.
- Decomposed (NFD) input works, and offsets still refer to the caller's original string.
- Lone surrogates are replaced with U+FFFD for tokenization only.
- Input longer than 32 tokens is truncated on the right, as in training, and `truncated` is
  `true`. Only the first ~30 tokens are seen, so a target after the cut cannot be found.

### Output

| field | meaning |
|---|---|
| `type` | argmax of the 8 type logits |
| `type_confidence` | softmax probability of `type` |
| `target` | `text[start:end]` of the caller's original string, or `null` |
| `target_span` | `[start, end]` Unicode code-point offsets into the original string, end exclusive; or `null` |
| `target_confidence` | span present: geometric mean of the softmax probability of each member token's predicted BIO tag. Span null: the minimum `P(O)` over the real tokens. |
| `truncated` | the note had more than 32 tokens |
| `model_version` | `gidi-finance-v1` |

Span decoding is deterministic and identical to training's `spans_from_tags`:

- Only the first span is returned. `B` starts it, consecutive `I` extend it, and the next `B`
  or an `O` ends it. An `I` without a preceding `B` also starts a span.
- Token offsets are trimmed of whitespace.
- If no token is tagged, `target` is `null`.

The runtime adds no business rules on top of the model.

Confidences are uncalibrated scores. There are no thresholds or abstention policy yet; that is
a later calibration phase.

### Plain ONNX Runtime integration

The Python wrapper is thin, and a native app can reimplement it with ONNX Runtime plus the
`tokenizers` library (Rust, or any binding that reads `tokenizer.json`):

1. NFC-normalize the text.
2. Tokenize with `tokenizer.json`, with truncation set to max_length 32 (longest-first, right side, stride 0), no padding. `<s>` and `</s>` come from the post-processor. `tokenizer.json` has truncation disabled, so it must be enabled explicitly.
3. Feed int64 `input_ids` and `attention_mask` of shape `[1, T]`. The outputs are `type_logits` `[1, 8]` and `tag_logits` `[1, T, 3]`.
4. Take the argmax and decode as above, using code-point offsets. Map them back if you normalized.

`vocab_map.json` is provenance only: it maps pruned ids to the original BamiBERT ids. The
bundled tokenizer already emits pruned ids.

## Local playground

`playground/` is a small React app (Vite, TypeScript, Tailwind v4, shadcn/ui) for developers.
`scripts/demo_ui.py` serves the built app and a JSON API from the standard library and calls the
real `GidiPredictor`, loaded once at startup. It needs Node and pnpm for the one-time build.

```sh
pnpm -C playground install && pnpm -C playground build && uv run python scripts/demo_ui.py
# then open http://127.0.0.1:8765
```

The build writes `playground/dist/`. Until it exists, `/` answers `503 frontend_not_built`
(the API still works). Rebuilding needs no server restart. For frontend work, run the server and
`pnpm -C playground dev`; Vite proxies `/api` to port 8765.

The page has a composer (Enter runs, Shift+Enter adds a line, Run and Clear buttons), the ten
preset notes, and one parsed result:

- type, target and value with confidences (one decimal; the raw number is in the tooltip) and
  half-open code-point spans `[start, end)`;
- model version, latency and the `truncated` flag;
- the input with the target and value spans highlighted;
- the raw JSON (copy button) and a Diagnostics tab (latency, round trip, offsets, backend,
  precision, architecture, opset, bundle path).

Empty input, an unreachable server, an inference error, a missing bundle file
(`missing_artifact`), a malformed response and a 10 s timeout each show their own inline alert.
The page follows the system light/dark theme and has a theme toggle.

The spans come from the returned code-point offsets; the page never searches for a string. It
renders the server's `segments` (`role` is `null`, `"target"` or `"value"`), whose text always
joins back to the input exactly, so characters outside the BMP (emoji) highlight correctly. The
two heads are independent, so their spans could in principle overlap; if they do, both spans are
reported unchanged, the overlap is outlined and flagged (`spans_overlap`), and Diagnostics shows
the raw offsets. A bundle without a value head (v1) returns no value keys; the page shows the
value as unavailable. The value is a span only: the page shows no number.

`GET /api/info` returns `model_version`, `bundle`, `max_length` and `has_value_head`, plus
`architecture`, `precision`, `backend`, `model_file`, `onnx_opset` and `bundle_path`. For the
preset `cho a Nam vay 1 triệu 20/10` the page shows a neutral "Known limitation" badge while the
returned value differs from `1 triệu`; the output is never changed.

It binds to `127.0.0.1` by default; `--host`, `--port` and `--bundle` override this. The default
bundle is `models/gidi-finance-v2`; `--bundle models/gidi-finance-v1` still works.

## Known limitations

- **Lender-first `X cho mượn/vay` → `borrow` is weaker than the full FFN-3072 reference,
  especially when unaccented.** Probe-v1 lender-first joint is 0.59 vs 0.78 across seeds. For
  example, `thg Khoa cho muon tam 300k` is predicted `lend`.
- On test, quality matches the FFN-3072 reference: seed 1 type accuracy is 0.952, span F1
  0.808 and joint 0.848. Probe-v1, the hard diagnostic set, is much lower: seed 1 joint is 0.605
  in FP32 and 0.580 in INT8. Title/name boundaries, insurance and loan-installment notes are
  the weak patterns. See `experiments/compression-v3/report.md`.
- INT8 predictions can differ from FP32 on unusual input. On 59 robustness inputs, 1 type
  differed and 4 spans differed (Arabic, Thai/Korean, quote-heavy, 300-word notes). On the
  golden suite there were 0 differences.
- **Multi-piece brand names can be cut to a sub-word.** Production case `mua sữa vinamilk hết
  500k` (prod-0001, `tests/data/production-regressions.jsonl`; gold target `vinamilk` [8, 16],
  value `500k` [21, 25]) returns target `v` in the browser and in Python FP32, and `vin` in
  Python INT8; both are wrong. This is a target-head error, not a runtime bug: tokens and
  offsets are identical in Python and TypeScript, and the character mapping is exact. The head
  tags ` v` B-TARGET, tags `amil` O, and is near-tied on `in`.
- There is no skipped class; see Purpose.

## Performance

Measured on an Apple Silicon Mac, CPU, 1 thread, batch 1, INT8, over 2000 calls. Latency
covers tokenization, the ONNX run and decoding.

| | |
|---|---|
| warm latency p50 / p95 | 1.19 / 1.61 ms |
| cold start (import + load + first predict) | 0.05 + 0.04 + 0.002 s |
| memory after load | +87.5 MB RSS over the imported runtime (131 MB peak process) |
| model file | 28.66 MB |
| tokenizer artifacts (tokenizer.json, config, vocab map) | 0.62 MB |
| bundle total | 29.29 MB |

## Verification

```sh
uv run python scripts/freeze_deployment.py --out models/gidi-finance-v1 --check
uv run python scripts/verify_deployment.py          # -> experiments/deployment-v1/verification.json
uv run pytest -q tests/test_inference.py
```

`verify_deployment.py` runs the golden suite (`experiments/deployment-v1/golden-suite.jsonl`)
through three paths: PyTorch, ONNX FP32 and ONNX INT8. It also runs the robustness inputs and
checks offsets, determinism, isolation (the runtime must not touch `datasets/`, `corpus/` or
`experiments/`), reproducibility and performance.

## Updating the model

A new model is a new version, for example `gidi-finance-v2`. Never edit `gidi-finance-v1` in
place.

1. Pick the checkpoint by a rule declared before packaging. Do not pick by test/probe results.
2. Point `scripts/freeze_deployment.py` at the new sources and protocol, then build a new
   `models/gidi-finance-vN/`.
3. Rebuild the golden suite's expected outputs from the new PyTorch reference, then run
   `verify_deployment.py` and pass the acceptance gate.
4. A change of taxonomy or semantics also needs a new annotation version
   (`configs/annotation-vN.yaml`).
5. Record the change in `experiments/deployment-vN/report.md` and the investigation journal.

## Web playground (gidi.8bu.dev)

The same React playground (`playground/`) also ships as a static site that runs
`gidi-finance-v2` **in the browser** (onnxruntime-web, WASM, INT8). No server does inference and no
note leaves the device. It is one Cloudflare Worker (`gidi`) with static assets; the 57.5 MB model
is in R2 (Workers assets are limited to 25 MiB per file) and streamed by the Worker at
`/models/<key>`.

- Source of truth is the immutable release `dist/releases/gidi-finance-v2/2.0.2/`.
  `playground/scripts/sync-release.mjs` checks `config.json` and `tokenizer.json` against the
  release `manifest.json` and copies them to `playground/public/release/` (gitignored), together
  with `web-release.json` (model URL, sha256, size). The browser verifies the model's sha256
  before creating the ORT session; a mismatch is a hard error.
- Build modes: `pnpm -C playground build:web` is the browser build; `pnpm -C playground build`
  stays the build served by `scripts/demo_ui.py` (Python `/api` backend).
- Deploy: `pnpm -C playground run deploy` (build:web + `wrangler deploy`; `run` is needed because `pnpm deploy` is a built-in). Config is
  `playground/wrangler.jsonc`: R2 binding `MODELS` -> bucket `gidi-models`, custom domain
  `gidi.8bu.dev`, only `/models/*` runs the Worker (`playground/worker/index.ts`: GET/HEAD, Range
  and conditional requests, `cache-control: public, max-age=31536000, immutable`).
- A new model release is a new R2 key (`<model>/<release>/model.int8.onnx`), never an overwrite:
  `wrangler r2 object put gidi-models/<key> --file <release model> --remote`.
- Local check: `pnpm -C playground build:web && pnpm -C playground exec wrangler dev` (add the
  model to the local R2 with the same `r2 object put ... --local`).
