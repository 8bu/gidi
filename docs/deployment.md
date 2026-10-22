# Deployment: gidi-finance-v1 to v4

## gidi-finance-v4 (current)

`models/gidi-finance-v4/` (release 4.0.0) is the clef-distill-v2 D4 seed-1 INT8 encoder (28,658,533 B,
sha256 `a702d35ec450…`) with the same runtime contract as v3: rule-parser value, whole-word target
snap, ONNX inputs `input_ids`/`attention_mask`, outputs `type_logits`/`tag_logits`. Protocol:
`experiments/deployment-v4/protocol.json`; spec `releases/gidi-finance-v4.json`; freeze with
`uv run python scripts/freeze_deployment.py --out models/gidi-finance-v4 --protocol experiments/deployment-v4/protocol.json`.
Quality and the missed pre-set criterion: `experiments/clef-distill-v2/report.md`. The live demo
serves v4 as static-asset parts of `gidi-finance-v4/4.0.0/model.int8.onnx` (see "Web playground"); the R2 copies of v3 and v4 stay as an archive. The
Hugging Face/GitHub publication is not done for v4. Everything below describes v3; v4 differs only
by the model weights and the names `gidi-finance-v4` / `4.0.0`.


## gidi-finance-v3

`models/gidi-finance-v3/` is immutable, like v1 and v2. Protocol and sources:
`experiments/deployment-v3/protocol.json`. Packaging and publishing: [releasing.md](releasing.md).

v3 is a system of three parts, all driven by the bundle's `config.json`:

1. **One INT8 encoder** (type head + target BIO head, same ONNX interface as v1: outputs
   `type_logits` and `tag_logits`). Trained on annotation-v3 (debt-only notes are `borrow` /
   `lend`, [annotation-v3.md](annotation-v3.md)); the receiver of a gift is the target (a labelling
   rule of the test set that this model's training data does not apply yet).
2. **Whole-word target snap** (`"target_snap": "words"`): the decoded first target span is
   extended to whole words (`gidi.inference.decode.snap_span_to_words`).
3. **Rule-based value span** (`"value_source": "rule-parser"`,
   `"value_parser": {"name": "gidi.value_parser", "version": "1"}`): the value is
   `gidi.value_parser.parse_value(text)` on the caller's string. There is no neural value head
   and `value_labels` / `value_decoding` are absent.

```python
GidiPredictor.from_bundle("models/gidi-finance-v3").predict("mượn chú hai 5 xị").to_dict()
# {"type": "borrow", ..., "target": "chú hai", "target_span": [5, 12], ...,
#  "value_text": "5 xị", "value_span": [13, 17], "value_confidence": None,
#  "truncated": False, "model_version": "gidi-finance-v3"}
```

- `GidiPredictor.from_bundle` reads `value_source` and `target_snap`. `snap_words=None` (the
  default) follows the bundle (on for v3, off for v1 and v2); `snap_words=True/False` overrides
  it. v1 and v2 bundles carry neither key, so their behaviour is unchanged.
- The output keys are the v2 keys. `value_confidence` is `None` (JSON `null`): the parser has no
  score. `has_value_head` is true when predictions carry a value span (v2 head or v3 parser).
- The parser version is pinned. `gidi.value_parser.VERSION` must equal the bundle's
  `value_parser.version`, otherwise `GidiPredictor` raises `BundleError`. Bump `VERSION` with
  any change that can alter `parse_value`; that needs a new bundle.
- The encoder sees the first 32 tokens; the parser reads the whole note.
- **Plain ONNX Runtime integration:** the graph has two outputs; argmax the tags, decode the
  first span, snap to words, and get the value from the parser (or a port of the same version).

| file | bytes | sha256 |
|---|---|---|
| `model.int8.onnx` | 28,658,533 | `61c17d27b92e1ac4ac1add0af73344cb1c68bdbbc19db58532ce6d38d9bb7315` |
| `config.json` | 932 | `e349cb300b3a1bdd18342f513608d0ab9a0a7a19014785ac47481d557c85dfeb` |
| `tokenizer.json`, `tokenizer_config.json`, `vocab_map.json` | as v1 | identical to v1 |

- The bundle is 29.28 MB (v2: 58.14 MB). The FP32 reference is
  `models/gidi-finance-v3-onnx/model.onnx` (114.0 MB, a byte copy of the encoder's FP32 export).
- CPU, 1 thread, warm, batch 1: p50 1.41 ms, p95 1.76 ms with or without the snap
  (`experiments/annotation-v3-retrain-v4/results.json`).
- Quality: `experiments/deployment-v3/hv02-release-check.json` scores the bundles exactly as
  shipped on `human-value-02` (190 scored notes, LLM-labelled by two independent labellers plus
  adjudication). v3: type 0.9211, target exact 0.8526, value exact 1.0, end-to-end 0.7895;
  `gidi-finance-v2` on the same notes: 0.7579, 0.7474, 0.9842, 0.5947.
- Known limitations: the counterparty is right about 85% of the time (multi-word brand and shop
  names, gift receivers, a few debt directions); the test set is LLM-labelled and small; confidences
  are uncalibrated; see the model card (`releases/templates/gidi-finance-v3.README.*.md`).

```sh
uv run python scripts/freeze_deployment.py --out models/gidi-finance-v3 \
  --protocol experiments/deployment-v3/protocol.json --check
uv run python scripts/score_hv02_release.py    # writes experiments/deployment-v3/hv02-release-check.json once
```

The rest of this document describes `gidi-finance-v2` and `gidi-finance-v1`.

## gidi-finance-v2

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

The React app in `playground/` ships as a static site that runs
`gidi-finance-v3` **in the browser** (onnxruntime-web, WASM, INT8 encoder for type and target;
the whole-word target snap and the rule-based value parser are TypeScript ports of
`gidi.inference.decode.snap_to_words` and `gidi.value_parser`). No server does inference and no
note leaves the device. It is one Cloudflare Worker (`gidi`) that is **assets only** (no Worker code
runs, so no request is billed): the 28.7 MB models are static assets too, cut into 16 MiB parts at
`/models/<id>/<version>/model.int8.onnx.NN` (Workers assets are limited to 25 MiB per file). Cost
and traffic model: [cost-and-traffic.md](cost-and-traffic.md).

- `/` is the notes app (`playground/src/app/`): a sticky note in the centre, one paper stack per
  type around it, each with the sum of its amounts. A sent note flies into its stack. Tapping a
  stack lists its records; the user can correct type, target and amount (the model's reading is
  kept). Notes stay in the browser (IndexedDB) with JSON export/import. The app turns the value
  span into VND in `playground/src/app/amount.ts` (app-only; a bare number under 1,000 with no
  unit or currency suffix counts in thousands; slang `củ`/`chai` = 1 million, `xị`/`lít`/`cành` =
  100k, `tỏi` = 1 billion; the app widens the value span over `cành`/`tỏi`, which the pinned
  parser does not read). Works on phones and tablets (touch).
- **Language and model.** The app runs one model per language: Vietnamese `gidi-finance-v4`
  4.0.0 and English `gidi-en-v2` 2.0.1 (29,166,949 B, sha256 `c169b23f…1933`, served as parts like
  v4). The default comes from the visitor's country, with no Worker: a zone **Response Header
  Transform Rule** on `gidi.8bu.dev` sets `Server-Timing: geo;desc=<ip.src.country>` on every
  response (rule and expression: [cost-and-traffic.md](cost-and-traffic.md), "Cloudflare rule"),
  and the page reads it from its own navigation entry (`performance.getEntriesByType("navigation")`,
  `hintFromServerTiming` in `src/lib/language.ts`): `VN` gives `vi`, any other country `en`. No
  cookie, no storage or log of the IP or country. A visible `VI | EN` switch in the header (and
  the menu, which shows the active model name and version) sets `gidi:lang` in localStorage; a
  stored choice always beats the hint (`playground/src/lib/language.ts`, unit tests in
  `language.test.ts`). With no hint at all (dev server, or the rule missing) the browser
  language decides. Switching retires the other model (aborts a running download or disposes its
  worker and ORT session) and loads the new one with the same progress readout and SHA-256 check.
- **One database per language.** Notes live in IndexedDB `gidi-vi` (Vietnamese, VND) and `gidi-en`
  (English, USD); the app shows, totals, exports and imports only the active language's database,
  and a switch never mixes or copies notes. Each database also has a `meta` store (the model it
  last ran, the migration record). The pre-language store `gidi-notes` is copied once, raw and
  verified, into `gidi-vi` (records get `currency: "VND"`); `gidi-notes` itself is never changed
  or deleted, so it is the rollback. If the copy fails, `gidi-notes` is served and the copy is
  retried next start. Code: `playground/src/app/notes-db.ts` (migration), `notes-core.ts` (store),
  tests `notes-core.test.ts` (migration, failed migration, isolation, switching, export/import).
  Local check of both defaults: `pnpm -C playground dev` then open `/?geo=VN` or `/?geo=US` (the
  override exists only in the dev server; a production build ignores `?geo=`).
  UI strings are two typed dictionaries in `playground/src/app/i18n.ts`.
- **English amounts.** The English bundle has no value head or parser, so the app finds the amount
  with a deterministic en-US scan of the note (`findEnglishValue` in `app/amount.ts`: `12`, `12.50`,
  `1,200`, `5k`, `1.2m`, `3 million`, `2 grand`; a `$` sign or a money word wins, otherwise the last
  number; times, dates and `5pm` are ignored). Amounts are US dollars (`$1,234.50`), a bare number
  is not thousands and no Vietnamese unit applies. Each note has a `currency` (`USD`/`VND`).
  Limitation: other currencies in an English note (`£`, `€`, `₹`, `S$`, `quid`, `rupees`, `sgd`)
  are not recognised or converted; the number is kept and shown as dollars.
- `/lab` is the developer playground (spans, confidences, diagnostics, raw JSON); it stays on
  the Vietnamese model.
- The site needs a secure context (HTTPS or `localhost`): the model sha256 check uses
  `crypto.subtle`. To try a local build on a phone, expose it over HTTPS (for example
  `cloudflared tunnel --url http://localhost:8787`), not a plain LAN IP.
- Usage analytics: PostHog project "Gidi (gidi.8bu.dev)" (US cloud, id 648258), wired in
  `playground/src/lib/analytics.ts` (event names and properties are the typed
  `AnalyticsEventMap`). Production bundles only, loaded as a lazy chunk. A random anonymous ID
  kept in localStorage (no cookie, no person profile, never identified) so returning visitors and
  retention work; no autocapture, session replay, heatmaps, surveys, `$pageleave` or web vitals
  (PostHog bills per event; caps and sampling: [cost-and-traffic.md](cost-and-traffic.md)). Failures go to
  PostHog error tracking with safe context: [error-tracking.md](error-tracking.md).
  Events carry counts and types,
  never note text, targets or amounts. `VITE_POSTHOG_KEY` / `VITE_POSTHOG_HOST` override the
  project at build time. Open the app once with `?internal=1` on your own browsers (`?internal=0`
  undoes it): their events get `internal: true`, and the project's default test-account filter
  (`internal` not set, `$host` = `gidi.8bu.dev`) drops them from insights.
- Search and answer engines: `playground/index.html` carries the meta, Open Graph and JSON-LD
  (`WebApplication`) plus a static fallback inside `#root` that non-JS crawlers read (hidden once
  JS runs; React replaces it). Static pages `/gioi-thieu` (vi) and `/about` (en) are plain HTML
  in `playground/public/` with `AboutPage`/`FAQPage` JSON-LD; keep their facts and FAQ JSON-LD in
  sync with the visible text and with the current release numbers. `llms.txt`, `llms-full.txt`,
  `robots.txt` (blocks only `/models/`) and `sitemap.xml` sit beside them; `/lab` gets
  `X-Robots-Tag: noindex` from `_headers`. Icons and social cards are rendered from
  `playground/assets-src/` with `build.sh` (needs rsvg-convert, ImageMagick and the Be Vietnam
  Pro font).
- Google Search Console: domain property `gidi.8bu.dev` (the owner's Google account), verified
  by the TXT record `google-site-verification=IU7C…` on `gidi.8bu.dev`; keep it, or the
  property loses verification. `sitemap.xml` is submitted there.

- Source of truth is the immutable bundles `models/gidi-finance-v4/` (release 4.0.0) and
  `models/gidi-en-v2-bundle/` (release 2.0.1), listed per language in `playground/web-models.json`
  (releases to ship, the `active` one, a `blocked` list, the manifest `schema` / `min_client`).
  `playground/scripts/sync-release.mjs` checks each bundle's files against its `manifest.json`,
  copies `config.json` and `tokenizer.json` to `playground/public/release/<model>/<release>/`,
  cuts the model into 16 MiB parts under `playground/public/models/<model>/<release>/` (all
  gitignored) and writes `public/release/manifest.json` (below). The browser downloads the parts in
  parallel, joins them and verifies the whole file against the manifest's sha256 before creating
  the ORT session; a mismatch is a hard error. `playground/src/lib/models.ts` only holds UI labels
  and the record-to-language map; keep its ids in sync with `web-models.json`.

### Model rollout runbook (no manual cache purge)

`/release/manifest.json` is the single source of truth for what each language runs: per language
`active`, `blocked`, and every release's id, version, sha256, size, `base` and `parts`, plus
`schema` and `min_client`. It is the only file that changes under a fixed URL: served
`Cache-Control: no-cache` (`public/_headers`) and fetched with `cache: "no-cache"`, so every load
revalidates it (ETag, a 304 costs nothing) and a stale copy can never win. Everything else is
immutable at a versioned path: `/assets/*` (fingerprinted), `/release/<id>/<version>/*`,
`/models/<id>/<version>/*`. HTML keeps the static-assets default (`max-age=0, must-revalidate`), so a
deploy is picked up on the next load. (The manifest has its own name because the first parts
release served an immutable `web-release.json` that some browsers still hold; the new code never
reads that path.)

Client (`src/runtime/manifest.ts`, `release-manager.ts`, `model-download.ts`, tests beside them): per
language it reads the release recorded in Cache Storage (`gidi-model:<lang>`, with a `release.json`
record written after the parts) and the manifest, then:

| Situation | What runs |
|---|---|
| nothing cached | download the active release, verify sha256, store, run |
| active release cached | run it (re-hashed) |
| cached is older/different than `active` (rollout **or** rollback) | run the cached one now; 3 s later fetch `active` in the background, verify sha256, and only then replace the cache entries (the next load runs it) |
| cached release is in `blocked` | drop it, download `active` in front, run it |
| `active` is `null` | the language is off: cache dropped, error shown |
| manifest unreachable or broken | run the cached release; nothing is forced |
| a background download fails | keep running the cached one; the error is reported, nothing else changes |
| `schema` / `min_client` newer than the code | tell the visitor to reload, and reload once by itself (`sessionStorage`) |

Publish a model: add the bundle to `web-models.json` (a new `version`, same `id` family), set
`active`, `pnpm -C playground build:web`, `node playground/scripts/verify-release.mjs
http://127.0.0.1:8787 --hash` against `pnpm -C playground preview:site`, `pnpm -C playground run
deploy`, then `node playground/scripts/verify-release.mjs https://gidi.8bu.dev`. Visitors move to
it in the background on their next visit; keep the previous release in `releases` so it can be
rolled back. Roll back or kill a bad model: set `active` back (or `blocked: ["<bad version>"]`, or
`active: null` to switch a language off) in `web-models.json` and run `pnpm -C playground run
deploy`: only the manifest changes, no code and no cache purge. Bump `min_client` (and
`CLIENT_VERSION` in `src/runtime/manifest.ts`) only with a client change older code must not run
against. A model family rename (`gidi-finance-v5`) also needs `src/lib/models.ts`.
- The TS runtime (`playground/src/runtime/`) must give the same type, target and value as
  `GidiPredictor.from_bundle("models/gidi-finance-v3")`. Check it with
  `uv run python scripts/dump_web_parity.py && pnpm -C playground parity` (report:
  `experiments/deployment-v3/web-parity.md`). English: `uv run python
  scripts/dump_web_parity_en.py --fixture && pnpm -C playground parity:en` compares the TS runtime
  on `gidi-en-v2` with `GidiPredictor` (tokenizer sweep, ids/offsets, decoder on the reference
  logits and the word snap must be exact; ORT numerics are reported, not gated). The `<mask>`
  token of the English tokenizer is `lstrip` (it swallows the whitespace before it); the TS
  tokenizer implements it. `pnpm -C playground test` runs the unit tests (language decision,
  English amounts, committed Python fixture for English tokens and the word snap). The parser and the Unicode classes it needs
  (`str.isalpha`, `isalnum`, `isupper`, `P*`, fold) are tabulated from Python by
  `scripts/gen_runtime_unicode_tables.py` (`unicode-tables.ts`, generated, do not edit).
- Build modes: `pnpm -C playground build:web` is the browser build; `pnpm -C playground build`
  stays the build served by `scripts/demo_ui.py` (Python `/api` backend).
- Deploy: `pnpm -C playground run deploy` (build:web + `wrangler deploy`; `run` is needed because `pnpm deploy` is a built-in). Config is
  `playground/wrangler.jsonc`: assets only (no `main`, no R2 binding, observability off), custom
  domain `gidi.8bu.dev`, SPA fallback. Every request is a static-asset request: free, unlimited,
  never a Worker invocation. The model parts and `/release/*` are immutable (`public/_headers`).
  Verified model bytes are also kept in the browser's Cache Storage
  (`playground/src/runtime/model-download.ts`, tests in `model-download.test.ts`), so a returning
  visitor does not download the model again. The R2 bucket `gidi-models` is no longer read by the
  site; its objects stay as an archive and as the rollback of the pre-cost-guard Worker version.
- A new model release is a new entry in `playground/scripts/sync-release.mjs` and
  `playground/src/lib/models.ts`; its parts land under a new path, never overwriting one (they are
  immutable for a year).
- Local check: `pnpm -C playground build:web && pnpm -C playground preview:site` (`wrangler dev`
  serves `dist/`).
