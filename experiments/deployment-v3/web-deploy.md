# Web playground deployment: gidi-finance-v3 3.0.0 at https://gidi.8bu.dev

Parity: `web-parity.md` / `web-parity.json` (from `scripts/dump_web_parity.py` and
`pnpm -C playground parity`). Screenshots: `web-screenshots/`.

## Deployment

| | |
|---|---|
| Worker | `gidi` (static assets + `/models/*` Worker), version id `ffeb3beb-5d18-4b53-a3a8-8f4b600525e8` |
| R2 bucket / binding | `gidi-models` / `MODELS` |
| R2 key | `gidi-finance-v3/3.0.0/model.int8.onnx` (28,658,533 B, sha256 `61c17d27b92e1ac4ac1add0af73344cb1c68bdbbc19db58532ce6d38d9bb7315`, uploaded from `models/gidi-finance-v3/` with `--remote`, downloaded back and hashed) |
| Public model URL | `https://gidi.8bu.dev/models/gidi-finance-v3/3.0.0/model.int8.onnx` (GET from the live site hashes to the same sha256) |
| Kept for rollback | `gidi-finance-v2/2.0.2/model.int8.onnx` (57,503,558 B), still served |

The browser path: the page fetches `/release/gidi-finance-v3/3.0.0/{web-release.json,config.json,
tokenizer.json}`, downloads the model, verifies the sha256, creates one ORT session (inputs
`input_ids`, `attention_mask`; outputs `type_logits`, `tag_logits`). The target is snapped to whole
words and the value comes from the TypeScript port of `gidi.value_parser`; `value_confidence` is
`null`.

## Live check (fresh headless Chromium tab on the deployed site)

The page loaded ("gidi-finance-v3", "SHA-256 61c17d27b92e… verified", loaded in 3.8 s). The raw JSON
equals `GidiPredictor.from_bundle("models/gidi-finance-v3").predict(...)` for these notes:

| note | type | target | value |
|---|---|---|---|
| `mua sữa vinamilk hết 500k` | expense | none (known target-head miss, gold is `vinamilk`; identical in Python) | `500k` [21, 25] |
| `còn nợ chị Mai 700k` | borrow | `Mai` [11, 14] | `700k` [15, 19] |
| `quà sinh nhật bé Na 300k` | expense | `Na` [17, 19] | `300k` [20, 24] |

`cho a Nam vay 1 triệu 20/10` (checked on a local `wrangler dev` build of the same bytes) gives lend /
`Nam` / `1 triệu`: the v2 known limitation (value `1 triệu 20/10`) is gone.
