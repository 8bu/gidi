# Web playground deployment: https://gidi.8bu.dev

Machine-readable numbers: `web-deploy.json`. Screenshots: `web-screenshots/`.
All browser numbers below were measured on the **deployed** site, in fresh incognito browser
contexts (no HTTP cache), with the eval `browser` helper: HeadlessChrome 150.0.7871.24, macOS on
Apple Silicon (12 cores), network = this workstation's real connection to Cloudflare. "Mobile" is
390x844 viewport emulation at DPR 3 with touch; it is **not** a real phone and has no CPU or
network throttling, so its timings are not representative of a phone.

## Deployment

| | |
|---|---|
| Worker | `gidi` (static assets + `/models/*` Worker), `workers_dev: false` |
| Current version id | `a0ca4f18-59f3-49d6-afa7-f93cdd8fb058` (earlier today: `03952912-…`, `8b808673-…`) |
| Compatibility date | 2026-10-03 |
| Custom domain | `gidi.8bu.dev` (Workers custom domain) |
| R2 bucket / binding | `gidi-models` / `MODELS` |
| R2 key | `gidi-finance-v2/2.0.2/model.int8.onnx` (57,503,558 B, sha256 `bc7945c3…81994e`, uploaded from the immutable release with `--remote`) |
| Public model URL | `https://gidi.8bu.dev/models/gidi-finance-v2/2.0.2/model.int8.onnx` |
| Config | `playground/wrangler.jsonc`; Worker `playground/worker/index.ts`; deploy `pnpm -C playground run deploy` |

The Worker only runs for `/models/*` (`run_worker_first`); everything else is served by the assets
binding with SPA fallback. `public/_headers` makes `/assets/*` and `/release/*` immutable.

## Response headers (curl, `Origin: https://example.org` sent)

| Path | Status | Length (identity) | cache-control | etag | content-type | content-encoding with `Accept-Encoding: gzip, br` |
|---|---|---|---|---|---|---|
| `/` | 200 | n/a (HTTP/2 stream) | `public, max-age=0, must-revalidate` | n/a | text/html | br |
| `/assets/index-DelKQhlh.js` | 200 | 377,160 | `public, max-age=31536000, immutable` | `"6bc671d0…"` (weak `W/` when compressed) | text/javascript | br (Chrome negotiated zstd) |
| `/assets/ort-wasm-simd-threaded-DcHrbrbl.wasm` | 200 | 14,239,897 | `public, max-age=31536000, immutable` | `"be5b16aa…"` | application/wasm | br (Chrome got zstd, 3.27 MB) |
| `/release/gidi-finance-v2/2.0.2/tokenizer.json` | 200 | 574,195 | `public, max-age=31536000, immutable` | `"a05cb60f…"` | application/json | br |
| `/models/gidi-finance-v2/2.0.2/model.int8.onnx` | 200 | **57,503,558** | `public, max-age=31536000, immutable` | `"b901afae6d6c73db2a7f03e10ffdef46"` | application/octet-stream | **none** (stays identity, so `content-length` is present and download progress is exact) |

- Model route also verified (local `wrangler dev` and live): HEAD, `Range` (206 + `content-range`),
  `If-None-Match` (304), 404 for unknown keys, 405 for non-GET/HEAD. A full GET downloaded 57,503,558
  B and its sha256 equals the release's.
- **CORS**: no `access-control-*` headers are sent and none are needed: page, Web Worker, wasm and
  model are all same-origin.
- **COOP/COEP**: not sent; `crossOriginIsolated` is `false`. Not needed: the runtime uses
  single-threaded ORT wasm (no SharedArrayBuffer). Multi-threaded wasm would require both headers.
- Cloudflare injects its **Web Analytics beacon** (`static.cloudflareinsights.com/beacon.min.js`,
  `POST /cdn-cgi/rum`, 300 B) into the HTML. That is a zone-level auto-injection I did not configure
  and did not touch; it never sees notes (inference is local), but it is a third-party request. Turn
  it off in the Cloudflare dashboard (Web Analytics) if that matters.

## First-load download (browser resource timing + worker network events)

Desktop run, bytes over the wire (what the browser received, Chrome negotiated zstd/br):

| Resource | Bytes |
|---|---:|
| HTML document | 1,126 |
| `index-*.js` (app) | 123,733 |
| `index-*.css` | 13,428 |
| `browser-backend-*.js` | 2,240 |
| 3 Geist woff2 subsets used | 54,816 |
| `web-release.json` (page) | 501 |
| worker script `worker-*.js` | 31,527 |
| worker import `load-local-*.js` (size from curl, zstd) | 30,813 |
| `config.json` | 1,415 |
| `tokenizer.json` | 101,611 |
| ORT wasm (14.2 MB raw) | 3,267,370 |
| **model** | **57,510,174** (57,503,558 body + headers/framing) |
| **Total** | **61,138,754 (61.1 MB)**, 94% is the model, 5% is wasm |

The 390px run is the same within 0.1% (61,108,396 B without the curl-sized chunk).
Repeat visits: assets and the model are served `immutable`, so the browser may keep them in its HTTP
cache; I did not measure a warm-cache reload separately.

## Timings in the browser (fresh context, no cache)

| | Desktop 1280x900 | Mobile 390x844 (emulated) |
|---|---:|---:|
| HTML `load` event | 0.92 s | 0.38 s |
| First model progress visible (from navigation) | 2.3 s | 1.1 s |
| Model download (request to last byte) | 1.90 s (TTFB 0.27 s, ~35 MB/s) | 2.08 s (TTFB 0.39 s) |
| After the last model byte until UI ready | 0.48 s | 0.58 s |
| Navigation start to UI ready (usable) | 4.4 s | 3.4 s |
| Runtime's own "loaded in" (shown in UI) | 3.07 s | 3.05 s |
| First inference (UI-measured, includes worker round trip) | 39.1 ms | 38.3 ms |
| Warm latency, 34 notes, median / p95 / min / max | **19.3 / 21.2 / 13.8 / 23.3 ms** | 19.3 / 22.3 / 13.8 / 27.9 ms |

How the phases were split: the runtime does not report phase times, so they are derived from
worker network events (CDP `Network.*` on the worker) and DOM polling. The worker fetches
`tokenizer.json`/`config.json`, then the model, then verifies sha256, then fetches the ORT wasm,
then creates the session. "Last model byte until ready" therefore contains sha256 + wasm
download/compile + session creation; the wasm response ended ~2-19 ms before the UI was ready, so
session creation itself is on the order of 10 ms. **sha256 verify** (WebCrypto over the 57.5 MB
file, measured separately in the page) took ~31 ms. Download and wasm fetch are sequential; fetching
the wasm in parallel with the model would save ~0.5 s of first load (not done).

Latency is `performance.now()` around the worker `predict()` round trip, so it includes
postMessage. Python native INT8 for comparison is ~1-2 ms; the browser is ~15-20x slower per note
(single-thread wasm), still imperceptible.

Memory (desktop): main-thread JS heap 13.9 MB at ready; worker JS heap 3.5 MB with a 57.6 MB
ArrayBuffer backing store (the model bytes) at ready. The renderer process that hosts the page and
worker was **~617 MB RSS after load and ~635 MB after 70+ inferences** (an empty headless renderer
is ~100 MB), so the model session costs roughly 0.5 GB of resident memory in this browser. That is
a real cost on low-memory phones (see open issues). `performance.memory` is not available inside
workers; heap figures come from CDP `Runtime.getHeapUsage`, RSS from `ps` on the new renderer.

## No server inference

- `GET /api/info` and `GET /api/predict` return the SPA `index.html` (200, text/html);
  `POST /api/predict` returns 405 from the assets layer. There is no Python backend and the Worker
  does not handle `/api`.
- Fresh context with `/api/*` blocked at the network layer (CDP `Network.setBlockedURLs` on page and
  worker): the page loads, the model downloads and verifies, `mượn chú hai 5 xị` returns
  borrow / chú hai / 5 xị. My own probes into `/api` from the page and from the worker failed with
  `Failed to fetch`; the page made no other `/api` request. (Puppeteer's generic request
  interception made the worker's model fetch hang in 4 of 6 loads, 0 of 6 without it, so the
  block list uses CDP instead. That is a harness artifact: no hang in those 6 plain loads or in the 4 full-scenario runs.)

## UI smoke (all on the live site)

Passed on desktop and 390px: loading state with progress (`*-loading.png`), ready banner with
"Runs locally in your browser" + sha prefix + load time, all 10 presets, target and value
highlighting (`mark` elements, labels target/value), Known limitation badge only on the Nam
preset, raw JSON panel, Copy (clipboard text equals the JSON panel, button shows "Copied"), Clear
(textarea emptied, back to "No result yet"), submit by Enter and by the Run button, Enter on empty
note shows "Nothing to parse", Diagnostics tab (shows model sha256, load time, backend, opset 17),
light and dark themes, no horizontal overflow at 390px, no console errors.

Error states (tampering with `web-release.json` through CDP `Fetch`, then clearing the tamper and
clicking **Retry**):

- wrong sha256 -> "Model integrity check failed" with expected vs downloaded hash, header badge
  "Model not loaded", Run and presets disabled; Retry loads the model and inference works.
- model URL 404 -> "Could not load the model" with `HTTP 404` detail; same recovery via Retry.

Screenshots (`web-screenshots/`): `desktop-*` and `mobile-390-*` (loading, ready-light,
preset 0/2/9 light, preset 9 dark, diagnostics), `error-sha-mismatch.png`, `error-model-404.png`,
`api-blocked-inference.png`.

## Parity of the 10 presets with the Python INT8 runtime

Python reference: `GidiPredictor.from_bundle` over the release's model/config/tokenizer
(onnxruntime 1.30.0; browser uses onnxruntime-web 1.30.0). The raw JSON panel was compared with
`Prediction.to_dict()` for each preset.

| Preset | type / target / value (browser = Python) | max \|Δconfidence\| |
|---|---|---:|
| cơm tấm 100 | expense / none / 100 | 5.4e-7 |
| bún bò 50k | expense / none / 50k | 2.2e-7 |
| mượn chú hai 5 xị | borrow / chú hai / 5 xị | 5.9e-6 |
| ăn 2 tô phở 70 | expense / none / 70 | 8.2e-7 |
| tiền điện tháng 10 hết 612 | expense / none / 612 | 3.0e-7 |
| trả góp kỳ 3 1tr5 | repayment_out / none / 1tr5 | 2.5e-6 |
| 20/10 mua quà mẹ 500 | expense / none / 500 | **2.8e-4** |
| mua 3 vé 150k | expense / none / 150k | **2.2e-2** |
| đổ 5 lít xăng 120k | expense / none / 120k | 2.2e-7 |
| cho a Nam vay 1 triệu 20/10 | lend / Nam / `1 triệu 20/10` (+ Known limitation badge) | 2.2e-7 |

All discrete fields (type, target, spans, value text and spans, truncated) are exactly equal for
all 10. **Two presets miss the 1e-4 confidence tolerance of the contract's parity gate**:
`mua 3 vé 150k` type_confidence 0.9593 (browser) vs 0.9374 (Python), and
`20/10 mua quà mẹ 500` target_confidence 0.99925 vs 0.99897. Same ORT version on both sides, so
this is INT8 kernel numerics (wasm vs native arm64 MLAS), not a port bug; the predicted labels
of the 10 presets are unaffected. Across the whole corpus, `web-parity.md` (WebRuntime) reports
2388 inputs, 349 with a confidence delta beyond 1e-4 and 1 input where a discrete field differs
because the ORT logits differ (TS decoding of the Python logits is exact). That report is the
authority for parity; this page only covers the 10 presets in the deployed browser.

## Open issues

1. Two preset confidence deltas above 1e-4 (above); corpus-wide numbers are in `web-parity.md` (349/2388 confidence-only, 1/2388 discrete flip). The 10 presets are exact in every discrete field.
2. ~0.6 GB resident memory for the renderer after loading the model; first load is 61 MB. Fine on
   desktop, likely heavy on low-end phones; not tested on a real phone.
3. The ORT wasm is fetched only after the model finishes (about 0.5 s of avoidable serial time).
4. Cloudflare Web Analytics beacon is auto-injected on the zone (see above).
5. `pnpm deploy` is a pnpm built-in; the script is run with `pnpm run deploy`.
