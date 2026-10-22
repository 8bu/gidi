# Cost and traffic (gidi.8bu.dev)

Goal: no surprise bill under heavy traffic. Prices below were read on 2026-10-22 from the pages
linked in "Sources". The site is on the Workers **Paid** plan (account "71Z", renews monthly; R2 Paid
too), the `8bu.dev` zone is on the Free plan.

## Result in one table

| | before (Worker `gidi` + R2) | now (assets only) |
|---|---|---|
| Worker invocations, first visit | 2 (`/`, model) | 0 |
| Worker invocations, returning visit | 1 (`/`) | 0 |
| 200 requests (100 `/` + 100 model range reads) | 206 invocations, 208 ms CPU, 107 cache subrequests | 0 invocations, 0 ms CPU, 0 subrequests (`/`: 99 of 100 edge HIT; model range reads are served by the asset service, free whatever the cache status) |
| 3 first + 5 returning visits (62 requests) | 11 invocations, 10.5 ms CPU, 8 subrequests | 0 invocations, 0 ms CPU, 0 subrequests |
| R2 operations per visit | 0 on an edge-cache hit, 1 Class B per data-centre miss | 0 (R2 is no longer read) |
| Workers Logs / traces | on, 100 % sampled | off (there is no Worker code to log) |
| PostHog events per visit | about 5 | about 3 |

"Before" was measured on 2026-10-22 against production with `playground/scripts/traffic-probe.mjs`
(Cloudflare GraphQL `workersInvocationsAdaptive`, script `gidi`; real traffic is a few dozen requests a
day, so the counts are the probe's own). Nothing on the page is billed per request any more:
requests to static assets are free and unlimited, egress is free.

## Per-request cost chain

| Step | Served by | Billed |
|---|---|---|
| `/` (HTML, revalidated each visit, 304 when unchanged) | static asset | free |
| `/assets/*` JS, CSS, 14 MB ORT wasm (fingerprinted, immutable) | static asset | free |
| `/release/manifest.json` (`no-cache`, 304 when unchanged) | static asset | free |
| `/release/<id>/<version>/config.json`, `tokenizer.json` (immutable) | static asset | free |
| `/models/<id>/<version>/model.int8.onnx.01/.02` (16 MiB parts, immutable) | static asset | free |
| Geoip language hint | zone Response Header Transform Rule | free (Free plan) |
| PostHog events and errors | PostHog (client side) | PostHog pricing |

- (a) **First visit**: HTML + manifest + JS/CSS/wasm (about 15 MB) + 2 model parts (about 29 MB) for
  the visitor's language: about 45 MB of free egress, 0 billed operations. Then the verified model is
  kept in Cache Storage.
- (b) **Returning visit**: HTML and manifest revalidate (304, a few hundred bytes); JS, CSS, wasm
  and the model come from the browser's HTTP cache or Cache Storage: no model download.
- (c) **Language switch**: the other language's two parts (29 MB) once; afterwards Cache Storage.

## Worst case (monthly visits)

Assumptions: 50 % first visits; PostHog 3 events a visit now (`$pageview`, `model_loaded`, about one
`notes_sent`), about 5 before (also `$pageleave` and `stack_opened`). Cloudflare numbers assume nothing
else on the account; the 10 M request and 30 M CPU-ms allowances are shared with the other Workers of the
account (this month: 128 k requests, 231 k CPU ms, R2 12 k Class B, 0.44 GB-month: all inside the free
tiers, Billing -> Billable usage).

| Visits / month | Workers + R2 before | Workers + R2 now | PostHog before (5 ev.) | PostHog now (3 ev.) |
|---|---|---|---|---|
| 100 k | $0 | $0 | $0 | $0 |
| 1 M | $0 | $0 | about $153 | about $84 |
| 10 M | about $1.50 | $0 | about $1,528 | about $938 |

(Plus the fixed $5 Workers Paid plan, unchanged.) PostHog: first 1 M events free, then $0.00005,
$0.0000343 from 2 M, $0.0000295 from 15 M. **PostHog is now the only metered cost**, and only a billing
limit (below) caps it: anyone can post events to the public project key.

Denial of wallet, 1,000 requests a second to `/` for a month (2.6 billion): before about $830
($0.30 per million requests over 10 M, $0.02 per million CPU ms over 30 M); now $0: static assets are
not billed, so no rate-limit rule is needed to protect the bill. Scanners hitting other paths get the
SPA page from the same free asset service (they never reached R2 or a Worker).

## What changed

- **No Worker.** `playground/wrangler.jsonc` has no `main`, no R2 binding and observability off; the
  Worker (`worker/index.ts`) is deleted. The Free-plan trap (a `run_worker_first` route answers 429 once
  the 100 k a day request limit is hit) is gone with it.
- **Models as static assets.** `scripts/sync-release.mjs` cuts each model into 16 MiB parts (assets are
  limited to 25 MiB a file); the browser downloads them in parallel, joins them and checks the
  sha256 of the whole file. R2 `gidi-models` is untouched (archive, 0.44 GB-month, inside the free 10 GB).
- **Geoip default without a Worker.** Zone rule "gidi geo hint (Server-Timing)" on `8bu.dev`: when
  `http.host eq "gidi.8bu.dev" and http.request.uri.path eq "/"`, set the response header `Server-Timing`
  to `concat("geo;desc=", ip.src.country)`. The page reads it from `performance.getEntriesByType
  ("navigation")`. Chromium keeps it on a 304 revalidation (checked). Missing rule = browser language.
  `?geo=` does nothing in production (only in the dev server).
- **Caching.** Fingerprinted assets, release config/tokenizer and model parts are immutable;
  `/release/manifest.json` is `no-cache`; HTML revalidates (default). Verified model bytes live in
  Cache Storage. Rollout, rollback and stale-cache handling: `docs/deployment.md`, "Model rollout
  runbook".
- **PostHog volume** (`src/lib/analytics.ts`): no `$pageleave`, web vitals, rageclick or other SDK
  extras (`before_send` allows only `$pageview`, `$exception` and our events); `model_loaded` once per
  language per page load, `stack_opened` once per type, `model_load_failed` at most 3; 200 events a page
  load at most. Errors: `docs/error-tracking.md`.
- **Checks:** `scripts/traffic-probe.mjs` (invocations and cache status, before/after) and
  `scripts/verify-release.mjs` (manifest, headers, part sizes, sha256, geo header).

## Risks that remain

1. PostHog events and exceptions (above): capped by the billing limits below.
2. Adding Worker code again (a `main` in `wrangler.jsonc`) brings per-request billing back, and with
   `run_worker_first` on `/`, an attacker's flood. Keep new logic in the client or in a rule.
3. Deploys remove old fingerprinted files; an open old tab that lazy-loads a removed chunk reloads
   itself once (`vite:preloadError`) and reports it.
4. Static asset limits: 20,000 files and 25 MiB a file on Free, 100,000 on Paid (now about 40 files).
5. About 3 minutes on 2026-10-22 an immutable `web-release.json` in the new format was served at the
   old path; browsers that fetched it in that window hold it. The new code never reads that path.

## Billing caps and alerts (done 2026-10-22)

1. **PostHog billing limits** (https://us.posthog.com/organization/billing): Product analytics
   **$25 a month**, Error tracking **$5 a month**. Every other product (Session replay, Destinations,
   Feature flags & Experiments, Surveys, Sources, AI Observability, Logs, PostHog AI, Replay vision,
   Self-driving inbox, PostHog Desktop, Workflows) already had the lowest limit, **$1 a month**, and no
   usage; left as is. Data over a limit is dropped (https://posthog.com/docs/billing/limits-alerts), so
   at about 1 M visits a month the $25 cap would cut events before the $84 projection above.
2. **Cloudflare budget alerts** (Manage account -> Billing -> Billable usage -> Budget alerts): the
   existing $10 alert ("Default budget alert (auto-created)") plus a new "Billing Budget Alert $25"
   (threshold $25, email hvanlong@pm.me). Platform Notifications also hold account-wide "Usage based
   billing" alerts for Workers requests, D1 usage and D1 writes (email), so a Workers request spike is
   covered; no separate alert was added.
3. **Zone rule kept**: `8bu.dev` -> Rules -> Response Header Transform Rules -> "gidi geo hint
   (Server-Timing)" is still there and Active (`http.host eq "gidi.8bu.dev"` and path `/`, `Server-Timing`
   = `concat("geo;desc=", ip.src.country)`). Deleting it only changes the first-visit default language
   to the browser's.
4. **PostHog alerts**: set up, see `docs/error-tracking.md` ("Alerts").

## Sources

- Workers pricing and limits: https://developers.cloudflare.com/workers/platform/pricing/ ,
  https://developers.cloudflare.com/workers/platform/limits/
- Static assets are free and unlimited: https://developers.cloudflare.com/workers/static-assets/billing-and-limitations/
- R2: https://developers.cloudflare.com/r2/pricing/ (egress free, 10 M Class B and 10 GB-month free)
- Observability: https://developers.cloudflare.com/observability/pricing/
- Response header transform rules: https://developers.cloudflare.com/rules/transform/response-header-modification/
- PostHog: https://posthog.com/pricing , https://posthog.com/docs/billing/limits-alerts
