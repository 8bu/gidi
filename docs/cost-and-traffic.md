# Cost and traffic (gidi.8bu.dev)

Goal: no surprise bill under heavy traffic. Prices below were read on 2026-10-22 from the pages
linked in "Sources". The site is on the Workers **Paid** plan (account "71Z", renews monthly; R2 Paid
too), the `8bu.dev` zone is on the Free plan.

## Result in one table

| | before (Worker `gidi` + R2) | now (assets only) |
|---|---|---|
| Worker invocations, first visit | 2 (`/`, model) | 0 |
| Worker invocations, returning visit | 1 (`/`) | 0 |
| 200 requests (100 `/` + 100 model reads) | 206 invocations, 208 ms CPU, 107 cache subrequests | 0 invocations, 0 ms CPU, 0 subrequests (`/`: 99 of 100 edge HIT; the model reads were ranged GETs, which the asset service answers in full, see "Load test 2026-10-22"; served by the asset service, free whatever the cache status) |
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
- **Checks:** `scripts/traffic-probe.mjs` (invocations and cache status, before/after),
  `scripts/verify-release.mjs` (manifest, headers, part sizes, sha256, geo header) and
  `scripts/load-test.mjs` (concurrent clients, latency percentiles, sha256 of whole model downloads).

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

## Load test 2026-10-22

**Method.** `playground/scripts/load-test.mjs`, repeatable:
`node playground/scripts/load-test.mjs <url> --duration 60 --concurrency 50 --scenarios html,manifest,manifest-304,js,model --script <worker> --json out.json`.
Production was not loaded. The target was a throw-away copy of the deployed `dist/` (the bytes of production
version `1784a26c`, commit `8ea1a0a`) as a separate assets-only Worker `gidi-loadtest` on
`https://gidi-loadtest.nguyenhoang.workers.dev` (`workers_dev: true`, no route, no custom domain; version
`1326b67d-d758-4de6-89c2-6ed48f4f0688`; config in `/tmp`, not in the repo). It was deleted afterwards
(`wrangler delete`; the hostname answers 404 and the account's Worker list no longer shows it).
Production received about 210 requests in the whole session: `verify-release.mjs --hash`, a browser smoke test,
`traffic-probe.mjs --burst 50` (100 requests) and 22 conditional manifest GETs.

Each scenario ran 50 closed-loop clients for 60 s one after the other (a request or model download in flight when
the time ran out was allowed to finish), from one macOS arm64 laptop in Vietnam with Node 26.3's built-in `fetch`.
The laptop's link was measured at 7.3 MB/s for one stream and 10.2 MB/s on speed.cloudflare.com; the edge
answered from Singapore (SIN) and Hong Kong (HKG). The window was 2026-10-22 16:24 to 16:32 UTC.

| scenario (50 clients, 60 s) | requests | errors | req/s | p50 / p95 / p99 ms | TTFB p50 ms | cf-cache-status |
|---|---|---|---|---|---|---|
| HTML `/` (11,727 B) | 17,062 | 0 | 279 | 109 / 453 / 956 | 88 | HIT 17,015, MISS 47 |
| manifest, 200 (no validators, 1.4 kB) | 19,095 | 0 | 312 | 92 / 398 / 794 | 84 | HIT 19,042, MISS 53 |
| manifest, 304 (`If-None-Match`, 0 B body) | 22,061 | 0 | 362 | 82 / 358 / 699 | 82 | HIT 22,010, MISS 51 |
| main JS chunk (232,930 B decoded, 35 MB/s decoded) | 9,153 | 0 | 151 | 239 / 765 / 1,355 | 126 | HIT 9,133, MISS 20 |
| whole model: 2 parts, sha256 checked (28,658,533 B) | 50 downloads | 0 | n/a | 117,571 / 130,496 / 143,284 per download | 421 | MISS 37, HIT 13 (second part) |

67,471 requests, 0 errors (no network error, no unexpected status, no sha256 mismatch), about 3.8 GB (decoded
bytes) moved.

- **Worker invocations and R2 (Cloudflare GraphQL, window 16:23:29 to 16:32:30 UTC):** script `gidi-loadtest`
  0 invocations, 0 CPU ms, 0 subrequests; account-wide R2 operations: 1, a `ListBuckets` from the account
  audit below (an API call of mine), not a read. The other Workers of the account logged 16 invocations in the
  same window, so the API had caught up when it reported 0.
- **Production probe** (`traffic-probe.mjs https://gidi.8bu.dev --first 0 --returning 0 --burst 50`, 16:32 to
  16:34 UTC): 50 GET `/` (50 edge HIT) and 50 ranged GETs of a model part (MISS 34, HIT 16), all 200; script
  `gidi`: 0 invocations, 0 CPU ms, 0 subrequests (polled eight times over about 7 minutes).
- **Model parts under parallel clients.** 50 clients downloaded both parts at once, joined them and matched the
  manifest's sha256: 50 of 50 correct, no truncation or error. The aggregate stayed at 10 MB/s because the
  laptop's link is the limit (a download took about 2 minutes each), so this shows the edge serves 50 full
  parallel downloads cleanly, not where the edge tops out.
- **Manifest 304 path.** 22,061 answers of 304 with no body and `cf-cache-status: HIT`, 362 requests a second,
  p50 82 ms, equal to its time to first byte, so the round trip is the whole cost (the 200 path is 92 ms and carries 1.4 kB). On production, 20
  sequential conditional GETs (each on a new TLS connection): 20 of 20 `304`, 0 B, HIT, 0.29 to 0.46 s each.
  It costs nothing: no Worker, no R2.
- **`Range` is ignored by the asset service.** `Range: bytes=0-1023` on a model part returns `200`, the whole
  11,881,317 B, and no `accept-ranges`. The model reads in the "now" column of "Result in one table" were ranged
  GETs, so they moved whole parts, not 1 KiB. Invocations and cost are unaffected (egress is free), but a burst of 50 moves about
  600 MB. `traffic-probe.mjs` now sends a HEAD for the model file in the burst.

**Limits of the test.**

- One client machine on one residential link, two edge locations, closed-loop (a new request only after the
  last answer), 60 s a scenario. HTML, manifest and JS numbers include my round trip (about 80 ms to the
  first byte) and, for 50 connections at 300 requests a second, one Node process; the JS and model rows are
  bound by the 10 MB/s link, so their latency is the link, not the edge.
- The target was `workers.dev`, not `gidi.8bu.dev`: same asset service and same bytes, but not the `8bu.dev`
  zone, so zone rules (the geo header, any future WAF or rate limit) were not exercised. Production only got the
  small probe above.
- No browser: no ORT/WASM start, no PostHog traffic, no Cache Storage. Nothing here says how fast a phone loads the
  model, only that the edge delivers it correctly under parallel load.
- Cloudflare's own capacity was never stressed; "0 invocations" is the cost result, and the latency figures are
  bounded by the client, not by the edge.

**Fair use of bandwidth.** No bandwidth cap or fair-use figure for static assets is published. Workers Paid
says: "There are no additional charges for data transfer (egress) or throughput (bandwidth)"
(https://developers.cloudflare.com/workers/platform/pricing/); requests to static assets are "free and
unlimited" (https://developers.cloudflare.com/workers/static-assets/billing-and-limitations/); the limits page
lists only 20,000 files (Free) or 100,000 (Paid) a version and 25 MiB a file
(https://developers.cloudflare.com/workers/platform/limits/#static-assets). The one content rule is the CDN
clause of the Service-Specific Terms (https://www.cloudflare.com/service-specific-terms-application-services/):
for Free, Pro and Business zones, "Cloudflare offers specific Paid Services (e.g., the Developer Platform,
Images, and Stream) that you must use in order to serve video and other large files via the CDN", and Cloudflare
may limit use "if you use or are suspected of using the CDN without such Paid Services to serve video or a
disproportionate percentage of pictures, audio files, or other large files". The models are served by the Workers
Developer Platform (static assets, account on Workers Paid), so [INFERENCE] they fall under the allowed
case; the blog post that moved the rule out of the old section 2.8 says the same for content hosted by Cloudflare
services (https://blog.cloudflare.com/updated-tos/). The general terms (section 2.2.1 of the Self-Serve
Subscription Agreement, https://www.cloudflare.com/terms/) still forbid an "undue burden" on the Services. At
1 M first visits a month the site would move about 45 TB; there is no published threshold, so the risk is a
notice, not a bill. Keeping parts immutable and in Cache Storage (already done) is the mitigation.

**Account check (2026-10-22, read-only, nothing deleted).** Sources: Cloudflare API and GraphQL with the
wrangler token, and Billing -> Billable usage (cycle 13 Oct to 12 Nov, 10 of 31 days observed: total $0.00,
projected $0.00, every line inside its included tier).

| Product | Inventory | Recent usage |
|---|---|---|
| Workers (18 after deleting `gidi-loadtest`) | 71z-demos, 71zone, bytelane, cookdi, crewdoku, customer-portal-api, customer-portal-web, cvspire, cyclone-portfolio, gidi, meet71, portf-api, portf-web, quet-web, sk-prototype, uptimeflare-8bu-worker, vietlott, vldex-web | cycle: 127.9 k requests of 10 M, 231 k CPU ms of 30 M. Last 30 days (GraphQL): 352,594 requests; `gidi` 383 requests, 453 CPU ms (the Worker era, before the assets-only cut-over) |
| R2 (9 buckets) | 71media, 71z-backups, camen-updates, customer-portal-branding, cvx, ght, gidi-models, obsidian-relay, sv-media | cycle: Class B 12.4 k of 10 M, Class A 272 of 1 M, 0.44 GB-month of 10; stored now 2.34 GB (71z-backups 1.27, camen-updates 0.79, gidi-models 0.14) |
| D1 (4) | meet71-db, quet-web, uptimeflare-8bu, m8t-staging | cycle: 2.65 k rows written, 5.57 k read |
| KV (2) | 71z-demos-content, BOOKINGS | cycle: 290 writes, 2.81 k reads |
| Durable Objects (1) | `cookdi_Room` (script `cookdi`) | 2,305 requests in 30 days, SQL storage 0 |
| Queues, Vectorize | none | none |
| Other | Hyperdrive `portal-hd`, `portf-hd-vps`; Pages project `uptimeflare-8bu`; zones `71.zone` and `8bu.dev`, both Free | Workers AI "Regular Twitch Neurons" 164.6 k, Email Service 16 sent, both $0.00 |

Unexpected on that day, since resolved in "Account audit 2026-10-22" below: (1) `__unknown__`, 116,260 requests and
541.6 s of CPU (33 % and 53 % of the account's 30-day Workers totals), is the deleted `skive-8bu-dev` Worker; (2)
`portf-api`'s 79 % errors are a cron trigger with no handler; (3) the 164.6 k Workers AI neurons are REST calls to
`@cf/cloudflare/clef`. None of these touches Gidi, and no line is billed. The wrangler OAuth token has no billing
scope (`/subscriptions` answers 403), so the plan and invoices come from the dashboard.

## Account audit 2026-10-22

Run 2026-10-23 (UTC), read-mostly: Cloudflare GraphQL (`workersInvocationsAdaptive`, `aiInferenceAdaptiveGroups`,
`httpRequests*Groups`) and REST with the wrangler OAuth token, `wrangler tail`, and the dashboard (Billing ->
Billable usage, Subscriptions, Invoices, WAF) in BrowserOS. GraphQL counts are adaptive (rounded to tens), window
2026-09-23 to 2026-10-23. Billing cycle 13 Oct to 12 Nov, 11 of 31 days observed: **total $0.00, projected
$0.00**, every line inside its included tier. The three items of the earlier check, resolved:

| | what it is | owner project | cause |
|---|---|---|---|
| `__unknown__` | the deleted Worker `skive-8bu-dev`, an allowlist reverse proxy on `skive.8bu.dev` | `~/Projects/skive` (`wrangler.jsonc`: custom domain `skive.8bu.dev`) | one 9-hour burst on 2026-10-03, nothing since |
| `portf-api` 79 % errors | Worker behind `8bu.dev/api/*` (chat, retrieve, blog, stats) | `~/Projects/8bu.dev/services/api` | a `*/5 * * * *` cron trigger with no `scheduled()` handler; user requests never errored |
| 164.6 k neurons | not a Worker: REST calls to `@cf/cloudflare/clef` | `~/Projects/clef-browse-fast` (the `clef-browse` / `browser_task` MCP tool, created 2026-10-10) | an agent tool calling Workers AI with an account API token, about 800 calls a day |

### 1. `__unknown__`: 116,450 requests, 550.6 s CPU (33 % and 53 % of the 30-day totals)

- **All of it is 2026-10-03, 08:00 to 16:59 UTC** (peak 28.7 k requests in the 15:00 hour), 116,390 `success` and 60
  `clientDisconnected`, about 4.7 ms CPU a request, nine script versions of one Worker. Nothing before or after.
  Zero requests since, so it is not a live cost.
- **Same day on the zone:** `8bu.dev` had 141,483 requests that day (normally 2,000 to 5,000); host `skive.8bu.dev`
  alone had 103,527 answered 200, 9,640 answered 530 and 9,617 answered 302. Top paths are the proxied sites
  (`/choigame/*`, `/gamevui2/*`, `/crazygamevn/*`, `/coolmath/*`, `/yaw/*`, `/lichess/*`) fetched by Chrome from VN
  and IN, plus `ClaudeBot` and `GPTBot`. Nine version ids of `__unknown__` match no version of any live Worker.
- **Why "unknown":** the script no longer exists (not in the Worker list; `skive.8bu.dev` no longer resolves), so
  GraphQL has no name to give it [INFERENCE]. The repo's `wrangler.jsonc` names it `skive-8bu-dev`. After the burst the host
  returned 530 for 5 to 9 k requests a day until the DNS record went away; the last 23 hours have none.
- **Routes:** every route on `8bu.dev` points to an existing script (`portf-api`, `portf-web`,
  `customer-portal-api`, `customer-portal-web`, `vldex-web`). There is no stale route and no dispatch namespace
  (Workers for Platforms is not enabled on the account).
- **Cost:** 116 k requests and 550 k CPU ms are 1.2 % and 1.8 % of the monthly allowance, $0.00. A day like that
  10x is 1.2 M requests and 5.5 M CPU ms: still $0.00. 100x (11.6 M requests, 55 M CPU ms in a month) is about
  $1.00 ($0.48 requests + $0.50 CPU).
- **If it is ever redeployed,** give it `limits.cpu_ms` and a per-IP limit first: it was an open (allowlisted)
  proxy whose traffic included AI crawlers.

### 2. `portf-api`: 7,970 of 10,450 requests in 30 days were `scriptThrewException`

- **Cause (seen with `wrangler tail`):** the event is `{"cron":"*/5 * * * *"}` and the exception is `Handler does
  not export a scheduled() function`. The cron is on the deployed script (`GET .../scripts/portf-api/schedules`),
  not in `wrangler.toml`, and nothing in the repo or its git history mentions it. 288 ticks a day is exactly the
  250 to 300 daily exceptions since 2026-09-23. Every real request (about 2,470 in 30 days, mostly scanners
  asking for `/api/.env`, `/api/v1/workflows` and the like, answered 404 by the app) was `success`.
- **Fixed and deployed 2026-10-23** (8bu.dev repo, commit `dc64877`, version `a55ca576`): `[env.portf.triggers]
  crons = []`; `schedules` is now empty. One more tick still fired at 10:35 UTC (trigger removal took a few
  minutes to reach the scheduler); the 10:40 and 10:45 UTC five-minute buckets have none, where every bucket
  before the deploy had one.
- **Cost:** the failing ticks were 8.6 k invocations a month, $0.00; the problem was noise, not money.

### 3. Workers AI: 180.4 k neurons in the cycle so far (164.6 k a day earlier), billed $0.00

- **By model** (`aiInferenceAdaptiveGroups`, since 13 Oct): `@cf/cloudflare/clef` 8,913 calls with `requestSource` =
  `rest api`, 8.45 M input tokens (output is not billed); a further 46 `llama-4-scout` calls (129,758 input
  tokens, 13 Oct) from a binding. Clef lists $0.24 per M input tokens: 8.45 M tokens is $2.03, about 184 k
  neurons at $0.011 per 1,000, which matches the dashboard's 180.44 k "Regular Twitch Neurons" within 2 %.
  The 10-day run is 0.5 to 1.8 M tokens a day (about 0.7 M at present).
- **Who:** a REST caller, so no Worker binding is involved [INFERENCE from `requestSource`, the model name, and the
  start date: `clef-browse-fast` was created 2026-10-10 as "browser task runner for BrowserOS Neo with Clef
  decisions" and reads `CLOUDFLARE_ACCOUNT_ID` / `CLOUDFLARE_AUTH_TOKEN`; the account's first clef and clef-flash
  calls are 2026-10-10]. The API does not say which token.
- **Workers with an `ai` binding:** `portf-api` (bge-m3 embeddings: 84 calls and 597 input tokens in 30 days,
  negligible) and `71z-demos` (a few llama, gemma, mistral and glm calls on 10-10 to 10-13, under 160 k tokens;
  its endpoints were not audited).
- **Cost at the current rate:** about 0.7 M clef tokens a day = $0.17 a day, about $5 a month
  before the free 10,000 neurons a day. 10x is about $50 a month, 100x about $500. Nothing on the account caps
  Workers AI spend; the token is the only gate.

### Public endpoint that can burn Workers AI: `portf-api` `POST /api/chat` and `POST /api/retrieve`

- Both are public and unauthenticated; each call embeds the text (`@cf/baai/bge-m3`, $0.012 per M input tokens).
  A 2,000-character body is at most about 700 tokens, so one call costs at most about $0.0000084; a chat turn
  embeds at least twice. `/retrieve` also upserts a row into `unanswered` on a miss (an unbounded table).
  `/api/blog` (5-minute in-isolate cache, Substack fetch) and `/api/stats` (three counts) cost no neurons.
- **The existing limiter does not limit.** `wrangler.toml` binds `CHAT_LIMIT` (20 per 60 s) and the code treats
  it as a hard gate, but measured on the live API from one IP: 30 sequential `/chat` calls, 64 sequential
  `/retrieve` calls and 120 parallel `/retrieve` calls in 3.9 s were all `200`, none `429`. A throw-away Worker
  with the same binding (limit 20, period 60) denied the 21st call inside one invocation but denied none of 40
  separate requests. Cloudflare documents it: "Rate limits are approximate and apply separately in each Cloudflare
  location. Requests can exceed your configured limit while counts update in the background"
  (https://developers.cloudflare.com/workers/runtime-apis/bindings/rate-limit/), with Durable Objects as the strict option. The probe
  Worker (`acct-audit-rl-probe`) was mine and is deleted.
- **Worst case for the account, at 38 requests a second sustained (100 M a month):** about $30 in Workers requests
  plus bge-m3 at the maximum body size about $840 less the free 10,000 neurons a day (about $3 a month), so
  roughly $0.9 k; at 10x the current traffic (about 100 k requests a month) it is cents. Realistic exposure is
  small because the API is only called by the site's own pages, but nothing stops a script.
- **Changed:** `/retrieve` now runs behind the same limiter as `/chat` (commit `dc64877`, with a test); that
  caps one client at roughly the binding's accuracy, not strictly. **Not changed (owner, zone rules):** `8bu.dev`
  has no rate-limiting rule (WAF -> Rate limiting rules is empty); the Free plan allows one. A rule on
  `http.request.uri.path in {"/api/chat" "/api/retrieve"}` and method POST, 10 requests per 10 s per IP, action
  block, gives a strict, shared counter at no cost. I did not create it: the wrangler token has no zone-rules
  scope and zone rules are the owner's call.

### Other changes in `portf-api` (same commit and deploy)

- `[env.portf.limits] cpu_ms = 500` (observed p99 is about 75 ms; the default is 30,000 ms). Wall-clock time
  (SSE) is not limited.
- Smoke after deploy, against production: `/api/healthz` 200 (`db: up`), `/api/stats` 200, `/api/blog` 200,
  `/api/.env` 404, `POST /api/chat` 200 (SSE stream), `POST /api/retrieve` 200. API gates: typecheck, lint and
  112 vitest tests pass; `wrangler deploy --dry-run` clean. The commit is local in `~/Projects/8bu.dev`
  and not pushed (the branch already had an unpushed commit of yours and uncommitted `apps/web` edits, which I left).

### Everything billable on the account

Subscriptions (dashboard): **Workers Paid** (invoice 13 Oct: $5.00), **R2 Paid**, **Images Stream Basic**,
**Realtime**, Zero Trust Teams Free; zones `71.zone` and `8bu.dev` on Free; payment methods on file: two cards. Invoices:
13 Oct $5.00, 4 Oct $1.50, 13 Sep to 13 Apr $0.00, 13 Feb and 13 Jan $6.00 each. I could not attribute the 4 Oct $1.50
(the invoice detail is a PDF) and the Images Stream Basic and Realtime plans show no usage line; the Images, Stream
and AI Gateway dashboard pages did not render in a background tab and the token has no scope for them.

| Product | Inventory | Usage this cycle (11 days) | Included, then price | 10x / 100x of a month at this rate |
|---|---|---|---|---|
| Workers requests | 18 Workers (`71zone` 129 k, `71z-demos` 38 k, `crewdoku` 16 k, `portf-web` 14 k, `portf-api` 10 k, `uptimeflare-8bu-worker` 8 k in 30 days) | 134.56 k | 10 M, then $0.30 / M | $0 / about $8 |
| Workers CPU | same | 251.52 k ms | 30 M ms, then $0.02 / M ms | $0 / about $1 |
| Workers AI | `portf-api`, `71z-demos` bindings; REST (clef) | 180.44 k neurons | 10 k a day, then $0.011 / 1,000 | about $50 / about $500 a month (clef at the current rate) |
| R2 (9 buckets: `71media`, `71z-backups`, `camen-updates`, `customer-portal-branding`, `cvx`, `ght`, `gidi-models`, `obsidian-relay`, `sv-media`) | 2.34 GB stored | Class B 12.84 k, Class A 300, 0.52 GB-month | 10 M B, 1 M A, 10 GB-month; egress free | $0 / $0 |
| D1 (`meet71-db`, `quet-web`, `uptimeflare-8bu`, `m8t-staging`) | 4 | 2.94 k rows written, 6.16 k read | 50 M written, 25 B read | $0 / $0 |
| KV (`71z-demos-content`, `BOOKINGS`) | 2 | 290 writes, 2.81 k reads, 171 lists, 113 deletes | 1 M, 10 M | $0 / $0 |
| Durable Objects | `cookdi_Room` (`cookdi`) | SQL storage 0 | 5 GB-month | $0 / $0 |
| Email Service | sending enabled | 16 sent | 3,000 | $0 / $0 |
| Queues, Vectorize, Tunnels | none | none | n/a | n/a |
| Hyperdrive | `portal-hd`, `portf-hd-vps` | not metered | included on Paid | n/a |
| Pages | `uptimeflare-8bu` (`uptime.8bu.dev`) | not shown separately in Billable usage | | [INFERENCE] Functions bill as Workers requests |
| Images, Stream, AI Gateway, Zaraz, domain registrations | plan "Images Stream Basic" only | no billable line | flat plan | unknown (see above) |

Workers crons: `portf-api` (now none), `uptimeflare-8bu-worker` (`*/5`, 8,270 invocations in 30 days, all
`success`) and `customer-portal-api` (`*/15`, 2,760, all `success`). Observability at 100 % sampling with
persisted logs: `71z-demos`, `customer-portal-api`, `cookdi` (Workers Logs events are billed past 20 M a month; at
today's volume nothing). Of the eight Workers whose settings I read (`portf-api`, `portf-web`, `gidi`, `71zone`,
`71z-demos`, `uptimeflare-8bu-worker`, `customer-portal-api`, `cookdi`) only `portf-api` has `limits.cpu_ms`, set today.

### What remains for the owner

1. Create the WAF rate-limiting rule above on `8bu.dev` (free plan, one rule), or replace `CHAT_LIMIT` with a
   Durable Object limiter. Either makes the `portf-api` AI exposure strict.
2. Confirm that `~/Projects/clef-browse-fast` is the caller; if so, put its token on a narrow scope (Workers AI read/run
   only) and add a usage-based billing notification in Billing -> Notifications; AI spend has no other cap.
3. Nothing was deleted. Candidates to review, none of them urgent: the `Images Stream Basic` and `Realtime`
   subscriptions (they show no usage; find the $1.50 invoice of 4 Oct), Workers with almost no traffic in 30 days
   (`vldex-web` 10 requests, `cookdi` 110), R2 `71z-backups` (1.27 GB, a backup) and `camen-updates` (0.79 GB).
4. Push `~/Projects/8bu.dev` (commit `dc64877` is local) when `apps/web` is in a state to push.
5. 2026-10-23: the `vldex-web` Worker and its route `vldex.8bu.dev/*` were dropped (no cron, secret or custom domain existed).
   No D1, KV, R2 or Queue was tied to it, so no data store was left; the repo `~/Projects/vldex` is kept. The config backup is at `~/Documents/cloudflare-backups/vldex-web-2026-10-22/` (outside git, no secrets).

## Sources

- Workers pricing and limits: https://developers.cloudflare.com/workers/platform/pricing/ ,
  https://developers.cloudflare.com/workers/platform/limits/
- Static assets are free and unlimited: https://developers.cloudflare.com/workers/static-assets/billing-and-limitations/
- R2: https://developers.cloudflare.com/r2/pricing/ (egress free, 10 M Class B and 10 GB-month free)
- Observability: https://developers.cloudflare.com/observability/pricing/
- Response header transform rules: https://developers.cloudflare.com/rules/transform/response-header-modification/
- PostHog: https://posthog.com/pricing , https://posthog.com/docs/billing/limits-alerts
- Cloudflare Service-Specific Terms (CDN): https://www.cloudflare.com/service-specific-terms-application-services/
  ; Self-Serve Subscription Agreement: https://www.cloudflare.com/terms/ ; Terms update (old section 2.8):
  https://blog.cloudflare.com/updated-tos/ ; Workers limits: https://developers.cloudflare.com/workers/platform/limits/
