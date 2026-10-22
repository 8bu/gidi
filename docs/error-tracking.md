# Error tracking (PostHog)

Failures on the page reach the owner as PostHog `$exception` events (project "Gidi (gidi.8bu.dev)",
id 648258, US cloud). Code: `playground/src/lib/error-report.ts` (context, scrubbing, limits),
`analytics.ts` (`captureError`, `before_send`), `main.tsx` (page-level listeners). Session recording,
autocapture and heatmaps stay off.

## What is captured

| Stage tag | Where it comes from | `kind` examples |
|---|---|---|
| `boot` | uncaught errors and unhandled rejections, a script/style/chunk that failed to load, React render errors (`ErrorBoundary`) | `uncaught`, `unhandled_rejection`, `resource_load`, `chunk_load`, `react_render` |
| `load` | manifest fetch/parse, model download and sha256 check, worker start or crash, ORT session, Cache Storage (open/store/damaged copy), background prefetch of a newer release | `manifest_fetch`, `manifest_invalid`, `manifest_outdated_client`, `release_blocked`, `release_disabled`, `cache_store`, `cache_damaged`, `prefetch`, `model_load_failed` |
| `switch` | the same failures after the visitor changed language | same kinds |
| `classify` | tokenizer, model run, decoding, word snap, amount parsing | `classify_failed` |
| `save` | IndexedDB open, migration, write, quota | `idb_open`, `idb_write`, `idb_migration` |

Stale-cache and manifest mismatches carry a `recovery` property saying which fallback ran
(`run_cached_release`, `evict_and_download_active`, `reload`, `reload_once`, `keep_running_cached_release`,
`memory_only`, `serve_legacy_db`, ...). `model_load_failed` is also still sent as a normal event with
its download numbers.

Every error event has: `stage`, `kind`, `recovery`, `app_build` (git hash, `-dirty` when the tree
had changes), `client_version`, `language`, `model` (id and version that ran), `manifest_schema`,
`manifest_active`, `online`, `effective_type`, `save_data`. PostHog adds browser, OS and `$host`.
Events from your own browsers (`?internal=1`, `gidi:internal` in localStorage) carry `internal: true`.

## What is never sent

- Note text, targets and amounts. Messages of `classify` and `save` errors are replaced by
  `[message omitted: <stage>]` (only the error type and the code location stay); in other stages
  any text the app has seen in a note (`registerSensitive`) is masked.
- URLs keep no query string or hash (also `$current_url` and `$referrer` on every event).

## Cost control

Dropped before sending: `ResizeObserver` noise, `Script error.`, aborts (a language switch aborts a
download), errors with a browser-extension frame, fetch failures while the browser is offline, and
a `<script>`/`<link>` from another origin that fails to load (the Cloudflare Web Analytics beacon,
blocked by ad blockers and Firefox tracking protection).
Each distinct error (type, message with numbers folded, top frame) is sent once per page load, at most
10 per page load, at most 3 in a burst and then one every 5 s. PostHog bills them as *exceptions*,
a separate product from events: the first 100,000 a month are free, then $0.00037 each up to
325,000 and $0.00014 after (https://posthog.com/pricing). Set a billing limit for Error tracking too
(`docs/cost-and-traffic.md`, checklist).

## Where to look

PostHog -> **Error tracking** (https://us.posthog.com/project/648258/error_tracking). Issues are
grouped by PostHog; open one and use the properties above (`stage`, `kind`, `model`, `app_build`).
In **Activity** the default "filter out internal and test users" view hides your own events: turn
it off to see a test error you triggered yourself.

Try it locally: `VITE_ANALYTICS_DEV=1 pnpm -C playground dev`, open `/?internal=1`, then in the
console `__gidiDebug.capture("load")`, `__gidiDebug.throwUncaught()` or
`__gidiDebug.rejectPromise()` (the hook exists only in the dev server).

## Alerts (set 2026-10-22)

PostHog's native error-tracking alerts (Error tracking -> Configuration -> Alerting) only offer Slack,
Discord, GitHub, GitLab, Teams, Linear and PagerDuty as destinations, no email, so the email alerts are
Product analytics **insight alerts** (hourly check, notify hvanlong@pm.me, the account owner). Both read
insights that count only real traffic: `internal` not set **and** `$host` equals `gidi.8bu.dev` (an exact
match: `$host` carries the port on local previews such as `127.0.0.1:8821`, and any other host, a tunnel
or a `workers.dev` test deployment, is not production either; the first version used `not_icontains`
`127.0.0.1` / `localhost`, which missed everything else):

1. **Gidi: new error issues (>2 distinct a day)**: insight "Gidi: production exceptions per day"
   (`$exception`, distinct `$exception_issue_id` per day). Fires when a day has more than 2 distinct
   error issues; set above today's baseline of 2.
2. **Gidi: model_load_failed spike (>20 a day)**: insight "Gidi: model_load_failed per day" (event
   `model_load_failed`, per day). Fires above 20 a day. A bad release or a broken cache shows up here first.

Local previews and other non-production hosts: a project **suppression rule** (Error tracking ->
Configuration -> Suppression rules, id `01a16386-1405-0000-26a2-3da17c6784be`) drops their exceptions
before they become issues. Its filters (match any): `$host` is not `gidi.8bu.dev`, `$host` contains
`127.0.0.1`, `$host` contains `localhost`. An event with no `$host` is not suppressed (the safe side).
Checked 2026-10-22 with four synthetic `$exception` events posted to the capture endpoint (internal,
distinct id `gidi-filter-test`): hosts `127.0.0.1:8821`, `localhost:5173` and a tunnel host were
dropped, the `gidi.8bu.dev` one arrived as an issue (set to Resolved). The saved insights, run with and
without the `$host` condition, differ by exactly the local rows on 2026-10-22: distinct error issues 2
vs 6 (4 local issues on `127.0.0.1:8821`), `model_load_failed` 15 vs 17 (2 local rows). Turn the rule off (or delete it) while you test the error path
from a local build or a preview deployment, then switch it on again. PostHog cannot delete events or
issues from the UI or API: test issues are set to Resolved (they reopen if the same error comes back).

## Source maps (opt-in, 2026-10-22)

Stack frames resolve to `src/` files (for example `notes-db.ts:openVietnamese:145`) once the build you
deploy was made with `pnpm -C playground sourcemaps:upload` (`scripts/upload-sourcemaps.mjs`):

1. builds `build:web` with `GIDI_SOURCEMAPS=1` (Vite writes hidden maps next to the chunks) and
   `GIDI_BUILD_ID` = the `app_build` string (git hash, `-dirty` when `playground/` has changes);
2. `posthog-cli sourcemap process` injects a chunk id into each chunk in `dist/assets`, uploads the maps
   as release `gidi-playground@<app_build>` and deletes them (`--delete-after`);
3. fails if any `.map` is left in `dist/`.

The injected chunks are what must be served, so deploy that same `dist/` without rebuilding:
`pnpm -C playground sourcemaps:upload && pnpm -C playground exec wrangler deploy`
(`pnpm run deploy` rebuilds without ids: it still works, but traces are minified again). Commit first so
the build id is not `-dirty`. A map is never a public file (`/assets/*.map` answers the SPA page).

Setup done once: `posthog-cli` (`curl -LsSf https://download.posthog.com/cli | sh`, in `~/.posthog`) and
a personal API key "gidi-sourcemaps (posthog-cli)" with the single scope `error_tracking:write`, limited
to project 648258 (create one at https://us.posthog.com/settings/user-api-keys; PostHog asks for the
password again). The key lives only in the macOS Keychain (service `gidi-posthog-cli`; store it with
`security add-generic-password -U -a posthog -s gidi-posthog-cli -w`), or in `$POSTHOG_CLI_API_KEY`, or in
`~/.config/posthog/gidi.env` (`POSTHOG_CLI_API_KEY=...`, `chmod 600`). Never commit it. Symbol sets:
https://us.posthog.com/project/648258/error_tracking?activeTab=configuration (Symbol sets).

Checked 2026-10-22 with a local production build (`wrangler dev`, `/?internal=1`, a blocked IndexedDB open):
the exception arrived as `app_build` `e20055e` with frames `notes-db.ts:openVietnamese:145` and
`openNotesDb:58`; the test issue was set to Resolved.

## Recheck checklist (written 2026-10-22)

Run in PostHog -> SQL editor (or the query API from a signed-in tab). Real traffic is
`properties.internal is null and properties.$host = 'gidi.8bu.dev'`: `$host` carries the port on
local previews (`127.0.0.1:8821`), so `not in ('localhost','127.0.0.1')` does NOT drop them. Times
are shown in the project time zone (UTC+7); use `formatDateTime(..., 'UTC')` to compare with deploys.

What 2026-10-22 showed: 18 `model_load_failed` since 2026-10-21 from 6 distinct visitors (2 more
rows were local previews of the owner). All 6 are Windows desktop Chrome 142, screen 1920x1080,
US cities Dulles VA / Des Moines IA / WA (cloud data centres), `effective_type` 4g,
`loaded_bytes` 0, always the same 3-event shape: `model worker failed to start` (3-22 s after the
pageview), then `Failed to fetch` (manifest fetch, 60-270 ms), then `Failed to fetch` again
after the automatic language switch; then `stack_opened`, `$pageleave` within seconds. Real visitors
(Chrome 154/155, Edge, Safari, VN/EU/US) have 0 failures in the same period. Six of six failures
came from outside the ~3 min bad-deploy window (3 before, 3 after 10:00 UTC), so it is not the
stale-cache bug. Reading: automated or network-restricted browsers (scanners, sandboxes) where the
worker script and then every same-origin fetch fail. Not an app bug, [INFERENCE] but not proven.

1. **Real failure rate** (decide from this one):
   ```sql
   select toDate(timestamp) d,
          countIf(event = 'model_load_failed') failed,
          countIf(event = 'model_loaded') loaded,
          count(distinct if(event = 'model_load_failed', distinct_id, null)) failed_users
   from events
   where timestamp >= '2026-10-23' and properties.internal is null
     and properties.$host = 'gidi.8bu.dev'
   group by d order by d
   ```
2. **Who fails** (bot profile or real people):
   ```sql
   select properties.$os os, properties.$browser b, properties.$browser_version v,
          properties.$geoip_country_code c, properties.$geoip_city_name city,
          properties.error err, count() n, count(distinct distinct_id) users
   from events
   where event = 'model_load_failed' and timestamp >= '2026-10-23'
     and properties.internal is null and properties.$host = 'gidi.8bu.dev'
   group by os, b, v, c, city, err order by n desc
   ```
3. **Stale-cache check**: failures with `loaded_bytes` above 0 or an `$exception` of kind
   `release_blocked`, `manifest_outdated_client`, `cache_damaged` or `chunk_load`:
   ```sql
   select properties.kind k, properties.recovery r, count(), count(distinct distinct_id)
   from events
   where event = '$exception' and timestamp >= '2026-10-23'
     and properties.internal is null and properties.$host = 'gidi.8bu.dev'
   group by k, r order by 3 desc
   ```

Decision rules, once at least 30 distinct real visitors loaded the model since 2026-10-23 (count
of `model_loaded` users in query 1) and at least 5 failing visitors exist:

- Failures only from the bot profile (Windows Chrome 142, 1920 wide, US data-centre cities, 3-event
  shape): benign. Optionally drop that cohort from alerts; no code change.
- Failed share of visitors under 3 %, spread over normal browsers, `loaded_bytes` 0: flaky networks
  or blockers. Add a bounded auto retry of the worker start only if Safari/mobile users appear.
- Any `loaded_bytes` > 0 and < `total_bytes`, or kinds `cache_damaged`, `release_blocked`,
  `manifest_outdated_client` outside a deploy hour: a real bug in the part download or cache
  rollout. Open `playground/src/runtime` (part join, Cache Storage) and reproduce with a throttled
  connection.
- Fewer than 30 loads or fewer than 5 failing visitors: too thin, repeat in a week.

Also confirm the beacon issue is gone: query `$exception` with `properties.kind = 'resource_load'`
and `app_build` `8ea1a0a` or later (the filter went live 2026-10-22 16:10 UTC, Worker version
`1784a26c-adc3-4473-b207-2b9869236b55`, source maps uploaded for `gidi-playground@8ea1a0a`; the version
before it, `b7a26a68-89fe-4d35-ac51-906216a63ab7`, is the rollback); expected 0 for third-party URLs.
