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
download), errors with a browser-extension frame, and fetch failures while the browser is offline.
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

## Suggested alerts (set by hand in PostHog)

1. **New issue**: Error tracking -> Configuration (alerts) -> "New issue created" -> your email or
   Slack. Catches every new kind of failure once.
2. **Model load failures spiking**: Product analytics -> new Trends insight on the event
   `model_load_failed` (or `$exception` with `stage` = `load`), `internal` not set, interval day;
   Alerts -> alert when the value is above e.g. 20 a day (set from your normal level), or when it
   rises more than 100% against the previous day. A bad release or a broken cache shows up here first.

## Limits

Source maps are not uploaded: PostHog needs a personal API key for `posthog-cli sourcemap upload`,
and shipping `.map` files in `dist/` would publish the source. Stack frames show minified file and
line numbers plus the `app_build` hash, so build that commit locally to read a frame.
