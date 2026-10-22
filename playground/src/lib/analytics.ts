import type { PostHog } from "posthog-js"

import type { TxType } from "@/app/types"
import {
  createErrorGate,
  errorProperties,
  scrubException,
  scrubPageUrls,
  setErrorSink,
  type ErrorStage,
} from "@/lib/error-report"
import type { Lang } from "@/lib/language"

/**
 * Anonymous usage counts for the notes app (PostHog, US zone, project "Gidi").
 *
 * Privacy contract: the app promises that a note never leaves the device, so no event carries
 * note text, a target name or an amount. Events say only what happened and to which type.
 * Visitors get a random anonymous ID that PostHog keeps in localStorage (no cookie, no person
 * profile, never identified), so returning visitors, retention and multi-day funnels work.
 * Events carry the app language (`vi`/`en`, the model's), never the country: the geoip default is
 * decided by the Worker and is not sent. Autocapture, session replay, heatmaps, surveys and exception capture are off, here and in the
 * project settings.
 *
 * Internal traffic: opening the app once with `?internal=1` marks this browser as the owner's
 * (a `gidi:internal` flag in localStorage; `?internal=0` clears it). Every event from it then
 * carries `internal: true`, and the project's test-account filter hides those events. They are
 * still sent, so the owner can watch them live.
 *
 * The SDK is a lazy chunk so it never delays the first paint, and it only loads in a production
 * bundle: dev servers and previews send nothing.
 *
 * Cost: PostHog bills per event (1M a month free, then $0.00005 falling to $0.0000343; see
 * docs/cost-and-traffic.md). A visit therefore sends `$pageview`, one `model_loaded`, and the
 * events the visitor causes. Anything else the SDK would add on its own (`$pageleave`, web vitals,
 * rageclicks, `$set`, flag calls) is dropped in `before_send`, `model_loaded` and `stack_opened`
 * are sent once per page load (per language / per type), failures at most three times, and a page
 * load stops after `PAGE_BUDGET` events so a loop can never run up the bill.
 */

/** Public project token. Override at build time to point at a dev project. */
const KEY =
  import.meta.env.VITE_POSTHOG_KEY ??
  "phc_r3UaVwvpE3N8jcbE7xcwL7fDX2AXGaYr7oEvTK7ezjS6"
const HOST = import.meta.env.VITE_POSTHOG_HOST ?? "https://us.i.posthog.com"
const UI_HOST = "https://us.posthog.com"
const INTERNAL_KEY = "gidi:internal"

/** Read, and apply then strip, the `?internal=1|0` switch. Storage may be blocked: then false. */
function isInternal(): boolean {
  try {
    const url = new URL(window.location.href)
    const flag = url.searchParams.get("internal")
    if (flag === "1") window.localStorage.setItem(INTERNAL_KEY, "1")
    if (flag === "0") window.localStorage.removeItem(INTERNAL_KEY)
    if (flag !== null) {
      url.searchParams.delete("internal")
      window.history.replaceState(window.history.state, "", url)
    }
    return window.localStorage.getItem(INTERNAL_KEY) === "1"
  } catch {
    return false
  }
}

/** Every event and its properties. Keep names stable: they are the PostHog taxonomy. */
export interface AnalyticsEventMap {
  /** The on-device model finished loading; `language` is the model's (and the app's) language. */
  model_loaded: { ms: number; language: Lang }
  /** The model could not load; `error` is the app's own message, never note text. The rest says how far the download got and on what connection. */
  model_load_failed: {
    ms: number
    language: Lang
    error: string
    error_name: string
    /** Bytes received before the failure, and the expected total (null if unknown). */
    loaded_bytes: number
    total_bytes: number | null
    effective_type: string | null
    save_data: boolean | null
    online: boolean
    device_memory: number | null
  }
  /** One send from the composer; a pasted list is one send with several lines. */
  notes_sent: {
    language: Lang
    lines: number
    classified: number
    failed: number
    with_amount: number
    with_target: number
    types: Partial<Record<TxType, number>>
  }
  /** The visitor chose a language (and so a model) in the app; never fired for the geoip default. */
  language_changed: { to: Lang }
  /** A stack sheet opened. */
  stack_opened: { type: TxType }
  /** The user corrected a record; `to_type` differs from `from_type` when retyped. */
  note_edited: {
    from_type: TxType
    to_type: TxType
    type_changed: boolean
    target_changed: boolean
    amount_changed: boolean
  }
  note_deleted: { type: TxType }
  notes_exported: { count: number }
  notes_imported: { count: number }
  /** An outbound link in the menu (`/lab` shows up as its own pageview). */
  link_clicked: { link: "github" | "huggingface" | "author" }
}

/** The only SDK-made event kept; every app event is in `AnalyticsEventMap` and has no `$`. */
const SDK_EVENTS = new Set(["$pageview", "$exception"])

/** Hard stop per page load, counted over everything that reaches `capture`. */
const PAGE_BUDGET = 200

/** Per page load: how often an event may be sent, split by one property (`by`) if given. */
const PAGE_LIMITS: Partial<
  Record<keyof AnalyticsEventMap, { max: number; by?: string }>
> = {
  model_loaded: { max: 1, by: "language" },
  model_load_failed: { max: 3 },
  stack_opened: { max: 1, by: "type" },
}

const sentThisPage = new Map<string, number>()
let pageBudget = PAGE_BUDGET

/** Spend one event of this page load's allowance, unless it is a repeat the app does not need. */
function admit(event: string, properties: Record<string, unknown>): boolean {
  if (pageBudget <= 0) return false
  const limit = PAGE_LIMITS[event as keyof AnalyticsEventMap]
  if (limit !== undefined) {
    const key = `${event}:${limit.by === undefined ? "" : String(properties[limit.by])}`
    const sent = sentThisPage.get(key) ?? 0
    if (sent >= limit.max) return false
    sentThisPage.set(key, sent + 1)
  }
  pageBudget -= 1
  return true
}

let client: Promise<PostHog | null> | null = null

/** `VITE_ANALYTICS_DEV=1 pnpm dev` sends from the dev server too (to try the error path for real). */
const enabled =
  import.meta.env.PROD || import.meta.env.VITE_ANALYTICS_DEV === "1"

const errorGate = createErrorGate()

export function initAnalytics(): void {
  if (!enabled || client !== null) return
  setErrorSink(captureError)
  const internal = isInternal()
  client = import("posthog-js").then(
    ({ default: posthog }) => {
      posthog.init(KEY, {
        api_host: HOST,
        ui_host: UI_HOST,
        defaults: "2025-05-24",
        persistence: "localStorage",
        person_profiles: "identified_only",
        autocapture: false,
        capture_pageleave: false,
        capture_performance: false,
        rageclick: false,
        capture_dead_clicks: false,
        capture_heatmaps: false,
        // Uncaught errors and unhandled rejections are captured by main.tsx (own listeners), so
        // they do not depend on the project's remote exception-autocapture switch.
        capture_exceptions: false,
        disable_session_recording: true,
        disable_surveys: true,
        disable_product_tours: true,
        disable_conversations: true,
        disable_web_experiments: true,
        advanced_disable_flags: true,
        disable_external_dependency_loading: true,
        before_send: (event) => {
          if (!event) return event
          if (event.event.startsWith("$") && !SDK_EVENTS.has(event.event)) {
            return null
          }
          scrubPageUrls(event.properties)
          if (internal) event.properties.internal = true
          if (event.event === "$exception") {
            return scrubException(event, errorGate)
          }
          return event
        },
      })
      return posthog
    },
    () => null
  )
}

export function track<E extends keyof AnalyticsEventMap>(
  event: E,
  properties: AnalyticsEventMap[E]
): void {
  if (client === null || !admit(event, properties)) return
  void client.then((posthog) => posthog?.capture(event, properties))
}

/**
 * Report a failure the app caught (or let through) to the owner's error tracking: PostHog
 * `$exception` events with a stage tag and safe context (see `error-report.ts`, which also
 * scrubs, dedupes and caps them). Never throws, never sends note text.
 */
export function captureError(
  error: unknown,
  stage: ErrorStage,
  extra: { kind?: string; recovery?: string } = {}
): void {
  if (client === null) return
  const failure = error instanceof Error ? error : new Error(String(error))
  void client.then((posthog) =>
    posthog?.captureException(failure, errorProperties(stage, extra))
  )
}
