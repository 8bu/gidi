import type { PostHog } from "posthog-js"

import type { TxType } from "@/app/types"

/**
 * Anonymous usage counts for the notes app (PostHog, US zone, project "Gidi").
 *
 * Privacy contract: the app promises that a note never leaves the device, so no event carries
 * note text, a target name or an amount. Events say only what happened and to which type.
 * Visitors get a random anonymous ID that PostHog keeps in localStorage (no cookie, no person
 * profile, never identified), so returning visitors, retention and multi-day funnels work.
 * Autocapture, session replay, heatmaps, surveys and exception capture are off, here and in the
 * project settings.
 *
 * Internal traffic: opening the app once with `?internal=1` marks this browser as the owner's
 * (a `gidi:internal` flag in localStorage; `?internal=0` clears it). Every event from it then
 * carries `internal: true`, and the project's test-account filter hides those events. They are
 * still sent, so the owner can watch them live.
 *
 * The SDK is a lazy chunk so it never delays the first paint, and it only loads in a production
 * bundle: dev servers and previews send nothing.
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
  /** The on-device model finished loading. */
  model_loaded: { ms: number }
  /** The model could not load; `error` is the app's own message, never note text. */
  model_load_failed: { ms: number; error: string }
  /** One send from the composer; a pasted list is one send with several lines. */
  notes_sent: {
    lines: number
    classified: number
    failed: number
    with_amount: number
    with_target: number
    types: Partial<Record<TxType, number>>
  }
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

let client: Promise<PostHog | null> | null = null

export function initAnalytics(): void {
  if (!import.meta.env.PROD || client !== null) return
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
        capture_dead_clicks: false,
        capture_heatmaps: false,
        capture_exceptions: false,
        disable_session_recording: true,
        disable_surveys: true,
        disable_product_tours: true,
        disable_conversations: true,
        disable_web_experiments: true,
        advanced_disable_flags: true,
        disable_external_dependency_loading: true,
        before_send: (event) => {
          if (event && internal) event.properties.internal = true
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
  void client?.then((posthog) => posthog?.capture(event, properties))
}
