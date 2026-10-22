/**
 * Decides which model release a language runs: reads the manifest (always revalidated), compares
 * it with the release recorded in Cache Storage and returns what to load now and what to fetch
 * in the background. Every effect is injected, so `release-manager.test.ts` covers rollout,
 * rollback, a stale or unreachable manifest, a blocked release and an outdated client under node.
 */

import {
  ManifestError,
  parseManifest,
  planLoad,
  type ReleaseEntry,
} from "./manifest.ts"
import { STORE_PREFIX } from "./model-download.ts"

export const MANIFEST_URL = "/release/manifest.json"

export interface ReleaseDeps {
  /** Parsed JSON of the manifest from the network; rejects when it cannot be reached. */
  fetchManifest(signal: AbortSignal): Promise<unknown>
  readStored(cacheName: string): Promise<ReleaseEntry | null>
  dropStore(cacheName: string): Promise<void>
  /** Short stage tag + error, for the owner's error tracking; never throws. */
  report(stage: string, error: unknown, recovery?: string): void
}

export interface Resolved {
  cacheName: string
  run: ReleaseEntry
  /** Fetch this in the background once `run` is loaded; `null` when nothing is newer. */
  update: ReleaseEntry | null
  /** The manifest could not be used and the cached release runs as it is. */
  offline: boolean
  /** What the manifest said (for error context); `null` when it could not be used. */
  manifest: { schema: number; active: string | null } | null
}

export const cacheNameFor = (lang: string): string => `${STORE_PREFIX}${lang}`

/** The manifest from the network. `no-cache` revalidates every time, so a stale copy never wins. */
export async function fetchManifestJson(
  signal: AbortSignal,
  fetchImpl: typeof fetch = fetch
): Promise<unknown> {
  const response = await fetchImpl(MANIFEST_URL, { cache: "no-cache", signal })
  if (!response.ok)
    throw new Error(`GET ${MANIFEST_URL} failed: HTTP ${response.status}`)
  return response.json()
}

export async function resolveRelease(
  lang: string,
  deps: ReleaseDeps,
  signal: AbortSignal
): Promise<Resolved> {
  const cacheName = cacheNameFor(lang)
  const cached = await deps.readStored(cacheName)

  let manifest
  try {
    manifest = parseManifest(await deps.fetchManifest(signal))
  } catch (error) {
    if (signal.aborted) throw error
    // A manifest for a newer app: the code is stale, not the model. The caller reloads.
    if (error instanceof ManifestError && error.problem === "outdated_client") {
      deps.report("manifest_outdated_client", error, "reload")
      throw error
    }
    deps.report(
      error instanceof ManifestError ? "manifest_invalid" : "manifest_fetch",
      error,
      cached === null ? undefined : "run_cached_release"
    )
    // Unreachable or broken manifest: keep running what is cached, never force a re-download.
    if (cached !== null)
      return {
        cacheName,
        run: cached,
        update: null,
        offline: true,
        manifest: null,
      }
    throw error
  }

  try {
    const plan = planLoad(manifest.models[lang], cached)
    if (plan.evictCached) {
      deps.report(
        "release_blocked",
        new Error(
          `cached release ${cached?.version} is blocked by the manifest`
        ),
        "evict_and_download_active"
      )
      await deps.dropStore(cacheName)
    }
    return {
      cacheName,
      run: plan.run,
      update: plan.update,
      offline: false,
      manifest: {
        schema: manifest.schema,
        active: manifest.models[lang]?.active ?? null,
      },
    }
  } catch (error) {
    if (error instanceof ManifestError && error.problem === "disabled") {
      deps.report("release_disabled", error, "evict_cache")
      await deps.dropStore(cacheName)
    }
    throw error
  }
}
