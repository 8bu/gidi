/**
 * Fetch the model: every part in parallel, joined into one buffer and checked against the sha256
 * of the whole file. Verified bytes are kept in Cache Storage so a returning visitor never
 * downloads the model again. No ONNX or DOM dependency, so `model-download.test.ts` runs it
 * under node with a fake `fetch` and `caches`.
 */

import { parseRelease, type ReleaseEntry } from "./manifest.ts"
import {
  ModelIntegrityError,
  type LoadOptions,
  type ModelPart,
} from "./types.ts"

const sumSizes = (parts: readonly ModelPart[]): number =>
  parts.reduce((n, part) => n + part.size, 0)

/**
 * Download every part in parallel straight into one buffer, reporting combined progress. The
 * sizes come from the release manifest; a part that ends early or runs long (a 404 turned into the
 * SPA page, a truncated transfer) fails here, before the sha256 check.
 */
async function downloadParts(
  parts: readonly ModelPart[],
  onProgress: LoadOptions["onProgress"],
  signal?: AbortSignal
): Promise<Uint8Array> {
  const total = sumSizes(parts)
  const buffer = new Uint8Array(total)
  const received = parts.map(() => 0)
  const inner = new AbortController()
  const abort = (): void => inner.abort(signal?.reason)
  signal?.addEventListener("abort", abort, { once: true })
  let offset = 0
  try {
    await Promise.all(
      parts.map(async (part, index) => {
        const start = offset
        offset += part.size
        const response = await fetch(part.url, { signal: inner.signal })
        if (!response.ok || response.body === null) {
          throw new Error(`GET ${part.url} failed: HTTP ${response.status}`)
        }
        const reader = response.body.getReader()
        let at = start
        for (;;) {
          const { done, value } = await reader.read()
          if (done) break
          if (at + value.length > start + part.size) {
            throw new Error(
              `${part.url} is longer than the ${part.size} B listed`
            )
          }
          buffer.set(value, at)
          at += value.length
          received[index] = at - start
          onProgress?.(
            received.reduce((n, r) => n + r, 0),
            total
          )
        }
        if (at !== start + part.size) {
          throw new Error(
            `${part.url} ended after ${at - start} of ${part.size} B`
          )
        }
      })
    )
  } catch (error) {
    inner.abort()
    throw error
  } finally {
    signal?.removeEventListener("abort", abort)
  }
  return buffer
}

/**
 * Verified model bytes live in Cache Storage, one cache per language (`gidi-model:<lang>`), keyed
 * by part URL, plus a `release.json` record of the release they belong to (written last, so its
 * presence means the parts are complete). A returning visitor, or one switching language back and
 * forth, never downloads the model again even when the browser's HTTP cache dropped it.
 * Everything here is best effort: where Cache Storage is missing or full the model is just
 * downloaded again, and `report` (if given) hears what went wrong.
 */
export interface ModelStore {
  /** Cache name, e.g. `gidi-model:vi`. */
  name: string
  /** The release the parts belong to, stored as `release.json`. */
  release: ReleaseEntry
  /** Called with a short stage tag and the error; never throws. */
  report?: (stage: string, error: unknown) => void
}

export const STORE_PREFIX = "gidi-model:"

const metaUrl = (parts: readonly ModelPart[]): string =>
  new URL("/__gidi/release.json", parts[0].url).href

async function openCache(
  name: string,
  report?: ModelStore["report"]
): Promise<Cache | null> {
  try {
    if (typeof caches === "undefined") return null
    return await caches.open(name)
  } catch (error) {
    report?.("cache_open", error)
    return null
  }
}

async function readCached(
  cache: Cache,
  parts: readonly ModelPart[]
): Promise<Uint8Array | null> {
  try {
    const model = new Uint8Array(sumSizes(parts))
    let at = 0
    for (const part of parts) {
      const hit = await cache.match(part.url)
      if (hit === undefined) return null
      const bytes = new Uint8Array(await hit.arrayBuffer())
      if (bytes.length !== part.size) return null
      model.set(bytes, at)
      at += bytes.length
    }
    return model
  } catch {
    return null
  }
}

/** Store verified bytes, then the release record, then drop everything else in this language's cache. */
async function storeCached(
  cache: Cache,
  parts: readonly ModelPart[],
  model: Uint8Array,
  store: ModelStore
): Promise<void> {
  try {
    let at = 0
    for (const part of parts) {
      const bytes = model.subarray(at, at + part.size)
      await cache.put(
        part.url,
        new Response(bytes as BufferSource, {
          headers: { "content-type": "application/octet-stream" },
        })
      )
      at += part.size
    }
    const meta = metaUrl(parts)
    await cache.put(meta, new Response(JSON.stringify(store.release)))
    const keep = new Set([meta, ...parts.map((part) => part.url)])
    for (const key of await cache.keys()) {
      if (!keep.has(key.url)) await cache.delete(key)
    }
  } catch (error) {
    // Quota or private mode: the next visit downloads the model again.
    store.report?.("cache_store", error)
  }
}

async function evictCached(cache: Cache): Promise<void> {
  try {
    for (const key of await cache.keys()) await cache.delete(key)
  } catch {
    // Nothing more to do.
  }
}

async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", bytes as BufferSource)
  return Array.from(new Uint8Array(digest), (b) =>
    b.toString(16).padStart(2, "0")
  ).join("")
}

/**
 * The verified model bytes. Order: Cache Storage copy (re-hashed, since a damaged copy must never
 * reach the ORT session), else a download that is hashed before it is stored.
 */
export async function fetchModel(
  parts: readonly ModelPart[],
  sha256: string,
  onProgress: LoadOptions["onProgress"],
  signal?: AbortSignal,
  store?: ModelStore
): Promise<Uint8Array> {
  const expected = sha256.toLowerCase()
  const cache =
    store === undefined ? null : await openCache(store.name, store.report)
  if (cache !== null) {
    const cached = await readCached(cache, parts)
    if (cached !== null) {
      if ((await sha256Hex(cached)) === expected) {
        onProgress?.(cached.length, cached.length)
        return cached
      }
      // A damaged copy: forget it and fetch a fresh one.
      store?.report?.(
        "cache_damaged",
        new ModelIntegrityError("cached model failed its sha256")
      )
      await evictCached(cache)
    }
  }
  const model = await downloadParts(parts, onProgress, signal)
  const actual = await sha256Hex(model)
  if (actual !== expected) {
    throw new ModelIntegrityError(
      `model sha256 mismatch: expected ${sha256}, downloaded ${actual}`
    )
  }
  if (cache !== null && store !== undefined)
    await storeCached(cache, parts, model, store)
  return model
}

/** The release recorded in a language's cache, or `null` (nothing cached, or an unreadable record). */
export async function readStoredRelease(
  name: string
): Promise<ReleaseEntry | null> {
  try {
    const cache = await openCache(name)
    if (cache === null) return null
    const keys = await cache.keys()
    const key = keys.find((k) => k.url.endsWith("/__gidi/release.json"))
    if (key === undefined) return null
    const hit = await cache.match(key.url)
    return hit === undefined ? null : parseRelease(await hit.json())
  } catch {
    return null
  }
}

/** Forget a language's cached model (a blocked or switched-off release). */
export async function dropStore(name: string): Promise<void> {
  try {
    if (typeof caches !== "undefined") await caches.delete(name)
  } catch {
    // Nothing more to do.
  }
}

/** Delete caches of this app that no language uses any more (older cache layouts). */
export async function pruneStores(keep: readonly string[]): Promise<void> {
  try {
    if (typeof caches === "undefined") return
    for (const name of await caches.keys()) {
      if (name.startsWith(STORE_PREFIX) && !keep.includes(name))
        await caches.delete(name)
    }
  } catch {
    // Nothing more to do.
  }
}

/**
 * Download `release` in the background and store it, replacing the language's cached release only
 * after its sha256 passed. Returns whether the cache now holds it; failures leave the cache as it
 * was (the running model is never touched) and go to `store.report`.
 */
export async function prefetchRelease(
  parts: readonly ModelPart[],
  sha256: string,
  signal: AbortSignal | undefined,
  store: ModelStore
): Promise<boolean> {
  try {
    await fetchModel(parts, sha256, undefined, signal, store)
    return (await readStoredRelease(store.name))?.sha256 === sha256
  } catch (error) {
    if (signal?.aborted !== true) store.report?.("prefetch", error)
    return false
  }
}
