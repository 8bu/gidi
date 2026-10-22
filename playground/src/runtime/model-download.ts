/**
 * Fetch the model: every part in parallel, joined into one buffer and checked against the sha256
 * of the whole file. Verified bytes are kept in Cache Storage so a returning visitor never
 * downloads the model again. No ONNX or DOM dependency, so `model-download.test.ts` runs it
 * under node with a fake `fetch` and `caches`.
 */

import {
  ModelIntegrityError,
  type LoadOptions,
  type ModelPart,
} from "./types.ts"

const sumSizes = (parts: readonly ModelPart[]): number =>
  parts.reduce((n, part) => n + part.size, 0)

/**
 * Download every part in parallel straight into one buffer, reporting combined progress. The
 * sizes come from `web-release.json`; a part that ends early or runs long (a 404 turned into the
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
 * Verified model bytes live in Cache Storage (one cache per model family, keyed by part URL), so
 * a returning visitor, or one switching language back and forth, never downloads the model again
 * even when the browser's HTTP cache dropped it. Everything here is best effort: where Cache
 * Storage is missing or full the model is just downloaded again.
 */
async function openModelCache(
  parts: readonly ModelPart[]
): Promise<Cache | null> {
  try {
    if (typeof caches === "undefined") return null
    const family = new URL(parts[0].url).pathname.split("/")[2] ?? "model"
    return await caches.open(`gidi-model:${family}`)
  } catch {
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

/** Store verified bytes and drop parts of other releases of the same family. */
async function storeCached(
  cache: Cache,
  parts: readonly ModelPart[],
  model: Uint8Array
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
    for (const key of await cache.keys()) {
      if (!parts.some((part) => part.url === key.url)) await cache.delete(key)
    }
  } catch {
    // Quota or private mode: the next visit downloads the model again.
  }
}

async function evictCached(
  cache: Cache,
  parts: readonly ModelPart[]
): Promise<void> {
  try {
    await Promise.all(parts.map((part) => cache.delete(part.url)))
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
  signal?: AbortSignal
): Promise<Uint8Array> {
  const expected = sha256.toLowerCase()
  const cache = await openModelCache(parts)
  if (cache !== null) {
    const cached = await readCached(cache, parts)
    if (cached !== null) {
      if ((await sha256Hex(cached)) === expected) {
        onProgress?.(cached.length, cached.length)
        return cached
      }
      // A damaged copy: forget it and fetch a fresh one.
      await evictCached(cache, parts)
    }
  }
  const model = await downloadParts(parts, onProgress, signal)
  const actual = await sha256Hex(model)
  if (actual !== expected) {
    throw new ModelIntegrityError(
      `model sha256 mismatch: expected ${sha256}, downloaded ${actual}`
    )
  }
  if (cache !== null) await storeCached(cache, parts, model)
  return model
}
