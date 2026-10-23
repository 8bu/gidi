import assert from "node:assert/strict"
import { createHash } from "node:crypto"
import { afterEach, beforeEach, test } from "node:test"

import type { ReleaseEntry } from "./manifest.ts"
import {
  dropStore,
  fetchModel,
  prefetchRelease,
  pruneStores,
  readStoredRelease,
  type ModelStore,
} from "./model-download.ts"
import { ModelIntegrityError, type ModelPart } from "./types.ts"

const sha = (bytes: Uint8Array): string =>
  createHash("sha256").update(bytes).digest("hex")

const model = Uint8Array.from({ length: 1000 }, (_, i) => (i * 7) % 256)
const parts: ModelPart[] = [
  { url: "https://x.test/models/fam/1/model.int8.onnx.01", size: 400 },
  { url: "https://x.test/models/fam/1/model.int8.onnx.02", size: 400 },
  { url: "https://x.test/models/fam/1/model.int8.onnx.03", size: 200 },
]
const slices = [
  model.subarray(0, 400),
  model.subarray(400, 800),
  model.subarray(800),
]
const SHA = sha(model)

const releaseFor = (list: ModelPart[]): ReleaseEntry => ({
  id: "fam",
  version: list[0].url.includes("/2/") ? "2" : "1",
  sha256: SHA,
  size_bytes: 1000,
  onnx_opset: 17,
  tokenizer_type: "bpe",
  base: "/release/fam/1",
  parts: list.map((p) => ({ url: p.url, size_bytes: p.size })),
})
const storeFor = (
  list: ModelPart[] = parts,
  reports: string[] = []
): ModelStore => ({
  name: "gidi-model:vi",
  release: releaseFor(list),
  report: (stage) => void reports.push(stage),
})

const realFetch = globalThis.fetch
const realCaches = (globalThis as { caches?: unknown }).caches
let requested: string[] = []
let served: Uint8Array[] = []
let stores: Map<string, Map<string, Uint8Array>> = new Map()

function fakeCaches(): unknown {
  return {
    async open(name: string) {
      const store = stores.get(name) ?? new Map<string, Uint8Array>()
      stores.set(name, store)
      return {
        async match(url: string) {
          const hit = store.get(url)
          return hit === undefined ? undefined : new Response(hit.slice())
        },
        async put(url: string, response: Response) {
          store.set(url, new Uint8Array(await response.arrayBuffer()))
        },
        async keys() {
          return [...store.keys()].map((url) => ({ url }))
        },
        async delete(request: string | { url: string }) {
          return store.delete(
            typeof request === "string" ? request : request.url
          )
        },
      }
    },
  }
}

beforeEach(() => {
  requested = []
  served = [...slices]
  stores = new Map()
  ;(globalThis as { caches?: unknown }).caches = fakeCaches()
  globalThis.fetch = (async (input: string | URL | Request) => {
    const url = String(input)
    requested.push(url)
    const index = parts.findIndex((part) => part.url === url)
    if (index === -1) return new Response("nope", { status: 404 })
    return new Response(served[index].slice())
  }) as typeof fetch
})

afterEach(() => {
  globalThis.fetch = realFetch
  ;(globalThis as { caches?: unknown }).caches = realCaches
})

test("parts download in parallel, join in order and are checked against the whole-file sha256", async () => {
  const progress: number[] = []
  const bytes = await fetchModel(
    parts,
    SHA,
    (loaded, total) => {
      assert.equal(total, 1000)
      progress.push(loaded)
    },
    undefined,
    storeFor()
  )
  assert.deepEqual(bytes, model)
  assert.equal(requested.length, 3)
  assert.equal(progress.at(-1), 1000)
})

test("a returning visit reads the verified copy from Cache Storage and fetches nothing", async () => {
  await fetchModel(parts, SHA, undefined, undefined, storeFor())
  requested = []
  const progress: Array<[number, number | null]> = []
  const again = await fetchModel(
    parts,
    SHA,
    (l, t) => progress.push([l, t]),
    undefined,
    storeFor()
  )
  assert.deepEqual(again, model)
  assert.deepEqual(requested, [])
  assert.deepEqual(progress, [[1000, 1000]])
})

test("a wrong hash is a hard error and nothing is stored", async () => {
  await assert.rejects(
    fetchModel(parts, "0".repeat(64), undefined, undefined, storeFor()),
    ModelIntegrityError
  )
  assert.equal([...stores.values()].flatMap((s) => [...s.keys()]).length, 0)
})

test("a damaged cached copy is evicted and downloaded again", async () => {
  await fetchModel(parts, SHA, undefined, undefined, storeFor())
  const [store] = stores.values()
  const damaged = store.get(parts[1].url) as Uint8Array
  damaged[5] ^= 0xff
  requested = []
  assert.deepEqual(
    await fetchModel(parts, SHA, undefined, undefined, storeFor()),
    model
  )
  assert.equal(requested.length, 3)
  assert.deepEqual(store.get(parts[1].url), slices[1])
})

test("a part with the wrong length (a 404 turned into the SPA page) is an error", async () => {
  served[1] = new Uint8Array(399)
  await assert.rejects(
    fetchModel(parts, SHA, undefined, undefined, storeFor()),
    /ended after 399 of 400/
  )
  served[1] = new Uint8Array(401)
  await assert.rejects(
    fetchModel(parts, SHA, undefined, undefined, storeFor()),
    /longer than/
  )
})

test("an HTTP error names the part", async () => {
  globalThis.fetch = (async () =>
    new Response("x", { status: 503 })) as typeof fetch
  await assert.rejects(
    fetchModel(parts, SHA, undefined, undefined, storeFor()),
    /HTTP 503/
  )
})

test("without Cache Storage the model is simply downloaded each time", async () => {
  ;(globalThis as { caches?: unknown }).caches = undefined
  assert.deepEqual(
    await fetchModel(parts, SHA, undefined, undefined, storeFor()),
    model
  )
  assert.deepEqual(
    await fetchModel(parts, SHA, undefined, undefined, storeFor()),
    model
  )
  assert.equal(requested.length, 6)
})

const META = "https://x.test/__gidi/release.json"

test("storing a release writes its record last and drops what belonged to the previous one", async () => {
  await fetchModel(parts, SHA, undefined, undefined, storeFor())
  const [store] = stores.values()
  assert.ok(store.has(META))
  assert.equal((await readStoredRelease("gidi-model:vi"))?.version, "1")
  const next: ModelPart[] = parts.map((p) => ({
    ...p,
    url: p.url.replace("/1/", "/2/"),
  }))
  const realParts = parts.splice(0, parts.length, ...next)
  await fetchModel(parts, SHA, undefined, undefined, storeFor(next))
  assert.deepEqual(
    [...store.keys()].sort(),
    [...next.map((p) => p.url), META].sort()
  )
  assert.equal((await readStoredRelease("gidi-model:vi"))?.version, "2")
  parts.splice(0, parts.length, ...realParts)
})

test("a background prefetch switches the cache only after the sha256 passed", async () => {
  await fetchModel(parts, SHA, undefined, undefined, storeFor())
  const next: ModelPart[] = parts.map((p) => ({
    ...p,
    url: p.url.replace("/1/", "/2/"),
  }))
  const realParts = parts.splice(0, parts.length, ...next)
  // The new files are corrupt: the cache keeps the old release untouched.
  served = [slices[0], slices[1].slice().fill(1), slices[2]]
  const reports: string[] = []
  assert.equal(
    await prefetchRelease(next, SHA, undefined, storeFor(next, reports)),
    false
  )
  assert.deepEqual(reports, ["prefetch"])
  assert.equal((await readStoredRelease("gidi-model:vi"))?.version, "1")
  // Now they are right.
  served = [...slices]
  assert.equal(
    await prefetchRelease(next, SHA, undefined, storeFor(next)),
    true
  )
  assert.equal((await readStoredRelease("gidi-model:vi"))?.version, "2")
  parts.splice(0, parts.length, ...realParts)
})

test("a failed prefetch (network down) leaves the running release cached and reports once", async () => {
  await fetchModel(parts, SHA, undefined, undefined, storeFor())
  globalThis.fetch = (async () => {
    throw new TypeError("Failed to fetch")
  }) as typeof fetch
  const next: ModelPart[] = parts.map((p) => ({
    ...p,
    url: p.url.replace("/1/", "/2/"),
  }))
  const reports: string[] = []
  assert.equal(
    await prefetchRelease(next, SHA, undefined, storeFor(next, reports)),
    false
  )
  assert.deepEqual(reports, ["prefetch"])
  assert.equal((await readStoredRelease("gidi-model:vi"))?.version, "1")
})

test("an aborted prefetch is silent", async () => {
  const abort = new AbortController()
  abort.abort()
  globalThis.fetch = (async () => {
    throw new DOMException("aborted", "AbortError")
  }) as typeof fetch
  const reports: string[] = []
  assert.equal(
    await prefetchRelease(parts, SHA, abort.signal, storeFor(parts, reports)),
    false
  )
  assert.deepEqual(reports, [])
})

test("dropStore and pruneStores remove a language's cache and caches of older layouts", async () => {
  await fetchModel(parts, SHA, undefined, undefined, storeFor())
  let deleted: string[] = []
  const fake = (
    globalThis as {
      caches: {
        keys?: () => Promise<string[]>
        delete?: (n: string) => Promise<boolean>
      }
    }
  ).caches
  fake.keys = async () => [
    "gidi-model:vi",
    "gidi-model:gidi-finance-v4",
    "other",
  ]
  fake.delete = async (n) => {
    deleted.push(n)
    return stores.delete(n)
  }
  await pruneStores(["gidi-model:vi", "gidi-model:en"])
  assert.deepEqual(deleted, ["gidi-model:gidi-finance-v4"])
  deleted = []
  await dropStore("gidi-model:vi")
  assert.deepEqual(deleted, ["gidi-model:vi"])
  assert.equal(stores.size, 0)
})
