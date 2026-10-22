import assert from "node:assert/strict"
import { createHash } from "node:crypto"
import { afterEach, beforeEach, test } from "node:test"

import { fetchModel } from "./model-download.ts"
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
  const bytes = await fetchModel(parts, SHA, (loaded, total) => {
    assert.equal(total, 1000)
    progress.push(loaded)
  })
  assert.deepEqual(bytes, model)
  assert.equal(requested.length, 3)
  assert.equal(progress.at(-1), 1000)
})

test("a returning visit reads the verified copy from Cache Storage and fetches nothing", async () => {
  await fetchModel(parts, SHA, undefined)
  requested = []
  const progress: Array<[number, number | null]> = []
  const again = await fetchModel(parts, SHA, (l, t) => progress.push([l, t]))
  assert.deepEqual(again, model)
  assert.deepEqual(requested, [])
  assert.deepEqual(progress, [[1000, 1000]])
})

test("a wrong hash is a hard error and nothing is stored", async () => {
  await assert.rejects(
    fetchModel(parts, "0".repeat(64), undefined),
    ModelIntegrityError
  )
  assert.equal([...stores.values()].flatMap((s) => [...s.keys()]).length, 0)
})

test("a damaged cached copy is evicted and downloaded again", async () => {
  await fetchModel(parts, SHA, undefined)
  const [store] = stores.values()
  const damaged = store.get(parts[1].url) as Uint8Array
  damaged[5] ^= 0xff
  requested = []
  assert.deepEqual(await fetchModel(parts, SHA, undefined), model)
  assert.equal(requested.length, 3)
  assert.deepEqual(store.get(parts[1].url), slices[1])
})

test("a part with the wrong length (a 404 turned into the SPA page) is an error", async () => {
  served[1] = new Uint8Array(399)
  await assert.rejects(
    fetchModel(parts, SHA, undefined),
    /ended after 399 of 400/
  )
  served[1] = new Uint8Array(401)
  await assert.rejects(fetchModel(parts, SHA, undefined), /longer than/)
})

test("an HTTP error names the part", async () => {
  globalThis.fetch = (async () =>
    new Response("x", { status: 503 })) as typeof fetch
  await assert.rejects(fetchModel(parts, SHA, undefined), /HTTP 503/)
})

test("without Cache Storage the model is simply downloaded each time", async () => {
  ;(globalThis as { caches?: unknown }).caches = undefined
  assert.deepEqual(await fetchModel(parts, SHA, undefined), model)
  assert.deepEqual(await fetchModel(parts, SHA, undefined), model)
  assert.equal(requested.length, 6)
})

test("a newer release of the same family replaces the older cached parts", async () => {
  await fetchModel(parts, SHA, undefined)
  const next: ModelPart[] = parts.map((p) => ({
    ...p,
    url: p.url.replace("/1/", "/2/"),
  }))
  const original = parts.map((p) => ({ ...p }))
  parts.splice(0, parts.length, ...next)
  await fetchModel(parts, SHA, undefined)
  const [store] = stores.values()
  assert.deepEqual([...store.keys()].sort(), next.map((p) => p.url).sort())
  parts.splice(0, parts.length, ...original)
})
