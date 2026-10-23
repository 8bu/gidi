import assert from "node:assert/strict"
import { test } from "node:test"

import { ManifestError, type ReleaseEntry } from "./manifest.ts"
import {
  cacheNameFor,
  fetchManifestJson,
  MANIFEST_URL,
  resolveRelease,
  type ReleaseDeps,
} from "./release-manager.ts"

const release = (version: string): ReleaseEntry => ({
  id: "fam",
  version,
  sha256: version.padEnd(64, "0"),
  size_bytes: 10,
  onnx_opset: 17,
  tokenizer_type: "bpe",
  base: `/release/fam/${version}`,
  parts: [{ url: `/models/fam/${version}/model.int8.onnx.01`, size_bytes: 10 }],
})

const manifestJson = (
  active: string | null,
  blocked: string[] = [],
  extra = {}
) => ({
  schema: 1,
  min_client: 1,
  models: {
    vi: {
      active,
      blocked,
      releases: { "1.0.0": release("1.0.0"), "2.0.0": release("2.0.0") },
    },
  },
  ...extra,
})

function harness(opts: {
  manifest?: unknown
  fail?: boolean
  cached?: ReleaseEntry | null
}) {
  const reports: string[] = []
  const dropped: string[] = []
  const deps: ReleaseDeps = {
    fetchManifest: async () => {
      if (opts.fail) throw new TypeError("Failed to fetch")
      return opts.manifest
    },
    readStored: async () => opts.cached ?? null,
    dropStore: async (name) => void dropped.push(name),
    report: (stage, _e, recovery) =>
      void reports.push(recovery ? `${stage}:${recovery}` : stage),
  }
  return { deps, reports, dropped }
}

const signal = new AbortController().signal

test("first visit: the active release, nothing to update", async () => {
  const { deps, reports } = harness({ manifest: manifestJson("2.0.0") })
  const r = await resolveRelease("vi", deps, signal)
  assert.equal(r.run.version, "2.0.0")
  assert.equal(r.update, null)
  assert.equal(r.cacheName, cacheNameFor("vi"))
  assert.deepEqual(reports, [])
})

test("new model rollout: the cached release runs now, the new one is queued for the background", async () => {
  const { deps } = harness({
    manifest: manifestJson("2.0.0"),
    cached: release("1.0.0"),
  })
  const r = await resolveRelease("vi", deps, signal)
  assert.equal(r.run.version, "1.0.0")
  assert.equal(r.update?.version, "2.0.0")
})

test("rollback by manifest: pointing active at the older release queues it", async () => {
  const { deps } = harness({
    manifest: manifestJson("1.0.0"),
    cached: release("2.0.0"),
  })
  const r = await resolveRelease("vi", deps, signal)
  assert.equal(r.run.version, "2.0.0")
  assert.equal(r.update?.version, "1.0.0")
})

test("emergency block: the bad cached release is dropped and the active one runs in front", async () => {
  const { deps, dropped, reports } = harness({
    manifest: manifestJson("1.0.0", ["2.0.0"]),
    cached: release("2.0.0"),
  })
  const r = await resolveRelease("vi", deps, signal)
  assert.equal(r.run.version, "1.0.0")
  assert.equal(r.update, null)
  assert.deepEqual(dropped, [cacheNameFor("vi")])
  assert.deepEqual(reports, ["release_blocked:evict_and_download_active"])
})

test("a language switched off drops its cache and errors", async () => {
  const { deps, dropped } = harness({
    manifest: manifestJson(null),
    cached: release("2.0.0"),
  })
  await assert.rejects(
    resolveRelease("vi", deps, signal),
    (e: unknown) => e instanceof ManifestError && e.problem === "disabled"
  )
  assert.deepEqual(dropped, [cacheNameFor("vi")])
})

test("offline or manifest unreachable: the cached release keeps running, nothing is forced", async () => {
  const { deps, reports } = harness({ fail: true, cached: release("1.0.0") })
  const r = await resolveRelease("vi", deps, signal)
  assert.equal(r.run.version, "1.0.0")
  assert.equal(r.update, null)
  assert.equal(r.offline, true)
  assert.deepEqual(reports, ["manifest_fetch:run_cached_release"])
})

test("offline with nothing cached is an error the app can retry", async () => {
  const { deps } = harness({ fail: true })
  await assert.rejects(resolveRelease("vi", deps, signal), /Failed to fetch/)
})

test("a broken manifest falls back to the cached release too", async () => {
  const { deps, reports } = harness({
    manifest: { schema: 1 },
    cached: release("1.0.0"),
  })
  const r = await resolveRelease("vi", deps, signal)
  assert.equal(r.run.version, "1.0.0")
  assert.deepEqual(reports, ["manifest_invalid:run_cached_release"])
})

test("an old client meeting a newer manifest is told to reload, even with a cached model", async () => {
  for (const manifest of [
    manifestJson("2.0.0", [], { schema: 2 }),
    manifestJson("2.0.0", [], { min_client: 9 }),
  ]) {
    const { deps, reports } = harness({ manifest, cached: release("1.0.0") })
    await assert.rejects(
      resolveRelease("vi", deps, signal),
      (e: unknown) =>
        e instanceof ManifestError && e.problem === "outdated_client"
    )
    assert.deepEqual(reports, ["manifest_outdated_client:reload"])
  }
})

test("the manifest is always revalidated, so an old copy in the HTTP cache cannot win", async () => {
  let seen: RequestInit | undefined
  let url = ""
  const fake = (async (input: string, init?: RequestInit) => {
    url = input
    seen = init
    return new Response(JSON.stringify(manifestJson("2.0.0")))
  }) as unknown as typeof fetch
  const json = (await fetchManifestJson(signal, fake)) as { schema: number }
  assert.equal(json.schema, 1)
  assert.equal(url, MANIFEST_URL)
  assert.equal(seen?.cache, "no-cache")
  const failing = (async () =>
    new Response("x", { status: 503 })) as unknown as typeof fetch
  await assert.rejects(fetchManifestJson(signal, failing), /HTTP 503/)
})

test("an aborted resolve is not reported as a manifest failure", async () => {
  const abort = new AbortController()
  const { deps, reports } = harness({ fail: true, cached: release("1.0.0") })
  abort.abort()
  await assert.rejects(resolveRelease("vi", deps, abort.signal))
  assert.deepEqual(reports, [])
})
