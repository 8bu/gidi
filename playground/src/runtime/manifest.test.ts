import assert from "node:assert/strict"
import { test } from "node:test"

import {
  CLIENT_VERSION,
  ManifestError,
  parseManifest,
  planLoad,
  type ReleaseEntry,
} from "./manifest.ts"

const release = (
  version: string,
  sha = version.padEnd(64, "0")
): ReleaseEntry => ({
  id: "fam",
  version,
  sha256: sha,
  size_bytes: 10,
  onnx_opset: 17,
  base: `/release/fam/${version}`,
  parts: [{ url: `/models/fam/${version}/model.int8.onnx.01`, size_bytes: 10 }],
})

const entry = (
  active: string | null,
  blocked: string[] = [],
  versions = ["1.0.0", "2.0.0"]
) => ({
  active,
  blocked,
  releases: Object.fromEntries(versions.map((v) => [v, release(v)])),
})

const raw = (overrides: Record<string, unknown> = {}) => ({
  schema: 1,
  min_client: 1,
  models: { vi: entry("2.0.0") },
  ...overrides,
})

test("a valid manifest parses", () => {
  const manifest = parseManifest(raw())
  assert.equal(manifest.models.vi.active, "2.0.0")
  assert.equal(manifest.models.vi.releases["2.0.0"].parts.length, 1)
})

test("a manifest for a newer app is outdated_client, whatever its shape", () => {
  for (const bad of [
    { schema: 2 },
    { min_client: CLIENT_VERSION + 1 },
    { schema: 2, models: "new shape" },
  ]) {
    assert.throws(
      () => parseManifest(raw(bad)),
      (e: unknown) =>
        e instanceof ManifestError && e.problem === "outdated_client"
    )
  }
})

test("a malformed manifest is invalid, not a crash", () => {
  for (const bad of [
    null,
    "x",
    {},
    raw({ models: null }),
    raw({ models: { vi: { active: 3 } } }),
    raw({
      models: {
        vi: { active: null, blocked: [], releases: { "1": { id: "x" } } },
      },
    }),
  ]) {
    assert.throws(
      () => parseManifest(bad),
      (e: unknown) => e instanceof ManifestError && e.problem === "invalid"
    )
  }
})

test("first visit downloads the active release", () => {
  const plan = planLoad(parseManifest(raw()).models.vi, null)
  assert.equal(plan.run.version, "2.0.0")
  assert.equal(plan.update, null)
})

test("the cached active release is used as is", () => {
  const plan = planLoad(parseManifest(raw()).models.vi, release("2.0.0"))
  assert.equal(plan.run.version, "2.0.0")
  assert.equal(plan.update, null)
})

test("rollout: a newer active release runs the cached one now and is fetched in the background", () => {
  const plan = planLoad(parseManifest(raw()).models.vi, release("1.0.0"))
  assert.equal(plan.run.version, "1.0.0")
  assert.equal(plan.update?.version, "2.0.0")
  assert.equal(plan.evictCached, false)
})

test("rollback by manifest: active points back to the older release", () => {
  const plan = planLoad(
    parseManifest(raw({ models: { vi: entry("1.0.0") } })).models.vi,
    release("2.0.0")
  )
  assert.equal(plan.run.version, "2.0.0")
  assert.equal(plan.update?.version, "1.0.0")
})

test("emergency: a blocked cached release never runs; the active one downloads in front", () => {
  const manifest = parseManifest(
    raw({ models: { vi: entry("1.0.0", ["2.0.0"]) } })
  )
  const plan = planLoad(manifest.models.vi, release("2.0.0"))
  assert.equal(plan.run.version, "1.0.0")
  assert.equal(plan.update, null)
  assert.equal(plan.evictCached, true)
})

test("same version but another sha256 is a different release", () => {
  const plan = planLoad(
    parseManifest(raw()).models.vi,
    release("2.0.0", "f".repeat(64))
  )
  assert.equal(plan.run.sha256, "f".repeat(64))
  assert.equal(plan.update?.sha256, "2.0.0".padEnd(64, "0"))
})

test("a disabled language, a missing language and a bad active are errors", () => {
  const disabled = parseManifest(raw({ models: { vi: entry(null) } }))
  assert.throws(
    () => planLoad(disabled.models.vi, release("2.0.0")),
    (e: unknown) => e instanceof ManifestError && e.problem === "disabled"
  )
  assert.throws(
    () => planLoad(undefined, null),
    (e: unknown) => e instanceof ManifestError && e.problem === "invalid"
  )
  const missing = parseManifest(raw({ models: { vi: entry("9.9.9") } }))
  assert.throws(
    () => planLoad(missing.models.vi, null),
    (e: unknown) => e instanceof ManifestError && e.problem === "invalid"
  )
})
