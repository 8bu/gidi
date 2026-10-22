// Concurrent load test for the static site (a temporary workers.dev deployment, never production):
//
//   node scripts/load-test.mjs https://gidi-loadtest.<subdomain>.workers.dev \
//        [--duration 60] [--concurrency 50] [--scenarios html,manifest,manifest-304,js,model] \
//        [--script gidi-loadtest] [--json out.json]
//
// Each scenario runs `--concurrency` closed-loop clients for `--duration` seconds (a request or
// model download that has started is allowed to finish), one scenario after the other:
//   html          GET /
//   manifest      GET /release/manifest.json, no validators (200)
//   manifest-304  the same with If-None-Match from the first answer (expects 304)
//   js            GET the main /assets/index-*.js chunk
//   path:/x       GET any path, e.g. path:/assets/ort-wasm-simd-threaded-<hash>.wasm (14 MB)
//   model         GET both 16 MiB parts of the first language's active release, sha256 of the
//                 joined bytes checked against the manifest (a "request" is one whole model)
// Reports per scenario: requests, errors (network error or unexpected status or sha mismatch),
// p50/p95/p99 total latency, p50 time to first byte, throughput, status and cf-cache-status counts.
// With `--script NAME`, then asks the Cloudflare GraphQL API (token from `wrangler auth token`)
// how many Worker invocations that script and any R2 bucket logged in the run's window; both must
// be 0 for an assets-only site. The API lags a minute or two, so it waits and polls.
import { createHash } from "node:crypto"
import { execFileSync } from "node:child_process"
import { writeFileSync } from "node:fs"

const args = process.argv.slice(2)
const base = args.find((a) => a.startsWith("http"))?.replace(/\/$/, "")
if (!base) throw new Error("usage: load-test.mjs <https://target> [options]")
const opt = (name, fallback) => {
  const i = args.indexOf(`--${name}`)
  return i === -1 ? fallback : args[i + 1]
}
const duration = Number(opt("duration", 60))
const concurrency = Number(opt("concurrency", 50))
const scenarioNames = opt("scenarios", "html,manifest,manifest-304,js,model").split(",")
const script = opt("script", null)
const jsonOut = opt("json", null)
const ACCOUNT = "d3fc0198c8197514b262b12e3c6639a4"
const UA = { "user-agent": "gidi-load-test/1" }

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
const pct = (sorted, p) => (sorted.length === 0 ? NaN : sorted[Math.min(sorted.length - 1, Math.ceil((p / 100) * sorted.length) - 1)])

/** GET, drain the body (optionally into a hash), and time headers and body separately. */
async function get(path, { headers = {}, hash = null, timeoutMs = 20_000 } = {}) {
  const t0 = performance.now()
  const res = await fetch(base + path, { headers: { ...UA, ...headers }, signal: AbortSignal.timeout(timeoutMs) })
  const ttfb = performance.now() - t0
  let bytes = 0
  for await (const chunk of res.body ?? []) {
    bytes += chunk.length
    hash?.update(chunk)
  }
  return { status: res.status, bytes, ttfb, cache: res.headers.get("cf-cache-status") ?? "-", etag: res.headers.get("etag") }
}

const html = await (await fetch(base + "/", { headers: UA })).text()
const manifestRes = await fetch(base + "/release/manifest.json", { headers: UA })
const etag = manifestRes.headers.get("etag")
const manifest = await manifestRes.json()
const mainJs = html.match(/\/assets\/index-[\w-]+\.js/)?.[0]
const lang = Object.values(manifest.models)[0]
const release = lang.releases[lang.active]

const scenarios = {
  html: { expect: [200], once: () => get("/") },
  manifest: { expect: [200], once: () => get("/release/manifest.json", { headers: { "cache-control": "no-cache" } }) },
  "manifest-304": { expect: [304], once: () => get("/release/manifest.json", { headers: { "if-none-match": etag, "cache-control": "no-cache" } }) },
  js: { expect: [200], once: () => get(mainJs) },
  model: {
    expect: [200],
    async once() {
      const hash = createHash("sha256")
      const t0 = performance.now()
      let bytes = 0
      let ttfb = 0
      let cache = "-"
      for (const part of release.parts) {
        const r = await get(part.url, { hash, timeoutMs: 600_000 })
        if (r.status !== 200) return { ...r, bytes: bytes + r.bytes }
        bytes += r.bytes
        ttfb = ttfb || r.ttfb
        cache = r.cache
      }
      const ok = bytes === release.size_bytes && hash.digest("hex") === release.sha256
      return { status: ok ? 200 : 599, bytes, ttfb, cache, ms: performance.now() - t0 }
    },
  },
}

async function runScenario(name) {
  const sc = name.startsWith("path:") ? { expect: [200], once: () => get(name.slice(5), { timeoutMs: 60_000 }) } : scenarios[name]
  if (!sc) throw new Error(`unknown scenario ${name}`)
  const lat = []
  const ttfb = []
  const status = {}
  const cache = {}
  let bytes = 0
  let errors = 0
  const errorKinds = {}
  const start = performance.now()
  const stopAt = Date.now() + duration * 1000
  const client = async () => {
    while (Date.now() < stopAt) {
      const t0 = performance.now()
      try {
        const r = await sc.once()
        lat.push(r.ms ?? performance.now() - t0)
        ttfb.push(r.ttfb)
        bytes += r.bytes
        status[r.status] = (status[r.status] ?? 0) + 1
        cache[r.cache] = (cache[r.cache] ?? 0) + 1
        if (!sc.expect.includes(r.status)) errors++
      } catch (e) {
        errors++
        const k = e?.cause?.code ?? e?.name ?? "error"
        errorKinds[k] = (errorKinds[k] ?? 0) + 1
      }
    }
  }
  await Promise.all(Array.from({ length: concurrency }, client))
  const wall = (performance.now() - start) / 1000
  lat.sort((a, b) => a - b)
  ttfb.sort((a, b) => a - b)
  const n = lat.length + Object.values(errorKinds).reduce((a, b) => a + b, 0)
  return {
    scenario: name,
    requests: n,
    errors,
    errorRate: n === 0 ? null : errors / n,
    errorKinds,
    wallSeconds: Number(wall.toFixed(1)),
    rps: Number((n / wall).toFixed(1)),
    mbPerSecond: Number((bytes / 1e6 / wall).toFixed(1)),
    totalMB: Number((bytes / 1e6).toFixed(1)),
    p50ms: Math.round(pct(lat, 50)),
    p95ms: Math.round(pct(lat, 95)),
    p99ms: Math.round(pct(lat, 99)),
    ttfbP50ms: Math.round(pct(ttfb, 50)),
    ttfbP95ms: Math.round(pct(ttfb, 95)),
    status,
    cache,
  }
}

async function graphql(query) {
  const token = execFileSync("npx", ["wrangler", "auth", "token"], { encoding: "utf8" }).trim().split("\n").pop()
  const res = await fetch("https://api.cloudflare.com/client/v4/graphql", {
    method: "POST",
    headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
    body: JSON.stringify({ query }),
  })
  return res.json()
}

async function analytics(from, to) {
  const invocations = await graphql(`query { viewer { accounts(filter:{accountTag:"${ACCOUNT}"}) {
    workersInvocationsAdaptive(limit:1000, filter:{scriptName:"${script}", datetime_geq:"${from}", datetime_leq:"${to}"}) {
      sum { requests errors subrequests cpuTimeUs } } } } }`)
  const r2 = await graphql(`query { viewer { accounts(filter:{accountTag:"${ACCOUNT}"}) {
    r2OperationsAdaptiveGroups(limit:1000, filter:{datetime_geq:"${from}", datetime_leq:"${to}"}) {
      sum { requests } dimensions { actionType bucketName } } } } }`)
  const rows = invocations.data?.viewer?.accounts?.[0]?.workersInvocationsAdaptive ?? []
  return {
    invocations: rows.reduce((a, r) => a + r.sum.requests, 0),
    cpuMs: rows.reduce((a, r) => a + r.sum.cpuTimeUs / 1000, 0),
    subrequests: rows.reduce((a, r) => a + r.sum.subrequests, 0),
    r2Ops: (r2.data?.viewer?.accounts?.[0]?.r2OperationsAdaptiveGroups ?? []).reduce((a, r) => a + r.sum.requests, 0),
    errors: [...(invocations.errors ?? []), ...(r2.errors ?? [])].map((e) => e.message),
  }
}

console.log(`target ${base}: ${concurrency} clients x ${duration}s per scenario; model ${release.id} ${release.version} (${release.parts.length} parts)`)
const t0 = new Date()
const results = []
for (const name of scenarioNames) {
  const r = await runScenario(name)
  results.push(r)
  console.log(
    `${name.padEnd(13)} req ${String(r.requests).padStart(6)}  err ${r.errors} (${((r.errorRate ?? 0) * 100).toFixed(2)}%)  ` +
      `${r.rps} req/s  ${r.mbPerSecond} MB/s  p50 ${r.p50ms} ms  p95 ${r.p95ms} ms  p99 ${r.p99ms} ms  ttfb p50 ${r.ttfbP50ms} ms  ` +
      `status ${JSON.stringify(r.status)}  cache ${JSON.stringify(r.cache)}` +
      (r.errors > 0 ? `  errors ${JSON.stringify(r.errorKinds)}` : "")
  )
}
const t1 = new Date()
let analyticsResult = null
if (script) {
  const from = new Date(t0.getTime() - 30_000).toISOString()
  const to = new Date(t1.getTime() + 30_000).toISOString()
  const deadline = Date.now() + 6 * 60_000
  await sleep(90_000)
  while (true) {
    analyticsResult = await analytics(from, to)
    console.log(`analytics: Worker invocations ${analyticsResult.invocations}, R2 operations ${analyticsResult.r2Ops}`)
    if (Date.now() > deadline || analyticsResult.errors.length === 0) break
    await sleep(45_000)
  }
  console.log(`window ${from} .. ${to}: script ${script}: invocations ${analyticsResult.invocations}, CPU ${analyticsResult.cpuMs.toFixed(1)} ms, subrequests ${analyticsResult.subrequests}; R2 operations (all buckets) ${analyticsResult.r2Ops}`)
  if (analyticsResult.errors.length > 0) console.log(`analytics errors: ${analyticsResult.errors.join("; ")}`)
}
if (jsonOut) writeFileSync(jsonOut, JSON.stringify({ base, duration, concurrency, startedAt: t0.toISOString(), results, analytics: analyticsResult }, null, 2))
