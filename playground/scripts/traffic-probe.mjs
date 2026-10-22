// Cost probe for the deployed site: replays N first visits and M returning visits, then asks the
// Cloudflare GraphQL API how many Worker invocations the script `gidi` logged in that window.
//
//   node scripts/traffic-probe.mjs https://gidi.8bu.dev --first 3 --returning 0
//   node scripts/traffic-probe.mjs https://gidi.8bu.dev --first 0 --returning 5
//   node scripts/traffic-probe.mjs https://gidi.8bu.dev --first 0 --returning 0 --burst 200
// (one phase per run, so the invocation count divides cleanly by the number of visits)
//
// A first visit fetches `/`, the manifest, every file the page and its JS chunks reference, both
// models' release files and the first language's model parts. A returning visit fetches only `/` (the rest sits in the browser's
// cache: immutable headers, plus Cache Storage for the model). The burst sends `--burst` GETs to `/`
// and to one model file (Range 0-1023, so it moves no model bytes) and reports the edge cache
// status. Invocation counts come from `workersInvocationsAdaptive`; the API lags by a minute or
// two, so the script polls for up to 6 minutes. It needs `wrangler login` (the token is read with
// `wrangler auth token`); `--no-analytics` only prints the per-request table.
import { execFileSync } from "node:child_process"

const args = process.argv.slice(2)
const base = (args.find((a) => a.startsWith("http")) ?? "https://gidi.8bu.dev").replace(/\/$/, "")
const opt = (name, fallback) => {
  const i = args.indexOf(`--${name}`)
  return i === -1 ? fallback : Number(args[i + 1])
}
const firstVisits = opt("first", 1)
const returning = opt("returning", 1)
const burst = opt("burst", 0)
const analytics = !args.includes("--no-analytics")
const ACCOUNT = "d3fc0198c8197514b262b12e3c6639a4"
const SCRIPT = "gidi"

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

/** GET and drain the body; the model parts are tens of MB, so only count bytes. */
async function hit(path, init = {}) {
  const res = await fetch(base + path, { ...init, headers: { "user-agent": "gidi-cost-probe/1", ...init.headers } })
  let bytes = 0
  for await (const chunk of res.body ?? []) bytes += chunk.length
  return {
    path,
    status: res.status,
    bytes,
    type: (res.headers.get("content-type") ?? "").split(";")[0],
    cache: res.headers.get("cf-cache-status") ?? "-",
    cc: res.headers.get("cache-control") ?? "-",
    body: res,
  }
}

async function text(path) {
  const res = await fetch(base + path, { headers: { "user-agent": "gidi-cost-probe/1" } })
  return res.text()
}

/** Every `/assets/...` file the page references, following JS chunks one level at a time. */
async function crawlAssets(html) {
  const seen = new Set()
  const queue = [...html.matchAll(/["'(](\/assets\/[\w.-]+)/g)].map((m) => m[1])
  while (queue.length > 0) {
    const path = queue.pop()
    if (seen.has(path)) continue
    seen.add(path)
    if (!path.endsWith(".js")) continue
    const js = await text(path)
    for (const m of js.matchAll(/["'`(,]\.?\/?(assets\/[\w.-]+\.(?:js|css|wasm|woff2))/g)) queue.push(`/${m[1]}`)
    for (const m of js.matchAll(/["'`]\.\/([\w.-]+\.(?:js|css))/g)) queue.push(`/assets/${m[1]}`)
  }
  return [...seen].filter((p) => /\.(js|css|wasm|woff2)$/.test(p)).sort()
}

async function manifest() {
  return JSON.parse(await text("/release/manifest.json"))
}

/** Per language: the active release's config/tokenizer base path and model part URLs. */
function activeReleases(m) {
  return Object.values(m.models).map((lang) => lang.releases[lang.active])
}

async function firstVisit(assets, releases) {
  const rows = [await hit("/"), await hit("/release/manifest.json")]
  for (const path of assets) rows.push(await hit(path))
  for (const rel of releases) {
    for (const f of ["config.json", "tokenizer.json"]) rows.push(await hit(`${rel.base}/${f}`))
  }
  rows.push(...(await Promise.all(releases[0].parts.map((p) => hit(p.url)))))
  return rows
}

async function invocations(from, to) {
  const token = execFileSync("npx", ["wrangler", "auth", "token"], { encoding: "utf8" }).trim().split("\n").pop()
  const query = `query { viewer { accounts(filter:{accountTag:"${ACCOUNT}"}) {
    workersInvocationsAdaptive(limit:1000, filter:{scriptName:"${SCRIPT}", datetime_geq:"${from}", datetime_leq:"${to}"}) {
      sum { requests errors subrequests cpuTimeUs } } } } }`
  const res = await fetch("https://api.cloudflare.com/client/v4/graphql", {
    method: "POST",
    headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
    body: JSON.stringify({ query }),
  })
  const json = await res.json()
  const rows = json.data?.viewer?.accounts?.[0]?.workersInvocationsAdaptive ?? []
  return rows.reduce(
    (a, r) => ({
      requests: a.requests + r.sum.requests,
      cpuMs: a.cpuMs + r.sum.cpuTimeUs / 1000,
      subrequests: a.subrequests + r.sum.subrequests,
    }),
    { requests: 0, cpuMs: 0, subrequests: 0 }
  )
}

const html = await text("/")
const assets = await crawlAssets(html)
const releases = activeReleases(await manifest())
console.log(`target ${base}: ${assets.length} asset files, models ${releases.flatMap((r) => r.parts.map((p) => p.url)).join(", ")}`)

const t0 = new Date()
const rows = []
for (let i = 0; i < firstVisits; i++) rows.push(...(await firstVisit(assets, releases)))
for (let i = 0; i < returning; i++) rows.push(await hit("/"))

const table = new Map()
for (const r of rows) {
  const k = `${r.path} ${r.status} ${r.cache}`
  table.set(k, (table.get(k) ?? 0) + 1)
}
console.log(`\nrequests (path status cf-cache-status): ${rows.length}`)
for (const [k, n] of table) console.log(`  ${String(n).padStart(3)}x ${k}`)

const burstStats = {}
if (burst > 0) {
  const modelPath = releases[0].parts.at(-1).url
  const results = []
  const tasks = []
  const flush = async () => results.push(...(await Promise.all(tasks.splice(0))))
  for (let i = 0; i < burst; i++) {
    tasks.push(hit("/").then((r) => ["/", r]))
    tasks.push(hit(modelPath, { headers: { range: "bytes=0-1023" } }).then((r) => ["model-range", r]))
    if (tasks.length >= 40) await flush()
  }
  await flush()
  for (const [name, r] of results) {
    const s = (burstStats[name] ??= { n: 0, cache: {}, status: {} })
    s.n++
    s.cache[r.cache] = (s.cache[r.cache] ?? 0) + 1
    s.status[r.status] = (s.status[r.status] ?? 0) + 1
  }
  console.log(`\nburst of ${burst} pairs:`)
  for (const [name, s] of Object.entries(burstStats)) {
    console.log(`  ${name}: ${s.n} requests, status ${JSON.stringify(s.status)}, cf-cache-status ${JSON.stringify(s.cache)}`)
  }
}
const tEnd = new Date()

if (analytics) {
  const from = new Date(t0.getTime() - 30_000).toISOString()
  const to = new Date(tEnd.getTime() + 30_000).toISOString()
  const expected = rows.length + (burst > 0 ? burst * 2 : 0)
  let got = { requests: 0, cpuMs: 0, subrequests: 0 }
  const deadline = Date.now() + 6 * 60_000
  while (Date.now() < deadline) {
    await sleep(45_000)
    const latest = await invocations(from, to)
    if (latest.requests >= got.requests) got = latest // the API occasionally answers with nothing
    process.stdout.write(`  analytics: ${got.requests} Worker invocations so far\n`)
    if (got.requests > 0 && Date.now() - tEnd.getTime() > 120_000) break
  }
  console.log(
    `\nWorker invocations in window ${from} .. ${to} (includes any real traffic): ${got.requests} ` +
      `of ${expected} requests sent; CPU ${got.cpuMs.toFixed(1)} ms; subrequests (R2/cache) ${got.subrequests}`
  )
  const visits = firstVisits + returning
  if (burst === 0 && visits > 0) {
    console.log(`= ${(got.requests / visits).toFixed(2)} Worker invocations per visit (${firstVisits} first, ${returning} returning)`)
  }
}
