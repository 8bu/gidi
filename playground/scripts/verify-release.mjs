// Post-deploy (or pre-deploy, against `wrangler dev`) check of the model rollout contract:
//
//   node scripts/verify-release.mjs https://gidi.8bu.dev [--hash]
//
// - /release/manifest.json is `no-cache`, has a supported schema, every language has an active
//   release that is not blocked;
// - every release's config.json / tokenizer.json / model parts exist, are immutable, and the parts
//   add up to the listed size (with --hash: the joined parts match the manifest's sha256);
// - the page is served and (on the production host) carries the `Server-Timing: geo` hint.
// Exit code 1 on any failure.
import { createHash } from "node:crypto"

const args = process.argv.slice(2)
const base = (args.find((a) => a.startsWith("http")) ?? "https://gidi.8bu.dev").replace(/\/$/, "")
const hash = args.includes("--hash")
let failed = 0
const check = (ok, what) => {
  console.log(`${ok ? "ok  " : "FAIL"} ${what}`)
  if (!ok) failed++
}
const get = (path, init) => fetch(base + path, { headers: { "user-agent": "gidi-verify-release/1" }, ...init })

const res = await get("/release/manifest.json", { cache: "no-store" })
check(res.ok, `manifest 200 (${res.status})`)
check((res.headers.get("cache-control") ?? "").includes("no-cache"), `manifest cache-control no-cache (${res.headers.get("cache-control")})`)
const manifest = await res.json()
check(manifest.schema === 1 && Number.isInteger(manifest.min_client), `schema ${manifest.schema}, min_client ${manifest.min_client}`)

for (const [lang, entry] of Object.entries(manifest.models)) {
  const release = entry.releases[entry.active]
  check(release !== undefined && !entry.blocked.includes(entry.active), `${lang}: active ${entry.active} exists and is not blocked`)
  for (const [version, r] of Object.entries(entry.releases)) {
    for (const file of ["config.json", "tokenizer.json"]) {
      const head = await get(`${r.base}/${file}`, { method: "HEAD" })
      check(head.ok && (head.headers.get("cache-control") ?? "").includes("immutable"), `${lang} ${version} ${file} immutable`)
    }
    let total = 0
    const sha = createHash("sha256")
    for (const part of r.parts) {
      const head = await get(part.url, { method: "HEAD" })
      let length = Number(head.headers.get("content-length"))
      if (!(length > 0)) {
        // Some servers omit Content-Length on HEAD: ask for one byte and read the total.
        const one = await get(part.url, { headers: { range: "bytes=0-0" } })
        length = Number((one.headers.get("content-range") ?? "").split("/")[1])
        const body = await one.arrayBuffer()
        if (!(length > 0)) length = body.byteLength // the server ignored Range and sent it all
      }
      check(head.ok && length === part.size_bytes && (head.headers.get("cache-control") ?? "").includes("immutable"), `${lang} ${version} ${part.url.split("/").pop()} ${length} B, immutable`)
      total += part.size_bytes
      if (hash) sha.update(Buffer.from(await (await get(part.url)).arrayBuffer()))
    }
    check(total === r.size_bytes, `${lang} ${version} parts add up to ${r.size_bytes} B`)
    if (hash) check(sha.digest("hex") === r.sha256, `${lang} ${version} joined parts match the manifest sha256`)
  }
}

const page = await get("/")
check(page.ok && (page.headers.get("content-type") ?? "").includes("text/html"), "page serves HTML")
if (base === "https://gidi.8bu.dev") {
  check(/geo;desc=[A-Z0-9]{2}/.test(page.headers.get("server-timing") ?? ""), `page has Server-Timing geo (${page.headers.get("server-timing")})`)
}
process.exit(failed === 0 ? 0 : 1)
