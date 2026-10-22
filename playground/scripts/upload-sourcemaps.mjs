// Opt-in: build the browser bundle with hidden source maps, give every chunk a PostHog chunk id,
// upload the maps to PostHog error tracking (release = the `app_build` the app reports), then
// delete the maps so `dist/` never ships them. Deploy the SAME `dist/` afterwards:
//   pnpm sourcemaps:upload && pnpm exec wrangler deploy
// (a plain `pnpm run deploy` rebuilds without ids, so traces stay minified; nothing else breaks).
//
// Key: a personal API key with only the `error_tracking:write` scope, scoped to project 648258.
// Never stored in the repo. Looked up in this order:
//   1. $POSTHOG_CLI_API_KEY
//   2. macOS Keychain item  service "gidi-posthog-cli"  (security add-generic-password -U -a posthog
//      -s gidi-posthog-cli -w)
//   3. ~/.config/posthog/gidi.env  (POSTHOG_CLI_API_KEY=..., chmod 600)
// Other env: POSTHOG_CLI_PROJECT_ID (default 648258), POSTHOG_CLI_HOST (default https://us.posthog.com).
import { execFileSync, spawnSync } from "node:child_process"
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs"
import { homedir } from "node:os"
import { dirname, join, resolve } from "node:path"
import { fileURLToPath } from "node:url"

const playground = resolve(dirname(fileURLToPath(import.meta.url)), "..")
const dist = join(playground, "dist")
const RELEASE_NAME = "gidi-playground"

const run = (cmd) =>
  execFileSync("sh", ["-c", cmd], { cwd: playground, encoding: "utf8" }).trim()

// Same value as `buildId()` in vite.config.ts; passed down as GIDI_BUILD_ID so they cannot differ.
function buildId() {
  const dirty = run("git status --porcelain -- .") ? "-dirty" : ""
  return run("git rev-parse --short HEAD") + dirty
}

function apiKey() {
  if (process.env.POSTHOG_CLI_API_KEY) return process.env.POSTHOG_CLI_API_KEY
  if (process.platform === "darwin") {
    const r = spawnSync(
      "security",
      ["find-generic-password", "-s", "gidi-posthog-cli", "-w"],
      {
        encoding: "utf8",
      }
    )
    if (r.status === 0 && r.stdout.trim()) return r.stdout.trim()
  }
  const file = join(homedir(), ".config", "posthog", "gidi.env")
  if (existsSync(file)) {
    const m = /^POSTHOG_CLI_API_KEY=(.+)$/m.exec(readFileSync(file, "utf8"))
    if (m) return m[1].trim()
  }
  return null
}

function cli() {
  for (const candidate of [
    "posthog-cli",
    join(homedir(), ".posthog", "posthog-cli"),
  ]) {
    if (spawnSync(candidate, ["--version"], { stdio: "ignore" }).status === 0)
      return candidate
  }
  throw new Error(
    "posthog-cli not found: curl -LsSf https://download.posthog.com/cli | sh"
  )
}

function mapFiles(dir) {
  return readdirSync(dir, { recursive: true, withFileTypes: true })
    .filter((e) => e.isFile() && e.name.endsWith(".map"))
    .map((e) => join(e.parentPath, e.name))
}

const key = apiKey()
if (!key) {
  console.error(
    "No PostHog key. Set POSTHOG_CLI_API_KEY, or store it in the macOS Keychain (service gidi-posthog-cli)."
  )
  process.exit(1)
}
const bin = cli()
const build = buildId()
if (build.endsWith("-dirty")) {
  console.warn(
    `Working tree has changes: the build id is ${build}. Commit first for a stable id.`
  )
}

const env = {
  ...process.env,
  GIDI_SOURCEMAPS: "1",
  GIDI_BUILD_ID: build,
  POSTHOG_CLI_API_KEY: key,
  POSTHOG_CLI_PROJECT_ID: process.env.POSTHOG_CLI_PROJECT_ID ?? "648258",
  POSTHOG_CLI_HOST: process.env.POSTHOG_CLI_HOST ?? "https://us.posthog.com",
}
const sh = (cmd, args) => {
  const r = spawnSync(cmd, args, { cwd: playground, env, stdio: "inherit" })
  if (r.status !== 0)
    throw new Error(`${cmd} ${args.join(" ")} failed (${r.status})`)
}

sh("pnpm", ["run", "build:web"])
if (
  !statSync(join(dist, "assets")).isDirectory() ||
  mapFiles(dist).length === 0
) {
  throw new Error("The build wrote no source maps; nothing to upload.")
}
sh(bin, [
  "sourcemap",
  "process",
  "--directory",
  join(dist, "assets"),
  "--release-name",
  RELEASE_NAME,
  "--release-version",
  build,
  "--delete-after",
])
const left = mapFiles(dist)
if (left.length > 0)
  throw new Error(`Source maps still in dist/: ${left.join(", ")}`)
console.log(
  `Uploaded source maps for ${RELEASE_NAME}@${build}; dist/ has no .map files.`
)
