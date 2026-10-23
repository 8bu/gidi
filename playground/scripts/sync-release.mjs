// Publish the web-facing files of the immutable model bundles for the static site and write the
// manifest the app reads. Source of truth for what is served: web-models.json (per language: the
// releases to ship, the `active` one, a `blocked` list, and the manifest `schema` / `min_client`).
//
// For every release listed there, in this order:
//   - the bundle's manifest.json is read and every copied byte is checked against it;
//   - config.json and tokenizer.json go to public/release/<id>/<version>/ (immutable URLs);
//   - model.int8.onnx is cut into PART_BYTES parts under public/models/<id>/<version>/ (Workers
//     Static Assets take files up to 25 MiB, are free and never invoke a Worker); the browser joins
//     them and checks the sha256 of the whole file;
// then public/release/manifest.json is written: the one small file the app revalidates on every
// load (`Cache-Control: no-cache`, see public/_headers). A new model = a new <id>/<version> path,
// never an overwrite. Keep MODELS in src/lib/models.ts (UI labels, record ownership) in sync.
import { createHash } from "node:crypto"
import { mkdir, readFile, rm, writeFile } from "node:fs/promises"
import { dirname, join, resolve } from "node:path"
import { fileURLToPath } from "node:url"

const here = dirname(fileURLToPath(import.meta.url))
const playground = resolve(here, "..")

const MODEL_FILE = "model.int8.onnx"
const COPIED = ["config.json", "tokenizer.json"]
// Below the 25 MiB per-file limit of Workers Static Assets, and two parts download in parallel.
const PART_BYTES = 16 * 1024 * 1024

const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex")

async function sync({ id, version, bundle }) {
  const bundleDir = resolve(playground, "..", "models", bundle)
  const outDir = join(playground, "public", "release", id, version)
  const partsDir = join(playground, "public", "models", id, version)
  const manifest = JSON.parse(await readFile(join(bundleDir, "manifest.json"), "utf8"))
  if (manifest.model_version !== id) {
    throw new Error(`${bundleDir}/manifest.json is ${manifest.model_version}, expected ${id}`)
  }
  const listed = (name) => {
    const entry = manifest.files[name]
    if (!entry) throw new Error(`manifest.json has no file ${name}`)
    return entry
  }
  const verified = async (name) => {
    const entry = listed(name)
    const bytes = await readFile(join(bundleDir, name))
    if (bytes.length !== entry.bytes || sha256(bytes) !== entry.sha256) {
      throw new Error(`${name} does not match manifest.json (size or sha256); refusing to copy`)
    }
    return bytes
  }

  await mkdir(outDir, { recursive: true })
  await mkdir(partsDir, { recursive: true })
  for (const name of COPIED) await writeFile(join(outDir, name), await verified(name))

  const config = JSON.parse(await readFile(join(outDir, "config.json"), "utf8"))
  const modelBytes = await verified(MODEL_FILE)
  const parts = []
  for (let start = 0, n = 1; start < modelBytes.length; start += PART_BYTES, n++) {
    const bytes = modelBytes.subarray(start, start + PART_BYTES)
    const name = `${MODEL_FILE}.${String(n).padStart(2, "0")}`
    await writeFile(join(partsDir, name), bytes)
    parts.push({ url: `/models/${id}/${version}/${name}`, size_bytes: bytes.length })
  }
  const model = listed(MODEL_FILE)
  console.log(
    `synced ${id} ${version} (model ${model.bytes} B in ${parts.length} parts, ` +
      `sha256 ${model.sha256.slice(0, 12)}…)`
  )
  return {
    id,
    version,
    sha256: model.sha256,
    size_bytes: model.bytes,
    onnx_opset: config.onnx?.opset ?? null,
    tokenizer_type: config.tokenizer_type ?? "bpe",
    base: `/release/${id}/${version}`,
    parts,
  }
}

const source = JSON.parse(await readFile(join(playground, "web-models.json"), "utf8"))
await rm(join(playground, "public", "release"), { recursive: true, force: true })
await rm(join(playground, "public", "models"), { recursive: true, force: true })

const models = {}
for (const [lang, entry] of Object.entries(source.languages)) {
  const releases = {}
  for (const release of entry.releases) releases[release.version] = await sync(release)
  if (entry.active !== null && !releases[entry.active]) {
    throw new Error(`${lang}: active ${entry.active} is not in its releases`)
  }
  if (entry.blocked.includes(entry.active)) throw new Error(`${lang}: active ${entry.active} is blocked`)
  models[lang] = { active: entry.active, blocked: entry.blocked, releases }
}

await mkdir(join(playground, "public", "release"), { recursive: true })
await writeFile(
  join(playground, "public", "release", "manifest.json"),
  JSON.stringify({ schema: source.schema, min_client: source.min_client, models }, null, 2) + "\n"
)
console.log("wrote public/release/manifest.json")
