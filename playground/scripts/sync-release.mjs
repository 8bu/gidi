// Copy the web-facing files of the immutable model bundles into public/release/ and write one
// web-release.json per model: gidi-finance-v4 (Vietnamese, models/gidi-finance-v4, the bytes of
// release 4.0.0) and gidi-en-v2 (English, models/gidi-en-v2-bundle, the bytes of release 2.0.1).
// Every copied byte is checked against the bundle's manifest.json first; the model itself is not
// copied whole: it is cut into PART_BYTES parts under public/models/<id>/<version>/ (Workers
// Static Assets take files up to 25 MiB, are free and never invoke a Worker), and web-release.json
// lists the parts plus the sha256 of the whole file, which the browser verifies after joining
// them. Keep MODELS in sync with src/lib/models.ts.
import { createHash } from "node:crypto"
import { mkdir, readFile, rm, writeFile } from "node:fs/promises"
import { dirname, join, resolve } from "node:path"
import { fileURLToPath } from "node:url"

const MODELS = [
  { id: "gidi-finance-v4", version: "4.0.0", bundle: "gidi-finance-v4" },
  { id: "gidi-en-v2", version: "2.0.1", bundle: "gidi-en-v2-bundle" },
]

const here = dirname(fileURLToPath(import.meta.url))
const playground = resolve(here, "..")

const MODEL_FILE = "model.int8.onnx"
// Below the 25 MiB per-file limit of Workers Static Assets, and two parts download in parallel.
const PART_BYTES = 16 * 1024 * 1024
const COPIED = ["config.json", "tokenizer.json"]

const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex")

async function sync({ id: MODEL_VERSION, version: RELEASE_VERSION, bundle }) {
const bundleDir = resolve(playground, "..", "models", bundle)
const outDir = join(playground, "public", "release", MODEL_VERSION, RELEASE_VERSION)
const partsDir = join(playground, "public", "models", MODEL_VERSION, RELEASE_VERSION)
const manifest = JSON.parse(await readFile(join(bundleDir, "manifest.json"), "utf8"))
if (manifest.model_version !== MODEL_VERSION) {
  throw new Error(`${bundleDir}/manifest.json is ${manifest.model_version}, expected ${MODEL_VERSION}`)
}
const listed = (name) => {
  const entry = manifest.files[name]
  if (!entry) throw new Error(`manifest.json has no file ${name}`)
  return entry
}

await mkdir(outDir, { recursive: true })
await mkdir(partsDir, { recursive: true })

for (const name of COPIED) {
  const entry = listed(name)
  const bytes = await readFile(join(bundleDir, name))
  if (bytes.length !== entry.bytes || sha256(bytes) !== entry.sha256) {
    throw new Error(`${name} does not match manifest.json (size or sha256); refusing to copy`)
  }
  await writeFile(join(outDir, name), bytes)
}

const config = JSON.parse(await readFile(join(outDir, "config.json"), "utf8"))
const model = listed(MODEL_FILE)
const modelBytes = await readFile(join(bundleDir, MODEL_FILE))
if (modelBytes.length !== model.bytes || sha256(modelBytes) !== model.sha256) {
  throw new Error(`${MODEL_FILE} does not match manifest.json (size or sha256); refusing to copy`)
}
const parts = []
for (let start = 0, n = 1; start < modelBytes.length; start += PART_BYTES, n++) {
  const bytes = modelBytes.subarray(start, start + PART_BYTES)
  const name = `${MODEL_FILE}.${String(n).padStart(2, "0")}`
  await writeFile(join(partsDir, name), bytes)
  parts.push({ url: `/models/${MODEL_VERSION}/${RELEASE_VERSION}/${name}`, size_bytes: bytes.length })
}
await writeFile(
  join(outDir, "web-release.json"),
  JSON.stringify(
    {
      model_version: MODEL_VERSION,
      release_version: RELEASE_VERSION,
      onnx_opset: config.onnx.opset,
      model: {
        sha256: model.sha256,
        size_bytes: model.bytes,
        parts,
      },
    },
    null,
    2
  ) + "\n"
)

console.log(
  `synced ${MODEL_VERSION} ${RELEASE_VERSION} -> ${outDir} (model ${model.bytes} B in ${parts.length} parts, ` +
    `sha256 ${model.sha256.slice(0, 12)}…)`
)
}

await rm(join(playground, "public", "release"), { recursive: true, force: true })
await rm(join(playground, "public", "models"), { recursive: true, force: true })
for (const model of MODELS) await sync(model)
