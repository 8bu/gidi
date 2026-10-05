// Copy the web-facing files of the immutable gidi-finance-v3 bundle (models/gidi-finance-v3, the
// bytes of release 3.0.0) into public/release/ and write web-release.json. Every copied byte is
// checked against the bundle's manifest.json first; the model itself is not copied (it is served
// from R2), but its sha256 and size are read from the manifest and published in web-release.json
// so the browser can verify its download.
import { createHash } from "node:crypto"
import { mkdir, readFile, rm, writeFile } from "node:fs/promises"
import { dirname, join, resolve } from "node:path"
import { fileURLToPath } from "node:url"

const MODEL_VERSION = "gidi-finance-v3"
const RELEASE_VERSION = "3.0.0"

const here = dirname(fileURLToPath(import.meta.url))
const playground = resolve(here, "..")
const bundleDir = resolve(playground, "..", "models", MODEL_VERSION)
const outDir = join(playground, "public", "release", MODEL_VERSION, RELEASE_VERSION)

const MODEL_FILE = "model.int8.onnx"
const COPIED = ["config.json", "tokenizer.json"]

const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex")

const manifest = JSON.parse(await readFile(join(bundleDir, "manifest.json"), "utf8"))
if (manifest.model_version !== MODEL_VERSION) {
  throw new Error(`${bundleDir}/manifest.json is ${manifest.model_version}, expected ${MODEL_VERSION}`)
}
const listed = (name) => {
  const entry = manifest.files[name]
  if (!entry) throw new Error(`manifest.json has no file ${name}`)
  return entry
}

await rm(join(playground, "public", "release"), { recursive: true, force: true })
await mkdir(outDir, { recursive: true })

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
await writeFile(
  join(outDir, "web-release.json"),
  JSON.stringify(
    {
      model_version: MODEL_VERSION,
      release_version: RELEASE_VERSION,
      onnx_opset: config.onnx.opset,
      model: {
        url: `/models/${MODEL_VERSION}/${RELEASE_VERSION}/${MODEL_FILE}`,
        sha256: model.sha256,
        size_bytes: model.bytes,
      },
    },
    null,
    2
  ) + "\n"
)

console.log(
  `synced ${MODEL_VERSION} ${RELEASE_VERSION} -> ${outDir} (model ${model.bytes} B, ` +
    `sha256 ${model.sha256.slice(0, 12)}…)`
)
