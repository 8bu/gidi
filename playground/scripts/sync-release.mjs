// Copy the web-facing files of the immutable gidi-finance-v2 release into public/release/ and write
// web-release.json. Every copied byte is checked against the release manifest.json first; the
// model itself is not copied (it is served from R2), but its sha256 and size are read from the
// manifest and published in web-release.json so the browser can verify its download.
import { createHash } from "node:crypto"
import { mkdir, readFile, rm, writeFile } from "node:fs/promises"
import { dirname, join, resolve } from "node:path"
import { fileURLToPath } from "node:url"

const MODEL_VERSION = "gidi-finance-v2"
const RELEASE_VERSION = "2.0.2"

const here = dirname(fileURLToPath(import.meta.url))
const playground = resolve(here, "..")
const releaseDir = resolve(playground, "..", "dist", "releases", MODEL_VERSION, RELEASE_VERSION)
const outDir = join(playground, "public", "release", MODEL_VERSION, RELEASE_VERSION)

const MODEL_ARTIFACT = "model/int8/model.int8.onnx"
const COPIED = [
  ["runtime/config.json", "config.json"],
  ["tokenizer/tokenizer.json", "tokenizer.json"],
]

const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex")

const manifest = JSON.parse(await readFile(join(releaseDir, "manifest.json"), "utf8"))
if (manifest.model_version !== MODEL_VERSION || manifest.release_version !== RELEASE_VERSION) {
  throw new Error(
    `${releaseDir}/manifest.json is ${manifest.model_version} ${manifest.release_version}, ` +
      `expected ${MODEL_VERSION} ${RELEASE_VERSION}`
  )
}
const artifact = (path) => {
  const entry = manifest.artifacts.find((item) => item.path === path)
  if (!entry) throw new Error(`manifest.json has no artifact ${path}`)
  return entry
}

await rm(join(playground, "public", "release"), { recursive: true, force: true })
await mkdir(outDir, { recursive: true })

for (const [source, target] of COPIED) {
  const entry = artifact(source)
  const bytes = await readFile(join(releaseDir, source))
  if (bytes.length !== entry.size_bytes || sha256(bytes) !== entry.sha256) {
    throw new Error(`${source} does not match manifest.json (size or sha256); refusing to copy`)
  }
  await writeFile(join(outDir, target), bytes)
}

const model = artifact(MODEL_ARTIFACT)
await writeFile(
  join(outDir, "web-release.json"),
  JSON.stringify(
    {
      model_version: MODEL_VERSION,
      release_version: RELEASE_VERSION,
      onnx_opset: manifest.runtime.opset,
      model: {
        url: `/models/${MODEL_VERSION}/${RELEASE_VERSION}/model.int8.onnx`,
        sha256: model.sha256,
        size_bytes: model.size_bytes,
      },
    },
    null,
    2
  ) + "\n"
)

console.log(
  `synced ${MODEL_VERSION} ${RELEASE_VERSION} -> ${outDir} (model ${model.size_bytes} B, ` +
    `sha256 ${model.sha256.slice(0, 12)}…)`
)
