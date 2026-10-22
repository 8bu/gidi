/**
 * The release manifest (`/release/manifest.json`, written by `scripts/sync-release.mjs` from
 * `web-models.json`): the single source of truth for which model each language runs. It is the
 * only file that changes under a fixed URL, so it is served `no-cache` and fetched with
 * `cache: "no-cache"`; model files are immutable and live at versioned paths.
 *
 * Pure and dependency-free so `manifest.test.ts` can run every rollout case under node.
 */

/** Shape of the manifest this client understands. */
export const MANIFEST_SCHEMA = 1
/** Bump with a client change that older manifests must not reach (`min_client`). */
export const CLIENT_VERSION = 1

export interface ReleasePart {
  url: string
  size_bytes: number
}

export interface ReleaseEntry {
  id: string
  version: string
  sha256: string
  size_bytes: number
  onnx_opset: number | null
  /** Directory of `config.json` and `tokenizer.json` (immutable, versioned). */
  base: string
  parts: ReleasePart[]
}

export interface LanguageEntry {
  /** The release to run; `null` switches the language off (a bad model, no code deploy). */
  active: string | null
  /** Versions that must never run, even from the browser's cache. */
  blocked: string[]
  releases: Record<string, ReleaseEntry>
}

export interface Manifest {
  schema: number
  min_client: number
  models: Record<string, LanguageEntry>
}

export type ManifestProblem =
  | "outdated_client" // the manifest is newer than this code: reload to get the new code
  | "invalid" // the manifest is malformed or misses the language
  | "disabled" // the language is switched off in the manifest

export class ManifestError extends Error {
  readonly problem: ManifestProblem
  constructor(problem: ManifestProblem, message: string) {
    super(message)
    this.name = "ManifestError"
    this.problem = problem
  }
}

const isRecord = (v: unknown): v is Record<string, unknown> =>
  typeof v === "object" && v !== null && !Array.isArray(v)

export function parseRelease(raw: unknown): ReleaseEntry | null {
  if (!isRecord(raw) || !Array.isArray(raw.parts) || raw.parts.length === 0) {
    return null
  }
  const parts: ReleasePart[] = []
  for (const part of raw.parts) {
    if (
      !isRecord(part) ||
      typeof part.url !== "string" ||
      !Number.isInteger(part.size_bytes)
    ) {
      return null
    }
    parts.push({ url: part.url, size_bytes: part.size_bytes as number })
  }
  if (
    typeof raw.id !== "string" ||
    typeof raw.version !== "string" ||
    typeof raw.sha256 !== "string" ||
    !Number.isInteger(raw.size_bytes) ||
    typeof raw.base !== "string"
  ) {
    return null
  }
  return {
    id: raw.id,
    version: raw.version,
    sha256: raw.sha256,
    size_bytes: raw.size_bytes as number,
    onnx_opset: typeof raw.onnx_opset === "number" ? raw.onnx_opset : null,
    base: raw.base,
    parts,
  }
}

/** Throws `ManifestError`. Compatibility is judged first: a newer schema may have another shape. */
export function parseManifest(
  raw: unknown,
  clientVersion: number = CLIENT_VERSION
): Manifest {
  if (
    !isRecord(raw) ||
    !Number.isInteger(raw.schema) ||
    !Number.isInteger(raw.min_client)
  ) {
    throw new ManifestError("invalid", "the release manifest is malformed")
  }
  if (
    raw.schema !== MANIFEST_SCHEMA ||
    (raw.min_client as number) > clientVersion
  ) {
    throw new ManifestError(
      "outdated_client",
      `manifest schema ${String(raw.schema)} / min_client ${String(raw.min_client)} ` +
        `needs a newer app than schema ${MANIFEST_SCHEMA} / client ${clientVersion}`
    )
  }
  if (!isRecord(raw.models)) {
    throw new ManifestError("invalid", "the release manifest has no models")
  }
  const models: Record<string, LanguageEntry> = {}
  for (const [lang, entry] of Object.entries(raw.models)) {
    if (
      !isRecord(entry) ||
      !(entry.active === null || typeof entry.active === "string") ||
      !Array.isArray(entry.blocked) ||
      !isRecord(entry.releases)
    ) {
      throw new ManifestError(
        "invalid",
        `the release manifest is malformed for ${lang}`
      )
    }
    const releases: Record<string, ReleaseEntry> = {}
    for (const [version, release] of Object.entries(entry.releases)) {
      const parsed = parseRelease(release)
      if (parsed === null) {
        throw new ManifestError(
          "invalid",
          `release ${lang} ${version} is malformed`
        )
      }
      releases[version] = parsed
    }
    models[lang] = {
      active: entry.active,
      blocked: entry.blocked.filter((v): v is string => typeof v === "string"),
      releases,
    }
  }
  return {
    schema: MANIFEST_SCHEMA,
    min_client: raw.min_client as number,
    models,
  }
}

export interface LoadPlan {
  /** The release to load now. */
  run: ReleaseEntry
  /**
   * A newer (or rolled-back) release to fetch in the background and switch to on the next load,
   * once its sha256 has passed; `null` when `run` is already the active one.
   */
  update: ReleaseEntry | null
  /** The cached release was blocked or damaged; its entries must go. */
  evictCached: boolean
}

/**
 * What to load for one language. `cached` is the release recorded in Cache Storage (or `null`).
 * - active release already cached: run it;
 * - nothing cached, or the cached release is blocked: download the active one and run it;
 * - otherwise run the cached release and prefetch the active one (no forced re-download, no
 *   waiting; the switch happens on the next load, after the download verified).
 */
export function planLoad(
  entry: LanguageEntry | undefined,
  cached: ReleaseEntry | null
): LoadPlan {
  if (entry === undefined) {
    throw new ManifestError(
      "invalid",
      "the release manifest does not list this language"
    )
  }
  if (entry.active === null) {
    throw new ManifestError("disabled", "this language's model is switched off")
  }
  const active = entry.releases[entry.active]
  if (active === undefined || entry.blocked.includes(active.version)) {
    throw new ManifestError(
      "invalid",
      `active release ${entry.active} is missing or blocked`
    )
  }
  if (cached === null) return { run: active, update: null, evictCached: false }
  if (cached.version === active.version && cached.sha256 === active.sha256) {
    return { run: active, update: null, evictCached: false }
  }
  if (entry.blocked.includes(cached.version)) {
    return { run: active, update: null, evictCached: true }
  }
  return { run: cached, update: active, evictCached: false }
}
