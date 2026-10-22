import type { Lang } from "./language.ts"

/**
 * The two on-device models, one per language. The files under `releaseBase` are written by
 * `scripts/sync-release.mjs` from the immutable bundles (each model is verified in the browser
 * against the SHA-256 in its own `web-release.json`); the weights live in R2 under
 * `/models/<id>/<version>/model.int8.onnx`.
 */
export interface ModelSpec {
  lang: Lang
  /** `model_version` of the bundle, e.g. `gidi-finance-v4`. */
  id: string
  version: string
  releaseBase: string
}

export const MODELS: Readonly<Record<Lang, ModelSpec>> = {
  vi: {
    lang: "vi",
    id: "gidi-finance-v4",
    version: "4.0.0",
    releaseBase: "/release/gidi-finance-v4/4.0.0",
  },
  en: {
    lang: "en",
    id: "gidi-en-v2",
    version: "2.0.1",
    releaseBase: "/release/gidi-en-v2/2.0.1",
  },
}

/** Which language a record's `modelVersion` (the bundle's `model_version`) belongs to. */
export function langOfModel(modelVersion: string): Lang | null {
  for (const spec of Object.values(MODELS)) {
    if (spec.id === modelVersion) return spec.lang
  }
  return null
}
