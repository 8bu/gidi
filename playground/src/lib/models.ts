import type { Lang } from "./language.ts"

/**
 * The two on-device models, one per language: UI labels and which language a stored record
 * belongs to (`langOfModel`). Which release actually runs, its files and its SHA-256 come from
 * `/release/manifest.json` (see `runtime/manifest.ts`); a new release of the same family changes
 * `web-models.json` only, while the labels here follow it on the next code deploy.
 */
export interface ModelSpec {
  lang: Lang
  /** `model_version` of the bundle, e.g. `gidi-finance-v4`. */
  id: string
  version: string
}

export const MODELS: Readonly<Record<Lang, ModelSpec>> = {
  vi: {
    lang: "vi",
    id: "gidi-finance-v4",
    version: "4.0.0",
  },
  en: {
    lang: "en",
    id: "gidi-en-v4",
    version: "4.0.0",
  },
}

/** Which language a record's `modelVersion` (the bundle's `model_version`) belongs to. */
export function langOfModel(modelVersion: string): Lang | null {
  for (const spec of Object.values(MODELS)) {
    if (spec.id === modelVersion) return spec.lang
  }
  return null
}
