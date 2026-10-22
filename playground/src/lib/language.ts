/**
 * Which language (and so which model) the notes app starts in. Pure and dependency-free: the
 * Worker and the browser share it, and `language.test.ts` runs it under node.
 *
 * Precedence: the visitor's own choice (`gidi:lang` in localStorage) always wins; then the
 * Worker's coarse geoip hint (country `VN` -> Vietnamese, anything else or unknown -> English);
 * with no hint at all (dev server, a path the Worker does not front) the browser language.
 */

export const LANGS = ["vi", "en"] as const
export type Lang = (typeof LANGS)[number]

export const LANG_STORAGE_KEY = "gidi:lang"
/** `<meta name="…" content="vi|en">` the Worker injects into the page; never a cookie. */
export const LANG_HINT_META = "gidi-lang-hint"

export const isLang = (value: unknown): value is Lang =>
  typeof value === "string" && (LANGS as readonly string[]).includes(value)

/** ISO 3166-1 alpha-2 country (any case) of the request; `null`/unknown/other -> English. */
export const languageForCountry = (country: string | null | undefined): Lang =>
  country?.trim().toUpperCase() === "VN" ? "vi" : "en"

export interface LanguageInputs {
  /** The visitor's saved choice, as read from storage (unvalidated). */
  stored?: string | null
  /** The Worker's hint, as read from the meta tag (unvalidated). */
  hint?: string | null
  /** `navigator.language`. */
  browser?: string | null
}

export function decideLanguage({
  stored,
  hint,
  browser,
}: LanguageInputs): Lang {
  if (isLang(stored)) return stored
  if (isLang(hint)) return hint
  return browser?.toLowerCase().startsWith("vi") ? "vi" : "en"
}
