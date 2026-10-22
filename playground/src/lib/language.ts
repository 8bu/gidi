/**
 * Which language (and so which model) the notes app starts in. Pure and dependency-free;
 * `language.test.ts` runs it under node.
 *
 * Precedence: the visitor's own choice (`gidi:lang` in localStorage) always wins; then the
 * coarse geoip hint (country `VN` -> Vietnamese, anything else -> English); with no hint at all
 * (dev server, or the Cloudflare rule below is missing) the browser language.
 *
 * The hint comes from a zone Response Header Transform Rule on `gidi.8bu.dev` that sets
 * `Server-Timing: geo;desc=<ip.src.country>`; the page reads it from its own navigation entry.
 * No Worker runs for the page (docs/cost-and-traffic.md), and nothing stores or logs the country.
 */

export const LANGS = ["vi", "en"] as const
export type Lang = (typeof LANGS)[number]

export const LANG_STORAGE_KEY = "gidi:lang"
/** Name of the `Server-Timing` metric the Cloudflare rule uses for the visitor's country. */
export const GEO_TIMING_NAME = "geo"

export const isLang = (value: unknown): value is Lang =>
  typeof value === "string" && (LANGS as readonly string[]).includes(value)

/** ISO 3166-1 alpha-2 country (any case) of the request; `null`/unknown/other -> English. */
export const languageForCountry = (country: string | null | undefined): Lang =>
  country?.trim().toUpperCase() === "VN" ? "vi" : "en"

/** The hint carried by the page's own `Server-Timing` entries; `null` when the rule did not run. */
export function hintFromServerTiming(
  timings: readonly { name: string; description: string }[] | undefined
): Lang | null {
  const geo = timings?.find((timing) => timing.name === GEO_TIMING_NAME)
  return geo === undefined ? null : languageForCountry(geo.description)
}

export interface LanguageInputs {
  /** The visitor's saved choice, as read from storage (unvalidated). */
  stored?: string | null
  /** The geoip hint (unvalidated). */
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
