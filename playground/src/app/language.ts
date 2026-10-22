import { useMemo, useSyncExternalStore } from "react"

import { track } from "@/lib/analytics"
import {
  decideLanguage,
  LANG_HINT_META,
  LANG_STORAGE_KEY,
  type Lang,
} from "@/lib/language"
import { MODELS, type ModelSpec } from "@/lib/models"

import { formatAmount, formatAmountCompact } from "./amount"
import { STRINGS, type Strings } from "./i18n"
import type { Currency, NoteRecord } from "./types"

/**
 * The language of the app and so which model runs. The visitor's saved choice (`gidi:lang`)
 * always beats the Worker's geoip hint; see `@/lib/language`. Module-level state, like the model
 * in `use-gidi.ts`, so every component and the model loader read the same answer.
 */

function readStored(): string | null {
  try {
    return window.localStorage.getItem(LANG_STORAGE_KEY)
  } catch {
    return null
  }
}

/** `<html lang>`, the tab title and the description follow the language. */
function applyDocument(lang: Lang): void {
  const strings = STRINGS[lang]
  document.documentElement.lang = lang
  document.title = strings.docTitle
  document
    .querySelector('meta[name="description"]')
    ?.setAttribute("content", strings.docDescription)
}

let current: Lang = decideLanguage({
  stored: readStored(),
  hint: document.querySelector<HTMLMetaElement>(
    `meta[name="${LANG_HINT_META}"]`
  )?.content,
  browser: navigator.language,
})
applyDocument(current)
const listeners = new Set<() => void>()

export const getLanguage = (): Lang => current
export const getStrings = (): Strings => STRINGS[current]

/** The visitor chose `lang`: remember it (it beats geoip from now on) and tell everyone. */
export function setLanguage(lang: Lang): void {
  if (lang === current) return
  try {
    window.localStorage.setItem(LANG_STORAGE_KEY, lang)
  } catch {
    // Private mode: the choice lasts for this page only.
  }
  current = lang
  applyDocument(lang)
  track("language_changed", { to: lang })
  for (const listener of listeners) listener()
}

export function subscribe(listener: () => void): () => void {
  listeners.add(listener)
  return () => {
    listeners.delete(listener)
  }
}

/** The currency of a language's database. */
export const currencyOf = (lang: Lang): Currency =>
  lang === "vi" ? "VND" : "USD"

export interface I18n {
  lang: Lang
  t: Strings
  /** `Intl` locale of the language. */
  locale: string
  model: ModelSpec
  /** `1.200.000 ₫` / `$1,200.00`: an amount in the active language's currency. */
  money(n: number): string
  moneyCompact(n: number): string
  /** A record's amount in its own currency. */
  recordMoney(record: NoteRecord, amount?: number | null): string
  recordMoneyCompact(record: NoteRecord): string
}

export function useI18n(): I18n {
  const lang = useSyncExternalStore(subscribe, getLanguage, getLanguage)
  return useMemo(() => {
    const currency = currencyOf(lang)
    return {
      lang,
      t: STRINGS[lang],
      locale: STRINGS[lang].locale,
      model: MODELS[lang],
      money: (n) => formatAmount(n, currency),
      moneyCompact: (n) => formatAmountCompact(n, currency),
      recordMoney: (record, amount = record.amount) =>
        amount === null ? "" : formatAmount(amount, record.currency),
      recordMoneyCompact: (record) =>
        record.amount === null
          ? ""
          : formatAmountCompact(record.amount, record.currency),
    }
  }, [lang])
}

export function useLanguage(): Lang {
  return useSyncExternalStore(subscribe, getLanguage, getLanguage)
}
