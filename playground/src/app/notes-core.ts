/**
 * The notes of the app, one dataset per language (`gidi-vi`, `gidi-en`; see `notes-db.ts`).
 * Only the active language's notes are in the snapshot; the other dataset is never read into it,
 * copied, exported or changed. No DOM or React here, so the tests drive it with `fake-indexeddb`.
 */

import { reportError } from "../lib/error-report.ts"
import type { Lang } from "../lib/language.ts"
import { MODELS } from "../lib/models.ts"
import {
  openForLanguage,
  readAllNotes,
  writeMeta,
  writeNotes,
  type Opened,
} from "./notes-db.ts"
import {
  TX_TYPES,
  type Currency,
  type ModelReading,
  type NotePatch,
  type NoteRecord,
  type StackSummary,
  type TxType,
} from "./types.ts"

const EXPORT_APP = "gidi-notes"
const EXPORT_VERSION = 1
const LATEST_PER_STACK = 3

export interface Snapshot {
  lang: Lang
  notes: NoteRecord[]
  /** The active language's database has been read. */
  ready: boolean
  summaries: Record<TxType, StackSummary>
}

export interface ImportMessages {
  notJson: string
  notGidiExport: string
  unsupportedVersion: string
  invalidRecord(index: number): string
}

export interface NotesStore {
  getSnapshot(): Snapshot
  subscribe(listener: () => void): () => void
  /** Make `lang` the active dataset (loading it on first use). */
  setLanguage(lang: Lang): Promise<void>
  add(record: NoteRecord): Promise<void>
  update(id: string, patch: NotePatch): Promise<void>
  remove(id: string): Promise<void>
  exportJson(): string
  importJson(json: string): Promise<number>
  /** Resolves once the active language is loaded. */
  loaded(): Promise<void>
}

const currencyOf = (lang: Lang): Currency => (lang === "vi" ? "VND" : "USD")

function newestFirst(a: NoteRecord, b: NoteRecord): number {
  return b.createdAt - a.createdAt || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0)
}

function summarize(notes: NoteRecord[]): Record<TxType, StackSummary> {
  const summaries = Object.fromEntries(
    TX_TYPES.map((type): [TxType, StackSummary] => [
      type,
      { type, count: 0, total: 0, latest: [] },
    ])
  ) as Record<TxType, StackSummary>
  for (const note of notes) {
    const stack = summaries[note.type]
    stack.count += 1
    stack.total += note.amount ?? 0
    if (stack.latest.length < LATEST_PER_STACK) stack.latest.push(note)
  }
  return summaries
}

function snapshotOf(lang: Lang, notes: NoteRecord[], ready: boolean): Snapshot {
  return { lang, notes, ready, summaries: summarize(notes) }
}

// --------------------------------------------------------------------------- validation

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value)
const isFiniteNumber = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value)
const isNullableString = (value: unknown): value is string | null =>
  value === null || typeof value === "string"
const isNullableNumber = (value: unknown): value is number | null =>
  value === null || isFiniteNumber(value)
const isTxType = (value: unknown): value is TxType =>
  typeof value === "string" && (TX_TYPES as readonly string[]).includes(value)
const isNullableSpan = (value: unknown): value is [number, number] | null =>
  value === null ||
  (Array.isArray(value) &&
    value.length === 2 &&
    isFiniteNumber(value[0]) &&
    isFiniteNumber(value[1]))

function isModelReading(value: unknown): value is ModelReading {
  return (
    isRecord(value) &&
    isTxType(value.type) &&
    isFiniteNumber(value.typeConfidence) &&
    isNullableString(value.target) &&
    isNullableSpan(value.targetSpan) &&
    isFiniteNumber(value.targetConfidence) &&
    isNullableString(value.valueText) &&
    isNullableSpan(value.valueSpan) &&
    isNullableNumber(value.amount) &&
    typeof value.modelVersion === "string"
  )
}

/** `currency` is optional in stored and imported records (older ones have none). */
function isNoteRecord(
  value: unknown
): value is Omit<NoteRecord, "currency"> & { currency?: Currency } {
  return (
    isRecord(value) &&
    typeof value.id === "string" &&
    value.id !== "" &&
    typeof value.text === "string" &&
    isFiniteNumber(value.createdAt) &&
    isFiniteNumber(value.updatedAt) &&
    isTxType(value.type) &&
    isNullableString(value.target) &&
    isNullableNumber(value.amount) &&
    (value.currency === undefined ||
      value.currency === "USD" ||
      value.currency === "VND") &&
    isModelReading(value.original)
  )
}

interface Dataset {
  notes: NoteRecord[]
  db: IDBDatabase | null
  loading: Promise<void> | null
  loaded: boolean
}

export function createNotesStore(options: {
  factory: IDBFactory | undefined
  lang: Lang
  messages: () => ImportMessages
}): NotesStore {
  const datasets: Record<Lang, Dataset> = {
    vi: { notes: [], db: null, loading: null, loaded: false },
    en: { notes: [], db: null, loading: null, loaded: false },
  }
  let active = options.lang
  const listeners = new Set<() => void>()
  let snapshot = snapshotOf(active, [], false)

  const publish = (lang: Lang): void => {
    const data = datasets[lang]
    // A change to the inactive dataset never reaches the UI.
    if (lang !== active) return
    snapshot = snapshotOf(lang, [...data.notes].sort(newestFirst), data.loaded)
    for (const listener of listeners) listener()
  }

  /** Open and read `lang`'s database once. Never rejects; falls back to memory. */
  const load = (lang: Lang): Promise<void> => {
    const data = datasets[lang]
    data.loading ??= (async () => {
      let stored: unknown[] = []
      try {
        if (options.factory !== undefined) {
          const opened: Opened = await openForLanguage(options.factory, lang)
          data.db = opened.db
          stored = await readAllNotes(opened.db)
          if (!opened.fromLegacy) {
            const spec = MODELS[lang]
            await writeMeta(opened.db, "model", `${spec.id} ${spec.version}`)
          }
        }
      } catch (cause) {
        reportError(cause, "save", {
          kind: "idb_open",
          recovery: "memory_only",
        })
        console.warn(
          "gidi: IndexedDB unavailable, notes stay in memory.",
          cause
        )
        data.db = null
      }
      data.notes = stored
        .filter(isNoteRecord)
        .map((n) => ({ ...n, currency: n.currency ?? currencyOf(lang) }))
      data.loaded = true
      publish(lang)
    })()
    return data.loading
  }

  /** Best-effort write; the in-memory dataset stays the source of truth. */
  const persist = (
    lang: Lang,
    writes: NoteRecord[],
    deleteId: string | null = null
  ): Promise<void> => {
    const { db } = datasets[lang]
    if (db === null) return Promise.resolve()
    return writeNotes(db, writes, deleteId).catch((cause: unknown) => {
      reportError(cause, "save", { kind: "idb_write", recovery: "memory_only" })
      console.warn("gidi: could not save to IndexedDB, kept in memory.", cause)
    })
  }

  const setNotes = (lang: Lang, notes: NoteRecord[]): void => {
    datasets[lang].notes = notes
    publish(lang)
  }

  return {
    getSnapshot: () => snapshot,
    subscribe(listener) {
      listeners.add(listener)
      void load(active)
      return () => {
        listeners.delete(listener)
      }
    },
    async setLanguage(lang) {
      if (lang === active) return
      active = lang
      publish(lang)
      await load(lang)
    },
    loaded: () => load(active),
    async add(record) {
      // A note belongs to the database of the language that read it, even if the visitor
      // switched language while it was flying into its stack.
      const lang = record.currency === "VND" ? "vi" : "en"
      await load(lang)
      const rest = datasets[lang].notes.filter((n) => n.id !== record.id)
      setNotes(lang, [record, ...rest])
      await persist(lang, [record])
    },
    async update(id, patch) {
      const lang = active
      await load(lang)
      const current = datasets[lang].notes.find((n) => n.id === id)
      if (current === undefined) return
      const next: NoteRecord = { ...current, ...patch, updatedAt: Date.now() }
      setNotes(
        lang,
        datasets[lang].notes.map((n) => (n.id === id ? next : n))
      )
      await persist(lang, [next])
    },
    async remove(id) {
      const lang = active
      await load(lang)
      setNotes(
        lang,
        datasets[lang].notes.filter((n) => n.id !== id)
      )
      await persist(lang, [], id)
    },
    exportJson: () =>
      JSON.stringify(
        { app: EXPORT_APP, version: EXPORT_VERSION, notes: snapshot.notes },
        null,
        2
      ),
    async importJson(json) {
      const messages = options.messages()
      let parsed: unknown
      try {
        parsed = JSON.parse(json)
      } catch {
        throw new Error(messages.notJson)
      }
      const list = Array.isArray(parsed)
        ? parsed
        : isRecord(parsed)
          ? parsed.notes
          : null
      if (
        !Array.isArray(list) ||
        (isRecord(parsed) && parsed.app !== EXPORT_APP)
      ) {
        throw new Error(messages.notGidiExport)
      }
      if (isRecord(parsed) && parsed.version !== EXPORT_VERSION) {
        throw new Error(messages.unsupportedVersion)
      }
      const lang = active
      const incoming: NoteRecord[] = []
      for (const [index, item] of list.entries()) {
        if (!isNoteRecord(item))
          throw new Error(messages.invalidRecord(index + 1))
        // Imported notes join the active language's dataset, in its currency.
        incoming.push({ ...item, currency: currencyOf(lang) })
      }
      await load(lang)
      const byId = new Map(datasets[lang].notes.map((n) => [n.id, n]))
      const written: NoteRecord[] = []
      for (const note of incoming) {
        const existing = byId.get(note.id)
        if (existing === undefined || note.updatedAt > existing.updatedAt) {
          byId.set(note.id, note)
          written.push(note)
        }
      }
      if (written.length > 0) {
        setNotes(lang, [...byId.values()])
        await persist(lang, written)
      }
      return written.length
    },
  }
}
