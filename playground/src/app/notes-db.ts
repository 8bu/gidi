/**
 * IndexedDB persistence of the notes app, one database per language: `gidi-vi` and `gidi-en`.
 * Nothing is shared or copied between them. Each holds the `notes` store and a small `meta`
 * store for settings that belong to that language (the model it last ran, the migration record).
 *
 * Before languages the app used one database, `gidi-notes` (Vietnamese only). `migrateLegacy`
 * copies it into `gidi-vi` once. The legacy database is never modified or deleted, so it is the
 * rollback: if the copy fails, the Vietnamese notes keep being served from it and the copy is
 * retried on the next start.
 *
 * Pure functions over an `IDBFactory`, so the tests run them on `fake-indexeddb`.
 */

import { reportError } from "../lib/error-report.ts"
import type { Lang } from "../lib/language.ts"

export const LEGACY_DB_NAME = "gidi-notes"
export const STORE_NAME = "notes"
export const META_STORE = "meta"
const DB_VERSION = 1

export const dbName = (lang: Lang): string => `gidi-${lang}`

function wrap<T>(request: IDBRequest<T>): Promise<T> {
  const { promise, resolve, reject } = Promise.withResolvers<T>()
  request.onsuccess = () => resolve(request.result)
  request.onerror = () => reject(request.error)
  return promise
}

function done(tx: IDBTransaction): Promise<void> {
  const { promise, resolve, reject } = Promise.withResolvers<void>()
  tx.oncomplete = () => resolve()
  tx.onerror = () => reject(tx.error)
  tx.onabort = () => reject(tx.error)
  return promise
}

/** `indexedDB.open` with the usual handlers; `upgrade` creates the stores of a new database. */
function open(
  factory: IDBFactory,
  name: string,
  upgrade: (db: IDBDatabase) => void
): Promise<IDBDatabase> {
  const { promise, resolve, reject } = Promise.withResolvers<IDBDatabase>()
  const request = factory.open(name, DB_VERSION)
  request.onupgradeneeded = () => upgrade(request.result)
  request.onsuccess = () => resolve(request.result)
  request.onerror = () => reject(request.error)
  request.onblocked = () => reject(new Error(`${name} is blocked`))
  return promise
}

/** Open (creating if needed) a database with the notes and meta stores; rejects if refused. */
export function openNotesDb(
  factory: IDBFactory,
  name: string
): Promise<IDBDatabase> {
  return open(factory, name, (db) => {
    db.createObjectStore(STORE_NAME, { keyPath: "id" })
    db.createObjectStore(META_STORE, { keyPath: "key" })
  })
}

export const readAllNotes = (db: IDBDatabase): Promise<unknown[]> =>
  wrap(db.transaction(STORE_NAME, "readonly").objectStore(STORE_NAME).getAll())

export function writeNotes(
  db: IDBDatabase,
  writes: readonly unknown[],
  deleteId: string | null
): Promise<void> {
  const tx = db.transaction(STORE_NAME, "readwrite")
  const store = tx.objectStore(STORE_NAME)
  for (const note of writes) store.put(note)
  if (deleteId !== null) store.delete(deleteId)
  return done(tx)
}

export async function readMeta(db: IDBDatabase, key: string): Promise<unknown> {
  const row = (await wrap(
    db.transaction(META_STORE, "readonly").objectStore(META_STORE).get(key)
  )) as { key: string; value: unknown } | undefined
  return row?.value
}

export function writeMeta(
  db: IDBDatabase,
  key: string,
  value: unknown
): Promise<void> {
  const tx = db.transaction(META_STORE, "readwrite")
  tx.objectStore(META_STORE).put({ key, value })
  return done(tx)
}

/** The legacy database as it is (version 1, `notes` store only); never upgraded. */
function openLegacy(
  factory: IDBFactory,
  created?: { value: boolean }
): Promise<IDBDatabase> {
  return open(factory, LEGACY_DB_NAME, (db) => {
    if (created) created.value = true
    db.createObjectStore(STORE_NAME, { keyPath: "id" })
  })
}

/** Whether the legacy database exists, without creating it. */
async function legacyExists(factory: IDBFactory): Promise<boolean> {
  if (typeof factory.databases === "function") {
    return (await factory.databases()).some((d) => d.name === LEGACY_DB_NAME)
  }
  // No `databases()`: opening creates an absent database; detect that and remove it again.
  const created = { value: false }
  const db = await openLegacy(factory, created)
  db.close()
  if (created.value) {
    await wrap(
      factory.deleteDatabase(LEGACY_DB_NAME) as unknown as IDBRequest<undefined>
    )
    return false
  }
  return true
}

/** A legacy (Vietnamese) record gets `currency: "VND"`; everything else is kept as stored. */
function withCurrency(record: unknown): unknown {
  if (typeof record !== "object" || record === null || "currency" in record)
    return record
  return { ...record, currency: "VND" }
}

export interface Opened {
  db: IDBDatabase
  /** `true` when the Vietnamese notes are served from the untouched legacy database. */
  fromLegacy: boolean
}

/**
 * Open the Vietnamese database, copying the legacy one into it first when that has not happened.
 * The copy is raw (every stored record, valid or not), one transaction together with the
 * `migrated` marker, and verified by reading it back; any failure leaves `gidi-vi` unmarked
 * and returns the legacy database instead, so no note is ever lost or hidden.
 */
export async function openVietnamese(factory: IDBFactory): Promise<Opened> {
  const target = await openNotesDb(factory, dbName("vi"))
  try {
    if ((await readMeta(target, "migrated")) !== undefined) {
      return { db: target, fromLegacy: false }
    }
    if (!(await legacyExists(factory))) {
      await writeMeta(target, "migrated", { from: null, count: 0 })
      return { db: target, fromLegacy: false }
    }
    const legacy = await openLegacy(factory)
    try {
      const records = await readAllNotes(legacy)
      const tx = target.transaction([STORE_NAME, META_STORE], "readwrite")
      const notes = tx.objectStore(STORE_NAME)
      for (const record of records) notes.put(withCurrency(record))
      tx.objectStore(META_STORE).put({
        key: "migrated",
        value: { from: LEGACY_DB_NAME, count: records.length },
      })
      await done(tx)
      const copied = await readAllNotes(target)
      if (copied.length < records.length) {
        throw new Error("migration verification failed")
      }
    } finally {
      legacy.close()
    }
    return { db: target, fromLegacy: false }
  } catch (cause) {
    reportError(cause, "save", {
      kind: "idb_migration",
      recovery: "serve_legacy_db",
    })
    console.warn(
      "gidi: could not copy the notes to gidi-vi, using gidi-notes.",
      cause
    )
    // The marker may have been written before verification failed: remove it so it retries.
    try {
      const tx = target.transaction(META_STORE, "readwrite")
      tx.objectStore(META_STORE).delete("migrated")
      await done(tx)
    } catch {
      // Ignore: the next start verifies again anyway.
    }
    target.close()
    return { db: await openLegacy(factory), fromLegacy: true }
  }
}

/** The database of `lang`; Vietnamese goes through the migration. */
export async function openForLanguage(
  factory: IDBFactory,
  lang: Lang
): Promise<Opened> {
  if (lang === "vi") return openVietnamese(factory)
  return { db: await openNotesDb(factory, dbName(lang)), fromLegacy: false }
}
