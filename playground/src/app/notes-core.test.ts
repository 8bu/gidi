import assert from "node:assert/strict"
import { test } from "node:test"

import { IDBDatabase as FDBDatabase, IDBFactory } from "fake-indexeddb"

import { createNotesStore } from "./notes-core.ts"
import {
  dbName,
  LEGACY_DB_NAME,
  openNotesDb,
  openVietnamese,
  readAllNotes,
  readMeta,
  writeNotes,
} from "./notes-db.ts"
import type { NoteRecord } from "./types.ts"

const messages = () => ({
  notJson: "not json",
  notGidiExport: "not export",
  unsupportedVersion: "version",
  invalidRecord: (i: number) => `bad ${i}`,
})

function note(
  id: string,
  text: string,
  currency: "USD" | "VND",
  amount = 10
): NoteRecord {
  return {
    id,
    text,
    createdAt: Number(id.replace(/\D/g, "")) || 1,
    updatedAt: 1,
    type: "expense",
    target: null,
    amount,
    currency,
    original: {
      type: "expense",
      typeConfidence: 0.9,
      target: null,
      targetSpan: null,
      targetConfidence: 0.9,
      valueText: null,
      valueSpan: null,
      amount,
      modelVersion: currency === "VND" ? "gidi-finance-v4" : "gidi-en-v2",
    },
  }
}

/** A pre-language database: `gidi-notes`, `notes` store, records without `currency`. */
async function seedLegacy(
  factory: IDBFactory,
  records: unknown[]
): Promise<void> {
  const { promise, resolve, reject } = Promise.withResolvers<IDBDatabase>()
  const request = factory.open(LEGACY_DB_NAME, 1)
  request.onupgradeneeded = () =>
    request.result.createObjectStore("notes", { keyPath: "id" })
  request.onsuccess = () => resolve(request.result)
  request.onerror = () => reject(request.error)
  const db = await promise
  await writeNotes(db, records, null)
  db.close()
}

const legacyRecords = () =>
  ["a1", "a2", "a3"].map((id, i) => {
    const legacy: Record<string, unknown> = {
      ...note(id, `phở ${i}`, "VND", 45_000),
    }
    delete legacy.currency
    return { ...legacy, extra: "kept" }
  })

async function raw(
  factory: IDBFactory,
  name: string
): Promise<Record<string, unknown>[]> {
  const db = await openNotesDb(factory, name)
  const rows = (await readAllNotes(db)) as Record<string, unknown>[]
  db.close()
  return rows.sort((a, b) => String(a.id).localeCompare(String(b.id)))
}

test("an existing gidi-notes store is copied to gidi-vi with VND and left untouched", async () => {
  const factory = new IDBFactory()
  const legacy = legacyRecords()
  await seedLegacy(factory, legacy)

  const { db, fromLegacy } = await openVietnamese(factory)
  assert.equal(fromLegacy, false)
  assert.equal(db.name, dbName("vi"))
  const migrated = (await readAllNotes(db)) as Record<string, unknown>[]
  assert.equal(migrated.length, 3)
  for (const row of migrated) {
    assert.equal(row.currency, "VND")
    assert.equal(row.extra, "kept")
  }
  assert.deepEqual(await readMeta(db, "migrated"), {
    from: LEGACY_DB_NAME,
    count: 3,
  })
  db.close()

  // The legacy database is the rollback: exactly what it was (no currency added, not upgraded).
  const afterLegacy = await raw(factory, LEGACY_DB_NAME)
  assert.deepEqual(afterLegacy, legacy)
})

test("the migration runs once: later edits and deletes in gidi-vi are not undone", async () => {
  const factory = new IDBFactory()
  await seedLegacy(factory, legacyRecords())
  const first = await openVietnamese(factory)
  await writeNotes(first.db, [], "a1")
  first.db.close()

  const second = await openVietnamese(factory)
  const ids = ((await readAllNotes(second.db)) as { id: string }[])
    .map((r) => r.id)
    .sort()
  assert.deepEqual(ids, ["a2", "a3"])
  second.db.close()
})

test("without a legacy store gidi-vi simply starts empty and no gidi-notes is created", async () => {
  const factory = new IDBFactory()
  const { db } = await openVietnamese(factory)
  assert.deepEqual(await readAllNotes(db), [])
  db.close()
  const names = (await factory.databases()).map((d) => d.name)
  assert.deepEqual(names, ["gidi-vi"])
})

test("a failed copy serves the legacy notes, marks nothing, and retries next start", async () => {
  const factory = new IDBFactory()
  await seedLegacy(factory, legacyRecords())

  const original = FDBDatabase.prototype.transaction
  FDBDatabase.prototype.transaction = function (
    this: InstanceType<typeof FDBDatabase>,
    ...args: Parameters<typeof original>
  ) {
    if (
      this.name === "gidi-vi" &&
      Array.isArray(args[0]) &&
      args[0].length === 2
    ) {
      throw new Error("disk full")
    }
    return original.apply(this, args)
  } as typeof original
  const warn = console.warn
  console.warn = () => undefined
  try {
    const failed = await openVietnamese(factory)
    assert.equal(failed.fromLegacy, true)
    assert.equal(failed.db.name, LEGACY_DB_NAME)
    assert.equal(((await readAllNotes(failed.db)) as unknown[]).length, 3)
    failed.db.close()
  } finally {
    FDBDatabase.prototype.transaction = original
    console.warn = warn
  }

  const retried = await openVietnamese(factory)
  assert.equal(retried.fromLegacy, false)
  assert.equal(((await readAllNotes(retried.db)) as unknown[]).length, 3)
  retried.db.close()
})

test("the two languages are separate databases; switching keeps both datasets", async () => {
  const factory = new IDBFactory()
  await seedLegacy(factory, legacyRecords())
  const store = createNotesStore({ factory, lang: "vi", messages })
  store.subscribe(() => undefined)
  await store.loaded()
  assert.equal(store.getSnapshot().notes.length, 3)

  await store.setLanguage("en")
  assert.equal(store.getSnapshot().lang, "en")
  assert.deepEqual(store.getSnapshot().notes, [], "English starts empty")
  await store.add(note("e1", "venmo'd mom 50", "USD", 50))
  assert.deepEqual(
    store.getSnapshot().notes.map((n) => n.id),
    ["e1"]
  )
  assert.equal(store.getSnapshot().summaries.expense.total, 50)

  await store.setLanguage("vi")
  assert.equal(store.getSnapshot().notes.length, 3)
  assert.ok(store.getSnapshot().notes.every((n) => n.currency === "VND"))
  assert.equal(store.getSnapshot().summaries.expense.total, 135_000)

  // Reload: a new store over the same databases sees both datasets, still apart.
  const again = createNotesStore({ factory, lang: "en", messages })
  again.subscribe(() => undefined)
  await again.loaded()
  assert.deepEqual(
    again.getSnapshot().notes.map((n) => n.id),
    ["e1"]
  )
  await again.setLanguage("vi")
  assert.equal(again.getSnapshot().notes.length, 3)

  assert.deepEqual(
    (await raw(factory, "gidi-en")).map((r) => r.id),
    ["e1"]
  )
  assert.deepEqual(
    (await raw(factory, "gidi-vi")).map((r) => r.id),
    ["a1", "a2", "a3"]
  )
})

test("edit and delete act on the active database only", async () => {
  const factory = new IDBFactory()
  const store = createNotesStore({ factory, lang: "vi", messages })
  store.subscribe(() => undefined)
  await store.add(note("x1", "cà phê", "VND", 30_000))
  await store.setLanguage("en")
  await store.add(note("x1", "coffee", "USD", 4))

  await store.update("x1", { amount: 5 })
  await store.remove("x1")
  assert.deepEqual(store.getSnapshot().notes, [])

  await store.setLanguage("vi")
  assert.deepEqual(
    store.getSnapshot().notes.map((n) => [n.id, n.amount]),
    [["x1", 30_000]]
  )
  assert.deepEqual(await raw(factory, "gidi-en"), [])
})

test("export covers the active language only; import joins the active database in its currency", async () => {
  const factory = new IDBFactory()
  const store = createNotesStore({ factory, lang: "vi", messages })
  store.subscribe(() => undefined)
  await store.add(note("v1", "phở", "VND", 45_000))
  await store.setLanguage("en")
  await store.add(note("e1", "pizza", "USD", 12))

  const exported = JSON.parse(store.exportJson()) as { notes: NoteRecord[] }
  assert.deepEqual(
    exported.notes.map((n) => n.id),
    ["e1"]
  )

  await store.setLanguage("vi")
  const viExport = store.exportJson()
  await store.setLanguage("en")
  assert.equal(await store.importJson(viExport), 1)
  const english = store.getSnapshot().notes
  assert.deepEqual(english.map((n) => n.id).sort(), ["e1", "v1"])
  assert.ok(english.every((n) => n.currency === "USD"))
  await store.setLanguage("vi")
  assert.deepEqual(
    store.getSnapshot().notes.map((n) => n.id),
    ["v1"]
  )
  assert.equal(store.getSnapshot().notes[0].currency, "VND")
})

test("a note that finishes after a language switch lands in its own database", async () => {
  const factory = new IDBFactory()
  const store = createNotesStore({ factory, lang: "vi", messages })
  store.subscribe(() => undefined)
  await store.setLanguage("en")
  await store.add(note("late1", "bủn xỉn", "VND", 1000))
  assert.deepEqual(
    store.getSnapshot().notes,
    [],
    "the English view does not show it"
  )
  await store.setLanguage("vi")
  assert.deepEqual(
    store.getSnapshot().notes.map((n) => n.id),
    ["late1"]
  )
})

test("each database records the model it last ran", async () => {
  const factory = new IDBFactory()
  const store = createNotesStore({ factory, lang: "en", messages })
  store.subscribe(() => undefined)
  await store.loaded()
  await store.setLanguage("vi")
  for (const [lang, model] of [
    ["en", "gidi-en-v2 2.0.1"],
    ["vi", "gidi-finance-v4 4.0.0"],
  ] as const) {
    const db = await openNotesDb(factory, dbName(lang))
    assert.equal(await readMeta(db, "model"), model)
    db.close()
  }
})

test("a bad import changes nothing and the file errors are reported", async () => {
  const store = createNotesStore({
    factory: new IDBFactory(),
    lang: "en",
    messages,
  })
  store.subscribe(() => undefined)
  await assert.rejects(store.importJson("{"), /not json/)
  await assert.rejects(store.importJson('{"app":"x","notes":[]}'), /not export/)
  await assert.rejects(store.importJson('[{"id":"a"}]'), /bad 1/)
  assert.deepEqual(store.getSnapshot().notes, [])
})
