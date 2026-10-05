import { useMemo, useSyncExternalStore } from "react"

import {
  TX_TYPES,
  type ModelReading,
  type NotePatch,
  type NoteRecord,
  type StackSummary,
  type TxType,
} from "./types"

const DB_NAME = "gidi-notes"
const STORE_NAME = "notes"
const EXPORT_APP = "gidi-notes"
const EXPORT_VERSION = 1
const LATEST_PER_STACK = 3

interface Snapshot {
  notes: NoteRecord[]
  ready: boolean
  summaries: Record<TxType, StackSummary>
}

export interface NotesApi extends Snapshot {
  add(record: NoteRecord): Promise<void>
  update(id: string, patch: NotePatch): Promise<void>
  remove(id: string): Promise<void>
  exportJson(): string
  importJson(json: string): Promise<number>
}

// --------------------------------------------------------------------------- persistence

/** Open the database; `null` when IndexedDB is missing or refuses (private mode, blocked). */
function openDatabase(): Promise<IDBDatabase | null> {
  return new Promise((resolve) => {
    if (typeof indexedDB === "undefined") {
      resolve(null)
      return
    }
    try {
      const request = indexedDB.open(DB_NAME, 1)
      request.onupgradeneeded = () => {
        request.result.createObjectStore(STORE_NAME, { keyPath: "id" })
      }
      request.onsuccess = () => resolve(request.result)
      request.onerror = () => resolve(null)
      request.onblocked = () => resolve(null)
    } catch {
      resolve(null)
    }
  })
}

function readAllNotes(db: IDBDatabase): Promise<unknown[]> {
  return new Promise((resolve, reject) => {
    const request = db
      .transaction(STORE_NAME, "readonly")
      .objectStore(STORE_NAME)
      .getAll()
    request.onsuccess = () => resolve(request.result as unknown[])
    request.onerror = () => reject(request.error)
  })
}

function writeNotes(
  db: IDBDatabase,
  writes: NoteRecord[],
  deleteId: string | null
): Promise<void> {
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite")
    const store = tx.objectStore(STORE_NAME)
    for (const note of writes) store.put(note)
    if (deleteId !== null) store.delete(deleteId)
    tx.oncomplete = () => resolve()
    tx.onerror = () => reject(tx.error)
    tx.onabort = () => reject(tx.error)
  })
}

// --------------------------------------------------------------------------- state

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

function snapshotOf(notes: NoteRecord[], ready: boolean): Snapshot {
  return { notes, ready, summaries: summarize(notes) }
}

let snapshot: Snapshot = snapshotOf([], false)
let database: IDBDatabase | null = null
let loading: Promise<void> | null = null
const listeners = new Set<() => void>()

function publish(notes: NoteRecord[], ready: boolean): void {
  snapshot = snapshotOf([...notes].sort(newestFirst), ready)
  for (const listener of listeners) listener()
}

/** Load once: open the database and read every note. Never rejects; falls back to memory. */
function ensureLoaded(): Promise<void> {
  loading ??= (async () => {
    let stored: unknown[] = []
    try {
      database = await openDatabase()
      if (database !== null) stored = await readAllNotes(database)
    } catch (cause) {
      console.warn("gidi: IndexedDB unavailable, notes stay in memory.", cause)
      database = null
    }
    publish(stored.filter(isNoteRecord), true)
  })()
  return loading
}

/** Best-effort write; the in-memory snapshot stays the source of truth. */
function persist(
  writes: NoteRecord[],
  deleteId: string | null = null
): Promise<void> {
  if (database === null) return Promise.resolve()
  return writeNotes(database, writes, deleteId).catch((cause: unknown) => {
    console.warn("gidi: could not save to IndexedDB, kept in memory.", cause)
  })
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener)
  void ensureLoaded()
  return () => {
    listeners.delete(listener)
  }
}

const getSnapshot = (): Snapshot => snapshot

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

function isNoteRecord(value: unknown): value is NoteRecord {
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
    isModelReading(value.original)
  )
}

// --------------------------------------------------------------------------- actions

async function add(record: NoteRecord): Promise<void> {
  await ensureLoaded()
  const rest = snapshot.notes.filter((n) => n.id !== record.id)
  publish([record, ...rest], true)
  await persist([record])
}

async function update(id: string, patch: NotePatch): Promise<void> {
  await ensureLoaded()
  const current = snapshot.notes.find((n) => n.id === id)
  if (current === undefined) return
  const next: NoteRecord = { ...current, ...patch, updatedAt: Date.now() }
  publish(
    snapshot.notes.map((n) => (n.id === id ? next : n)),
    true
  )
  await persist([next])
}

async function remove(id: string): Promise<void> {
  await ensureLoaded()
  publish(
    snapshot.notes.filter((n) => n.id !== id),
    true
  )
  await persist([], id)
}

function exportJson(): string {
  return JSON.stringify(
    { app: EXPORT_APP, version: EXPORT_VERSION, notes: snapshot.notes },
    null,
    2
  )
}

/**
 * Merge an exported file by id: a note is written when it is new or newer (`updatedAt`) than the
 * stored one. Returns how many notes were written. Throws a Vietnamese message on a bad file;
 * nothing is imported then.
 */
async function importJson(json: string): Promise<number> {
  let parsed: unknown
  try {
    parsed = JSON.parse(json)
  } catch {
    throw new Error("Tệp không phải JSON hợp lệ.")
  }
  const list = Array.isArray(parsed)
    ? parsed
    : isRecord(parsed)
      ? parsed.notes
      : null
  if (!Array.isArray(list) || (isRecord(parsed) && parsed.app !== EXPORT_APP)) {
    throw new Error("Tệp này không phải bản xuất của Gidi.")
  }
  if (isRecord(parsed) && parsed.version !== EXPORT_VERSION) {
    throw new Error("Phiên bản tệp xuất chưa được hỗ trợ.")
  }
  const incoming: NoteRecord[] = []
  for (const [index, item] of list.entries()) {
    if (!isNoteRecord(item)) {
      throw new Error(`Ghi chú thứ ${index + 1} trong tệp không hợp lệ.`)
    }
    incoming.push(item)
  }

  await ensureLoaded()
  const byId = new Map(snapshot.notes.map((n) => [n.id, n]))
  const written: NoteRecord[] = []
  for (const note of incoming) {
    const existing = byId.get(note.id)
    if (existing === undefined || note.updatedAt > existing.updatedAt) {
      byId.set(note.id, note)
      written.push(note)
    }
  }
  if (written.length > 0) {
    publish([...byId.values()], snapshot.ready)
    await persist(written)
  }
  return written.length
}

// --------------------------------------------------------------------------- hook

/** All notes of this device. State is module-level: every caller sees the same notes. */
export function useNotes(): NotesApi {
  const state = useSyncExternalStore(subscribe, getSnapshot, getSnapshot)
  return useMemo(
    () => ({ ...state, add, update, remove, exportJson, importJson }),
    [state]
  )
}
