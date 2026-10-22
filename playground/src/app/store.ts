import { useMemo, useSyncExternalStore } from "react"

import {
  getLanguage,
  getStrings,
  subscribe as onLanguageChange,
} from "./language"
import { createNotesStore, type Snapshot } from "./notes-core"
import type { NotePatch, NoteRecord } from "./types"

export interface NotesApi extends Snapshot {
  add(record: NoteRecord): Promise<void>
  update(id: string, patch: NotePatch): Promise<void>
  remove(id: string): Promise<void>
  exportJson(): string
  importJson(json: string): Promise<number>
}

/** One store for the page: the notes of the active language, see `notes-core.ts`. */
const store = createNotesStore({
  factory: typeof indexedDB === "undefined" ? undefined : indexedDB,
  lang: getLanguage(),
  messages: getStrings,
})

// The visitor switched language: the other language's dataset becomes the active one.
onLanguageChange(() => void store.setLanguage(getLanguage()))

/** The notes of the active language. State is module-level: every caller sees the same notes. */
export function useNotes(): NotesApi {
  const state = useSyncExternalStore(
    store.subscribe,
    store.getSnapshot,
    store.getSnapshot
  )
  return useMemo(
    () => ({
      ...state,
      add: store.add,
      update: store.update,
      remove: store.remove,
      exportJson: store.exportJson,
      importJson: store.importJson,
    }),
    [state]
  )
}
