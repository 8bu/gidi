/** Shared contract of the notes app (`/`). The developer playground lives at `/lab`. */

export const TX_TYPES = [
  "expense",
  "income",
  "borrow",
  "lend",
  "repayment_in",
  "repayment_out",
  "transfer",
  "refund",
] as const

export type TxType = (typeof TX_TYPES)[number]

/** What the model said when the note was sent. Never changed by user edits. */
export interface ModelReading {
  type: TxType
  typeConfidence: number
  target: string | null
  targetSpan: [number, number] | null
  targetConfidence: number
  valueText: string | null
  valueSpan: [number, number] | null
  /** VND parsed from `valueText`; null when there is no amount or it cannot be read. */
  amount: number | null
  modelVersion: string
}

/** One saved note. `type`, `target` and `amount` are the current (maybe user-corrected) values. */
export interface NoteRecord {
  id: string
  text: string
  createdAt: number
  updatedAt: number
  type: TxType
  target: string | null
  amount: number | null
  original: ModelReading
}

export type NotePatch = Partial<Pick<NoteRecord, "type" | "target" | "amount">>

export interface StackSummary {
  type: TxType
  count: number
  /** Sum of `amount` over the stack's notes, VND. */
  total: number
  /** Newest first, at most 3. */
  latest: NoteRecord[]
}
