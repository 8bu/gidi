"use client"

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
  type CSSProperties,
  type MouseEvent,
  type ReactNode,
  type Ref,
} from "react"
import { createPortal } from "react-dom"
import { AnimatePresence, motion } from "motion/react"
import { PencilIcon, RotateCcwIcon, Trash2Icon } from "lucide-react"

import { Button } from "@/components/ui/button"
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet"
import { track } from "@/lib/analytics"
import { cn } from "@/lib/utils"
import { formatVnd, formatVndCompact } from "./amount"
import { useFly } from "./fly-layer"
import {
  RecordEditor,
  TypeSwatch,
  isEdited,
  restorePatch,
} from "./record-editor"
import { useNotes } from "./store"
import { TYPE_META } from "./taxonomy"
import type { ModelReading, NotePatch, NoteRecord, TxType } from "./types"

/** Side panel from here up; below it a bottom drawer (phones and tablet portrait). */
const DESKTOP_QUERY = "(min-width: 1024px)"
const TOAST_MS = 5000
const ROW_SPRING = { type: "spring", stiffness: 520, damping: 42 } as const
const TARGETS_COLLAPSED = 6

function subscribeDesktop(onChange: () => void) {
  const query = window.matchMedia(DESKTOP_QUERY)
  query.addEventListener("change", onChange)
  return () => query.removeEventListener("change", onChange)
}

function useIsDesktop(): boolean {
  return useSyncExternalStore(
    subscribeDesktop,
    () => window.matchMedia(DESKTOP_QUERY).matches,
    () => true
  )
}

/* ----------------------------------------------------------- formatting */

const DAY_TIME = new Intl.DateTimeFormat("vi-VN", {
  day: "2-digit",
  month: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
})

function formatWhen(timestamp: number): string {
  const date = new Date(timestamp)
  const year =
    date.getFullYear() === new Date().getFullYear()
      ? ""
      : `/${date.getFullYear()}`
  const [day, time] = DAY_TIME.format(date).split(", ")
  return time ? `${day}${year}, ${time}` : `${day}${year}`
}

const percent = (confidence: number) => `${Math.round(confidence * 100)}%`

/** Original note with the model's target and value spans marked (offsets are code points). */
function highlight(text: string, original: ModelReading): ReactNode[] {
  const chars = Array.from(text)
  const marks: { start: number; end: number; kind: "target" | "value" }[] = []
  const add = (span: [number, number] | null, kind: "target" | "value") => {
    if (!span) return
    const start = Math.max(0, span[0])
    const end = Math.min(chars.length, span[1])
    if (start < end) marks.push({ start, end, kind })
  }
  add(original.targetSpan, "target")
  add(original.valueSpan, "value")
  marks.sort((a, b) => a.start - b.start)

  const nodes: ReactNode[] = []
  let cursor = 0
  for (const mark of marks) {
    if (mark.start < cursor) continue
    if (mark.start > cursor)
      nodes.push(chars.slice(cursor, mark.start).join(""))
    nodes.push(
      <mark
        key={`${mark.kind}-${mark.start}`}
        className={cn(
          "rounded-sm px-0.5 font-medium",
          mark.kind === "target"
            ? "bg-span-target text-span-target-foreground"
            : "bg-span-value text-span-value-foreground"
        )}
      >
        {chars.slice(mark.start, mark.end).join("")}
      </mark>
    )
    cursor = mark.end
  }
  if (cursor < chars.length) nodes.push(chars.slice(cursor).join(""))
  return nodes
}

/* ------------------------------------------------------------- targets */

interface TargetStat {
  key: string
  name: string
  count: number
  total: number
}

const targetKey = (target: string) => target.trim().toLocaleLowerCase("vi")

function summarizeTargets(records: NoteRecord[]): TargetStat[] {
  const byKey = new Map<string, TargetStat>()
  for (const record of records) {
    if (record.target === null || record.target.trim() === "") continue
    const key = targetKey(record.target)
    const stat = byKey.get(key) ?? {
      key,
      name: record.target.trim(),
      count: 0,
      total: 0,
    }
    stat.count += 1
    stat.total += record.amount ?? 0
    byKey.set(key, stat)
  }
  return [...byKey.values()].sort(
    (a, b) => b.total - a.total || b.count - a.count
  )
}

/* --------------------------------------------------------------- toast */

interface ToastState {
  nonce: number
  message: string
  undo?: () => void
}

function Toast({
  toast,
  onDismiss,
}: {
  toast: ToastState | null
  onDismiss: () => void
}) {
  const nonce = toast?.nonce
  useEffect(() => {
    if (nonce === undefined) return
    const timer = window.setTimeout(onDismiss, TOAST_MS)
    return () => window.clearTimeout(timer)
  }, [nonce, onDismiss])

  return createPortal(
    <div className="pointer-events-none fixed inset-x-0 bottom-[max(1.25rem,env(safe-area-inset-bottom))] z-[70] flex justify-center pr-[max(1rem,env(safe-area-inset-right))] pl-[max(1rem,env(safe-area-inset-left))]">
      <AnimatePresence>
        {toast && (
          <motion.div
            key={toast.nonce}
            data-snackbar
            role="status"
            aria-live="polite"
            initial={{ opacity: 0, y: 24, scale: 0.96 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 12, scale: 0.98 }}
            transition={{ type: "spring", stiffness: 420, damping: 34 }}
            className="pointer-events-auto relative flex items-center gap-3 overflow-hidden rounded-lg bg-foreground py-2 pr-2 pl-4 text-sm text-background shadow-lg"
          >
            <span>{toast.message}</span>
            {toast.undo && (
              <Button
                size="sm"
                variant="secondary"
                className="pointer-coarse:h-11 pointer-coarse:px-4 pointer-coarse:text-sm"
                onClick={() => {
                  toast.undo?.()
                  onDismiss()
                }}
              >
                <RotateCcwIcon data-icon="inline-start" />
                Hoàn tác
              </Button>
            )}
            <motion.span
              aria-hidden
              className="absolute inset-x-0 bottom-0 h-0.5 origin-left bg-background/40"
              initial={{ scaleX: 1 }}
              animate={{ scaleX: 0 }}
              transition={{ duration: TOAST_MS / 1000, ease: "linear" }}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </div>,
    document.body
  )
}

/* ----------------------------------------------------------------- row */

interface RecordRowProps {
  record: NoteRecord
  editing: boolean
  onToggleEdit(): void
  onSave(patch: NotePatch, from: DOMRect | null): void
  onRestore(from: DOMRect | null): void
  onDelete(): void
  /** Supplied by AnimatePresence (popLayout) so it can pin the row while it exits. */
  ref?: Ref<HTMLLIElement>
}

function RecordRow({
  record,
  editing,
  onToggleEdit,
  onSave,
  onRestore,
  onDelete,
  ref,
}: RecordRowProps) {
  const own = useRef<HTMLLIElement | null>(null)
  const setRef = (el: HTMLLIElement | null) => {
    own.current = el
    if (typeof ref === "function") ref(el)
    else if (ref) ref.current = el
  }
  const [showOriginal, setShowOriginal] = useState(false)
  const edited = isEdited(record)
  const { original } = record
  const rect = () => own.current?.getBoundingClientRect() ?? null

  /** Touch fast path: a tap on the card itself opens its editor. */
  const onCardClick = (event: MouseEvent<HTMLLIElement>) => {
    if (editing || !window.matchMedia("(pointer: coarse)").matches) return
    if (
      event.target instanceof Element &&
      event.target.closest("button, a, input, textarea, select, [data-no-open]")
    )
      return
    // Selecting text to copy it is not a tap.
    if (window.getSelection()?.toString()) return
    onToggleEdit()
  }

  return (
    <motion.li
      ref={setRef}
      layout
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.96 }}
      transition={ROW_SPRING}
      onClick={onCardClick}
      className="gidi-sheet px-3.5 py-3"
      style={
        {
          "--sheet-paper": `var(--paper-${record.type})`,
          "--sheet-ink": `var(--ink-${record.type})`,
        } as CSSProperties
      }
    >
      <div className="flex items-start justify-between gap-2">
        <p className="min-w-0 text-[0.95rem] leading-6 break-words whitespace-pre-wrap">
          {highlight(record.text, original)}
        </p>
        <div className="-mt-1 -mr-1.5 flex shrink-0 pointer-coarse:-mt-2.5 pointer-coarse:-mr-2.5">
          <Button
            variant="ghost"
            size="icon-sm"
            aria-label="Sửa ghi chú"
            aria-pressed={editing}
            onClick={onToggleEdit}
            className={cn(
              "pointer-coarse:size-11",
              editing && "bg-foreground/10"
            )}
          >
            <PencilIcon />
          </Button>
          <Button
            variant="ghost"
            size="icon-sm"
            aria-label="Xóa ghi chú"
            onClick={onDelete}
            className="pointer-coarse:size-11"
          >
            <Trash2Icon />
          </Button>
        </div>
      </div>

      <div className="mt-2 flex items-end justify-between gap-3">
        <div className="min-w-0 text-xs">
          <p
            className={cn(
              "truncate text-sm font-medium",
              record.target === null && "font-normal italic opacity-60"
            )}
          >
            {record.target ?? "Chưa rõ đối tượng"}
          </p>
          <p className="mt-0.5 flex items-center gap-1.5 opacity-70">
            <time dateTime={new Date(record.createdAt).toISOString()}>
              {formatWhen(record.createdAt)}
            </time>
          </p>
        </div>
        <div className="flex shrink-0 flex-col items-end gap-1">
          {edited && (
            <button
              type="button"
              onClick={() => setShowOriginal((v) => !v)}
              aria-expanded={showOriginal}
              title="Xem mô hình đã đọc gì"
              className="rounded-full border border-current/30 px-2 text-[0.7rem] leading-5 font-medium outline-none hover:bg-foreground/5 focus-visible:ring-2 focus-visible:ring-current pointer-coarse:min-h-8 pointer-coarse:px-3.5 pointer-coarse:text-xs"
            >
              đã sửa
            </button>
          )}
          <p className="text-lg leading-6 font-semibold tabular-nums">
            {record.amount === null ? (
              <span className="text-sm font-normal italic opacity-60">
                Chưa có số tiền
              </span>
            ) : (
              formatVnd(record.amount)
            )}
          </p>
        </div>
      </div>

      <AnimatePresence initial={false}>
        {edited && showOriginal && (
          <motion.div
            key="original"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
            className="overflow-hidden"
          >
            <div
              data-no-open
              className="mt-3 rounded-md bg-background/55 p-3 text-xs text-foreground"
            >
              <p className="font-medium">Mô hình đã đọc ban đầu</p>
              <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1">
                <dt className="text-muted-foreground">Loại</dt>
                <dd>
                  {TYPE_META[original.type].label}{" "}
                  <span className="text-muted-foreground tabular-nums">
                    ({percent(original.typeConfidence)})
                  </span>
                </dd>
                <dt className="text-muted-foreground">Đối tượng</dt>
                <dd>
                  {original.target ?? "không có"}{" "}
                  <span className="text-muted-foreground tabular-nums">
                    ({percent(original.targetConfidence)})
                  </span>
                </dd>
                <dt className="text-muted-foreground">Số tiền</dt>
                <dd className="tabular-nums">
                  {original.valueText ?? "không có"}
                  {original.amount !== null &&
                    ` = ${formatVnd(original.amount)}`}
                </dd>
              </dl>
              <Button
                variant="outline"
                size="sm"
                className="mt-3 pointer-coarse:h-11 pointer-coarse:px-3.5"
                onClick={() => onRestore(rect())}
              >
                <RotateCcwIcon data-icon="inline-start" />
                Khôi phục
              </Button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {editing && (
        <div data-no-open className="text-foreground">
          <RecordEditor
            record={record}
            onSave={(patch) => onSave(patch, rect())}
            onCancel={onToggleEdit}
            onRestore={() => onRestore(rect())}
          />
        </div>
      )}
    </motion.li>
  )
}

/* --------------------------------------------------------------- sheet */

export interface StackSheetProps {
  /** Type of the opened stack; `null` while closed. */
  type: TxType | null
  onClose(): void
}

export function StackSheet({ type, onClose }: StackSheetProps) {
  // Keep the last opened stack so the sheet can slide out with its content intact.
  const [shown, setShown] = useState<TxType>("expense")
  if (type !== null && type !== shown) setShown(type)

  const desktop = useIsDesktop()
  const open = type !== null
  const [editingId, setEditingId] = useState<string | null>(null)
  const [toast, setToast] = useState<ToastState | null>(null)
  const dismissToast = useCallback(() => setToast(null), [])
  const toastNonce = useRef(0)

  const showToast = (message: string, undo?: () => void) => {
    toastNonce.current += 1
    setToast({ nonce: toastNonce.current, message, undo })
  }

  const close = () => {
    setEditingId(null)
    onClose()
  }

  return (
    <>
      <Sheet open={open} onOpenChange={(next) => (next ? undefined : close())}>
        <SheetContent
          side={desktop ? "right" : "bottom"}
          showCloseButton
          className={desktop ? undefined : "h-[85dvh] sm:h-[min(85dvh,46rem)]"}
          onEscapeKeyDown={(event) => {
            if (editingId !== null) {
              event.preventDefault()
              setEditingId(null)
            }
          }}
          onInteractOutside={(event) => {
            const target = event.target
            if (target instanceof Element && target.closest("[data-snackbar]"))
              event.preventDefault()
          }}
        >
          <SheetBody
            key={shown}
            type={shown}
            editingId={editingId}
            setEditingId={setEditingId}
            showToast={showToast}
          />
        </SheetContent>
      </Sheet>
      <Toast toast={toast} onDismiss={dismissToast} />
    </>
  )
}

interface SheetBodyProps {
  type: TxType
  editingId: string | null
  setEditingId(id: string | null): void
  showToast(message: string, undo?: () => void): void
}

function SheetBody({
  type,
  editingId,
  setEditingId,
  showToast,
}: SheetBodyProps) {
  const { notes, add, update, remove } = useNotes()
  const fly = useFly()
  const [targetFilter, setTargetFilter] = useState<string | null>(null)
  const [allTargets, setAllTargets] = useState(false)
  const [flying, setFlying] = useState<ReadonlySet<string>>(new Set())

  const ofStack = useMemo(
    () => notes.filter((n) => n.type === type && !flying.has(n.id)),
    [notes, type, flying]
  )
  const targets = useMemo(() => summarizeTargets(ofStack), [ofStack])
  const activeFilter =
    targetFilter !== null && targets.some((t) => t.key === targetFilter)
      ? targetFilter
      : null
  const visible = useMemo(
    () =>
      activeFilter === null
        ? ofStack
        : ofStack.filter(
            (n) => n.target !== null && targetKey(n.target) === activeFilter
          ),
    [ofStack, activeFilter]
  )

  const total = ofStack.reduce((sum, n) => sum + (n.amount ?? 0), 0)
  const shownTargets = allTargets
    ? targets
    : targets.slice(0, TARGETS_COLLAPSED)

  /** Apply a patch; a changed type flies the record into its new stack. */
  const apply = (
    record: NoteRecord,
    patch: NotePatch,
    from: DOMRect | null
  ) => {
    setEditingId(null)
    const retyped = patch.type !== undefined && patch.type !== record.type
    track("note_edited", {
      from_type: record.type,
      to_type: patch.type ?? record.type,
      type_changed: retyped,
      target_changed:
        patch.target !== undefined && patch.target !== record.target,
      amount_changed:
        patch.amount !== undefined && patch.amount !== record.amount,
    })
    const run = () =>
      update(record.id, patch).catch((e: unknown) =>
        showToast(e instanceof Error ? e.message : "Không lưu được chỉnh sửa.")
      )
    if (!retyped) {
      void run()
      return
    }
    const landing: NoteRecord = { ...record, ...patch }
    const origin =
      from ??
      new DOMRect(
        window.innerWidth / 2 - 140,
        window.innerHeight / 2 - 60,
        280,
        120
      )
    setFlying((current) => new Set(current).add(record.id))
    fly(landing, origin, {
      save: async () => {
        try {
          await run()
        } finally {
          setFlying((current) => {
            const next = new Set(current)
            next.delete(record.id)
            return next
          })
        }
      },
    })
  }

  const handleDelete = (record: NoteRecord) => {
    setEditingId(null)
    track("note_deleted", { type: record.type })
    remove(record.id).then(
      () => showToast("Đã xóa ghi chú", () => void add(record)),
      (e: unknown) =>
        showToast(e instanceof Error ? e.message : "Không xóa được ghi chú.")
    )
  }

  const body = (
    <>
      {targets.length > 0 && (
        <section aria-label="Đối tượng" className="px-4 pt-4">
          <div className="flex items-baseline justify-between">
            <h3 className="text-xs font-medium text-muted-foreground">
              Đối tượng
            </h3>
            {activeFilter !== null && (
              <button
                type="button"
                onClick={() => setTargetFilter(null)}
                className="rounded-sm text-xs text-muted-foreground underline underline-offset-2 outline-none hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring pointer-coarse:min-h-10 pointer-coarse:px-2"
              >
                Bỏ lọc
              </button>
            )}
          </div>
          <motion.ul layout className="mt-2 flex flex-wrap gap-1.5">
            {shownTargets.map((t) => {
              const active = activeFilter === t.key
              return (
                <motion.li layout key={t.key}>
                  <button
                    type="button"
                    aria-pressed={active}
                    onClick={() => setTargetFilter(active ? null : t.key)}
                    className={cn(
                      "flex max-w-full items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs transition-colors outline-none focus-visible:ring-3 focus-visible:ring-ring/50 pointer-coarse:min-h-10 pointer-coarse:px-3.5",
                      active
                        ? "border-foreground bg-foreground text-background"
                        : "border-border bg-background hover:bg-muted"
                    )}
                  >
                    <span className="max-w-40 truncate font-medium">
                      {t.name}
                    </span>
                    <span
                      className={cn(
                        "tabular-nums",
                        active ? "opacity-80" : "text-muted-foreground"
                      )}
                    >
                      {t.count} · {formatVndCompact(t.total)}
                    </span>
                  </button>
                </motion.li>
              )
            })}
            {targets.length > TARGETS_COLLAPSED && (
              <motion.li layout>
                <button
                  type="button"
                  onClick={() => setAllTargets((v) => !v)}
                  className="rounded-full px-2.5 py-1 text-xs text-muted-foreground underline underline-offset-2 outline-none hover:text-foreground focus-visible:ring-3 focus-visible:ring-ring/50 pointer-coarse:min-h-10 pointer-coarse:px-3.5"
                >
                  {allTargets
                    ? "Thu gọn"
                    : `Xem thêm ${targets.length - TARGETS_COLLAPSED}`}
                </button>
              </motion.li>
            )}
          </motion.ul>
        </section>
      )}

      <section aria-label="Ghi chú" className="px-4 py-4">
        {visible.length === 0 ? (
          <p className="rounded-lg border border-dashed px-4 py-8 text-center text-sm text-muted-foreground">
            {ofStack.length === 0
              ? "Chồng này đang trống. Viết một ghi chú ở giữa màn hình để bắt đầu."
              : "Không có ghi chú nào khớp bộ lọc."}
          </p>
        ) : (
          <ul className="relative flex flex-col gap-3">
            <AnimatePresence initial={false} mode="popLayout">
              {visible.map((record) => (
                <RecordRow
                  key={record.id}
                  record={record}
                  editing={editingId === record.id}
                  onToggleEdit={() =>
                    setEditingId(editingId === record.id ? null : record.id)
                  }
                  onSave={(patch, from) => apply(record, patch, from)}
                  onRestore={(from) =>
                    apply(record, restorePatch(record), from)
                  }
                  onDelete={() => handleDelete(record)}
                />
              ))}
            </AnimatePresence>
          </ul>
        )}
      </section>
    </>
  )

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <SheetHeader
        className="gap-3 border-b px-4 pt-4 pb-3 lg:pt-5"
        style={{
          background: `color-mix(in oklab, var(--paper-${type}) 55%, var(--background))`,
        }}
      >
        <div className="flex items-center gap-2 pr-10 pointer-coarse:pr-14">
          <TypeSwatch type={type} className="size-3.5" />
          <SheetTitle className="text-lg">{TYPE_META[type].label}</SheetTitle>
        </div>
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <p
            className="text-3xl leading-none font-semibold tracking-tight tabular-nums"
            style={{ color: `var(--ink-${type})` }}
          >
            {formatVnd(total)}
          </p>
          <p className="text-sm text-muted-foreground tabular-nums">
            {ofStack.length} ghi chú
          </p>
        </div>
        <SheetDescription className="text-xs">
          {TYPE_META[type].hint}
        </SheetDescription>
      </SheetHeader>
      <div className="min-h-0 flex-1 touch-pan-y overflow-y-auto overscroll-contain [-webkit-overflow-scrolling:touch]">
        {body}
      </div>
    </div>
  )
}
