import { useEffect, useState, type CSSProperties } from "react"
import {
  animate,
  AnimatePresence,
  motion,
  useAnimate,
  useMotionValue,
  useReducedMotion,
  useTransform,
  type MotionStyle,
} from "motion/react"
import { formatVnd, formatVndCompact } from "./amount"
import type { NoteRecord, StackSummary, TxType } from "./types"

export type PileSize = "mini" | "xs" | "sm" | "md" | "lg"

/** Narrows an untrusted string (a DOM dataset value) to a pile size. */
export function isPileSize(value: string | undefined): value is PileSize {
  return (
    value === "mini" ||
    value === "xs" ||
    value === "sm" ||
    value === "md" ||
    value === "lg"
  )
}

interface SizeSpec {
  /** Width of the whole button. */
  w: number
  /** Paper area (where sheets sit). */
  sheetW: number
  sheetH: number
  showText: boolean
  /** Label and total under the paper. Off for `mini`, which is only paper. */
  caption: boolean
  /** Gap between paper and caption. */
  gapClass: string
  labelClass: string
  totalClass: string
  textClass: string
}

const SIZES: Record<PileSize, SizeSpec> = {
  // Paper only: for a keyboard-shortened screen. The button is exactly 44px tall.
  mini: {
    w: 60,
    sheetW: 54,
    sheetH: 36,
    showText: false,
    caption: false,
    gapClass: "gap-0",
    labelClass: "text-[11px]",
    totalClass: "text-[13px]",
    textClass: "text-[9px]",
  },
  // Fits a 4-column grid on a narrow phone.
  xs: {
    w: 72,
    sheetW: 58,
    sheetH: 44,
    showText: false,
    caption: true,
    gapClass: "gap-1",
    labelClass: "text-[11px]",
    totalClass: "text-[13px]",
    textClass: "text-[9px]",
  },
  sm: {
    w: 80,
    sheetW: 62,
    sheetH: 58,
    showText: false,
    caption: true,
    gapClass: "gap-2",
    labelClass: "text-[11px]",
    totalClass: "text-[13px]",
    textClass: "text-[9px]",
  },
  md: {
    w: 124,
    sheetW: 98,
    sheetH: 90,
    showText: true,
    caption: true,
    gapClass: "gap-2",
    labelClass: "text-xs",
    totalClass: "text-sm",
    textClass: "text-[11px]",
  },
  lg: {
    w: 172,
    sheetW: 138,
    sheetH: 124,
    showText: true,
    caption: true,
    gapClass: "gap-2",
    labelClass: "text-sm",
    totalClass: "text-lg",
    textClass: "text-xs",
  },
}

/** Outer footprint (px) of a pile per size, for layouts that position piles by hand. */
export const STACK_FOOTPRINT: Record<PileSize, { w: number; h: number }> = {
  mini: { w: SIZES.mini.w, h: 44 },
  xs: { w: SIZES.xs.w, h: 88 },
  sm: { w: SIZES.sm.w, h: 106 },
  md: { w: SIZES.md.w, h: 142 },
  lg: { w: SIZES.lg.w, h: 184 },
}

export interface StackPileProps {
  type: TxType
  summary: StackSummary
  label: string
  onOpen(): void
  size?: PileSize
}

/** Inset (px) of every sheet inside the slot that holds the pile's paper. */
const PAD = 7

/** FNV-1a: stable pseudo-random numbers per note id, so a sheet always lies the same way. */
function seeded(id: string): { r: number; x: number; y: number } {
  let h = 0x811c9dc5
  for (let i = 0; i < id.length; i++) {
    h ^= id.charCodeAt(i)
    h = Math.imul(h, 0x01000193)
  }
  const unit = (shift: number) => (((h >>> shift) & 0xff) / 255) * 2 - 1
  return { r: unit(0), x: unit(8), y: unit(16) }
}

/**
 * Resting pose of a sheet: its size inside the slot (px), and rotation (deg) and offset (px)
 * about the slot centre. Pure, so a flying card can land on exactly this pose.
 */
export interface SheetPose {
  width: number
  height: number
  rotate: number
  x: number
  y: number
}

export function sheetPose(
  recordId: string,
  index: number,
  size: PileSize
): SheetPose {
  const spec = SIZES[size]
  const s = seeded(recordId)
  return {
    width: spec.sheetW - PAD * 2,
    height: spec.sheetH - PAD * 2,
    rotate: s.r * (index === 0 ? 3.5 : 6.5),
    x: s.x * 4,
    y: index * 2.5 + s.y * 2,
  }
}

/** Pose of the top sheet of a pile: where a record that just arrived comes to rest. */
export function topSheetPose(recordId: string, size: PileSize): SheetPose {
  return sheetPose(recordId, 0, size)
}

/**
 * Ids of records that reached their pile by flight. Their sheet mounts already at rest, because
 * the flying card has done the arriving. Read once, when the sheet mounts.
 */
const arrivedIds = new Set<string>()

export function markArrived(id: string): void {
  arrivedIds.add(id)
}

export function clearArrived(id: string): void {
  arrivedIds.delete(id)
}

const EASE_OUT: [number, number, number, number] = [0.22, 1, 0.36, 1]

function Ticker({
  value,
  format,
  reduced,
}: {
  value: number
  format: (n: number) => string
  reduced: boolean
}) {
  const mv = useMotionValue(value)
  const text = useTransform(mv, (v) => format(Math.round(v)))
  useEffect(() => {
    if (reduced) {
      mv.set(value)
      return
    }
    const controls = animate(mv, value, { duration: 0.75, ease: EASE_OUT })
    return () => controls.stop()
  }, [value, reduced, mv])
  return <motion.span>{text}</motion.span>
}

/** Text and amount printed on the top sheet. Shared with the flying card's last frame. */
export function SheetFace({
  record,
  size,
}: {
  record: NoteRecord
  size: PileSize
}) {
  const spec = SIZES[size]
  if (!spec.showText) return null
  return (
    <span className="flex h-full flex-col p-2 text-left">
      <span className="block min-h-0 overflow-hidden">
        <span
          className={`line-clamp-2 leading-snug break-words ${spec.textClass}`}
        >
          {record.text}
        </span>
      </span>
      {record.amount !== null ? (
        <span className="mt-auto pt-1 text-[10px] font-semibold tabular-nums opacity-70">
          {formatVndCompact(record.amount)}
        </span>
      ) : null}
    </span>
  )
}

function Sheet({
  record,
  index,
  size,
  reduced,
}: {
  record: NoteRecord
  index: number
  size: PileSize
  reduced: boolean
}) {
  const [arrived] = useState(() => arrivedIds.has(record.id))
  const { rotate, x, y } = sheetPose(record.id, index, size)
  return (
    <motion.span
      data-sheet={record.id}
      data-sheet-index={index}
      className="gidi-sheet absolute block overflow-hidden"
      style={
        {
          left: PAD,
          right: PAD,
          top: PAD,
          bottom: PAD,
          "--sheet-paper": `var(--paper-${record.type})`,
          "--sheet-ink": `var(--ink-${record.type})`,
          zIndex: 10 - index,
        } as MotionStyle
      }
      initial={
        arrived
          ? false
          : reduced
            ? { opacity: 0, rotate, x, y }
            : { opacity: 0, rotate: rotate + 5, x, y: y - 16, scale: 0.94 }
      }
      animate={{ opacity: 1, rotate, x, y, scale: 1 }}
      exit={{ opacity: 0, scale: 0.9, transition: { duration: 0.16 } }}
      transition={
        reduced
          ? { duration: 0.15 }
          : { type: "spring", stiffness: 380, damping: 28 }
      }
    >
      {index === 0 ? <SheetFace record={record} size={size} /> : null}
    </motion.span>
  )
}

function EmptySlot({
  type,
  label,
  spec,
}: {
  type: TxType
  label: string
  spec: SizeSpec
}) {
  return (
    <span
      className="gidi-slot absolute block"
      style={
        {
          left: PAD,
          right: PAD,
          top: PAD,
          bottom: PAD,
          "--slot-ink": `var(--ink-${type})`,
          zIndex: 10,
        } as CSSProperties
      }
    >
      <span
        className={`flex h-full items-center justify-center px-1 text-center leading-tight font-medium ${spec.labelClass}`}
      >
        {spec.caption ? label : null}
      </span>
    </span>
  )
}

export function StackPile({
  type,
  summary,
  label,
  onOpen,
  size = "md",
}: StackPileProps) {
  const spec = SIZES[size]
  const reduced = useReducedMotion() ?? false
  const [scope, animateScope] = useAnimate<HTMLSpanElement>()

  const { count, total } = summary
  const sheets = summary.latest.slice(0, 3)
  const empty = count === 0
  const ink = `var(--ink-${type})`

  useEffect(() => {
    const onLanded = (e: Event) => {
      const detail = (e as CustomEvent<{ type: TxType }>).detail
      const el = scope.current
      if (!detail || !el || detail.type !== type) return
      if (reduced) {
        void animateScope(el, { opacity: [1, 0.55, 1] }, { duration: 0.35 })
        return
      }
      // Short squash, anchored at the bottom edge so the pile stays on the table.
      void animateScope(
        el,
        { scaleX: [1, 1.05, 0.99, 1], scaleY: [1, 0.93, 1.02, 1] },
        { duration: 0.36, ease: "easeOut", times: [0, 0.3, 0.7, 1] }
      )
    }
    window.addEventListener("gidi:landed", onLanded)
    return () => window.removeEventListener("gidi:landed", onLanded)
  }, [type, reduced, scope, animateScope])

  const ariaLabel = empty
    ? `${label}: chưa có ghi chú`
    : `${label}: ${count} ghi chú, tổng ${formatVnd(total)}`

  return (
    <motion.button
      type="button"
      data-stack={type}
      aria-label={ariaLabel}
      title={empty ? ariaLabel : `${label}: tổng ${formatVnd(total)}`}
      onClick={onOpen}
      className={`group relative flex shrink-0 cursor-pointer touch-manipulation flex-col items-center rounded-xl p-1 outline-none select-none [-webkit-tap-highlight-color:transparent] [-webkit-touch-callout:none] focus-visible:ring-[3px] focus-visible:ring-ring/60 focus-visible:ring-offset-2 focus-visible:ring-offset-background active:bg-foreground/[0.08] ${spec.gapClass}`}
      style={{ width: spec.w, minWidth: 44, minHeight: 44 }}
      whileHover={reduced ? undefined : { y: -4 }}
      whileTap={reduced ? undefined : { scale: 0.94 }}
      transition={{ type: "spring", stiffness: 420, damping: 26 }}
    >
      <span
        ref={scope}
        data-stack-slot={type}
        data-stack-size={size}
        className="relative block"
        style={{
          width: spec.sheetW,
          height: spec.sheetH,
          transformOrigin: "50% 100%",
        }}
      >
        {empty ? (
          <EmptySlot type={type} label={label} spec={spec} />
        ) : (
          <AnimatePresence initial={false}>
            {sheets
              .map((record, index) => ({ record, index }))
              .reverse()
              .map(({ record, index }) => (
                <Sheet
                  key={record.id}
                  record={record}
                  index={index}
                  size={size}
                  reduced={reduced}
                />
              ))}
          </AnimatePresence>
        )}
        {!empty ? (
          <span
            className="absolute -top-1 -right-1 z-20 min-w-5 rounded-full px-1.5 py-px text-center text-[10px] leading-4 font-semibold tabular-nums shadow-sm"
            style={{ background: ink, color: "var(--background)" }}
          >
            <Ticker value={count} format={String} reduced={reduced} />
          </span>
        ) : null}
      </span>

      {spec.caption ? (
        <span className="flex w-full flex-col items-center gap-px leading-tight">
          {empty ? (
            <>
              <span
                className={`font-medium text-muted-foreground ${spec.labelClass}`}
              >
                Trống
              </span>
              <span
                className={`font-semibold text-muted-foreground/70 tabular-nums ${spec.totalClass}`}
              >
                0 ₫
              </span>
            </>
          ) : (
            <>
              <span
                className={`w-full truncate font-medium ${spec.labelClass}`}
                style={{ color: ink }}
              >
                {label}
              </span>
              <span
                className={`font-semibold tabular-nums ${spec.totalClass}`}
                style={{ color: "var(--foreground)" }}
              >
                <Ticker
                  value={total}
                  format={formatVndCompact}
                  reduced={reduced}
                />
              </span>
            </>
          )}
        </span>
      ) : null}
    </motion.button>
  )
}
