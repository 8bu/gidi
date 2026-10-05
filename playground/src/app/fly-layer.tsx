"use client"

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react"
import { createPortal } from "react-dom"
import {
  animate,
  useReducedMotion,
  type AnimationPlaybackControls,
} from "motion/react"

import { cn } from "@/lib/utils"
import { formatVnd } from "./amount"
import { useNotes } from "./store"
import { TYPE_META } from "./taxonomy"
import {
  clearArrived,
  isPileSize,
  markArrived,
  SheetFace,
  topSheetPose,
  type PileSize,
  type SheetPose,
} from "./stack-pile"
import type { NoteRecord, TxType } from "./types"

/** A rectangle in viewport coordinates. `DOMRect` satisfies it. */
export interface FlyRect {
  left: number
  top: number
  width: number
  height: number
}

export interface FlyOptions {
  /**
   * Persists the record once the paper lands. Defaults to `add(record)`.
   * Re-typing an existing record passes `(r) => update(id, { type: r.type })`.
   */
  save?: (record: NoteRecord) => Promise<void>
}

/**
 * Flies a paper card from `from` to the stack of `record.type`, then saves and
 * announces the landing with the `gidi:landed` window event.
 */
export type FlyFn = (
  record: NoteRecord,
  from: FlyRect,
  options?: FlyOptions
) => void

interface Flight {
  id: number
  record: NoteRecord
  from: FlyRect
  save: (record: NoteRecord) => Promise<void>
}

const FlyContext = createContext<FlyFn | null>(null)

export function useFly(): FlyFn {
  const fly = useContext(FlyContext)
  if (!fly) throw new Error("useFly must be used inside <FlyProvider>")
  return fly
}

export function FlyProvider({
  children,
  onError,
}: {
  children: ReactNode
  onError?: (message: string) => void
}) {
  const { add } = useNotes()
  const [flights, setFlights] = useState<Flight[]>([])
  const nextId = useRef(0)

  const latest = useRef({ add, onError })
  useEffect(() => {
    latest.current = { add, onError }
  })

  const fly = useCallback<FlyFn>((record, from, options) => {
    nextId.current += 1
    const flight: Flight = {
      id: nextId.current,
      record,
      from: {
        left: from.left,
        top: from.top,
        width: from.width,
        height: from.height,
      },
      save: options?.save ?? ((r) => latest.current.add(r)),
    }
    setFlights((current) => [...current, flight])
  }, [])

  const finish = useCallback((id: number) => {
    setFlights((current) => current.filter((f) => f.id !== id))
  }, [])

  const fail = useCallback((error: unknown) => {
    const message = error instanceof Error ? error.message : String(error)
    latest.current.onError?.(`Không lưu được ghi chú: ${message}`)
  }, [])

  return (
    <FlyContext.Provider value={fly}>
      {children}
      {createPortal(
        <div
          aria-hidden
          className="pointer-events-none fixed inset-0 z-[100] overflow-hidden"
        >
          {flights.map((flight) => (
            <FlightCard
              key={flight.id}
              flight={flight}
              onDone={finish}
              onError={fail}
            />
          ))}
        </div>,
        document.body
      )}
    </FlyContext.Provider>
  )
}

const EDGE_MARGIN = 28
/** Both borders of `.gidi-sheet` (1px each side): the face sits inside them. */
const SHEET_BORDER = 2
const FLIGHT_SECONDS = 0.66
/** Decelerates into the slot, with no overshoot and no settling tail. */
const FLIGHT_EASE: [number, number, number, number] = [0.3, 0.7, 0.1, 1]
/** Progress at which the paper counts as touching the pile. */
const CONTACT_AT = 0.98

function findStack(type: TxType): HTMLElement | null {
  return findVisible(`[data-stack~="${type}"]`)
}

function findSlot(type: TxType): HTMLElement | null {
  return findVisible(`[data-stack-slot="${type}"]`)
}

function findVisible(selector: string): HTMLElement | null {
  for (const el of document.querySelectorAll<HTMLElement>(selector)) {
    const rect = el.getBoundingClientRect()
    if (rect.width > 0 && rect.height > 0) return el
  }
  return null
}

function slotSize(type: TxType): PileSize {
  const size = findSlot(type)?.dataset.stackSize
  return size !== undefined && isPileSize(size) ? size : "md"
}

/** Where and how the flying card must end up, read live from the pile's slot. */
interface Destination {
  /** Centre of the landed sheet, viewport px. */
  cx: number
  cy: number
  /** Current scale of the slot (hover, tap, landing squash). */
  sx: number
  sy: number
  /** The slot is fully on screen, so the card can merge with the sheet. */
  onScreen: boolean
}

function readDestination(type: TxType, pose: SheetPose): Destination {
  const vw = window.innerWidth
  const vh = window.innerHeight
  const slot = findSlot(type)
  if (!slot) {
    return {
      cx: vw / 2,
      cy: vh - EDGE_MARGIN * 2,
      sx: 1,
      sy: 1,
      onScreen: false,
    }
  }
  const rect = slot.getBoundingClientRect()
  const sx = rect.width / slot.offsetWidth
  const sy = rect.height / slot.offsetHeight
  const onScreen =
    rect.left >= 0 && rect.right <= vw && rect.top >= 0 && rect.bottom <= vh
  const cx = rect.left + rect.width / 2 + pose.x * sx
  const cy = rect.top + rect.height / 2 + pose.y * sy
  return {
    cx: onScreen ? cx : Math.min(Math.max(cx, EDGE_MARGIN), vw - EDGE_MARGIN),
    cy: onScreen ? cy : Math.min(Math.max(cy, EDGE_MARGIN), vh - EDGE_MARGIN),
    sx,
    sy,
    onScreen,
  }
}

function ensureVisible(type: TxType, instant: boolean) {
  const el = findStack(type)
  if (!el) return
  const rect = el.getBoundingClientRect()
  const fullyVisible =
    rect.left >= 0 &&
    rect.right <= window.innerWidth &&
    rect.top >= 0 &&
    rect.bottom <= window.innerHeight
  if (fullyVisible) return
  el.scrollIntoView({
    behavior: instant ? "auto" : "smooth",
    block: "nearest",
    inline: "center",
  })
}

function random(min: number, max: number) {
  return min + Math.random() * (max - min)
}

function smoothstep(edge0: number, edge1: number, x: number) {
  const t = Math.min(Math.max((x - edge0) / (edge1 - edge0), 0), 1)
  return t * t * (3 - 2 * t)
}

/** Resolves after React has committed and the browser has painted twice (or after a short cap). */
function afterCommit(): Promise<void> {
  return new Promise((resolve) => {
    const cap = window.setTimeout(resolve, 150)
    requestAnimationFrame(() =>
      requestAnimationFrame(() => {
        window.clearTimeout(cap)
        resolve()
      })
    )
  })
}

/** Blend `from` into `to` as `amount` goes 0 to 1; exactly `to` at 1. */
function mixVar(from: string, to: string, amount: number): string {
  return amount >= 1
    ? to
    : `color-mix(in oklab, ${from} ${(1 - amount) * 100}%, ${to} ${amount * 100}%)`
}

function FlightCard({
  flight,
  onDone,
  onError,
}: {
  flight: Flight
  onDone: (id: number) => void
  onError: (error: unknown) => void
}) {
  const reduced = useReducedMotion() ?? false
  const cardRef = useRef<HTMLDivElement>(null)
  const shadowRef = useRef<HTMLDivElement>(null)
  const bodyRef = useRef<HTMLDivElement>(null)
  const bigRef = useRef<HTMLDivElement>(null)
  const detailRef = useRef<HTMLDivElement>(null)
  const faceRef = useRef<HTMLDivElement>(null)
  const landed = useRef(false)
  // Stable across StrictMode's effect replay.
  const [start] = useState(() => random(-1.2, 1.2))
  const [size] = useState(() => slotSize(flight.record.type))

  useEffect(() => {
    const card = cardRef.current
    const shadow = shadowRef.current
    const body = bodyRef.current
    const big = bigRef.current
    const detail = detailRef.current
    const face = faceRef.current
    if (!card || !shadow || !body || !big || !detail || !face) return
    const { record, from, id } = flight
    const type = record.type
    const pose = topSheetPose(record.id, size)
    const startX = from.left + from.width / 2
    const startY = from.top + from.height / 2

    /** Draw the card at progress `p`; at 1 it is the pile's top sheet, pixel for pixel. */
    const paint = (p: number) => {
      const dest = readDestination(type, pose)
      const tint = smoothstep(0.1, 0.75, p)
      const dx = dest.cx - startX
      const dy = dest.cy - startY
      const arc = Math.min(Math.max(Math.hypot(dx, dy) * 0.22, 36), 150)
      const w = p >= 1 ? pose.width : from.width + (pose.width - from.width) * p
      const h =
        p >= 1 ? pose.height : from.height + (pose.height - from.height) * p
      const x = startX + dx * p
      const y = startY + dy * p - arc * 4 * p * (1 - p)
      const rotate = start + (pose.rotate - start) * p
      const scaleX = 1 + (dest.sx - 1) * p
      const scaleY = 1 + (dest.sy - 1) * p
      card.style.width = `${w}px`
      card.style.height = `${h}px`
      card.style.transform = `translate3d(${x - w / 2}px, ${y - h / 2}px, 0) rotate(${rotate}deg) scale(${scaleX}, ${scaleY})`
      body.style.setProperty(
        "--sheet-paper",
        mixVar("var(--paper-note)", `var(--paper-${type})`, tint)
      )
      body.style.setProperty(
        "--sheet-ink",
        mixVar("var(--ink-note)", `var(--ink-${type})`, tint)
      )
      shadow.style.opacity = String(1 - smoothstep(0.2, 0.95, p))
      const fit = Math.min(w / from.width, h / from.height)
      big.style.transform = `translate(-50%, -50%) scale(${fit})`
      big.style.opacity = String(1 - smoothstep(0.5, 0.85, p))
      detail.style.opacity = String(tint)
      face.style.opacity = String(smoothstep(0.55, 0.95, p))
      // The face is laid out once, in the sheet's own box, and scaled with the card, so
      // its text wraps exactly as it will on the sheet and never reflows mid-flight.
      face.style.transform =
        p >= 1
          ? "none"
          : `scale(${(w - SHEET_BORDER) / (pose.width - SHEET_BORDER)}, ${(h - SHEET_BORDER) / (pose.height - SHEET_BORDER)})`
    }

    let flightEnded: () => void = () => {}
    const ended = new Promise<void>((resolve) => {
      flightEnded = resolve
    })
    let raf = 0
    let fade: AnimationPlaybackControls | null = null

    // After the flight the pile may still move (landing squash, hover): keep following it.
    const hold = () => {
      paint(1)
      raf = requestAnimationFrame(hold)
    }

    const settle = async () => {
      let failed = false
      try {
        await flight.save(record)
      } catch (error) {
        failed = true
        onError(error)
      }
      await ended
      await afterCommit()
      const slot = findSlot(type)
      const top = slot?.querySelector<HTMLElement>(
        `[data-sheet="${record.id}"]`
      )
      const merged =
        !failed &&
        top?.dataset.sheetIndex === "0" &&
        readDestination(type, pose).onScreen
      const finish = () => {
        clearArrived(record.id)
        onDone(id)
      }
      if (merged) {
        // The sheet under the card is at rest in the same pose: just lift the card off.
        finish()
        return
      }
      cancelAnimationFrame(raf)
      fade = animate(1, 0, {
        duration: 0.2,
        ease: "easeOut",
        onUpdate: (v) => {
          card.style.opacity = String(v)
        },
        onComplete: finish,
      })
    }

    const contact = () => {
      if (landed.current) return
      landed.current = true
      markArrived(record.id)
      window.dispatchEvent(new CustomEvent("gidi:landed", { detail: { type } }))
      void settle()
    }

    ensureVisible(type, reduced)

    if (reduced) {
      // Fade in place on the stack instead of flying.
      paint(1)
      card.style.opacity = "0"
      const controls = animate(0, 1, {
        duration: 0.26,
        ease: "easeOut",
        onUpdate: (v) => {
          card.style.opacity = String(v)
        },
        onComplete: () => {
          raf = requestAnimationFrame(hold)
          flightEnded()
          contact()
        },
      })
      return () => {
        controls.stop()
        fade?.stop()
        cancelAnimationFrame(raf)
      }
    }

    paint(0)
    const controls = animate(0, 1, {
      duration: FLIGHT_SECONDS,
      ease: FLIGHT_EASE,
      onUpdate: (p) => {
        // Re-read every frame: the page may be scrolling the stack into view.
        paint(p)
        if (p >= CONTACT_AT) contact()
      },
      onComplete: () => {
        paint(1)
        raf = requestAnimationFrame(hold)
        flightEnded()
        contact()
      },
    })
    return () => {
      controls.stop()
      fade?.stop()
      cancelAnimationFrame(raf)
    }
  }, [flight, reduced, start, size, onDone, onError])

  const { record, from } = flight
  const pose = topSheetPose(record.id, size)
  const meta = TYPE_META[record.type]
  const compact = from.height < 190
  return (
    <div
      ref={cardRef}
      className="absolute top-0 left-0"
      style={{
        width: from.width,
        height: from.height,
        transformOrigin: "50% 50%",
        willChange: "transform",
      }}
    >
      <div
        ref={shadowRef}
        className="absolute inset-0 rounded-[6px_6px_8px_8px]"
        style={{ boxShadow: "var(--paper-shadow-3)" }}
      />
      <div
        ref={bodyRef}
        className="gidi-sheet absolute inset-0 overflow-hidden"
      >
        <div
          ref={bigRef}
          className={cn(
            "absolute top-1/2 left-1/2 flex flex-col justify-between",
            compact ? "p-3" : "p-5"
          )}
          style={{ width: from.width, height: from.height }}
        >
          <p
            className={cn(
              "break-words whitespace-pre-wrap",
              compact
                ? "line-clamp-2 text-sm leading-5"
                : "line-clamp-6 text-[1.05rem] leading-7"
            )}
          >
            {record.text}
          </p>
          <div
            ref={detailRef}
            className="flex items-end justify-between gap-3 opacity-0"
          >
            <span
              className={
                compact ? "text-sm font-medium" : "text-xl font-medium"
              }
            >
              {meta.label}
            </span>
            <span
              className={cn(
                "font-semibold tabular-nums",
                compact ? "text-lg" : "text-3xl"
              )}
            >
              {record.amount === null ? "" : formatVnd(record.amount)}
            </span>
          </div>
        </div>
        <div
          ref={faceRef}
          className="absolute top-0 left-0 opacity-0"
          style={{
            width: pose.width - SHEET_BORDER,
            height: pose.height - SHEET_BORDER,
            transformOrigin: "0 0",
          }}
        >
          <SheetFace record={record} size={size} />
        </div>
      </div>
    </div>
  )
}
