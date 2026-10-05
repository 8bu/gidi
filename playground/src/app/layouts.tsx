import { useLayoutEffect, useRef, useState, type ReactNode } from "react"
import { motion, useReducedMotion } from "motion/react"
import { STACK_FOOTPRINT, StackPile, type PileSize } from "./stack-pile"
import { TYPE_META } from "./taxonomy"
import { TX_TYPES, type StackSummary, type TxType } from "./types"

export interface LayoutProps {
  summaries: Record<TxType, StackSummary>
  composer: ReactNode
  onOpenStack(type: TxType): void
}

/** Below this width the page stacks: composer on top, a 4x2 grid of piles under it. */
const PHONE_MAX = 640
/** A short, not-too-wide stage (a phone on its side) puts composer left and the grid right. */
const SPLIT_MAX_STAGE_H = 436
const SPLIT_MAX_W = 1000
/**
 * The visible region the shell gives the layout: the (visual) viewport minus the top bar and the
 * bottom safe area. `--gidi-vvh` shrinks while the on-screen keyboard is open.
 */
const STAGE_H =
  "calc(var(--gidi-vvh, 100dvh) - var(--gidi-topbar, 4rem) - var(--gidi-safe-bottom, 0px))"

/** Matches the composer's `max-w-[26rem]`. */
const COMPOSER_MAX_W = 416
const FALLBACK_COMPOSER_H = 300
/** Clear space kept between the composer paper (it tilts and casts a shadow) and a pile. */
const COMPOSER_MARGIN = 12
const PILE_MARGIN = 4
/** Clear space between a pile and the stage edge. */
const EDGE = 8
/** Extra clearance a ring needs to win back from the split layout, so the two never flicker. */
const RING_HYSTERESIS = 28
/** Padding the stacked layout puts around the composer (py-3) and the grid (pt-1, pb-3), px. */
const STACK_CHROME = 24 + 16 + 8

/** Largest first. `mini` (paper only) is the last resort for the grid. */
const RING_SIZES: readonly PileSize[] = ["md", "sm", "xs"]
const GRID_SIZES: readonly PileSize[] = ["md", "sm", "xs", "mini"]

type Mode = "stack" | "split" | "ring"

interface Box {
  w: number
  h: number
}

/** Rounded size of an element, kept in sync by a ResizeObserver. `null` until first measured. */
function useBox<T extends HTMLElement>() {
  const ref = useRef<T>(null)
  const [box, setBox] = useState<Box | null>(null)
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const ro = new ResizeObserver(([entry]) => {
      const w = Math.round(entry.contentRect.width)
      const h = Math.round(entry.contentRect.height)
      setBox((prev) => (prev && prev.w === w && prev.h === h ? prev : { w, h }))
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  return [ref, box] as const
}

/** Entrance: a pile rises onto the desk, staggered by `index`. Fades only for reduced motion. */
function Rise({
  index,
  className,
  children,
}: {
  index: number
  className?: string
  children: ReactNode
}) {
  const reduced = useReducedMotion() ?? false
  return (
    <motion.div
      className={className}
      initial={reduced ? { opacity: 0 } : { opacity: 0, y: 22, scale: 0.9 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={
        reduced
          ? { duration: 0.2 }
          : {
              type: "spring",
              stiffness: 260,
              damping: 24,
              delay: 0.1 + index * 0.05,
            }
      }
    >
      {children}
    </motion.div>
  )
}

/* ----------------------------------------------------------------- plans */

/** Superellipse (squircle) easing so a ring of piles hugs a rectangular viewport. */
const squircle = (v: number) => Math.sign(v) * Math.abs(v) ** 0.77

interface RingPlan {
  size: PileSize
  /** Pile centres relative to the stage centre, in `TX_TYPES` order. */
  points: { x: number; y: number }[]
}

/**
 * Piles on an ellipse as tall and wide as the stage allows. `null` when, at this size, a pile would
 * touch the composer or another pile, so the caller can try a smaller size.
 */
function ringPlan(
  size: PileSize,
  stageW: number,
  stageH: number,
  composerH: number,
  slack: number
): RingPlan | null {
  const fp = STACK_FOOTPRINT[size]
  const rx = Math.min(660, stageW / 2 - fp.w / 2 - EDGE)
  const ry = stageH / 2 - fp.h / 2 - EDGE
  if (rx <= 0 || ry <= 0) return null

  const points = TX_TYPES.map((_, i) => {
    // Clockwise from the top-left, offset half a step so no pile sits dead above or below the note.
    const angle = ((-112.5 + 45 * i) * Math.PI) / 180
    return {
      x: rx * squircle(Math.cos(angle)),
      y: ry * squircle(Math.sin(angle)),
    }
  })

  const clearW =
    Math.min(COMPOSER_MAX_W, stageW) / 2 + COMPOSER_MARGIN + slack + fp.w / 2
  const clearH = composerH / 2 + COMPOSER_MARGIN + slack + fp.h / 2
  for (const p of points) {
    if (Math.abs(p.x) < clearW && Math.abs(p.y) < clearH) return null
  }
  for (let i = 0; i < points.length; i++) {
    for (let j = i + 1; j < points.length; j++) {
      if (
        Math.abs(points[i].x - points[j].x) < fp.w + PILE_MARGIN &&
        Math.abs(points[i].y - points[j].y) < fp.h + PILE_MARGIN
      ) {
        return null
      }
    }
  }
  return { size, points }
}

/** Largest size whose 4x2 grid fits a `w` x `h` area; `mini` (paper only) if none does. */
function gridSize(w: number, h: number, gapX: number, gapY: number): PileSize {
  const cell = (w - gapX * 3) / 4
  for (const size of GRID_SIZES) {
    const fp = STACK_FOOTPRINT[size]
    if (fp.w <= cell && fp.h * 2 + gapY <= h) return size
  }
  return "mini"
}

/* ---------------------------------------------------------------- layout */

const ROOT: Record<Mode, string> = {
  ring: "relative w-full overflow-hidden",
  stack: "relative flex w-full flex-col overflow-x-clip",
  split: "relative flex w-full flex-row overflow-x-clip",
}

const COMPOSER_ZONE: Record<Mode, string> = {
  ring: "pointer-events-none absolute inset-x-0 top-0 z-10 grid place-items-center",
  stack: "flex flex-1 flex-col px-3 py-3 group-data-[keyboard=open]/app:py-1",
  split: "flex w-1/2 min-w-0 flex-col px-3 py-2",
}

const PILES_ZONE: Record<Mode, string> = {
  ring: "pointer-events-none absolute inset-x-0 top-0",
  stack: "grid grid-cols-4 justify-items-center gap-x-1 gap-y-2 px-3 pt-1 pb-3",
  split:
    "grid w-1/2 grid-cols-4 content-center justify-items-center gap-x-1 gap-y-4 px-3 py-2",
}

/**
 * One DOM shape for every mode (only classes change), so the composer is never remounted and keeps
 * focus while the window resizes or the keyboard opens.
 *
 * - stack (width < 640): composer on top, 4x2 grid of piles under it.
 * - split (a phone on its side, or a ring that cannot fit): composer left, 4x2 grid right.
 * - ring: piles on an ellipse around the composer, sized to what the stage allows.
 */
export function RingLayout({ summaries, composer, onOpenStack }: LayoutProps) {
  const [stageRef, stage] = useBox<HTMLDivElement>()
  const [composerRef, composerBox] = useBox<HTMLDivElement>()
  const reduced = useReducedMotion() ?? false
  const [lastMode, setLastMode] = useState<Mode>("stack")

  const stageW = stage?.w ?? 0
  const stageH = stage?.h ?? 0
  const composerH = composerBox?.h ?? FALLBACK_COMPOSER_H

  let mode: Mode = "stack"
  let plan: RingPlan | null = null
  if (stage !== null && stageW >= PHONE_MAX) {
    if (stageH < SPLIT_MAX_STAGE_H && stageW < SPLIT_MAX_W) {
      mode = "split"
    } else {
      const slack = lastMode === "split" ? RING_HYSTERESIS : 0
      for (const size of RING_SIZES) {
        plan = ringPlan(size, stageW, stageH, composerH, slack)
        if (plan) break
      }
      mode = plan ? "ring" : "split"
    }
  }
  if (mode !== lastMode) setLastMode(mode)
  const ring = mode === "ring" ? plan : null

  const grid =
    mode === "stack"
      ? gridSize(stageW - 24, stageH - composerH - STACK_CHROME, 4, 8)
      : gridSize(stageW / 2 - 24, stageH - 16, 4, 16)

  return (
    <div
      className={ROOT[mode]}
      style={mode === "ring" ? { height: STAGE_H } : { minHeight: STAGE_H }}
    >
      {/* Exactly the visible stage; measured, never grows with its content. */}
      <div
        ref={stageRef}
        aria-hidden
        className="pointer-events-none absolute inset-x-0 top-0"
        style={{ height: STAGE_H }}
      />

      <div
        className={COMPOSER_ZONE[mode]}
        style={mode === "ring" ? { height: STAGE_H } : undefined}
      >
        <div
          ref={composerRef}
          className="pointer-events-auto mx-auto my-auto w-full max-w-[26rem]"
        >
          {composer}
        </div>
      </div>

      <div
        className={PILES_ZONE[mode]}
        style={mode === "ring" ? { height: STAGE_H } : undefined}
      >
        {stage === null
          ? null
          : ring
            ? TX_TYPES.map((type, i) => {
                const fp = STACK_FOOTPRINT[ring.size]
                const x = stageW / 2 + ring.points[i].x - fp.w / 2
                const y = stageH / 2 + ring.points[i].y - fp.h / 2
                return (
                  <motion.div
                    key={type}
                    className="pointer-events-auto absolute top-0 left-0"
                    style={{ width: fp.w }}
                    initial={{
                      x: stageW / 2 - fp.w / 2,
                      y: stageH / 2 - fp.h / 2,
                      scale: reduced ? 1 : 0.5,
                      opacity: 0,
                    }}
                    animate={{ x, y, scale: 1, opacity: 1 }}
                    transition={
                      reduced
                        ? { duration: 0.2 }
                        : {
                            type: "spring",
                            stiffness: 170,
                            damping: 22,
                            delay: 0.1 + i * 0.05,
                          }
                    }
                  >
                    <StackPile
                      type={type}
                      summary={summaries[type]}
                      label={TYPE_META[type].label}
                      onOpen={() => onOpenStack(type)}
                      size={ring.size}
                    />
                  </motion.div>
                )
              })
            : TX_TYPES.map((type, i) => (
                <Rise key={type} index={i} className="shrink-0">
                  <StackPile
                    type={type}
                    summary={summaries[type]}
                    label={TYPE_META[type].label}
                    onOpen={() => onOpenStack(type)}
                    size={grid}
                  />
                </Rise>
              ))}
      </div>
    </div>
  )
}
