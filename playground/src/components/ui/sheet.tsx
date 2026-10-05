"use client"

import * as React from "react"
import { cn } from "cn"
import { Dialog as SheetPrimitive } from "radix-ui"
import { XIcon } from "lucide-react"

import { Button } from "@/components/ui/button"

const SheetCloseContext = React.createContext<() => void>(() => undefined)

function Sheet({
  onOpenChange,
  ...props
}: React.ComponentProps<typeof SheetPrimitive.Root>) {
  const close = React.useCallback(() => onOpenChange?.(false), [onOpenChange])
  return (
    <SheetCloseContext.Provider value={close}>
      <SheetPrimitive.Root
        data-slot="sheet"
        onOpenChange={onOpenChange}
        {...props}
      />
    </SheetCloseContext.Provider>
  )
}

function SheetTrigger({
  ...props
}: React.ComponentProps<typeof SheetPrimitive.Trigger>) {
  return <SheetPrimitive.Trigger data-slot="sheet-trigger" {...props} />
}

function SheetClose({
  ...props
}: React.ComponentProps<typeof SheetPrimitive.Close>) {
  return <SheetPrimitive.Close data-slot="sheet-close" {...props} />
}

function SheetPortal({
  ...props
}: React.ComponentProps<typeof SheetPrimitive.Portal>) {
  return <SheetPrimitive.Portal data-slot="sheet-portal" {...props} />
}

function SheetOverlay({
  className,
  ...props
}: React.ComponentProps<typeof SheetPrimitive.Overlay>) {
  return (
    <SheetPrimitive.Overlay
      data-slot="sheet-overlay"
      className={cn(
        "fixed inset-0 z-50 bg-black/30 duration-200 supports-backdrop-filter:backdrop-blur-[2px] data-open:animate-in data-open:fade-in-0 data-closed:animate-out data-closed:fade-out-0",
        className
      )}
      {...props}
    />
  )
}

const SWIPE_CLOSE_DISTANCE = 96
const SWIPE_CLOSE_VELOCITY = 0.5
/** Movement before a press turns into a drag, so taps stay taps. */
const DRAG_SLOP = 6
/** Window the release velocity is measured over. */
const VELOCITY_WINDOW_MS = 100
/** Pointer-down on these never starts a drag. */
const NO_DRAG = "button, a, input, textarea, select, [role=combobox]"
/** Visual viewport shrinks by more than this when the on-screen keyboard is up. */
const KEYBOARD_MIN_PX = 80

interface DragBind {
  onPointerDown(event: React.PointerEvent<HTMLElement>): void
  onPointerMove(event: React.PointerEvent<HTMLElement>): void
  onPointerUp(event: React.PointerEvent<HTMLElement>): void
  onPointerCancel(event: React.PointerEvent<HTMLElement>): void
}

/** Set inside a bottom drawer: spread on any element that should drag it. */
const DrawerDragContext = React.createContext<DragBind | null>(null)

interface DragState {
  id: number
  startY: number
  dy: number
  dragging: boolean
  samples: { y: number; t: number }[]
}

/**
 * Pointer drag for the bottom drawer: follows the finger, closes when pulled
 * far or fast enough, otherwise springs back.
 */
function useDrawerDrag(
  contentRef: React.RefObject<HTMLDivElement | null>
): DragBind {
  const close = React.useContext(SheetCloseContext)
  const state = React.useRef<DragState | null>(null)
  const timer = React.useRef<number | undefined>(undefined)
  React.useEffect(() => () => window.clearTimeout(timer.current), [])

  const onPointerDown = (event: React.PointerEvent<HTMLElement>) => {
    if (state.current) return
    if (event.pointerType === "mouse" && event.button !== 0) return
    if (event.target instanceof Element && event.target.closest(NO_DRAG)) return
    state.current = {
      id: event.pointerId,
      startY: event.clientY,
      dy: 0,
      dragging: false,
      samples: [{ y: event.clientY, t: event.timeStamp }],
    }
  }

  const onPointerMove = (event: React.PointerEvent<HTMLElement>) => {
    const s = state.current
    const el = contentRef.current
    if (!s || !el || s.id !== event.pointerId) return
    const dy = event.clientY - s.startY
    if (!s.dragging) {
      if (Math.abs(dy) < DRAG_SLOP) return
      if (dy < 0) {
        // Pulling up is not a gesture of the drawer.
        state.current = null
        return
      }
      s.dragging = true
      event.currentTarget.setPointerCapture(event.pointerId)
      el.style.transition = "none"
    }
    s.dy = Math.max(0, dy - DRAG_SLOP)
    s.samples.push({ y: event.clientY, t: event.timeStamp })
    while (
      s.samples.length > 2 &&
      event.timeStamp - s.samples[0].t > VELOCITY_WINDOW_MS
    ) {
      s.samples.shift()
    }
    el.style.transform = `translateY(${s.dy}px)`
  }

  const finish = (
    event: React.PointerEvent<HTMLElement>,
    cancelled: boolean
  ) => {
    const s = state.current
    const el = contentRef.current
    if (!s || s.id !== event.pointerId) return
    state.current = null
    if (!s.dragging || !el) return
    const first = s.samples[0]
    const last = s.samples[s.samples.length - 1]
    const velocity = (last.y - first.y) / Math.max(1, last.t - first.t)
    const shouldClose =
      !cancelled &&
      (s.dy > SWIPE_CLOSE_DISTANCE ||
        (velocity > SWIPE_CLOSE_VELOCITY && s.dy > 12))
    if (shouldClose) {
      el.style.transition = "transform 200ms cubic-bezier(0.32, 0.72, 0, 1)"
      el.style.transform = "translateY(100%)"
      // The slide-out is already done; skip the stock exit animation.
      el.style.animation = "none"
      timer.current = window.setTimeout(close, 190)
    } else {
      el.style.transition = "transform 280ms cubic-bezier(0.32, 0.72, 0, 1)"
      el.style.transform = ""
    }
  }

  return {
    onPointerDown,
    onPointerMove,
    onPointerUp: (event) => finish(event, false),
    onPointerCancel: (event) => finish(event, true),
  }
}

/**
 * While the on-screen keyboard is up, lift the drawer above it and cap its
 * height to the visible area (fixed elements stay put behind the keyboard).
 */
function useKeyboardLift(contentRef: React.RefObject<HTMLDivElement | null>) {
  React.useEffect(() => {
    const vv = window.visualViewport
    const el = contentRef.current
    if (!vv || !el) return
    const sync = () => {
      const lift = Math.round(window.innerHeight - vv.height - vv.offsetTop)
      if (lift > KEYBOARD_MIN_PX) {
        el.style.bottom = `${lift}px`
        el.style.maxHeight = `min(85dvh, ${Math.round(vv.height) - 8}px)`
        el.dataset.keyboard = "open"
      } else {
        el.style.bottom = ""
        el.style.maxHeight = ""
        delete el.dataset.keyboard
      }
    }
    sync()
    vv.addEventListener("resize", sync)
    vv.addEventListener("scroll", sync)
    return () => {
      vv.removeEventListener("resize", sync)
      vv.removeEventListener("scroll", sync)
    }
  }, [contentRef])
}

/** Drawer chrome: grab handle, drag behaviour shared with the header, keyboard lift. */
function DrawerShell({
  contentRef,
  children,
}: {
  contentRef: React.RefObject<HTMLDivElement | null>
  children: React.ReactNode
}) {
  const bind = useDrawerDrag(contentRef)
  useKeyboardLift(contentRef)
  return (
    <DrawerDragContext.Provider value={bind}>
      <div
        data-slot="sheet-handle"
        className="flex h-6 shrink-0 cursor-grab touch-none items-center justify-center active:cursor-grabbing pointer-coarse:h-7"
        {...bind}
        aria-hidden
      >
        <span className="h-1 w-10 rounded-full bg-foreground/15" />
      </div>
      {children}
    </DrawerDragContext.Provider>
  )
}

function SheetContent({
  className,
  children,
  side = "right",
  showCloseButton = true,
  ...props
}: React.ComponentProps<typeof SheetPrimitive.Content> & {
  side?: "right" | "bottom"
  showCloseButton?: boolean
}) {
  const ref = React.useRef<HTMLDivElement | null>(null)
  const content = (
    <>
      {children}
      {showCloseButton && (
        <SheetPrimitive.Close data-slot="sheet-close" asChild>
          <Button
            variant="ghost"
            className={cn(
              "absolute pointer-coarse:size-11",
              side === "bottom"
                ? "top-5 right-[max(0.75rem,env(safe-area-inset-right))] pointer-coarse:top-2 pointer-coarse:right-[max(0.5rem,env(safe-area-inset-right))]"
                : "top-3 right-3 pointer-coarse:top-2.5 pointer-coarse:right-[max(0.625rem,env(safe-area-inset-right))]"
            )}
            size="icon-sm"
          >
            <XIcon />
            <span className="sr-only">Đóng</span>
          </Button>
        </SheetPrimitive.Close>
      )}
    </>
  )
  return (
    <SheetPortal>
      <SheetOverlay />
      <SheetPrimitive.Content
        ref={ref}
        data-slot="sheet-content"
        data-side={side}
        className={cn(
          "fixed z-50 flex flex-col bg-background text-sm shadow-xl outline-none motion-reduce:animate-none",
          "data-open:animate-in data-open:duration-300 data-closed:animate-out data-closed:duration-200",
          side === "right" &&
            "inset-y-0 right-0 h-full w-full max-w-[30rem] border-l pr-[env(safe-area-inset-right)] data-open:slide-in-from-right data-closed:slide-out-to-right",
          side === "bottom" &&
            "inset-x-0 bottom-0 mx-auto max-h-[min(85dvh,calc(100dvh-env(safe-area-inset-top)-2rem))] w-full rounded-t-2xl border-t pr-[env(safe-area-inset-right)] pb-[env(safe-area-inset-bottom)] pl-[env(safe-area-inset-left)] data-[keyboard=open]:pb-0 sm:max-w-xl sm:border-x data-open:slide-in-from-bottom data-closed:slide-out-to-bottom",
          className
        )}
        {...props}
      >
        {side === "bottom" ? (
          <DrawerShell contentRef={ref}>{content}</DrawerShell>
        ) : (
          content
        )}
      </SheetPrimitive.Content>
    </SheetPortal>
  )
}

function SheetHeader({ className, ...props }: React.ComponentProps<"div">) {
  const drag = React.useContext(DrawerDragContext)
  return (
    <div
      data-slot="sheet-header"
      className={cn(
        "flex flex-col gap-1 p-4",
        drag && "touch-none select-none",
        className
      )}
      {...drag}
      {...props}
    />
  )
}

function SheetTitle({
  className,
  ...props
}: React.ComponentProps<typeof SheetPrimitive.Title>) {
  return (
    <SheetPrimitive.Title
      data-slot="sheet-title"
      className={cn(
        "font-heading text-base font-medium text-foreground",
        className
      )}
      {...props}
    />
  )
}

function SheetDescription({
  className,
  ...props
}: React.ComponentProps<typeof SheetPrimitive.Description>) {
  return (
    <SheetPrimitive.Description
      data-slot="sheet-description"
      className={cn("text-sm text-muted-foreground", className)}
      {...props}
    />
  )
}

export {
  Sheet,
  SheetClose,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
}
