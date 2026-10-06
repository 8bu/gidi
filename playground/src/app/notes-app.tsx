"use client"

import "./theme.css"

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type CSSProperties,
  type RefObject,
} from "react"
import { AnimatePresence, MotionConfig, motion } from "motion/react"

import { TooltipProvider } from "@/components/ui/tooltip"
import { track } from "@/lib/analytics"
import { AppHeader, APP_HEADER_HEIGHT } from "./app-header"
import { FlyProvider } from "./fly-layer"
import { RingLayout } from "./layouts"
import { StackSheet } from "./stack-sheet"
import { StickyComposer } from "./sticky-composer"
import { useNotes } from "./store"
import { useGidi } from "./use-gidi"
import type { TxType } from "./types"

const NOTICE_MS = 4500

const EXAMPLES = [
  "mua sữa vinamilk hết 500k",
  "còn nợ chị Mai 700k",
  "quà sinh nhật bé Na 300k",
  "lương tháng 10 về 15tr",
  "chuyển khoản cho mẹ 2tr",
]

/* ------------------------------------------------------------ keyboard */

/** Share of the full viewport height below which the soft keyboard counts as open. */
const KEYBOARD_RATIO = 0.75

/**
 * Publishes the visible viewport on `root` as `--gidi-vvh` (px) and returns whether a soft
 * keyboard is up (coarse pointer and the visual viewport lost more than a quarter of its height).
 * While the keyboard is open the window is pinned to the top so the layout region is the visible one.
 */
function useSoftKeyboard(root: RefObject<HTMLElement | null>) {
  const [open, setOpen] = useState(false)

  useEffect(() => {
    const el = root.current
    const vv = window.visualViewport
    if (!el) return
    const coarse = window.matchMedia("(pointer: coarse)")
    // Browsers that resize the layout viewport for the keyboard (Android) shrink innerHeight
    // too, so remember the tallest height seen for the current width.
    let width = vv?.width ?? window.innerWidth
    let full = window.innerHeight

    const update = () => {
      const height = vv?.height ?? window.innerHeight
      const nextWidth = vv?.width ?? window.innerWidth
      if (Math.abs(nextWidth - width) > 1) {
        width = nextWidth
        full = window.innerHeight
      }
      full = Math.max(full, window.innerHeight)
      const zoomed = (vv?.scale ?? 1) > 1.01
      const keyboard =
        coarse.matches && !zoomed && height < full * KEYBOARD_RATIO
      if (zoomed) el.style.removeProperty("--gidi-vvh")
      else el.style.setProperty("--gidi-vvh", `${height}px`)
      setOpen(keyboard)
      if (keyboard && (window.scrollY !== 0 || window.scrollX !== 0)) {
        window.scrollTo(0, 0)
      }
    }

    update()
    vv?.addEventListener("resize", update)
    vv?.addEventListener("scroll", update)
    window.addEventListener("resize", update)
    coarse.addEventListener("change", update)
    return () => {
      vv?.removeEventListener("resize", update)
      vv?.removeEventListener("scroll", update)
      window.removeEventListener("resize", update)
      coarse.removeEventListener("change", update)
      el.style.removeProperty("--gidi-vvh")
    }
  }, [root])

  return open
}

/* --------------------------------------------------------------- notice */

function Notice({
  message,
  onDone,
}: {
  message: string | null
  onDone: () => void
}) {
  useEffect(() => {
    if (message === null) return
    const timer = window.setTimeout(onDone, NOTICE_MS)
    return () => window.clearTimeout(timer)
  }, [message, onDone])

  return (
    <div
      className="pointer-events-none fixed inset-x-0 z-40 flex justify-center pr-[max(1rem,env(safe-area-inset-right))] pl-[max(1rem,env(safe-area-inset-left))]"
      style={{ top: APP_HEADER_HEIGHT }}
    >
      <AnimatePresence>
        {message !== null && (
          <motion.p
            key={message}
            role="status"
            initial={{ opacity: 0, y: -12, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ type: "spring", stiffness: 420, damping: 34 }}
            className="pointer-events-auto rounded-lg bg-foreground px-4 py-2 text-sm text-background shadow-lg"
          >
            {message}
          </motion.p>
        )}
      </AnimatePresence>
    </div>
  )
}

/* -------------------------------------------------------------- empty */

function ExampleChips({ onPick }: { onPick: (text: string) => void }) {
  return (
    <div className="mt-6 flex flex-col items-center gap-2.5 max-sm:mt-3 max-sm:gap-1.5 [@media(max-height:500px)]:hidden">
      <p className="text-xs text-muted-foreground max-sm:sr-only">
        Chưa có ghi chú nào. Thử một ví dụ:
      </p>
      <ul className="flex flex-wrap justify-center gap-2 max-sm:w-full max-sm:[scrollbar-width:none] max-sm:flex-nowrap max-sm:justify-start max-sm:overflow-x-auto max-sm:px-1 max-sm:py-1">
        {EXAMPLES.map((text, i) => (
          <motion.li
            key={text}
            className="max-sm:shrink-0"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{
              type: "spring",
              stiffness: 300,
              damping: 26,
              delay: 0.25 + i * 0.05,
            }}
          >
            <button
              type="button"
              onClick={() => onPick(text)}
              className="inline-flex items-center rounded-full border bg-card/85 px-3 py-1.5 text-xs text-card-foreground shadow-xs transition-[transform,background-color] outline-none hover:-translate-y-0.5 hover:bg-card focus-visible:ring-3 focus-visible:ring-ring/50 active:translate-y-0 active:scale-[0.98] pointer-coarse:min-h-11 pointer-coarse:px-4 pointer-coarse:text-[13px]"
            >
              {text}
            </button>
          </motion.li>
        ))}
      </ul>
    </div>
  )
}

/* ----------------------------------------------------------------- page */

function Page({ notify }: { notify: (message: string) => void }) {
  const { notes, ready, summaries, exportJson, importJson } = useNotes()
  const gidi = useGidi()
  const [openType, setOpenType] = useState<TxType | null>(null)
  const [draft, setDraft] = useState("")
  const inputRef = useRef<HTMLTextAreaElement | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    const previousLang = document.documentElement.lang
    const previousTitle = document.title
    document.documentElement.lang = "vi"
    document.title = "Gidi"
    return () => {
      document.documentElement.lang = previousLang
      document.title = previousTitle
    }
  }, [])

  const exportNotes = () => {
    const blob = new Blob([exportJson()], { type: "application/json" })
    const url = URL.createObjectURL(blob)
    const link = document.createElement("a")
    link.href = url
    link.download = `gidi-ghi-chu-${new Date().toISOString().slice(0, 10)}.json`
    link.click()
    window.setTimeout(() => URL.revokeObjectURL(url), 1000)
    track("notes_exported", { count: notes.length })
    notify(`Đã xuất ${notes.length} ghi chú`)
  }

  const importFile = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    event.target.value = ""
    if (!file) return
    file
      .text()
      .then((text) => importJson(text))
      .then(
        (count) => {
          track("notes_imported", { count })
          notify(
            count === 0
              ? "Không có ghi chú mới trong tệp này."
              : `Đã nhập ${count} ghi chú`
          )
        },
        (e: unknown) =>
          notify(e instanceof Error ? e.message : "Không nhập được tệp này.")
      )
  }

  const rootRef = useRef<HTMLDivElement>(null)
  const keyboardOpen = useSoftKeyboard(rootRef)

  const pickExample = (text: string) => {
    setDraft(text)
    inputRef.current?.focus()
  }

  const empty = ready && notes.length === 0

  const composer = (
    <div>
      <StickyComposer
        ref={inputRef}
        value={draft}
        onValueChange={setDraft}
        classify={gidi.classify}
        status={gidi.status}
        progress={gidi.progress}
        error={gidi.error}
        onRetry={gidi.retry}
      />
      {empty && <ExampleChips onPick={pickExample} />}
    </div>
  )

  return (
    <div
      ref={rootRef}
      data-keyboard={keyboardOpen ? "open" : "closed"}
      className="gidi-desk group/app relative min-h-[var(--gidi-vvh,100dvh)] overflow-x-clip pr-[env(safe-area-inset-right)] pb-[var(--gidi-safe-bottom)] pl-[env(safe-area-inset-left)] text-foreground [--gidi-safe-bottom:env(safe-area-inset-bottom)] data-[keyboard=open]:[--gidi-safe-bottom:0px]"
      style={{ "--gidi-topbar": APP_HEADER_HEIGHT } as CSSProperties}
    >
      <AppHeader
        summaries={summaries}
        noteCount={notes.length}
        model={{
          status: gidi.status,
          progress: gidi.progress,
          retry: gidi.retry,
        }}
        onExport={exportNotes}
        onImport={() => fileRef.current?.click()}
      />
      <input
        ref={fileRef}
        type="file"
        accept="application/json,.json"
        className="hidden"
        onChange={importFile}
        tabIndex={-1}
        aria-hidden
      />

      <motion.main
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.25 }}
      >
        <RingLayout
          summaries={summaries}
          composer={composer}
          onOpenStack={(type) => {
            track("stack_opened", { type })
            setOpenType(type)
          }}
        />
      </motion.main>

      <StackSheet type={openType} onClose={() => setOpenType(null)} />
    </div>
  )
}

export function NotesApp() {
  const [notice, setNotice] = useState<string | null>(null)
  const clearNotice = useCallback(() => setNotice(null), [])

  return (
    <TooltipProvider>
      <MotionConfig reducedMotion="user">
        <FlyProvider onError={setNotice}>
          <Page notify={setNotice} />
        </FlyProvider>
        <Notice message={notice} onDone={clearNotice} />
      </MotionConfig>
    </TooltipProvider>
  )
}
