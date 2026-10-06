"use client"

import {
  useEffect,
  useId,
  useRef,
  useState,
  type KeyboardEvent,
  type Ref,
} from "react"
import { animate, useAnimate, useReducedMotion } from "motion/react"
import { RotateCcwIcon, SendHorizontalIcon } from "lucide-react"

import { Button } from "@/components/ui/button"
import { useCoarsePointer } from "@/hooks/use-coarse-pointer"
import { track } from "@/lib/analytics"
import { cn } from "@/lib/utils"
import { useFly } from "./fly-layer"
import type { NoteRecord, TxType } from "./types"

export interface StickyComposerProps {
  value: string
  onValueChange(text: string): void
  classify(text: string): Promise<NoteRecord>
  status: "loading" | "ready" | "error"
  /** 0..1 while the model downloads, `null` when unknown. */
  progress: number | null
  error: string | null
  onRetry(): void
  ref?: Ref<HTMLTextAreaElement>
}

/** Only show "reading" if classify takes longer than this, so a fast reply never flashes. */
const READING_DELAY_MS = 220
/** Gap between cards when one send holds several lines, so each one reads as its own flight. */
const FLIGHT_STAGGER_MS = 110
const LINE = "1.75rem"
const INK_LINE = "color-mix(in oklab, var(--ink-note) 15%, transparent)"

export function StickyComposer({
  value,
  onValueChange,
  classify,
  status,
  progress,
  error,
  onRetry,
  ref,
}: StickyComposerProps) {
  const fly = useFly()
  const coarse = useCoarsePointer()
  const reduced = useReducedMotion() ?? false
  const [scope, animateScope] = useAnimate<HTMLDivElement>()
  const paperRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement | null>(null)
  const valueRef = useRef(value)
  const sending = useRef(false)
  const [reading, setReading] = useState(false)
  const [message, setMessage] = useState<string | null>(null)
  const messageId = useId()

  useEffect(() => {
    valueRef.current = value
  })

  const setTextarea = (el: HTMLTextAreaElement | null) => {
    textareaRef.current = el
    if (typeof ref === "function") ref(el)
    else if (ref) ref.current = el
  }

  const complain = (text: string) => {
    setMessage(text)
    if (!reduced) {
      void animateScope(
        scope.current,
        { x: [0, -8, 7, -5, 3, 0] },
        { duration: 0.42, ease: "easeOut" }
      )
    }
  }

  /** The old sheet has left with the fly layer; a fresh one slides up from beneath the pad. */
  const peelFresh = () => {
    const paper = paperRef.current
    if (!paper) return
    if (reduced) {
      void animate(paper, { opacity: [0, 1] }, { duration: 0.2 })
      return
    }
    void animate(
      paper,
      { y: [20, 0], scale: [0.96, 1], opacity: [0, 1] },
      {
        y: { type: "spring", stiffness: 420, damping: 32 },
        scale: { type: "spring", stiffness: 420, damping: 32 },
        opacity: { duration: 0.16 },
      }
    )
  }

  const send = async () => {
    if (sending.current) return
    const text = value.trim()
    if (text === "") {
      complain("Hãy viết một dòng ghi chú trước khi gửi.")
      return
    }
    if (status === "error") {
      complain(error ?? "Mô hình chưa chạy được. Hãy thử tải lại.")
      return
    }
    if (status === "loading") {
      complain("Mô hình đang tải. Gửi được ngay khi tải xong.")
      return
    }
    // Focus inside the user gesture so the soft keyboard stays up for the next note.
    if (coarse) textareaRef.current?.focus({ preventScroll: true })

    sending.current = true
    const timer = window.setTimeout(() => setReading(true), READING_DELAY_MS)
    try {
      // One line is one note: a pasted list becomes one record per line.
      const lines = text
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter((line) => line !== "")
      const records: NoteRecord[] = []
      const failed: string[] = []
      let firstError: unknown = null
      for (const line of lines) {
        try {
          records.push(await classify(line))
        } catch (e) {
          failed.push(line)
          firstError ??= e
        }
      }
      const types: Partial<Record<TxType, number>> = {}
      for (const record of records)
        types[record.type] = (types[record.type] ?? 0) + 1
      track("notes_sent", {
        lines: lines.length,
        classified: records.length,
        failed: failed.length,
        with_amount: records.filter((r) => r.amount !== null).length,
        with_target: records.filter((r) => r.target !== null).length,
        types,
      })
      const rect = paperRef.current?.getBoundingClientRect()
      if (rect) {
        records.forEach((record, index) => {
          if (index === 0) fly(record, rect)
          else
            window.setTimeout(
              () => fly(record, rect),
              index * FLIGHT_STAGGER_MS
            )
        })
      }
      // Lines the model could not read stay on the pad so nothing typed is lost.
      if (valueRef.current.trim() === text) onValueChange(failed.join("\n"))
      if (records.length > 0) {
        setMessage(null)
        peelFresh()
      }
      textareaRef.current?.focus()
      if (firstError !== null) throw firstError
    } catch (e) {
      complain(
        e instanceof Error
          ? e.message
          : "Không đọc được ghi chú này. Thử lại nhé."
      )
    } finally {
      window.clearTimeout(timer)
      setReading(false)
      sending.current = false
    }
  }

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== "Enter" || event.shiftKey) return
    // Vietnamese IMEs use Enter to commit a composition; that must not send.
    // Touch keyboards send "Enter" too (enterKeyHint="send"), so this path serves them as well.
    if (event.nativeEvent.isComposing || event.keyCode === 229) return
    event.preventDefault()
    void send()
  }

  const percent = progress === null ? null : Math.round(progress * 100)
  const blocked = status !== "ready"

  return (
    <div ref={scope} className="relative w-full">
      {/* The pad: two sheets peeking out from under the top one. */}
      <div
        aria-hidden
        className="gidi-sheet absolute inset-0 translate-y-2 -rotate-[1.8deg] opacity-80"
        style={{ boxShadow: "var(--paper-shadow-1)" }}
      />
      <div
        aria-hidden
        className="gidi-sheet absolute inset-0 translate-y-1 rotate-[1.2deg]"
        style={{ boxShadow: "var(--paper-shadow-1)" }}
      />

      <div
        ref={paperRef}
        className="gidi-sheet relative flex min-h-[14rem] flex-col px-5 pt-6 pb-4 group-data-[keyboard=open]/app:min-h-0 group-data-[keyboard=open]/app:px-4 group-data-[keyboard=open]/app:pt-3 group-data-[keyboard=open]/app:pb-2.5 sm:min-h-[17rem] [@media(max-height:500px)]:min-h-0 [@media(max-height:500px)]:pt-4 [@media(max-height:500px)]:pb-3"
        style={{ boxShadow: "var(--paper-shadow-3)" }}
      >
        <label
          htmlFor={`${messageId}-note`}
          className="mb-1 text-xs font-medium tracking-wide opacity-60 group-data-[keyboard=open]/app:mb-0"
        >
          Ghi chú mới
        </label>

        <textarea
          id={`${messageId}-note`}
          ref={setTextarea}
          value={value}
          onChange={(event) => {
            onValueChange(event.target.value)
            if (message !== null) setMessage(null)
          }}
          onKeyDown={onKeyDown}
          rows={4}
          spellCheck={false}
          autoComplete="off"
          enterKeyHint="send"
          aria-invalid={message !== null}
          aria-describedby={message !== null ? messageId : undefined}
          placeholder={"mua sữa vinamilk hết 500k\nmỗi dòng là một ghi chú"}
          className="min-h-28 w-full flex-1 resize-none bg-transparent text-[max(1.05rem,16px)] outline-none group-data-[keyboard=open]/app:h-14 group-data-[keyboard=open]/app:min-h-14 group-data-[keyboard=open]/app:flex-none placeholder:text-[color-mix(in_oklab,var(--ink-note)_42%,transparent)] [@media(max-height:500px)]:h-[5.25rem] [@media(max-height:500px)]:min-h-[5.25rem] [@media(max-height:500px)]:flex-none"
          style={{
            lineHeight: LINE,
            backgroundImage: `repeating-linear-gradient(to bottom, transparent 0, transparent calc(${LINE} - 1px), ${INK_LINE} calc(${LINE} - 1px), ${INK_LINE} ${LINE})`,
            backgroundAttachment: "local",
          }}
        />

        {status === "loading" && (
          <div className="mt-2" role="status">
            <div className="h-[3px] overflow-hidden rounded-full bg-[color-mix(in_oklab,var(--ink-note)_14%,transparent)]">
              <div
                className={cn(
                  "h-full rounded-full bg-[var(--ink-note)] transition-[width] duration-300",
                  percent === null && "w-1/3 animate-pulse"
                )}
                style={
                  percent === null
                    ? undefined
                    : { width: `${Math.max(percent, 4)}%` }
                }
              />
            </div>
            <p className="mt-1.5 text-xs opacity-70">
              Mô hình 29 MB đang tải về máy bạn
              {percent === null ? "" : ` (${percent}%)`}. Bạn cứ viết sẵn, mọi
              thứ chạy ngay trên thiết bị.
            </p>
          </div>
        )}

        <div className="mt-3 flex items-center justify-between gap-3 group-data-[keyboard=open]/app:mt-2">
          <div className="min-w-0 text-xs">
            {message !== null ? (
              <p
                id={messageId}
                role="alert"
                className="flex items-center gap-1.5 font-medium text-[var(--ink-expense)]"
              >
                {message}
                {status === "error" && (
                  <button
                    type="button"
                    onClick={onRetry}
                    className="inline-flex items-center gap-1 rounded-sm underline underline-offset-2 outline-none focus-visible:ring-2 focus-visible:ring-current pointer-coarse:min-h-11 pointer-coarse:px-1"
                  >
                    <RotateCcwIcon className="size-3" />
                    Tải lại
                  </button>
                )}
              </p>
            ) : status === "error" ? (
              <button
                type="button"
                onClick={onRetry}
                className="inline-flex items-center gap-1 rounded-sm font-medium text-[var(--ink-expense)] underline underline-offset-2 outline-none focus-visible:ring-2 focus-visible:ring-current pointer-coarse:min-h-11"
              >
                <RotateCcwIcon className="size-3" />
                Không tải được mô hình. Thử lại
              </button>
            ) : (
              <p className="opacity-60">
                {coarse
                  ? "Chạm Gửi để cất ghi chú"
                  : "Enter để gửi, Shift + Enter xuống dòng"}
              </p>
            )}
          </div>

          <Button
            type="button"
            size="lg"
            aria-disabled={blocked}
            // Keep focus (and the soft keyboard) in the textarea when the button is pressed.
            onMouseDown={(event) => event.preventDefault()}
            onClick={() => void send()}
            className={cn(
              "shrink-0 gap-2 px-3.5 transition-opacity pointer-coarse:h-11 pointer-coarse:min-w-20 pointer-coarse:px-4",
              blocked && "opacity-50 hover:bg-primary",
              reading && "opacity-80"
            )}
          >
            {reading ? "Đang đọc" : "Gửi"}
            <SendHorizontalIcon data-icon="inline-end" />
          </Button>
        </div>
      </div>
    </div>
  )
}
