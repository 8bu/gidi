"use client"

import { useEffect, type CSSProperties, type ReactNode } from "react"
import {
  AnimatePresence,
  MotionConfig,
  animate,
  motion,
  useMotionValue,
  useReducedMotion,
  useTransform,
} from "motion/react"
import { DropdownMenu } from "radix-ui"
import {
  BoxIcon,
  CheckIcon,
  CodeIcon,
  DownloadIcon,
  EllipsisIcon,
  FlaskConicalIcon,
  GlobeIcon,
  MonitorIcon,
  MoonIcon,
  RotateCcwIcon,
  ShieldCheckIcon,
  SunIcon,
  TriangleAlertIcon,
  UploadIcon,
} from "lucide-react"

import { useTheme } from "@/components/theme-provider"
import { track } from "@/lib/analytics"
import { cn } from "@/lib/utils"
import { formatVnd, formatVndCompact } from "./amount"
import { TYPE_META } from "./taxonomy"
import { TX_TYPES, type StackSummary, type TxType } from "./types"

/**
 * Cash in and out over every type by its direction: Thu = income, borrow, repayment_in, refund;
 * Chi = expense, lend, repayment_out. Transfers move money between own accounts and are left out.
 */
function cashFlow(summaries: Record<TxType, StackSummary>): {
  income: number
  expense: number
} {
  let income = 0
  let expense = 0
  for (const type of TX_TYPES) {
    const { sign } = TYPE_META[type]
    if (sign > 0) income += summaries[type].total
    else if (sign < 0) expense += summaries[type].total
  }
  return { income, expense }
}

export interface HeaderProps {
  summaries: Record<TxType, StackSummary>
  noteCount: number
  model: {
    status: "loading" | "ready" | "error"
    /** Download progress 0..1, null when unknown. */
    progress: number | null
    retry(): void
  }
  onExport(): void
  onImport(): void
}

/*
 * No bar at all. The desk stays empty; a wordmark sits top-left, a small paper slip floats
 * top-centre (loading, totals or error, one state at a time), and a single round paper button
 * on the right opens everything else.
 */

const EASE_OUT = [0.22, 1, 0.36, 1] as const
const PILL_SPRING = { type: "spring", stiffness: 420, damping: 38 } as const

/** White-ish slip on the desk: `.gidi-sheet` painted with the card colour. */
const SLIP_VARS = {
  "--sheet-paper": "var(--card)",
  "--sheet-ink": "var(--foreground)",
} as CSSProperties

/* ---------------------------------------------------------------- numbers */

/** Counts to `value` when it changes (no animation on first paint or reduced motion). */
function Tick({
  value,
  format,
}: {
  value: number
  format: (n: number) => string
}) {
  const reduce = useReducedMotion()
  const motionValue = useMotionValue(value)
  const text = useTransform(motionValue, (n) => format(n))

  useEffect(() => {
    if (reduce) {
      motionValue.set(value)
      return
    }
    const controls = animate(motionValue, value, {
      duration: 0.7,
      ease: EASE_OUT,
    })
    return () => controls.stop()
  }, [value, reduce, motionValue])

  return (
    <>
      <span className="sr-only">{formatVnd(value)}</span>
      <span aria-hidden>
        <motion.span>{text}</motion.span>
      </span>
    </>
  )
}

function Percent({ value }: { value: number }) {
  return <Tick value={value} format={(n) => `${Math.round(n)}%`} />
}

/* ------------------------------------------------------------- pill states */

function Column({
  label,
  value,
  color,
  strong,
}: {
  label: string
  value: number
  color?: string
  strong?: boolean
}) {
  return (
    <div className="flex min-w-0 flex-col items-center px-2.5 leading-none sm:px-4 [&:not(:first-child)]:border-l [&:not(:first-child)]:border-dashed [&:not(:first-child)]:border-foreground/20">
      <dt className="text-[11px] font-medium text-foreground/70">{label}</dt>
      <dd
        className={cn(
          "mt-1 text-[13px] tabular-nums sm:text-sm",
          strong ? "font-bold" : "font-semibold"
        )}
        style={color ? { color } : undefined}
      >
        <Tick value={value} format={formatVndCompact} />
      </dd>
    </div>
  )
}

function Totals({ income, expense }: { income: number; expense: number }) {
  const net = income - expense
  return (
    <dl
      aria-label="Tổng hợp thu chi"
      className="m-0 flex items-center"
      title={`Thu ${formatVnd(income)}  ·  Chi ${formatVnd(expense)}  ·  Còn lại ${formatVnd(net)}`}
    >
      <Column label="Thu" value={income} color="var(--ink-income)" />
      <Column label="Chi" value={expense} color="var(--ink-expense)" />
      <Column
        label="Còn lại"
        value={net}
        color={net < 0 ? "var(--ink-expense)" : undefined}
        strong
      />
    </dl>
  )
}

function Idle() {
  return (
    <p className="flex items-center gap-2 px-1 text-xs font-medium text-foreground/75 sm:text-[13px]">
      <ShieldCheckIcon className="size-4 shrink-0" aria-hidden />
      <span className="truncate">Chạy trên thiết bị</span>
    </p>
  )
}

function Loading({ progress }: { progress: number | null }) {
  const percent = progress === null ? null : Math.round(progress * 100)
  return (
    <div className="flex items-center gap-3 px-1 sm:min-w-[22rem]">
      <div className="min-w-0 flex-1 leading-tight">
        <p
          role="status"
          className="truncate text-xs font-medium sm:text-[13px]"
        >
          <span className="sm:hidden">Đang tải mô hình 29 MB</span>
          <span className="hidden sm:inline">Đang tải mô hình, 29 MB</span>
        </p>
        <p className="hidden truncate text-[11px] text-foreground/70 sm:block">
          Chỉ tải một lần, sau đó chạy ngay trên thiết bị
        </p>
      </div>
      {percent !== null && (
        <span className="shrink-0 text-sm font-semibold tabular-nums">
          <Percent value={percent} />
        </span>
      )}
    </div>
  )
}

function ProgressFill({ progress }: { progress: number | null }) {
  const reduce = useReducedMotion()
  return (
    <div
      aria-hidden
      className="pointer-events-none absolute inset-0 overflow-hidden rounded-[inherit]"
    >
      {progress === null ? (
        <motion.div
          className="absolute inset-y-0 left-0 w-2/5"
          style={{ background: "var(--paper-note)" }}
          initial={{ x: "-100%" }}
          animate={reduce ? { x: "0%" } : { x: ["-100%", "250%"] }}
          transition={
            reduce
              ? { duration: 0 }
              : { duration: 1.6, ease: "easeInOut", repeat: Infinity }
          }
        />
      ) : (
        <motion.div
          className="absolute inset-0 origin-left border-r border-[color-mix(in_oklab,var(--ink-note)_35%,transparent)]"
          style={{ background: "var(--paper-note)" }}
          initial={false}
          animate={{ scaleX: Math.max(0.04, Math.min(1, progress)) }}
          transition={{ duration: reduce ? 0 : 0.35, ease: EASE_OUT }}
        />
      )}
    </div>
  )
}

function Failed({ onRetry }: { onRetry: () => void }) {
  return (
    <div className="flex items-center gap-2 pl-1">
      <TriangleAlertIcon
        className="size-4 shrink-0 text-destructive"
        aria-hidden
      />
      <p
        role="alert"
        className="truncate text-xs font-medium text-destructive sm:text-[13px]"
      >
        <span className="sm:hidden">Không tải được</span>
        <span className="hidden sm:inline">Không tải được mô hình</span>
      </p>
      <button
        type="button"
        onClick={onRetry}
        className="inline-flex h-7 shrink-0 items-center gap-1 rounded-full bg-destructive/10 px-2.5 text-xs font-semibold text-destructive transition-colors outline-none hover:bg-destructive/20 focus-visible:ring-2 focus-visible:ring-ring motion-reduce:transition-none pointer-coarse:h-11 pointer-coarse:px-3.5"
      >
        <RotateCcwIcon className="size-3" aria-hidden />
        Thử lại
      </button>
    </div>
  )
}

function Pill({
  summaries,
  noteCount,
  model,
}: Pick<HeaderProps, "summaries" | "noteCount" | "model">) {
  const state =
    model.status === "loading"
      ? "loading"
      : model.status === "error"
        ? "error"
        : noteCount > 0
          ? "totals"
          : "idle"

  let content: ReactNode
  if (state === "loading") content = <Loading progress={model.progress} />
  else if (state === "error") content = <Failed onRetry={model.retry} />
  else if (state === "totals") content = <Totals {...cashFlow(summaries)} />
  else content = <Idle />

  return (
    <motion.div
      layout
      transition={{ layout: PILL_SPRING }}
      className="gidi-sheet relative flex h-11 max-w-full min-w-0 items-center px-3 sm:px-4"
      style={{
        ...SLIP_VARS,
        borderRadius: 9999,
        boxShadow: "var(--paper-shadow-2)",
        ...(state === "error"
          ? {
              borderColor:
                "color-mix(in oklab, var(--destructive) 45%, transparent)",
            }
          : null),
      }}
    >
      {state === "loading" && <ProgressFill progress={model.progress} />}
      <AnimatePresence mode="wait" initial={false}>
        <motion.div
          key={state}
          layout="position"
          className="relative min-w-0"
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -4 }}
          transition={{ duration: 0.16, ease: EASE_OUT }}
        >
          {content}
        </motion.div>
      </AnimatePresence>
    </motion.div>
  )
}

/* --------------------------------------------------------------- wordmark */

function Wordmark() {
  return (
    <h1 className="flex items-center gap-2 text-xl leading-none font-extrabold tracking-tighter select-none sm:text-2xl">
      {/* A small sticky note, slightly off-square, like it was just stuck down. */}
      <span
        aria-hidden
        className="gidi-sheet hidden size-[18px] -rotate-6 rounded-[3px] min-[420px]:block sm:size-5"
        style={{ borderRadius: 4 }}
      />
      Gidi
    </h1>
  )
}

/* ------------------------------------------------------------------- menu */

const ITEM =
  "relative flex cursor-default items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm outline-none select-none data-[highlighted]:bg-accent data-[highlighted]:text-accent-foreground pointer-coarse:min-h-11 [&_svg]:size-4 [&_svg]:shrink-0 [&_svg]:text-muted-foreground"

const THEMES = [
  { value: "light", label: "Sáng", Icon: SunIcon },
  { value: "dark", label: "Tối", Icon: MoonIcon },
  { value: "system", label: "Theo hệ thống", Icon: MonitorIcon },
] as const

function Menu({
  onExport,
  onImport,
}: Pick<HeaderProps, "onExport" | "onImport">) {
  const { theme, setTheme } = useTheme()
  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        <button
          type="button"
          aria-label="Menu"
          className="gidi-sheet inline-flex size-10 items-center justify-center transition-[transform,box-shadow] duration-150 outline-none hover:-translate-y-px focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background active:translate-y-0 active:scale-[0.96] data-[state=open]:shadow-none motion-reduce:transition-none motion-reduce:hover:translate-y-0 motion-reduce:active:scale-100 pointer-coarse:size-11"
          style={{
            ...SLIP_VARS,
            borderRadius: 9999,
            boxShadow: "var(--paper-shadow-2)",
          }}
        >
          <EllipsisIcon className="size-5" aria-hidden />
        </button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content
          align="end"
          sideOffset={8}
          collisionPadding={12}
          className="z-50 w-64 max-w-[calc(100vw-1.5rem)] rounded-2xl bg-popover p-1.5 text-popover-foreground shadow-[var(--paper-shadow-3)] ring-1 ring-foreground/10 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95 data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95 motion-reduce:animate-none"
        >
          <DropdownMenu.Label className="px-2.5 pt-1.5 pb-1 text-xs font-medium text-muted-foreground">
            Giao diện
          </DropdownMenu.Label>
          <DropdownMenu.RadioGroup
            value={theme}
            onValueChange={(value) => {
              if (value === "light" || value === "dark" || value === "system") {
                setTheme(value)
              }
            }}
          >
            {THEMES.map(({ value, label, Icon }) => (
              <DropdownMenu.RadioItem
                key={value}
                value={value}
                className={ITEM}
              >
                <Icon aria-hidden />
                {label}
                <DropdownMenu.ItemIndicator className="ml-auto">
                  <CheckIcon aria-hidden className="!text-foreground" />
                </DropdownMenu.ItemIndicator>
              </DropdownMenu.RadioItem>
            ))}
          </DropdownMenu.RadioGroup>
          <DropdownMenu.Separator className="mx-1 my-1.5 h-px bg-border" />
          <DropdownMenu.Item className={ITEM} onSelect={onExport}>
            <DownloadIcon aria-hidden />
            Xuất ghi chú (JSON)
          </DropdownMenu.Item>
          <DropdownMenu.Item className={ITEM} onSelect={onImport}>
            <UploadIcon aria-hidden />
            Nhập ghi chú (JSON)
          </DropdownMenu.Item>
          <DropdownMenu.Separator className="mx-1 my-1.5 h-px bg-border" />
          <DropdownMenu.Item className={ITEM} asChild>
            <a href="/lab">
              <FlaskConicalIcon aria-hidden />
              Phòng thử nghiệm
            </a>
          </DropdownMenu.Item>
          <DropdownMenu.Separator className="mx-1 my-1.5 h-px bg-border" />
          <DropdownMenu.Item className={ITEM} asChild>
            <a
              href="https://github.com/8bu/gidi"
              target="_blank"
              rel="noopener"
              onClick={() => track("link_clicked", { link: "github" })}
            >
              <CodeIcon aria-hidden />
              Mã nguồn (GitHub)
            </a>
          </DropdownMenu.Item>
          <DropdownMenu.Item className={ITEM} asChild>
            <a
              href="https://huggingface.co/x8bu/gidi-finance"
              target="_blank"
              rel="noopener"
              onClick={() => track("link_clicked", { link: "huggingface" })}
            >
              <BoxIcon aria-hidden />
              Mô hình (Hugging Face)
            </a>
          </DropdownMenu.Item>
          <DropdownMenu.Item className={ITEM} asChild>
            <a
              href="https://8bu.dev"
              target="_blank"
              rel="noopener"
              onClick={() => track("link_clicked", { link: "author" })}
            >
              <GlobeIcon aria-hidden />
              Tác giả: 8bu.dev
            </a>
          </DropdownMenu.Item>
          <DropdownMenu.Separator className="mx-1 my-1.5 h-px bg-border" />
          <p className="flex items-start gap-2.5 px-2.5 pt-1 pb-1.5 text-xs leading-snug text-muted-foreground">
            <ShieldCheckIcon className="mt-px size-4 shrink-0" aria-hidden />
            Mô hình chạy trên thiết bị. Ghi chú không rời khỏi máy. Chỉ đếm lượt
            dùng ẩn danh, không có nội dung, tên hay số tiền.
          </p>
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  )
}

/* ----------------------------------------------------------------- header */

export function AppHeader({
  summaries,
  noteCount,
  model,
  onExport,
  onImport,
}: HeaderProps) {
  return (
    <MotionConfig reducedMotion="user">
      <motion.header
        initial={{ opacity: 0, y: -6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35, ease: EASE_OUT }}
        className="relative z-20 grid h-[calc(4rem+env(safe-area-inset-top))] grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-2 px-3 pt-[env(safe-area-inset-top)] sm:grid-cols-[1fr_auto_1fr] sm:gap-4 sm:px-6"
      >
        <div className="justify-self-start">
          <Wordmark />
        </div>
        <div className="flex min-w-0 justify-center">
          <Pill summaries={summaries} noteCount={noteCount} model={model} />
        </div>
        <div className="justify-self-end">
          <Menu onExport={onExport} onImport={onImport} />
        </div>
      </motion.header>
    </MotionConfig>
  )
}

/** Vertical space the ring layout leaves for the header. */
export const APP_HEADER_HEIGHT = "calc(4rem + env(safe-area-inset-top))"
