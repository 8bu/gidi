"use client"

import { useId, useRef, useState, type FocusEvent, type FormEvent } from "react"
import { RotateCcwIcon } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { cn } from "@/lib/utils"
import { amountToVnd, formatVnd, formatVndCompact } from "./amount"
import { TYPE_META } from "./taxonomy"
import { TX_TYPES, type NotePatch, type NoteRecord, type TxType } from "./types"

/** Whether the user's current values differ from what the model read. */
export function isEdited(record: NoteRecord): boolean {
  const { original } = record
  return (
    record.type !== original.type ||
    (record.target ?? null) !== (original.target ?? null) ||
    record.amount !== original.amount
  )
}

/** Patch that puts a record back to the model's reading. */
export function restorePatch(record: NoteRecord): NotePatch {
  const { original } = record
  return {
    type: original.type,
    target: original.target,
    amount: original.amount,
  }
}

type ParsedAmount = { ok: true; value: number | null } | { ok: false }

/** Plain digits and grouped thousands ("1.200.000") read directly; "500k", "1,2tr" go through the model's parser. */
export function parseAmountInput(text: string): ParsedAmount {
  const t = text.trim()
  if (t === "") return { ok: true, value: null }
  if (/^\d+$/.test(t) || /^\d{1,3}([.,]\d{3})+$/.test(t)) {
    const value = Number(t.replace(/[.,]/g, ""))
    return Number.isSafeInteger(value) ? { ok: true, value } : { ok: false }
  }
  const value = amountToVnd(t)
  return value === null ? { ok: false } : { ok: true, value }
}

/** Small square of the type's paper colour. */
export function TypeSwatch({
  type,
  className,
}: {
  type: TxType
  className?: string
}) {
  return (
    <span
      aria-hidden
      className={cn("inline-block size-3 shrink-0 rounded-[3px]", className)}
      style={{
        background: `var(--paper-${type})`,
        boxShadow: `inset 0 0 0 1px color-mix(in oklab, var(--ink-${type}) 45%, transparent)`,
      }}
    />
  )
}

export interface RecordEditorProps {
  record: NoteRecord
  onSave(patch: NotePatch): void
  onCancel(): void
  onRestore(): void
}

/**
 * On a touch device the keyboard hides the lower part of the screen: once the
 * visual viewport has resized, bring the editor and the focused field back in.
 */
function revealOnFocus(form: HTMLFormElement, field: HTMLElement) {
  if (!window.matchMedia("(pointer: coarse)").matches) return
  const reveal = () => {
    form.scrollIntoView({ block: "nearest" })
    field.scrollIntoView({ block: "nearest" })
  }
  reveal()
  const vv = window.visualViewport
  if (!vv) return
  let settled = false
  const settle = () => {
    if (settled) return
    settled = true
    vv.removeEventListener("resize", settle)
    window.clearTimeout(fallback)
    reveal()
  }
  // Fallback when the keyboard was already up and no resize follows.
  const fallback = window.setTimeout(settle, 450)
  vv.addEventListener("resize", settle)
}

export function RecordEditor({
  record,
  onSave,
  onCancel,
  onRestore,
}: RecordEditorProps) {
  const formRef = useRef<HTMLFormElement>(null)
  const [amountText, setAmountText] = useState(
    record.amount === null ? "" : String(record.amount)
  )
  const [target, setTarget] = useState(record.target ?? "")
  const [type, setType] = useState<TxType>(record.type)
  const id = useId()

  const parsed = parseAmountInput(amountText)
  const amountInvalid = !parsed.ok

  const submit = (event: FormEvent) => {
    event.preventDefault()
    if (!parsed.ok) return
    const nextTarget = target.trim() === "" ? null : target.trim()
    const patch: NotePatch = {}
    if (type !== record.type) patch.type = type
    if (nextTarget !== record.target) patch.target = nextTarget
    if (parsed.value !== record.amount) patch.amount = parsed.value
    if (Object.keys(patch).length === 0) onCancel()
    else onSave(patch)
  }

  const onFocus = (event: FocusEvent<HTMLFormElement>) => {
    const field = event.target
    if (formRef.current && field instanceof HTMLInputElement)
      revealOnFocus(formRef.current, field)
  }

  return (
    <form
      ref={formRef}
      onSubmit={submit}
      onFocus={onFocus}
      className="mt-3 grid gap-3 border-t border-dashed border-foreground/15 pt-3"
      aria-label="Sửa ghi chú"
    >
      <div className="grid gap-1.5">
        <Label htmlFor={`${id}-amount`}>Số tiền</Label>
        <Input
          id={`${id}-amount`}
          value={amountText}
          onChange={(event) => setAmountText(event.target.value)}
          placeholder="500k, 1,2tr hoặc 500000"
          autoComplete="off"
          autoFocus
          inputMode="text"
          enterKeyHint="done"
          aria-invalid={amountInvalid}
          aria-describedby={`${id}-amount-preview`}
          className="h-9 tabular-nums pointer-coarse:h-11 pointer-coarse:text-base!"
        />
        <p
          id={`${id}-amount-preview`}
          aria-live="polite"
          className={cn(
            "min-h-4 text-xs tabular-nums",
            amountInvalid ? "text-destructive" : "text-muted-foreground"
          )}
        >
          {!parsed.ok
            ? "Chưa đọc được số tiền. Thử 500k, 1,2tr hoặc 500000."
            : parsed.value === null
              ? "Để trống nếu ghi chú không có số tiền."
              : `${formatVndCompact(parsed.value)}  ·  ${formatVnd(parsed.value)}`}
        </p>
      </div>

      <div className="grid gap-1.5">
        <Label htmlFor={`${id}-target`}>Đối tượng</Label>
        <Input
          id={`${id}-target`}
          value={target}
          onChange={(event) => setTarget(event.target.value)}
          placeholder="Để trống nếu không có"
          autoComplete="off"
          maxLength={80}
          enterKeyHint="done"
          className="h-9 pointer-coarse:h-11 pointer-coarse:text-base!"
        />
      </div>

      <div className="grid gap-1.5">
        <Label htmlFor={`${id}-type`}>Loại</Label>
        <Select
          value={type}
          onValueChange={(value) => setType(value as TxType)}
        >
          <SelectTrigger id={`${id}-type`}>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {TX_TYPES.map((t) => (
              <SelectItem key={t} value={t}>
                <TypeSwatch type={t} />
                {TYPE_META[t].label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {type !== record.type && (
          <p className="text-xs text-muted-foreground">
            Khi lưu, ghi chú sẽ bay sang chồng “{TYPE_META[type].label}”.
          </p>
        )}
      </div>

      <div className="flex flex-wrap items-center justify-between gap-2">
        {isEdited(record) ? (
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="pointer-coarse:h-11 pointer-coarse:px-3"
            onClick={onRestore}
          >
            <RotateCcwIcon data-icon="inline-start" />
            Khôi phục
          </Button>
        ) : (
          <span />
        )}
        <div className="flex gap-2">
          <Button
            type="button"
            variant="outline"
            className="pointer-coarse:h-11 pointer-coarse:px-4"
            onClick={onCancel}
          >
            Hủy
          </Button>
          <Button
            type="submit"
            className="pointer-coarse:h-11 pointer-coarse:px-5"
            disabled={amountInvalid}
          >
            Lưu
          </Button>
        </div>
      </div>
    </form>
  )
}
