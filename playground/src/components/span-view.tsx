import { cva } from "class-variance-authority"
import { cn } from "cn"

import type { Segment, Span } from "@/lib/api"
import { spanLabel } from "@/lib/format"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"

const markVariants = cva(
  "rounded-sm border-b-2 px-0.5 py-px outline-none focus-visible:ring-2 focus-visible:ring-ring after:ml-1 after:align-middle after:text-[10px] after:leading-none after:font-semibold after:tracking-wide after:uppercase after:opacity-75 after:content-[attr(data-label)]",
  {
    variants: {
      role: {
        target:
          "border-span-target-foreground/60 bg-span-target text-span-target-foreground",
        value:
          "border-dashed border-span-value-foreground/60 bg-span-value text-span-value-foreground",
        overlap:
          "border-dotted border-destructive bg-span-target text-span-target-foreground ring-1 ring-destructive/60",
      },
    },
  }
)

const LABELS = { target: "Target", value: "Value", overlap: "Target + value" } as const

type Group = { text: string; role: Segment["role"]; overlap: boolean }

/** Merge neighbouring segments that render identically; the text still joins back exactly. */
function groupSegments(segments: Segment[]): Group[] {
  const groups: Group[] = []
  for (const segment of segments) {
    const overlap = segment.overlap === true
    const last = groups.at(-1)
    if (last && last.role === segment.role && last.overlap === overlap) {
      last.text += segment.text
    } else {
      groups.push({ text: segment.text, role: segment.role, overlap })
    }
  }
  return groups
}

type SpanViewProps = {
  segments: Segment[]
  targetSpan: Span | null
  valueSpan: Span | null | undefined
}

export function SpanView({ segments, targetSpan, valueSpan }: SpanViewProps) {
  return (
    <p
      lang="vi"
      aria-label="Input with highlighted spans"
      className="rounded-lg border bg-muted/40 px-3 py-2.5 text-[15px] leading-8 break-words whitespace-pre-wrap"
    >
      {groupSegments(segments).map((group, index) => {
        if (group.role === null) return <span key={index}>{group.text}</span>
        const role = group.overlap ? "overlap" : group.role
        const span = group.role === "target" ? targetSpan : valueSpan
        return (
          <Tooltip key={index}>
            <TooltipTrigger asChild>
              {/* The label is CSS-generated so the DOM text is exactly the input text. */}
              <mark
                tabIndex={0}
                data-label={group.overlap ? "overlap" : group.role}
                className={cn(markVariants({ role }))}
              >
                {group.text}
              </mark>
            </TooltipTrigger>
            <TooltipContent>
              {LABELS[role]}
              {span ? ` ${spanLabel(span)}` : ""}
            </TooltipContent>
          </Tooltip>
        )
      })}
    </p>
  )
}

export function SpanLegend({ hasValue }: { hasValue: boolean }) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
      <span className="flex items-center gap-1.5">
        <mark className={cn(markVariants({ role: "target" }), "px-1.5")}>target</mark>
        counterparty
      </span>
      {hasValue ? (
        <span className="flex items-center gap-1.5">
          <mark className={cn(markVariants({ role: "value" }), "px-1.5")}>value</mark>
          amount text
        </span>
      ) : null}
    </div>
  )
}
