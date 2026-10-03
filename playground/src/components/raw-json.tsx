import { useEffect, useRef, useState } from "react"
import { CheckIcon, CopyIcon } from "lucide-react"

import { Button } from "@/components/ui/button"

async function copyText(text: string) {
  try {
    await navigator.clipboard.writeText(text)
    return true
  } catch {
    // Clipboard API is unavailable outside secure contexts; fall back to a hidden textarea.
    const area = document.createElement("textarea")
    area.value = text
    area.setAttribute("readonly", "")
    area.style.position = "fixed"
    area.style.opacity = "0"
    document.body.appendChild(area)
    area.select()
    const ok = document.execCommand("copy")
    area.remove()
    return ok
  }
}

export function RawJson({ raw }: { raw: Record<string, unknown> }) {
  // Two-number arrays (the spans) on one line; values are untouched.
  const json = JSON.stringify(raw, null, 2).replace(
    /\[\n\s*(-?\d+),\n\s*(-?\d+)\n\s*\]/g,
    "[$1, $2]"
  )
  const [state, setState] = useState<"idle" | "copied" | "failed">("idle")
  const timer = useRef<number>(undefined)

  useEffect(() => () => window.clearTimeout(timer.current), [])

  const copy = async () => {
    setState((await copyText(json)) ? "copied" : "failed")
    window.clearTimeout(timer.current)
    timer.current = window.setTimeout(() => setState("idle"), 1600)
  }

  return (
    <div className="relative rounded-lg border bg-muted/40">
      <Button
        variant="ghost"
        size="xs"
        className="absolute top-1.5 right-1.5"
        onClick={copy}
        aria-label="Copy raw JSON"
      >
        {state === "copied" ? (
          <CheckIcon data-icon="inline-start" />
        ) : (
          <CopyIcon data-icon="inline-start" />
        )}
        {state === "copied" ? "Copied" : state === "failed" ? "Copy failed" : "Copy"}
      </Button>
      <pre
        tabIndex={0}
        aria-label="Raw prediction JSON"
        className="max-h-80 overflow-auto p-3 pr-20 font-mono text-xs leading-relaxed break-words whitespace-pre-wrap outline-none focus-visible:ring-2 focus-visible:ring-ring"
      >
        {json}
      </pre>
    </div>
  )
}
