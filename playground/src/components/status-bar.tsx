import type { ReactNode } from "react"

import type { InfoState, RunState } from "@/hooks/use-playground"
import { ms } from "@/lib/format"
import { Separator } from "@/components/ui/separator"

function Item({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex min-w-0 items-baseline gap-1.5">
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="font-mono break-all tabular-nums">{children}</dd>
    </div>
  )
}

export function StatusBar({ run, infoState }: { run: RunState; infoState: InfoState }) {
  const info = infoState.status === "ready" ? infoState.info : null
  const runtime = info
    ? [info.backend, info.precision].filter(Boolean).join(" · ") || "n/a"
    : infoState.status === "offline"
      ? "offline"
      : "connecting"
  const response = run.status === "success" ? run.response : null
  return (
    <footer className="border-t">
      <dl className="mx-auto flex w-full max-w-6xl flex-wrap items-center gap-x-4 gap-y-1 px-4 py-2.5 text-xs sm:px-6">
        <Item label="Runtime">{runtime}</Item>
        <Separator orientation="vertical" className="hidden h-3 sm:block" />
        <Item label="Latency">
          {run.status === "running" ? "running" : response ? ms(response.latencyMs) : "—"}
        </Item>
        <Separator orientation="vertical" className="hidden h-3 sm:block" />
        <Item label="Truncated">
          {response ? (response.result.truncated ? "yes" : "no") : "—"}
        </Item>
        <Separator orientation="vertical" className="hidden h-3 sm:block" />
        <Item label="Model">{response?.result.model_version ?? info?.model_version ?? "—"}</Item>
      </dl>
    </footer>
  )
}
