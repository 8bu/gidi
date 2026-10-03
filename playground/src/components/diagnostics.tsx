import type { ReactNode } from "react"

import type { PredictResponse, RuntimeInfo, Span } from "@/lib/api"
import { ms, spanLabel } from "@/lib/format"

function Item({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex min-w-0 flex-col gap-0.5">
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd className="font-mono text-xs break-all">{children}</dd>
    </div>
  )
}

const spanOrNull = (span: Span | null | undefined) => spanLabel(span) ?? "null"

type DiagnosticsProps = {
  response: PredictResponse
  roundTripMs: number
  info: RuntimeInfo | null
}

export function Diagnostics({ response, roundTripMs, info }: DiagnosticsProps) {
  const { result, latencyMs, spansOverlap } = response
  const hasValue = "value_text" in result
  const unknown = "n/a"
  return (
    <dl className="grid grid-cols-1 gap-x-6 gap-y-3 rounded-lg border bg-muted/40 p-3 min-[480px]:grid-cols-2">
      <Item label="Inference latency">{ms(latencyMs)}</Item>
      <Item label="Round trip">{ms(roundTripMs)}</Item>
      <Item label="Model version">{result.model_version}</Item>
      <Item label="Truncated">{String(result.truncated)}</Item>
      <Item label="Target offsets">{spanOrNull(result.target_span)}</Item>
      <Item label="Value offsets">{hasValue ? spanOrNull(result.value_span) : unknown}</Item>
      {spansOverlap !== null ? (
        <Item label="Spans overlap">{String(spansOverlap)}</Item>
      ) : null}
      <Item label="Backend">{info?.backend ?? unknown}</Item>
      <Item label="Precision">{info?.precision ?? unknown}</Item>
      <Item label="Architecture">{info?.architecture ?? unknown}</Item>
      <Item label="ONNX opset">{info?.onnx_opset ?? unknown}</Item>
      <Item label="Max length">{info ? `${info.max_length} tokens` : unknown}</Item>
      <Item label="Model file">{info?.model_file ?? unknown}</Item>
      <Item label="Bundle">{info?.bundle_path ?? info?.bundle ?? unknown}</Item>
    </dl>
  )
}
