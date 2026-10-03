import type { ReactNode } from "react"
import { InfoIcon, TextCursorInputIcon } from "lucide-react"

import type { PredictResponse, RuntimeInfo } from "@/lib/api"
import { KNOWN_LIMITATION } from "@/lib/presets"
import { ms, pct, spanLabel } from "@/lib/format"
import { Badge } from "@/components/ui/badge"
import { Empty, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from "@/components/ui/empty"
import { Separator } from "@/components/ui/separator"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { Diagnostics } from "@/components/diagnostics"
import { RawJson } from "@/components/raw-json"
import { SpanLegend, SpanView } from "@/components/span-view"

function Row({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="grid grid-cols-[4.5rem_minmax(0,1fr)] gap-x-3 sm:grid-cols-[5.5rem_minmax(0,1fr)]">
      <dt className="pt-1 text-xs font-medium tracking-wide text-muted-foreground uppercase">
        {label}
      </dt>
      <dd className="flex min-w-0 flex-col gap-1.5">{children}</dd>
    </div>
  )
}

function Confidence({ value, hint }: { value: number; hint?: string }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Badge variant="secondary" tabIndex={0} className="font-mono tabular-nums">
          {pct(value)}
        </Badge>
      </TooltipTrigger>
      <TooltipContent>
        {hint ? `${hint} Raw: ${value}` : `Raw confidence: ${value}`}
      </TooltipContent>
    </Tooltip>
  )
}

function SpanText({ span }: { span: readonly [number, number] }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span className="font-mono text-xs text-muted-foreground tabular-nums">
          chars {spanLabel(span)}
        </span>
      </TooltipTrigger>
      <TooltipContent>Unicode code points, start inclusive, end exclusive</TooltipContent>
    </Tooltip>
  )
}

function Primary({ children }: { children: ReactNode }) {
  return <p className="text-lg leading-snug font-semibold break-words">{children}</p>
}

const NONE = <span className="text-lg leading-snug text-muted-foreground">—</span>

export function ResultEmpty() {
  return (
    <Empty className="border py-10">
      <EmptyHeader>
        <EmptyMedia variant="icon">
          <TextCursorInputIcon />
        </EmptyMedia>
        <EmptyTitle>No result yet</EmptyTitle>
        <EmptyDescription>
          Run a note or pick a preset to see its transaction type, target and value.
        </EmptyDescription>
      </EmptyHeader>
    </Empty>
  )
}

export function ResultSkeleton() {
  return (
    <div role="status" aria-label="Running inference" className="flex flex-col gap-4">
      {[0, 1, 2].map((row) => (
        <div key={row} className="flex flex-col gap-4">
          <div className="grid grid-cols-[4.5rem_minmax(0,1fr)] gap-x-3 sm:grid-cols-[5.5rem_minmax(0,1fr)]">
            <Skeleton className="mt-1 h-3 w-12" />
            <div className="flex flex-col gap-2">
              <Skeleton className="h-6 w-2/3" />
              <Skeleton className="h-5 w-32" />
            </div>
          </div>
          <Separator />
        </div>
      ))}
      <Skeleton className="h-14 w-full" />
      <Skeleton className="h-8 w-48" />
    </div>
  )
}

type ResultViewProps = {
  text: string
  response: PredictResponse
  roundTripMs: number
  info: RuntimeInfo | null
}

export function ResultView({ text, response, roundTripMs, info }: ResultViewProps) {
  const { result, latencyMs, segments, spansOverlap } = response
  const hasValue = "value_text" in result
  const valueText = result.value_text ?? null
  const showKnownLimitation =
    text === KNOWN_LIMITATION.text && hasValue && valueText !== KNOWN_LIMITATION.expectedValue

  return (
    <div className="flex min-w-0 flex-col gap-4">
      <dl className="flex flex-col gap-4">
        <Row label="Type">
          <Primary>{result.type}</Primary>
          <div className="flex flex-wrap items-center gap-2">
            <Confidence value={result.type_confidence} />
          </div>
        </Row>
        <Separator />
        <Row label="Target">
          {result.target === null ? NONE : <Primary>{result.target}</Primary>}
          <div className="flex flex-wrap items-center gap-2">
            <Confidence
              value={result.target_confidence}
              hint={
                result.target === null
                  ? "No target span. Confidence is the lowest P(O) over the note's tokens."
                  : undefined
              }
            />
            {result.target_span ? (
              <SpanText span={result.target_span} />
            ) : (
              <Badge variant="outline">no target found</Badge>
            )}
          </div>
        </Row>
        <Separator />
        <Row label="Value">
          {!hasValue ? (
            <>
              {NONE}
              <p className="text-xs text-muted-foreground">
                This bundle ({result.model_version}) has no value head.
              </p>
            </>
          ) : (
            <>
              {valueText === null ? NONE : <Primary>{valueText}</Primary>}
              <div className="flex flex-wrap items-center gap-2">
                <Confidence
                  value={result.value_confidence ?? 0}
                  hint={
                    valueText === null
                      ? "No value span. Confidence is the lowest P(O) over the note's tokens."
                      : undefined
                  }
                />
                {result.value_span ? (
                  <SpanText span={result.value_span} />
                ) : (
                  <Badge variant="outline">no value found</Badge>
                )}
                {showKnownLimitation ? (
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Badge variant="outline" tabIndex={0}>
                        <InfoIcon data-icon="inline-start" />
                        Known limitation
                      </Badge>
                    </TooltipTrigger>
                    <TooltipContent>
                      The expected value for this note is “{KNOWN_LIMITATION.expectedValue}”. The
                      output is shown exactly as the model returned it.
                    </TooltipContent>
                  </Tooltip>
                ) : null}
              </div>
            </>
          )}
        </Row>
        <Separator />
        <Row label="Runtime">
          <div className="flex flex-wrap items-center gap-2 text-sm">
            <span className="font-mono text-xs break-all">{result.model_version}</span>
            <Badge variant="secondary" className="font-mono tabular-nums">
              {ms(latencyMs)}
            </Badge>
            <Badge variant={result.truncated ? "default" : "outline"}>
              {result.truncated ? "truncated" : "not truncated"}
            </Badge>
          </div>
        </Row>
      </dl>

      <Separator />

      <section aria-labelledby="spans-heading" className="flex flex-col gap-2">
        <h3
          id="spans-heading"
          className="text-xs font-medium tracking-wide text-muted-foreground uppercase"
        >
          Spans in input
        </h3>
        <SpanView
          segments={segments}
          targetSpan={result.target_span}
          valueSpan={result.value_span}
        />
        <SpanLegend hasValue={hasValue} />
        {spansOverlap ? (
          <p className="text-xs text-muted-foreground">
            Target and value spans overlap. The shared text is outlined; both raw spans are in
            Diagnostics.
          </p>
        ) : null}
      </section>

      <Tabs defaultValue="json">
        <TabsList>
          <TabsTrigger value="json">Raw JSON</TabsTrigger>
          <TabsTrigger value="diagnostics">Diagnostics</TabsTrigger>
        </TabsList>
        <TabsContent value="json">
          <RawJson raw={response.raw} />
        </TabsContent>
        <TabsContent value="diagnostics">
          <Diagnostics
            response={response}
            roundTripMs={roundTripMs}
            info={info}
          />
        </TabsContent>
      </Tabs>
    </div>
  )
}
