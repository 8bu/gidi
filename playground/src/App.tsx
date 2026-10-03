import { useEffect, useRef } from "react"
import { PlugZapIcon } from "lucide-react"

import { Composer } from "@/components/composer"
import { ErrorAlert } from "@/components/error-alert"
import { ModelLoader } from "@/components/model-loader"
import { ResultEmpty, ResultSkeleton, ResultView } from "@/components/result-view"
import { StatusBar } from "@/components/status-bar"
import { ThemeToggle } from "@/components/theme-toggle"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Spinner } from "@/components/ui/spinner"
import { TooltipProvider, Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { usePlayground, type InfoState } from "@/hooks/use-playground"
import { BROWSER_MODE } from "@/lib/backend"
import { displayName } from "@/lib/format"
import { PRESETS } from "@/lib/presets"

function RuntimeBadges({ infoState }: { infoState: InfoState }) {
  if (infoState.status === "loading") {
    const { progress } = infoState
    const percent = progress?.total ? Math.round((progress.loaded / progress.total) * 100) : null
    return (
      <Badge variant="outline">
        <Spinner data-icon="inline-start" />
        {BROWSER_MODE
          ? percent === null
            ? "Loading model"
            : `Loading model ${percent}%`
          : "Connecting"}
      </Badge>
    )
  }
  if (infoState.status === "offline") {
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <Badge variant="destructive" tabIndex={0}>
            <PlugZapIcon data-icon="inline-start" />
            {BROWSER_MODE ? "Model not loaded" : "Runtime offline"}
          </Badge>
        </TooltipTrigger>
        <TooltipContent>{infoState.error.message}</TooltipContent>
      </Tooltip>
    )
  }
  const { info } = infoState
  const build = [info.precision, info.backend].filter(Boolean).join(" · ")
  return (
    <>
      <Badge variant="secondary" className="font-mono">
        {info.model_version}
      </Badge>
      {build ? (
        <Badge variant="outline" className="hidden font-mono sm:inline-flex">
          {build}
        </Badge>
      ) : null}
    </>
  )
}

export function App() {
  const { text, setText, run, infoState, canRun, retry, submit, clear } = usePlayground()
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const resultRef = useRef<HTMLDivElement>(null)
  const running = run.status === "running"
  const info = infoState.status === "loading" ? null : infoState.info
  const title = info ? displayName(info.model_version) : "Gidi Finance"

  useEffect(() => {
    document.title = `${title} playground`
  }, [title])

  // Stacked layout: bring the result into view when a run starts.
  useEffect(() => {
    if (run.status === "running" && !window.matchMedia("(min-width: 1024px)").matches) {
      resultRef.current?.scrollIntoView({
        block: "nearest",
        behavior: "smooth",
      })
    }
  }, [run.status])

  const runPreset = (preset: string) => {
    setText(preset)
    void submit(preset)
  }

  const emptyInputError = run.status === "error" && run.error.kind === "empty_input"
  const announcement =
    run.status === "running"
      ? "Running inference"
      : run.status === "success"
        ? `Result: type ${run.response.result.type}`
        : run.status === "error"
          ? `Error: ${run.error.message}`
          : ""

  return (
    <TooltipProvider>
      <div className="flex min-h-dvh flex-col">
        <header className="border-b">
          <div className="mx-auto flex w-full max-w-6xl flex-wrap items-center justify-between gap-x-4 gap-y-2 px-4 py-3 sm:px-6">
            <div className="min-w-0">
              <h1 className="text-lg leading-tight font-semibold">{title}</h1>
              <p className="text-sm text-muted-foreground">
                Extract transaction semantics from Vietnamese notes
              </p>
            </div>
            <div className="flex items-center gap-2">
              <RuntimeBadges infoState={infoState} />
              <ThemeToggle />
            </div>
          </div>
        </header>
        <p role="status" className="sr-only">
          {announcement}
        </p>

        <main className="mx-auto grid w-full max-w-6xl flex-1 content-start gap-4 px-4 py-4 sm:px-6 lg:grid-cols-2">
          {BROWSER_MODE ? (
            <div className="min-w-0 lg:col-span-2">
              <ModelLoader infoState={infoState} onRetry={() => void retry()} />
            </div>
          ) : null}
          <Card className="min-w-0 self-start lg:sticky lg:top-4">
            <CardHeader>
              <CardTitle>Input</CardTitle>
              <CardDescription>One note, parsed when you run it.</CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-4">
              <Composer
                value={text}
                onChange={setText}
                onSubmit={() => void submit(text)}
                onClear={() => {
                  clear()
                  textareaRef.current?.focus()
                }}
                running={running}
                disabled={!canRun}
                canClear={text !== "" || run.status !== "idle"}
                invalid={emptyInputError}
                textareaRef={textareaRef}
              />
              {emptyInputError ? <ErrorAlert error={run.error} /> : null}
              <section aria-labelledby="presets-heading" className="flex flex-col gap-2">
                <h2
                  id="presets-heading"
                  className="text-xs font-medium tracking-wide text-muted-foreground uppercase"
                >
                  Presets
                </h2>
                <div className="flex flex-wrap gap-2">
                  {PRESETS.map((preset) => (
                    <Button
                      key={preset}
                      variant="outline"
                      size="sm"
                      lang="vi"
                      className="max-sm:h-9"
                      disabled={running || !canRun}
                      onClick={() => runPreset(preset)}
                    >
                      {preset}
                    </Button>
                  ))}
                </div>
              </section>
            </CardContent>
          </Card>

          <Card ref={resultRef} className="min-w-0 scroll-mt-4 self-start">
            <CardHeader>
              <CardTitle>Parsed result</CardTitle>
              <CardDescription>Type, target and value of the current note.</CardDescription>
            </CardHeader>
            <CardContent>
              {run.status === "idle" ? <ResultEmpty /> : null}
              {run.status === "running" ? <ResultSkeleton /> : null}
              {run.status === "error" && !emptyInputError ? <ErrorAlert error={run.error} /> : null}
              {run.status === "error" && emptyInputError ? <ResultEmpty /> : null}
              {run.status === "success" ? (
                <ResultView
                  text={run.text}
                  response={run.response}
                  roundTripMs={run.roundTripMs}
                  info={info}
                />
              ) : null}
            </CardContent>
          </Card>
        </main>

        <StatusBar run={run} infoState={infoState} />
      </div>
    </TooltipProvider>
  )
}

export default App
