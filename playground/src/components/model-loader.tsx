import { CheckCircle2Icon, DownloadCloudIcon, LockIcon, RefreshCwIcon } from "lucide-react"

import type { InfoState } from "@/hooks/use-playground"
import { ErrorAlert } from "@/components/error-alert"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { Spinner } from "@/components/ui/spinner"
import { ms } from "@/lib/format"

const mb = (bytes: number) => `${(bytes / 1_000_000).toFixed(1)} MB`

const LOCAL_NOTE = "Runs locally in your browser. Your notes never leave this device."

/** Browser mode only: the one-time model download, its failure, and the "runs locally" note. */
export function ModelLoader({ infoState, onRetry }: { infoState: InfoState; onRetry: () => void }) {
  if (infoState.status === "offline") {
    return (
      <div className="flex flex-col items-start gap-2">
        <ErrorAlert error={infoState.error} />
        <Button variant="outline" size="sm" onClick={onRetry}>
          <RefreshCwIcon data-icon="inline-start" />
          Retry
        </Button>
      </div>
    )
  }

  if (infoState.status === "loading") {
    const progress = infoState.progress
    const total = progress?.total ?? null
    const fraction = progress && total ? Math.min(progress.loaded / total, 1) : null
    const starting = progress !== null && total !== null && progress.loaded >= total
    return (
      <Alert>
        <DownloadCloudIcon />
        <AlertTitle>
          {starting ? "Verifying and starting the model…" : "Downloading the model…"}
        </AlertTitle>
        <AlertDescription className="flex flex-col gap-2">
          <span>One-time download, cached by your browser afterwards. {LOCAL_NOTE}</span>
          <div
            role="progressbar"
            aria-label="Model download"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={fraction === null ? undefined : Math.round(fraction * 100)}
            className="h-2 w-full overflow-hidden rounded-full bg-muted"
          >
            <div
              className={
                fraction === null
                  ? "h-full w-1/3 animate-pulse rounded-full bg-primary"
                  : "h-full rounded-full bg-primary transition-[width]"
              }
              style={fraction === null ? undefined : { width: `${fraction * 100}%` }}
            />
          </div>
          <span className="flex items-center gap-1.5 font-mono text-xs tabular-nums">
            {starting ? <Spinner /> : null}
            {progress
              ? total
                ? `${mb(progress.loaded)} / ${mb(total)} (${Math.round((fraction ?? 0) * 100)}%)`
                : mb(progress.loaded)
              : "Starting download"}
          </span>
        </AlertDescription>
      </Alert>
    )
  }

  const { info } = infoState
  return (
    <p className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm text-muted-foreground">
      <LockIcon className="size-4 shrink-0" />
      <span>{LOCAL_NOTE}</span>
      <span className="inline-flex items-center gap-1 text-xs">
        <CheckCircle2Icon className="size-3.5" />
        <span className="font-mono">
          {info.model_sha256 ? `SHA-256 ${info.model_sha256.slice(0, 12)}… verified` : "verified"}
          {typeof info.load_ms === "number" ? ` · loaded in ${ms(info.load_ms)}` : ""}
        </span>
      </span>
    </p>
  )
}
