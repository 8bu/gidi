import type { ComponentType } from "react"
import {
  ClockAlertIcon,
  FileWarningIcon,
  PlugZapIcon,
  ServerCrashIcon,
  TextCursorInputIcon,
  TriangleAlertIcon,
} from "lucide-react"

import type { ErrorKind, PlaygroundError } from "@/lib/api"
import { REQUEST_TIMEOUT_MS } from "@/lib/api"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"

type Copy = { icon: ComponentType; title: string; hint: string }

const COPY: Record<ErrorKind, Copy> = {
  empty_input: {
    icon: TextCursorInputIcon,
    title: "Nothing to parse",
    hint: "Type or paste a note first, or pick a preset.",
  },
  unavailable: {
    icon: PlugZapIcon,
    title: "Runtime unavailable",
    hint: "The playground server did not answer. Check that `uv run python scripts/demo_ui.py` is still running, then run the note again.",
  },
  timeout: {
    icon: ClockAlertIcon,
    title: "Request timed out",
    hint: `The server did not respond within ${REQUEST_TIMEOUT_MS / 1000} s. Run the note again.`,
  },
  inference: {
    icon: ServerCrashIcon,
    title: "Inference failed",
    hint: "The runtime raised an error while parsing this note.",
  },
  missing_artifact: {
    icon: FileWarningIcon,
    title: "Model artifact missing",
    hint: "A file of the model bundle could not be read. Rebuild or restore the bundle, then restart the server.",
  },
  malformed: {
    icon: TriangleAlertIcon,
    title: "Unexpected response",
    hint: "The server answered, but the payload does not match the playground's schema.",
  },
  rejected: {
    icon: TriangleAlertIcon,
    title: "Request rejected",
    hint: "The server refused this request.",
  },
}

export function ErrorAlert({ error }: { error: PlaygroundError }) {
  const { icon: Icon, title, hint } = COPY[error.kind]
  const detail = error.kind === "empty_input" || error.kind === "timeout" ? null : error.message
  return (
    <Alert variant="destructive">
      <Icon />
      <AlertTitle>{title}</AlertTitle>
      <AlertDescription className="flex flex-col gap-1.5">
        <span>{hint}</span>
        {detail ? (
          <code className="font-mono text-xs break-words whitespace-pre-wrap">
            {error.code ? `${error.code}: ` : ""}
            {detail}
          </code>
        ) : null}
      </AlertDescription>
    </Alert>
  )
}
