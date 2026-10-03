import { useCallback, useEffect, useRef, useState } from "react"

import {
  fetchInfo,
  PlaygroundError,
  predict,
  type PredictResponse,
  type RuntimeInfo,
} from "@/lib/api"

export type RunState =
  | { status: "idle" }
  | { status: "running"; text: string }
  | { status: "success"; text: string; response: PredictResponse; roundTripMs: number }
  | { status: "error"; text: string; error: PlaygroundError }

export type InfoState =
  | { status: "loading" }
  | { status: "ready"; info: RuntimeInfo }
  // `info` keeps the last runtime facts seen before the server went away.
  | { status: "offline"; error: PlaygroundError; info: RuntimeInfo | null }

function asPlaygroundError(cause: unknown): PlaygroundError {
  return cause instanceof PlaygroundError
    ? cause
    : new PlaygroundError(
        "unavailable",
        cause instanceof Error ? cause.message : "Unknown error."
      )
}

export function usePlayground() {
  const [text, setText] = useState("")
  const [run, setRun] = useState<RunState>({ status: "idle" })
  const [infoState, setInfoState] = useState<InfoState>({ status: "loading" })
  const running = useRef(false)

  const goOffline = useCallback((error: PlaygroundError) => {
    setInfoState((previous) => ({
      status: "offline",
      error,
      info: previous.status === "loading" ? null : previous.info,
    }))
  }, [])

  const refreshInfo = useCallback(async () => {
    try {
      setInfoState({ status: "ready", info: await fetchInfo() })
    } catch (cause) {
      goOffline(asPlaygroundError(cause))
    }
  }, [goOffline])

  useEffect(() => {
    // Initial load only; later refreshes come from a failed or recovered run.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void refreshInfo()
  }, [refreshInfo])

  const submit = useCallback(
    async (note: string) => {
      if (running.current) return
      if (note.trim() === "") {
        setRun({
          status: "error",
          text: note,
          error: new PlaygroundError(
            "empty_input",
            "Type or paste a note first, or pick a preset.",
            "empty_input"
          ),
        })
        return
      }
      running.current = true
      setRun({ status: "running", text: note })
      const started = performance.now()
      try {
        const response = await predict(note)
        setRun({
          status: "success",
          text: note,
          response,
          roundTripMs: performance.now() - started,
        })
        if (infoState.status !== "ready") void refreshInfo()
      } catch (cause) {
        const error = asPlaygroundError(cause)
        setRun({ status: "error", text: note, error })
        if (error.kind === "unavailable") goOffline(error)
      } finally {
        running.current = false
      }
    },
    [infoState.status, refreshInfo, goOffline]
  )

  const clear = useCallback(() => {
    if (running.current) return
    setText("")
    setRun({ status: "idle" })
  }, [])

  return { text, setText, run, infoState, submit, clear }
}
