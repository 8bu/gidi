import { useCallback, useEffect, useRef, useState } from "react"

import {
  BROWSER_MODE,
  loadBrowserBackend,
  type BrowserBackend,
  type LoadProgress,
} from "@/lib/backend"
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
  | {
      status: "success"
      text: string
      response: PredictResponse
      roundTripMs: number
    }
  | { status: "error"; text: string; error: PlaygroundError }

export type InfoState =
  // `progress` is the model download in browser mode (null before the first byte arrives).
  | { status: "loading"; progress: LoadProgress | null }
  | { status: "ready"; info: RuntimeInfo }
  // `info` keeps the last runtime facts seen before the server went away.
  | { status: "offline"; error: PlaygroundError; info: RuntimeInfo | null }

function asPlaygroundError(cause: unknown): PlaygroundError {
  return cause instanceof PlaygroundError
    ? cause
    : new PlaygroundError("unavailable", cause instanceof Error ? cause.message : "Unknown error.")
}

export function usePlayground() {
  const [text, setText] = useState("")
  const [run, setRun] = useState<RunState>({ status: "idle" })
  const [infoState, setInfoState] = useState<InfoState>({
    status: "loading",
    progress: null,
  })
  const running = useRef(false)
  const backend = useRef<BrowserBackend | null>(null)
  const loading = useRef<AbortController | null>(null)

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

  /** Browser mode: download, verify and start the model. Safe to call again to retry. */
  const loadModel = useCallback(async () => {
    loading.current?.abort()
    const controller = new AbortController()
    loading.current = controller
    setInfoState({ status: "loading", progress: null })
    try {
      const loaded = await loadBrowserBackend((progress) => {
        if (!controller.signal.aborted) setInfoState({ status: "loading", progress })
      }, controller.signal)
      if (controller.signal.aborted) {
        void loaded.dispose()
        return
      }
      backend.current = loaded
      setInfoState({ status: "ready", info: loaded.info })
    } catch (cause) {
      if (!controller.signal.aborted) goOffline(asPlaygroundError(cause))
    }
  }, [goOffline])

  useEffect(() => {
    // Initial load only; later refreshes come from a failed or recovered run.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void (BROWSER_MODE ? loadModel() : refreshInfo())
    return () => loading.current?.abort()
  }, [loadModel, refreshInfo])

  const retry = BROWSER_MODE ? loadModel : refreshInfo
  /** In browser mode nothing can run before the model is loaded. */
  const canRun = !BROWSER_MODE || infoState.status === "ready"

  const submit = useCallback(
    async (note: string) => {
      if (running.current || !canRun) return
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
        const loaded = backend.current
        if (BROWSER_MODE && loaded === null) {
          throw new PlaygroundError("model_load", "The model is not loaded.")
        }
        const response = loaded ? await loaded.predict(note) : await predict(note)
        setRun({
          status: "success",
          text: note,
          response,
          roundTripMs: performance.now() - started,
        })
        if (!BROWSER_MODE && infoState.status !== "ready") void refreshInfo()
      } catch (cause) {
        const error = asPlaygroundError(cause)
        setRun({ status: "error", text: note, error })
        if (!BROWSER_MODE && error.kind === "unavailable") goOffline(error)
      } finally {
        running.current = false
      }
    },
    [canRun, infoState.status, refreshInfo, goOffline]
  )

  const clear = useCallback(() => {
    if (running.current) return
    setText("")
    setRun({ status: "idle" })
  }, [])

  return { text, setText, run, infoState, canRun, retry, submit, clear }
}
