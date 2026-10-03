/**
 * Browser entry. The model download, sha256 check, onnxruntime-web session and every prediction
 * run in a dedicated Web Worker, so the UI thread never blocks on the ~0.4 s session creation
 * or the ~25 ms per-note inference. Without `Worker` (tests, odd embeds) it falls back to the
 * calling thread.
 */

import {
  EmptyInputError,
  ModelIntegrityError,
  type LoadOptions,
  type Prediction,
  type PredictorInfo,
  type WebPredictor,
} from "./types.ts"
import type {
  SerializedError,
  WorkerRequest,
  WorkerResponse,
} from "./worker-protocol.ts"

function revive({ name, message }: SerializedError): Error {
  if (name === "EmptyInputError") return new EmptyInputError(message)
  if (name === "ModelIntegrityError") return new ModelIntegrityError(message)
  const error = new Error(message)
  error.name = name
  return error
}

function abortError(signal: AbortSignal): unknown {
  return (
    signal.reason ??
    new DOMException("The operation was aborted.", "AbortError")
  )
}

export async function loadPredictor(
  options: LoadOptions
): Promise<WebPredictor> {
  if (typeof Worker === "undefined") {
    // Dynamic: keeps onnxruntime-web out of the main-thread chunk when the worker is used.
    const { loadPredictorLocal } = await import("./load-local.ts")
    return loadPredictorLocal(options)
  }
  const { onProgress, signal, ...workerOptions } = options
  signal?.throwIfAborted()

  const worker = new Worker(new URL("./worker.ts", import.meta.url), {
    type: "module",
  })
  const send = (request: WorkerRequest): void => worker.postMessage(request)

  const info = await new Promise<PredictorInfo>((resolve, reject) => {
    const onAbort = (): void => {
      worker.terminate()
      reject(abortError(signal as AbortSignal))
    }
    signal?.addEventListener("abort", onAbort, { once: true })
    const settle = (): void => signal?.removeEventListener("abort", onAbort)
    worker.onerror = (event) => {
      settle()
      worker.terminate()
      reject(new Error(event.message || "model worker failed to start"))
    }
    worker.onmessage = ({ data }: MessageEvent<WorkerResponse>) => {
      if (data.type === "progress") onProgress?.(data.loaded, data.total)
      else if (data.type === "loaded") {
        settle()
        resolve(data.info)
      } else if (data.type === "load-error") {
        settle()
        worker.terminate()
        reject(revive(data.error))
      }
    }
    send({ type: "load", options: workerOptions })
  })

  let nextId = 0
  const pending = new Map<
    number,
    {
      resolve: (value: Prediction | undefined) => void
      reject: (error: Error) => void
    }
  >()
  worker.onerror = null
  worker.onmessage = ({ data }: MessageEvent<WorkerResponse>) => {
    if (
      data.type === "result" ||
      data.type === "disposed" ||
      data.type === "error"
    ) {
      const call = pending.get(data.id)
      pending.delete(data.id)
      if (data.type === "result") call?.resolve(data.prediction)
      else if (data.type === "disposed") call?.resolve(undefined)
      else call?.reject(revive(data.error))
    }
  }
  const call = (
    request: (id: number) => WorkerRequest
  ): Promise<Prediction | undefined> =>
    new Promise((resolve, reject) => {
      const id = nextId++
      pending.set(id, { resolve, reject })
      send(request(id))
    })

  return {
    info,
    async predict(text) {
      const prediction = await call((id) => ({ type: "predict", id, text }))
      return prediction as Prediction
    },
    async dispose() {
      await call((id) => ({ type: "dispose", id }))
      worker.terminate()
    },
  }
}
