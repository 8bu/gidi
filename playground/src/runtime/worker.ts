/// <reference lib="webworker" />
/** Worker entry: owns the downloaded model, the ORT session and the predictor. */

import { loadPredictorLocal } from "./load-local.ts"
import type {
  SerializedError,
  WorkerRequest,
  WorkerResponse,
} from "./worker-protocol.ts"
import type { WebPredictor } from "./types.ts"

const post = (message: WorkerResponse): void => self.postMessage(message)

const serialize = (error: unknown): SerializedError =>
  error instanceof Error
    ? { name: error.name, message: error.message }
    : { name: "Error", message: String(error) }

let predictor: WebPredictor | null = null

self.onmessage = async ({ data }: MessageEvent<WorkerRequest>) => {
  switch (data.type) {
    case "load":
      try {
        predictor = await loadPredictorLocal({
          ...data.options,
          onProgress: (loaded, total) =>
            post({ type: "progress", loaded, total }),
        })
        post({ type: "loaded", info: predictor.info })
      } catch (error) {
        post({ type: "load-error", error: serialize(error) })
      }
      break
    case "predict":
      try {
        if (predictor === null) throw new Error("predictor is not loaded")
        post({
          type: "result",
          id: data.id,
          prediction: await predictor.predict(data.text),
        })
      } catch (error) {
        post({ type: "error", id: data.id, error: serialize(error) })
      }
      break
    case "dispose":
      try {
        await predictor?.dispose()
        predictor = null
        post({ type: "disposed", id: data.id })
      } catch (error) {
        post({ type: "error", id: data.id, error: serialize(error) })
      }
      break
  }
}
