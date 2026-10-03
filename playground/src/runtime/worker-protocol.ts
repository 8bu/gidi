/** Messages between `load.ts` (main thread) and `worker.ts` (owns the model and ORT session). */

import type { LoadOptions, Prediction, PredictorInfo } from "./types.ts"

export type WorkerLoadOptions = Omit<LoadOptions, "onProgress" | "signal">

export type WorkerRequest =
  | { type: "load"; options: WorkerLoadOptions }
  | { type: "predict"; id: number; text: string }
  | { type: "dispose"; id: number }

export interface SerializedError {
  name: string
  message: string
}

export type WorkerResponse =
  | { type: "progress"; loaded: number; total: number | null }
  | { type: "loaded"; info: PredictorInfo }
  | { type: "load-error"; error: SerializedError }
  | { type: "result"; id: number; prediction: Prediction }
  | { type: "disposed"; id: number }
  | { type: "error"; id: number; error: SerializedError }
