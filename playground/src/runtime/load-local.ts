/**
 * Download + verify the model, create the onnxruntime-web session, assemble the
 * predictor. The ORT wasm binary is part of the Vite build (`?url` import), never a CDN.
 */

import * as ort from "onnxruntime-web/wasm"
import wasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.wasm?url"

import { parseConfig } from "./config.ts"
import { createOrtRunner } from "./ort-runner.ts"
import { createPredictor } from "./predictor.ts"
import { createTokenizer } from "./tokenizer.ts"
import { fetchModel } from "./model-download.ts"
import type { LoadOptions, WebPredictor } from "./types.ts"

async function fetchJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(url, { signal })
  if (!response.ok)
    throw new Error(`GET ${url} failed: HTTP ${response.status}`)
  return (await response.json()) as T
}

export async function loadPredictorLocal(
  options: LoadOptions
): Promise<WebPredictor> {
  const {
    baseUrl,
    modelParts,
    modelSha256,
    modelVersion,
    releaseVersion,
    tokenizerType,
    store,
    onProgress,
    onDiagnostic,
    signal,
  } = options
  const base = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`

  const [configJson, tokenizerJson] = await Promise.all([
    fetchJson<unknown>(`${base}config.json`, signal),
    fetchJson<unknown>(`${base}tokenizer.json`, signal),
  ])
  const config = parseConfig(configJson)
  if (modelVersion !== config.modelVersion) {
    throw new Error(
      `the manifest lists ${modelVersion}, config.json is for ${config.modelVersion}`
    )
  }
  if (tokenizerType !== config.tokenizerType) {
    throw new Error(
      `the manifest lists a ${tokenizerType} tokenizer, config.json says ${config.tokenizerType}`
    )
  }
  const tokenizer = createTokenizer(
    tokenizerJson,
    config.maxLength,
    config.tokenizerType
  )

  const model = await fetchModel(
    modelParts,
    modelSha256,
    onProgress,
    signal,
    store && {
      ...store,
      report: (stage, error) =>
        onDiagnostic?.(
          stage,
          error instanceof Error
            ? { name: error.name, message: error.message }
            : { name: "Error", message: String(error) }
        ),
    }
  )
  signal?.throwIfAborted()

  ort.env.wasm.wasmPaths = { wasm: wasmUrl }
  ort.env.wasm.numThreads = 1
  ort.env.wasm.proxy = false
  const runner = await createOrtRunner(ort, model, config)
  return createPredictor({
    config,
    tokenizer,
    runner,
    releaseVersion,
  })
}
