/**
 * Download + verify the model, create the onnxruntime-web session, assemble the
 * predictor. The ORT wasm binary is part of the Vite build (`?url` import), never a CDN.
 */

import * as ort from "onnxruntime-web/wasm"
import wasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.wasm?url"

import { parseConfig } from "./config.ts"
import { createOrtRunner } from "./ort-runner.ts"
import { createPredictor } from "./predictor.ts"
import { BundleTokenizer } from "./tokenizer.ts"
import { fetchModel } from "./model-download.ts"
import type { LoadOptions, WebPredictor } from "./types.ts"

interface WebRelease {
  model_version: string
  release_version: string
}

async function fetchJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(url, { signal })
  if (!response.ok)
    throw new Error(`GET ${url} failed: HTTP ${response.status}`)
  return (await response.json()) as T
}

export async function loadPredictorLocal(
  options: LoadOptions
): Promise<WebPredictor> {
  const { baseUrl, modelParts, modelSha256, onProgress, signal } = options
  const base = baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`

  const [release, configJson, tokenizerJson] = await Promise.all([
    fetchJson<WebRelease>(`${base}web-release.json`, signal),
    fetchJson<unknown>(`${base}config.json`, signal),
    fetchJson<unknown>(`${base}tokenizer.json`, signal),
  ])
  const config = parseConfig(configJson)
  if (release.model_version !== config.modelVersion) {
    throw new Error(
      `web-release.json is for ${release.model_version}, config.json for ${config.modelVersion}`
    )
  }
  const tokenizer = new BundleTokenizer(tokenizerJson, config.maxLength)

  const model = await fetchModel(modelParts, modelSha256, onProgress, signal)
  signal?.throwIfAborted()

  ort.env.wasm.wasmPaths = { wasm: wasmUrl }
  ort.env.wasm.numThreads = 1
  ort.env.wasm.proxy = false
  const runner = await createOrtRunner(ort, model, config)
  return createPredictor({
    config,
    tokenizer,
    runner,
    releaseVersion: release.release_version,
  })
}
