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
import {
  ModelIntegrityError,
  type LoadOptions,
  type WebPredictor,
} from "./types.ts"

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

/** Stream `url` into one buffer, reporting progress; `total` is `null` when unknown. */
async function downloadModel(
  url: string,
  onProgress: LoadOptions["onProgress"],
  signal?: AbortSignal
): Promise<Uint8Array> {
  const response = await fetch(url, { signal })
  if (!response.ok || response.body === null) {
    throw new Error(`GET ${url} failed: HTTP ${response.status}`)
  }
  // A compressed transfer reports the encoded length; the body then yields more bytes.
  const encoded = response.headers.get("content-encoding") ?? "identity"
  const length = Number(response.headers.get("content-length"))
  const total = encoded === "identity" && length > 0 ? length : null
  let buffer = new Uint8Array(total ?? 64 * 1024 * 1024)
  let loaded = 0
  const reader = response.body.getReader()
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    if (loaded + value.length > buffer.length) {
      const grown = new Uint8Array(
        Math.max(buffer.length * 2, loaded + value.length)
      )
      grown.set(buffer.subarray(0, loaded))
      buffer = grown
    }
    buffer.set(value, loaded)
    loaded += value.length
    onProgress?.(loaded, total)
  }
  onProgress?.(loaded, loaded)
  return buffer.subarray(0, loaded)
}

async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", bytes as BufferSource)
  return Array.from(new Uint8Array(digest), (b) =>
    b.toString(16).padStart(2, "0")
  ).join("")
}

export async function loadPredictorLocal(
  options: LoadOptions
): Promise<WebPredictor> {
  const { baseUrl, modelUrl, modelSha256, onProgress, signal } = options
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

  const model = await downloadModel(modelUrl, onProgress, signal)
  const actual = await sha256Hex(model)
  if (actual !== modelSha256.toLowerCase()) {
    throw new ModelIntegrityError(
      `model sha256 mismatch: expected ${modelSha256}, downloaded ${actual}`
    )
  }
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
