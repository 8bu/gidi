/**
 * `ModelRunner` on onnxruntime-web's WASM execution provider (CPU kernels, single thread).
 * Shared by the browser loader and the Node parity runner; the caller supplies the `ort`
 * module (and, in the browser, has pointed `ort.env.wasm.wasmPaths` at self-hosted files).
 */

import type * as Ort from "onnxruntime-web/wasm"

import type { BundleConfig } from "./config.ts"
import type { ModelOutputs, ModelRunner } from "./predictor.ts"

export type OrtModule = typeof Ort

const sameNames = (a: readonly string[], b: readonly string[]): boolean =>
  a.length === b.length && a.every((name, i) => name === b[i])

export async function createOrtRunner(
  ort: OrtModule,
  model: Uint8Array,
  config: BundleConfig
): Promise<ModelRunner> {
  const session = await ort.InferenceSession.create(model, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
    executionMode: "sequential",
    intraOpNumThreads: 1,
    interOpNumThreads: 1,
  })
  if (
    !sameNames(session.inputNames, config.inputNames) ||
    !sameNames(session.outputNames, config.outputNames)
  ) {
    await session.release()
    throw new Error(
      `model exposes inputs ${session.inputNames} / outputs ${session.outputNames}, ` +
        `config.json expects ${config.inputNames} / ${config.outputNames}`
    )
  }
  const [inputName, maskName] = config.inputNames
  const [typeName, tagName, valueName] = config.outputNames

  return {
    backend: "onnxruntime-web (wasm)",
    async run(ids, attentionMask) {
      const dims = [1, ids.length]
      const feeds = {
        [inputName]: new ort.Tensor(
          "int64",
          BigInt64Array.from(ids, BigInt),
          dims
        ),
        [maskName]: new ort.Tensor(
          "int64",
          BigInt64Array.from(attentionMask, BigInt),
          dims
        ),
      }
      const out = await session.run(feeds)
      const get = (name: string): Float32Array => out[name].data as Float32Array
      const outputs: ModelOutputs = {
        typeLogits: Float32Array.from(get(typeName)),
        tagLogits: Float32Array.from(get(tagName)),
        valueLogits: Float32Array.from(get(valueName)),
      }
      for (const tensor of Object.values(out)) tensor.dispose()
      return outputs
    },
    dispose: () => session.release(),
  }
}
