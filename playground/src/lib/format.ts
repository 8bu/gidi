import type { Span } from "@/lib/api"

export const pct = (confidence: number) => `${(confidence * 100).toFixed(1)}%`

/** Half-open code-point span, as returned by the runtime. */
export const spanLabel = (span: Span | null | undefined) =>
  span ? `[${span[0]}, ${span[1]})` : null

export const ms = (value: number) => `${value.toFixed(value < 10 ? 2 : 1)} ms`

/** `gidi-finance-v2` -> `Gidi Finance v2` */
export function displayName(modelVersion: string) {
  return modelVersion
    .split("-")
    .map((word) => (/^v\d/.test(word) ? word : word.charAt(0).toUpperCase() + word.slice(1)))
    .join(" ")
}
