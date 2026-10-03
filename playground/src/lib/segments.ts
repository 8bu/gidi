import type { Segment, Span } from "@/lib/api"

/** Port of `highlight_segments` / `_spans_overlap` in `scripts/demo_ui.py` (code-point spans). */
export function spansOverlap(first: Span | null | undefined, second: Span | null | undefined) {
  if (!first || !second) return false
  return first[0] < second[1] && second[0] < first[1]
}

/**
 * Split `text` at the code-point spans; segment texts always join back to `text`. Where the two
 * spans overlap the segment has role "target" and `overlap: true`.
 */
export function highlightSegments(
  text: string,
  targetSpan: Span | null | undefined,
  valueSpan: Span | null | undefined
): Segment[] {
  const points = Array.from(text)
  const cuts = new Set<number>([0, points.length])
  for (const span of [targetSpan, valueSpan]) {
    if (span) {
      cuts.add(span[0])
      cuts.add(span[1])
    }
  }
  const sorted = [...cuts].sort((a, b) => a - b)
  const segments: Segment[] = []
  for (let index = 0; index + 1 < sorted.length; index += 1) {
    const start = sorted[index]
    const end = sorted[index + 1]
    const inTarget = !!targetSpan && targetSpan[0] <= start && end <= targetSpan[1]
    const inValue = !!valueSpan && valueSpan[0] <= start && end <= valueSpan[1]
    segments.push({
      text: points.slice(start, end).join(""),
      role: inTarget ? "target" : inValue ? "value" : null,
      ...(inTarget && inValue ? { overlap: true } : {}),
    })
  }
  return segments
}
