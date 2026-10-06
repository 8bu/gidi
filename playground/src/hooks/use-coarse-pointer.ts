import { useSyncExternalStore } from "react"

const COARSE_QUERY = "(pointer: coarse)"

function subscribeCoarse(onChange: () => void) {
  const query = window.matchMedia(COARSE_QUERY)
  query.addEventListener("change", onChange)
  return () => query.removeEventListener("change", onChange)
}

/** True when the primary pointer is a finger (phones, tablets); reacts to input-mode changes. */
export function useCoarsePointer(): boolean {
  return useSyncExternalStore(
    subscribeCoarse,
    () => window.matchMedia(COARSE_QUERY).matches,
    () => false
  )
}
