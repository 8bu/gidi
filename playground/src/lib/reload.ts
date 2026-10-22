/** Reload the page once per tab session (a stale page meeting a new deploy); never loops. */
export function reloadOnce(): void {
  try {
    if (window.sessionStorage.getItem("gidi:reloaded") === "1") return
    window.sessionStorage.setItem("gidi:reloaded", "1")
  } catch {
    return
  }
  window.location.reload()
}
