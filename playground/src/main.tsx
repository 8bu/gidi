import { lazy, StrictMode, Suspense } from "react"
import { createRoot } from "react-dom/client"

import "./index.css"
import { ThemeProvider } from "@/components/theme-provider.tsx"
import { ErrorBoundary } from "@/components/error-boundary"
import { captureError, initAnalytics } from "@/lib/analytics"
import { setErrorContext, stripUrl, type ErrorStage } from "@/lib/error-report"
import { reloadOnce } from "@/lib/reload"
import { CLIENT_VERSION } from "@/runtime/manifest"

setErrorContext({ build: __APP_BUILD__, client_version: CLIENT_VERSION })
initAnalytics()
installBootHandlers()

// Dev server only (`VITE_ANALYTICS_DEV=1 pnpm dev` also sends): trigger each error path by hand.
if (import.meta.env.DEV) {
  ;(window as unknown as { __gidiDebug: unknown }).__gidiDebug = {
    throwUncaught: () =>
      window.setTimeout(() => {
        throw new Error("gidi debug: uncaught error")
      }),
    rejectPromise: () =>
      void Promise.reject(new Error("gidi debug: unhandled rejection")),
    capture: (stage: ErrorStage = "load") =>
      captureError(new Error("gidi debug: captured error"), stage, {
        kind: "debug",
      }),
  }
}

/**
 * A script, stylesheet or lazy chunk that fails to load (usually a stale page after a deploy)
 * is reported, and the page reloads once to pick up the new build.
 */
function installBootHandlers(): void {
  window.addEventListener("unhandledrejection", (event) => {
    captureError(event.reason, "boot", { kind: "unhandled_rejection" })
  })
  window.addEventListener(
    "error",
    (event) => {
      if (!(event.target instanceof HTMLElement)) {
        captureError(event.error ?? new Error(event.message), "boot", {
          kind: "uncaught",
        })
        return
      }
      const target = event.target
      if (
        target instanceof HTMLScriptElement ||
        target instanceof HTMLLinkElement
      ) {
        const url =
          target instanceof HTMLScriptElement ? target.src : target.href
        captureError(
          new Error(`resource failed to load: ${stripUrl(url)}`),
          "boot",
          {
            kind: "resource_load",
          }
        )
      }
    },
    true
  )
  window.addEventListener("vite:preloadError", () => {
    captureError(new Error("lazy chunk failed to load"), "boot", {
      kind: "chunk_load",
      recovery: "reload_once",
    })
    reloadOnce()
  })
}

// Each page is its own chunk: `/` never loads the developer playground and vice versa.
const App = lazy(() => import("./App.tsx"))
const NotesApp = lazy(() =>
  import("@/app/notes-app").then((module) => ({ default: module.NotesApp }))
)

const isLab = window.location.pathname.startsWith("/lab")

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ThemeProvider>
      <ErrorBoundary>
        <Suspense fallback={null}>{isLab ? <App /> : <NotesApp />}</Suspense>
      </ErrorBoundary>
    </ThemeProvider>
  </StrictMode>
)
