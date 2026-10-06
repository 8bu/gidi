import { lazy, StrictMode, Suspense } from "react"
import { createRoot } from "react-dom/client"

import "./index.css"
import { ThemeProvider } from "@/components/theme-provider.tsx"
import { initAnalytics } from "@/lib/analytics"

initAnalytics()

// Each page is its own chunk: `/` never loads the developer playground and vice versa.
const App = lazy(() => import("./App.tsx"))
const NotesApp = lazy(() =>
  import("@/app/notes-app").then((module) => ({ default: module.NotesApp }))
)

const isLab = window.location.pathname.startsWith("/lab")

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ThemeProvider>
      <Suspense fallback={null}>{isLab ? <App /> : <NotesApp />}</Suspense>
    </ThemeProvider>
  </StrictMode>
)
