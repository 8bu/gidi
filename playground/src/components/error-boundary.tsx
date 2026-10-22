import { Component, type ReactNode } from "react"

import { captureError } from "@/lib/analytics"

/** Last line of defence: a render error is reported and replaced by a reload prompt. */
export class ErrorBoundary extends Component<
  { children: ReactNode },
  { failed: boolean }
> {
  state = { failed: false }

  static getDerivedStateFromError(): { failed: boolean } {
    return { failed: true }
  }

  componentDidCatch(error: Error): void {
    captureError(error, "boot", { kind: "react_render" })
  }

  render(): ReactNode {
    if (!this.state.failed) return this.props.children
    return (
      <div role="alert" className="p-6 text-center text-sm">
        Something went wrong. / Đã có lỗi xảy ra.{" "}
        <button className="underline" onClick={() => window.location.reload()}>
          Reload / Tải lại
        </button>
      </div>
    )
  }
}
