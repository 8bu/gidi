import assert from "node:assert/strict"
import { beforeEach, test } from "node:test"

import {
  createErrorGate,
  ERROR_CAP_PER_PAGE,
  errorProperties,
  fingerprint,
  isBenign,
  isThirdPartyUrl,
  redact,
  registerSensitive,
  reportError,
  scrubException,
  scrubPageUrls,
  setErrorContext,
  setErrorSink,
} from "./error-report.ts"

const NOTE = "tra no chi Mai 500k tien nha"

test("isThirdPartyUrl separates other origins from the app's own assets", () => {
  const origin = "https://gidi.8bu.dev"
  assert.equal(
    isThirdPartyUrl("https://static.cloudflareinsights.com/beacon.min.js", origin),
    true
  )
  assert.equal(isThirdPartyUrl("https://gidi.8bu.dev/assets/index-1.js", origin), false)
  assert.equal(isThirdPartyUrl("/assets/index-1.js", origin), false)
  assert.equal(isThirdPartyUrl("", origin), false)
})

/** What posthog-js builds for an Error: a `$exception` event with `$exception_list`. */
function exceptionEvent(error: Error, props: Record<string, unknown> = {}) {
  return {
    event: "$exception",
    properties: {
      ...props,
      $exception_list: [
        {
          type: error.name,
          value: error.message,
          stacktrace: {
            type: "raw",
            frames: [
              {
                filename: "https://gidi.8bu.dev/assets/App.js?v=2#x",
                function: "classify",
                lineno: 10,
              },
              {
                filename: "https://gidi.8bu.dev/assets/worker.js",
                function: "predict",
                lineno: 99,
              },
            ],
          },
        },
      ],
    },
  }
}

beforeEach(() => {
  setErrorContext({
    build: "abc123",
    client_version: 1,
    language: "vi",
    model: "gidi-finance-v4 4.0.0",
    manifest_schema: 1,
    manifest_active: "4.0.0",
  })
  registerSensitive(NOTE, "Mai", "500k")
})

test("a thrown error reaches the capture path with context and without the note", () => {
  const sent: unknown[] = []
  setErrorSink((error, stage, extra) => {
    sent.push(
      scrubException(
        exceptionEvent(error as Error, errorProperties(stage, extra)),
        createErrorGate()
      )
    )
  })
  reportError(
    new Error(`cannot read "${NOTE}" at https://gidi.8bu.dev/x?note=${NOTE}`),
    "classify",
    {
      kind: "classify_failed",
    }
  )
  const [event] = sent as ReturnType<typeof exceptionEvent>[]
  const json = JSON.stringify(event)
  assert.ok(event, "the error was captured")
  assert.doesNotMatch(json, /tra no chi|Mai|500k|note=/)
  const p = event.properties as Record<string, unknown>
  assert.equal(p.stage, "classify")
  assert.equal(p.kind, "classify_failed")
  assert.equal(p.app_build, "abc123")
  assert.equal(p.language, "vi")
  assert.equal(p.model, "gidi-finance-v4 4.0.0")
  assert.equal(p.manifest_schema, 1)
  assert.equal(p.manifest_active, "4.0.0")
  assert.equal("online" in p, true)
  assert.equal(
    event.properties.$exception_list[0].stacktrace.frames[0].filename,
    "https://gidi.8bu.dev/assets/App.js"
  )
  setErrorSink(null)
})

test("load errors keep their message, minus URL queries and any registered note text", () => {
  const event = scrubException(
    exceptionEvent(
      new Error(
        `GET https://gidi.8bu.dev/models/a.01?token=1 failed for ${NOTE}`
      ),
      { stage: "load" }
    ),
    createErrorGate()
  )
  assert.ok(event)
  const value = event.properties.$exception_list[0].value
  assert.match(value, /GET https:\/\/gidi\.8bu\.dev\/models\/a\.01 failed/)
  assert.doesNotMatch(value, /token|tra no chi/)
})

test("classify and save errors send no message at all", () => {
  for (const stage of ["classify", "save"]) {
    const event = scrubException(
      exceptionEvent(new TypeError("boom unknown text"), { stage }),
      createErrorGate()
    )
    assert.equal(
      event?.properties.$exception_list[0].value,
      `[message omitted: ${stage}]`
    )
    assert.equal(event?.properties.$exception_list[0].type, "TypeError")
  }
})

test("SDK-made errors (window.onerror) get the context and a boot stage", () => {
  const event = scrubException(
    exceptionEvent(new Error("x is not a function")),
    createErrorGate()
  )
  const p = event?.properties as Record<string, unknown> | undefined
  assert.equal(p?.stage, "boot")
  assert.equal(p?.app_build, "abc123")
})

test("benign errors are dropped", () => {
  const drop = (error: Error, extra: Record<string, unknown> = {}) =>
    scrubException(exceptionEvent(error, extra), createErrorGate())
  assert.equal(
    drop(
      new Error("ResizeObserver loop completed with undelivered notifications.")
    ),
    null
  )
  assert.equal(drop(new Error("Script error.")), null)
  assert.equal(
    drop(
      Object.assign(new Error("The user aborted a request."), {
        name: "AbortError",
      })
    ),
    null
  )
  const ext = exceptionEvent(new Error("boom"))
  ext.properties.$exception_list[0].stacktrace.frames.push({
    filename: "chrome-extension://abc/x.js",
    function: "f",
    lineno: 1,
  })
  assert.equal(scrubException(ext, createErrorGate()), null)
  assert.equal(isBenign({ value: "Failed to fetch" }, false), true)
  assert.equal(isBenign({ value: "Failed to fetch" }, true), false)
})

test("the same error is sent once per page load, and at most ten distinct ones", () => {
  const gate = createErrorGate(() => 0)
  const first = scrubException(exceptionEvent(new Error("same 1")), gate)
  const again = scrubException(exceptionEvent(new Error("same 2")), gate) // digits fold into one fingerprint
  assert.ok(first)
  assert.equal(again, null)
  let t = 0
  const spaced = createErrorGate(() => (t += 10_000))
  let sent = 0
  for (let i = 0; i < 30; i++) {
    if (
      scrubException(
        exceptionEvent(new Error(`distinct problem ${"x".repeat(i)}`)),
        spaced
      )
    )
      sent++
  }
  assert.equal(sent, ERROR_CAP_PER_PAGE)
})

test("a burst is rate limited", () => {
  const gate = createErrorGate(() => 0)
  let sent = 0
  for (let i = 0; i < 8; i++) {
    if (
      scrubException(exceptionEvent(new Error(`burst ${"y".repeat(i)}`)), gate)
    )
      sent++
  }
  assert.equal(sent, 3)
})

test("redact and fingerprint basics; page URLs lose query and hash", () => {
  assert.equal(
    redact("see https://a.dev/p?q=1#h end"),
    "see https://a.dev/p end"
  )
  assert.equal(
    fingerprint({ type: "E", value: "id 12345 and abcdef0123456789" }),
    fingerprint({ type: "E", value: "id 9 and 0123456789abcdef" })
  )
  const props: Record<string, unknown> = {
    $current_url: "https://gidi.8bu.dev/?internal=1&x=2#a",
    $referrer: "$direct",
  }
  scrubPageUrls(props)
  assert.equal(props.$current_url, "https://gidi.8bu.dev/")
  assert.equal(props.$referrer, "$direct")
})
