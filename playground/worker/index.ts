/**
 * Serves the model file from R2 at `/models/<key>`. Everything else (the SPA, ORT wasm,
 * release metadata) is a static asset and never reaches this Worker (`run_worker_first`).
 */

const MODELS_PREFIX = "/models/"

/** Model keys embed the release version and the file is immutable, so cache forever. */
const CACHE_CONTROL = "public, max-age=31536000, immutable"

function notFound(): Response {
  return new Response("Not found", {
    status: 404,
    headers: {
      "content-type": "text/plain; charset=utf-8",
      "cache-control": "no-store",
    },
  })
}

function objectHeaders(object: R2Object): Headers {
  const headers = new Headers()
  object.writeHttpMetadata(headers)
  headers.set("content-type", "application/octet-stream")
  headers.set("cache-control", CACHE_CONTROL)
  headers.set("etag", object.httpEtag)
  headers.set("accept-ranges", "bytes")
  return headers
}

function contentRange(range: R2Range | undefined, size: number): string | null {
  if (!range) return null
  // R2Range is a union of {offset, length?} and {suffix}; the runtime may carry both keys.
  const { offset, length, suffix } = range as {
    offset?: number
    length?: number
    suffix?: number
  }
  let start: number
  let end: number
  if (typeof suffix === "number") {
    start = Math.max(size - suffix, 0)
    end = size - 1
  } else {
    start = offset ?? 0
    end = length === undefined ? size - 1 : Math.min(start + length - 1, size - 1)
  }
  return `bytes ${start}-${end}/${size}`
}

async function serveModel(request: Request, env: Env, key: string): Promise<Response> {
  if (request.method === "HEAD") {
    const head = await env.MODELS.head(key)
    if (head === null) return notFound()
    const headers = objectHeaders(head)
    headers.set("content-length", String(head.size))
    return new Response(null, { status: 200, headers })
  }

  const object = await env.MODELS.get(key, {
    onlyIf: request.headers,
    range: request.headers,
  })
  if (object === null) return notFound()

  const headers = objectHeaders(object)
  if (!("body" in object)) {
    // Conditional request matched (If-None-Match / If-Modified-Since): nothing to send.
    return new Response(null, { status: 304, headers })
  }
  // R2 reports `range` only for a Range request, but be explicit about which one we answered.
  const partial = request.headers.has("range") ? contentRange(object.range, object.size) : null
  if (partial === null) {
    headers.set("content-length", String(object.size))
    return new Response(object.body, { status: 200, headers })
  }
  headers.set("content-range", partial)
  return new Response(object.body, { status: 206, headers })
}

export default {
  async fetch(request, env): Promise<Response> {
    const url = new URL(request.url)
    if (!url.pathname.startsWith(MODELS_PREFIX)) return notFound()
    if (request.method !== "GET" && request.method !== "HEAD") {
      return new Response("Method not allowed", {
        status: 405,
        headers: { allow: "GET, HEAD" },
      })
    }
    let key: string
    try {
      key = decodeURIComponent(url.pathname.slice(MODELS_PREFIX.length))
    } catch {
      return notFound()
    }
    if (key === "" || key.split("/").some((part) => part === "" || part === "." || part === "..")) {
      return notFound()
    }
    return serveModel(request, env, key)
  },
} satisfies ExportedHandler<Env>
