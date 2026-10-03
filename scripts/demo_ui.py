"""Local playground server for the ``gidi-finance`` runtime (default bundle: v2).

Stdlib ``http.server``: the built React app from ``playground/dist`` (build it first, see
``docs/deployment.md``), ``GET /api/info`` and ``POST /api/predict``. The predictor is loaded
once at startup and called as-is; this script adds no thresholds, rules, overrides or
persistence.

    uv run python scripts/demo_ui.py [--bundle DIR] [--host 127.0.0.1] [--port 8765]
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import threading
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from gidi.inference import BundleError, EmptyInputError, GidiPredictor
from gidi.inference.bundle import CONFIG_FILE, MODEL_FILE, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = "models/gidi-finance-v2"
DIST_DIR = REPO_ROOT / "playground" / "dist"
MAX_BODY_BYTES = 64 * 1024
DEFAULT_PORT = 8765
BUILD_HINT = "run: pnpm -C playground install && pnpm -C playground build"

Payload = dict[str, Any]

# Explicit so a host's mime database cannot serve a module script as text/plain.
CONTENT_TYPES = {
    ".html": "text/html",
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".css": "text/css",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".map": "application/json",
    ".txt": "text/plain",
    ".woff2": "font/woff2",
    ".woff": "font/woff",
    ".png": "image/png",
    ".ico": "image/x-icon",
}


def content_type_for(path: Path) -> str:
    kind = CONTENT_TYPES.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0]
    kind = kind or "application/octet-stream"
    return f"{kind}; charset=utf-8" if kind.startswith("text/") or kind.endswith("json") else kind


def static_file(dist_dir: Path | None, url_path: str) -> Path | None:
    """Map a URL path to a regular file inside ``dist_dir``; ``/`` is ``index.html``.

    Returns ``None`` for anything else, including paths that resolve outside ``dist_dir``
    (``..`` segments, percent-encoded separators, symlinks). There is no SPA fallback.
    """
    if dist_dir is None:
        return None
    relative = "index.html" if url_path == "/" else unquote(url_path).lstrip("/")
    if not relative or "\x00" in relative or "\\" in relative:
        return None
    root = dist_dir.resolve()
    try:
        candidate = (root / relative).resolve()
    except (OSError, ValueError):
        return None
    if not candidate.is_relative_to(root) or not candidate.is_file():
        return None
    return candidate


def describe_bundle(bundle_dir: Path) -> Payload:
    """Runtime facts for ``/api/info`` read from the bundle's ``config.json``; best effort."""
    try:
        raw = json.loads((bundle_dir / CONFIG_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = {}
    architecture = raw.get("architecture")
    onnx = raw.get("onnx")
    return {
        "architecture": architecture.get("kind") if isinstance(architecture, dict) else None,
        "precision": "int8" if "int8" in MODEL_FILE else None,
        "backend": "onnxruntime CPU",
        "model_file": MODEL_FILE,
        "onnx_opset": onnx.get("opset") if isinstance(onnx, dict) else None,
        "bundle_path": str(bundle_dir.resolve()),
    }


def _spans_overlap(first: Sequence[int] | None, second: Sequence[int] | None) -> bool:
    if first is None or second is None:
        return False
    return first[0] < second[1] and second[0] < first[1]


def highlight_segments(
    text: str, target_span: Sequence[int] | None, value_span: Sequence[int] | None = None
) -> list[dict[str, Any]]:
    """Split ``text`` at the code-point spans; segment texts always join back to ``text``.

    Each segment has ``text`` and ``role``: ``None``, ``"target"`` or ``"value"``. The model's two
    heads are independent, so the spans can overlap. The overlap is not hidden: it is one segment
    with role ``"target"`` (target takes precedence) and ``"overlap": true``; the result dict
    still carries both spans unchanged.
    """
    cuts = {0, len(text)}
    for span in (target_span, value_span):
        if span is not None:
            cuts.update(span)
    points = sorted(cuts)
    segments: list[dict[str, Any]] = []
    for start, end in zip(points, points[1:], strict=False):
        in_target = target_span is not None and target_span[0] <= start and end <= target_span[1]
        in_value = value_span is not None and value_span[0] <= start and end <= value_span[1]
        segment: dict[str, Any] = {
            "text": text[start:end],
            "role": "target" if in_target else "value" if in_value else None,
        }
        if in_target and in_value:
            segment["overlap"] = True
        segments.append(segment)
    return segments


def _error(status: int, code: str, message: str) -> tuple[int, Payload]:
    return status, {"ok": False, "error": {"code": code, "message": message}}


def handle_predict(predictor: Any, body: bytes, lock: Any = None) -> tuple[int, Payload]:
    """Validate a ``/api/predict`` body and run the predictor; returns ``(status, payload)``."""
    try:
        request = json.loads(body.decode("utf-8"))
    except (ValueError, RecursionError):  # JSONDecodeError and UnicodeDecodeError are ValueErrors
        return _error(400, "invalid_json", "request body is not valid UTF-8 JSON")
    if not isinstance(request, dict) or "text" not in request:
        return _error(400, "invalid_request", 'body must be a JSON object with a "text" field')
    text = request["text"]
    if not isinstance(text, str):
        return _error(400, "text_not_string", '"text" must be a string')
    try:
        with lock if lock is not None else nullcontext():
            started = time.perf_counter()
            result = predictor.predict(text).to_dict()
            latency_ms = (time.perf_counter() - started) * 1000.0
    except EmptyInputError as exc:
        return _error(400, "empty_input", str(exc) or "text is empty")
    except (BundleError, FileNotFoundError) as exc:  # a bundle artifact vanished after startup
        traceback.print_exc(file=sys.stderr)
        return _error(503, "missing_artifact", f"{type(exc).__name__}: {exc}")
    except Exception as exc:  # the server must survive any inference failure
        traceback.print_exc(file=sys.stderr)
        return _error(500, "inference_error", f"{type(exc).__name__}: {exc}")
    payload: Payload = {
        "ok": True,
        "result": result,
        "latency_ms": latency_ms,
        "segments": highlight_segments(text, result["target_span"], result.get("value_span")),
    }
    if "value_span" in result:  # bundle with a value head
        payload["spans_overlap"] = _spans_overlap(result["target_span"], result["value_span"])
    return 200, payload


def _make_handler(
    predictor: Any, dist_dir: Path | None, info: Payload
) -> type[BaseHTTPRequestHandler]:
    lock = threading.Lock()  # one ORT session shared by all request threads
    api_routes = {"/api/info": "GET", "/api/predict": "POST"}

    class Handler(BaseHTTPRequestHandler):
        server_version = "GidiPlayground"

        def _send(self, status: int, content_type: str, data: bytes, **headers: str) -> None:
            headers.setdefault("Cache_Control", "no-store")
            try:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Connection", "close")
                for name, value in headers.items():
                    self.send_header(name.replace("_", "-"), value)
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass  # client went away; nothing to report

        def _send_json(self, status: int, payload: Payload, **headers: str) -> None:
            try:
                data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            except UnicodeEncodeError:  # lone surrogate in the echoed text: escape it instead
                data = json.dumps(payload, ensure_ascii=True).encode("ascii")
            self._send(status, "application/json; charset=utf-8", data, **headers)

        def _read_body(self) -> bytes | None:
            """Return the request body, or ``None`` after sending the error response."""
            content_type = self.headers.get("Content-Type")
            mime = None if content_type is None else content_type.split(";", 1)[0].strip().lower()
            if mime is not None and mime != "application/json":
                status, payload = _error(
                    415, "unsupported_media_type", "Content-Type must be application/json"
                )
                self._send_json(status, payload)
                return None
            raw_length = self.headers.get("Content-Length")
            if raw_length is None:
                status, payload = _error(411, "invalid_request", "Content-Length is required")
                self._send_json(status, payload)
                return None
            try:
                length = int(raw_length)
                if length < 0:
                    raise ValueError
            except ValueError:
                status, payload = _error(400, "invalid_request", "invalid Content-Length")
                self._send_json(status, payload)
                return None
            if length > MAX_BODY_BYTES:
                status, payload = _error(
                    413, "too_large", f"request body exceeds {MAX_BODY_BYTES} bytes"
                )
                self._send_json(status, payload)
                return None
            return self.rfile.read(length)

        def _serve_static(self, url_path: str, file: Path) -> None:
            try:
                data = file.read_bytes()
            except OSError:
                status, payload = _error(404, "not_found", f"no such path: {url_path}")
                self._send_json(status, payload)
                return
            headers = {}
            if url_path.startswith("/assets/"):  # Vite fingerprints these file names
                headers["Cache_Control"] = "public, max-age=31536000, immutable"
            self._send(200, content_type_for(file), data, **headers)

        def _dispatch(self) -> None:
            path = self.path.split("?", 1)[0].split("#", 1)[0]
            file = static_file(dist_dir, path)
            allowed = api_routes.get(path, "GET" if file is not None else None)
            if path == "/" and file is None:
                status, payload = _error(
                    503, "frontend_not_built", f"playground/dist is missing; {BUILD_HINT}"
                )
                self._send_json(status, payload)
            elif allowed is None:
                status, payload = _error(404, "not_found", f"no such path: {path}")
                self._send_json(status, payload)
            elif self.command != allowed:
                status, payload = _error(
                    405, "method_not_allowed", f"{path} only accepts {allowed}"
                )
                self._send_json(status, payload, Allow=allowed)
            elif path == "/api/info":
                self._send_json(200, info)
            elif path == "/api/predict":
                body = self._read_body()
                if body is not None:
                    self._send_json(*handle_predict(predictor, body, lock))
            else:
                assert file is not None
                self._serve_static(path, file)

        do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_HEAD = do_OPTIONS = _dispatch

        def log_request(self, code: Any = "-", size: Any = "-") -> None:
            sys.stderr.write(f"{self.command} {self.path} {code}\n")

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            sys.stderr.write((format % args) + "\n")

    return Handler


def make_server(
    predictor: Any,
    host: str,
    port: int,
    dist_dir: Path | None,
    bundle: str,
    max_length: int = 32,
    runtime: Mapping[str, Any] | None = None,
) -> ThreadingHTTPServer:
    """Build (bind, not serve) the playground server around an already-loaded ``predictor``.

    ``dist_dir`` is the built frontend (``None`` or a missing ``index.html`` makes ``/`` answer
    503 ``frontend_not_built``); ``runtime`` adds fields (see ``describe_bundle``) to ``/api/info``.
    """
    info: Payload = {
        "model_version": getattr(predictor, "model_version", "gidi-finance-v1"),
        "bundle": bundle,
        "max_length": max_length,
        "has_value_head": bool(getattr(predictor, "has_value_head", False)),
        **(runtime or {}),
    }
    return ThreadingHTTPServer((host, port), _make_handler(predictor, dist_dir, info))


def main(
    argv: Sequence[str] | None = None,
    ready: Callable[[ThreadingHTTPServer], None] | None = None,
) -> int:
    """Load the predictor once, serve until Ctrl-C. ``ready(server)`` runs just before serving."""
    parser = argparse.ArgumentParser(description="Gidi finance model playground (local only).")
    parser.add_argument(
        "--bundle", default=None, help=f"bundle directory (default: {DEFAULT_BUNDLE} in the repo)"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)

    bundle = args.bundle or DEFAULT_BUNDLE
    bundle_dir = Path(args.bundle) if args.bundle else REPO_ROOT / DEFAULT_BUNDLE
    try:
        max_length = load_config(bundle_dir).max_length
        predictor = GidiPredictor.from_bundle(bundle_dir)
    except (BundleError, OSError) as exc:
        parser.error(f"cannot load bundle {bundle_dir}: {exc}")

    if not (DIST_DIR / "index.html").is_file():
        print(f"warning: {DIST_DIR} has no index.html; {BUILD_HINT}", file=sys.stderr, flush=True)
    server = make_server(
        predictor, args.host, args.port, DIST_DIR, bundle, max_length, describe_bundle(bundle_dir)
    )
    try:
        print(f"Gidi playground: http://{args.host}:{server.server_address[1]}", flush=True)
        if ready is not None:
            ready(server)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
