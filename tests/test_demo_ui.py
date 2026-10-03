"""Playground server: HTTP contract, offset-safe highlighting, one predictor load, 500 isolation."""

from __future__ import annotations

import http.client
import importlib.util
import json
import threading
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from gidi.inference import GidiPredictor

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "models" / "gidi-finance-v1"

pytestmark = pytest.mark.skipif(
    not (BUNDLE / "model.int8.onnx").is_file(), reason="deployment bundle not built"
)

_SPEC = importlib.util.spec_from_file_location("demo_ui", ROOT / "scripts" / "demo_ui.py")
demo_ui = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(demo_ui)

HTML = "<!doctype html><title>stub</title>"
NOTE = "trả nợ chị Mai 500k"


def write_dist(root: Path) -> Path:
    """A stand-in for ``playground/dist`` with a secret one level above it."""
    dist = root / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text(HTML, encoding="utf-8")
    (dist / "assets" / "app-1a2b.js").write_text("export const x = 1\n", encoding="utf-8")
    (dist / "assets" / "app-1a2b.css").write_text("body{}\n", encoding="utf-8")
    (dist / "assets" / "font-1a2b.woff2").write_bytes(b"wOF2\x00\x01")
    (root / "secret.txt").write_text("top secret", encoding="utf-8")
    return dist


def serve(predictor, dist=None, runtime=None):
    server = demo_ui.make_server(
        predictor, "127.0.0.1", 0, dist, "models/gidi-finance-v1", runtime=runtime
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def stop(server, thread):
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def url(server, path):
    return f"http://127.0.0.1:{server.server_address[1]}{path}"


def request(server, path, *, body=None, method=None, headers=None):
    """Return ``(status, parsed_json_or_text)`` for any HTTP status."""
    data = body if isinstance(body, bytes) or body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url(server, path),
        data=data,
        method=method,
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            status, raw = response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        status, raw = exc.code, exc.read().decode("utf-8")
    return status, json.loads(raw) if raw.startswith(("{", "[")) else raw


def predict(server, text):
    return request(server, "/api/predict", body={"text": text})


@pytest.fixture(scope="module")
def predictor():
    return GidiPredictor.from_bundle(BUNDLE)


@pytest.fixture(scope="module")
def dist_dir(tmp_path_factory):
    return write_dist(tmp_path_factory.mktemp("playground"))


@pytest.fixture(scope="module")
def server(predictor, dist_dir):
    srv, thread = serve(predictor, dist_dir)
    yield srv
    stop(srv, thread)


def test_predict_returns_prediction_latency_and_segments(server):
    status, body = predict(server, NOTE)
    assert status == 200 and body["ok"] is True
    assert {"type", "type_confidence", "target", "target_span", "model_version"} <= set(
        body["result"]
    )
    assert body["result"]["model_version"] == "gidi-finance-v1"
    assert body["latency_ms"] >= 0
    assert "".join(s["text"] for s in body["segments"]) == NOTE


def test_info_reports_the_loaded_bundle(server):
    assert request(server, "/api/info") == (
        200,
        {
            "model_version": "gidi-finance-v1",
            "bundle": "models/gidi-finance-v1",
            "max_length": 32,
            "has_value_head": False,
        },
    )


def test_info_adds_runtime_facts_for_the_ui(predictor):
    runtime = demo_ui.describe_bundle(BUNDLE)
    assert runtime["precision"] == "int8" and runtime["backend"] == "onnxruntime CPU"
    assert runtime["model_file"] == "model.int8.onnx"
    assert runtime["bundle_path"] == str(BUNDLE.resolve())
    srv, thread = serve(predictor, runtime=runtime)
    try:
        status, info = request(srv, "/api/info")
    finally:
        stop(srv, thread)
    assert status == 200
    assert info["model_version"] == "gidi-finance-v1" and info["has_value_head"] is False
    assert {key: info[key] for key in runtime} == runtime


def test_describe_bundle_of_a_value_bundle_names_the_architecture(v2_bundle):
    runtime = demo_ui.describe_bundle(v2_bundle)
    assert runtime["architecture"] == json.loads((v2_bundle / "config.json").read_text())[
        "architecture"
    ].get("kind")
    assert runtime["onnx_opset"] == 17


def test_describe_bundle_tolerates_a_missing_config(tmp_path):
    runtime = demo_ui.describe_bundle(tmp_path)
    assert runtime["architecture"] is None and runtime["onnx_opset"] is None


def fetch(server, path, method="GET"):
    """Raw request line (no client-side path normalisation): ``(status, headers, body)``."""
    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
    conn.request(method, path)
    response = conn.getresponse()
    result = response.status, response, response.read()
    conn.close()
    return result


def test_index_and_assets_are_served_with_their_content_types(server):
    status, response, body = fetch(server, "/")
    assert (status, body.decode()) == (200, HTML)
    assert response.getheader("Content-Type") == "text/html; charset=utf-8"
    assert response.getheader("Cache-Control") == "no-store"
    cases = {
        "/assets/app-1a2b.js": "text/javascript; charset=utf-8",
        "/assets/app-1a2b.css": "text/css; charset=utf-8",
        "/assets/font-1a2b.woff2": "font/woff2",
    }
    for path, content_type in cases.items():
        status, response, body = fetch(server, path)
        assert status == 200 and body
        assert response.getheader("Content-Type") == content_type
        assert "immutable" in response.getheader("Cache-Control")
    assert fetch(server, "/assets/app-1a2b.js")[2] == b"export const x = 1\n"


@pytest.mark.parametrize(
    "path",
    [
        "/../secret.txt",
        "/assets/../../secret.txt",
        "/%2e%2e/secret.txt",
        "/assets/%2e%2e/%2e%2e/secret.txt",
        "/assets/..%2f..%2fsecret.txt",
        "/assets/..%5c..%5csecret.txt",
        "/assets/app-1a2b.js%00.png",
        "//etc/passwd",
    ],
)
def test_static_paths_cannot_escape_the_dist_directory(server, path):
    status, _, body = fetch(server, path)
    assert status == 404 and b"top secret" not in body


def test_only_the_root_falls_back_to_the_index(server):
    for path in ("/assets", "/assets/", "/assets/missing.js", "/some/client/route", "/index.htm"):
        status, _, body = fetch(server, path)
        assert status == 404 and json.loads(body)["error"]["code"] == "not_found"
    status, _, body = fetch(server, "/", method="POST")
    assert status == 405 and json.loads(body)["error"]["code"] == "method_not_allowed"


@pytest.mark.parametrize("dist", [None, "empty"])
def test_missing_frontend_build_is_a_clear_503_and_the_api_still_works(predictor, tmp_path, dist):
    srv, thread = serve(predictor, None if dist is None else tmp_path)
    try:
        status, payload = request(srv, "/")
        assert (status, payload["error"]["code"]) == (503, "frontend_not_built")
        assert "pnpm -C playground build" in payload["error"]["message"]
        assert request(srv, "/assets/app.js")[0] == 404
        assert predict(srv, NOTE)[0] == 200
    finally:
        stop(srv, thread)


@pytest.mark.parametrize("text", ["", "   ", "\n\t "])
def test_blank_text_is_empty_input(server, text):
    status, body = predict(server, text)
    assert (status, body["ok"], body["error"]["code"]) == (400, False, "empty_input")


@pytest.mark.parametrize(
    ("body", "code"),
    [
        (b"{not json", "invalid_json"),
        (b"\xff\xfe", "invalid_json"),
        ([NOTE], "invalid_request"),
        ({"note": NOTE}, "invalid_request"),
        ({"text": 123}, "text_not_string"),
        ({"text": None}, "text_not_string"),
    ],
)
def test_malformed_bodies_get_specific_400s(server, body, code):
    status, payload = request(server, "/api/predict", body=body)
    assert (status, payload["error"]["code"]) == (400, code)


def test_oversized_content_length_is_rejected_before_reading(server):
    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
    conn.putrequest("POST", "/api/predict")
    conn.putheader("Content-Type", "application/json")
    conn.putheader("Content-Length", str(64 * 1024 + 1))
    conn.endheaders()  # no body is ever sent
    response = conn.getresponse()
    assert response.status == 413
    assert json.loads(response.read())["error"]["code"] == "too_large"
    conn.close()


def test_missing_content_length_and_wrong_media_type(server):
    conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=10)
    conn.putrequest("POST", "/api/predict")
    conn.putheader("Content-Type", "application/json")
    conn.endheaders()
    response = conn.getresponse()
    assert response.status == 411
    assert json.loads(response.read())["error"]["code"] == "invalid_request"
    conn.close()

    status, payload = request(
        server,
        "/api/predict",
        body=b'{"text": "x"}',
        headers={"Content-Type": "text/plain"},
    )
    assert (status, payload["error"]["code"]) == (415, "unsupported_media_type")


def test_unknown_path_and_wrong_method(server):
    assert request(server, "/etc/passwd")[0] == 404
    assert request(server, "/nope", body={"text": "x"})[1]["error"]["code"] == "not_found"
    status, payload = request(server, "/api/predict")  # GET on a POST-only path
    assert (status, payload["error"]["code"]) == (405, "method_not_allowed")
    status, payload = request(server, "/api/info", body={})
    assert (status, payload["error"]["code"]) == (405, "method_not_allowed")


def test_highlight_without_span_is_one_plain_segment():
    assert demo_ui.highlight_segments("ăn phở 45k", None) == [{"text": "ăn phở 45k", "role": None}]


@pytest.mark.parametrize(
    "text",
    [
        NOTE,
        "😀 " + NOTE,
        "🎉😀 cho Tuấn vay 5 triệu",
        unicodedata.normalize("NFD", NOTE),
        unicodedata.normalize("NFD", "😀 Tuấn cho vay 5 triệu"),
    ],
)
def test_segments_keep_codepoint_offsets(server, predictor, text):
    result = predictor.predict(text).to_dict()
    segments = demo_ui.highlight_segments(text, result["target_span"])
    status, body = predict(server, text)
    assert status == 200 and body["result"] == result
    for got in (segments, body["segments"]):
        assert "".join(s["text"] for s in got) == text
        assert {s["role"] for s in got} <= {None, "target"}
        targets = [s["text"] for s in got if s["role"] == "target"]
        if result["target_span"] is None:
            assert targets == [] and len(got) == 1
        else:
            start, end = result["target_span"]
            assert targets == [text[start:end]]


def test_v1_bundle_has_no_value_keys_anywhere(server):
    assert request(server, "/api/info")[1]["has_value_head"] is False
    body = predict(server, NOTE)[1]
    assert not any(key.startswith("value") for key in body["result"])
    assert "spans_overlap" not in body


def roles(segments):
    return [(s["text"], s["role"], s.get("overlap", False)) for s in segments]


def test_highlight_two_disjoint_spans_and_adjacent_spans():
    text = "😀 mượn chú hai 5 xị"
    assert roles(demo_ui.highlight_segments(text, [2, 6], [15, 19])) == [
        ("😀 ", None, False),
        ("mượn", "target", False),
        (" chú hai ", None, False),
        ("5 xị", "value", False),
    ]
    assert roles(demo_ui.highlight_segments("ab12", [0, 2], [2, 4])) == [
        ("ab", "target", False),
        ("12", "value", False),
    ]
    assert roles(demo_ui.highlight_segments("cơm 100", None, [4, 7])) == [
        ("cơm ", None, False),
        ("100", "value", False),
    ]


@pytest.mark.parametrize(
    ("target", "value", "expected"),
    [
        ([2, 8], [5, 10], [("ab", None), ("cde", "target"), ("fgh", "target"), ("ij", "value")]),
        ([2, 5], [2, 5], [("ab", None), ("cde", "target"), ("fghij", None)]),
        ([0, 10], [3, 6], [("abc", "target"), ("def", "target"), ("ghij", "target")]),
    ],
)
def test_overlapping_spans_render_as_target_and_are_flagged(target, value, expected):
    text = "abcdefghij"
    got = demo_ui.highlight_segments(text, target, value)
    assert "".join(s["text"] for s in got) == text
    assert [(s["text"], s["role"]) for s in got] == expected
    flagged = {s["text"] for s in got if s.get("overlap")}
    shared = text[max(target[0], value[0]) : min(target[1], value[1])]
    assert flagged == {shared}


class SpanStub:
    """Predictor stand-in whose ``predict(...).to_dict()`` returns a fixed result."""

    model_version = "stub-v2"
    has_value_head = True

    def __init__(self, result):
        self.result = result

    def predict(self, text):
        return self

    def to_dict(self):
        return self.result


def test_handle_predict_reports_overlapping_spans_unchanged():
    result = {
        "type": "expense",
        "target": "cde",
        "target_span": [2, 5],
        "value_text": "efg",
        "value_span": [4, 7],
    }
    body = json.dumps({"text": "abcdefghij"}).encode()
    status, payload = demo_ui.handle_predict(SpanStub(result), body)
    assert status == 200 and payload["result"] == result
    assert payload["spans_overlap"] is True
    assert roles(payload["segments"]) == [
        ("ab", None, False),
        ("cd", "target", False),
        ("e", "target", True),
        ("fg", "value", False),
        ("hij", None, False),
    ]


def test_predictor_is_loaded_once_per_process(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(demo_ui, "DIST_DIR", write_dist(tmp_path))
    calls = []
    real = GidiPredictor.from_bundle

    def counting(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(demo_ui.GidiPredictor, "from_bundle", counting)
    ready, servers = threading.Event(), []

    def on_ready(srv):
        servers.append(srv)
        ready.set()

    argv = ["--bundle", str(BUNDLE), "--port", "0"]
    thread = threading.Thread(target=demo_ui.main, args=(argv, on_ready), daemon=True)
    thread.start()
    assert ready.wait(timeout=60)
    srv = servers[0]
    try:
        for text in (NOTE, "ăn phở 45k", "Tuấn cho vay 5 triệu"):
            assert predict(srv, text)[0] == 200
        _, info = request(srv, "/api/info")
        assert info["max_length"] == 32 and info["bundle_path"] == str(BUNDLE.resolve())
        assert (info["precision"], info["backend"]) == ("int8", "onnxruntime CPU")
        assert request(srv, "/")[1] == HTML
    finally:
        srv.shutdown()
        thread.join(timeout=10)
    assert len(calls) == 1
    assert f"Gidi playground: http://127.0.0.1:{srv.server_address[1]}" in capsys.readouterr().out


class ExplodingPredictor:
    model_version = "stub"

    def predict(self, text):
        raise RuntimeError("boom")


def test_inference_failure_is_a_500_and_the_server_keeps_serving():
    srv, thread = serve(ExplodingPredictor())
    try:
        for _ in range(2):
            status, payload = predict(srv, NOTE)
            assert status == 500
            assert payload["error"] == {
                "code": "inference_error",
                "message": "RuntimeError: boom",
            }
        assert request(srv, "/api/info")[0] == 200
    finally:
        stop(srv, thread)


class MissingArtifactPredictor:
    model_version = "stub"

    def predict(self, text):
        raise FileNotFoundError("model.int8.onnx")


def test_a_vanished_bundle_file_is_a_503_missing_artifact():
    srv, thread = serve(MissingArtifactPredictor())
    try:
        status, payload = predict(srv, NOTE)
        assert status == 503
        assert payload["error"]["code"] == "missing_artifact"
        assert "model.int8.onnx" in payload["error"]["message"]
        assert request(srv, "/api/info")[0] == 200
    finally:
        stop(srv, thread)


@pytest.fixture(scope="module")
def v2_server(v2_bundle):
    srv, thread = serve(GidiPredictor.from_bundle(v2_bundle))
    yield srv
    stop(srv, thread)


def test_value_bundle_playground_exposes_value_and_two_role_segments(v2_server):
    assert request(v2_server, "/api/info")[1]["has_value_head"] is True
    text = "😀 trả nợ chị Mai 500k"
    status, body = predict(v2_server, text)
    assert status == 200
    result = body["result"]
    assert {"value_text", "value_span", "value_confidence"} <= set(result)
    assert body["spans_overlap"] is False
    assert "".join(s["text"] for s in body["segments"]) == text
    by_role = {s["role"]: s["text"] for s in body["segments"] if s["role"]}
    assert by_role == {
        "target": text[slice(*result["target_span"])],
        "value": text[slice(*result["value_span"])],
    }
    assert by_role["value"] == "chị"  # the fixture's value span: the token before the target


def test_value_bundle_segments_without_a_value_span_are_target_only(v2_server):
    status, body = predict(v2_server, "ăn phở 45k")
    assert status == 200 and body["result"]["value_span"] is None
    assert {s["role"] for s in body["segments"]} <= {None, "target"}
