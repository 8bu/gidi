"""A tiny fake project for release tests: small pinned files, a spec, a template.

``make_fake_project(tmp_path)`` writes a project root with a bundle, an FP32 stand-in, two
protocols and a README template, plus a committed-style spec for the model ``fake-v1``. Model
files are a few random-looking bytes unless ``onnx_bundle`` points at a real value-head bundle
(e.g. the ``v2_bundle`` fixture); then smoke inference works and the spec's smoke cases hold.

    project = make_fake_project(tmp_path)
    release_dir = project.build("1.0.0")          # default dist root: tmp_path / "out"
    report = verify_release(release_dir, spec=project.spec, root=project.root, smoke=False)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gidi.release import ReleaseSpec, build_release, load_spec, release_dir_for

MODEL = "fake-v1"
SMOKE_NOTES = ("mượn chú hai 5 xị", "ăn 2 tô phở 70")
PIN_CHECKPOINT = "c" * 64


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fake_config(model_version: str = MODEL) -> dict[str, Any]:
    return {
        "model_version": model_version,
        "annotation_version": "annotation-v2",
        "types": ["expense", "income", "borrow", "lend"],
        "tags": ["O", "B-TARGET", "I-TARGET"],
        "value_labels": ["O", "B-VALUE", "I-VALUE"],
        "value_decoding": {
            "method": "crf_viterbi",
            "start_transitions": [0.0, 0.0, 0.0],
            "end_transitions": [0.0, 0.0, 0.0],
            "transitions": [[0.0] * 3 for _ in range(3)],
            "span_selection": "highest_confidence",
            "confidence": "geometric mean",
        },
        "max_length": 32,
        "onnx": {
            "inputs": ["input_ids", "attention_mask"],
            "outputs": ["type_logits", "tag_logits", "value_logits"],
            "opset": 17,
        },
    }


def _write(path: Path, data: bytes) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


@dataclass
class FakeProject:
    root: Path
    spec_path: Path
    dist: Path

    @property
    def spec(self) -> ReleaseSpec:
        return load_spec(self.spec_path)

    def release_dir(self, version: str, dist: Path | None = None) -> Path:
        return release_dir_for(MODEL, version, dist or self.dist)

    def build(self, version: str = "1.0.0", dist: Path | None = None, **kwargs: Any) -> Path:
        """Build into ``dist`` (default ``self.dist``); unversioned unless told otherwise."""
        kwargs.setdefault("allow_unversioned", True)
        return build_release(
            self.spec,
            version,
            root=self.root,
            out_dir=self.release_dir(version, dist),
            **kwargs,
        )

    def edit_spec(self, mutate) -> Path:
        """Write a modified copy of the spec next to it and return its path."""
        data = json.loads(self.spec_path.read_text(encoding="utf-8"))
        mutate(data)
        path = self.spec_path.with_name("edited.json")
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path


def make_fake_project(
    tmp_path: Path,
    *,
    onnx_bundle: Path | None = None,
    template: str | None = None,
) -> FakeProject:
    root = tmp_path / "project"
    bundle = root / "models" / MODEL
    fp32_dir = root / "models" / f"{MODEL}-onnx"
    config = json.dumps(fake_config(), ensure_ascii=False, indent=2) + "\n"

    sources: dict[str, bytes] = {}
    if onnx_bundle is None:
        sources["model.int8.onnx"] = b"int8-onnx-bytes" * 50
        fp32 = b"fp32-onnx-bytes" * 80
        sources["tokenizer.json"] = b'{"tokenizer": true}\n'
        sources["tokenizer_config.json"] = b'{"cfg": 1}\n'
        sources["vocab_map.json"] = b'{"0": 0}\n'
    else:
        sources["model.int8.onnx"] = fp32 = (onnx_bundle / "model.int8.onnx").read_bytes()
        for name in ("tokenizer.json", "tokenizer_config.json", "vocab_map.json"):
            sources[name] = (onnx_bundle / name).read_bytes()
    sources["config.json"] = config.encode("utf-8")
    for name, data in sources.items():
        _write(bundle / name, data)
    _write(fp32_dir / "model.onnx", fp32)

    experiment_protocol = _write(root / "experiments" / "exp" / "protocol.json", b'{"exp": 1}\n')
    deployment_protocol = _write(root / "experiments" / "dep" / "protocol.json", b'{"dep": 1}\n')
    requirements = {"python": ">=3.12", "onnxruntime": ">=1.30.0"}
    bundle_manifest = {
        "model_version": MODEL,
        "protocol": {
            "path": "experiments/dep/protocol.json",
            "sha256": sha256_bytes(deployment_protocol),
        },
        "sources": {
            "checkpoint_model_safetensors": {"sha256": PIN_CHECKPOINT},
            "source_experiment_protocol": {"sha256": sha256_bytes(experiment_protocol)},
            "onnx_fp32": {"sha256": sha256_bytes(fp32)},
        },
        "files": {name: {"sha256": sha256_bytes(data)} for name, data in sources.items()},
        "runtime_requirements": requirements,
    }
    manifest_bytes = _write(
        bundle / "manifest.json",
        (json.dumps(bundle_manifest, indent=2) + "\n").encode("utf-8"),
    )

    template_text = template or (
        "---\nlanguage: vi\nlibrary_name: onnx\ntags:\n  - test\n---\n\n"
        "# {model_version} {release_version}\n\nint8 {int8_size_mb} `{int8_sha256}`\n"
        "fp32 {fp32_size_mb}; archives {int8_archive} {fp32_archive}; max {max_length}\n"
    )
    _write(root / "releases" / "templates" / f"{MODEL}.README.md", template_text.encode("utf-8"))

    def entry(source: str, path: str, kind: str, data: bytes) -> dict[str, Any]:
        return {
            "source": source,
            "path": path,
            "kind": kind,
            "sha256": sha256_bytes(data),
            "size_bytes": len(data),
        }

    b = f"models/{MODEL}"
    files = [
        entry(
            f"{b}/model.int8.onnx",
            "model/int8/model.int8.onnx",
            "onnx-int8",
            sources["model.int8.onnx"],
        ),
        entry(f"{b}-onnx/model.onnx", "model/fp32/model.onnx", "onnx-fp32", fp32),
        entry(
            f"{b}/tokenizer.json",
            "tokenizer/tokenizer.json",
            "tokenizer",
            sources["tokenizer.json"],
        ),
        entry(
            f"{b}/tokenizer_config.json",
            "tokenizer/tokenizer_config.json",
            "tokenizer-config",
            sources["tokenizer_config.json"],
        ),
        entry(
            f"{b}/vocab_map.json",
            "tokenizer/vocab_map.json",
            "vocab-map",
            sources["vocab_map.json"],
        ),
        entry(f"{b}/config.json", "runtime/config.json", "runtime-config", sources["config.json"]),
    ]
    common = {
        "config.json": "runtime/config.json",
        "tokenizer.json": "tokenizer/tokenizer.json",
        "tokenizer_config.json": "tokenizer/tokenizer_config.json",
        "vocab_map.json": "tokenizer/vocab_map.json",
        "README.md": "README.md",
        "release-manifest.json": "manifest.json",
    }
    smoke: list[dict[str, Any]] = []
    if onnx_bundle is not None:
        from gidi.inference.predictor import GidiPredictor

        predictor = GidiPredictor.from_bundle(bundle)
        for text in SMOKE_NOTES:
            got = predictor.predict(text).to_dict()
            expected = {key: got[key] for key in ("type", "target", "value_text")}
            smoke.append({"text": text, "expected": expected})
    spec = {
        "spec_format": 1,
        "model_version": MODEL,
        "source": {
            "experiment": "fake-experiment",
            "seed": 1,
            "experiment_protocol": {
                "path": "experiments/exp/protocol.json",
                "sha256": sha256_bytes(experiment_protocol),
            },
            "deployment_protocol": {
                "path": "experiments/dep/protocol.json",
                "sha256": sha256_bytes(deployment_protocol),
            },
            "bundle": {"path": b, "manifest_sha256": sha256_bytes(manifest_bytes)},
            "checkpoint_sha256": PIN_CHECKPOINT,
        },
        "files": files,
        "readmes": {"README.md": f"releases/templates/{MODEL}.README.md"},
        "variants": {
            "int8": {
                "members": {"model.int8.onnx": "model/int8/model.int8.onnx", **common},
                "model_file": "model.int8.onnx",
            },
            "fp32": {
                "members": {"model.onnx": "model/fp32/model.onnx", **common},
                "model_file": "model.onnx",
            },
        },
        "smoke": smoke,
    }
    spec_path = root / "releases" / f"{MODEL}.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return FakeProject(root=root, spec_path=spec_path, dist=tmp_path / "out")
