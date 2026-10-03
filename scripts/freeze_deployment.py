#!/usr/bin/env python
"""Freeze a deployment bundle (default ``gidi-finance-v1``) from the artifacts a protocol names.

Usage:
    uv run python scripts/freeze_deployment.py --out models/gidi-finance-v1
    uv run python scripts/freeze_deployment.py --out models/gidi-finance-v1 --check
    uv run python scripts/freeze_deployment.py --out models/gidi-finance-v2 \\
        --protocol experiments/deployment-v2/protocol.json     # value-head bundle

Nothing is trained or re-exported: the INT8 ONNX, tokenizer and vocab map are byte copies of the
source artifacts named in the protocol (default ``experiments/deployment-v1/protocol.json``);
``config.json`` and ``manifest.json`` are derived from the checkpoint metadata and the files
themselves. A protocol with ``"value_head": true`` additionally expects the ``value_logits`` ONNX
output and writes ``value_labels``, the CRF ``value_decoding`` (read from the ``value_head.crf.*``
tensors of the checkpoint, float32-exact) and the dual-encoder ``architecture`` into
``config.json``. A protocol with ``"base_checkpoint"`` (the v2 bundle: the value checkpoint carries
no compression metadata) takes the compression/vocab/FFN provenance and the layer map from that
base checkpoint, requires its tokenizer and HF config to equal the value checkpoint's, and records
its weights in the manifest.

Default mode builds the bundle in a temporary sibling directory and moves it into place. If
``--out`` already exists the bundle is rebuilt anyway and compared byte for byte: identical is a
no-op, different is an error (a frozen bundle is never overwritten). ``--check`` does the same
comparison without ever writing ``--out``; it exits nonzero on any difference.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import onnx
import yaml
from safetensors import safe_open

from gidi.export.onnx_export import INPUT_NAMES, OUTPUT_NAMES
from gidi.inference.bundle import (
    BUNDLE_FILES,
    CONFIG_FILE,
    MANIFEST_FILE,
    MODEL_FILE,
    TOKENIZER_CONFIG_FILE,
    TOKENIZER_FILE,
    VALUE_LABELS,
    VALUE_OUTPUT,
    VOCAB_MAP_FILE,
    sha256_file,
)
from gidi.modeling.checkpoint import META_FILE, WEIGHTS_FILE
from gidi.modeling.preprocessing import DEFAULT_MAX_LENGTH, TAGS, TYPES

PROTOCOL = Path("experiments/deployment-v1/protocol.json")
ANNOTATION_CONFIG = Path("configs/annotation-v1.yaml")
DEPLOYMENT_SEED = 1
FP32_ONNX_FILE = "model.onnx"
VALUE_HEAD_PREFIX = "value_head."
VALUE_CRF_KEYS = ("start_transitions", "end_transitions", "transitions")
VALUE_MLP_HIDDEN = 256
VALUE_CONFIDENCE = (
    "geometric mean of softmax(value_logits) P(Viterbi tag) over span tokens; "
    "no span: min P(O) over real tokens"
)
RUNTIME_REQUIREMENTS = {
    "python": ">=3.12",
    "onnxruntime": ">=1.30.0",
    "tokenizers": ">=0.23.2",
    "numpy": ">=2.0",
}


def _json_text(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _entry(path: Path) -> dict:
    return {"path": path.as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _parameter_counts(weights: Path) -> dict[str, int]:
    """Parameter counts from the safetensors header: total, path B (``value_head.*``), path A."""
    with safe_open(str(weights), framework="numpy") as f:
        sizes = {key: math.prod(f.get_slice(key).get_shape()) for key in f.keys()}  # noqa: SIM118
    total = sum(sizes.values())
    path_b = sum(n for key, n in sizes.items() if key.startswith(VALUE_HEAD_PREFIX))
    return {"total": total, "path_a": total - path_b, "path_b": path_b}


def _value_decoding(weights: Path) -> dict:
    """The ``config.json`` ``value_decoding`` block; CRF scores from the checkpoint tensors.

    The tensors are float32; ``tolist`` yields the exact double of each float32, whose JSON
    repr round-trips, so a float64 Viterbi in the runtime equals the torch CRF cast to float64.
    """
    shapes = {"start_transitions": (3,), "end_transitions": (3,), "transitions": (3, 3)}
    crf = {}
    with safe_open(str(weights), framework="numpy") as f:
        keys = set(f.keys())  # noqa: SIM118
        for name in VALUE_CRF_KEYS:
            key = f"{VALUE_HEAD_PREFIX}crf.{name}"
            _require(key in keys, f"checkpoint has no CRF tensor {key}")
            tensor = f.get_tensor(key)
            _require(tensor.dtype == np.float32, f"{key} is {tensor.dtype}, expected float32")
            _require(tensor.shape == shapes[name], f"{key} has shape {tensor.shape}")
            crf[name] = tensor.tolist()
    return {
        "method": "crf_viterbi",
        "start_transitions": crf["start_transitions"],
        "end_transitions": crf["end_transitions"],
        "transitions": crf["transitions"],
        "span_selection": "highest_confidence",
        "confidence": VALUE_CONFIDENCE,
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"error: {message}")


def _onnx_interface(path: Path) -> dict:
    model = onnx.load(str(path), load_external_data=False)
    opsets = [i.version for i in model.opset_import if i.domain in ("", "ai.onnx")]
    return {
        "inputs": [i.name for i in model.graph.input],
        "outputs": [o.name for o in model.graph.output],
        "opset": max(opsets),
    }


def build_bundle(out_dir: Path, created_at: str, protocol_path: Path = PROTOCOL) -> None:
    """Write the complete bundle into the (new, empty) directory ``out_dir``.

    ``protocol_path`` names the sources. A protocol with ``"value_head": true`` (a v2 bundle)
    expects the extra ``value_logits`` ONNX output and records ``value_labels``, the CRF
    ``value_decoding`` and the dual-encoder ``architecture`` in ``config.json`` (and
    ``value_labels`` plus parameter counts in ``manifest.json``); it needs
    ``"value_decoding": "crf_viterbi"``. An optional ``"annotation_config"`` overrides the
    annotation config path and ``"seed"`` the expected checkpoint seed. ``"base_checkpoint"``
    names the checkpoint that carries the compression provenance when the source checkpoint does
    not; ``"source_experiment"`` (with ``"source_experiment_protocol_sha256"``) adds the
    experiment protocol to the sources and cross-checks the parameter counts against it.
    """
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    value_head = bool(protocol.get("value_head", False))
    if value_head:
        _require(
            protocol.get("value_decoding") == "crf_viterbi",
            f"unsupported value_decoding {protocol.get('value_decoding')!r}; expected crf_viterbi",
        )
    output_names = (*OUTPUT_NAMES, VALUE_OUTPUT) if value_head else OUTPUT_NAMES
    checkpoint = Path(protocol["source_checkpoint"])
    onnx_dir = Path(protocol["source_onnx"])
    meta = json.loads((checkpoint / META_FILE).read_text(encoding="utf-8"))
    hf_config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    annotation_config = Path(protocol.get("annotation_config", ANNOTATION_CONFIG))
    annotation = yaml.safe_load(annotation_config.read_text(encoding="utf-8"))

    base = Path(protocol["base_checkpoint"]) if "base_checkpoint" in protocol else None
    provenance = meta
    if base is not None:
        _require((base / META_FILE).is_file(), f"base checkpoint metadata missing: {base}")
        provenance = json.loads((base / META_FILE).read_text(encoding="utf-8"))
    compression = provenance["compression"]
    vocab_dir = Path(compression["vocab"]["spec"])
    ffn_dir = Path(compression["ffn"]["map"])
    sources = {
        "checkpoint_model_safetensors": checkpoint / WEIGHTS_FILE,
        "onnx_fp32": onnx_dir / FP32_ONNX_FILE,
        "onnx_int8": onnx_dir / MODEL_FILE,
        "tokenizer_json": checkpoint / TOKENIZER_FILE,
        "tokenizer_config_json": checkpoint / TOKENIZER_CONFIG_FILE,
        "vocab_map_json": vocab_dir / VOCAB_MAP_FILE,
        "ffn_map_json": ffn_dir / "ffn_map.json",
    }
    if base is not None:
        sources["base_checkpoint_model_safetensors"] = base / WEIGHTS_FILE
    experiment_protocol = None
    if "source_experiment" in protocol:
        experiment_protocol = Path(protocol["source_experiment"]) / "protocol.json"
        sources["source_experiment_protocol"] = experiment_protocol
    for name, path in sources.items():
        _require(path.is_file(), f"source {name} missing: {path}")
    source_entries = {name: _entry(path) for name, path in sources.items()}

    if base is not None:
        # The value checkpoint must sit on the recorded base: same weights, tokenizer, HF config.
        frozen_base = meta.get("frozen_base", {}).get("model_safetensors_sha256")
        _require(
            frozen_base == source_entries["base_checkpoint_model_safetensors"]["sha256"],
            "base checkpoint weights differ from the checkpoint's recorded frozen_base sha256",
        )
        for name, file in (
            ("tokenizer_json", TOKENIZER_FILE),
            ("tokenizer_config_json", TOKENIZER_CONFIG_FILE),
        ):
            _require(
                sha256_file(base / file) == source_entries[name]["sha256"],
                f"{file} differs between the base checkpoint and the source checkpoint",
            )
        base_hf_config = json.loads((base / "config.json").read_text(encoding="utf-8"))
        _require(base_hf_config == hf_config, "HF config differs from the base checkpoint's")
        _require(provenance["seed"] == protocol.get("seed", DEPLOYMENT_SEED), "base seed differs")
    if experiment_protocol is not None:
        expected = protocol.get("source_experiment_protocol_sha256")
        _require(
            expected in (None, source_entries["source_experiment_protocol"]["sha256"]),
            f"{experiment_protocol} does not match source_experiment_protocol_sha256",
        )

    # Consistency of the sources with each other and with the code that trained them.
    seed = protocol.get("seed", DEPLOYMENT_SEED)
    _require(meta["seed"] == seed, f"checkpoint seed {meta['seed']} != {seed}")
    _require(tuple(meta["types"]) == TYPES, "checkpoint types differ from preprocessing.TYPES")
    _require(tuple(meta["tags"]) == TAGS, "checkpoint tags differ from preprocessing.TAGS")
    _require(meta["max_length"] == DEFAULT_MAX_LENGTH, "checkpoint max_length differs")
    _require(
        list(annotation["types"]) == list(TYPES),
        f"{annotation['version']} types differ from TYPES",
    )
    if value_head:
        _require(
            tuple(meta.get("value_tags", ())) == VALUE_LABELS,
            f"checkpoint value_tags {meta.get('value_tags')} differ from {list(VALUE_LABELS)}",
        )
    _require(
        compression["vocab"]["map_sha256"] == source_entries["vocab_map_json"]["sha256"],
        "vocab_map.json does not match the checkpoint's recorded map_sha256",
    )
    _require(
        compression["ffn"]["map_sha256"] == source_entries["ffn_map_json"]["sha256"],
        "ffn_map.json does not match the checkpoint's recorded map_sha256",
    )
    _require(
        hf_config["max_position_embeddings"] == compression["positions"]["rows_after"],
        "position rows differ between config.json and checkpoint metadata",
    )
    _require(hf_config["vocab_size"] == compression["vocab"]["rows_after"], "vocab_size mismatch")
    _require(
        hf_config["intermediate_size"] == compression["ffn"]["intermediate_after"],
        "ffn intermediate size mismatch",
    )
    interface = _onnx_interface(sources["onnx_int8"])
    _require(
        tuple(interface["inputs"]) == INPUT_NAMES and tuple(interface["outputs"]) == output_names,
        f"unexpected ONNX interface {interface}",
    )
    if value_head:
        fp32_interface = _onnx_interface(sources["onnx_fp32"])
        _require(
            tuple(fp32_interface["inputs"]) == INPUT_NAMES
            and tuple(fp32_interface["outputs"]) == output_names,
            f"unexpected FP32 ONNX interface {fp32_interface}",
        )

    layer_map = provenance["init_layer_map"]
    encoder = {
        "layers": hf_config["num_hidden_layers"],
        "hidden": hf_config["hidden_size"],
        "heads": hf_config["num_attention_heads"],
        "ffn_intermediate": hf_config["intermediate_size"],
        "position_rows": hf_config["max_position_embeddings"],
        "vocab_size": hf_config["vocab_size"],
        "encoder_layers_from_bamibert": [layer_map[k] for k in sorted(layer_map, key=int)],
    }
    value_config: dict = {}
    value_manifest: dict = {}
    architecture: dict = encoder
    if value_head:
        weights = checkpoint / WEIGHTS_FILE
        parameters = _parameter_counts(weights)
        _require(parameters["path_b"] > 0, f"{weights} has no {VALUE_HEAD_PREFIX}* tensors")
        if experiment_protocol is not None:
            declared = json.loads(experiment_protocol.read_text(encoding="utf-8"))
            declared = declared["architecture"]["params"]
            expected = {
                "total": declared["total"],
                "path_a": declared["v1_encoder_frozen"]
                + declared["v1_type_and_target_heads_frozen"],
                "path_b": declared["trainable_total"],
            }
            _require(parameters == expected, f"parameter counts {parameters} != {expected}")
        head = (
            f"Linear({encoder['hidden']},{VALUE_MLP_HIDDEN})-GELU-Linear({VALUE_MLP_HIDDEN},3)+CRF"
        )
        architecture = {
            "kind": "dual_encoder",
            "path_a": {**encoder, "role": "type+target", "frozen_from": "gidi-finance-v1"},
            "path_b": {**encoder, "role": "value", "head": head},
            "shared_tokenization": True,
        }
        value_config = {
            "value_labels": list(VALUE_LABELS),
            "value_decoding": _value_decoding(weights),
        }
        value_manifest = {
            "value_labels": list(VALUE_LABELS),
            "value_decoding": {
                "method": "crf_viterbi",
                "rule": protocol.get("value_decoding_rule"),
                "crf_source": f"{VALUE_HEAD_PREFIX}crf.* in checkpoint_model_safetensors",
            },
            "parameters": {
                **parameters,
                "path_a_note": "frozen gidi-finance-v1 encoder + type head + target head",
                "path_b_note": "value encoder + emission MLP + CRF",
            },
        }

    config = {
        "model_version": protocol["version"],
        "annotation_version": annotation["version"],
        "types": list(meta["types"]),
        "tags": list(meta["tags"]),
        **value_config,
        "max_length": meta["max_length"],
        "onnx": interface,
        "architecture": architecture,
        "normalization": "NFC",
        "truncation": "tokens beyond max_length (incl. <s>,</s>) dropped",
    }

    out_dir.mkdir(parents=True)
    for name, source in (
        (MODEL_FILE, "onnx_int8"),
        (TOKENIZER_FILE, "tokenizer_json"),
        (TOKENIZER_CONFIG_FILE, "tokenizer_config_json"),
        (VOCAB_MAP_FILE, "vocab_map_json"),
    ):
        shutil.copyfile(sources[source], out_dir / name)
    (out_dir / CONFIG_FILE).write_text(_json_text(config), encoding="utf-8")

    files = {
        name: {k: v for k, v in _entry(out_dir / name).items() if k != "path"}
        for name in BUNDLE_FILES
    }
    total = sum(entry["bytes"] for entry in files.values())
    manifest = {
        "model_version": protocol["version"],
        "annotation_version": annotation["version"],
        **value_manifest,
        "created_at": created_at,
        "seed": meta["seed"],
        "protocol": _entry(protocol_path),
        "sources": source_entries,
        "files": files,
        "compression": {
            "positions": compression["positions"],
            "vocab": {
                "rows_before": compression["vocab"]["rows_before"],
                "rows_after": compression["vocab"]["rows_after"],
                "policy": compression["vocab"]["spec"],
            },
            "ffn": {
                "intermediate_before": compression["ffn"]["intermediate_before"],
                "intermediate_after": compression["ffn"]["intermediate_after"],
                "criterion": compression["ffn"]["criterion"],
            },
        },
        "architecture_summary": protocol["architecture"],
        "seed_selection": protocol["seed_selection"],
        "runtime_requirements": RUNTIME_REQUIREMENTS,
        "int8_size_mb": files[MODEL_FILE]["bytes"] / 1e6,
        "files_total_bytes": total,
        "files_total_mb": total / 1e6,
        "files_total_note": "sum of the files listed under 'files'; excludes manifest.json",
    }
    (out_dir / MANIFEST_FILE).write_text(_json_text(manifest), encoding="utf-8")
    for path in out_dir.iterdir():
        path.chmod(0o444)


def differences(built: Path, existing: Path) -> list[str]:
    """Names of files that are missing, extra or byte-different between two bundles."""
    names = {p.name for p in built.iterdir()} | {p.name for p in existing.iterdir()}
    problems = []
    for name in sorted(names):
        a, b = built / name, existing / name
        if not a.is_file() or not b.is_file():
            problems.append(f"{name}: missing in {'existing' if a.is_file() else 'rebuilt'} bundle")
        elif a.stat().st_size != b.stat().st_size or sha256_file(a) != sha256_file(b):
            problems.append(f"{name}: bytes differ")
    return problems


def _existing_created_at(out: Path) -> str | None:
    try:
        return json.loads((out / MANIFEST_FILE).read_text(encoding="utf-8"))["created_at"]
    except (OSError, KeyError, ValueError):
        return None


def _remove(path: Path) -> None:
    for child in path.iterdir():
        child.chmod(0o644)
    shutil.rmtree(path, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="bundle directory")
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROTOCOL,
        help="deployment protocol JSON naming the sources (default: the gidi-finance-v1 one)",
    )
    parser.add_argument(
        "--check", action="store_true", help="rebuild in a temp dir and compare, never write --out"
    )
    args = parser.parse_args(argv)
    out: Path = args.out

    if args.check and not out.is_dir():
        print(f"FAIL: {out} does not exist", file=sys.stderr)
        return 1

    existing_created_at = _existing_created_at(out) if out.is_dir() else None
    created_at = existing_created_at or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}-", dir=out.parent))
    built = staging / out.name
    try:
        build_bundle(built, created_at, args.protocol)
        if out.is_dir():
            problems = differences(built, out)
            if problems:
                verb = "does not match" if args.check else "refusing to overwrite; differs from"
                print(f"FAIL: {out} {verb} a rebuild from the sources:", file=sys.stderr)
                for problem in problems:
                    print(f"  {problem}", file=sys.stderr)
                return 1
            print(f"OK: {out} is byte-identical to a rebuild from the sources")
            return 0
        built.rename(out)
        print(f"built {out}")
        return 0
    finally:
        if built.is_dir():
            _remove(built)
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
