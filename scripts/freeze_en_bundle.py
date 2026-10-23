#!/usr/bin/env python
"""Freeze an English deployment bundle (``gidi-en-v2``, ``gidi-en-v4``) from a protocol's artifacts.

Usage:
    uv run python scripts/freeze_en_bundle.py --out models/gidi-en-v4-bundle \\
        --protocol experiments/gidi-en-deployment-v4/protocol.json
    uv run python scripts/freeze_en_bundle.py --out models/gidi-en-v4-bundle \\
        --protocol experiments/gidi-en-deployment-v4/protocol.json --check

Nothing is trained or re-exported: the INT8 ONNX, tokenizer and vocabulary map are byte copies of
the source artifacts named in the protocol; ``config.json`` and ``manifest.json`` are derived from
the checkpoint metadata and the files themselves. Two bundle kinds, chosen by the protocol:

* ``gidi-en-v2`` (roberta, byte-level BPE): no value head and no value parser (``config.json``
  has neither ``value_labels`` nor ``value_source``), ``target_snap: "words"``.
* ``gidi-en-v4`` (``"value_head": true``, MiniLM, uncased WordPiece): a third ONNX output
  ``value_logits`` (labels O/B-VALUE/I-VALUE) decoded by per-token argmax, the highest-confidence
  span, ``value_snap: "words"`` and ``tokenizer_type: "wordpiece"``.

The freeze logic and the ``--check`` semantics are those of ``scripts/freeze_deployment.py`` (a
frozen bundle is never overwritten).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import yaml
from freeze_deployment import (
    FP32_ONNX_FILE,
    RUNTIME_REQUIREMENTS,
    _entry,
    _existing_created_at,
    _json_text,
    _onnx_interface,
    _remove,
    _require,
    differences,
)

from gidi.export.onnx_export import INPUT_NAMES, OUTPUT_NAMES
from gidi.inference.bundle import (
    BUNDLE_FILES,
    CONFIG_FILE,
    MANIFEST_FILE,
    MODEL_FILE,
    TARGET_SNAP_WORDS,
    TOKENIZER_CONFIG_FILE,
    TOKENIZER_FILE,
    TOKENIZER_TYPE_WORDPIECE,
    VALUE_DECODING_ARGMAX,
    VALUE_LABELS,
    VALUE_OUTPUT,
    VALUE_SNAP_WORDS,
    VALUE_SPAN_SELECTION,
    VOCAB_MAP_FILE,
)
from gidi.modeling.checkpoint import META_FILE, WEIGHTS_FILE
from gidi.modeling.preprocessing import DEFAULT_MAX_LENGTH, TAGS, TYPES

PROTOCOL = Path("experiments/gidi-en-deployment-v2/protocol.json")


def build_bundle(out_dir: Path, created_at: str, protocol_path: Path = PROTOCOL) -> None:
    """Write the complete bundle into the (new, empty) directory ``out_dir``."""
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    _require(protocol.get("target_snap") == TARGET_SNAP_WORDS, "protocol needs target_snap words")
    checkpoint = Path(protocol["source_checkpoint"])
    onnx_dir = Path(protocol["source_onnx"])
    meta = json.loads((checkpoint / META_FILE).read_text(encoding="utf-8"))
    hf_config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    init_report = json.loads(Path(protocol["init_report"]).read_text(encoding="utf-8"))
    annotation = yaml.safe_load(Path(protocol["annotation_config"]).read_text(encoding="utf-8"))

    sources = {
        "checkpoint_model_safetensors": checkpoint / WEIGHTS_FILE,
        "onnx_fp32": onnx_dir / FP32_ONNX_FILE,
        "onnx_int8": onnx_dir / MODEL_FILE,
        "tokenizer_json": checkpoint / TOKENIZER_FILE,
        "tokenizer_config_json": checkpoint / TOKENIZER_CONFIG_FILE,
        "vocab_map_json": Path(protocol["vocab_map"]),
        "init_build_report_json": Path(protocol["init_report"]),
        "source_experiment_protocol": Path(protocol["source_experiment"]) / "protocol.json",
    }
    for name, path in sources.items():
        _require(path.is_file(), f"source {name} missing: {path}")
    source_entries = {name: _entry(path) for name, path in sources.items()}

    seed = protocol["seed"]
    _require(meta["seed"] == seed, f"checkpoint seed {meta['seed']} != {seed}")
    _require(tuple(meta["types"]) == TYPES, "checkpoint types differ from preprocessing.TYPES")
    _require(tuple(meta["tags"]) == TAGS, "checkpoint tags differ from preprocessing.TAGS")
    _require(meta["max_length"] == DEFAULT_MAX_LENGTH, "checkpoint max_length differs")
    _require(
        list(annotation["types"]) == list(TYPES),
        f"{annotation['version']} types differ from TYPES",
    )
    value_head = bool(protocol.get("value_head", False))
    init_source = protocol.get("init_source", "roberta-base")
    _require(init_report["source"] == init_source, f"init source is not {init_source}")
    _require(hf_config["vocab_size"] == init_report["vocab_rows"], "vocab_size mismatch")
    _require(hf_config["max_position_embeddings"] == init_report["positions"], "position rows")
    _require(hf_config["intermediate_size"] == init_report["intermediate"], "ffn size mismatch")
    outputs = (*OUTPUT_NAMES, VALUE_OUTPUT) if value_head else OUTPUT_NAMES
    if value_head:
        _require(tuple(meta.get("value_tags", ())) == VALUE_LABELS, "checkpoint value_tags differ")
        _require("value" in meta.get("heads", ()), "checkpoint has no value head")
    for kind in ("onnx_int8", "onnx_fp32"):
        interface = _onnx_interface(sources[kind])
        _require(
            tuple(interface["inputs"]) == INPUT_NAMES and tuple(interface["outputs"]) == outputs,
            f"unexpected {kind} ONNX interface {interface}",
        )
    interface = _onnx_interface(sources["onnx_int8"])

    layer_map = init_report["layer_map"]
    architecture = {
        "layers": hf_config["num_hidden_layers"],
        "hidden": hf_config["hidden_size"],
        "heads": hf_config["num_attention_heads"],
        "ffn_intermediate": hf_config["intermediate_size"],
        "position_rows": hf_config["max_position_embeddings"],
        "vocab_size": hf_config["vocab_size"],
        protocol.get("layers_key", "encoder_layers_from_roberta_base"): [
            layer_map[k] for k in sorted(layer_map, key=int)
        ],
    }
    config = {
        "model_version": protocol["version"],
        "annotation_version": annotation["version"],
        "types": list(meta["types"]),
        "tags": list(meta["tags"]),
        "target_snap": protocol["target_snap"],
        "max_length": meta["max_length"],
        "onnx": interface,
        "architecture": architecture,
        "normalization": "NFC",
        "truncation": protocol.get(
            "truncation", "tokens beyond max_length (incl. <s>,</s>) dropped"
        ),
    }
    value_rules: dict[str, object] = {}
    if value_head:
        value_rules = {
            "value_labels": list(VALUE_LABELS),
            "value_decoding": {
                "method": VALUE_DECODING_ARGMAX,
                "span_selection": VALUE_SPAN_SELECTION,
            },
            "value_snap": VALUE_SNAP_WORDS,
            "tokenizer_type": TOKENIZER_TYPE_WORDPIECE,
        }
        _require(
            protocol.get("tokenizer_type") == TOKENIZER_TYPE_WORDPIECE,
            "a value-head bundle needs tokenizer_type wordpiece in the protocol",
        )
        # Keys after ``tags``/``target_snap`` so the v2 key order is untouched.
        config = {**config, **value_rules}

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
        "target_snap": protocol["target_snap"],
        "label_provenance": protocol["label_provenance"],
        "created_at": created_at,
        "seed": meta["seed"],
        "protocol": _entry(protocol_path),
        "sources": source_entries,
        "files": files,
        "compression": {
            "positions": {"rows_after": init_report["positions"]},
            "vocab": {
                "rows_after": init_report["vocab_rows"],
                "policy": init_report.get("vocab_policy") or init_report["vocab"]["policy"],
            },
            "ffn": {"intermediate_after": init_report["intermediate"]},
        },
        "architecture_summary": protocol["architecture"],
        "seed_selection": protocol["seed_selection"],
        "runtime_requirements": RUNTIME_REQUIREMENTS,
        **value_rules,
        "int8_size_mb": files[MODEL_FILE]["bytes"] / 1e6,
        "files_total_bytes": total,
        "files_total_mb": total / 1e6,
        "files_total_note": "sum of the files listed under 'files'; excludes manifest.json",
    }
    (out_dir / MANIFEST_FILE).write_text(_json_text(manifest), encoding="utf-8")
    for path in out_dir.iterdir():
        path.chmod(0o444)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="bundle directory")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL, help="deployment protocol JSON")
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
