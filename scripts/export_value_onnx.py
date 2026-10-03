#!/usr/bin/env python
"""Export a value-span checkpoint to ONNX (FP32 + INT8) with the extra ``value_logits`` output.

Usage:
    uv run python scripts/export_value_onnx.py --checkpoint models/value-span-v1/seed1 \\
        --out models/value-span-v1-onnx/seed1 --report experiments/value-span-v1/export-seed1.json

Writes ``model.onnx`` and ``model.int8.onnx`` into ``--out`` and a JSON report to ``--report``.
Outputs are ``type_logits``, ``tag_logits`` (the v1 names/order) and ``value_logits`` ``[batch,
seq, 3]`` (O, B-VALUE, I-VALUE); inputs, opset (17), dynamic axes and the INT8 settings (dynamic,
QInt8 weights) are the v1 export's. The report holds the FP32 parity vs PyTorch per output, the
INT8 size and INT8-vs-FP32 agreement, and the size difference to a v1 INT8 model (the value
head: 768*3+3 = 2,307 parameters for the linear head). Sample notes come from ``--notes``
(fallback: built-in list).
The report also holds file sizes/sha256, the declared opset, the recomputed frozen (path A)
state hash and a content comparison of the v1 INT8 / v1 source FP32 initializers against the new
graphs (``path_a_initializers``). ``--read-only`` chmods the outputs to 0444 (for a frozen
deployment source such as ``models/gidi-finance-v2-onnx``).
"""

from __future__ import annotations

import argparse
import json
import platform
import random
from pathlib import Path

import onnxruntime
import torch

from gidi.corpus.jsonl import read_jsonl
from gidi.export.onnx_export import SAMPLE_NOTES
from gidi.export.onnx_export_value import (
    compare_initializers,
    export_value_onnx,
    onnx_interface,
    onnx_opset,
    quantize_value_int8,
)
from gidi.inference.bundle import sha256_file
from gidi.modeling.checkpoint import WEIGHTS_FILE
from gidi.modeling.value import load_value_checkpoint
from gidi.training.train_value_head import frozen_state_sha256

DEFAULT_NOTES = Path("datasets/annotation-v1/splits/test.jsonl")
V1_INT8 = Path("models/gidi-finance-v1/model.int8.onnx")
V1_FP32 = Path("models/compression-v3-onnx/pos32-vocabB8000-ffn2048-4x768-seed1/model.onnx")


def sample_notes(path: Path, n: int, seed: int) -> tuple[list[str], str]:
    if path.exists():
        texts = [r["text"] for r in read_jsonl(path)]
        if texts:
            return random.Random(seed).sample(texts, min(n, len(texts))), str(path)
    return list(SAMPLE_NOTES), "builtin"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="directory for the ONNX files")
    parser.add_argument("--report", type=Path, required=True, help="JSON report path")
    parser.add_argument("--notes", type=Path, default=DEFAULT_NOTES, help="JSONL with a text field")
    parser.add_argument("--num-notes", type=int, default=50)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--v1-int8", type=Path, default=V1_INT8, help="v1 INT8 file for the size delta"
    )
    parser.add_argument(
        "--v1-fp32",
        type=Path,
        default=V1_FP32,
        help="v1 source FP32 ONNX file for the path-A initializer comparison",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace an existing report")
    parser.add_argument(
        "--read-only", action="store_true", help="chmod the ONNX files and the report to 0444"
    )
    args = parser.parse_args(argv)
    if args.report.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite existing report: {args.report} (pass --overwrite)")
    fp32_path = args.out / "model.onnx"
    int8_path = args.out / "model.int8.onnx"
    if args.overwrite:  # a previous --read-only run left 0444 files behind
        for stale in (args.report, fp32_path, int8_path):
            stale.unlink(missing_ok=True)

    model, tokenizer, meta = load_value_checkpoint(args.checkpoint)
    max_length = int(meta.get("max_length", 32))
    notes, notes_source = sample_notes(args.notes, args.num_notes, args.seed)

    fp32 = export_value_onnx(model, tokenizer, fp32_path, max_length, args.opset, notes)
    int8 = quantize_value_int8(fp32_path, int8_path, tokenizer, notes, max_length)

    v1_delta = None
    if args.v1_int8.exists():
        v1_bytes = args.v1_int8.stat().st_size
        v1_delta = {
            "v1_int8": str(args.v1_int8),
            "v1_int8_bytes": v1_bytes,
            "v2_int8_bytes": int8_path.stat().st_size,
            "delta_bytes": int8_path.stat().st_size - v1_bytes,
        }
    comparisons = {}
    if args.v1_int8.exists():
        comparisons["int8_vs_v1_int8"] = compare_initializers(args.v1_int8, int8_path)
    if args.v1_fp32.exists():
        comparisons["fp32_vs_v1_source_fp32"] = compare_initializers(args.v1_fp32, fp32_path)
    frozen_sha = frozen_state_sha256(model)
    report = {
        "checkpoint": str(args.checkpoint),
        "encoder": meta.get("encoder"),
        "value_head_arch": meta.get("value_head_arch"),
        "max_length": max_length,
        "param_count": sum(p.numel() for p in model.parameters()),
        "value_head_params": sum(p.numel() for p in model.value_head.parameters()),
        "checkpoint_safetensors_mb": (args.checkpoint / WEIGHTS_FILE).stat().st_size / 1e6,
        "checkpoint_safetensors_sha256": sha256_file(args.checkpoint / WEIGHTS_FILE),
        "frozen_state_sha256": {
            "recomputed": frozen_sha,
            "checkpoint_metadata": meta.get("frozen_state_sha256"),
            "equal": frozen_sha == meta.get("frozen_state_sha256"),
        },
        "notes": {"source": notes_source, "count": len(notes), "seed": args.seed},
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in (fp32_path, int8_path)
        },
        "opset": {
            "requested": args.opset,
            "fp32": onnx_opset(fp32_path),
            "int8": onnx_opset(int8_path),
        },
        "interface": {
            "fp32": onnx_interface(fp32_path),
            "int8": onnx_interface(int8_path),
        },
        "fp32": fp32,
        "int8": int8,
        "int8_size_delta_vs_v1": v1_delta,
        "path_a_initializers": comparisons,
        "environment": {
            "torch": torch.__version__,
            "onnxruntime": onnxruntime.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.read_only:
        for path in (fp32_path, int8_path, args.report):
            path.chmod(0o444)

    parity = fp32["parity"]
    print(
        f"exporter={fp32['exporter']} outputs={report['interface']['fp32']['outputs']} "
        f"parity max_abs_diff={parity['max_abs_diff']:.2e} "
        f"(value_logits {parity['max_abs_diff_value_logits']:.2e})"
    )
    print(f"size FP32 {int8['fp32_size_mb']:.3f} MB -> INT8 {int8['int8_size_mb']:.3f} MB")
    agree = int8["agreement"]
    print(
        f"INT8 agreement type={agree['type_agreement']:.3f} tag={agree['tag_agreement']:.3f} "
        f"value={agree['value_agreement']:.3f}"
    )
    if v1_delta:
        print(f"INT8 size delta vs v1: {v1_delta['delta_bytes']:+d} bytes")
    print(f"report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
