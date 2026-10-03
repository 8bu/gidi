#!/usr/bin/env python
"""Export a trained Gidi checkpoint to ONNX (FP32 + INT8) and measure size, parity and latency.

Usage:
    uv run python scripts/export_onnx.py --checkpoint models/baseline-v1/<slug>/<run>/ \\
        --out models/baseline-v1-onnx/<slug>/ --report experiments/baseline-v1/export/<slug>.json

Writes ``model.onnx`` and ``model.int8.onnx`` into ``--out`` and a JSON report to ``--report``.
Sample notes come from the test split (fallback: a small built-in list).
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
from gidi.export.onnx_export import (
    SAMPLE_NOTES,
    benchmark_latency,
    benchmark_torch_latency,
    export_onnx,
    make_session,
    quantize_int8,
)
from gidi.modeling.checkpoint import WEIGHTS_FILE, load_checkpoint

TEST_SPLIT = Path("datasets/annotation-v1/splits/test.jsonl")


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
    parser.add_argument("--notes", type=Path, default=TEST_SPLIT, help="JSONL with a text field")
    parser.add_argument("--num-notes", type=int, default=50)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="replace an existing report")
    args = parser.parse_args(argv)

    if args.report.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite existing report: {args.report} (pass --overwrite)")

    model, tokenizer, meta = load_checkpoint(args.checkpoint)
    model.eval()
    max_length = int(meta.get("max_length", 32))
    notes, notes_source = sample_notes(args.notes, args.num_notes, args.seed)

    fp32_path = args.out / "model.onnx"
    int8_path = args.out / "model.int8.onnx"
    fp32 = export_onnx(model, tokenizer, fp32_path, max_length, args.opset, sample_texts=notes)
    int8 = quantize_int8(fp32_path, int8_path, tokenizer, notes, max_length)

    thread_modes = {"threads_1": 1, "threads_default": None}
    latency: dict[str, dict] = {}
    for label, threads in thread_modes.items():
        latency[label] = {
            "torch_fp32": benchmark_torch_latency(
                model, tokenizer, notes, args.runs, args.warmup, max_length, threads
            ),
            "onnx_fp32": benchmark_latency(
                make_session(fp32_path, threads),
                tokenizer,
                notes,
                args.runs,
                args.warmup,
                max_length,
            ),
            "onnx_int8": benchmark_latency(
                make_session(int8_path, threads),
                tokenizer,
                notes,
                args.runs,
                args.warmup,
                max_length,
            ),
        }
    latency["torch_threads_default"] = torch.get_num_threads()

    weights = args.checkpoint / WEIGHTS_FILE
    report = {
        "checkpoint": str(args.checkpoint),
        "encoder": meta.get("encoder"),
        "max_length": max_length,
        "param_count": sum(p.numel() for p in model.parameters()),
        "checkpoint_safetensors_mb": weights.stat().st_size / 1e6,
        "notes": {"source": notes_source, "count": len(notes), "seed": args.seed},
        "fp32": fp32,
        "int8": int8,
        "latency": latency,
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

    print(f"exporter={fp32['exporter']} parity max_abs_diff={fp32['parity']['max_abs_diff']:.2e}")
    print(f"size FP32 {int8['fp32_size_mb']:.1f} MB -> INT8 {int8['int8_size_mb']:.1f} MB")
    agree = int8["agreement"]
    print(f"INT8 agreement type={agree['type_agreement']:.3f} tag={agree['tag_agreement']:.3f}")
    for label in thread_modes:
        for name, r in latency[label].items():
            print(
                f"{label:<16} {name:<11} total p50 {r['total']['p50_ms']:.2f} ms "
                f"p95 {r['total']['p95_ms']:.2f} ms (inference p50 {r['inference']['p50_ms']:.2f})"
            )
    print(f"report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
