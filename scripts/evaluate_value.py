#!/usr/bin/env python
"""Evaluate a value-span model (type + target + value) on a merged annotation-v2 records file.

Usage:
    uv run python scripts/evaluate_value.py --checkpoint models/value-span-v1/seed1 \\
        --records <merged-test.jsonl> --train <merged-train.jsonl> \\
        --out experiments/value-span-v1/eval-seed1.json \\
        --predictions experiments/value-span-v1/eval-seed1-predictions.jsonl
    # an exported ONNX graph (FP32 or INT8); the tokenizer comes from --checkpoint/--tokenizer
    uv run python scripts/evaluate_value.py \\
        --onnx models/value-span-v1-onnx/seed1/model.int8.onnx \\
        --tokenizer models/value-span-v1/seed1 --records <merged-test.jsonl> --out <json>

Reports type accuracy / macro-F1, target span F1 and exact match, type+target joint, and (on
``value_status == complete`` records) value span F1 (exact-span and token/char overlap), value
exact match, present/null accuracy and the full joint (type AND target AND value exact), overall
and per slice: value-span categories (``gidi.annotation.value_span``), ``surface_pattern``
seen/unseen vs ``--train``, ``amount_length`` (multi-token or spaced amounts), ``accented``,
``source_batch``. ``--predictions`` writes one row per record (gold vs prediction, per-head
correctness, confidences, slice tags) for manual inspection. Evaluation only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

from gidi.corpus.jsonl import read_jsonl, write_jsonl
from gidi.evaluation.value_eval import evaluate_records, value_span_categories
from gidi.evaluation.value_predict import onnx_logits, torch_logits
from gidi.export.onnx_export import make_session
from gidi.modeling.tokenization import load_tokenizer
from gidi.modeling.value import load_value_checkpoint


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--records", type=Path, required=True, help="merged records to score")
    parser.add_argument("--checkpoint", type=Path, help="value checkpoint dir (PyTorch back end)")
    parser.add_argument("--onnx", type=Path, help="ONNX file to score instead of --checkpoint")
    parser.add_argument("--tokenizer", type=Path, help="tokenizer dir (default: --checkpoint)")
    parser.add_argument("--train", type=Path, help="training records, for the unseen-pattern slice")
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True, help="metrics JSON")
    parser.add_argument("--predictions", type=Path, help="per-record JSONL")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.onnx is None and args.checkpoint is None:
        parser.error("give --checkpoint (PyTorch) or --onnx (with --tokenizer or --checkpoint)")
    if args.onnx is not None and args.tokenizer is None and args.checkpoint is None:
        parser.error("--onnx needs --tokenizer (or --checkpoint) for the tokenizer")
    for path in (args.out, args.predictions):
        if path is not None and path.exists() and not args.overwrite:
            raise SystemExit(f"refusing to overwrite {path} (pass --overwrite)")

    records = read_jsonl(args.records)
    train_records = read_jsonl(args.train) if args.train else None
    max_length = args.max_length
    if args.onnx is None:
        model, tokenizer, meta = load_value_checkpoint(args.checkpoint)
        max_length = int(meta.get("max_length", max_length))
        device = torch.device("cpu")

        def logits_fn(data):
            return torch_logits(model, data, device)

        backend = {"kind": "torch", "checkpoint": str(args.checkpoint)}
    else:
        tokenizer = load_tokenizer(str(args.tokenizer or args.checkpoint))
        session = make_session(args.onnx, threads=1)

        def logits_fn(data):
            return onnx_logits(session, data)

        backend = {"kind": "onnx", "path": str(args.onnx), "sha256": _sha256(args.onnx)}

    categories = value_span_categories(required=False)
    if categories is None:
        print(
            "warning: gidi.annotation.value_span is not importable; no category slices",
            file=sys.stderr,
        )
    metrics, rows, _ = evaluate_records(
        records,
        logits_fn,
        tokenizer,
        max_length,
        train_records=train_records,
        categories=categories,
    )
    report: dict[str, Any] = {
        "backend": backend,
        "records": {"path": str(args.records), "sha256": _sha256(args.records)},
        "train": {"path": str(args.train), "sha256": _sha256(args.train)} if args.train else None,
        **metrics,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.predictions:
        write_jsonl(args.predictions, rows, overwrite=args.overwrite)

    overall, value = metrics["overall"], metrics["value"]
    print(
        f"n={metrics['n']} (value complete {metrics['n_value_complete']}) "
        f"type acc {overall['type']['accuracy']:.4f} F1 {overall['type']['macro_f1']:.4f} "
        f"target F1 {overall['target']['f1']:.4f} exact {overall['target']['exact_match']:.4f}"
    )
    token_f1 = value["token"]["f1"] if value["token"] else float("nan")
    print(
        f"value span F1 {value['span']['f1']:.4f} token F1 {token_f1:.4f} "
        f"exact {value['exact_match']:.4f} present acc {value['present_accuracy']:.4f} "
        f"full joint {metrics['full_joint']['accuracy']:.4f}"
    )
    print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
