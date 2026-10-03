#!/usr/bin/env python
"""Evaluate exported ONNX models (FP32 + INT8) on the frozen test split and the probe set.

Usage:
    uv run python scripts/evaluate_onnx.py \\
        --onnx-dir models/compression-v1-onnx/baseline-4x768-seed1 --checkpoint <ckpt-dir> \\
        --onnx-dir models/compression-v1-onnx/pos32-4x768-seed1 --checkpoint <ckpt-dir> \\
        --out experiments/compression-v1/positions/onnx-eval.json

``--onnx-dir`` and ``--checkpoint`` are repeated and paired in order; the checkpoint supplies the
tokenizer (and the PyTorch reference predictions). Each model is scored with 1-thread sessions.
With more than one model, every later model is compared with the first: prediction identity for
torch/FP32/INT8, and raw-logit differences when both models see the same input ids.
Evaluation only: nothing here feeds back into model construction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from evaluate_probe import load_probe

from gidi.evaluation.metrics import evaluate
from gidi.export.onnx_export import INPUT_NAMES, OUTPUT_NAMES, make_session
from gidi.modeling.checkpoint import load_checkpoint
from gidi.modeling.preprocessing import TYPES, spans_from_tags
from gidi.training.train import Prepared, load_split, predict, prepare

TEST = Path("datasets/annotation-v1/distillation-v1/test.jsonl")
PROBE = Path("datasets/probe-v1")
BATCH = 64

Spans = list[dict[str, Any] | None]


def onnx_outputs(session: Any, data: Prepared) -> tuple[np.ndarray, list[np.ndarray]]:
    """Type logits ``(N, types)`` and per-record tag logits ``(len_i, tags)`` (padding trimmed)."""
    enc = data.encoded
    type_rows: list[np.ndarray] = []
    tag_rows: list[np.ndarray] = []
    for i in range(0, len(data), BATCH):
        ids = enc.input_ids[i : i + BATCH]
        mask = enc.attention_mask[i : i + BATCH]
        width = int(mask.sum(dim=1).max())
        feed = {
            INPUT_NAMES[0]: ids[:, :width].numpy(),
            INPUT_NAMES[1]: mask[:, :width].numpy(),
        }
        type_logits, tag_logits = session.run(list(OUTPUT_NAMES), feed)
        type_rows.append(type_logits)
        tag_rows.extend(tag_logits[j, : int(mask[j].sum())] for j in range(ids.shape[0]))
    return np.concatenate(type_rows), tag_rows


def decode(data: Prepared, type_logits: np.ndarray, tag_logits: list[np.ndarray]) -> tuple:
    types = [TYPES[int(k)] for k in type_logits.argmax(-1)]
    spans: Spans = [
        spans_from_tags(data.encoded.offsets[i], tags.argmax(-1).tolist(), data.records[i]["text"])
        for i, tags in enumerate(tag_logits)
    ]
    return types, spans


def span_key(span: dict[str, Any] | None) -> tuple[int, int] | None:
    return None if span is None else (int(span["start"]), int(span["end"]))


def score(records: list[dict], types: list[str], spans: Spans) -> dict[str, Any]:
    overall = evaluate(records, types, spans, slices=())["overall"]
    joint = [
        t == r["type"] and span_key(s) == span_key(r["target"])
        for r, t, s in zip(records, types, spans, strict=True)
    ]
    return {
        "type_accuracy": overall["type"]["accuracy"],
        "type_macro_f1": overall["type"]["macro_f1"],
        "span_f1": overall["target"]["f1"],
        "span_exact": overall["target"]["exact_match"],
        "joint": sum(joint) / len(joint),
    }


def same_predictions(a: tuple, b: tuple) -> dict[str, Any]:
    """Per-record agreement of two ``(types, spans)`` predictions."""
    types_a, spans_a = a
    types_b, spans_b = b
    type_same = [x == y for x, y in zip(types_a, types_b, strict=True)]
    span_same = [span_key(x) == span_key(y) for x, y in zip(spans_a, spans_b, strict=True)]
    both = [t and s for t, s in zip(type_same, span_same, strict=True)]
    n = len(both)
    return {
        "type_agreement": sum(type_same) / n,
        "span_agreement": sum(span_same) / n,
        "joint_agreement": sum(both) / n,
        "identical": all(both),
        "n_different": n - sum(both),
    }


def max_abs_diff(a: tuple, b: tuple) -> float:
    (type_a, tags_a), (type_b, tags_b) = a, b
    diffs = [float(np.abs(type_a - type_b).max())]
    diffs += [float(np.abs(x - y).max()) for x, y in zip(tags_a, tags_b, strict=True)]
    return max(diffs)


def run_model(onnx_dir: Path, checkpoint: Path, splits: dict[str, list[dict]]) -> dict[str, Any]:
    model, tokenizer, meta = load_checkpoint(checkpoint)
    max_length = int(meta["max_length"])
    sessions = {
        "fp32": make_session(onnx_dir / "model.onnx", threads=1),
        "int8": make_session(onnx_dir / "model.int8.onnx", threads=1),
    }
    result: dict[str, Any] = {
        "onnx_dir": str(onnx_dir),
        "checkpoint": str(checkpoint),
        "sizes_mb": {
            "fp32": (onnx_dir / "model.onnx").stat().st_size / 1e6,
            "int8": (onnx_dir / "model.int8.onnx").stat().st_size / 1e6,
        },
        "splits": {},
    }
    raw: dict[str, Any] = {}
    for name, records in splits.items():
        data = prepare(tokenizer, records, max_length)
        outputs = {kind: onnx_outputs(s, data) for kind, s in sessions.items()}
        preds = {"torch": predict(model, data, torch.device("cpu"))}
        preds |= {kind: decode(data, *out) for kind, out in outputs.items()}
        preds_metrics = {kind: score(records, *p) for kind, p in preds.items()}
        result["splits"][name] = {
            "n": len(records),
            "metrics": preds_metrics,
            "int8_minus_fp32": {
                k: preds_metrics["int8"][k] - preds_metrics["fp32"][k]
                for k in preds_metrics["fp32"]
            },
            "agreement_with_torch": {
                kind: same_predictions(preds[kind], preds["torch"]) for kind in sessions
            },
        }
        raw[name] = {"data": data, "outputs": outputs, "preds": preds}
    result["_raw"] = raw
    return result


def compare(ref: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ref["splits"]:
        a, b = ref["_raw"][name], other["_raw"][name]
        same_inputs = torch.equal(a["data"].encoded.input_ids, b["data"].encoded.input_ids)
        entry: dict[str, Any] = {"same_input_ids": same_inputs}
        for kind in ("fp32", "int8"):
            entry[kind] = {"predictions": same_predictions(a["preds"][kind], b["preds"][kind])}
            if same_inputs:
                diff = max_abs_diff(a["outputs"][kind], b["outputs"][kind])
                entry[kind]["logits_max_abs_diff"] = diff
                entry[kind]["logits_bitwise_equal"] = diff == 0.0
        entry["torch"] = {"predictions": same_predictions(a["preds"]["torch"], b["preds"]["torch"])}
        out[name] = entry
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--onnx-dir", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if len(args.onnx_dir) != len(args.checkpoint):
        raise SystemExit("--onnx-dir and --checkpoint must be given the same number of times")
    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {args.out} (pass --overwrite)")

    splits = {"test": load_split(TEST), "probe": load_probe(PROBE)}
    runs = [run_model(d, c, splits) for d, c in zip(args.onnx_dir, args.checkpoint, strict=True)]

    report: dict[str, Any] = {"threads": 1, "models": {}, "comparison_vs_first": {}}
    for run in runs:
        name = Path(run["onnx_dir"]).name
        for kind in ("fp32", "int8"):
            for split, s in run["splits"].items():
                m = s["metrics"]
                print(
                    f"{name:<24} {split:<5} {kind:<5} acc {m[kind]['type_accuracy']:.4f} "
                    f"f1 {m[kind]['type_macro_f1']:.4f} span_f1 {m[kind]['span_f1']:.4f} "
                    f"exact {m[kind]['span_exact']:.4f} joint {m[kind]['joint']:.4f}"
                )
        report["models"][name] = {k: v for k, v in run.items() if k != "_raw"}
    for run in runs[1:]:
        key = f"{Path(run['onnx_dir']).name}_vs_{Path(runs[0]['onnx_dir']).name}"
        report["comparison_vs_first"][key] = compare(runs[0], run)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
