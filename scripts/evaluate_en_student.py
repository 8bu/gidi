#!/usr/bin/env python
"""Score the exported gidi-en student (int8 and fp32 ONNX) against ``protocol.json``.

Usage:
    uv run python scripts/evaluate_en_student.py --onnx-dir models/gidi-en-v1-onnx/seed1 \\
        --checkpoint models/gidi-en-v1/seed1 --out experiments/gidi-en-student-v1/results.json

Every score is reported twice: ``int8``/``fp32`` with the raw BIO span, and ``int8_snap`` with the
span expanded to whole-word boundaries (``snap_words``, see ``docs/annotation-en-v1.md``).
Sets: ``primary`` = complete test notes with consensus labels, ``secondary`` = all complete test
notes, ``dev`` = complete dev notes (sanity only). Pass/fail is computed from the thresholds in
``experiments/gidi-en-data/protocol.json``; nothing is tuned on any set.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from evaluate_onnx import decode, onnx_outputs, score, span_key

from gidi.evaluation.metrics import type_metrics
from gidi.export.onnx_export import make_session
from gidi.modeling.checkpoint import load_checkpoint
from gidi.training.english import load_labeled, primary_ids
from gidi.training.train import prepare

PROTOCOL = Path("experiments/gidi-en-data/protocol.json")


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def evaluate_set(records: list[dict], types: list[str], spans: list) -> dict[str, Any]:
    gold = [r["type"] for r in records]
    tm = type_metrics(gold, types)
    result = score(records, types, spans)
    result["n"] = len(records)
    result["per_type"] = {
        t: {"f1": c["f1"], "precision": c["precision"], "recall": c["recall"], "n": c["support"]}
        for t, c in tm["per_class"].items()
        if c["support"] > 0
    }
    return result


def check(primary: dict, secondary: dict, protocol: dict) -> dict[str, Any]:
    pa, sa = protocol["acceptance"]["primary"], protocol["acceptance"]["secondary"]
    per_type_min = pa["per_type_f1_min_for_types_with_n_ge_10"]
    primary_checks = {
        "type_macro_f1": (primary["type_macro_f1"], pa["type_macro_f1_min"]),
        "target_exact": (primary["span_exact"], pa["target_exact_match_min"]),
        "joint": (primary["joint"], pa["joint_exact_match_min"]),
    }
    secondary_checks = {
        "type_macro_f1": (secondary["type_macro_f1"], sa["type_macro_f1_min"]),
        "target_exact": (secondary["span_exact"], sa["target_exact_match_min"]),
    }
    per_type = {t: (c["f1"], per_type_min) for t, c in primary["per_type"].items() if c["n"] >= 10}

    def verdict(items: dict) -> dict[str, Any]:
        return {k: {"value": v, "min": m, "pass": v >= m} for k, (v, m) in items.items()}

    out = {
        "primary": verdict(primary_checks),
        "primary_per_type_f1": verdict(per_type),
        "secondary": verdict(secondary_checks),
    }
    out["pass"] = all(c["pass"] for group in out.values() for c in group.values())
    return out


def worst_errors(
    records: list[dict], types: list[str], spans: list, probs: np.ndarray, n: int = 10
) -> list[dict[str, Any]]:
    """Errors ranked by: both type and target wrong first, then by the confidence of the type."""
    rows = []
    for r, t, s, p in zip(records, types, spans, probs, strict=True):
        type_wrong = t != r["type"]
        span_wrong = span_key(s) != span_key(r["target"])
        if not (type_wrong or span_wrong):
            continue
        rows.append(
            {
                "id": r["id"],
                "text": r["text"],
                "gold_type": r["type"],
                "pred_type": t,
                "gold_target": r["target"]["text"] if r["target"] else None,
                "pred_target": s["text"] if s else None,
                "pred_type_prob": round(float(p.max()), 3),
            }
        )
    rows.sort(
        key=lambda x: (
            -((x["gold_type"] != x["pred_type"]) + (x["gold_target"] != x["pred_target"])),
            -x["pred_type_prob"],
        )
    )
    return rows[:n]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--onnx-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, default=PROTOCOL)
    args = ap.parse_args(argv)

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    _, tokenizer, meta = load_checkpoint(args.checkpoint)
    max_length = int(meta["max_length"])
    test = load_labeled("test")
    keep = primary_ids("test")
    sets = {
        "primary": [r for r in test if r["id"] in keep],
        "secondary": test,
        "dev": load_labeled("dev"),
    }
    int8_path = args.onnx_dir / "model.int8.onnx"
    results: dict[str, Any] = {
        "int8_file": str(int8_path),
        "int8_bytes": int8_path.stat().st_size,
        "fp32_bytes": (args.onnx_dir / "model.onnx").stat().st_size,
        "n": {k: len(v) for k, v in sets.items()},
    }
    errors: dict[str, Any] = {}
    for kind, name, snap in (
        ("int8", "model.int8.onnx", False),
        ("int8_snap", "model.int8.onnx", True),
        ("fp32", "model.onnx", False),
    ):
        session = make_session(args.onnx_dir / name, threads=1)
        results[kind] = {}
        for set_name, records in sets.items():
            data = prepare(tokenizer, records, max_length)
            type_logits, tag_logits = onnx_outputs(session, data)
            types, spans = decode(data, type_logits, tag_logits, snap_words=snap)
            results[kind][set_name] = evaluate_set(records, types, spans)
            if kind in ("int8", "int8_snap") and set_name == "secondary":
                errors[kind] = worst_errors(records, types, spans, softmax(type_logits))
    results["protocol"] = check(results["int8"]["primary"], results["int8"]["secondary"], protocol)
    results["protocol_snap"] = check(
        results["int8_snap"]["primary"], results["int8_snap"]["secondary"], protocol
    )
    results["worst_errors_secondary"] = errors
    assert not math.isnan(results["int8"]["primary"]["type_macro_f1"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({k: results[k] for k in ("int8_bytes", "protocol", "protocol_snap")}, indent=1)[
            :3000
        ]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
