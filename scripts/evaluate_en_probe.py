#!/usr/bin/env python
"""Score the exported gidi-en student (int8 ONNX) on an English evaluation-only probe.

Usage:
    uv run python scripts/evaluate_en_probe.py --probe datasets/probe-en-slang \
        --onnx-dir models/gidi-en-v1-onnx/seed1 --checkpoint models/gidi-en-v1/seed1 \
        --out experiments/gidi-en-student-v1/slang-probe.json

Scores the ``complete`` notes of the probe overall and split by label source (``consensus`` vs
``claude-adjudicated``). Metrics are the ones of ``evaluate_en_student.py`` (type macro-F1, target
exact match, joint exact match). Nothing here trains on or tunes against the probe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_en_student import evaluate_set, softmax, worst_errors
from evaluate_onnx import decode, onnx_outputs

from gidi.corpus.jsonl import read_jsonl
from gidi.export.onnx_export import make_session
from gidi.modeling.checkpoint import load_checkpoint
from gidi.modeling.preprocessing import TYPES
from gidi.training.train import prepare


def load_probe(probe: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    texts = {r["id"]: r["text"] for r in read_jsonl(probe / "notes.jsonl")}
    source = {r["id"]: r["source"] for r in read_jsonl(probe / "provenance.jsonl")}
    records = []
    for label in read_jsonl(probe / "labels.jsonl"):
        if label["annotation_status"] != "complete":
            continue
        if label["type"] not in TYPES:
            raise ValueError(f"{label['id']!r} has unknown type {label['type']!r}")
        text, target = texts[label["id"]], label["target"]
        if target is not None and text[target["start"] : target["end"]] != target["text"]:
            raise ValueError(f"{label['id']!r} target is not an exact substring")
        records.append({"id": label["id"], "text": text, "type": label["type"], "target": target})
    return records, source


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--probe", type=Path, required=True)
    ap.add_argument("--onnx-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--snap-words", action="store_true", help="expand predicted spans to whole-word boundaries"
    )
    args = ap.parse_args(argv)

    records, source = load_probe(args.probe)
    _, tokenizer, meta = load_checkpoint(args.checkpoint)
    session = make_session(args.onnx_dir / "model.int8.onnx", threads=1)
    data = prepare(tokenizer, records, int(meta["max_length"]))
    type_logits, tag_logits = onnx_outputs(session, data)
    types, spans = decode(data, type_logits, tag_logits, snap_words=args.snap_words)

    def subset(keep: set[str]) -> dict[str, Any]:
        idx = [i for i, r in enumerate(records) if source[r["id"]] in keep]
        return evaluate_set(
            [records[i] for i in idx], [types[i] for i in idx], [spans[i] for i in idx]
        )

    results = {
        "int8_file": str(args.onnx_dir / "model.int8.onnx"),
        "snap_words": args.snap_words,
        "all": subset({"consensus", "claude-adjudicated"}),
        "consensus": subset({"consensus"}),
        "adjudicated": subset({"claude-adjudicated"}),
        "worst_errors": worst_errors(records, types, spans, softmax(type_logits), n=15),
        "predictions": [
            {
                "id": r["id"],
                "text": r["text"],
                "source": source[r["id"]],
                "gold": [r["type"], r["target"] and r["target"]["text"]],
                "pred": [t, s and r["text"][s["start"] : s["end"]]],
            }
            for r, t, s in zip(records, types, spans, strict=True)
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name in ("all", "consensus", "adjudicated"):
        r = results[name]
        print(name, r["n"], {k: round(v, 3) for k, v in r.items() if isinstance(v, float)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
