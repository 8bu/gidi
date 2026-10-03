#!/usr/bin/env python
"""Score existing checkpoints on the diagnostic probe set (evaluation only; never trains).

Usage:
    uv run python scripts/evaluate_probe.py [--probe-dir datasets/probe-v1]
        [--weights-dir models/baseline-v1-e20] [--out experiments/baseline-v1/probe-eval.json]

Scores every checkpoint directory ``<weights-dir>/<model>/<run>/`` on the complete probe labels,
sliced by ``pattern`` and ``accented``, and writes per-run metrics plus per-record predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gidi.annotation.schema import load_config, trainable
from gidi.corpus.jsonl import read_jsonl
from gidi.evaluation.metrics import evaluate
from gidi.modeling.checkpoint import load_checkpoint
from gidi.training.train import predict, prepare


def load_probe(probe_dir: Path) -> list[dict]:
    notes = {n["id"]: n for n in read_jsonl(probe_dir / "notes.jsonl")}
    labels = trainable(read_jsonl(probe_dir / "labels.jsonl"), load_config())
    return [
        {
            "id": label["id"],
            "text": notes[label["id"]]["text"],
            "pattern": notes[label["id"]]["pattern"],
            "type": label["type"],
            "target": label["target"],
        }
        for label in labels
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--probe-dir", type=Path, default=Path("datasets/probe-v1"))
    parser.add_argument("--weights-dir", type=Path, default=Path("models/baseline-v1-e20"))
    parser.add_argument("--out", type=Path, default=Path("experiments/baseline-v1/probe-eval.json"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {args.out} (pass --overwrite)")

    records = load_probe(args.probe_dir)
    device = torch.device("cpu")
    runs = {}
    for ckpt in sorted(p.parent for p in args.weights_dir.glob("*/*/gidi_checkpoint.json")):
        model, tokenizer, meta = load_checkpoint(ckpt)
        data = prepare(tokenizer, records, int(meta["max_length"]))
        types, spans = predict(model, data, device)
        name = f"{ckpt.parent.name}/{ckpt.name}"
        runs[name] = {
            "checkpoint": str(ckpt),
            "metrics": evaluate(records, types, spans, slices=("pattern", "accented")),
            "predictions": [
                {"id": r["id"], "pred_type": t, "pred_span": s}
                for r, t, s in zip(records, types, spans, strict=True)
            ],
        }
        print(f"{name}: type acc {runs[name]['metrics']['overall']['type']['accuracy']:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"n": len(records), "records": records, "runs": runs}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    print(f"{len(records)} probe records x {len(runs)} runs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
