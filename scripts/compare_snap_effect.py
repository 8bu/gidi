#!/usr/bin/env python
r"""Paired effect of the word snap on the target span: notes fixed / broken / changed-but-wrong.

Usage:
    uv run python scripts/compare_snap_effect.py \\
        [--out experiments/annotation-v3-retrain-v4/snap-effect.json]

Re-runs the INT8 models of the retrain-v4 evaluation (old encoder, retrain-v2 seed 1, retrain-v4
seed 1) on the frozen test, probe-v1 and human-value-01 with and without
``GidiPredictor(snap_words=True)`` and counts, per set, the notes whose target span is exactly
the gold only with the snap (``fixed``), only without it (``broken``), and the notes the snap
changed that stay wrong (``still_wrong``). Same sets and loaders as
``scripts/evaluate_encoder_retrain_v4.py``; no human-value-02.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

import evaluate_encoder_retrain as base  # noqa: E402
import evaluate_encoder_retrain_v3 as v3e  # noqa: E402

from gidi.inference import GidiPredictor  # noqa: E402

EXP = ROOT / "experiments" / "annotation-v3-retrain-v4"
MODELS = {
    "old": None,
    "run2_seed1": v3e.V2_ONNX / "seed1" / "model.int8.onnx",
    "run4_seed1": EXP / "onnx" / "seed1" / "model.int8.onnx",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=EXP / "snap-effect.json")
    args = parser.parse_args(argv)
    sets, _ = base.load_sets(smoke=False)
    out: dict[str, dict[str, dict[str, object]]] = {}
    for name, model in MODELS.items():
        plain, snapped = (
            GidiPredictor.from_bundle(
                base.OLD_BUNDLE, model_path=model, intra_op_threads=1, snap_words=snap
            )
            for snap in (False, True)
        )
        out[name] = {}
        for set_name, records in sets.items():
            _, a = base.predict_all(plain, records)
            _, b = base.predict_all(snapped, records)
            fixed, broken, still_wrong, changed = [], [], [], 0
            for r, x, y in zip(records, a, b, strict=True):
                gold = base.key(r["target"])
                if base.key(x) != base.key(y):
                    changed += 1
                    if base.key(y) == gold:
                        fixed.append(r["text"])
                    elif base.key(x) == gold:
                        broken.append(r["text"])
                    else:
                        still_wrong.append(r["text"])
            out[name][set_name] = {
                "n": len(records),
                "changed": changed,
                "fixed": len(fixed),
                "broken": len(broken),
                "still_wrong": len(still_wrong),
                "broken_notes": broken,
            }
    with args.out.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    for name, per_set in out.items():
        for set_name, c in per_set.items():
            print(
                f"{name:11s} {set_name:6s} changed {c['changed']:3d} fixed {c['fixed']:3d} "
                f"broken {c['broken']:3d} still wrong {c['still_wrong']:3d}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
