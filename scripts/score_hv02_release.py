#!/usr/bin/env python
"""Score the two release bundles on human-value-02 and check v3 against the run-4 numbers.

Usage:
    uv run python scripts/score_hv02_release.py \\
        [--out experiments/deployment-v3/hv02-release-check.json]

Both bundles are loaded exactly as shipped, with no flags: ``models/gidi-finance-v3`` (INT8
encoder, whole-word snap and rule value parser from its ``config.json``) and
``models/gidi-finance-v2`` (type/target from the v1 encoder, value from its CRF head, no snap).
The 190 complete notes are ``datasets/annotation-v3/human-value-02`` (LLM labels: two independent
labellers plus adjudication, not human labels); ``systems_agreed`` is the subset the two labellers
agreed on.

The v3 row must equal the ``run4`` row of ``experiments/annotation-v3-retrain-v5/results-hv02.json``
(the same ONNX, snap on, parser) in every metric; the script exits 1 otherwise. Refuses to
overwrite ``--out``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

import evaluate_encoder_retrain as base  # noqa: E402

from gidi.inference import GidiPredictor  # noqa: E402

HV02 = ROOT / "datasets" / "annotation-v3" / "human-value-02"
RUN4_RESULTS = ROOT / "experiments" / "annotation-v3-retrain-v5" / "results-hv02.json"
DEFAULT_OUT = ROOT / "experiments" / "deployment-v3" / "hv02-release-check.json"
BUNDLES = {
    "gidi-finance-v3": ROOT / "models" / "gidi-finance-v3",
    "gidi-finance-v2": ROOT / "models" / "gidi-finance-v2",
}
SCALARS = (
    "n",
    "type_accuracy",
    "type_macro_f1",
    "target_exact",
    "span_f1",
    "value_exact",
    "type_target_exact",
    "end_to_end_exact",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists: the release check is written once")

    queue, labels = HV02 / "review-queue-all.jsonl", HV02 / "labels.jsonl"
    records, excluded = base.load_human(queue, labels)
    agreed = {
        r["id"]
        for r in base.read_jsonl(HV02 / "labels-provenance.jsonl")
        if r["provenance"] == "agreed"
    }
    subset = [i for i, r in enumerate(records) if r["id"] in agreed]

    systems: dict[str, Any] = {}
    systems_agreed: dict[str, Any] = {}
    for name, bundle in BUNDLES.items():
        predictor = GidiPredictor.from_bundle(bundle)
        types, spans, values = base.predict_all_v7(predictor, records)
        systems[name] = base.human_block(records, types, spans, values)
        systems_agreed[name] = base.human_block(
            [records[i] for i in subset],
            [types[i] for i in subset],
            [spans[i] for i in subset],
            [values[i] for i in subset],
        )

    run4 = json.loads(RUN4_RESULTS.read_text("utf-8"))
    mismatches = [
        f"{group}.{key}: {mine[key]} != {theirs[key]}"
        for group, mine, theirs in (
            ("systems", systems["gidi-finance-v3"], run4["systems"]["run4"]),
            ("systems_agreed", systems_agreed["gidi-finance-v3"], run4["systems_agreed"]["run4"]),
        )
        for key in SCALARS
        if mine[key] != theirs[key]
    ]
    results = {
        "experiment": "deployment-v3",
        "set": "human-value-02",
        "label_source": "LLM labels (two independent labellers + adjudication), not human labels",
        "n_complete": len(records),
        "n_excluded_not_complete": excluded,
        "n_agreed": len(subset),
        "labels_sha256": base.sha256_file(labels),
        "bundles": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "model_sha256": base.sha256_file(path / "model.int8.onnx"),
            }
            for name, path in BUNDLES.items()
        },
        "v3_equals_run4": not mismatches,
        "systems": systems,
        "systems_agreed": systems_agreed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(results, indent=1, ensure_ascii=False) + "\n")
    for group, rows in (("all", systems), ("agreed", systems_agreed)):
        for name, block in rows.items():
            print(
                f"{group:6s} {name:16s} n {block['n']} type {block['type_accuracy']:.4f} "
                f"target {block['target_exact']:.4f} value {block['value_exact']:.4f} "
                f"e2e {block['end_to_end_exact']:.4f}"
            )
    for line in mismatches:
        print(f"MISMATCH {line}")
    print("v3 equals run4:", not mismatches)
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
