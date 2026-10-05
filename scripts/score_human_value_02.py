#!/usr/bin/env python
"""Score the retrain-v4 release candidate on the human-labelled test set human-value-02, once.

Usage (after the user has labelled the set):
    uv run python scripts/score_human_value_02.py \\
        [--out experiments/annotation-v3-retrain-v4/results-hv02.json]

Refuses to run until ``datasets/annotation-v3/human-value-02/labels.jsonl`` exists, and refuses to
overwrite ``--out`` (the set is scored once). Systems, all INT8 encoder (type + target) + the
unchanged rule value parser ``gidi.value_parser.parse_value``:

* ``candidate``: retrain-v4 seed 1 with ``snap_words=True`` (the release candidate fixed in
  ``experiments/annotation-v3-retrain-v4/protocol.json``); its ONNX hash must equal the one the
  held-out run (``results.json``) scored;
* ``candidate_nosnap``: the same model without the snap (isolates the snap);
* ``old``: the deployed encoder ``models/gidi-finance-v1`` (no snap);
* ``run2``: retrain-v2 seed 1 (no snap), the best earlier single model; ``run2_snap``: with it.

Reported: type accuracy / macro-F1, target exact, value exact, end-to-end exact (type + target +
value all exact), per-class recall, the notes each system gets wrong, and the verdict against the
protocol criterion (candidate end-to-end exact >= 0.95). ``--smoke`` scores in-sample debt-01
labels instead, writes only outside ``experiments/`` and never touches human-value-02.
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
import evaluate_encoder_retrain_v3 as v3e  # noqa: E402

from gidi.inference import GidiPredictor  # noqa: E402

HV02 = ROOT / "datasets" / "annotation-v3" / "human-value-02"
EXP = ROOT / "experiments" / "annotation-v3-retrain-v4"
PROTOCOL = EXP / "protocol.json"
CANDIDATE_MODEL = EXP / "onnx" / "seed1" / "model.int8.onnx"
HELD_OUT_RESULTS = EXP / "results.json"
RUN2_MODEL = v3e.V2_ONNX / "seed1" / "model.int8.onnx"
DEFAULT_OUT = EXP / "results-hv02.json"
THRESHOLD = 0.95


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--smoke", action="store_true", help="in-sample debt-01; debug run")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists: human-value-02 is scored once")
    if args.smoke:
        if ROOT / "experiments" in args.out.resolve().parents:
            raise SystemExit("--smoke must not write into experiments/")
        queue, labels = base.DEBT / "queue.jsonl", base.DEBT / "labels.jsonl"
    else:
        queue, labels = HV02 / "review-queue.jsonl", HV02 / "labels.jsonl"
        if not labels.is_file():
            raise SystemExit(
                f"{labels} does not exist yet: label human-value-02 first (Quet), then run this"
            )
    for needed in (PROTOCOL, CANDIDATE_MODEL, RUN2_MODEL):
        if not needed.is_file():
            raise SystemExit(f"missing {needed}")
    if not args.smoke:
        scored = json.loads(HELD_OUT_RESULTS.read_text("utf-8"))["models"]["v4_seed1"]["sha256"]
        if base.sha256_file(CANDIDATE_MODEL) != scored:
            raise SystemExit("candidate ONNX differs from the model scored in results.json")

    records, excluded = base.load_human(queue, labels)
    if not records:
        raise SystemExit("no complete label to score")
    queue_ids = {r["id"] for r in base.read_jsonl(queue)}
    unknown = sorted({r["id"] for r in base.read_jsonl(labels)} - queue_ids)
    if unknown:
        raise SystemExit(f"labels for ids outside the queue: {unknown[:5]}")
    parser_values = base.value_spans(records)

    plan: dict[str, tuple[Path | None, bool]] = {
        "candidate": (CANDIDATE_MODEL, True),
        "candidate_nosnap": (CANDIDATE_MODEL, False),
        "old": (None, False),
        "run2": (RUN2_MODEL, False),
        "run2_snap": (RUN2_MODEL, True),
    }
    systems: dict[str, Any] = {}
    wrong: dict[str, list[dict[str, Any]]] = {}
    preds = {}
    for name, (model, snap) in plan.items():
        predictor = GidiPredictor.from_bundle(
            base.OLD_BUNDLE, model_path=model, intra_op_threads=1, snap_words=snap
        )
        types, spans = base.predict_all(predictor, records)
        preds[name] = (types, spans)
        block = base.human_block(records, types, spans, parser_values)
        block["recall_by_type"] = v3e.recall_by_type(records, types)
        systems[name] = block
    for name in plan:
        others = {k: v for k, v in preds.items() if k != name}
        rows = v3e.wrong_rows(records, preds[name], parser_values, others)
        for row in rows:
            row["kind"] = "+".join(
                k
                for k, ok in (
                    ("type", row["type_ok"]),
                    ("target", row["target_ok"]),
                    ("value", row["value_ok"]),
                )
                if not ok
            )
        wrong[name] = rows

    candidate = systems["candidate"]["end_to_end_exact"]
    results = {
        "experiment": "annotation-v3-retrain-v4",
        "set": "human-value-02" if not args.smoke else "debt-01 (smoke)",
        "smoke": args.smoke,
        "n_complete": len(records),
        "n_excluded_not_complete": excluded,
        "labels_sha256": base.sha256_file(labels),
        "queue_sha256": base.sha256_file(queue),
        "protocol_sha256": base.sha256_file(PROTOCOL),
        "models": {
            "candidate": {
                "path": str(CANDIDATE_MODEL),
                "sha256": base.sha256_file(CANDIDATE_MODEL),
            },
            "old": {
                "path": str(base.OLD_BUNDLE / "model.int8.onnx"),
                "sha256": base.sha256_file(base.OLD_BUNDLE / "model.int8.onnx"),
            },
            "run2": {"path": str(RUN2_MODEL), "sha256": base.sha256_file(RUN2_MODEL)},
        },
        "criterion": {"candidate_end_to_end_exact_at_least": THRESHOLD},
        "candidate_end_to_end_exact": candidate,
        "verdict": "PASS" if candidate >= THRESHOLD else "FAIL",
        "systems": systems,
        "wrong_notes": wrong,
        "wrong_by_kind": {
            name: {
                kind: sum(r["kind"] == kind for r in rows)
                for kind in sorted({r["kind"] for r in rows})
            }
            for name, rows in wrong.items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(results, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {args.out} (n={len(records)})")
    for name, block in systems.items():
        print(
            f"{name:16s} type acc {block['type_accuracy']:.4f} F1 {block['type_macro_f1']:.4f} "
            f"target {block['target_exact']:.4f} value {block['value_exact']:.4f} "
            f"e2e {block['end_to_end_exact']:.4f}"
        )
    print(f"verdict: {results['verdict']} (candidate e2e {candidate:.4f}, need {THRESHOLD})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
