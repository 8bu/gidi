#!/usr/bin/env python
"""Score the retrain-v5 release candidate on the test set human-value-02 (200 notes), once.

Usage (after the labels exist):
    uv run python scripts/score_human_value_02_v5.py \\
        [--out experiments/annotation-v3-retrain-v5/results-hv02.json]

The 200 notes are ``review-queue-all.jsonl`` (the 150 of ``review-queue.jsonl`` followed by the 50
of ``review-queue-b.jsonl``). Their labels are ``labels.jsonl``: **LLM labels**, not human labels
(the user approved this for the test set): two independent LLM labellers (A and B), the records on
which they agreed as is and the others adjudicated by a third pass
(``labels-provenance.jsonl``: ``agreed`` / ``adjudicated``; ``manifest-labels.json``). Every system
is scored on all complete notes (``systems``) and on the ``agreed`` subset only
(``systems_agreed``, the labels no single model decided).

Refuses to run until ``datasets/annotation-v3/human-value-02/labels.jsonl`` exists, and refuses to
overwrite ``--out`` (the set is scored once). Systems, all INT8 encoder (type + target) + the
unchanged rule value parser ``gidi.value_parser.parse_value``:

* ``candidate``: retrain-v5 seed 1 with ``snap_words=True`` (the release candidate fixed in
  ``experiments/annotation-v3-retrain-v5/protocol.json``); its ONNX hash must equal the one the
  held-out run (``results.json``) scored;
* ``candidate_nosnap``: the same model without the snap (isolates the snap);
* ``run4``: retrain-v4 seed 1 with ``snap_words=True`` (the previous candidate); its ONNX hash must
  equal the one the run 4 held-out run scored;
* ``old``: the deployed encoder ``models/gidi-finance-v1`` (no snap);
* ``run2``: retrain-v2 seed 1 (no snap), the best earlier single model; ``run2_snap``: with it.

Reported: type accuracy / macro-F1, target exact, value exact, end-to-end exact (type + target +
value all exact), per-class recall, the notes each system gets wrong, and the verdict against the
protocol criterion (candidate end-to-end exact >= 0.95) on all complete notes (``verdict``) and on
the agreed subset (``verdict_agreed``). ``--smoke`` scores in-sample debt-01 labels instead,
writes only outside ``experiments/`` and never touches human-value-02.
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
EXP = ROOT / "experiments" / "annotation-v3-retrain-v5"
EXP4 = ROOT / "experiments" / "annotation-v3-retrain-v4"
PROTOCOL = EXP / "protocol.json"
CANDIDATE_MODEL = EXP / "onnx" / "seed1" / "model.int8.onnx"
HELD_OUT_RESULTS = EXP / "results.json"
RUN4_MODEL = EXP4 / "onnx" / "seed1" / "model.int8.onnx"
RUN4_RESULTS = EXP4 / "results.json"
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
        queue, labels = HV02 / "review-queue-all.jsonl", HV02 / "labels.jsonl"
        if not labels.is_file():
            raise SystemExit(
                f"{labels} does not exist yet: label human-value-02 first (Quet), then run this"
            )
    for needed in (PROTOCOL, CANDIDATE_MODEL, RUN4_MODEL, RUN2_MODEL):
        if not needed.is_file():
            raise SystemExit(f"missing {needed}")
    if not args.smoke:
        scored = json.loads(HELD_OUT_RESULTS.read_text("utf-8"))["models"]["v5_seed1"]["sha256"]
        if base.sha256_file(CANDIDATE_MODEL) != scored:
            raise SystemExit("candidate ONNX differs from the model scored in results.json")
        scored4 = json.loads(RUN4_RESULTS.read_text("utf-8"))["models"]["v4_seed1"]["sha256"]
        if base.sha256_file(RUN4_MODEL) != scored4:
            raise SystemExit("run 4 ONNX differs from the model scored in the run 4 results.json")

    records, excluded = base.load_human(queue, labels)
    if not records:
        raise SystemExit("no complete label to score")
    queue_ids = {r["id"] for r in base.read_jsonl(queue)}
    unknown = sorted({r["id"] for r in base.read_jsonl(labels)} - queue_ids)
    if unknown:
        raise SystemExit(f"labels for ids outside the queue: {unknown[:5]}")
    parser_values = base.value_spans(records)
    provenance_path = HV02 / "labels-provenance.jsonl"
    agreed_ids: set[str] | None = None
    if not args.smoke:
        if not provenance_path.is_file():
            raise SystemExit(f"{provenance_path} does not exist: needed for the agreed subset")
        agreed_ids = {
            r["id"] for r in base.read_jsonl(provenance_path) if r["provenance"] == "agreed"
        }
    agreed_idx = (
        [i for i, r in enumerate(records) if r["id"] in agreed_ids]
        if agreed_ids is not None
        else []
    )

    plan: dict[str, tuple[Path | None, bool]] = {
        "candidate": (CANDIDATE_MODEL, True),
        "candidate_nosnap": (CANDIDATE_MODEL, False),
        "run4": (RUN4_MODEL, True),
        "old": (None, False),
        "run2": (RUN2_MODEL, False),
        "run2_snap": (RUN2_MODEL, True),
    }
    systems: dict[str, Any] = {}
    systems_agreed: dict[str, Any] = {}
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
        if agreed_ids is not None:
            sub = [records[i] for i in agreed_idx]
            sub_types = [types[i] for i in agreed_idx]
            sub_block = base.human_block(
                sub,
                sub_types,
                [spans[i] for i in agreed_idx],
                [parser_values[i] for i in agreed_idx],
            )
            sub_block["recall_by_type"] = v3e.recall_by_type(sub, sub_types)
            systems_agreed[name] = sub_block
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
    candidate_agreed = (
        systems_agreed["candidate"]["end_to_end_exact"] if agreed_ids is not None else None
    )
    results = {
        "experiment": "annotation-v3-retrain-v5",
        "set": "human-value-02" if not args.smoke else "debt-01 (smoke)",
        "smoke": args.smoke,
        "n_complete": len(records),
        "n_excluded_not_complete": excluded,
        "labels_sha256": base.sha256_file(labels),
        "label_source": None
        if args.smoke
        else "LLM labels (two independent labellers + adjudication), not human labels",
        "n_agreed": len(agreed_idx) if agreed_ids is not None else None,
        "provenance_sha256": None if args.smoke else base.sha256_file(provenance_path),
        "queue_sha256": base.sha256_file(queue),
        "protocol_sha256": base.sha256_file(PROTOCOL),
        "models": {
            "candidate": {
                "path": str(CANDIDATE_MODEL),
                "sha256": base.sha256_file(CANDIDATE_MODEL),
            },
            "run4": {"path": str(RUN4_MODEL), "sha256": base.sha256_file(RUN4_MODEL)},
            "old": {
                "path": str(base.OLD_BUNDLE / "model.int8.onnx"),
                "sha256": base.sha256_file(base.OLD_BUNDLE / "model.int8.onnx"),
            },
            "run2": {"path": str(RUN2_MODEL), "sha256": base.sha256_file(RUN2_MODEL)},
        },
        "criterion": {"candidate_end_to_end_exact_at_least": THRESHOLD},
        "candidate_end_to_end_exact": candidate,
        "verdict": "PASS" if candidate >= THRESHOLD else "FAIL",
        "candidate_end_to_end_exact_agreed": candidate_agreed,
        "verdict_agreed": (
            None
            if candidate_agreed is None
            else "PASS"
            if candidate_agreed >= THRESHOLD
            else "FAIL"
        ),
        "systems": systems,
        "systems_agreed": systems_agreed or None,
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
    if agreed_ids is not None:
        print(f"-- agreed subset (n={len(agreed_idx)}) --")
        for name, block in systems_agreed.items():
            print(
                f"{name:16s} type acc {block['type_accuracy']:.4f} F1 {block['type_macro_f1']:.4f} "
                f"target {block['target_exact']:.4f} value {block['value_exact']:.4f} "
                f"e2e {block['end_to_end_exact']:.4f}"
            )
        print(f"verdict_agreed: {results['verdict_agreed']} (candidate e2e {candidate_agreed:.4f})")
    print(f"verdict: {results['verdict']} (candidate e2e {candidate:.4f}, need {THRESHOLD})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
