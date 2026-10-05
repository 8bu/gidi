#!/usr/bin/env python
"""Score the three seeds of the annotation-v3 retrain v5 (word snap on) once.

Usage:
    uv run python scripts/evaluate_encoder_retrain_v5.py \\
        [--onnx-dir experiments/annotation-v3-retrain-v5/onnx] \\
        [--out experiments/annotation-v3-retrain-v5/results.json] [--smoke]

Systems (every one = encoder type + target span; value = the unchanged rule parser
``gidi.value_parser.parse_value``):

* ``v5_seed{1,2,3}_snap``: the retrained INT8 encoders with ``GidiPredictor(snap_words=True)``;
  ``v5_seed1`` is the release candidate (``protocol.json``, written before training);
  ``v5_seed1_nosnap`` isolates the data effect from the snap effect;
* ``v4_seed1_snap``: run 4 seed 1 (the previous candidate) rescored on the amended labels;
* ``old`` / ``old_snap``: ``models/gidi-finance-v1`` without / with the snap;
* ``v2_seed1`` / ``v2_seed1_snap``: retrain-v2 seed 1 without / with the snap.

Held-out sets, scored once and never trained on: the frozen test split (105), probe-v1 (81
complete labels) and ``datasets/annotation-v2/human-value-01`` (147 complete labels, annotation-v3
debt rule, **amendment-01**: the gift receiver is the target; every system is scored on the amended
labels). human-value-02 is not read here. The no-snap ``old`` and ``v2_seed1`` rows are asserted
equal to the stored retrain-v2 results on the frozen test and probe-v1, and the ``v4_seed1_snap``
row equal to the stored run 4 results on those two sets and on every human-value-01 note except
the amended ones, before anything is written. Refuses to overwrite ``--out``. ``--smoke`` swaps
every set for in-sample data and never writes into ``experiments/``.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

import evaluate_encoder_retrain as base  # noqa: E402
import evaluate_encoder_retrain_v2 as v2e  # noqa: E402
import evaluate_encoder_retrain_v3 as v3e  # noqa: E402

from gidi.inference import GidiPredictor  # noqa: E402
from gidi.modeling.preprocessing import TYPES  # noqa: E402

V2_RESULTS = v3e.V2_RESULTS
V2_ONNX = v3e.V2_ONNX
V4_ONNX = ROOT / "experiments" / "annotation-v3-retrain-v4" / "onnx" / "seed1" / "model.int8.onnx"
V4_RESULTS = ROOT / "experiments" / "annotation-v3-retrain-v4" / "results.json"
AMENDMENT = ROOT / "datasets" / "annotation-v2" / "human-value-01" / "amendment-01.json"
DEFAULT_ONNX = ROOT / "experiments" / "annotation-v3-retrain-v5" / "onnx"
DEFAULT_OUT = ROOT / "experiments" / "annotation-v3-retrain-v5" / "results.json"
PROTOCOL = ROOT / "experiments" / "annotation-v3-retrain-v5" / "protocol.json"
SEEDS = (1, 2, 3)
KEYS_ENC = v3e.KEYS_ENC
KEYS_HUMAN = v3e.KEYS_HUMAN


def wrong_kind(row: dict[str, Any]) -> str:
    return "+".join(
        name
        for name, ok in (
            ("type", row["type_ok"]),
            ("target", row["target_ok"]),
            ("value", row["value_ok"]),
        )
        if not ok
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--onnx-dir", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--smoke", action="store_true", help="in-sample data only; debug run")
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists: held-out scoring runs once")
    if args.smoke and ROOT / "experiments" in args.out.resolve().parents:
        raise SystemExit("--smoke must not write into experiments/")
    if not PROTOCOL.is_file():
        raise SystemExit(f"{PROTOCOL} must exist before scoring")

    v2 = json.loads(V2_RESULTS.read_text("utf-8"))
    sets, human_meta = base.load_sets(args.smoke)
    human = sets["human"]
    human_ids = {r["id"] for r in human}
    debt_ids = human_ids if args.smoke else set(base.DEBT_ONLY_IDS)
    v1_skipped = debt_ids if args.smoke else set(base.V1_SKIPPED_DEBT_IDS)
    if debt_ids - human_ids:
        raise SystemExit(f"debt-only ids not in the human set: {sorted(debt_ids - human_ids)}")
    slices = {
        "all_complete": human_ids,
        "debt_only": debt_ids,
        "debt_only_v1_skipped": v1_skipped,
        "non_debt": human_ids - debt_ids,
    }
    parser_values = base.value_spans(human)

    def load(model: Path | None, snap: bool) -> GidiPredictor:
        return GidiPredictor.from_bundle(
            base.OLD_BUNDLE, model_path=model, intra_op_threads=1, snap_words=snap
        )

    old_model = base.OLD_BUNDLE / "model.int8.onnx"
    v2_model = V2_ONNX / "seed1" / "model.int8.onnx"
    v4_model = V4_ONNX
    v5_models = {s: args.onnx_dir / f"seed{s}" / "model.int8.onnx" for s in SEEDS}
    plan: dict[str, tuple[Path | None, bool]] = {
        "old": (None, False),
        "old_snap": (None, True),
        "v2_seed1": (v2_model, False),
        "v2_seed1_snap": (v2_model, True),
        "v4_seed1_snap": (v4_model, True),
        "v5_seed1_nosnap": (v5_models[1], False),
        **{f"v5_seed{s}_snap": (m, True) for s, m in v5_models.items()},
    }
    systems: dict[str, Any] = {}
    preds: dict[str, dict[str, Any]] = {}
    predictors: dict[str, GidiPredictor] = {}
    for name, (model, snap) in plan.items():
        predictors[name] = load(model, snap)
        preds[name] = {s: base.predict_all(predictors[name], r) for s, r in sets.items()}
        systems[name] = v3e.system_metrics(sets, preds[name], parser_values, slices)

    if not args.smoke:  # the no-snap controls are valid only if the re-run reproduces run 2
        stored = v2["seeds"]["1"]["metrics"]
        for s in ("test", "probe"):
            v3e.assert_equal(f"v2 seed1 {s}", systems["v2_seed1"][s], stored[s], KEYS_ENC)
            v3e.assert_equal(f"old {s}", systems["old"][s], v2["old_encoder"][s], KEYS_ENC)
        stored4 = json.loads(V4_RESULTS.read_text("utf-8"))
        for s_ in ("test", "probe"):
            v3e.assert_equal(
                f"run 4 seed1 {s_}",
                systems["v4_seed1_snap"][s_],
                stored4["systems"]["v4_seed1_snap"][s_],
                KEYS_ENC,
            )
        amended = {c["id"] for c in json.loads(AMENDMENT.read_text("utf-8"))["changes"]}
        now_wrong = {
            r["id"]
            for r in v3e.wrong_rows(human, preds["v4_seed1_snap"]["human"], parser_values, {})
        }
        stored_wrong = {r["id"] for r in stored4["wrong_human_notes"]["v4_seed1_snap"]}
        if (now_wrong ^ stored_wrong) - amended:
            raise SystemExit(
                "run 4 seed1 human: wrong notes differ from the stored ones beyond the "
                f"amended labels: {sorted((now_wrong ^ stored_wrong) - amended)}"
            )

    seed_names = [f"v5_seed{k}_snap" for k in SEEDS]
    aggregates: dict[str, Any] = {"seeds": list(SEEDS), "metrics": {}}
    for s in ("test", "probe"):
        blocks = [systems[n][s] for n in seed_names]
        aggregates["metrics"][s] = v2e.agg_block(blocks, KEYS_ENC) | {
            "per_class_f1": {
                t: v2e.aggregate([b["per_class_f1"][t] for b in blocks]) for t in TYPES
            }
        }
    aggregates["metrics"]["human"] = {
        name: v2e.agg_block([systems[n]["human"][name] for n in seed_names], KEYS_HUMAN)
        for name in slices
    }

    others = {
        "old": preds["old"]["human"],
        "v4_seed1_snap": preds["v4_seed1_snap"]["human"],
        "v2_seed1": preds["v2_seed1"]["human"],
        "v2_seed1_snap": preds["v2_seed1_snap"]["human"],
        "v5_seed1_nosnap": preds["v5_seed1_nosnap"]["human"],
    }
    wrong = {
        name: v3e.wrong_rows(human, preds[name]["human"], parser_values, others)
        for name in seed_names
    }
    for rows in wrong.values():
        for row in rows:
            row["kind"] = wrong_kind(row)
    wrong_by_kind = {
        name: dict(sorted(_count(r["kind"] for r in rows).items())) for name, rows in wrong.items()
    }

    sample = random.Random(0).sample([r["text"] for r in sets["test"]], min(50, len(sets["test"])))
    latency = base.latency(
        {
            "candidate_snap_off": predictors["v5_seed1_nosnap"],
            "candidate_snap_on": predictors["v5_seed1_snap"],
        },
        sample,
        args.runs,
        args.warmup,
    )
    other_files = sum(
        p.stat().st_size for p in base.OLD_BUNDLE.iterdir() if p.name != old_model.name
    )
    sizes = {f"seed{s}_int8_onnx": m.stat().st_size for s, m in v5_models.items()}
    size = {
        **sizes,
        "bundle_other_files": other_files,
        "single_bundle": sizes["seed1_int8_onnx"] + other_files,
    }

    results: dict[str, Any] = {
        "experiment": "annotation-v3-retrain-v5",
        "smoke": args.smoke,
        "protocol_sha256": base.sha256_file(PROTOCOL),
        "release_candidate": "v5_seed1_snap",
        "old_model": {"path": str(old_model), "sha256": base.sha256_file(old_model)},
        "v2_seed1_model": {"path": str(v2_model), "sha256": base.sha256_file(v2_model)},
        "v4_seed1_model": {"path": str(v4_model), "sha256": base.sha256_file(v4_model)},
        "models": {
            f"v5_seed{s}": {"path": str(m), "sha256": base.sha256_file(m)}
            for s, m in v5_models.items()
        },
        "sets": {s: len(r) for s, r in sets.items()},
        "human_value_01": {
            **human_meta,
            "complete": len(human),
            "labels_sha256": base.sha256_file(base.HUMAN / "labels.jsonl"),
            "amendment_sha256": base.sha256_file(AMENDMENT),
        },
        "debt_only_ids": sorted(debt_ids),
        "systems": systems,
        "aggregate_v5_seeds": aggregates,
        "reused_from_v2": {
            "path": str(V2_RESULTS.relative_to(ROOT)),
            "v2_mean": v2["aggregate"],
        },
        "wrong_human_notes": wrong,
        "wrong_by_kind": wrong_by_kind,
        "size_bytes": size,
        "latency_int8_ms": latency,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "types": list(TYPES),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(results, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")
    for name in plan:
        m, h = systems[name]["test"], systems[name]["human"]["all_complete"]
        d = systems[name]["human"]["debt_only"]
        print(
            f"{name:16s} test acc {m['type_accuracy']:.4f} F1 {m['type_macro_f1']:.4f} "
            f"tgt {m['target_exact']:.4f} | probe F1 {systems[name]['probe']['type_macro_f1']:.4f}"
            f" | human acc {h['type_accuracy']:.4f} tgt {h['target_exact']:.4f} "
            f"e2e {h['end_to_end_exact']:.4f} debt e2e {d['end_to_end_exact']:.4f}"
        )
    print("wrong by kind:", json.dumps(wrong_by_kind))
    print(json.dumps(latency, indent=1))
    return 0


def _count(items) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        out[item] = out.get(item, 0) + 1
    return out


if __name__ == "__main__":
    raise SystemExit(main())
