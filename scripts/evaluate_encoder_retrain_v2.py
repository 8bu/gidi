#!/usr/bin/env python
"""Score the three seeds of the annotation-v3 retrain v2 once on the held-out sets.

Usage:
    uv run python scripts/evaluate_encoder_retrain_v2.py \\
        [--onnx-dir experiments/annotation-v3-retrain-v2/onnx] \\
        [--out experiments/annotation-v3-retrain-v2/results.json] [--smoke]

Systems: the retrained INT8 encoders ``<onnx-dir>/seed{1,2,3}/model.int8.onnx`` (same bundle
config, tokenizer and decoding as ``models/gidi-finance-v1``) + the unchanged rule value parser
``gidi.value_parser.parse_value``. Held-out sets, scored once and never trained on: the frozen
test split (105), probe-v1 (81 complete labels) and ``datasets/annotation-v2/human-value-01``
(147 complete labels, annotation-v3 debt rule). The helpers are those of
``scripts/evaluate_encoder_retrain.py`` (the run of ``experiments/annotation-v3-retrain``).

Reused, not rerun: the V7 and V8-old rows of human-value-01 and the first retrain ("v3-retrain-v1")
rows are read from ``experiments/annotation-v3-retrain/results.json``. The old encoder is run on
the frozen test and probe-v1 only to list per-note changes; its aggregate metrics are asserted
equal to the stored ones.

The release-candidate rule (``SELECTION_RULE``) was fixed before any seed was scored and is
written to the results. Refuses to overwrite ``--out``. ``--smoke`` swaps every set for in-sample
data and never writes into ``experiments/``.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

import evaluate_encoder_retrain as base  # noqa: E402

from gidi.inference import GidiPredictor  # noqa: E402
from gidi.modeling.preprocessing import TYPES  # noqa: E402

PRIOR = ROOT / "experiments" / "annotation-v3-retrain" / "results.json"
DEFAULT_ONNX = ROOT / "experiments" / "annotation-v3-retrain-v2" / "onnx"
DEFAULT_OUT = ROOT / "experiments" / "annotation-v3-retrain-v2" / "results.json"
SEEDS = (1, 2, 3)

SELECTION_RULE = (
    "Fixed before any seed was scored. Gate (G) on the frozen test and human-value-01, old = the "
    "deployed encoder models/gidi-finance-v1: (a) test type macro-F1 drop <= 0.01, (b) test "
    "target exact drop <= 0.02, (c) human-value-01 debt-only slice (n=13, encoder + unchanged "
    "parser) end-to-end exact >= 0.8. The release candidate is seed 1 (deployment convention) "
    "unless seed 1 fails G while the three-seed mean passes G; then it is the seed whose test "
    "type macro-F1 is the median of the three. Verdict ACCEPT iff the release candidate passes "
    "G, else REJECT (a mean that passes with a failing candidate is reported, not accepted)."
)
MAX_TYPE_F1_DROP = 0.01
MAX_TARGET_DROP = 0.02
MIN_DEBT_E2E = 0.8
EPS = 1e-9

REGRESSION_IDS = (
    "baseline-01-b590dd3df660",  # tra lai chi Mai 2tr           repayment_out
    "baseline-01-07ca1f56011d",  # de rieng tien sua xe 800k     transfer
    "targeted-annotation-v1-01-84622e5a18da",  # Nhi ck tra lai 120k tien sua  repayment_in
    "targeted-annotation-v1-01-9d98b78854de",  # đòi được nợ thằng Lâm 400k    repayment_in
)


def gate(test_old: dict, test_new: dict, debt_e2e: float) -> dict[str, Any]:
    f1_drop = test_old["type_macro_f1"] - test_new["type_macro_f1"]
    target_drop = test_old["target_exact"] - test_new["target_exact"]
    checks = {
        "test_type_macro_f1_drop": {
            "value": f1_drop,
            "limit": MAX_TYPE_F1_DROP,
            "pass": f1_drop <= MAX_TYPE_F1_DROP + EPS,
        },
        "test_target_exact_drop": {
            "value": target_drop,
            "limit": MAX_TARGET_DROP,
            "pass": target_drop <= MAX_TARGET_DROP + EPS,
        },
        "human_debt_end_to_end": {
            "value": debt_e2e,
            "limit": MIN_DEBT_E2E,
            "pass": debt_e2e >= MIN_DEBT_E2E - EPS,
        },
    }
    return {"checks": checks, "pass": all(c["pass"] for c in checks.values())}


def aggregate(values: list[float | None]) -> dict[str, Any]:
    if any(v is None for v in values):  # an empty slice (smoke runs only)
        return {"values": values, "mean": None, "std": None, "min": None, "max": None}
    return {
        "values": values,
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def agg_block(blocks: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, Any]:
    return {k: aggregate([b[k] for b in blocks]) for k in keys}


def mean_block(blocks: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, float]:
    return {k: statistics.fmean(b[k] for b in blocks) for k in keys}


DEBT_WORDS = ("nợ", "no", "trả", "tra", "mượn", "muon", "vay")
TRAIN_SETS = {
    "training_v1": Path("datasets/annotation-v3/training-v1/train.jsonl"),
    "training_v2": Path("datasets/annotation-v3/training-v2/train.jsonl"),
}


def inspect_training_data() -> dict[str, Any]:
    """Per-type note counts and how many notes of each type contain a debt word (exact token)."""
    out: dict[str, Any] = {"debt_words": list(DEBT_WORDS)}
    for name, rel in TRAIN_SETS.items():
        rows = [json.loads(line) for line in (ROOT / rel).read_text("utf-8").splitlines()]
        tokens = [set(re.findall(r"\w+", r["text"].lower())) for r in rows]
        per_type: dict[str, Any] = {}
        for kind in TYPES:
            idx = [i for i, r in enumerate(rows) if r["type"] == kind]
            per_type[kind] = {
                "n": len(idx),
                "any_debt_word": sum(bool(tokens[i] & set(DEBT_WORDS)) for i in idx),
                **{w: sum(w in tokens[i] for i in idx) for w in DEBT_WORDS},
            }
        out[name] = {"n": len(rows), "per_type": per_type}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--onnx-dir", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--smoke", action="store_true", help="in-sample data only; debug run")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists: held-out scoring runs once")
    if args.smoke and ROOT / "experiments" in args.out.resolve().parents:
        raise SystemExit("--smoke must not write into experiments/")

    prior = json.loads(PRIOR.read_text("utf-8"))
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

    old_model = base.OLD_BUNDLE / "model.int8.onnx"
    old = GidiPredictor.from_bundle(base.OLD_BUNDLE, intra_op_threads=1)
    old_preds = {s: base.predict_all(old, sets[s]) for s in ("test", "probe")}
    old_blocks = {s: base.encoder_block(sets[s], *old_preds[s]) for s in ("test", "probe")}
    if not args.smoke:  # the reused rows are valid only if the old encoder is unchanged
        for s in ("test", "probe"):
            stored = prior["metrics"][s]["old"]
            for k in ("type_accuracy", "type_macro_f1", "target_exact", "span_f1"):
                if abs(old_blocks[s][k] - stored[k]) > EPS:
                    raise SystemExit(f"old encoder {s} {k} differs from the stored result")

    parser_values = base.value_spans(human)
    seeds: dict[str, Any] = {}
    for seed in SEEDS:
        model = args.onnx_dir / f"seed{seed}" / "model.int8.onnx"
        new = GidiPredictor.from_bundle(base.OLD_BUNDLE, model_path=model, intra_op_threads=1)
        preds = {s: base.predict_all(new, recs) for s, recs in sets.items()}
        entry: dict[str, Any] = {
            "model": {"path": str(model), "sha256": base.sha256_file(model)},
            "size_bytes": model.stat().st_size,
            "metrics": {},
            "diffs": {},
        }
        for s in ("test", "probe"):
            entry["metrics"][s] = base.encoder_block(sets[s], *preds[s])
            entry["diffs"][s] = base.diff_rows(sets[s], old_preds[s], preds[s])
        entry["metrics"]["human"] = {}
        for name, ids in slices.items():
            recs, t, sp, v = base.pick(human, *preds["human"], parser_values, ids=ids)
            entry["metrics"]["human"][name] = base.human_block(recs, t, sp, v)
        entry["human_borrow_lend_confusion"] = base.borrow_lend_confusion(human, preds["human"][0])
        recs, t = base.pick(human, preds["human"][0], ids=slices["non_debt"])
        entry["human_borrow_lend_confusion_non_debt"] = base.borrow_lend_confusion(recs, t)
        entry["human_value_exact"] = entry["metrics"]["human"]["all_complete"]["value_exact"]
        entry["regression_notes"] = regression_rows(sets["test"], old_preds["test"], preds["test"])
        entry["gate"] = gate(
            old_blocks["test"],
            entry["metrics"]["test"],
            entry["metrics"]["human"]["debt_only"]["end_to_end_exact"],
        )
        seeds[str(seed)] = entry

    # aggregate over seeds and the gate of the three-seed mean
    keys_enc = ("type_accuracy", "type_macro_f1", "target_exact", "span_f1")
    keys_human = (*keys_enc, "type_target_exact", "value_exact", "end_to_end_exact")
    aggregates: dict[str, Any] = {"seeds": list(SEEDS), "metrics": {}}
    for s in ("test", "probe"):
        blocks = [seeds[str(k)]["metrics"][s] for k in SEEDS]
        aggregates["metrics"][s] = agg_block(blocks, keys_enc) | {
            "per_class_f1": {t: aggregate([b["per_class_f1"][t] for b in blocks]) for t in TYPES}
        }
    aggregates["metrics"]["human"] = {
        name: agg_block([seeds[str(k)]["metrics"]["human"][name] for k in SEEDS], keys_human)
        for name in slices
    }
    mean_test = mean_block([seeds[str(k)]["metrics"]["test"] for k in SEEDS], keys_enc)
    mean_debt = statistics.fmean(
        seeds[str(k)]["metrics"]["human"]["debt_only"]["end_to_end_exact"] for k in SEEDS
    )
    mean_gate = gate(old_blocks["test"], mean_test, mean_debt)
    seed1_gate = seeds["1"]["gate"]
    ranked = sorted(SEEDS, key=lambda k: seeds[str(k)]["metrics"]["test"]["type_macro_f1"])
    if seed1_gate["pass"] or not mean_gate["pass"]:
        candidate, why = 1, "seed 1 (deployment convention)"
        if not seed1_gate["pass"]:
            why += "; it fails the gate and so does the mean"
    else:
        candidate = ranked[1]
        why = "seed 1 fails the gate while the mean passes: median-test-F1 seed"
    cand_gate = seeds[str(candidate)]["gate"]
    selection = {
        "rule": SELECTION_RULE,
        "seed1_pass": seed1_gate["pass"],
        "mean_pass": mean_gate["pass"],
        "candidate_seed": candidate,
        "candidate_reason": why,
        "candidate_pass": cand_gate["pass"],
        "verdict": "ACCEPT" if cand_gate["pass"] else "REJECT",
    }

    results: dict[str, Any] = {
        "experiment": "annotation-v3-retrain-v2",
        "smoke": args.smoke,
        "training_data": inspect_training_data(),
        "old_model": {"path": str(old_model), "sha256": base.sha256_file(old_model)},
        "sets": {s: len(r) for s, r in sets.items()},
        "human_value_01": {**human_meta, "complete": len(human)},
        "debt_only_ids": sorted(debt_ids),
        "old_encoder": old_blocks,
        "seeds": seeds,
        "aggregate": aggregates,
        "mean_gate": mean_gate,
        "mean_test_metrics": mean_test,
        "selection": selection,
        "reused_from_v3_retrain_v1": {
            "path": str(PRIOR.relative_to(ROOT)),
            "new_model": prior["new_model"],
            "v7_model": prior["v7_model"],
            "test": {k: prior["metrics"]["test"][k] for k in ("old", "new")},
            "probe": {k: prior["metrics"]["probe"][k] for k in ("old", "new")},
            "human": prior["metrics"]["human"],
            "diffs_test": prior["diffs"]["test"],
            "diffs_probe": prior["diffs"]["probe"],
        },
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
    for k in SEEDS:
        m = seeds[str(k)]["metrics"]["test"]
        print(
            f"seed{k} test: acc {m['type_accuracy']:.4f} F1 {m['type_macro_f1']:.4f} "
            f"target {m['target_exact']:.4f} | debt e2e "
            f"{seeds[str(k)]['metrics']['human']['debt_only']['end_to_end_exact']:.4f} | "
            f"gate {seeds[str(k)]['gate']['pass']}"
        )
    print(json.dumps(selection | {"rule": "(see SELECTION_RULE)"}, indent=1))
    return 0


def regression_rows(
    records: list[dict[str, Any]],
    old: tuple[list[str], list[dict[str, Any] | None]],
    new: tuple[list[str], list[dict[str, Any] | None]],
) -> list[dict[str, Any]]:
    """Old / new prediction of the 4 notes whose type flipped in the first retrain."""
    rows = []
    for i, r in enumerate(records):
        if r["id"] not in REGRESSION_IDS:
            continue
        o_span, n_span = old[1][i], new[1][i]
        rows.append(
            {
                "id": r["id"],
                "text": r["text"],
                "gold_type": r["type"],
                "gold_target": None if r["target"] is None else r["target"]["text"],
                "old_type": old[0][i],
                "old_target": None if o_span is None else o_span["text"],
                "new_type": new[0][i],
                "new_target": None if n_span is None else n_span["text"],
                "type_ok": new[0][i] == r["type"],
            }
        )
    return sorted(rows, key=lambda row: REGRESSION_IDS.index(row["id"]))


if __name__ == "__main__":
    raise SystemExit(main())
