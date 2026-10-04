#!/usr/bin/env python
"""Score the OLD encoder 1 and the retrained one (annotation-v3) once on the held-out sets.

Usage:
    uv run python scripts/evaluate_encoder_retrain.py \\
        [--new-int8 experiments/annotation-v3-retrain/onnx/model.int8.onnx] \\
        [--out experiments/annotation-v3-retrain/results.json] [--smoke]

Systems on human-value-01 (each scored once, no tuning):

    v7      models/gidi-finance-v2 INT8: encoder + its own value head (trained on the
            annotation-v1 rule, debt-only = skipped)
    v8_old  models/gidi-finance-v1 INT8 encoder (also annotation-v1 rule) + the unchanged rule
            parser ``gidi.value_parser.parse_value``
    v8_new  the retrained INT8 ONNX (same bundle config, tokenizer and decoding) + the parser

On the frozen test and probe-v1 only the two encoders (old, new) are scored.

Held-out sets (never trained on): the frozen test split (105), probe-v1 (81 complete labels) and
``datasets/annotation-v2/human-value-01`` (150 labels, annotation-v3 debt rule; ``uncertain``
labels are excluded and counted). The debt-only slice of human-value-01 is the 13 notes whose
label was ``skipped`` before the second review turned debt-only notes into borrow/lend
(``DEBT_ONLY_IDS``). Refuses to overwrite ``--out``: the held-out scoring runs once. ``--smoke``
swaps every set for in-sample data (validation split, debt-01 labels) and writes to ``--out``
only if it is outside the experiment directory; it never touches held-out data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

from evaluate_probe import load_probe  # noqa: E402

from gidi.corpus.jsonl import read_jsonl  # noqa: E402
from gidi.evaluation.metrics import task_metrics  # noqa: E402
from gidi.inference import GidiPredictor  # noqa: E402
from gidi.modeling.preprocessing import TYPES  # noqa: E402
from gidi.value_parser import parse_value  # noqa: E402

OLD_BUNDLE = ROOT / "models" / "gidi-finance-v1"
V7_BUNDLE = ROOT / "models" / "gidi-finance-v2"
DEFAULT_NEW = ROOT / "experiments" / "annotation-v3-retrain" / "onnx" / "model.int8.onnx"
DEFAULT_OUT = ROOT / "experiments" / "annotation-v3-retrain" / "results.json"
TEST = ROOT / "datasets" / "annotation-v1" / "splits" / "test.jsonl"
VALIDATION = ROOT / "datasets" / "annotation-v1" / "splits" / "validation.jsonl"
PROBE_DIR = ROOT / "datasets" / "probe-v1"
HUMAN = ROOT / "datasets" / "annotation-v2" / "human-value-01"
DEBT = ROOT / "datasets" / "annotation-v3" / "debt-01"

# human-value-01 notes whose label was `skipped` before the second review (commit ea3c10f) and is
# borrow/lend now: the debt-only notes under the annotation-v3 rule. The first 8 are annotation-v1
# `skipped` notes too (the 9th, 3f29fa04ff7a, is `uncertain`); the other 5 were never queued.
DEBT_ONLY_IDS = (
    "baseline-01-1f8b559fe3c3",
    "baseline-01-3f8d7d6f6fac",
    "baseline-01-43b46e772c2b",
    "baseline-01-4a30fae51dfa",
    "baseline-01-5facc59b6ce4",
    "baseline-01-6b471f45cc33",
    "baseline-01-790ca645bdb7",
    "baseline-01-eff9d5387a64",
    "baseline-01-a74b152d7abb",
    "baseline-01-32d4f5a1a0c5",
    "baseline-01-bbba32c0b2c6",
    "baseline-01-79ef1ca54d34",
    "baseline-01-9cbf3e661f54",
)
V1_SKIPPED_DEBT_IDS = DEBT_ONLY_IDS[:8]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(span: dict[str, Any] | None) -> tuple[int, int] | None:
    return None if span is None else (span["start"], span["end"])


# --------------------------------------------------------------------------- data


def load_human(queue: Path, labels: Path) -> tuple[list[dict[str, Any]], int]:
    """Complete labelled notes (text, type, target, value) and the number of excluded others."""
    text_of = {r["id"]: r["text"] for r in read_jsonl(queue)}
    records, excluded = [], 0
    for label in read_jsonl(labels):
        if label["annotation_status"] != "complete":
            excluded += 1
            continue
        records.append(
            {
                "id": label["id"],
                "text": text_of[label["id"]],
                "type": label["type"],
                "target": label["target"],
                "value": label["value"],
            }
        )
    return records, excluded


def load_sets(smoke: bool) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if smoke:
        val = read_jsonl(VALIDATION)
        human, excluded = load_human(DEBT / "queue.jsonl", DEBT / "labels.jsonl")
        return {"test": val, "probe": val, "human": human}, {"uncertain_excluded": excluded}
    human, excluded = load_human(HUMAN / "review-queue.jsonl", HUMAN / "labels.jsonl")
    sets = {"test": read_jsonl(TEST), "probe": load_probe(PROBE_DIR), "human": human}
    return sets, {"uncertain_excluded": excluded}


# --------------------------------------------------------------------------- scoring


def predict_all(
    predictor: GidiPredictor, records: list[dict[str, Any]]
) -> tuple[list[str], list[dict[str, Any] | None]]:
    types, spans = [], []
    for r in records:
        p = predictor.predict(r["text"])
        types.append(p.type)
        spans.append(
            None
            if p.target_span is None
            else {
                "text": r["text"][p.target_span[0] : p.target_span[1]],
                "start": p.target_span[0],
                "end": p.target_span[1],
            }
        )
    return types, spans


def read_queue(smoke: bool) -> list[dict[str, Any]]:
    return [] if smoke else read_jsonl(HUMAN / "review-queue.jsonl")


def predict_all_v7(
    predictor: GidiPredictor, records: list[dict[str, Any]]
) -> tuple[list[str], list[dict[str, Any] | None], list[dict[str, Any] | None]]:
    """Type, target and value spans of the released V7 bundle (its own value head)."""
    types, spans, values = [], [], []
    for r in records:
        p = predictor.predict(r["text"])
        text = r["text"]
        types.append(p.type)
        t = p.target_span
        spans.append(None if t is None else {"text": text[t[0] : t[1]], "start": t[0], "end": t[1]})
        v = None if p.value is None else p.value.span
        values.append(
            None if v is None else {"text": text[v[0] : v[1]], "start": v[0], "end": v[1]}
        )
    return types, spans, values


def value_spans(records: list[dict[str, Any]]) -> list[dict[str, Any] | None]:
    out = []
    for r in records:
        span = parse_value(r["text"])
        out.append(
            None
            if span is None
            else {"text": r["text"][span.start : span.end], "start": span.start, "end": span.end}
        )
    return out


def rate(flags: list[bool]) -> dict[str, Any]:
    return {"n": len(flags), "exact": sum(flags) / len(flags) if flags else None, "k": sum(flags)}


def encoder_block(
    records: list[dict[str, Any]], types: list[str], spans: list[dict[str, Any] | None]
) -> dict[str, Any]:
    m = task_metrics(records, types, spans)
    return {
        "n": m["n"],
        "type_accuracy": m["type"]["accuracy"],
        "type_macro_f1": m["type"]["macro_f1"],
        "target_exact": m["target"]["exact_match"],
        "span_f1": m["target"]["f1"],
        "per_class_f1": {t: c["f1"] for t, c in m["type"]["per_class"].items()},
        "confusion_matrix": m["type"]["confusion_matrix"],
    }


def pick(records, *lists, ids: set[str]):
    idx = [i for i, r in enumerate(records) if r["id"] in ids]
    return ([records[i] for i in idx], *([lst[i] for i in idx] for lst in lists))


def human_block(
    records: list[dict[str, Any]],
    types: list[str],
    spans: list[dict[str, Any] | None],
    values: list[dict[str, Any] | None],
) -> dict[str, Any]:
    out = encoder_block(records, types, spans)
    type_ok = [t == r["type"] for t, r in zip(types, records, strict=True)]
    target_ok = [key(s) == key(r["target"]) for s, r in zip(spans, records, strict=True)]
    value_ok = [key(v) == key(r["value"]) for v, r in zip(values, records, strict=True)]
    out["value_exact"] = rate(value_ok)["exact"]
    out["type_target_exact"] = rate([a and b for a, b in zip(type_ok, target_ok, strict=True)])[
        "exact"
    ]
    out["end_to_end_exact"] = rate(
        [a and b and c for a, b, c in zip(type_ok, target_ok, value_ok, strict=True)]
    )["exact"]
    return out


def diff_rows(
    records: list[dict[str, Any]],
    old: tuple[list[str], list[dict[str, Any] | None]],
    new: tuple[list[str], list[dict[str, Any] | None]],
) -> list[dict[str, Any]]:
    rows = []
    for i, r in enumerate(records):
        o_t, o_s, n_t, n_s = old[0][i], old[1][i], new[0][i], new[1][i]
        if o_t == n_t and key(o_s) == key(n_s):
            continue
        rows.append(
            {
                "id": r["id"],
                "text": r["text"],
                "gold_type": r["type"],
                "gold_target": None if r["target"] is None else r["target"]["text"],
                "old_type": o_t,
                "old_target": None if o_s is None else o_s["text"],
                "new_type": n_t,
                "new_target": None if n_s is None else n_s["text"],
                "old_type_ok": o_t == r["type"],
                "new_type_ok": n_t == r["type"],
                "old_target_ok": key(o_s) == key(r["target"]),
                "new_target_ok": key(n_s) == key(r["target"]),
            }
        )
    return rows


def borrow_lend_confusion(
    records: list[dict[str, Any]], types: list[str]
) -> dict[str, dict[str, int]]:
    """Rows = gold borrow/lend, columns = predicted type (only non-zero cells)."""
    table: dict[str, Counter] = {"borrow": Counter(), "lend": Counter()}
    for r, t in zip(records, types, strict=True):
        if r["type"] in table:
            table[r["type"]][t] += 1
    return {gold: dict(cells) for gold, cells in table.items()}


# --------------------------------------------------------------------------- latency


def latency(predictors: dict[str, GidiPredictor], texts: list[str], runs: int, warmup: int):
    """Single-note ``predict`` wall time in ms, 1 thread, systems interleaved per round."""
    samples: dict[str, list[float]] = {name: [] for name in predictors}
    for p in predictors.values():
        for i in range(warmup):
            p.predict(texts[i % len(texts)])
    for i in range(runs):
        text = texts[i % len(texts)]
        for name, p in predictors.items():
            t0 = time.perf_counter()
            p.predict(text)
            samples[name].append((time.perf_counter() - t0) * 1e3)
    out = {}
    for name, xs in samples.items():
        xs = sorted(xs)
        out[name] = {
            "runs": len(xs),
            "p50_ms": statistics.median(xs),
            "p95_ms": xs[int(0.95 * (len(xs) - 1))],
            "mean_ms": statistics.fmean(xs),
        }
    return out


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--new-int8", type=Path, default=DEFAULT_NEW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--smoke", action="store_true", help="in-sample data only; debug run")
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists: held-out scoring runs once")
    if args.smoke and ROOT / "experiments" in args.out.resolve().parents:
        raise SystemExit("--smoke must not write into experiments/")

    sets, human_meta = load_sets(args.smoke)
    old_model = OLD_BUNDLE / "model.int8.onnx"
    v7_model = V7_BUNDLE / "model.int8.onnx"
    predictors = {
        "old": GidiPredictor.from_bundle(OLD_BUNDLE, intra_op_threads=1),
        "new": GidiPredictor.from_bundle(OLD_BUNDLE, model_path=args.new_int8, intra_op_threads=1),
    }
    v7 = GidiPredictor.from_bundle(V7_BUNDLE, intra_op_threads=1)
    human = sets["human"]
    human_ids = {r["id"] for r in human}
    debt_ids = human_ids if args.smoke else set(DEBT_ONLY_IDS)
    v1_skipped = debt_ids if args.smoke else set(V1_SKIPPED_DEBT_IDS)
    missing = debt_ids - human_ids
    if missing:
        raise SystemExit(f"debt-only ids not in the human set: {sorted(missing)}")

    preds = {
        name: {s: predict_all(p, recs) for s, recs in sets.items()}
        for name, p in predictors.items()
    }
    v7_types, v7_spans, v7_values = predict_all_v7(v7, human)
    parser_values = value_spans(human)
    # system -> (types, target spans, value spans) on human-value-01
    systems = {
        "v7": (v7_types, v7_spans, v7_values),
        "v8_old": (*preds["old"]["human"], parser_values),
        "v8_new": (*preds["new"]["human"], parser_values),
    }

    results: dict[str, Any] = {
        "experiment": "annotation-v3-retrain",
        "smoke": args.smoke,
        "old_model": {"path": str(old_model), "sha256": sha256_file(old_model)},
        "new_model": {"path": str(args.new_int8), "sha256": sha256_file(args.new_int8)},
        "v7_model": {"path": str(v7_model), "sha256": sha256_file(v7_model)},
        "systems": {
            "v7": "models/gidi-finance-v2 INT8: encoder + its own value head (CRF)",
            "v8_old": "models/gidi-finance-v1 INT8 encoder + gidi.value_parser.parse_value",
            "v8_new": "retrained INT8 encoder + gidi.value_parser.parse_value",
        },
        "sets": {s: len(r) for s, r in sets.items()},
        "human_value_01": {**human_meta, "complete": len(human)},
        "debt_only_ids": sorted(debt_ids),
        "metrics": {},
        "diffs": {},
    }
    for set_name in ("test", "probe"):
        recs = sets[set_name]
        results["metrics"][set_name] = {
            name: encoder_block(recs, *preds[name][set_name]) for name in predictors
        }
        results["diffs"][set_name] = diff_rows(recs, preds["old"][set_name], preds["new"][set_name])

    slices = {
        "all_complete": human_ids,
        "debt_only": debt_ids,
        "debt_only_v1_skipped": v1_skipped,
        "non_debt": human_ids - debt_ids,
    }
    human_metrics: dict[str, Any] = {}
    for slice_name, ids in slices.items():
        human_metrics[slice_name] = {}
        for name, (types, spans, values) in systems.items():
            recs, t, s, v = pick(human, types, spans, values, ids=ids)
            human_metrics[slice_name][name] = human_block(recs, t, s, v)
    results["metrics"]["human"] = human_metrics

    strata_of = {r["id"]: set(r.get("strata", ())) for r in read_queue(args.smoke)}
    strata_names = sorted({x for v in strata_of.values() for x in v} | {"null_value"})
    strata_ids = {
        name: {
            r["id"]
            for r in human
            if name in strata_of.get(r["id"], ()) or (name == "null_value" and r["value"] is None)
        }
        for name in strata_names
    }
    results["human_value_by_stratum"] = {
        name: {
            system: rate(
                [key(values[i]) == key(r["value"]) for i, r in enumerate(human) if r["id"] in ids]
            )
            for system, (_, _, values) in systems.items()
        }
        for name, ids in strata_ids.items()
    }
    results["human_borrow_lend_confusion"] = {}
    results["human_borrow_lend_confusion_non_debt"] = {}
    for name, (types, _, _) in systems.items():
        results["human_borrow_lend_confusion"][name] = borrow_lend_confusion(human, types)
        recs, t = pick(human, types, ids=slices["non_debt"])
        results["human_borrow_lend_confusion_non_debt"][name] = borrow_lend_confusion(recs, t)
    diffs = diff_rows(human, preds["old"]["human"], preds["new"]["human"])
    by_id = {r["id"]: r for r in human}
    for row in diffs:
        row["debt_only"] = row["id"] in debt_ids
        gold_value = by_id[row["id"]]["value"]
        row["gold_value"] = None if gold_value is None else gold_value["text"]
    results["diffs"]["human"] = diffs

    sample = random.Random(0).sample([r["text"] for r in sets["test"]], min(50, len(sets["test"])))
    results["latency_int8_ms"] = latency(predictors, sample, args.runs, args.warmup)
    old_bundle_other = sum(
        p.stat().st_size for p in OLD_BUNDLE.iterdir() if p.name != old_model.name
    )
    results["size_bytes"] = {
        "old_int8_onnx": old_model.stat().st_size,
        "new_int8_onnx": args.new_int8.stat().st_size,
        "bundle_other_files": old_bundle_other,
        "old_bundle": old_model.stat().st_size + old_bundle_other,
        "new_bundle": args.new_int8.stat().st_size + old_bundle_other,
    }
    results["environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "types": list(TYPES),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(results, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")
    for set_name in ("test", "probe"):
        for name in predictors:
            m = results["metrics"][set_name][name]
            print(
                f"{set_name:6s} {name}: acc {m['type_accuracy']:.4f} F1 {m['type_macro_f1']:.4f} "
                f"target {m['target_exact']:.4f} span F1 {m['span_f1']:.4f}"
            )
    for slice_name in slices:
        for name in systems:
            m = human_metrics[slice_name][name]
            if not m["n"]:
                continue
            print(
                f"human/{slice_name:20s} {name:6s}: n={m['n']} acc {m['type_accuracy']:.4f} "
                f"tgt {m['target_exact']:.4f} val {m['value_exact']:.4f} "
                f"e2e {m['end_to_end_exact']:.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
