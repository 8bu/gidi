"""Pure aggregation helpers for the baseline grid summary (``scripts/summarize_baseline.py``).

A *run* is a plain dict with the parsed contents of one run directory:
``{"model", "lr", "seed", "metrics", "checkpoint", ...}`` where ``metrics`` is ``metrics.json``
(``{"validation": <evaluate>, "test": <evaluate>}``) and ``checkpoint`` is ``checkpoint.json``.
Nothing here touches the filesystem, so every function can be checked on hand-built dicts.

Spreads use the *sample* standard deviation (``None`` for a single seed) and skip ``None``
values, keeping the number of values actually averaged in ``n``.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from gidi.modeling.preprocessing import TYPES

# name -> path into an ``evaluate`` split's ``overall`` block
HEADLINE: dict[str, tuple[str, ...]] = {
    "type_accuracy": ("type", "accuracy"),
    "type_macro_f1": ("type", "macro_f1"),
    "span_f1": ("target", "f1"),
    "span_exact_match": ("target", "exact_match"),
    "target_exact_match": ("target", "gold_target", "exact_match"),
    "null_accuracy": ("target", "gold_null", "null_accuracy"),
    "false_span_rate": ("target", "gold_null", "false_span_rate"),
}
SLICE_METRICS = ("type_accuracy", "type_macro_f1", "span_f1", "span_exact_match")
CLASS_METRICS = ("precision", "recall", "f1")


def selection_score(metrics: Mapping[str, Any]) -> float:
    """mean(type macro-F1, span F1) of an ``evaluate`` result; mirrors ``training.train``."""
    overall = metrics["overall"]
    return ((overall["type"]["macro_f1"] or 0.0) + (overall["target"]["f1"] or 0.0)) / 2


def _dig(block: Mapping[str, Any], path: Sequence[str]) -> Any:
    for key in path:
        block = block[key]
    return block


def headline(block: Mapping[str, Any]) -> dict[str, float | None]:
    """The reported scalars of one ``evaluate`` block (``overall`` or a slice entry)."""
    return {name: _dig(block, path) for name, path in HEADLINE.items()}


def mean_spread(values: Iterable[float | None]) -> dict[str, Any]:
    """``{n, mean, std, min, max}`` over the non-``None`` values (sample std, ``None`` if n<2)."""
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": len(xs),
        "mean": statistics.fmean(xs),
        "std": statistics.stdev(xs) if len(xs) > 1 else None,
        "min": min(xs),
        "max": max(xs),
    }


def _mean(values: Iterable[float | None]) -> float | None:
    return mean_spread(values)["mean"]


def val_score(run: Mapping[str, Any]) -> float:
    return selection_score(run["metrics"]["validation"])


def _split_counts(block: Mapping[str, Any]) -> dict[str, int]:
    target = block["overall"]["target"]
    return {
        "n": block["n"],
        "n_gold_target": target["gold_target"]["n"],
        "n_gold_null": target["gold_null"]["n"],
    }


def aggregate_config(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Across-seed summary of the runs of one (model, lr) configuration."""
    if not runs:
        raise ValueError("no runs to aggregate")
    runs = sorted(runs, key=lambda r: r["seed"])
    first = runs[0]
    out: dict[str, Any] = {
        "model": first["model"],
        "model_name": first.get("model_name", first["model"]),
        "lr": first["lr"],
        "n_seeds": len(runs),
        "seeds": [r["seed"] for r in runs],
        "per_seed": [
            {
                "seed": r["seed"],
                "best_epoch": r["checkpoint"]["best_epoch"],
                "epochs_run": r["checkpoint"]["epochs_run"],
                "val_score": val_score(r),
            }
            for r in runs
        ],
        "val_score": mean_spread(val_score(r) for r in runs),
        "wall_time_sec": mean_spread(r["checkpoint"].get("wall_time_sec") for r in runs),
        "sec_per_epoch": mean_spread(r["checkpoint"].get("sec_per_epoch") for r in runs),
        "param_count": first["checkpoint"].get("param_count"),
        "split_n": {
            split: _split_counts(first["metrics"][split]) for split in ("validation", "test")
        },
    }
    for split in ("validation", "test"):
        per_run = [headline(r["metrics"][split]["overall"]) for r in runs]
        out[split] = {name: mean_spread(p[name] for p in per_run) for name in HEADLINE}
    return out


def best_run(runs: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Run with the highest validation selection score (ties: lowest seed)."""
    return min(runs, key=lambda r: (-val_score(r), r["seed"]))


def best_config_per_model(configs: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """Per model, the config with the highest mean validation selection score (ties: lowest lr)."""
    best: dict[str, Mapping[str, Any]] = {}
    for cfg in sorted(configs, key=lambda c: (c["val_score"]["mean"] is None, c["lr"])):
        if cfg["val_score"]["mean"] is None:
            continue
        cur = best.get(cfg["model"])
        if cur is None or cfg["val_score"]["mean"] > cur["val_score"]["mean"]:
            best[cfg["model"]] = cfg
    return best


def _same(values: Sequence[Any], what: str) -> Any:
    if any(v != values[0] for v in values):
        raise ValueError(f"{what} differs across runs: {sorted(set(map(str, values)))}")
    return values[0]


def per_class_mean(runs: Sequence[Mapping[str, Any]], split: str = "test") -> dict[str, Any]:
    """Per-class P/R/F1 averaged over runs, with the (run-independent) support, in TYPES order."""
    out: dict[str, Any] = {}
    for label in TYPES:
        rows = [r["metrics"][split]["overall"]["type"]["per_class"][label] for r in runs]
        out[label] = {
            **{m: _mean(row[m] for row in rows) for m in CLASS_METRICS},
            "support": _same([row["support"] for row in rows], f"support of {label!r}"),
            "n_pred": _mean(row["n_pred"] for row in rows),
        }
    return out


def summed_confusion(runs: Sequence[Mapping[str, Any]], split: str = "test") -> list[list[int]]:
    """Element-wise sum of the gold-by-pred confusion matrices (TYPES order)."""
    total = [[0] * len(TYPES) for _ in TYPES]
    for r in runs:
        type_block = r["metrics"][split]["overall"]["type"]
        if type_block["labels"] != list(TYPES):
            raise ValueError(f"confusion labels {type_block['labels']} are not TYPES order")
        for i, row in enumerate(type_block["confusion_matrix"]):
            for j, count in enumerate(row):
                total[i][j] += count
    return total


def slice_means(
    runs: Sequence[Mapping[str, Any]], split: str = "test"
) -> dict[str, dict[str, dict[str, Any]]]:
    """``{slice: {value: {n, n_runs, <metric>: mean over runs}}}``; ``n`` must agree across runs."""
    names = sorted({name for r in runs for name in r["metrics"][split]["slices"]})
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for name in names:
        per_run = [r["metrics"][split]["slices"].get(name, {}) for r in runs]
        values = sorted({v for slices in per_run for v in slices})
        out[name] = {}
        for value in values:
            blocks = [slices[value] for slices in per_run if value in slices]
            heads = [headline(b) for b in blocks]
            entry: dict[str, Any] = {
                "n": _same([b["n"] for b in blocks], f"n of slice {name}={value}"),
                "n_runs": len(blocks),
            }
            entry.update({m: _mean(h[m] for h in heads) for m in SLICE_METRICS})
            out[name][value] = entry
    return out


def _span_key(span: Mapping[str, Any] | None) -> tuple[int, int] | None:
    return None if span is None else (int(span["start"]), int(span["end"]))


def _span_view(span: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if span is None:
        return None
    return {"text": span.get("text"), "start": int(span["start"]), "end": int(span["end"])}


def type_error(pred: Mapping[str, Any]) -> bool:
    return pred["gold_type"] != pred["pred_type"]


def span_error(pred: Mapping[str, Any]) -> bool:
    return _span_key(pred["gold_span"]) != _span_key(pred["pred_span"])


def failure_cases(predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Type errors and span errors (null == null is correct) among ``predictions_test`` rows."""
    return {
        "n": len(predictions),
        "type_errors": [
            {
                "id": p["id"],
                "text": p["text"],
                "gold_type": p["gold_type"],
                "pred_type": p["pred_type"],
            }
            for p in predictions
            if type_error(p)
        ],
        "span_errors": [
            {
                "id": p["id"],
                "text": p["text"],
                "gold_span": _span_view(p["gold_span"]),
                "pred_span": _span_view(p["pred_span"]),
            }
            for p in predictions
            if span_error(p)
        ],
    }


def shared_failures(
    predictions_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Records (matched by id) on which *every* model errs, per error kind.

    Each entry keeps the gold value and every model's prediction. Ids not predicted by all
    models are ignored; their count is reported as ``n_unmatched``.
    """
    indexed = {m: {p["id"]: p for p in preds} for m, preds in predictions_by_model.items()}
    if len(indexed) < 2:
        raise ValueError("need predictions from at least two models")
    common = set.intersection(*(set(ix) for ix in indexed.values()))
    n_all = len(set.union(*(set(ix) for ix in indexed.values())))
    ordered = sorted(common)
    models = sorted(indexed)
    ref = indexed[models[0]]
    type_rows, span_rows = [], []
    for rid in ordered:
        rows = {m: indexed[m][rid] for m in models}
        if all(type_error(p) for p in rows.values()):
            type_rows.append(
                {
                    "id": rid,
                    "text": ref[rid]["text"],
                    "gold_type": ref[rid]["gold_type"],
                    "pred_type": {m: p["pred_type"] for m, p in rows.items()},
                }
            )
        if all(span_error(p) for p in rows.values()):
            span_rows.append(
                {
                    "id": rid,
                    "text": ref[rid]["text"],
                    "gold_span": _span_view(ref[rid]["gold_span"]),
                    "pred_span": {m: _span_view(p["pred_span"]) for m, p in rows.items()},
                }
            )
    return {
        "models": models,
        "n_common": len(ordered),
        "n_unmatched": n_all - len(ordered),
        "type_errors": type_rows,
        "span_errors": span_rows,
    }
