"""Metrics for the two Gidi tasks: transaction type and target span.

Everything returned is plain JSON-able Python. Spans are compared as character offsets, so a
prediction is exact only when ``(start, end)`` equals the gold ``(start, end)``.

Conventions (kept explicit because small splits make them matter):

- Type ``macro_f1`` averages over classes with ``support > 0`` in the gold labels;
  ``macro_f1_all_classes`` averages over every class, scoring absent ones 0.
- Span ``exact_match`` counts null == null as exact. Span precision/recall/F1 count only
  non-null spans: a predicted non-null span is a true positive iff exact; precision is over
  predicted non-null spans, recall over gold non-null spans. Undefined ratios inside a non-empty
  set are 0.0; rates over an empty subset are ``None``.
- Every slice carries its own ``n``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from gidi.annotation.split import accented as text_is_accented
from gidi.modeling.preprocessing import TYPES

Span = Mapping[str, Any] | None


def _ratio(num: float, den: float) -> float:
    return num / den if den else 0.0


def _prf(tp: int, n_pred: int, n_gold: int) -> tuple[float, float, float]:
    precision, recall = _ratio(tp, n_pred), _ratio(tp, n_gold)
    return precision, recall, _ratio(2 * precision * recall, precision + recall)


def type_metrics(
    gold: Sequence[str], pred: Sequence[str], labels: Sequence[str] = TYPES
) -> dict[str, Any]:
    """Accuracy, macro-F1, per-class P/R/F1 + support, and a gold-by-pred confusion matrix."""
    if len(gold) != len(pred):
        raise ValueError(f"{len(gold)} gold types but {len(pred)} predictions")
    index = {label: i for i, label in enumerate(labels)}
    for name, values in (("gold", gold), ("predicted", pred)):
        unknown = sorted({v for v in values if v not in index})
        if unknown:
            raise ValueError(f"unknown {name} type labels: {unknown}")

    confusion = [[0] * len(labels) for _ in labels]
    for g, p in zip(gold, pred, strict=True):
        confusion[index[g]][index[p]] += 1

    per_class: dict[str, dict[str, Any]] = {}
    for label, i in index.items():
        tp = confusion[i][i]
        support = sum(confusion[i])
        n_pred = sum(row[i] for row in confusion)
        precision, recall, f1 = _prf(tp, n_pred, support)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "n_pred": n_pred,
        }

    n = len(gold)
    supported = [c["f1"] for c in per_class.values() if c["support"] > 0]
    all_f1 = [c["f1"] for c in per_class.values()]
    return {
        "n": n,
        "accuracy": _ratio(sum(confusion[i][i] for i in range(len(labels))), n) if n else None,
        "macro_f1": _ratio(sum(supported), len(supported)) if supported else None,
        "macro_f1_all_classes": _ratio(sum(all_f1), len(all_f1)) if n else None,
        "per_class": per_class,
        "labels": list(labels),
        "confusion_matrix": confusion,
    }


def _bounds(span: Span) -> tuple[int, int] | None:
    return None if span is None else (int(span["start"]), int(span["end"]))


def _span_group(gold: Sequence[tuple[int, int] | None], pred: Sequence[tuple[int, int] | None]):
    """Counts shared by every span view: n, exact matches, TP, predicted/gold non-null."""
    return {
        "n": len(gold),
        "n_exact": sum(g == p for g, p in zip(gold, pred, strict=True)),
        "tp": sum(g is not None and g == p for g, p in zip(gold, pred, strict=True)),
        "n_pred": sum(p is not None for p in pred),
        "n_gold": sum(g is not None for g in gold),
    }


def span_metrics(gold: Sequence[Span], pred: Sequence[Span]) -> dict[str, Any]:
    """Exact-match rate and span P/R/F1 overall, and restricted to gold-target / gold-null."""
    if len(gold) != len(pred):
        raise ValueError(f"{len(gold)} gold spans but {len(pred)} predictions")
    g_all = [_bounds(s) for s in gold]
    p_all = [_bounds(s) for s in pred]

    def scored(g: Sequence[tuple[int, int] | None], p: Sequence[tuple[int, int] | None]):
        c = _span_group(g, p)
        precision, recall, f1 = _prf(c["tp"], c["n_pred"], c["n_gold"])
        return {
            "n": c["n"],
            "exact_match": _ratio(c["n_exact"], c["n"]) if c["n"] else None,
            "precision": precision if c["n"] else None,
            "recall": recall if c["n"] else None,
            "f1": f1 if c["n"] else None,
            "tp": c["tp"],
            "n_pred": c["n_pred"],
            "n_gold": c["n_gold"],
        }

    with_target = [(g, p) for g, p in zip(g_all, p_all, strict=True) if g is not None]
    null_preds = [p for g, p in zip(g_all, p_all, strict=True) if g is None]
    n_false = sum(p is not None for p in null_preds)

    result = scored(g_all, p_all)
    result["gold_target"] = scored([g for g, _ in with_target], [p for _, p in with_target])
    result["gold_null"] = {
        "n": len(null_preds),
        "null_accuracy": _ratio(len(null_preds) - n_false, len(null_preds)) if null_preds else None,
        "false_span_rate": _ratio(n_false, len(null_preds)) if null_preds else None,
        "n_false_span": n_false,
    }
    return result


def task_metrics(records: Sequence[Mapping[str, Any]], type_preds, span_preds) -> dict[str, Any]:
    return {
        "n": len(records),
        "type": type_metrics([r["type"] for r in records], type_preds),
        "target": span_metrics([r["target"] for r in records], span_preds),
    }


def _slice_value(record: Mapping[str, Any], name: str) -> str:
    if name in record:
        value = record[name]
    elif name == "accented":
        value = text_is_accented(record["text"])
    else:
        return "unknown"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def evaluate(
    records: Sequence[Mapping[str, Any]],
    type_preds: Sequence[str],
    span_preds: Sequence[Span],
    slices: Sequence[str] = ("accented", "source_batch"),
) -> dict[str, Any]:
    """Overall and per-slice metrics for split ``records`` and aligned predictions.

    ``records`` follow the split contract (``type``, ``target``, and the slice fields;
    ``accented`` is derived from ``text`` if absent). ``type_preds`` are type names and
    ``span_preds`` are ``{"start", "end", ...}`` dicts or ``None``. Result::

        {"n": N, "overall": {n, type, target},
         "slices": {"accented": {"true": {n, type, target}, ...}, "source_batch": {...}}}
    """
    if not (len(records) == len(type_preds) == len(span_preds)):
        raise ValueError(
            f"{len(records)} records, {len(type_preds)} type predictions, "
            f"{len(span_preds)} span predictions"
        )
    out: dict[str, Any] = {
        "n": len(records),
        "overall": task_metrics(records, type_preds, span_preds),
        "slices": {},
    }
    for name in slices:
        keys = [_slice_value(r, name) for r in records]
        counts = Counter(keys)
        out["slices"][name] = {
            value: task_metrics(
                [r for r, k in zip(records, keys, strict=True) if k == value],
                [t for t, k in zip(type_preds, keys, strict=True) if k == value],
                [s for s, k in zip(span_preds, keys, strict=True) if k == value],
            )
            for value in sorted(counts)
        }
    return out
