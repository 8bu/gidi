"""Metrics for the annotation-v2 value span, next to the v1 type/target metrics.

Everything returned is plain JSON-able Python. Conventions (extending ``metrics.py``):

- Value metrics are computed on ``value_status == "complete"`` records only; ``uncertain``
  records are counted in ``n_value_masked`` and excluded (they are masked in training too).
  Type and target metrics use every record, as in v1.
- A value span is compared as character offsets. ``exact_match`` counts null == null (a
  ``no_amount`` note predicted without a value) as exact; ``span`` P/R/F1 is the v1 span metric
  (a prediction is a true positive iff its offsets equal the gold's); ``present_accuracy`` is
  the rate of agreeing "has an amount / has none" decisions.
- ``token`` / ``char`` are *overlap* P/R/F1 (micro over notes where either side has a span):
  tokens of the encoder (the tokens that carry the alignment, i.e. B/I labels) or characters.
  ``token`` needs the per-record token offsets and is ``None`` without them.
- ``full_joint``: type correct AND target exact AND value exact (value-complete records).
- Slices (value metrics + ``full_joint`` per slice, every slice carries its ``n``):
  ``category`` (tags from ``gidi.annotation.value_span``, a note can be in several),
  ``surface_pattern`` (gold value shape seen / unseen in the training records, digits -> ``N``),
  ``amount_length`` (``multi_token``: more than one encoder token or a space; BPE splits
  ``1tr5`` into several tokens), ``amount_space`` (the amount contains whitespace, e.g.
  ``5 xị``), plus ``accented`` and ``source_batch`` as in v1.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from gidi.evaluation.metrics import (
    Span,
    _bounds,
    _prf,
    _ratio,
    _slice_value,
    span_metrics,
    type_metrics,
)
from gidi.modeling.preprocessing import trim_span

Categories = Callable[[str], Sequence[str]]
Offsets = Sequence[Sequence[tuple[int, int]]]

_DIGITS = re.compile(r"\d+")


def surface_pattern(text: str) -> str:
    """Shape of an amount: lowercased, every digit run -> ``N``, whitespace collapsed.

    ``5 xị`` -> ``n xị``, ``1tr5`` -> ``ntrn``, ``1.500.000`` -> ``n.n.n``, ``50K`` -> ``nk``.
    """
    return " ".join(_DIGITS.sub("n", text.lower()).split())


def token_indices(text: str, offsets: Sequence[tuple[int, int]], span: Span) -> frozenset[int]:
    """Positions of the tokens a span covers: the alignment rule of ``encode``."""
    if span is None:
        return frozenset()
    start, end = int(span["start"]), int(span["end"])
    inside = set()
    for i, (s, e) in enumerate(offsets):
        s, e = trim_span(text, int(s), int(e))
        if s < e and s < end and e > start:
            inside.add(i)
    return frozenset(inside)


def _overlap(tp: int, n_pred: int, n_gold: int) -> dict[str, Any]:
    precision, recall, f1 = _prf(tp, n_pred, n_gold)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "n_pred": n_pred,
        "n_gold": n_gold,
    }


def overlap_metrics(
    gold: Sequence[frozenset | set], pred: Sequence[frozenset | set]
) -> dict[str, Any] | None:
    """Micro overlap P/R/F1 of two aligned sequences of sets; ``None`` if both are all empty."""
    n_gold = sum(len(g) for g in gold)
    n_pred = sum(len(p) for p in pred)
    if n_gold == 0 and n_pred == 0:
        return None
    tp = sum(len(g & p) for g, p in zip(gold, pred, strict=True))
    return _overlap(tp, n_pred, n_gold)


def _char_set(span: Span) -> frozenset[int]:
    return frozenset() if span is None else frozenset(range(int(span["start"]), int(span["end"])))


def span_overlap_metrics(
    records: Sequence[Mapping[str, Any]],
    gold: Sequence[Span],
    pred: Sequence[Span],
    offsets: Offsets | None,
) -> dict[str, Any]:
    """Token-level (needs ``offsets``) and character-level overlap P/R/F1 of two span lists."""
    char = overlap_metrics([_char_set(g) for g in gold], [_char_set(p) for p in pred])
    token = None
    if offsets is not None:
        rows = list(zip(records, offsets, strict=True))
        token = overlap_metrics(
            [token_indices(r["text"], o, g) for (r, o), g in zip(rows, gold, strict=True)],
            [token_indices(r["text"], o, p) for (r, o), p in zip(rows, pred, strict=True)],
        )
    return {"token": token, "char": char}


def _outcomes(gold: Sequence[Span], pred: Sequence[Span]) -> dict[str, int]:
    counts = Counter()
    for g, p in zip(gold, pred, strict=True):
        gb, pb = _bounds(g), _bounds(p)
        if gb == pb:
            counts["correct_null" if gb is None else "exact"] += 1
        elif gb is None:
            counts["false_value"] += 1
        elif pb is None:
            counts["missed_value"] += 1
        elif min(gb[1], pb[1]) > max(gb[0], pb[0]):
            counts["partial_overlap"] += 1
        else:
            counts["disjoint"] += 1
    return {
        k: counts[k]
        for k in (
            "exact",
            "correct_null",
            "false_value",
            "missed_value",
            "partial_overlap",
            "disjoint",
        )
    }


def value_span_metrics(
    records: Sequence[Mapping[str, Any]],
    pred_values: Sequence[Span],
    offsets: Offsets | None = None,
) -> dict[str, Any]:
    """Value metrics of ``records`` (already restricted to complete ones) vs ``pred_values``."""
    gold = [r["value"] for r in records]
    n = len(records)
    span = span_metrics(gold, pred_values)
    present_ok = sum((g is None) == (p is None) for g, p in zip(gold, pred_values, strict=True))
    return {
        "n": n,
        "exact_match": span["exact_match"],
        "present_accuracy": _ratio(present_ok, n) if n else None,
        "span": span,
        **span_overlap_metrics(records, gold, pred_values, offsets),
        "outcomes": _outcomes(gold, pred_values),
        "n_gold_value": sum(g is not None for g in gold),
        "n_gold_null": sum(g is None for g in gold),
    }


def _joint_flags(
    records: Sequence[Mapping[str, Any]], preds: Sequence[Mapping[str, Any]]
) -> list[bool]:
    return [
        p["type"] == r["type"]
        and _bounds(p["target"]) == _bounds(r["target"])
        and _bounds(p["value"]) == _bounds(r["value"])
        for r, p in zip(records, preds, strict=True)
    ]


def _block(
    records: Sequence[Mapping[str, Any]],
    preds: Sequence[Mapping[str, Any]],
    offsets: Offsets | None,
) -> dict[str, Any]:
    flags = _joint_flags(records, preds)
    return {
        "n": len(records),
        "value": value_span_metrics(records, [p["value"] for p in preds], offsets),
        "full_joint": _ratio(sum(flags), len(flags)) if flags else None,
    }


def gold_token_count(record: Mapping[str, Any], offsets: Sequence[tuple[int, int]]) -> int:
    return len(token_indices(record["text"], offsets, record["value"]))


def evaluate_value(
    records: Sequence[Mapping[str, Any]],
    preds: Sequence[Mapping[str, Any]],
    *,
    offsets: Offsets | None = None,
    train_records: Sequence[Mapping[str, Any]] | None = None,
    categories: Categories | None = None,
    slices: Sequence[str] = ("accented", "source_batch"),
) -> dict[str, Any]:
    """Overall and sliced metrics of aligned ``preds`` (``{"type", "target", "value"}``).

    ``offsets[i]`` are the token offsets of record ``i`` (``Encoded.offsets``); they enable the
    token-level overlap metrics and the ``multi_token`` slice. ``train_records`` enable the
    ``surface_pattern`` slice (``seen`` = the same digit-masked shape occurs among the
    ``complete`` training values). ``categories(text)`` enables the ``category`` slices.
    """
    if len(records) != len(preds):
        raise ValueError(f"{len(records)} records but {len(preds)} predictions")
    if offsets is not None and len(offsets) != len(records):
        raise ValueError(f"{len(records)} records but {len(offsets)} offset rows")
    idx = list(range(len(records)))
    complete = [i for i in idx if records[i].get("value_status") == "complete"]

    def pick(items, keep):
        return [items[i] for i in keep]

    task_types = type_metrics([r["type"] for r in records], [p["type"] for p in preds])
    target = span_metrics([r["target"] for r in records], [p["target"] for p in preds])
    target |= span_overlap_metrics(
        records, [r["target"] for r in records], [p["target"] for p in preds], offsets
    )
    type_target = [
        p["type"] == r["type"] and _bounds(p["target"]) == _bounds(r["target"])
        for r, p in zip(records, preds, strict=True)
    ]
    c_records, c_preds = pick(records, complete), pick(preds, complete)
    c_offsets = pick(offsets, complete) if offsets is not None else None
    joint = _joint_flags(c_records, c_preds)

    out: dict[str, Any] = {
        "n": len(records),
        "n_value_complete": len(complete),
        "n_value_masked": len(records) - len(complete),
        "overall": {
            "type": task_types,
            "target": target,
            "type_target_joint": _ratio(sum(type_target), len(type_target)) if records else None,
        },
        "value": value_span_metrics(c_records, [p["value"] for p in c_preds], c_offsets),
        "full_joint": {
            "n": len(joint),
            "accuracy": _ratio(sum(joint), len(joint)) if joint else None,
        },
        "slices": {},
    }

    def add_slice(name: str, keys: Sequence[str | Sequence[str] | None]) -> None:
        groups: dict[str, list[int]] = {}
        for k, key in enumerate(keys):
            if key is None:
                continue
            for label in [key] if isinstance(key, str) else key:
                groups.setdefault(label, []).append(k)
        out["slices"][name] = {
            label: _block(
                [c_records[k] for k in ks],
                [c_preds[k] for k in ks],
                [c_offsets[k] for k in ks] if c_offsets is not None else None,
            )
            for label, ks in sorted(groups.items())
        }

    if categories is not None:
        add_slice("category", [list(categories(r["text"])) or ["none"] for r in c_records])
    if train_records is not None:
        seen = {
            surface_pattern(t["value"]["text"])
            for t in train_records
            if t.get("value_status") == "complete" and t.get("value") is not None
        }
        add_slice(
            "surface_pattern",
            [
                "no_gold_value"
                if r["value"] is None
                else ("seen" if surface_pattern(r["value"]["text"]) in seen else "unseen")
                for r in c_records
            ],
        )
    if c_offsets is not None:
        add_slice(
            "amount_length",
            [
                "no_gold_value"
                if r["value"] is None
                else (
                    "multi_token"
                    if gold_token_count(r, o) > 1 or " " in r["value"]["text"]
                    else "single_token"
                )
                for r, o in zip(c_records, c_offsets, strict=True)
            ],
        )
    add_slice(
        "amount_space",
        [
            "no_gold_value"
            if r["value"] is None
            else ("space" if any(c.isspace() for c in r["value"]["text"]) else "no_space")
            for r in c_records
        ],
    )
    for name in slices:
        add_slice(name, [_slice_value(r, name) for r in c_records])
    return out
