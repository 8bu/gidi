"""Score a 3-head model on a merged annotation-v2 records file: metrics JSON + per-record rows."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from gidi.evaluation.value_metrics import (
    Categories,
    evaluate_value,
    gold_token_count,
    surface_pattern,
)
from gidi.evaluation.value_predict import Logits, decode_predictions, prediction_rows
from gidi.modeling.crf import CRFTransitions
from gidi.modeling.value import ValuePrepared, prepare_value


def value_span_categories(*, required: bool = True) -> Categories | None:
    """Category tags of a note from ``gidi.annotation.value_span.propose_value``.

    Imported lazily: that module is the rule-based proposer; evaluation only reads the
    categories it assigns to a note (``explicit_unit``, ``bare_number``, ``slang``,
    ``multi_number``, ``unaccented``, ...). With ``required=False`` a missing module yields
    ``None`` (no category slices) instead of an ``ImportError``.
    """
    try:
        from gidi.annotation.value_span import propose_value
    except ImportError:
        if required:
            raise
        return None

    def categories(text: str) -> Sequence[str]:
        proposal = propose_value(text)
        found = (
            proposal.get("categories")
            if isinstance(proposal, dict)
            else getattr(proposal, "categories", None)
        )
        return list(found or [])

    return categories


def evaluate_records(
    records: list[dict[str, Any]],
    logits_fn: Callable[[ValuePrepared], Logits],
    tokenizer: Any,
    max_length: int,
    *,
    train_records: Sequence[dict[str, Any]] | None = None,
    categories: Categories | None = None,
    crf: CRFTransitions | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """``(metrics, per-record rows, predictions)`` for ``records``.

    ``logits_fn`` maps the encoded split to raw logits (``torch_logits`` or ``onnx_logits``
    bound to a model/session). Encoding is non-strict: a value that no token boundary can
    reproduce is counted under ``alignment`` instead of aborting the evaluation. ``crf``: the
    value head's CRF transitions (``value_crf``) for Viterbi value decoding.
    """
    data = prepare_value(tokenizer, records, max_length, strict=False)
    preds = decode_predictions(data, logits_fn(data), crf)
    offsets = data.encoded.offsets
    metrics = evaluate_value(
        records,
        preds,
        offsets=offsets,
        train_records=train_records,
        categories=categories,
    )
    venc = data.value_encoded
    metrics["alignment"] = {
        "truncated": data.encoded.n_truncated,
        "target_span_truncated": data.encoded.n_span_truncated,
        "value_span_truncated": venc.n_value_span_truncated,
        "value_boundary_mismatch": venc.n_value_boundary_mismatch,
    }

    seen: set[str] | None = None
    if train_records is not None:
        seen = {
            surface_pattern(t["value"]["text"])
            for t in train_records
            if t.get("value_status") == "complete" and t.get("value") is not None
        }
    extra = []
    for r, o in zip(records, offsets, strict=True):
        row: dict[str, Any] = {}
        if categories is not None:
            row["categories"] = list(categories(r["text"]))
        if r.get("value") is not None:
            pattern = surface_pattern(r["value"]["text"])
            row["gold_value_pattern"] = pattern
            if seen is not None:
                row["pattern_seen_in_train"] = pattern in seen
            row["gold_value_tokens"] = gold_token_count(r, o)
        extra.append(row)
    return metrics, prediction_rows(records, preds, extra), preds
