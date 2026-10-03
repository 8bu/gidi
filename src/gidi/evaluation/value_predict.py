"""Per-record predictions of a 3-head (type, target, value) model: PyTorch or ONNX Runtime.

Both back ends produce raw logits per record (padding trimmed); ``decode_predictions`` turns them
into the same prediction dicts, so a PyTorch checkpoint and its ONNX export are scored by one
code path::

    {"type", "type_confidence", "target", "target_confidence", "value", "value_confidence"}

``target`` / ``value`` are ``{"text", "start", "end"}`` or ``None``. The target keeps the first
predicted span, the value the best one (``gidi.modeling.value_decode``); both confidences are
the v1 runtime's ``target_confidence`` computed on the respective head.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from gidi.inference.decode import DecodedSpan, decode_first_span, target_confidence, type_prediction
from gidi.modeling.crf import CRFTransitions
from gidi.modeling.preprocessing import TYPES
from gidi.modeling.value import GidiValueModel, ValuePrepared, value_crf
from gidi.modeling.value_decode import decode_value

OUTPUT_NAMES = ("type_logits", "tag_logits", "value_logits")
BATCH = 64

Logits = tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]


def _span_dict(text: str, span: DecodedSpan | None) -> dict[str, Any] | None:
    if span is None:
        return None
    return {"text": text[span.start : span.end], "start": span.start, "end": span.end}


@torch.no_grad()
def torch_logits(
    model: GidiValueModel, data: ValuePrepared, device: torch.device, batch_size: int = BATCH
) -> Logits:
    """Type logits ``[N, types]`` and per-record target/value logits ``[len_i, 3]``."""
    model.eval()
    enc = data.encoded
    types: list[np.ndarray] = []
    tags: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for i in range(0, len(data), batch_size):
        mask = enc.attention_mask[i : i + batch_size]
        width = int(mask.sum(dim=1).max())
        ids = enc.input_ids[i : i + batch_size, :width]
        type_l, tag_l, value_l = model(ids.to(device), mask[:, :width].to(device))
        types.append(type_l.float().cpu().numpy())
        tag_np, value_np = tag_l.float().cpu().numpy(), value_l.float().cpu().numpy()
        for j in range(ids.shape[0]):
            n = int(mask[j].sum())
            tags.append(tag_np[j, :n])
            values.append(value_np[j, :n])
    return np.concatenate(types), tags, values


def onnx_logits(session: Any, data: ValuePrepared, batch_size: int = BATCH) -> Logits:
    """Same as :func:`torch_logits` from an ``onnxruntime.InferenceSession`` (FP32 or INT8)."""
    enc = data.encoded
    types: list[np.ndarray] = []
    tags: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for i in range(0, len(data), batch_size):
        ids = enc.input_ids[i : i + batch_size]
        mask = enc.attention_mask[i : i + batch_size]
        width = int(mask.sum(dim=1).max())
        feed = {"input_ids": ids[:, :width].numpy(), "attention_mask": mask[:, :width].numpy()}
        type_l, tag_l, value_l = session.run(list(OUTPUT_NAMES), feed)
        types.append(type_l)
        for j in range(ids.shape[0]):
            n = int(mask[j].sum())
            tags.append(tag_l[j, :n])
            values.append(value_l[j, :n])
    return np.concatenate(types), tags, values


def decode_predictions(
    data: ValuePrepared, logits: Logits, crf: CRFTransitions | None = None
) -> list[dict[str, Any]]:
    """Decode logits of ``data``'s records (record order) into prediction dicts.

    ``crf``: the value head's CRF transitions (``value_crf``); value tags are then Viterbi-decoded.
    """
    type_logits, tag_logits, value_logits = logits
    preds: list[dict[str, Any]] = []
    for i, record in enumerate(data.records):
        text = record["text"]
        n = len(tag_logits[i])
        offsets = data.encoded.offsets[i][:n]
        real = [e > s for s, e in offsets]
        type_index, type_conf = type_prediction(type_logits[i])
        tag_ids = [int(k) for k in tag_logits[i].argmax(-1)]
        target = decode_first_span(offsets, tag_ids, text)
        value, value_conf = decode_value(value_logits[i], offsets, text, real, crf)
        preds.append(
            {
                "type": TYPES[type_index],
                "type_confidence": type_conf,
                "target": _span_dict(text, target),
                "target_confidence": target_confidence(tag_logits[i], tag_ids, target, real),
                "value": _span_dict(text, value),
                "value_confidence": value_conf,
            }
        )
    return preds


def predict_torch(
    model: GidiValueModel, data: ValuePrepared, device: torch.device | None = None
) -> list[dict[str, Any]]:
    logits = torch_logits(model, data, device or torch.device("cpu"))
    return decode_predictions(data, logits, value_crf(model))


def predict_onnx(session: Any, data: ValuePrepared) -> list[dict[str, Any]]:
    return decode_predictions(data, onnx_logits(session, data))


def prediction_rows(
    records: Sequence[dict[str, Any]],
    preds: Sequence[dict[str, Any]],
    extra: Sequence[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Inspection rows (gold vs prediction, per-head correctness) for a JSONL dump."""
    rows = []
    for k, (r, p) in enumerate(zip(records, preds, strict=True)):

        def key(span: dict[str, Any] | None) -> tuple[int, int] | None:
            return None if span is None else (span["start"], span["end"])

        complete = r.get("value_status") == "complete"
        type_ok = p["type"] == r["type"]
        target_ok = key(p["target"]) == key(r["target"])
        value_ok = key(p["value"]) == key(r["value"]) if complete else None
        rows.append(
            {
                "id": r["id"],
                "text": r["text"],
                "value_status": r.get("value_status"),
                "value_provenance": r.get("value_provenance"),
                "gold_type": r["type"],
                "pred_type": p["type"],
                "gold_target": r["target"],
                "pred_target": p["target"],
                "gold_value": r.get("value"),
                "pred_value": p["value"],
                "value_confidence": p["value_confidence"],
                "type_ok": type_ok,
                "target_ok": target_ok,
                "value_ok": value_ok,
                "full_joint_ok": (type_ok and target_ok and value_ok) if complete else None,
                **(extra[k] if extra is not None else {}),
            }
        )
    return rows
