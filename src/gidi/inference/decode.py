"""BIO decoding and confidences. Pure numpy/stdlib.

The span rule is the one of ``gidi.modeling.preprocessing.spans_from_tags`` (a test compares
the two): the first span starts at a ``B`` (or an ``I`` after a non-span token), consecutive
``I`` tokens extend it, a later ``B`` or an ``O`` ends it; bounds are the min start / max end of
the member tokens after trimming whitespace; tokens with empty trimmed offsets are skipped.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from gidi.inference.crf import CRFTransitions, viterbi_masked

TAG_O, TAG_B, TAG_I = 0, 1, 2


@dataclass(frozen=True)
class DecodedSpan:
    start: int
    end: int
    members: tuple[int, ...]  # token indices that make up the span


def trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Shrink ``[start, end)`` so it neither starts nor ends on whitespace (may become empty)."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def decode_first_span(
    offsets: Sequence[tuple[int, int]], tag_ids: Sequence[int], text: str
) -> DecodedSpan | None:
    start = end = None
    members: list[int] = []
    for i, ((s, e), tag) in enumerate(zip(offsets, tag_ids, strict=False)):
        s, e = trim_span(text, int(s), int(e))
        if s >= e:
            continue
        tag = int(tag)
        if tag == TAG_B:
            if start is not None:
                break
            start, end = s, e
            members.append(i)
        elif tag == TAG_I:
            if start is None:
                start, end = s, e
            else:
                start, end = min(start, s), max(end, e)
            members.append(i)
        elif start is not None:
            break
    if start is None:
        return None
    return DecodedSpan(start, end, tuple(members))


def log_softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise log-softmax in float64 (last axis)."""
    x = np.asarray(logits, dtype=np.float64)
    shifted = x - x.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def type_prediction(type_logits: np.ndarray) -> tuple[int, float]:
    """``(argmax class, softmax probability of it)``."""
    log_p = log_softmax(type_logits)
    index = int(np.argmax(type_logits))
    return index, float(math.exp(log_p[index]))


def target_confidence(
    tag_logits: np.ndarray, tag_ids: Sequence[int], span: DecodedSpan | None, real: Sequence[bool]
) -> float:
    """Geometric mean of P(predicted tag) over the span's tokens.

    Without a span: the minimum P(O) over the real (non-special) tokens, i.e. the model's
    weakest "this token is not a target" belief; ``1.0`` if the note has no real token.
    """
    log_p = log_softmax(tag_logits)
    if span is not None:
        picked = [log_p[i, int(tag_ids[i])] for i in span.members]
        return float(math.exp(sum(picked) / len(picked)))
    o_log_p = [log_p[i, TAG_O] for i, is_real in enumerate(real) if is_real]
    return float(math.exp(min(o_log_p))) if o_log_p else 1.0


def decode_all_spans(
    offsets: Sequence[tuple[int, int]], tag_ids: Sequence[int], text: str
) -> list[DecodedSpan]:
    """Every BIO span in token order, by the same rule as ``decode_first_span``.

    The first span of the result equals ``decode_first_span``'s. A ``B`` starts a new span, an
    ``I`` extends the current span (or starts one after an ``O``), an ``O`` ends it.
    """
    spans: list[DecodedSpan] = []
    start = end = None
    members: list[int] = []

    def flush() -> None:
        nonlocal start, end, members
        if start is not None:
            spans.append(DecodedSpan(start, end, tuple(members)))
        start = end = None
        members = []

    for i, ((s, e), tag) in enumerate(zip(offsets, tag_ids, strict=False)):
        s, e = trim_span(text, int(s), int(e))
        if s >= e:
            continue
        tag = int(tag)
        if tag == TAG_B:
            flush()
            start, end, members = s, e, [i]
        elif tag == TAG_I:
            if start is None:
                start, end, members = s, e, [i]
            else:
                start, end = min(start, s), max(end, e)
                members.append(i)
        else:
            flush()
    flush()
    return spans


def best_span_from_tags(
    logits: np.ndarray,
    tag_ids: Sequence[int],
    offsets: Sequence[tuple[int, int]],
    text: str,
    real: Sequence[bool],
) -> tuple[DecodedSpan | None, float]:
    """The most confident BIO span of ``tag_ids`` and its confidence.

    A note can have several amounts (``ăn 2 tô phở 70``), so unlike the target head, which keeps
    the first span, the value head keeps the span with the highest ``target_confidence`` (ties go
    to the earlier span). With no span the confidence is the minimum ``P(O)`` over the real
    tokens, as for the target head. ``logits`` is ``[tokens, 3]``.
    """
    spans = decode_all_spans(offsets, tag_ids, text)
    if not spans:
        return None, target_confidence(logits, tag_ids, None, real)
    scored = [(target_confidence(logits, tag_ids, span, real), span) for span in spans]
    best = max(range(len(scored)), key=lambda k: (scored[k][0], -k))
    confidence, span = scored[best]
    return span, confidence


def decode_value_crf(
    value_logits: np.ndarray,
    offsets: Sequence[tuple[int, int]],
    text: str,
    real: Sequence[bool],
    crf: CRFTransitions,
) -> tuple[DecodedSpan | None, float]:
    """Value-head decoding with a CRF: the best span and its confidence for one note.

    The tags are the Viterbi path over the ``real`` (non-special) positions, every other
    position is ``O``; spans and confidence follow ``best_span_from_tags``. This is the single
    CRF value-decoding path of the runtime and of the training-side metrics.
    """
    tag_ids = viterbi_masked(value_logits, real, crf, fill=TAG_O)
    return best_span_from_tags(value_logits, tag_ids, offsets, text, real)
