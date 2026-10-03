"""Decode the value BIO head: every predicted span, the best one, and its confidence.

The span rule is the one of the target head (``gidi.inference.decode.decode_first_span``): a span
starts at a ``B`` (or an ``I`` after a non-span token), consecutive ``I`` tokens extend it, a later
``B`` or an ``O`` ends it; bounds are the min start / max end of the member tokens after trimming
whitespace; tokens with empty trimmed offsets are skipped. The target head keeps the *first*
span. A note can legitimately produce several value spans (``ăn 2 tô phở 70``), so the value
head keeps the span with the highest confidence instead: the geometric mean of P(predicted tag)
over its tokens (ties go to the earlier span). ``value_confidence`` is computed exactly like
``target_confidence`` of the v1 runtime: that span's mean, or, when no span is predicted, the
minimum P(O) over the real tokens.

Pure numpy/stdlib: every primitive is the runtime's (``gidi.inference.decode``).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from gidi.inference.crf import CRFTransitions
from gidi.inference.decode import (
    DecodedSpan,
    best_span_from_tags,
    decode_all_spans,
    decode_value_crf,
)

VALUE_TAG_O, VALUE_TAG_B, VALUE_TAG_I = 0, 1, 2

__all__ = ["VALUE_TAG_B", "VALUE_TAG_I", "VALUE_TAG_O", "decode_all_spans", "decode_value"]


def decode_value(
    value_logits: np.ndarray,
    offsets: Sequence[tuple[int, int]],
    text: str,
    real: Sequence[bool],
    crf: CRFTransitions | None = None,
) -> tuple[DecodedSpan | None, float]:
    """Best value span (or ``None``) and ``value_confidence`` for one note.

    ``value_logits`` is ``[seq, 3]`` (order O, B-VALUE, I-VALUE), ``offsets`` one raw
    ``(start, end)`` per position and ``real`` marks the non-special, non-padding positions.
    With ``crf`` (an ``mlp-crf`` head) the tags are the Viterbi path over the ``real`` positions
    (others ``O``) instead of the per-token argmax; confidences still come from ``value_logits``.
    The CRF path is the runtime's ``decode_value_crf``.
    """
    if crf is not None:
        return decode_value_crf(value_logits, offsets, text, list(real), crf)
    tag_ids = [int(k) for k in np.asarray(value_logits).argmax(-1)]
    return best_span_from_tags(value_logits, tag_ids, offsets, text, real)
