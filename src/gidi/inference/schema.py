"""Public result types of the runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class EmptyInputError(ValueError):
    """The text is empty or whitespace-only; there is nothing to classify."""


@dataclass(frozen=True)
class ValuePrediction:
    """The value-span head's result for one note (bundles with a value head only).

    ``span`` is ``[start, end)`` in code points of the *caller's* string and ``text`` is
    ``text[start:end]`` sliced from that string. ``confidence`` follows the target formula: the
    geometric mean of the per-token probability of the predicted tag over the span's tokens, or,
    with no span, the minimum ``P(O)`` over the note's real tokens. The value is a span only;
    no number is ever derived from it.
    """

    text: str | None
    span: tuple[int, int] | None
    confidence: float


@dataclass(frozen=True)
class Prediction:
    """One classified note.

    ``target_span`` is ``[start, end)`` in code points of the *caller's* string (``end``
    exclusive), so ``target == text[start:end]``. ``type_confidence`` is the softmax probability
    of the predicted type. ``target_confidence`` is the geometric mean of the per-token
    probability of the predicted tag over the span's tokens, or, when there is no span, the
    minimum ``P(O)`` over the note's real tokens (how sure the model is that nothing is a target).

    ``value`` is ``None`` for a bundle without a value head (``to_dict`` then has no value keys);
    otherwise it carries the value span, decoded exactly like the target span.
    """

    type: str
    type_confidence: float
    target: str | None
    target_span: tuple[int, int] | None
    target_confidence: float
    truncated: bool
    model_version: str
    value: ValuePrediction | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "type": self.type,
            "type_confidence": self.type_confidence,
            "target": self.target,
            "target_span": None if self.target_span is None else list(self.target_span),
            "target_confidence": self.target_confidence,
        }
        if self.value is not None:
            out["value_text"] = self.value.text
            out["value_span"] = None if self.value.span is None else list(self.value.span)
            out["value_confidence"] = self.value.confidence
        out["truncated"] = self.truncated
        out["model_version"] = self.model_version
        return out


@dataclass(frozen=True)
class RawOutput:
    """Model outputs and tokenization of one note, before decoding.

    ``offsets`` are per-token ``(start, end)`` code-point offsets into the caller's original
    string (``(0, 0)`` for ``<s>``/``</s>``). ``normalized_text`` is the NFC text the model
    saw, ``normalized_offsets`` the same offsets in that text, and ``special_tokens_mask`` marks
    ``<s>``/``</s>``. ``value_logits`` and ``value_tags`` are ``None`` for a bundle without a
    value head; ``value_tags`` is the CRF Viterbi path over the real tokens as indices into
    ``O, B-VALUE, I-VALUE`` (``<s>``/``</s>`` are ``O``).
    """

    type_logits: np.ndarray  # float32 [num_types]
    tag_logits: np.ndarray  # float32 [tokens, num_tags]
    input_ids: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]
    truncated: bool
    normalized_text: str
    normalized_offsets: tuple[tuple[int, int], ...]
    special_tokens_mask: tuple[int, ...]
    value_logits: np.ndarray | None = None  # float32 [tokens, 3]: O, B-VALUE, I-VALUE
    value_tags: tuple[int, ...] | None = None  # Viterbi tag per token, ``len(input_ids)``
