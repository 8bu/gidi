"""Linear-chain CRF Viterbi decoding. Pure numpy.

Parameters follow the ``pytorch-crf`` layout: ``start [K]``, ``end [K]`` and
``transitions [K, K]`` where ``transitions[i, j]`` scores tag ``i`` followed by tag ``j``. The
training side (``gidi.modeling.crf``) re-uses this module, so both decode with one
implementation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np


class CRFTransitions(NamedTuple):
    """Numpy copies of a CRF's transition scores, for decoding outside torch."""

    start: np.ndarray  # [K]
    end: np.ndarray  # [K]
    transitions: np.ndarray  # [K, K]


def viterbi(emissions: np.ndarray, crf: CRFTransitions) -> list[int]:
    """Highest-scoring tag sequence for ``emissions [T, K]``; ``[]`` when ``T == 0``.

    Scores are accumulated in float64. Ties go to the lowest tag index.
    """
    emissions = np.asarray(emissions, dtype=np.float64)
    if len(emissions) == 0:
        return []
    trans = crf.transitions.astype(np.float64)
    score = crf.start.astype(np.float64) + emissions[0]
    back: list[np.ndarray] = []
    for t in range(1, len(emissions)):
        total = score[:, None] + trans + emissions[t][None, :]  # [prev, cur]
        back.append(total.argmax(axis=0))
        score = total.max(axis=0)
    best = int((score + crf.end).argmax())
    path = [best]
    for pointers in reversed(back):
        best = int(pointers[best])
        path.append(best)
    return path[::-1]


def viterbi_masked(
    emissions: np.ndarray, real: Sequence[bool], crf: CRFTransitions, fill: int = 0
) -> list[int]:
    """Viterbi over the ``real`` positions only; every other position gets ``fill``."""
    idx = [i for i, r in enumerate(real) if r]
    tags = [fill] * len(real)
    if idx:
        for i, tag in zip(idx, viterbi(np.asarray(emissions)[idx], crf), strict=True):
            tags[i] = tag
    return tags
