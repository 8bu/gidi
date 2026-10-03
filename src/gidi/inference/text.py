"""NFC normalization that remembers where every normalized character came from.

The model was trained on NFC text with offsets in NFC code points. A caller may hand over
decomposed text (``n`` + ``o`` + U+031B + U+0323 for ``nợ``) and expects spans back in *their*
string, so normalization keeps a map from NFC positions to original positions.

The original string is cut into groups that normalize independently of their neighbours. A group
that NFC leaves unchanged keeps a per-character map. A group that NFC rewrites (composition or
reordering) is atomic: a span boundary inside it is rounded outward to the group's edges, so the
mapped span always covers whole user-visible characters.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class NormalizedText:
    """``text`` is NFC of the original. The maps have length ``len(text) + 1``.

    ``start_map[i]`` is the original offset for a span *starting* at NFC offset ``i`` and
    ``end_map[i]`` the one for a span *ending* (exclusive) at ``i``. Both are ``None`` when the
    original was already NFC (identity).
    """

    text: str
    original_length: int
    start_map: tuple[int, ...] | None = None
    end_map: tuple[int, ...] | None = None

    def start_to_original(self, index: int) -> int:
        return index if self.start_map is None else self.start_map[index]

    def end_to_original(self, index: int) -> int:
        return index if self.end_map is None else self.end_map[index]

    def span_to_original(self, start: int, end: int) -> tuple[int, int]:
        return self.start_to_original(start), self.end_to_original(end)


def _groups(text: str) -> list[tuple[int, int]]:
    """Cut ``text`` into ``[start, end)`` groups whose NFC forms concatenate to ``NFC(text)``."""
    # Elementary clusters: a starter (combining class 0) plus the non-starters following it.
    # Reordering only happens inside a run of non-starters, so a cluster never needs its
    # neighbours for reordering, only for composition (handled by merging below).
    cuts = [i for i, ch in enumerate(text) if i == 0 or unicodedata.combining(ch) == 0]
    cuts.append(len(text))
    groups: list[tuple[int, int]] = []
    group_start, group_nfc = 0, ""
    for a, b in zip(cuts, cuts[1:], strict=False):
        cluster = text[a:b]
        cluster_nfc = unicodedata.normalize("NFC", cluster)
        if a == 0:
            group_nfc = cluster_nfc
            continue
        joined = unicodedata.normalize("NFC", text[group_start:b])
        if joined == group_nfc + cluster_nfc:
            groups.append((group_start, a))
            group_start, group_nfc = a, cluster_nfc
        else:
            group_nfc = joined
    groups.append((group_start, len(text)))
    return groups


def normalize_nfc(text: str) -> NormalizedText:
    """NFC-normalize ``text`` and build the NFC -> original offset maps."""
    if unicodedata.is_normalized("NFC", text):
        return NormalizedText(text, len(text))

    pieces: list[str] = []
    start_map: list[int] = []
    end_map: list[int] = [0]
    for a, b in _groups(text):
        group = text[a:b]
        group_nfc = unicodedata.normalize("NFC", group)
        pieces.append(group_nfc)
        if group_nfc == group:
            for j in range(a, b):
                start_map.append(j)
                end_map.append(j + 1)
        else:
            start_map.extend([a] * len(group_nfc))
            end_map.extend([b] * len(group_nfc))
    start_map.append(len(text))
    nfc = "".join(pieces)

    if nfc != unicodedata.normalize("NFC", text):
        # The grouping assumption failed; map every position to the whole string rather than
        # return wrong offsets.
        nfc = unicodedata.normalize("NFC", text)
        start_map = [0] * len(nfc) + [len(text)]
        end_map = [0] + [len(text)] * len(nfc)
    return NormalizedText(nfc, len(text), tuple(start_map), tuple(end_map))
