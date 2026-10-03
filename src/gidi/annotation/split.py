"""Deterministic, leakage-safe train/validation/test split of annotated records.

Records that are the same utterance in disguise are merged into *leakage groups*; a group is
never split across train/validation/test. Three rules merge records:

* templates: same text once the target span is masked (``cho Long mượn 1tr`` / ``cho Linh mượn
  200k`` -> ``cho § mượn N``) and amounts are masked; this also keeps exact duplicates together;
* mirrors: same bag of words (direction mirrors such as "Hùng trả nợ 500k" / "trả nợ Hùng 500k");
* near-duplicates: normalized similarity >= ``SIMILARITY_THRESHOLD``.

Groups are then distributed per stratum (majority type, majority accented flag) to hit the
requested ratios.
"""

from __future__ import annotations

import random
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from difflib import SequenceMatcher
from typing import Any

SPLITS: tuple[str, ...] = ("train", "validation", "test")
RATIOS: tuple[float, float, float] = (0.70, 0.15, 0.15)
SIMILARITY_THRESHOLD = 0.9
MAX_LENGTH_GAP = 6
STOPWORDS: frozenset[str] = frozenset(
    {"cho", "minh", "tao", "toi", "lai", "roi", "nha", "ok", "da", "dc", "duoc"}
)
MIN_GROUPS_FOR_COVERAGE = 3
TARGET_MASK = "§"

_AMOUNT = re.compile(r"\b\d[\d.,]*\s*(k|tr\d*|trieu|nghin|cu|d|%)?(?=\W|$)")


def normalize(text: str) -> str:
    """Lowercase, strip diacritics (đ -> d), mask amounts as ``N``, drop ``,``/``.``."""
    text = unicodedata.normalize("NFC", text).lower()
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    text = text.replace("đ", "d")
    text = _AMOUNT.sub(" N ", text)
    text = text.replace(",", " ").replace(".", " ")
    return " ".join(text.split())


def accented(text: str) -> bool:
    return any(ord(c) > 127 for c in text)


def template_key(record: dict[str, Any]) -> str:
    """Normalized text with the target span replaced by ``TARGET_MASK`` (when there is one).

    Target offsets are Unicode code points into the NFC text, as in the annotation labels.
    """
    text = unicodedata.normalize("NFC", record["text"])
    target = record.get("target")
    if target:
        start, end = target["start"], target["end"]
        if text[start:end] != target["text"]:
            raise ValueError(f"{record.get('id')}: target {target!r} does not match {text!r}")
        text = f"{text[:start]} {TARGET_MASK} {text[end:]}"
    return normalize(text)


def _bag_key(normalized: str) -> tuple[str, ...]:
    return tuple(sorted(t for t in normalized.split() if t not in STOPWORDS))


def leakage_groups(records: Sequence[dict[str, Any]]) -> list[list[str]]:
    """Group ids of records that share a target-masked template, mirror, or near-duplicate.

    Records may carry ``target`` (``{"text","start","end"}`` or ``None``) for template masking.

    Each group lists its ids sorted; groups are sorted by their first id.
    """
    ids = [str(r["id"]) for r in records]
    norms = [normalize(r["text"]) for r in records]
    parent = list(range(len(ids)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    first_by_template: dict[str, int] = {}
    first_by_bag: dict[tuple[str, ...], int] = {}
    for i, record in enumerate(records):
        template = template_key(record)
        if template in first_by_template:
            union(first_by_template[template], i)
        else:
            first_by_template[template] = i
        key = _bag_key(norms[i])
        if not key:
            continue
        if key in first_by_bag:
            union(first_by_bag[key], i)
        else:
            first_by_bag[key] = i

    for i in range(len(norms)):
        for j in range(i + 1, len(norms)):
            if find(i) == find(j) or abs(len(norms[i]) - len(norms[j])) > MAX_LENGTH_GAP:
                continue
            matcher = SequenceMatcher(None, norms[i], norms[j])
            if (
                matcher.real_quick_ratio() >= SIMILARITY_THRESHOLD
                and matcher.quick_ratio() >= SIMILARITY_THRESHOLD
                and matcher.ratio() >= SIMILARITY_THRESHOLD
            ):
                union(i, j)

    members: dict[int, list[str]] = {}
    for i, rid in enumerate(ids):
        members.setdefault(find(i), []).append(rid)
    return sorted((sorted(m) for m in members.values()), key=lambda m: m[0])


def _majority(values: Sequence[Any]) -> Any:
    counts = Counter(values)
    return min(counts, key=lambda v: (-counts[v], v))


def assign_splits(
    records: Sequence[dict[str, Any]],
    *,
    seed: str | int,
    ratios: tuple[float, float, float] = RATIOS,
    groups: list[list[str]] | None = None,
) -> dict[str, str]:
    """Assign every record id to a split, keeping each leakage group whole.

    Records carry ``id``, ``type`` and ``accented`` (plus ``text`` unless ``groups`` is given).
    """
    by_id = {str(r["id"]): r for r in records}
    if groups is None:
        groups = leakage_groups(records)
    groups = sorted(groups, key=lambda g: g[0])

    def stratum(group: list[str]) -> tuple[str, bool]:
        return (
            _majority([by_id[i]["type"] for i in group]),
            _majority([bool(by_id[i]["accented"]) for i in group]),
        )

    strata = [stratum(g) for g in groups]
    stratum_size: Counter[tuple[str, bool]] = Counter()
    for group, key in zip(groups, strata, strict=True):
        stratum_size[key] += len(group)

    order = list(range(len(groups)))
    random.Random(seed).shuffle(order)
    order.sort(key=lambda i: -len(groups[i]))  # stable: keeps the shuffled order within a size

    assigned: dict[tuple[str, bool], list[int]] = {k: [0, 0, 0] for k in stratum_size}
    group_split: dict[int, int] = {}
    for gi in order:
        key = strata[gi]
        deficits = [ratios[s] * stratum_size[key] - assigned[key][s] for s in range(3)]
        best = max(range(3), key=lambda s: (deficits[s], -s))
        group_split[gi] = best
        assigned[key][best] += len(groups[gi])

    _ensure_type_coverage(groups, group_split, by_id)
    return {rid: SPLITS[group_split[gi]] for gi, g in enumerate(groups) for rid in g}


def _ensure_type_coverage(
    groups: list[list[str]], group_split: dict[int, int], by_id: dict[str, dict[str, Any]]
) -> None:
    """Move whole groups so validation and test each hold every type with >= 3 groups."""
    group_types = [{by_id[i]["type"] for i in g} for g in groups]
    groups_per_type = Counter(t for types in group_types for t in types)
    for type_ in sorted(groups_per_type):
        if groups_per_type[type_] < MIN_GROUPS_FOR_COVERAGE:
            continue
        holders = [gi for gi, types in enumerate(group_types) if type_ in types]
        for target in (1, 2):
            if any(group_split[gi] == target for gi in holders):
                continue
            donors = (0, 3 - target)  # train first, then the other held-out split
            for donor in donors:
                candidates = [gi for gi in holders if group_split[gi] == donor]
                if len(candidates) < 2:
                    continue  # moving the last holder would uncover the donor
                move = min(candidates, key=lambda gi: (len(groups[gi]), gi))
                group_split[move] = target
                break
