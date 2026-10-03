"""Hard guard on what may enter the distillation pipeline (final teacher, targets, students).

Allowed: the targeted-v2 training data (frozen annotation-v1 train + targeted-02) and the frozen
validation split. Refused, loudly:

* the frozen test split (by id) and every diagnostic probe record (``probe-`` ids);
* targeted-03 / training-v3, an experiment artifact rejected as teacher data (by id prefix and
  by ``source_batch``).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from gidi.corpus.jsonl import read_jsonl

FROZEN_TEST = Path("datasets/annotation-v1/splits/test.jsonl")
FORBIDDEN_ID_PREFIXES = ("probe-", "targeted-annotation-v1-03-")
FORBIDDEN_SOURCE_BATCHES = frozenset({"targeted-annotation-v1-03"})


def forbidden_reason(record: dict[str, Any], test_ids: set[str]) -> str | None:
    """Why ``record`` may not be trained on in distillation-v1, or ``None`` if it may."""
    rid = str(record.get("id", ""))
    if rid in test_ids:
        return "frozen test record"
    for prefix in FORBIDDEN_ID_PREFIXES:
        if rid.startswith(prefix):
            return f"id prefix {prefix!r}"
    batch = record.get("source_batch") or (record.get("provenance") or {}).get("source_batch")
    if batch in FORBIDDEN_SOURCE_BATCHES:
        return f"source_batch {batch!r}"
    return None


def assert_distillation_trainable(
    records: Iterable[dict[str, Any]], root: Path = Path(".")
) -> None:
    """Raise ``ValueError`` listing every record that must not enter distillation training."""
    test_ids = {r["id"] for r in read_jsonl(root / FROZEN_TEST)}
    bad = [
        f"{r.get('id')!r}: {reason}"
        for r in records
        if (reason := forbidden_reason(r, test_ids)) is not None
    ]
    if bad:
        raise ValueError("records not allowed in distillation training:\n  " + "\n  ".join(bad))
