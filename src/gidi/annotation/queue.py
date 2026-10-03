"""Deterministic annotation queue: a fixed, seed-ordered sample of approved corpus records."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any


def queue_key(seed: str, record_id: str) -> str:
    return hashlib.sha256(f"{seed}:{record_id}".encode()).hexdigest()


def build_queue(
    records: Iterable[dict[str, Any]],
    *,
    size: int,
    seed: str,
    review_status: str,
    corpus: str,
) -> list[dict[str, Any]]:
    """Pick ``size`` records whose Quet status is ``review_status``, ordered by ``queue_key``.

    The order depends only on the seed and ids, not on file order, so re-running on the same
    corpus reproduces the queue exactly. Raises ``ValueError`` if too few records qualify.
    """
    eligible = [r for r in records if (r.get("quet") or {}).get("status") == review_status]
    if len(eligible) < size:
        raise ValueError(f"only {len(eligible)} {review_status!r} record(s); {size} requested")
    eligible.sort(key=lambda r: queue_key(seed, r["id"]))
    return [
        {"id": r["id"], "text": r["text"], "corpus": corpus, "position": position}
        for position, r in enumerate(eligible[:size], start=1)
    ]
