"""Stratum verdicts of a scored review, and the gate that training builds pass through.

``scripts/score_value_review.py`` compares the human labels of a reduced review with the rule
proposals and records, per stratum, whether the auto-accepted remainder may be trusted
(``strata_status``) together with the sha256 of every input it read. Training builds call
``load_gate``: no score file, no entry for the dataset, or an input that changed since scoring
(a new human label, for instance) means the verdicts are stale, and the build must stop.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from gidi.annotation import risk_review as rr

SCORE_PATH = Path("experiments/value-span-v1/review-score.json")
SCORE_COMMAND = "uv run python scripts/score_value_review.py"

PASSED = "passed"
FAILED = "failed"
PENDING = "pending"

_HARD_VERDICT = {"ok": PASSED, "review_entire_stratum": FAILED, "pending": PENDING}
_AUDIT_VERDICT = {"accept": PASSED, "expand_all": FAILED, "pending": PENDING}


class GateError(Exception):
    """The score is absent, lacks the dataset, or is stale; the message tells what to run."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stratum_verdicts(
    escalation: rr.Escalation, auto_by_stratum: Mapping[str, Sequence[str]]
) -> dict[str, str]:
    """``passed`` / ``failed`` / ``pending`` for every stratum that has auto-accepted records.

    The clean stratum follows the audit; every other stratum follows its own sample. A stratum
    with a record the escalation asks a human to review has failed, whatever its own sample says.
    """
    verdicts = {s: _HARD_VERDICT[d] for s, d in escalation.hard_strata.items()}
    verdicts[rr.CLEAN_STRATUM] = _AUDIT_VERDICT[escalation.clean_audit]
    escalated = set(escalation.ids)
    for stratum, ids in auto_by_stratum.items():
        if any(i in escalated for i in ids):
            verdicts[stratum] = FAILED
        else:
            verdicts.setdefault(stratum, PENDING)
    return verdicts


def trusted_strata(verdicts: Mapping[str, str]) -> frozenset[str]:
    return frozenset(s for s, v in verdicts.items() if v == PASSED)


def load_gate(root: Path, dataset: str, score_path: Path = SCORE_PATH) -> frozenset[str]:
    """Strata whose rule labels may train, from a fresh score of ``dataset``.

    Raises ``GateError`` when the score file or the dataset entry is missing, or when any input the
    scorer read (queue, provenance, proposals, human labels) changed since.
    """
    path = root / score_path
    if not path.exists():
        raise GateError(f"{score_path} is missing: score the reviews first ({SCORE_COMMAND})")
    entry = json.loads(path.read_text(encoding="utf-8")).get("datasets", {}).get(dataset)
    if entry is None:
        raise GateError(
            f"{score_path} has no score for {dataset!r}: run `{SCORE_COMMAND} --dataset {dataset}`"
        )
    stale = [
        name
        for name, digest in entry["inputs"].items()
        if not (root / name).exists() or sha256_file(root / name) != digest
    ]
    if stale:
        raise GateError(
            f"{score_path} is stale for {dataset!r}: changed since scoring: {', '.join(stale)}; "
            f"re-run `{SCORE_COMMAND} --dataset {dataset}`"
        )
    return trusted_strata(entry["strata_status"])
