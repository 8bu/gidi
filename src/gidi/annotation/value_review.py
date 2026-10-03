"""Tiered review of the existing in-scope notes' value spans.

``scripts/build_value_queue.py`` proposes a value span for every in-scope note. Instead of asking
a human to read all the flagged notes plus a large audit sample, the records are grouped with the
same policy as ``targeted-value-01`` (``gidi.annotation.risk_review``):

* ``must_review``: every flagged note with an objective, semantic ambiguity (``must_reasons``);
* ``hard_sample``: a few records per pattern stratum of the remaining flagged notes
  (``pattern_stratum``), spread over accents and the v1 type;
* ``clean_audit``: ``AUDIT_SIZE`` records drawn over the clean, auto-accepted notes;
* ``auto_accept``: everything else. Clean notes carry a rule label; unsampled flagged notes keep
  their rule *proposal* with status ``pending-stratum-check``. Neither is ever a human label, and
  both are trusted only if the scorer (``scripts/score_value_review.py``) marks their stratum as
  passed.

Every draw is a pure function of ``(SEED, record id)``, so the plan is reproducible. Nothing here
writes a label: human labels come only from Quet.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any

from gidi.annotation import risk_review as rr
from gidi.annotation.value_span import RULE_VERSION, ValueProposal, fold

NAME = "existing-value-review"
SEED = "annotation-v2:existing-value-review"
AUDIT_SIZE = 15
# Population of a stratum -> sample: all if tiny, 2 if small, 3 otherwise.
SAMPLE_ALL_UP_TO = 2
SAMPLE_LARGE_FROM = 6
SAMPLE_SMALL = 2
SAMPLE_LARGE = 3
# A failed clean audit is expanded to the clean population in stages of this many records.
CLEAN_EXPANSION_STAGE = 50
# The queue the previous layout asked for (127 flagged notes + a 40-record audit sample).
ORIGINAL_AUDIT_SAMPLE = 40

STATUS_PENDING = "pending-stratum-check"
STATUS_QUEUED = "queued"
PROVENANCE_RULE = "rule"

# Must-review reasons, in display order. The proposer categories come first.
REASON_ORDER: tuple[str, ...] = (
    *rr.PROPOSER_MUST,
    "bare_number",
    "number_words",
    "split_share",
    "cu_collision",
    "numeric_ambiguity",
)

# Exclusion reasons of the extra number of a multi-number note, in stratum priority order.
MULTI_NUMBER_REASONS: tuple[str, ...] = (
    "period_or_ordinal",
    "quantity",
    "date_or_time",
    "attached_unit",
    "weekday",
    "percent",
)
# Display order of the sampled strata; unknown strata follow alphabetically.
STRATUM_ORDER: tuple[str, ...] = (
    *(f"multi_number_{r}" for r in MULTI_NUMBER_REASONS),
    "multi_number_other",
    "slang_cu",
    "slang_other",
    "flagged_other",
)

ESCALATION_RULE: dict[str, Any] = {
    "error": (
        "the human value label differs from the proposal (span, or present versus null) or the "
        "human annotation_status is not 'complete'"
    ),
    "sampled_stratum": (
        "any error in a sampled stratum -> that stratum is expanded to full review (the scorer "
        "writes value-escalation-queue.jsonl)"
    ),
    "clean_audit": {
        "max_errors_to_accept": rr.ESCALATION_RULE["clean_audit"]["max_errors_to_accept"],
        "accept": "0-1 errors in the audit sample -> the auto-accepted clean set stands",
        "expand": (
            ">=2 errors -> stop; the clean auto-accepted population is reviewed in stages of "
            f"{CLEAN_EXPANSION_STAGE} records (value-escalation-queue.jsonl)"
        ),
    },
    "must_review": "errors do not trigger escalation (expected); they are reported",
    "unsampled": (
        "unsampled flagged records keep their rule proposal (status pending-stratum-check); they "
        "train only when the scorer marks their stratum passed, and are never human labels"
    ),
}


# ---------------------------------------------------------------------------------------------
# Reasons and strata
# ---------------------------------------------------------------------------------------------


def _words(text: str) -> list[str]:
    return re.findall(r"[^\W\d_]+", fold(text))


def split_share(text: str) -> bool:
    """Is ``chia`` (split) a word of the note? The amount may be the total or a per-person share."""
    return "chia" in _words(text)


def cu_collision(text: str, proposal: ValueProposal) -> bool:
    """A slang ``củ`` amount in a note where ``cu`` (``cũ``/``củ``) occurs again outside it."""
    chosen = proposal.chosen
    if chosen is None or rr.slang_unit(chosen.text) != "slang_cu":
        return False
    folded = fold(text)
    return any(
        not chosen.start <= m.start() < chosen.end
        for m in re.finditer(r"(?<![^\W\d_])cu(?![^\W\d_])", folded)
    )


def must_reasons(text: str, proposal: ValueProposal) -> list[str]:
    """Must-review reasons of a flagged note (``REASON_ORDER``); clean notes have none."""
    if not proposal.needs_review:
        return []
    cats = set(proposal.categories)
    found = {c for c in rr.PROPOSER_MUST if c in cats}
    found |= {c for c in ("bare_number", "number_words") if c in cats}
    if split_share(text):
        found.add("split_share")
    if cu_collision(text, proposal):
        found.add("cu_collision")
    if any(x.reason == "bare_beside_money" for x in proposal.excluded):
        found.add("numeric_ambiguity")
    return [r for r in REASON_ORDER if r in found]


def pattern_stratum(proposal: ValueProposal) -> str | None:
    """Sampled stratum of a flagged note (a record counts once); ``None`` for a clean one.

    A slang unit of the value span wins; else a multi-number note is split by the proposer's
    exclusion reason of the extra number.
    """
    if not proposal.needs_review:
        return None
    unit = rr.slang_unit(proposal.chosen.text if proposal.chosen else None)
    if unit:
        return unit
    if "multi_number" in proposal.categories:
        reasons = {x.reason for x in proposal.excluded}
        for reason in MULTI_NUMBER_REASONS:
            if reason in reasons:
                return f"multi_number_{reason}"
        return "multi_number_other"
    return "slang_other" if "slang" in proposal.categories else "flagged_other"


def stratum_rank(stratum: str) -> tuple[int, str]:
    order = STRATUM_ORDER.index(stratum) if stratum in STRATUM_ORDER else len(STRATUM_ORDER)
    return order, stratum


def sample_size(population: int) -> int:
    """How many records of a stratum with ``population`` members a human reviews."""
    if population <= SAMPLE_ALL_UP_TO:
        return population
    return SAMPLE_LARGE if population >= SAMPLE_LARGE_FROM else SAMPLE_SMALL


# ---------------------------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Facts:
    """What the draws need to know about one record."""

    id: str
    reasons: tuple[str, ...]  # must-review reasons
    stratum: str | None  # sampled stratum of a non-must flagged record; None when clean
    type: str | None  # v1 transaction type
    accented: bool
    form: str  # pattern signature of the proposed span
    target_present: bool  # v1 counterparty target present
    split: str


def plan_review(
    facts: Sequence[Facts], audit_size: int = AUDIT_SIZE, seed: str = SEED
) -> dict[str, rr.Decision]:
    """Group every record; the result does not depend on the order of ``facts``."""
    decisions: dict[str, rr.Decision] = {}
    rest: list[Facts] = []
    for f in facts:
        if f.reasons:
            decisions[f.id] = rr.Decision(f.id, rr.MUST, f.reasons[0], f.reasons)
        else:
            rest.append(f)

    for stratum in sorted({f.stratum for f in rest if f.stratum}, key=stratum_rank):
        members = [f for f in rest if f.stratum == stratum]
        picked = rr.pick_diverse(
            members,
            lambda f: f.id,
            lambda f: (f"type:{f.type}", f"accent:{'accented' if f.accented else 'unaccented'}"),
            sample_size(len(members)),
            seed,
            f"stratum:{stratum}",
        )
        chosen = {f.id for f in picked}
        for f in members:
            decisions[f.id] = rr.Decision(f.id, rr.HARD if f.id in chosen else rr.AUTO, stratum)

    clean = [f for f in rest if f.stratum is None]
    picked = rr.pick_diverse(
        clean,
        lambda f: f.id,
        lambda f: (
            f"type:{f.type}",
            f"accent:{'accented' if f.accented else 'unaccented'}",
            f"form:{f.form}",
            f"target:{'present' if f.target_present else 'null'}",
            f"split:{f.split}",
        ),
        audit_size,
        seed,
        "audit",
    )
    chosen = {f.id for f in picked}
    for f in clean:
        group = rr.AUDIT if f.id in chosen else rr.AUTO
        decisions[f.id] = rr.Decision(f.id, group, rr.CLEAN_STRATUM)
    return decisions


def rule_label_row(proposal_row: dict[str, Any]) -> dict[str, Any]:
    """Rule-provenance value label of an unsampled flagged record, from its Quet proposal row.

    Only proposals that carry a single amount span qualify; the caller trusts the label only when
    the scorer marked the record's stratum as passed. Never a human label.
    """
    if (
        proposal_row.get("annotation_status") != "complete"
        or proposal_row.get("type") != "amount"
        or proposal_row.get("target") is None
    ):
        raise ValueError(f"{proposal_row.get('id')}: proposal is not a single amount span")
    return {
        "id": proposal_row["id"],
        "annotation_status": "complete",
        "type": "amount",
        "target": proposal_row["target"],
        "provenance": PROVENANCE_RULE,
        "confidence": proposal_row["confidence"],
        "rule_version": RULE_VERSION,
    }


def stage_clean_ids(
    ids: Collection[str],
    reviewed: Collection[str],
    seed: str = SEED,
    size: int = CLEAN_EXPANSION_STAGE,
) -> list[str]:
    """Next expansion stage: the first ``size`` unreviewed ids by seeded hash."""
    todo = [i for i in ids if i not in reviewed]
    todo.sort(key=lambda i: rr.hash_key(seed, "stage", i))
    return todo[:size]
