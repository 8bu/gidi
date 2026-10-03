"""Risk-based review of a synthetic batch: grouping, auto-accept rules, scoring and escalation.

A batch whose notes carry generation intent and advisory proposals is split into four groups
instead of being reviewed in full:

* ``must_review``: every record with an objective risk signal (``assess_record``);
* ``hard_sample``: ``HARD_SAMPLE_PER_STRATUM`` records per hard-pattern stratum (``multi_number``,
  ``bare_number`` and each slang unit), from the records that are not must-review;
* ``clean_audit``: ``CLEAN_AUDIT_SIZE`` records stratified over the remaining clean records;
* ``auto_accept``: everything else (labelled ``synthetic-auto``, never ``human``).

Every draw is a pure function of ``(seed, record id)``, so the plan is reproducible. After a human
labelled the queue, ``record_errors`` and ``decide_escalation`` apply the predeclared rule
(``ESCALATION_RULE``) that decides whether the auto-accepted remainder may stand.

Nothing here writes a label: human labels come only from Quet.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from statistics import median
from typing import Any, TypeVar

from gidi.annotation.leakage import NEAR_DUP_THRESHOLD, nearest, reference
from gidi.annotation.schema import AnnotationConfig, validate_annotation
from gidi.annotation.value_span import (
    RULE_LABEL_EXTRA,
    RULE_VERSION,
    ValueConfig,
    ValueProposal,
    fold,
    to_quet_proposal,
    validate_quet_proposal,
    validate_span,
    validate_value_label,
)

NAME = "targeted-value-01"
SEED = "targeted-value-01:review"

MUST = "must_review"
HARD = "hard_sample"
AUDIT = "clean_audit"
AUTO = "auto_accept"
GROUPS: tuple[str, ...] = (MUST, HARD, AUDIT, AUTO)
QUEUED_GROUPS: tuple[str, ...] = (MUST, HARD, AUDIT)

PROVENANCE_AUTO = "synthetic-auto"
PROVENANCE_QUEUED = "queued-for-human"

HARD_SAMPLE_PER_STRATUM = 3
CLEAN_AUDIT_SIZE = 20
CLEAN_STRATUM = "clean"
# A record counts once, in the first stratum it matches.
HARD_STRATA: tuple[str, ...] = (
    "multi_number",
    "bare_number",
    "slang_xi",
    "slang_chai",
    "slang_lit",
    "slang_cu",
    "slang_ty",
    "slang_other",
)
_SLANG_WORDS: tuple[tuple[str, frozenset[str]], ...] = (
    ("slang_xi", frozenset({"xi"})),
    ("slang_chai", frozenset({"chai"})),
    ("slang_lit", frozenset({"lit"})),
    ("slang_cu", frozenset({"cu"})),
    ("slang_ty", frozenset({"ty", "ti"})),
)

MIN_CHARS = 3
MAX_CHARS = 120
MAX_TOKENS = 32  # the v1 student's max_length (special tokens included)
NEAR_WATCH_THRESHOLD = 0.85  # reported, never forces review

# Proposer categories that always send a record to a human, in display order.
PROPOSER_MUST: tuple[str, ...] = (
    "multiple_money_candidates",
    "no_candidate",
    "compound_amount",
    "unusual_punctuation",
)
REASON_ORDER: tuple[str, ...] = (
    *PROPOSER_MUST,
    "type_status_not_complete",
    "type_intent_mismatch",
    "target_intent_mismatch",
    "value_intent_mismatch",
    "invalid_type_proposal",
    "invalid_value_proposal",
    "invalid_intent_span",
    "duplicate_corpus",
    "near_duplicate_corpus",
    "duplicate_batch",
    "near_duplicate_batch",
    "length_out_of_bounds",
    "numeric_ambiguity",
)

ESCALATION_RULE: dict[str, Any] = {
    "error": (
        "a human label differs from the proposal on type, target span or value span, or the "
        "human annotation_status is not 'complete' (either pass)"
    ),
    "hard_stratum": "any error in a hard-pattern stratum -> review that entire stratum",
    "clean_audit": {
        "max_errors_to_accept": 1,
        "accept": "0-1 errors -> accept the auto-accepted remainder",
        "expand": ">=2 errors -> stop and expand review to all auto-accepted records",
    },
    "must_review": "errors do not trigger escalation (expected); they are reported",
}

T = TypeVar("T")


# ---------------------------------------------------------------------------------------------
# Draws
# ---------------------------------------------------------------------------------------------


def hash_key(seed: str, salt: str, record_id: str) -> str:
    return hashlib.sha256(f"{seed}:{salt}:{record_id}".encode()).hexdigest()


def pick_diverse[T](
    items: Sequence[T],
    ident: Callable[[T], str],
    features: Callable[[T], Iterable[str]],
    n: int,
    seed: str,
    salt: str,
) -> list[T]:
    """Greedy, order-independent draw of ``n`` items that spreads over ``features``.

    Each step takes the item whose features are least represented so far (the score is the sum of
    ``1 / (1 + times the feature was already picked)``); ties are broken by the seeded hash of the
    item id. The result does not depend on the order of ``items``.
    """
    pool = sorted(items, key=lambda x: hash_key(seed, salt, ident(x)))
    counts: Counter[str] = Counter()
    picked: list[T] = []
    while pool and len(picked) < n:
        best_i, best_score = 0, Fraction(-1)
        for i, item in enumerate(pool):
            score = sum((Fraction(1, 1 + counts[f]) for f in features(item)), Fraction(0))
            if score > best_score:  # strict: the earliest hash wins ties
                best_i, best_score = i, score
        item = pool.pop(best_i)
        counts.update(features(item))
        picked.append(item)
    return picked


# ---------------------------------------------------------------------------------------------
# Strata
# ---------------------------------------------------------------------------------------------


def slang_unit(span_text: str | None) -> str | None:
    """``slang_xi``/``slang_chai``/``slang_lit``/``slang_cu``/``slang_ty`` of a value span."""
    if not span_text:
        return None
    words = set(re.findall(r"[^\W\d_]+", fold(span_text)))
    for name, unit_words in _SLANG_WORDS:
        if words & unit_words:
            return name
    return None


def hard_stratum(categories: Iterable[str], span_text: str | None) -> str | None:
    """First hard-pattern stratum of a record, or ``None`` for a clean one."""
    cats = set(categories)
    if "multi_number" in cats:
        return "multi_number"
    if "bare_number" in cats:
        return "bare_number"
    unit = slang_unit(span_text)
    if unit:
        return unit
    return "slang_other" if "slang" in cats else None


# ---------------------------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DuplicateInfo:
    corpus_norm_match: bool
    corpus_sim: float
    batch_norm_match: bool
    batch_sim: float  # nearest note of a *different* generation group


def load_reference_texts(
    root: Path, exclude_ids: set[str], exclude_dirs: Sequence[Path] = ()
) -> tuple[list[str], list[str]]:
    """Unique ``text`` values of every existing corpus/dataset/scratch JSONL, and their files.

    Searched: ``corpus/raw``, ``corpus/reviewed``, ``datasets`` and ``generated`` (``*.jsonl``).
    Records of the batch itself are skipped by id, files named after the batch and files under
    ``exclude_dirs`` entirely.
    """
    texts: set[str] = set()
    files: list[str] = []
    skip = tuple(d.resolve() for d in exclude_dirs)
    for top in ("corpus/raw", "corpus/reviewed", "datasets", "generated"):
        base = root / top
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.jsonl")):
            if path.stem.startswith(NAME) or any(d in path.resolve().parents for d in skip):
                continue
            used = False
            with path.open(encoding="utf-8") as f:
                for lineno, line in enumerate(f, start=1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{path}:{lineno}: invalid JSON: {exc.msg}") from exc
                    text = row.get("text") if isinstance(row, dict) else None
                    if isinstance(text, str) and text.strip() and row.get("id") not in exclude_ids:
                        texts.add(unicodedata.normalize("NFC", text))
                        used = True
            if used:
                files.append(str(path.relative_to(root)))
    return sorted(texts), files


def scan_duplicates(
    batch: Sequence[Mapping[str, Any]], ref_texts: Sequence[str]
) -> dict[str, DuplicateInfo]:
    """Normalised-match flags and nearest similarity of each note against the corpus and batch.

    Uses ``gidi.annotation.leakage`` (amount-masked, diacritic-folded text). Notes of the same
    generation ``group`` (minimal pairs) do not count for the batch similarity.
    """
    refs = [reference(t, None, "corpus") for t in ref_texts]
    ref_norms = {r["norm"] for r in refs}
    cands = [reference(b["text"], None, "batch") for b in batch]
    norm_counts = Counter(c["norm"] for c in cands)
    out: dict[str, DuplicateInfo] = {}
    for i, (b, cand) in enumerate(zip(batch, cands, strict=True)):
        corpus_sim, _ = nearest(cand["norm"], refs)
        others = [
            c if batch[j].get("group") != b.get("group") else {"norm": ""}
            for j, c in enumerate(cands)
        ]
        batch_sim, _ = nearest(cand["norm"], others, skip=i)
        out[b["id"]] = DuplicateInfo(
            corpus_norm_match=cand["norm"] in ref_norms,
            corpus_sim=round(corpus_sim, 4),
            batch_norm_match=norm_counts[cand["norm"]] > 1,
            batch_sim=round(batch_sim, 4),
        )
    return out


# ---------------------------------------------------------------------------------------------
# Proposals and labels
# ---------------------------------------------------------------------------------------------


def span_key(span: Mapping[str, Any] | None) -> tuple[str, int, int] | None:
    return None if span is None else (span["text"], span["start"], span["end"])


def intent_summary(note: Mapping[str, Any]) -> str:
    target = note.get("intended_target")
    value = note.get("intended_value")
    parts = [
        f"type {note.get('intended_type')}",
        f"target {target['text']!r}" if target else "target null",
        f"value {value['text']!r}" if value else "value none",
        f"value_status {note.get('value_status')}",
    ]
    if note.get("value_note"):
        parts.append(f"note {note['value_note']!r}")
    return ", ".join(parts)


def value_proposal_row(note: Mapping[str, Any], proposal: ValueProposal) -> dict[str, Any]:
    """Quet value proposal (``type`` amount/no_amount, ``target`` = value span) plus the intent."""
    row = to_quet_proposal(note["id"], proposal)
    row["reason"] = f"{row['reason']} | generation intent: {intent_summary(note)}"
    return row


def type_label(proposal: Mapping[str, Any]) -> dict[str, Any]:
    """Quet label shape of a type/target proposal (confidence and reason dropped)."""
    keys = ("id", "annotation_status", "type", "target", "note")
    return {k: proposal[k] for k in keys if k in proposal}


def value_auto_label(record_id: str, proposal: ValueProposal) -> dict[str, Any]:
    """Rule-provenance value label of an auto-accepted record (never a human label)."""
    if proposal.chosen is None:
        raise ValueError(f"{record_id}: no single value span to auto-accept")
    return {
        "id": record_id,
        "annotation_status": "complete",
        "type": "amount",
        "target": proposal.chosen.as_span(),
        "provenance": "rule",
        "confidence": proposal.confidence,
        "rule_version": RULE_VERSION,
    }


# ---------------------------------------------------------------------------------------------
# Assessment
# ---------------------------------------------------------------------------------------------


def value_matches_intent(note: Mapping[str, Any], proposal: ValueProposal) -> bool:
    """Does the proposer agree with the generation intent for the value pass?"""
    status = note.get("value_status")
    if status == "complete":
        return span_key(proposal.span) == span_key(note.get("intended_value"))
    if status == "no_amount":
        return proposal.chosen is None and "multiple_money_candidates" not in proposal.categories
    if status == "uncertain":
        return proposal.chosen is None and "multiple_money_candidates" in proposal.categories
    return False


def assess_record(
    note: Mapping[str, Any],
    type_proposal: Mapping[str, Any],
    value: ValueProposal,
    dup: DuplicateInfo,
    token_count: int,
    type_config: AnnotationConfig,
    value_config: ValueConfig,
) -> list[str]:
    """Must-review reasons of one record (``REASON_ORDER``); empty means it may be auto-accepted."""
    text = note["text"]
    found: set[str] = {c for c in PROPOSER_MUST if c in value.categories}

    if type_proposal.get("annotation_status") != "complete":
        found.add("type_status_not_complete")
    if type_proposal.get("type") != note.get("intended_type"):
        found.add("type_intent_mismatch")
    if span_key(type_proposal.get("target")) != span_key(note.get("intended_target")):
        found.add("target_intent_mismatch")
    if not value_matches_intent(note, value):
        found.add("value_intent_mismatch")

    if validate_annotation(type_label(type_proposal), text, type_config):
        found.add("invalid_type_proposal")
    bad_value = validate_quet_proposal(to_quet_proposal(note["id"], value), text, value_config)
    if value.chosen is not None:
        bad_value += validate_value_label(
            value_auto_label(note["id"], value), text, value_config, extra_fields=RULE_LABEL_EXTRA
        )
    if bad_value:
        found.add("invalid_value_proposal")
    for field_name in ("intended_target", "intended_value"):
        span = note.get(field_name)
        if span is not None and validate_span(span, text, field_name):
            found.add("invalid_intent_span")

    if dup.corpus_norm_match:
        found.add("duplicate_corpus")
    if dup.corpus_sim >= NEAR_DUP_THRESHOLD:
        found.add("near_duplicate_corpus")
    if dup.batch_norm_match:
        found.add("duplicate_batch")
    if dup.batch_sim >= NEAR_DUP_THRESHOLD:
        found.add("near_duplicate_batch")

    if len(text) < MIN_CHARS or len(text) > MAX_CHARS or token_count > MAX_TOKENS:
        found.add("length_out_of_bounds")
    # A number dropped only because it sits beside a money expression (no unit, date, percent or
    # quantity evidence) is a decision by elimination.
    if any(x.reason == "bare_beside_money" for x in value.excluded):
        found.add("numeric_ambiguity")
    return [r for r in REASON_ORDER if r in found]


# ---------------------------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Facts:
    """What the draws need to know about one record."""

    id: str
    reasons: tuple[str, ...]
    hard_stratum: str | None
    type: str | None
    accented: bool
    form: str
    target_present: bool
    length: int
    group: str = ""


@dataclass(frozen=True)
class Decision:
    id: str
    group: str
    stratum: str
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def queued(self) -> bool:
        return self.group in QUEUED_GROUPS


def _accent(f: Facts) -> str:
    return "accent:accented" if f.accented else "accent:unaccented"


def plan_review(
    facts: Sequence[Facts],
    seed: str = SEED,
    per_stratum: int = HARD_SAMPLE_PER_STRATUM,
    audit_size: int = CLEAN_AUDIT_SIZE,
) -> dict[str, Decision]:
    """Group every record; the result does not depend on the order of ``facts``."""
    decisions: dict[str, Decision] = {}
    rest: list[Facts] = []
    for f in facts:
        if f.reasons:
            stratum = f.reasons[0]
            decisions[f.id] = Decision(f.id, MUST, stratum, f.reasons)
        else:
            rest.append(f)

    for stratum in HARD_STRATA:
        members = [f for f in rest if f.hard_stratum == stratum]
        picked = pick_diverse(
            members,
            lambda f: f.id,
            lambda f: (f"type:{f.type}", _accent(f)),
            per_stratum,
            seed,
            f"hard:{stratum}",
        )
        chosen = {f.id for f in picked}
        for f in members:
            decisions[f.id] = Decision(f.id, HARD if f.id in chosen else AUTO, stratum)

    clean = [f for f in rest if f.hard_stratum is None]
    cut = median(f.length for f in clean) if clean else 0

    def audit_features(f: Facts) -> tuple[str, ...]:
        return (
            f"type:{f.type}",
            _accent(f),
            f"form:{f.form}",
            f"target:{'present' if f.target_present else 'null'}",
            f"length:{'short' if f.length <= cut else 'long'}",
            f"group:{f.group}",
        )

    picked = pick_diverse(clean, lambda f: f.id, audit_features, audit_size, seed, "audit")
    chosen = {f.id for f in picked}
    for f in clean:
        decisions[f.id] = Decision(f.id, AUDIT if f.id in chosen else AUTO, CLEAN_STRATUM)
    return decisions


def queue_order(decisions: Mapping[str, Decision], seed: str = SEED) -> list[str]:
    """Queued ids: must-review first, then hard sample, then clean audit; seeded hash within."""
    rank = {g: i for i, g in enumerate(QUEUED_GROUPS)}
    queued = [d for d in decisions.values() if d.queued]
    queued.sort(key=lambda d: (rank[d.group], hash_key(seed, "order", d.id)))
    return [d.id for d in queued]


# ---------------------------------------------------------------------------------------------
# Scoring and escalation
# ---------------------------------------------------------------------------------------------


def pass_errors(human: Mapping[str, Any], proposal: Mapping[str, Any]) -> list[str]:
    """How one human label differs from its proposal: ``status``, ``type``, ``target``."""
    if human.get("annotation_status") != "complete":
        return ["status"]
    errors = []
    if human.get("type") != proposal.get("type"):
        errors.append("type")
    if span_key(human.get("target")) != span_key(proposal.get("target")):
        errors.append("target")
    return errors


def record_errors(passes: Mapping[str, tuple[Mapping[str, Any], Mapping[str, Any]]]) -> list[str]:
    """Errors of a record, ``<pass>:<what>``; ``passes`` maps a pass name to (human, proposal)."""
    return [
        f"{name}:{e}"
        for name, (human, proposal) in passes.items()
        for e in pass_errors(human, proposal)
    ]


@dataclass(frozen=True)
class Tally:
    """Errors among the labelled records of one group or stratum."""

    queued: int
    labelled: int
    errors: int

    @property
    def complete(self) -> bool:
        return self.labelled >= self.queued


@dataclass(frozen=True)
class Escalation:
    status: str  # escalated | pending | accept_auto_remainder
    hard_strata: dict[str, str]  # stratum -> review_entire_stratum | ok | pending
    clean_audit: str  # expand_all | accept | pending
    ids: tuple[str, ...]  # auto-accepted records that must now be reviewed
    reasons: tuple[str, ...]


def decide_escalation(
    strata: Mapping[str, Tally],
    audit: Tally,
    auto_by_stratum: Mapping[str, Sequence[str]],
    max_audit_errors: int = ESCALATION_RULE["clean_audit"]["max_errors_to_accept"],
    audit_scope: Collection[str] | None = None,
) -> Escalation:
    """Apply the predeclared rule. ``auto_by_stratum`` lists auto-accepted ids per stratum.

    A sampled-stratum failure reviews the auto-accepted records of that stratum. An audit failure
    expands to the auto-accepted records of the ``audit_scope`` strata (``None``: every stratum,
    including ``clean``). Strata are judged in the order of ``strata``.
    """
    reasons: list[str] = []
    ids: list[str] = []
    hard: dict[str, str] = {}
    for stratum, tally in strata.items():
        if tally.queued == 0:
            continue
        if tally.errors > 0:
            hard[stratum] = "review_entire_stratum"
            ids += auto_by_stratum.get(stratum, ())
            reasons.append(f"{stratum}: {tally.errors} error(s) in {tally.labelled} labelled")
        else:
            hard[stratum] = "ok" if tally.complete else "pending"

    if audit.errors > max_audit_errors:
        audit_decision = "expand_all"
        ids += [
            i
            for stratum, stratum_ids in auto_by_stratum.items()
            if audit_scope is None or stratum in audit_scope
            for i in stratum_ids
        ]
        reasons.append(f"clean_audit: {audit.errors} error(s) > {max_audit_errors}")
    elif audit.queued and not audit.complete:
        audit_decision = "pending"
    else:
        audit_decision = "accept"

    unique = tuple(dict.fromkeys(ids))
    if unique:
        status = "escalated"
    elif audit_decision == "pending" or "pending" in hard.values():
        status = "pending"
    else:
        status = "accept_auto_remainder"
    return Escalation(status, hard, audit_decision, unique, tuple(reasons))
