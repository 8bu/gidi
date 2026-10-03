#!/usr/bin/env python
r"""Score the reduced value reviews by stratum and apply the predeclared escalation rule.

Usage:
    uv run python scripts/score_value_review.py [--root R] [--dataset NAME|all] [--out FILE]

Two reviews share one policy (``gidi.annotation.risk_review``): ``existing-value-review`` (the
in-scope notes of gidi-finance-v1; ``scripts/build_value_queue.py``) and ``targeted-value-01``
(``scripts/build_targeted_value_queue.py``). Run it after the Quet passes over the reduced queue
(commands in ``experiments/value-span-v1/*-review-plan.md``). It compares the human labels
(only Quet writes them) with the advisory proposals per review group and sampled stratum, prints
the error rates and the escalation decision, and merges the result into
``experiments/value-span-v1/review-score.json`` (one entry per dataset).

An error is a human label that differs from the proposal (span, or present versus null; for
targeted-value-01 also type/target) or whose ``annotation_status`` is not ``complete``.
Escalation rule:

* an error in a sampled stratum -> that stratum is expanded to full review;
* clean audit: 0-1 errors let the auto-accepted clean set stand; >=2 stop it and expand the review
  of the clean population (existing-value-review: in stages of 50; targeted-value-01: all);
* must-review errors are expected: reported, never an escalation trigger.

Each stratum gets a verdict (``strata_status``: passed / failed / pending). Rule labels of
auto-accepted records train only in ``passed`` strata (``scripts/build_training_v2.py`` reads the
verdicts and refuses a stale score: the sha256 of every input is recorded).

When escalation is required the records to review are written as a Quet queue with proposals
(``*escalation-queue.jsonl``) next to the dataset; records that still lack a human label are
"pending" and the decision is then provisional.

Exits 2 with a message when the human labels of a selected dataset are absent or invalid. Never
writes a label. ``--dataset all`` (default) needs labels for both datasets; score one at a time
with ``--dataset NAME``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gidi.annotation import risk_review as rr
from gidi.annotation import value_review as vr
from gidi.annotation.review_gate import SCORE_PATH, sha256_file, stratum_verdicts
from gidi.annotation.schema import load_config, validate_file
from gidi.annotation.value_span import load_value_config, propose_value, validate_label_file
from gidi.corpus.jsonl import read_jsonl

V2 = Path("datasets/annotation-v2")
TARGETED = V2 / "targeted-value-01"
VALUE_SCHEMA = "configs/annotation-v2-value.quet.yaml"


class ScoreError(Exception):
    """Human labels absent or invalid; the message is shown to the user."""


@dataclass(frozen=True)
class Pass:
    """One Quet pass: its human label file and the proposals the reviewer saw."""

    name: str  # type | value
    labels: str
    proposals: str


@dataclass(frozen=True)
class Dataset:
    key: str
    seed: str
    directory: Path
    queue: str
    provenance: str
    texts: str  # file holding {id, text} for every record
    passes: tuple[Pass, ...]
    escalation_queue: str
    escalation_proposals: dict[str, str]  # pass name -> file
    rule: dict[str, Any]
    reason_order: tuple[str, ...]
    stratum_rank: Callable[[str], Any]
    audit_scope: frozenset[str] | None  # strata an audit failure expands (None: all)
    stage_size: int | None  # expand the clean stratum in stages of this size (None: at once)
    build_script: str
    plan_file: str
    proposal_rows: Callable[[Path, Dataset, list[str]], dict[str, list[dict]]]
    quet_commands: Callable[[Dataset], list[str]]


def _targeted_proposal_rows(root: Path, ds: Dataset, ids: list[str]) -> dict[str, list[dict]]:
    d = root / ds.directory
    notes = {n["id"]: n for n in read_jsonl(d / "generation-notes.jsonl")}
    types = {p["id"]: p for p in read_jsonl(d / "type-proposals.jsonl")}
    return {
        "type": [types[i] for i in ids],
        "value": [rr.value_proposal_row(notes[i], propose_value(notes[i]["text"])) for i in ids],
    }


def _existing_proposal_rows(root: Path, ds: Dataset, ids: list[str]) -> dict[str, list[dict]]:
    rows = {p["id"]: p for p in read_jsonl(root / ds.directory / "value-proposals.jsonl")}
    return {"value": [rows[i] for i in ids]}


def _targeted_commands(ds: Dataset) -> list[str]:
    d = ds.directory.as_posix()
    return [
        f"quet annotate {d}/{ds.escalation_queue} --schema configs/annotation-v1.yaml "
        f"--labels {d}/labels.jsonl --proposals {d}/{ds.escalation_proposals['type']}",
        f"quet annotate {d}/{ds.escalation_queue} --schema {VALUE_SCHEMA} "
        f"--labels {d}/value-labels.jsonl --proposals {d}/{ds.escalation_proposals['value']}",
    ]


def _existing_commands(ds: Dataset) -> list[str]:
    d = ds.directory.as_posix()
    return [
        f"quet annotate {d}/{ds.escalation_queue} --schema {VALUE_SCHEMA} "
        f"--labels {d}/value-labels.jsonl --proposals {d}/{ds.escalation_proposals['value']}"
    ]


def _hard_rank(stratum: str) -> tuple[int, str]:
    order = rr.HARD_STRATA.index(stratum) if stratum in rr.HARD_STRATA else len(rr.HARD_STRATA)
    return order, stratum


DATASETS: dict[str, Dataset] = {
    vr.NAME: Dataset(
        key=vr.NAME,
        seed=vr.SEED,
        directory=V2,
        queue="value-review-queue.jsonl",
        provenance="value-review-provenance.jsonl",
        texts="value-queue.jsonl",
        passes=(Pass("value", "value-labels.jsonl", "value-proposals.jsonl"),),
        escalation_queue="value-escalation-queue.jsonl",
        escalation_proposals={"value": "value-escalation-proposals.jsonl"},
        rule=vr.ESCALATION_RULE,
        reason_order=vr.REASON_ORDER,
        stratum_rank=vr.stratum_rank,
        audit_scope=frozenset({rr.CLEAN_STRATUM}),
        stage_size=vr.CLEAN_EXPANSION_STAGE,
        build_script="scripts/build_value_queue.py",
        plan_file="experiments/value-span-v1/existing-value-review-plan.md",
        proposal_rows=_existing_proposal_rows,
        quet_commands=_existing_commands,
    ),
    rr.NAME: Dataset(
        key=rr.NAME,
        seed=rr.SEED,
        directory=TARGETED,
        queue="review-queue.jsonl",
        provenance="review-provenance.jsonl",
        texts="generation-notes.jsonl",
        passes=(
            Pass("type", "labels.jsonl", "review-type-proposals.jsonl"),
            Pass("value", "value-labels.jsonl", "review-value-proposals.jsonl"),
        ),
        escalation_queue="escalation-queue.jsonl",
        escalation_proposals={
            "type": "escalation-type-proposals.jsonl",
            "value": "escalation-value-proposals.jsonl",
        },
        rule=rr.ESCALATION_RULE,
        reason_order=rr.REASON_ORDER,
        stratum_rank=_hard_rank,
        audit_scope=None,
        stage_size=None,
        build_script="scripts/build_targeted_value_queue.py",
        plan_file="experiments/value-span-v1/targeted-value-01-review-plan.md",
        proposal_rows=_targeted_proposal_rows,
        quet_commands=_targeted_commands,
    ),
}


def load_human_labels(root: Path, ds: Dataset, texts: dict[str, str]) -> dict[str, dict]:
    """Validated human labels per pass (``{pass: {id: label}}``), or ``ScoreError``."""
    d = root / ds.directory
    missing = [str(ds.directory / p.labels) for p in ds.passes if not (d / p.labels).exists()]
    if missing:
        raise ScoreError(
            f"human labels absent for {ds.key}: {', '.join(missing)} not found. Run the Quet "
            f"pass(es) over {ds.directory / ds.queue} first (commands in {ds.plan_file}); this "
            "script never creates labels."
        )
    labels: dict[str, dict] = {}
    for p in ds.passes:
        path = d / p.labels
        if p.name == "type":
            report = validate_file(path, texts, load_config(root / "configs/annotation-v1.yaml"))
            if not report.ok:
                errors = [f"{e.line}: {e.id}: {e.message}" for e in report.errors[:10]]
                raise ScoreError(f"invalid type/target labels in {path}:\n  " + "\n  ".join(errors))
            labels[p.name] = {r["id"]: r for r in read_jsonl(path)}
        else:
            value_report = validate_label_file(
                path, texts, load_value_config(root / "configs/annotation-v2.yaml")
            )
            if not value_report.ok:
                errors = value_report.errors[:10]
                raise ScoreError(f"invalid value labels in {path}:\n  " + "\n  ".join(errors))
            labels[p.name] = value_report.labels
    return labels


def tally_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labelled = [r for r in rows if r["labelled"]]
    errors = [r for r in labelled if r["errors"]]
    kinds = Counter(e for r in errors for e in r["errors"])
    return {
        "queued": len(rows),
        "labelled": len(labelled),
        "pending": len(rows) - len(labelled),
        "errors": len(errors),
        "error_rate": round(len(errors) / len(labelled), 4) if labelled else None,
        "error_kinds": dict(sorted(kinds.items())),
        "error_ids": [r["id"] for r in errors],
    }


def score(
    ds: Dataset,
    queue: list[dict],
    provenance: dict[str, dict],
    proposals: dict[str, dict[str, dict]],
    labels: dict[str, dict[str, dict]],
) -> tuple[dict[str, Any], rr.Escalation, dict[str, str]]:
    """Tallies, the escalation decision and the stratum verdicts (pure; no file access).

    ``proposals`` and ``labels`` map a pass name to ``{id: row}``.
    """
    rows = []
    for q in queue:
        i = q["id"]
        labelled = all(i in labels[p] for p in labels)
        errors = (
            rr.record_errors({p: (labels[p][i], proposals[p][i]) for p in labels})
            if labelled
            else []
        )
        rows.append(
            {
                "id": i,
                "group": q["review_group"],
                "stratum": q["stratum"],
                "reasons": q["reasons"],
                "labelled": labelled,
                "errors": errors,
            }
        )

    groups = {g: tally_rows([r for r in rows if r["group"] == g]) for g in rr.QUEUED_GROUPS}
    names = {r["stratum"] for r in rows if r["group"] == rr.HARD}
    names |= {p["stratum"] for p in provenance.values() if p["review_group"] == rr.AUTO} - {
        rr.CLEAN_STRATUM
    }
    strata = {
        s: tally_rows([r for r in rows if r["group"] == rr.HARD and r["stratum"] == s])
        for s in sorted(names, key=ds.stratum_rank)
    }
    by_reason = {}
    for reason in ds.reason_order:
        members = [r for r in rows if r["group"] == rr.MUST and reason in r["reasons"]]
        if members:
            t = tally_rows(members)
            by_reason[reason] = {k: t[k] for k in ("queued", "labelled", "errors", "error_rate")}

    auto_by_stratum: dict[str, list[str]] = {}
    for i, p in provenance.items():
        if p["review_group"] == rr.AUTO:
            auto_by_stratum.setdefault(p["stratum"], []).append(i)

    def as_tally(t: dict[str, Any]) -> rr.Tally:
        return rr.Tally(t["queued"], t["labelled"], t["errors"])

    escalation = rr.decide_escalation(
        {s: as_tally(t) for s, t in strata.items()},
        as_tally(groups[rr.AUDIT]),
        auto_by_stratum,
        audit_scope=ds.audit_scope,
    )
    detail = {
        "groups": groups,
        "strata": strata,
        "must_review_by_reason": by_reason,
        "error_details": {
            r["id"]: {"group": r["group"], "stratum": r["stratum"], "errors": r["errors"]}
            for r in rows
            if r["errors"]
        },
    }
    return detail, escalation, stratum_verdicts(escalation, auto_by_stratum)


def escalation_ids(
    ds: Dataset,
    escalation: rr.Escalation,
    provenance: dict[str, dict],
    labelled: set[str],
) -> list[str]:
    """Escalated records still lacking a human label; the clean stratum is released in stages."""
    todo = [i for i in escalation.ids if i not in labelled]
    if ds.stage_size is None:
        return todo
    clean = {i for i in todo if provenance[i]["stratum"] == rr.CLEAN_STRATUM}
    staged = set(vr.stage_clean_ids(clean, labelled, ds.seed, ds.stage_size))
    return [i for i in todo if i not in clean or i in staged]


def write_escalation_queue(
    root: Path,
    ds: Dataset,
    ids: list[str],
    texts: dict[str, dict],
    provenance: dict[str, dict],
) -> None:
    """Quet queue + proposals for the escalated records (stale files are removed)."""
    d = root / ds.directory
    names = [ds.escalation_queue, *ds.escalation_proposals.values()]
    if not ids:
        for name in names:
            (d / name).unlink(missing_ok=True)
        return
    queue = [
        {
            "id": i,
            "text": texts[i]["text"],
            "review_group": "escalation",
            "stratum": provenance[i]["stratum"],
            "reasons": [],
            **{k: texts[i][k] for k in ("split", "source") if k in texts[i]},
        }
        for i in ids
    ]
    rows = ds.proposal_rows(root, ds, ids)
    files = [ds.escalation_queue, *(ds.escalation_proposals[p.name] for p in ds.passes)]
    contents = [queue, *(rows[p.name] for p in ds.passes)]
    for name, content in zip(files, contents, strict=True):
        (d / name).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in content), encoding="utf-8"
        )


def print_summary(
    ds: Dataset, detail: dict[str, Any], escalation: rr.Escalation, verdicts: dict[str, str]
) -> None:
    def line(name: str, t: dict[str, Any]) -> str:
        rate = "n/a" if t["error_rate"] is None else f"{t['error_rate']:.1%}"
        return (
            f"  {name:<34} queued {t['queued']:>3}  labelled {t['labelled']:>3}  "
            f"errors {t['errors']:>3}  rate {rate}"
        )

    print(f"== {ds.key}")
    print("groups")
    for g, t in detail["groups"].items():
        print(line(g, t))
    print("sampled strata")
    for s, t in detail["strata"].items():
        print(line(s, t) + f"  -> {verdicts.get(s, 'n/a')}")
    print(f"  clean audit verdict: {verdicts.get(rr.CLEAN_STRATUM, 'n/a')}")
    print("must-review by reason (reported, never an escalation trigger)")
    for reason, t in detail["must_review_by_reason"].items():
        print(line(reason, t | {"pending": 0, "error_kinds": {}, "error_ids": []}))
    print(f"escalation: {escalation.status}")
    for r in escalation.reasons:
        print(f"  - {r}")
    print(f"  sampled strata: {escalation.hard_strata}")
    print(f"  clean audit: {escalation.clean_audit}")
    print(f"  auto-accepted records to review: {len(escalation.ids)}")


def score_dataset(root: Path, ds: Dataset) -> dict[str, Any]:
    """Score one dataset from its files, write its escalation queue, return its score entry."""
    d = root / ds.directory
    needed = (ds.queue, ds.provenance, ds.texts, *(p.proposals for p in ds.passes))
    absent = [n for n in needed if not (d / n).exists()]
    if absent:
        raise ScoreError(
            f"{', '.join(absent)} missing in {ds.directory}: run {ds.build_script} first"
        )
    texts = {r["id"]: r for r in read_jsonl(d / ds.texts)}
    queue = read_jsonl(d / ds.queue)
    provenance = {p["id"]: p for p in read_jsonl(d / ds.provenance)}
    labels = load_human_labels(root, ds, {i: r["text"] for i, r in texts.items()})
    queue_ids = {q["id"] for q in queue}
    if not any(queue_ids & set(pass_labels) for pass_labels in labels.values()):
        raise ScoreError(
            f"human labels absent for {ds.key}: the label files hold no label for any queued "
            f"record. Run the Quet pass(es) over {ds.directory / ds.queue} first "
            f"(commands in {ds.plan_file})."
        )
    proposals = {p.name: {r["id"]: r for r in read_jsonl(d / p.proposals)} for p in ds.passes}
    detail, escalation, verdicts = score(ds, queue, provenance, proposals, labels)

    labelled_all = set.intersection(*(set(v) for v in labels.values()))
    todo = escalation_ids(ds, escalation, provenance, labelled_all)
    write_escalation_queue(root, ds, todo, texts, provenance)
    queue_file = None
    if todo:
        clean_left = sum(
            provenance[i]["stratum"] == rr.CLEAN_STRATUM and i not in labelled_all
            for i in escalation.ids
        )
        queue_file = {
            "queue": str(ds.directory / ds.escalation_queue),
            "records_without_human_labels": len(todo),
            "clean_records_still_unreviewed": clean_left if ds.stage_size else None,
            "stage_size": ds.stage_size,
            "quet_commands": ds.quet_commands(ds),
        }

    print_summary(ds, detail, escalation, verdicts)
    input_names = (ds.queue, ds.provenance, ds.texts)
    input_names += tuple(n for p in ds.passes for n in (p.proposals, p.labels))
    return {
        "name": ds.key,
        "seed": ds.seed,
        "rule": ds.rule,
        "labels": {
            **{f"{p.name}_labels": len(labels[p.name]) for p in ds.passes},
            "queued": len(queue),
            "fully_labelled": sum(q["id"] in labelled_all for q in queue),
        },
        **detail,
        "escalation": {
            "status": escalation.status,
            "strata": escalation.hard_strata,
            "clean_audit": escalation.clean_audit,
            "reasons": list(escalation.reasons),
            "auto_accepted_ids_to_review": list(escalation.ids),
            "queue": queue_file,
        },
        "strata_status": verdicts,
        "inputs": {str(ds.directory / n): sha256_file(d / n) for n in input_names},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--dataset",
        choices=(*DATASETS, "all"),
        default="all",
        help="review to score (default: all; each selected dataset needs its human labels)",
    )
    parser.add_argument("--out", type=Path, default=SCORE_PATH)
    args = parser.parse_args(argv)
    root = args.root
    selected = list(DATASETS.values()) if args.dataset == "all" else [DATASETS[args.dataset]]

    entries: dict[str, Any] = {}
    try:
        for ds in selected:
            entries[ds.key] = score_dataset(root, ds)
    except ScoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    out = root / args.out
    merged = json.loads(out.read_text(encoding="utf-8"))["datasets"] if out.exists() else {}
    merged.update(entries)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"datasets": merged}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.out} ({', '.join(entries)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
