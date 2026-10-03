#!/usr/bin/env python
"""Build the annotation-v2 value-span files for the notes that feed gidi-finance-v1.

Usage:
    uv run python scripts/build_value_queue.py [--root R] [--out-dir D] [--report-dir D]
                                               [--force] [--check]

In-scope records (deduplicated by id) are the union of

* ``datasets/annotation-v1/distillation-v1/train.jsonl``: the 723 notes compression-v3 (and the
  frozen teacher) trained on = frozen train 501 + targeted-02 112 + frozen validation 110;
* ``datasets/annotation-v1/splits/validation.jsonl`` and ``splits/test.jsonl`` (the frozen
  evaluation splits, also reachable as ``training-v3/{validation,test}.jsonl``);
* ``datasets/probe-v1/notes.jsonl`` (evaluation only; texts are read, nothing there is changed).

Writes ``datasets/annotation-v2/``:

* ``value-queue.jsonl``: ``{id, text, split, source, source_batch}`` for every in-scope record;
* ``value-proposals.jsonl``: one advisory Quet proposal per record (``--proposals``);
* ``value-auto-labels.jsonl``: the clean, rule-accepted, high-confidence labels
  (``provenance: rule``). Quet never reads this file and it is never a human label;
* ``value-review-queue.jsonl``: the reduced queue a human reviews, ``{id, text, review_group,
  stratum, reasons, split, source}``: every must-review record, a few records per pattern
  stratum of the other flagged notes, and a small clean-audit sample (policy and draws:
  ``gidi.annotation.value_review``, same tiers as ``targeted-value-01``);
* ``value-review-provenance.jsonl``: per in-scope record its group, stratum and provenance.
  Flagged records outside the queue keep their rule proposal (``provenance: rule``, status
  ``pending-stratum-check``): they train only if ``scripts/score_value_review.py`` passes their
  stratum, and they are never human labels;
* ``value-manifest.json``: counts, the review plan, input/output sha256, seed.

and the plan report ``experiments/value-span-v1/existing-value-review-plan.{md,json}``.

``value-labels.jsonl`` (the human Quet output) is never created or modified here. Run the pass with

    quet annotate datasets/annotation-v2/value-review-queue.jsonl \\
      --schema configs/annotation-v2-value.quet.yaml \\
      --out datasets/annotation-v2/value-labels.jsonl \\
      --proposals datasets/annotation-v2/value-proposals.jsonl

then ``uv run python scripts/score_value_review.py --dataset existing-value-review``.

The build is deterministic. ``--check`` rebuilds in memory and verifies the files on disk.
Existing outputs are only replaced with ``--force``, and never if that would orphan ids already
labelled by a human.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from gidi.annotation import risk_review as rr
from gidi.annotation import value_review as vr
from gidi.annotation.value_span import (
    RULE_LABEL_EXTRA,
    RULE_VERSION,
    ValueProposal,
    load_value_config,
    pattern_signature,
    propose_value,
    to_quet_proposal,
    to_rule_label,
    validate_quet_proposal,
    validate_value_label,
)
from gidi.corpus.jsonl import read_jsonl

ANNOTATION = Path("datasets/annotation-v1")
OUT_DIR = Path("datasets/annotation-v2")
REPORT_DIR = Path("experiments/value-span-v1")
REPORT_STEM = "existing-value-review-plan"
INPUTS = {
    "train": ANNOTATION / "distillation-v1" / "train.jsonl",
    "validation": ANNOTATION / "splits" / "validation.jsonl",
    "test": ANNOTATION / "splits" / "test.jsonl",
    "probe": Path("datasets/probe-v1") / "notes.jsonl",
}
PROBE_LABELS = Path("datasets/probe-v1") / "labels.jsonl"
SOURCE_NAMES = {
    "train": "compression-v3-train",
    "validation": "frozen-validation",
    "test": "frozen-test",
    "probe": "probe-v1",
}
FILES = {
    "queue": "value-queue.jsonl",
    "proposals": "value-proposals.jsonl",
    "auto_labels": "value-auto-labels.jsonl",
    "review_queue": "value-review-queue.jsonl",
    "provenance": "value-review-provenance.jsonl",
    "manifest": "value-manifest.json",
}
ESCALATION_QUEUE = "value-escalation-queue.jsonl"
HUMAN_LABELS = "value-labels.jsonl"
SCORE_COMMAND = "uv run python scripts/score_value_review.py --dataset " + vr.NAME


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl_bytes(rows: list[dict]) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")


def json_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def quet_command(out_dir: Path) -> str:
    d = out_dir.as_posix()
    return (
        f"quet annotate {d}/{FILES['review_queue']} \\\n"
        f"  --schema configs/annotation-v2-value.quet.yaml \\\n"
        f"  --out {d}/{HUMAN_LABELS} \\\n"
        f"  --proposals {d}/{FILES['proposals']}"
    )


def load_scope(root: Path) -> tuple[list[dict], dict[str, dict]]:
    """In-scope records in a stable order (train, validation, test, probe; file order), and
    the v1 facts of each (``type``, ``target`` present) the draws spread over."""
    rows: dict[str, dict] = {}
    v1: dict[str, dict] = {}
    order: list[str] = []

    def add(record: dict, split: str, source_batch: str, facts: dict) -> None:
        rid, text = record["id"], record["text"]
        if unicodedata.normalize("NFC", text) != text:
            raise ValueError(f"{rid}: text is not NFC-normalized")
        if rid in rows:
            if rows[rid]["text"] != text:
                raise ValueError(f"{rid}: texts differ between input files")
            return
        rows[rid] = {
            "id": rid,
            "text": text,
            "split": split,
            "source": SOURCE_NAMES[split],
            "source_batch": source_batch,
        }
        v1[rid] = facts
        order.append(rid)

    def v1_facts(record: dict) -> dict:
        return {"type": record["type"], "target": record["target"] is not None}

    for r in read_jsonl(root / INPUTS["train"]):
        add(r, "train", r["source_batch"], v1_facts(r))
    validation = read_jsonl(root / INPUTS["validation"])
    validation_ids = {r["id"] for r in validation}
    for rid in validation_ids & set(rows):
        rows[rid]["split"] = "validation"
        rows[rid]["source"] = SOURCE_NAMES["validation"]
    for r in validation:
        add(r, "validation", r["source_batch"], v1_facts(r))
    for r in read_jsonl(root / INPUTS["test"]):
        if r["id"] in rows:
            raise ValueError(f"{r['id']}: frozen test note is also in a training input")
        add(r, "test", r["source_batch"], v1_facts(r))
    probe_labels = {r["id"]: r for r in read_jsonl(root / PROBE_LABELS)}
    for r in read_jsonl(root / INPUTS["probe"]):
        if not r["id"].startswith("probe-"):
            raise ValueError(f"{r['id']}: probe id lacks the 'probe-' prefix")
        if r["id"] in rows:
            raise ValueError(f"{r['id']}: probe note is also in another input")
        label = probe_labels[r["id"]]
        add(r, "probe", "probe-v1", {"type": label["type"], "target": label["target"] is not None})
    return [rows[i] for i in order], v1


def make_facts(
    scope: list[dict], v1: dict[str, dict], proposals: dict[str, ValueProposal]
) -> list[vr.Facts]:
    facts = []
    for r in scope:
        p = proposals[r["id"]]
        reasons = vr.must_reasons(r["text"], p)
        facts.append(
            vr.Facts(
                id=r["id"],
                reasons=tuple(reasons),
                stratum=None if reasons else vr.pattern_stratum(p),
                type=v1[r["id"]]["type"],
                accented="unaccented" not in p.categories,
                form=pattern_signature(p.chosen.text) if p.chosen else "none",
                target_present=v1[r["id"]]["target"],
                split=r["split"],
            )
        )
    return facts


def build_plan(
    scope: list[dict],
    proposals: dict[str, ValueProposal],
    decisions: dict[str, rr.Decision],
    order: list[str],
    out_dir: Path,
) -> dict:
    """The review plan: counts, original-queue breakdown, strata, escalation rule, command."""
    by_id = {r["id"]: r for r in scope}
    flagged = [r for r in scope if proposals[r["id"]].needs_review]
    must = [r["id"] for r in scope if decisions[r["id"]].group == rr.MUST]
    group_counts = Counter(decisions[r["id"]].group for r in scope)
    reason_counts = Counter(x for i in must for x in decisions[i].reasons)
    primary = Counter(proposals[r["id"]].review_reasons[0] for r in flagged)
    any_reason = Counter(x for r in flagged for x in proposals[r["id"]].review_reasons)
    patterns = Counter(vr.pattern_stratum(proposals[r["id"]]) for r in flagged)

    original: dict[str, dict[str, int]] = {}
    for r in flagged:
        reason = proposals[r["id"]].review_reasons[0]
        cell = original.setdefault(reason, {"records": 0, rr.MUST: 0, rr.HARD: 0, rr.AUTO: 0})
        cell["records"] += 1
        cell[decisions[r["id"]].group] += 1

    strata = {}
    for stratum in sorted(
        {d.stratum for d in decisions.values() if d.group in (rr.HARD, rr.AUTO)}
        - {rr.CLEAN_STRATUM},
        key=vr.stratum_rank,
    ):
        members = [i for i, d in decisions.items() if d.stratum == stratum]
        sampled = [i for i in order if decisions[i].stratum == stratum and decisions[i].queued]
        strata[stratum] = {
            "population": len(members),
            "sample": len(sampled),
            "unsampled": len(members) - len(sampled),
            "sampled": [{"id": i, "text": by_id[i]["text"]} for i in sampled],
        }
    clean = [i for i, d in decisions.items() if d.stratum == rr.CLEAN_STRATUM]
    audit = [i for i in order if decisions[i].group == rr.AUDIT]
    unsampled_flagged = sum(
        d.group == rr.AUTO and d.stratum != rr.CLEAN_STRATUM for d in decisions.values()
    )
    return {
        "name": vr.NAME,
        "seed": vr.SEED,
        "counts": {
            "in_scope": len(scope),
            "flagged": len(flagged),
            "clean_auto_accepted": len(clean),
            "must_review": group_counts[rr.MUST],
            "hard_sample": group_counts[rr.HARD],
            "clean_audit": group_counts[rr.AUDIT],
            "review_queue": len(order),
            "unsampled_flagged": unsampled_flagged,
            "genuinely_ambiguous": len(must),
        },
        "genuinely_ambiguous": (
            "must-review records: flagged notes whose value a rule cannot settle (several or no "
            "candidate, a bare number, spelled-out number words, a compound amount, odd "
            "punctuation, a split/share amount, a cu/cũ/củ collision). The other flagged notes "
            "carry explicit period/quantity/date/unit evidence for the extra number."
        ),
        "original_queue": {
            "total": len(flagged) + vr.ORIGINAL_AUDIT_SAMPLE,
            "flagged": len(flagged),
            "audit_sample": vr.ORIGINAL_AUDIT_SAMPLE,
            "flagged_by_primary_reason": dict(primary),
            "flagged_by_any_reason": dict(any_reason),
            "flagged_by_pattern": {s: patterns[s] for s in sorted(patterns, key=vr.stratum_rank)},
            "primary_reason_to_new_group": original,
        },
        "must_review_reason_counts": {
            r: reason_counts[r] for r in vr.REASON_ORDER if reason_counts[r]
        },
        "must_review": [
            {"id": i, "text": by_id[i]["text"], "reasons": list(decisions[i].reasons)}
            for i in order
            if decisions[i].group == rr.MUST
        ],
        "strata": strata,
        "clean": {
            "population": len(clean),
            "audit_sample": len(audit),
            "audit_sample_size": vr.AUDIT_SIZE,
            "audit": [{"id": i, "text": by_id[i]["text"]} for i in audit],
        },
        "sampling": {
            "per_stratum": (
                f"all if population <= {vr.SAMPLE_ALL_UP_TO}, {vr.SAMPLE_SMALL} if < "
                f"{vr.SAMPLE_LARGE_FROM}, else {vr.SAMPLE_LARGE}; spread over accents and v1 type"
            ),
            "clean_audit": "spread over v1 type, accents, span unit form, target present/null, "
            "split",
            "draw": "greedy least-represented-feature pick, ties by sha256(seed:salt:id)",
        },
        "escalation_rule": vr.ESCALATION_RULE,
        "quet_command": quet_command(out_dir),
        "score_command": SCORE_COMMAND,
    }


def _table(header: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines


def render_markdown(plan: dict) -> str:
    c, o = plan["counts"], plan["original_queue"]
    rule = plan["escalation_rule"]
    lines = [
        "# Existing-data value review plan (reduced, tiered)",
        "",
        "Generated by `scripts/build_value_queue.py` (deterministic; `--check` verifies). Seed "
        f"`{plan['seed']}`. Same policy as targeted-value-01: must-review, sampled strata, a "
        "clean audit, and a predeclared escalation rule. Nothing here is a human label.",
        "",
        "## Counts",
        "",
        f"- in scope: {c['in_scope']} (clean rule-accepted {c['clean_auto_accepted']}, "
        f"flagged {c['flagged']})",
        f"- **review queue: {c['review_queue']}** = must-review {c['must_review']} + "
        f"stratum samples {c['hard_sample']} + clean audit {c['clean_audit']} "
        f"(was {o['total']}: {o['flagged']} flagged + {o['audit_sample']} audit)",
        f"- flagged notes outside the queue: {c['unsampled_flagged']} (rule proposal kept, status "
        "`pending-stratum-check`; trained only if their stratum passes)",
        f"- genuinely ambiguous (must-review): {c['genuinely_ambiguous']}. "
        f"{plan['genuinely_ambiguous']}",
        "",
        "## The original 167-note queue",
        "",
        f"Flagged by the proposer: {o['flagged']}; audit sample: {o['audit_sample']}.",
        "",
        *_table(
            ["primary reason", "records", "must-review", "sampled", "unsampled"],
            [
                [k, v["records"], v[rr.MUST], v[rr.HARD], v[rr.AUTO]]
                for k, v in o["primary_reason_to_new_group"].items()
            ],
        ),
        "",
        "By pattern (all flagged notes; a note counts once): "
        + ", ".join(f"`{k}` {v}" for k, v in o["flagged_by_pattern"].items()),
        "",
        "## Must-review (all reviewed)",
        "",
        "Reasons (a record may have several): "
        + ", ".join(f"`{k}` {v}" for k, v in plan["must_review_reason_counts"].items()),
        "",
        *_table(
            ["text", "reasons"],
            [[f"`{m['text']}`", ", ".join(m["reasons"])] for m in plan["must_review"]],
        ),
        "",
        "## Sampled strata",
        "",
        f"Sample size: {plan['sampling']['per_stratum']}. The population excludes must-review "
        "records.",
        "",
        *_table(
            ["stratum", "population", "sample", "unsampled", "sampled notes"],
            [
                [
                    f"`{k}`",
                    v["population"],
                    v["sample"],
                    v["unsampled"],
                    "; ".join(f"`{s['text']}`" for s in v["sampled"]),
                ]
                for k, v in plan["strata"].items()
            ],
        ),
        "",
        "## Clean audit",
        "",
        f"{plan['clean']['audit_sample']} of {plan['clean']['population']} clean records "
        f"({plan['sampling']['clean_audit']}):",
        "",
        *[f"- `{a['text']}`" for a in plan["clean"]["audit"]],
        "",
        "## Escalation rule (predeclared)",
        "",
        f"- Error: {rule['error']}.",
        f"- (a) {rule['sampled_stratum']}.",
        f"- (b) Clean audit: {rule['clean_audit']['accept']}; {rule['clean_audit']['expand']}.",
        f"- (c) Must-review: {rule['must_review']}.",
        f"- (d) Unsampled: {rule['unsampled']}.",
        "",
        "## Quet command",
        "",
        "```sh",
        *plan["quet_command"].splitlines(),
        "```",
        "",
        "`--out` is strict: labels must be ids of the queue. Proposals are advisory; only the",
        "reviewer's confirmed decision is saved. Afterwards:",
        "",
        "```sh",
        plan["score_command"],
        "```",
        "",
        "The scorer writes `experiments/value-span-v1/review-score.json` and, when the rule "
        f"escalates, `{OUT_DIR.as_posix()}/{ESCALATION_QUEUE}` with its Quet command. "
        "`scripts/build_training_v2.py` refuses to run without a fresh score.",
        "",
    ]
    return "\n".join(lines)


def build(root: Path, out_dir: Path) -> tuple[dict[str, bytes], dict[str, bytes], dict]:
    """``(dataset files, report files, manifest)``; file maps are ``name -> bytes``."""
    config = load_value_config(root / "configs" / "annotation-v2.yaml")
    scope, v1 = load_scope(root)
    proposals_by_id = {r["id"]: propose_value(r["text"]) for r in scope}

    proposals, auto_labels, errors = [], [], []
    for r in scope:
        p = proposals_by_id[r["id"]]
        proposal = to_quet_proposal(r["id"], p)
        problems = validate_quet_proposal(proposal, r["text"], config)
        errors += [f"{r['id']}: proposal: {m}" for m in problems]
        proposals.append(proposal)
        if p.auto_accept:
            label = to_rule_label(r["id"], p)
            problems = validate_value_label(label, r["text"], config, extra_fields=RULE_LABEL_EXTRA)
            errors += [f"{r['id']}: auto label: {m}" for m in problems]
            auto_labels.append(label)
    if errors:
        raise ValueError("invalid generated records:\n  " + "\n  ".join(errors))

    decisions = vr.plan_review(make_facts(scope, v1, proposals_by_id))
    order = rr.queue_order(decisions, vr.SEED)
    by_id = {r["id"]: r for r in scope}
    review = [
        {
            "id": i,
            "text": by_id[i]["text"],
            "review_group": decisions[i].group,
            "stratum": decisions[i].stratum,
            "reasons": list(decisions[i].reasons),
            "split": by_id[i]["split"],
            "source": by_id[i]["source"],
        }
        for i in order
    ]
    provenance = [
        {
            "id": r["id"],
            "review_group": decisions[r["id"]].group,
            "stratum": decisions[r["id"]].stratum,
            "provenance": rr.PROVENANCE_QUEUED if decisions[r["id"]].queued else vr.PROVENANCE_RULE,
            "status": vr.STATUS_QUEUED if decisions[r["id"]].queued else vr.STATUS_PENDING,
        }
        for r in scope
    ]
    plan = build_plan(scope, proposals_by_id, decisions, order, out_dir)

    contents = {
        FILES["queue"]: jsonl_bytes(scope),
        FILES["proposals"]: jsonl_bytes(proposals),
        FILES["auto_labels"]: jsonl_bytes(auto_labels),
        FILES["review_queue"]: jsonl_bytes(review),
        FILES["provenance"]: jsonl_bytes(provenance),
    }
    manifest = {
        "name": "value-queue-v1",
        "annotation_version": config.version,
        "rule_version": RULE_VERSION,
        "seed": config.seed,
        "review_seed": vr.SEED,
        "inputs": {str(path): sha256_file(root / path) for path in INPUTS.values()}
        | {str(PROBE_LABELS): sha256_file(root / PROBE_LABELS)},
        "counts": {
            "in_scope": len(scope),
            "by_split": dict(Counter(r["split"] for r in scope)),
            "by_source": dict(Counter(r["source"] for r in scope)),
            "by_source_batch": dict(Counter(r["source_batch"] for r in scope)),
            "auto_accepted": len(auto_labels),
            "needs_review": plan["counts"]["flagged"],
            "review_queue": len(review),
            "review_queue_by_group": dict(Counter(e["review_group"] for e in review)),
            "review_queue_by_stratum": dict(Counter(e["stratum"] for e in review)),
            "unsampled_flagged": plan["counts"]["unsampled_flagged"],
            "proposal_types": dict(Counter(p["type"] for p in proposals)),
            "proposal_statuses": dict(Counter(p["annotation_status"] for p in proposals)),
        },
        "review": {
            "name": vr.NAME,
            "strata": {
                s: {k: v for k, v in t.items() if k != "sampled"} for s, t in plan["strata"].items()
            }
            | {rr.CLEAN_STRATUM: {"population": plan["clean"]["population"]}},
            "audit_sample": plan["clean"]["audit_sample"],
            "sampling": plan["sampling"],
            "escalation_rule": vr.ESCALATION_RULE,
            "unsampled_status": vr.STATUS_PENDING,
            "report": str(REPORT_DIR / f"{REPORT_STEM}.md"),
        },
        "human_labels": f"{HUMAN_LABELS} is written only by Quet and is never touched here",
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()},
    }
    contents[FILES["manifest"]] = json_bytes(manifest)
    reports = {
        f"{REPORT_STEM}.json": json_bytes(plan),
        f"{REPORT_STEM}.md": render_markdown(plan).encode("utf-8"),
    }
    return contents, reports, manifest


def orphaned_labels(out_dir: Path, review_ids: set[str]) -> list[str]:
    path = out_dir / HUMAN_LABELS
    if not path.exists():
        return []
    return sorted(r["id"] for r in read_jsonl(path) if r["id"] not in review_ids)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=None, help=f"default: {OUT_DIR}")
    parser.add_argument("--report-dir", type=Path, default=None, help=f"default: {REPORT_DIR}")
    parser.add_argument("--force", action="store_true", help="replace existing outputs")
    parser.add_argument("--check", action="store_true", help="verify the files on disk")
    args = parser.parse_args(argv)

    root = args.root
    out_dir = args.out_dir or root / OUT_DIR
    report_dir = args.report_dir or root / REPORT_DIR
    dataset_dir = OUT_DIR if args.out_dir is None else args.out_dir
    contents, reports, manifest = build(root, dataset_dir)
    targets = [(out_dir / n, d) for n, d in contents.items()]
    targets += [(report_dir / n, d) for n, d in reports.items()]

    if args.check:
        bad = []
        for path, data in targets:
            if not path.exists():
                bad.append(f"{path.name}: missing")
            elif path.read_bytes() != data:
                bad.append(f"{path.name}: differs from a fresh build")
        if bad:
            print("CHECK FAILED", *bad, sep="\n  ", file=sys.stderr)
            return 1
        print(f"ok: {len(targets)} files reproduce ({manifest['counts']['in_scope']} records)")
        return 0

    existing = [p.name for p, _ in targets if p.exists()]
    if existing and not args.force:
        print(f"refusing to overwrite {existing}; pass --force", file=sys.stderr)
        return 1
    review_ids = {json.loads(line)["id"] for line in contents[FILES["review_queue"]].splitlines()}
    orphans = orphaned_labels(out_dir, review_ids)
    if orphans:
        print(
            f"refusing: {HUMAN_LABELS} labels {len(orphans)} id(s) outside the new review queue "
            f"(e.g. {orphans[:3]}); Quet --out would reject them",
            file=sys.stderr,
        )
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    for path, data in targets:
        path.write_bytes(data)
    counts = manifest["counts"]
    print(
        f"wrote {len(targets)} files: {counts['in_scope']} records, {counts['auto_accepted']} "
        f"clean auto-accepted, review queue {counts['review_queue']}"
    )
    print("review queue by group:", counts["review_queue_by_group"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
