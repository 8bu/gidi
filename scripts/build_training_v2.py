#!/usr/bin/env python
"""Build annotation-v2 training records: the v1 splits plus value-span fields.

Usage:
    uv run python scripts/build_training_v2.py [--root R] [--out-dir D] [--dry-run] [--check]
                                               [--extra DIR [--extra-name NAME]]

Reads, from ``datasets/annotation-v2/``, the value pass outputs of ``build_value_queue.py``:
``value-queue.jsonl`` (texts), ``value-auto-labels.jsonl`` (clean rule labels),
``value-proposals.jsonl``, ``value-review-queue.jsonl``, ``value-review-provenance.jsonl`` and,
when it exists, ``value-labels.jsonl`` (the human Quet output). It refuses to run when the human
labels fail validation (span mechanics, ``uncertain`` without a note, ``no_amount`` with a span,
ids outside the value queue, duplicates), and when the review has no fresh score: it needs
``experiments/value-span-v1/review-score.json`` (``scripts/score_value_review.py``) whose inputs
(queue, provenance, proposals, human labels) are unchanged since scoring.

Each v1 record keeps every field and gains

    "value": {"text", "start", "end"} | null     code-point offsets, text[start:end] == text
    "value_status": "complete" | "uncertain"
    "value_provenance": "human" | "rule"

Per record the value comes from the human label when there is one (it always beats a rule label).
Otherwise from the rule label, but only if the record's stratum passed the score: clean notes
follow the clean audit and the flagged notes outside the review queue follow their pattern stratum
(their rule label is the proposal itself, never a human label). A record whose stratum has not
passed ("unverified") or that still awaits its review is ``value=null, value_status="uncertain"``.
``uncertain`` records still train type and target; only the value loss is masked for them. A human
``no_amount`` is ``value=null, complete`` (all ``O``).

Output (``datasets/annotation-v2/training-v1/``) mirrors ``datasets/annotation-v1/distillation-v1``,
the data compression-v3 trained on: ``train.jsonl`` (723 notes, including the 110 validation
notes, in-sample), ``validation.jsonl`` and ``test.jsonl`` with the *same ids per split and in the
same order as v1*, plus ``probe-v1-eval-only.jsonl`` (the 81 probe notes with value labels, for
evaluation; never train on it) and ``manifest.json``.

``--extra DIR`` adds a later batch reviewed by risk (e.g.
``datasets/annotation-v2/targeted-value-01``, built by ``build_targeted_value_queue.py``). ``DIR``
holds ``generation-notes.jsonl`` (``id, text, group``), the small Quet queue
``review-queue.jsonl``, ``review-provenance.jsonl``, the auto-accepted ``auto-labels.jsonl``
(type/target) and ``value-auto-labels.jsonl``, and, once Quet wrote them, the human
``labels.jsonl`` and ``value-labels.jsonl``. The batch needs a fresh score entry named after
``DIR`` too. Per record: both human labels win over the auto labels (annotator ``human``,
``value_provenance`` human); otherwise an auto-accepted record trains from its auto labels
(annotator ``synthetic-auto``, ``value_provenance`` rule) if its stratum passed the score, else it
is EXCLUDED and counted; a queued record without both human labels is EXCLUDED from training (not
masked) and counted in ``extra.review``. Complete type labels only. Groups are split by the seed
``value-span-v1`` over every group of the batch (so the assignment does not move as review labels
arrive; never across splits) into train/validation/test (``--extra-fractions``), appended after
the v1 rows, and leakage-checked with ``gidi.annotation.leakage`` against the frozen
validation/test splits and probes. Filter by ``source_batch`` to evaluate on the frozen v1 rows
alone.

``--dry-run`` prints the manifest and writes nothing; ``--check`` verifies the files on disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Collection
from pathlib import Path

from gidi.annotation import review_gate as gate
from gidi.annotation import risk_review as rr
from gidi.annotation import value_review as vr
from gidi.annotation.leakage import EVAL_REJECTS, check, load_references
from gidi.annotation.schema import load_config, validate_file
from gidi.annotation.split import accented
from gidi.annotation.value_span import (
    RULE_LABEL_EXTRA,
    SEED,
    ValueConfig,
    load_value_config,
    merged_value_fields,
    span_shape_warnings,
    validate_label_file,
    validate_merged_record,
    validate_value_label,
)
from gidi.corpus.jsonl import read_jsonl

V2 = Path("datasets/annotation-v2")
V1_DIR = Path("datasets/annotation-v1/distillation-v1")
PROBE_DIR = Path("datasets/probe-v1")
OUT_DIR = V2 / "training-v1"
SPLITS = ("train", "validation", "test")
PROBE_OUT = "probe-v1-eval-only.jsonl"
FAIL_ON = EVAL_REJECTS | {"train_duplicate", "batch_duplicate", "batch_near_other_group"}
EXTRA_REQUIRED = (
    "generation-notes.jsonl",
    "review-queue.jsonl",
    "review-provenance.jsonl",
    "auto-labels.jsonl",
    "value-auto-labels.jsonl",
)
EXTRA_FILES = (*EXTRA_REQUIRED, "labels.jsonl", "value-labels.jsonl")
EXTRA_COUNTS = (
    "human",
    "human_overrides_auto",
    "auto_accepted",
    "awaiting_review_excluded",
    "partial_human_labels_excluded",
    "not_complete_skipped",
    "stratum_unverified_excluded",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonl_bytes(rows: list[dict]) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")


def fail(messages: list[str], title: str) -> None:
    print(title, *messages[:40], sep="\n  ", file=sys.stderr)
    if len(messages) > 40:
        print(f"  ... and {len(messages) - 40} more", file=sys.stderr)
    raise SystemExit(1)


class ValueSources:
    """Human, rule and review state for the in-scope value pass.

    Rule labels (clean notes, and flagged notes outside the review queue, whose label is their
    proposal) are trusted only in the strata a fresh score marks ``passed``. A record in another
    stratum is "unverified": it gets no rule label and is masked.
    """

    def __init__(self, root: Path, config: ValueConfig):
        v2 = root / V2
        try:
            trusted = gate.load_gate(root, vr.NAME)
        except gate.GateError as exc:
            fail([str(exc)], "the existing-data value review is not scored:")
        queue = read_jsonl(v2 / "value-queue.jsonl")
        self.texts = {r["id"]: r["text"] for r in queue}
        self.provenance = {r["id"]: r for r in read_jsonl(v2 / "value-review-provenance.jsonl")}
        self.review = {r["id"]: r for r in read_jsonl(v2 / "value-review-queue.jsonl")}
        self.trusted = trusted
        clean = validate_label_file(
            v2 / "value-auto-labels.jsonl", self.texts, config, extra_fields=RULE_LABEL_EXTRA
        )
        if not clean.ok:
            fail(clean.errors, "invalid rule labels (value-auto-labels.jsonl):")
        proposals = {r["id"]: r for r in read_jsonl(v2 / "value-proposals.jsonl")}
        self.rule: dict[str, dict] = {}
        self.unverified: dict[str, str] = {}  # id -> stratum that has not passed
        problems = []
        for rid, prov in self.provenance.items():
            if prov["review_group"] != rr.AUTO:
                continue
            if prov["stratum"] not in trusted:
                self.unverified[rid] = prov["stratum"]
                continue
            label = clean.labels.get(rid) or vr.rule_label_row(proposals[rid])
            bad = validate_value_label(
                label, self.texts[rid], config, extra_fields=RULE_LABEL_EXTRA
            )
            problems += [f"{rid}: {m}" for m in bad]
            self.rule[rid] = label
        if problems:
            fail(problems, "invalid rule labels of the flagged records outside the queue:")
        self.human: dict[str, dict] = {}
        self.human_path = v2 / "value-labels.jsonl"
        if self.human_path.exists():
            human = validate_label_file(self.human_path, self.texts, config)
            if not human.ok:
                fail(human.errors, f"invalid human value labels ({self.human_path}):")
            self.human = human.labels

    def fields(self, record: dict) -> dict:
        """Value fields for one v1 record, with a ``source`` tag for counting."""
        rid = record["id"]
        if self.texts.get(rid) != record["text"] or rid not in self.provenance:
            fail([f"{rid}: not in the value queue or its text differs"], "value queue mismatch:")
        value, status, provenance, source = merged_value_fields(
            self.human.get(rid), self.rule.get(rid)
        )
        if source == "unlabelled" and rid in self.unverified:
            source = "unverified"
        return {
            "value": value,
            "value_status": status,
            "value_provenance": provenance,
            "_source": source,
        }


def merge(records: list[dict], sources: ValueSources, config: ValueConfig) -> list[dict]:
    merged, errors = [], []
    for record in records:
        fields = sources.fields(record)
        row = {**record, **{k: v for k, v in fields.items() if k != "_source"}}
        row["_source"] = fields["_source"]
        problems = validate_merged_record(row, config)
        errors += [f"{record['id']}: {p}" for p in problems]
        merged.append(row)
    if errors:
        fail(errors, "invalid merged records:")
    return merged


def strip_private(rows: list[dict]) -> list[dict]:
    return [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]


def probe_records(root: Path) -> list[dict]:
    labels = {r["id"]: r for r in read_jsonl(root / PROBE_DIR / "labels.jsonl")}
    prov = {r["id"]: r for r in read_jsonl(root / PROBE_DIR / "provenance.jsonl")}
    out = []
    for note in read_jsonl(root / PROBE_DIR / "notes.jsonl"):
        lab = labels[note["id"]]
        if lab["annotation_status"] != "complete":
            continue
        out.append(
            {
                "id": note["id"],
                "text": note["text"],
                "type": lab["type"],
                "target": lab["target"],
                "source_batch": "probe-v1",
                "accented": accented(note["text"]),
                "provenance": {
                    "annotator": prov.get(note["id"], {}).get("annotator", "ai"),
                    "source_batch": "probe-v1",
                },
            }
        )
    return out


def split_groups(groups: list[str], fractions: tuple[float, float, float]) -> dict[str, str]:
    """Assign whole groups to splits: ordered by sha256(seed:group), test first, then validation."""

    def key(group: str) -> str:
        return hashlib.sha256(f"{SEED}:split:{group}".encode()).hexdigest()

    ordered = sorted(set(groups), key=key)
    n = len(ordered)
    n_val = round(n * fractions[1])
    n_test = round(n * fractions[2])
    if n >= 3:
        n_val, n_test = max(n_val, 1), max(n_test, 1)
    assignment = {}
    for i, g in enumerate(ordered):
        assignment[g] = "test" if i < n_test else "validation" if i < n_test + n_val else "train"
    return assignment


def extra_records(
    root: Path,
    directory: Path,
    name: str,
    config: ValueConfig,
    v1_ids: set[str],
    trusted: Collection[str],
) -> tuple[list[dict], list[str], dict]:
    """Merged records of a risk-reviewed extra batch, all its groups, and the review counts.

    Per record: both human labels win; otherwise an auto-accepted record uses its auto labels
    (``synthetic-auto``) if its stratum is in ``trusted`` (passed the score) and is excluded and
    counted if not; a queued record without both human labels is excluded and counted.
    """
    missing = [n for n in EXTRA_REQUIRED if not (directory / n).exists()]
    if missing:
        fail([f"{directory / n}: missing" for n in missing], "incomplete extra batch:")
    notes = {r["id"]: r for r in read_jsonl(directory / "generation-notes.jsonl")}
    texts = {i: n["text"] for i, n in notes.items()}
    queued = {r["id"] for r in read_jsonl(directory / "review-queue.jsonl")}
    prov = {r["id"]: r for r in read_jsonl(directory / "review-provenance.jsonl")}
    problems = [
        f"{i}: queued id is not in generation-notes.jsonl" for i in sorted(queued - set(notes))
    ]

    type_config = load_config(root / "configs" / "annotation-v2.yaml")

    def type_labels(file: str) -> dict[str, dict]:
        path = directory / file
        if not path.exists():
            return {}
        report = validate_file(path, texts, type_config)
        if not report.ok:
            fail([f"{e.line}: {e.id}: {e.message}" for e in report.errors], f"invalid {path}:")
        return {r["id"]: r for r in read_jsonl(path)}

    def value_labels(file: str, extra: tuple[str, ...] = ()) -> dict[str, dict]:
        path = directory / file
        if not path.exists():
            return {}
        report = validate_label_file(path, texts, config, extra_fields=extra)
        if not report.ok:
            fail(report.errors, f"invalid {path}:")
        return report.labels

    human_t, human_v = type_labels("labels.jsonl"), value_labels("value-labels.jsonl")
    auto_t = type_labels("auto-labels.jsonl")
    auto_v = value_labels("value-auto-labels.jsonl", RULE_LABEL_EXTRA)

    counts: Counter[str] = Counter()
    records = []
    for rid, note in notes.items():
        ht, hv = human_t.get(rid), human_v.get(rid)
        if rid in v1_ids or rid.startswith("probe-"):
            problems.append(f"{rid}: id collides with a v1 or probe record")
            continue
        if (ht is None) != (hv is None):
            if rid in queued:
                counts["partial_human_labels_excluded"] += 1
            else:
                problems.append(f"{rid}: only one human label pass for a non-queued record")
            continue
        if ht is not None:
            lab, annotator, human_value, rule_value = ht, "human", hv, None
            counts["human"] += 1
            if rid not in queued:
                counts["human_overrides_auto"] += 1
        elif rid in queued:
            counts["awaiting_review_excluded"] += 1
            continue
        else:
            lab, annotator, human_value = auto_t.get(rid), rr.PROVENANCE_AUTO, None
            rule_value = auto_v.get(rid)
            if lab is None or rule_value is None:
                problems.append(f"{rid}: not queued and has no auto labels")
                continue
            if prov.get(rid, {}).get("provenance") != rr.PROVENANCE_AUTO:
                problems.append(f"{rid}: auto label without {rr.PROVENANCE_AUTO!r} provenance")
                continue
            if rule_value["type"] != "amount" or rule_value["target"] is None:
                problems.append(f"{rid}: auto value label is not a single amount span")
                continue
            if prov[rid].get("stratum") not in trusted:
                counts["stratum_unverified_excluded"] += 1
                continue
            counts["auto_accepted"] += 1
        if lab["annotation_status"] != "complete":
            counts["not_complete_skipped"] += 1
            continue
        value, status, provenance, _ = merged_value_fields(human_value, rule_value)
        record = {
            "id": rid,
            "text": note["text"],
            "type": lab["type"],
            "target": lab["target"],
            "source_batch": name,
            "accented": accented(note["text"]),
            "provenance": {"annotator": annotator, "source_batch": name},
            "group": note.get("group", rid),
            "value": value,
            "value_status": status,
            "value_provenance": provenance,
            "_source": "extra-human" if annotator == "human" else "extra-auto",
        }
        problems += [f"{rid}: {p}" for p in validate_merged_record(record, config)]
        records.append(record)
    if problems:
        fail(problems, f"invalid extra batch {directory}:")
    review = {
        "queued": len(queued),
        "trained": len(records),
        **{k: counts.get(k, 0) for k in EXTRA_COUNTS},
    }
    return records, [n.get("group", i) for i, n in notes.items()], review


def extra_leakage(root: Path, records: list[dict]) -> dict:
    eval_refs, train_refs = load_references(root)
    candidates = [
        {
            "text": r["text"],
            "intended_target": r["target"]["text"] if r["target"] else None,
            "group": r["group"],
        }
        for r in records
    ]
    results = check(candidates, eval_refs, train_refs)
    errors = []
    for rec, res in zip(records, results, strict=True):
        bad = sorted(set(res["reject"]) & FAIL_ON)
        if bad:
            errors.append(f"{rec['id']} {rec['text']!r}: {', '.join(bad)} (eval {res['eval_sim']})")
    if errors:
        fail(errors, "LEAKAGE in the extra batch (nothing built):")
    sims = sorted(r["eval_sim"] for r in results) or [0.0]
    return {
        "checked": len(records),
        "eval_refs": len(eval_refs),
        "fail_on": sorted(FAIL_ON),
        "failures": 0,
        "warnings": dict(Counter(w for r in results for w in r["warn"] + r["reject"])),
        "max_eval_similarity": sims[-1],
        "median_eval_similarity": sims[len(sims) // 2],
    }


def value_stats(rows: list[dict]) -> dict:
    return {
        "records": len(rows),
        "with_value_span": sum(r["value"] is not None for r in rows),
        "no_amount_complete": sum(
            r["value"] is None and r["value_status"] == "complete" for r in rows
        ),
        "masked_uncertain": sum(r["value_status"] != "complete" for r in rows),
        "by_provenance": dict(Counter(r["value_provenance"] for r in rows)),
        "by_source": dict(Counter(r.get("_source", "extra-human") for r in rows)),
    }


def build(root: Path, args: argparse.Namespace) -> tuple[dict[str, bytes], dict]:
    config = load_value_config(root / "configs" / "annotation-v2.yaml")
    sources = ValueSources(root, config)
    v1 = {s: read_jsonl(root / V1_DIR / f"{s}.jsonl") for s in SPLITS}
    merged = {s: merge(v1[s], sources, config) for s in SPLITS}
    for s in SPLITS:
        if [r["id"] for r in merged[s]] != [r["id"] for r in v1[s]]:
            fail([f"{s}: id order changed"], "v1 split drift:")
    probe = merge(probe_records(root), sources, config)

    extra_info: dict = {}
    extra_rows: dict[str, list[dict]] = {s: [] for s in SPLITS}
    if args.extra:
        directory = args.extra if args.extra.is_absolute() else root / args.extra
        name = args.extra_name or args.extra.name
        v1_ids = {r["id"] for rows in v1.values() for r in rows}
        try:
            extra_trusted = gate.load_gate(root, args.extra.name)
        except gate.GateError as exc:
            fail([str(exc)], f"the extra batch {args.extra.name} is not scored:")
        records, all_groups, review = extra_records(
            root, directory, name, config, v1_ids, extra_trusted
        )
        fractions = tuple(float(x) for x in args.extra_fractions.split(","))
        if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-6:
            fail(["--extra-fractions must be three numbers summing to 1"], "bad arguments:")
        # Over every group of the batch, so the split does not move as review labels arrive.
        assignment = split_groups(all_groups, fractions)
        leakage = extra_leakage(root, records)
        for r in records:
            extra_rows[assignment[r["group"]]].append(r)
        extra_info = {
            "name": name,
            "directory": str(args.extra),
            "records": len(records),
            "groups": len({r["group"] for r in records}),
            "groups_in_batch": len(assignment),
            "split_seed": SEED,
            "fractions": list(fractions),
            "per_split": {s: len(extra_rows[s]) for s in SPLITS},
            "groups_per_split": {s: len({r["group"] for r in extra_rows[s]}) for s in SPLITS},
            "review": review,
            "leakage_gate": leakage,
            "files": {
                n: sha256_file(directory / n) for n in EXTRA_FILES if (directory / n).exists()
            },
        }

    out_rows = {s: strip_private(merged[s] + extra_rows[s]) for s in SPLITS}
    contents = {f"{s}.jsonl": jsonl_bytes(out_rows[s]) for s in SPLITS}
    contents[PROBE_OUT] = jsonl_bytes(strip_private(probe))

    v2 = root / V2
    awaiting = sorted(
        r["id"] for rows in (*merged.values(), probe) for r in rows if r["_source"] == "unlabelled"
    )
    unverified = sorted(
        r["id"] for rows in (*merged.values(), probe) for r in rows if r["_source"] == "unverified"
    )
    shape = [
        f"{r['id']}: {r['value']['text']!r}: {w}"
        for rows in (*merged.values(), probe, *extra_rows.values())
        for r in rows
        if r["value"] is not None
        for w in span_shape_warnings(r["value"]["text"])
    ]
    changed = [
        r["id"]
        for rows in (*merged.values(), probe)
        for r in rows
        if r["_source"] == "human"
        and r["id"] in sources.rule
        and (r["value"] or {}).get("text") != sources.rule[r["id"]]["target"]["text"]
    ]
    manifest = {
        "name": "training-v1",
        "annotation_version": config.version,
        "purpose": (
            "annotation-v1 records of the distillation-v1 layout (the data compression-v3 trained "
            "on) plus value-span fields; train.jsonl contains the validation ids (in-sample, as "
            "in v1)"
        ),
        "counts": {s: len(out_rows[s]) for s in SPLITS} | {"probe_eval_only": len(probe)},
        "v1_counts": {s: len(v1[s]) for s in SPLITS},
        "value": {s: value_stats(merged[s] + extra_rows[s]) for s in SPLITS}
        | {"probe_eval_only": value_stats(probe)},
        "human_value_labels": {
            "file": str(V2 / "value-labels.jsonl"),
            "present": sources.human_path.exists(),
            "labels": len(sources.human),
            "human_differs_from_rule": len(changed),
        },
        "awaiting_human_review_masked": {
            "count": len(awaiting),
            "note": "needs review, no human label yet: value_status uncertain, value loss masked",
        },
        "stratum_unverified_masked": {
            "count": len(unverified),
            "by_stratum": dict(Counter(sources.unverified[i] for i in unverified)),
            "note": (
                "rule label withheld: the record's stratum has not passed the score; value_status "
                "uncertain, value loss masked"
            ),
        },
        "value_review": {
            "name": vr.NAME,
            "score": str(gate.SCORE_PATH),
            "trusted_strata": sorted(sources.trusted),
            "review_queue": len(sources.review),
            "records_by_group": dict(
                Counter(p["review_group"] for p in sources.provenance.values())
            ),
        },
        "span_shape_warnings": shape,
        "extra": extra_info,
        "evaluation_only": [PROBE_OUT],
        "inputs": {
            str(p): sha256_file(root / p)
            for p in (
                V1_DIR / "train.jsonl",
                V1_DIR / "validation.jsonl",
                V1_DIR / "test.jsonl",
                V2 / "value-queue.jsonl",
                V2 / "value-proposals.jsonl",
                V2 / "value-auto-labels.jsonl",
                V2 / "value-review-queue.jsonl",
                V2 / "value-review-provenance.jsonl",
                PROBE_DIR / "labels.jsonl",
                gate.SCORE_PATH,
            )
        }
        | (
            {str(V2 / "value-labels.jsonl"): sha256_file(v2 / "value-labels.jsonl")}
            if sources.human_path.exists()
            else {}
        ),
        "files": {n: hashlib.sha256(d).hexdigest() for n, d in contents.items()},
    }
    contents["manifest.json"] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    return contents, manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=None, help=f"default: {OUT_DIR}")
    parser.add_argument("--extra", type=Path, default=None, help="extra batch directory")
    parser.add_argument("--extra-name", default=None, help="source_batch (default: dir name)")
    parser.add_argument("--extra-fractions", default="0.7,0.15,0.15", help="train,validation,test")
    parser.add_argument("--dry-run", action="store_true", help="print the manifest, write nothing")
    parser.add_argument("--check", action="store_true", help="verify the files on disk")
    parser.add_argument("--force", action="store_true", help="replace existing outputs")
    args = parser.parse_args(argv)

    root = args.root
    out_dir = args.out_dir or root / OUT_DIR
    contents, manifest = build(root, args)

    if args.dry_run:
        print(
            json.dumps(
                {k: v for k, v in manifest.items() if k != "span_shape_warnings"},
                ensure_ascii=False,
                indent=2,
            )
        )
        print(f"span shape warnings: {len(manifest['span_shape_warnings'])}")
        return 0
    if args.check:
        bad = [
            f"{n}: {'missing' if not (out_dir / n).exists() else 'differs from a fresh build'}"
            for n, data in contents.items()
            if not (out_dir / n).exists() or (out_dir / n).read_bytes() != data
        ]
        if bad:
            fail(bad, "CHECK FAILED")
        print(f"ok: {len(contents)} files reproduce")
        return 0
    existing = [n for n in contents if (out_dir / n).exists()]
    if existing and not args.force:
        print(f"refusing to overwrite {existing}; pass --force", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in contents.items():
        (out_dir / name).write_bytes(data)
    print(f"wrote {len(contents)} files to {out_dir}: {manifest['counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
