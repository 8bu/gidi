#!/usr/bin/env python
"""Build ``datasets/annotation-v3/training-v5``: training-v4 + the gift relabel map + contrast-04.

Usage:
    uv run python scripts/build_annotation_v3_training_v5.py [--root R] [--check]

``train.jsonl`` = the 905 byte-identical ``training-v2`` records + the rest of ``training-v4``
(95 contrast-02 notes, 83 contrast-03 notes) with the map
``datasets/annotation-v3/gift-relabel-01/map.jsonl`` applied (user decision "a gift receiver is
the target": four LLM contrast-03 notes get the named receiver as target, one gift note with a
second plausible counterparty is dropped; every map row
is checked against the record it changes, and the relabelled records carry
``provenance.relabeled = "gift-relabel-01"``) + the ``complete`` notes of
``datasets/annotation-v3/contrast-04`` (the shop names the contrast-03 quota dropped, gift notes
with the receiver as target). ``validation.jsonl`` and ``test.jsonl`` are symlinks to the frozen
annotation-v1 splits, as in training-v2 .. v4.

Leakage gate (raises, writes nothing) against the frozen test split, probe-v1,
``datasets/annotation-v2/human-value-01`` and ``datasets/annotation-v3/human-value-02`` (batches a
and b, 200 notes): no id and no exact text (verbatim or NFC + strip + lowercase) of any train
record in a set; no non-base train record a near duplicate (the rules of
``scripts/build_human_value_queue.py``) of any note of a set; no non-base record at folded
char-3-gram Jaccard >= ``HV_CHAR3_LIMIT`` to a human-value-01 or human-value-02 note; no base note a
near duplicate of a human-value-02 note. human-value-02 is a test set: its texts are only scored
by this gate (never printed or stored); the manifest keeps the file hashes. ``--check`` rebuilds
in memory and compares the files on disk.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


v1 = _load("build_annotation_v3_training")
v2 = _load("build_annotation_v3_training_v2")
v4 = _load("build_annotation_v3_training_v4")
c1 = _load("build_contrast_01")
c4 = _load("build_contrast_04")
hv = v2.hv

NAME = "training-v5"
OUT = Path("datasets/annotation-v3/training-v5")
BASE = Path("datasets/annotation-v3/training-v2")
PREV = Path("datasets/annotation-v3/training-v4")
CONTRAST_04 = Path("datasets/annotation-v3/contrast-04")
RELABEL = Path("datasets/annotation-v3/gift-relabel-01")
HV02_QUEUES = c4.HV02_QUEUES
BASE_RECORDS = 905
PREV_RECORDS = 1083
HV_CHAR3_LIMIT = 0.6
INPUTS = (
    BASE / "train.jsonl",
    BASE / "manifest.json",
    PREV / "train.jsonl",
    PREV / "manifest.json",
    RELABEL / "map.jsonl",
    RELABEL / "manifest.json",
    CONTRAST_04 / "queue.jsonl",
    CONTRAST_04 / "labels.jsonl",
    CONTRAST_04 / "provenance.jsonl",
    CONTRAST_04 / "manifest.json",
    v1.DEBT / "manifest.json",
    v1.HUMAN / "labels.jsonl",
    v1.HUMAN / "review-queue.jsonl",
    v1.HUMAN / "freeze.json",
    v1.PROBE / "notes.jsonl",
    *HV02_QUEUES,
)


def contrast_04_records(root: Path) -> list[dict[str, Any]]:
    """The ``complete`` contrast-04 labels as training records, in labels.jsonl order."""
    text_of = {r["id"]: r["text"] for r in v1.read_jsonl(root / CONTRAST_04 / "queue.jsonl")}
    annotator = {
        r["id"]: r["annotator"] for r in v1.read_jsonl(root / CONTRAST_04 / "provenance.jsonl")
    }
    records = []
    for label in v1.read_jsonl(root / CONTRAST_04 / "labels.jsonl"):
        if label["annotation_status"] != "complete":
            continue
        rid, text = label["id"], text_of[label["id"]]
        if label["type"] not in v1.TYPES:
            raise ValueError(f"{rid}: contrast-04 label has type {label['type']!r}")
        target = label["target"]
        if target is not None:
            target = {k: target[k] for k in ("text", "start", "end")}
            if text[target["start"] : target["end"]] != target["text"]:
                raise ValueError(f"{rid}: target span does not slice the note text")
        records.append(
            {
                "id": rid,
                "text": text,
                "type": label["type"],
                "target": target,
                "source_batch": "contrast-04",
                "accented": v1.accented(text),
                "provenance": {"annotator": annotator[rid], "source_batch": "contrast-04"},
                "group": f"contrast-04:{rid}",
            }
        )
    return records


def apply_relabel(
    records: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply the gift relabel map; returns ``(records, dropped records)``. Verifies every row."""
    by_id = {r["id"]: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError("gift-relabel-01 map: duplicate ids")
    seen: set[str] = set()
    out, dropped = [], []
    for record in records:
        row = by_id.get(record["id"])
        if row is None:
            out.append(record)
            continue
        seen.add(record["id"])
        old = record["target"]["text"] if record["target"] else None
        if row["text"] != record["text"] or row["old"] != old:
            raise ValueError(f"{record['id']}: map row does not match the training record")
        if record["provenance"]["annotator"] != "llm":
            raise ValueError(f"{record['id']}: only LLM notes are relabelled")
        if row["action"] == "drop":
            dropped.append(record)
        elif row["action"] == "relabel" and row["new"] is not None:
            target = c1.span(record["text"], row["new"], "target")
            out.append(
                {
                    **record,
                    "target": target,
                    "provenance": {**record["provenance"], "relabeled": "gift-relabel-01"},
                }
            )
        else:
            raise ValueError(f"{record['id']}: bad map row {row!r}")
    if seen != set(by_id):
        raise ValueError(f"map ids not in training-v4: {sorted(set(by_id) - seen)}")
    return out, dropped


def build(root: Path) -> tuple[bytes, dict[str, Any]]:
    """Verify every precondition and return ``(train.jsonl bytes, manifest)``; writes nothing."""
    for rel in HV02_QUEUES:
        if not (root / rel).is_file():
            raise ValueError(f"{rel} is missing: the leakage gate needs human-value-02")
    base_bytes = (root / BASE / "train.jsonl").read_bytes()
    prev_bytes = (root / PREV / "train.jsonl").read_bytes()
    if not prev_bytes.startswith(base_bytes):
        raise ValueError("training-v4 must start with the training-v2 train records")
    prev = [json.loads(line) for line in prev_bytes.decode("utf-8").splitlines()]
    base = prev[:BASE_RECORDS]
    if len(prev) != PREV_RECORDS or len(base_bytes.splitlines()) != BASE_RECORDS:
        raise ValueError("unexpected training-v2 / training-v4 sizes")
    kept_v4 = prev[BASE_RECORDS:]
    rows = v1.read_jsonl(root / RELABEL / "map.jsonl")
    relabelled, map_dropped = apply_relabel(kept_v4, rows)
    contrast_04 = contrast_04_records(root)
    if not contrast_04:
        raise ValueError("contrast-04 has no complete note")
    excluded = json.loads((root / v1.DEBT / "manifest.json").read_text("utf-8"))[
        "excluded_human_value_01_ids"
    ]

    sets = v1.held_out(root)
    sets["human-value-02"] = [r for rel in HV02_QUEUES for r in v1.read_jsonl(root / rel)]
    if len(sets["human-value-02"]) < 190:
        raise ValueError("human-value-02 queues look truncated")

    # human-value-02 batch b (50 notes) was written after training-v4: a note inherited from
    # training-v4 that is a near duplicate of, or at char-3 >= the limit to, a human-value-02 note
    # leaves the training set (decided by the gate score only; the test note is not read). The new
    # contrast-04 notes were gated against both batches when they were built and must pass.
    hv02_set = hv.ReferenceSet([r["text"] for r in sets["human-value-02"]])
    hv02_refs = [hv.Reference(r["text"]) for r in sets["human-value-02"]]
    inherited, removed = [], []
    for record in relabelled:
        if hv02_set.match(hv.Reference(record["text"])):
            reason = "near-duplicate-human-value-02"
        elif v4.max_char3(record, hv02_refs) >= HV_CHAR3_LIMIT:
            reason = "char3-human-value-02"
        else:
            inherited.append(record)
            continue
        removed.append(
            {
                "id": record["id"],
                "text": record["text"],
                "type": record["type"],
                "target": record["target"]["text"] if record["target"] else None,
                "reason": reason,
            }
        )
    new = inherited + contrast_04
    records = base + new
    v1.assert_no_leakage(records, sets, excluded)
    sizes = v2.assert_no_near_duplicates(new, sets)
    v2.assert_no_near_duplicates(base, {"human-value-02": sets["human-value-02"]})
    base_near_dup = {}
    for name, rows_ in sets.items():
        ref_set = hv.ReferenceSet([r["text"] for r in rows_])
        base_near_dup[name] = sum(bool(ref_set.match(hv.Reference(r["text"]))) for r in base)
    char3, base_char3 = {}, {}
    for name in ("human-value-01", "human-value-02"):
        refs = [hv.Reference(r["text"]) for r in sets[name]]
        scores = {r["id"]: v4.max_char3(r, refs) for r in new}
        over = [f"{i}: {s:.2f}" for i, s in scores.items() if s >= HV_CHAR3_LIMIT]
        if over:
            raise ValueError(f"{name} char3 bound: " + "; ".join(over))
        char3[name] = round(max(scores.values()), 4)
        base_char3[name] = sum(v4.max_char3(r, refs) >= HV_CHAR3_LIMIT for r in base)

    body = base_bytes + b"".join(
        (json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8") for r in new
    )
    relabel_counts = Counter(r["action"] for r in rows)
    manifest = {
        "name": NAME,
        "annotation_version": "annotation-v3",
        "purpose": (
            "encoder-1 retrain set v5: training-v4 with the gift-receiver relabel map applied "
            "(user decision: a gift receiver is the target) + contrast-04 (shop names the "
            "contrast-03 quota dropped, gift notes with the receiver as target); validation is "
            "in-sample, the frozen test split, probe-v1, human-value-01 (amendment-01) and "
            "human-value-02 stay held out"
        ),
        "counts": {
            "training_v2_train": len(base),
            "training_v4_after_base": len(kept_v4),
            "gift_relabeled": relabel_counts["relabel"],
            "gift_dropped": len(map_dropped),
            "after_relabel": len(relabelled),
            "removed_for_human_value_02": len(removed),
            "after_removal": len(inherited),
            "contrast_04_complete": len(contrast_04),
            "train": len(records),
        },
        "type_counts": v1.type_counts(records),
        "type_counts_by_source": {
            "training_v2_train": v1.type_counts(base),
            "training_v4_after_relabel_and_removal": v1.type_counts(inherited),
            "contrast_04": v1.type_counts(contrast_04),
        },
        "source_counts": dict(Counter(r["source_batch"] for r in records)),
        "annotator_counts": dict(Counter(r["provenance"]["annotator"] for r in records)),
        "gift_relabel": {
            "map": str(RELABEL / "map.jsonl"),
            "applied": [{k: r[k] for k in ("id", "text", "old", "new", "action")} for r in rows],
        },
        "removed_for_human_value_02": removed,
        "removed_note": "training-v4 notes (after the relabel) removed because they are near "
        "duplicates of, or at folded char-3 Jaccard >= "
        f"{HV_CHAR3_LIMIT} to, a note of human-value-02 batch b (written after training-v4); score "
        "only, the test notes were not read",
        "excluded_human_value_01_ids": excluded,
        "leakage_gate": {
            "sets": sizes,
            "human_value_02_review_queue_sha256": {
                str(rel): v1.sha256_bytes((root / rel).read_bytes()) for rel in HV02_QUEUES
            },
            "rule": "no id, no verbatim or NFC+strip+lowercase text of any train record in a "
            "set; no non-base note is a near duplicate (exact, folded, sequence >= 0.9, token, "
            "char3 >= 0.8: scripts/build_human_value_queue.py) of any note of a set, and no "
            "base note of a human-value-02 note; no non-base note has folded char-3-gram "
            f"Jaccard >= {HV_CHAR3_LIMIT} to a human-value-01 or human-value-02 note",
            "max_char3_new_notes": char3,
            "base_notes_near_duplicate_of_set": base_near_dup,
            "base_notes_at_char3_ge_limit": base_char3,
            "base_note": "the 905 training-v2 notes are byte-identical to the earlier runs (same "
            "templated corpus text, not edited here): earlier held-out notes have near "
            "duplicates among them (counts above, informational, as in every earlier run); none "
            "is a near duplicate of human-value-02; some reach the stricter char-3 bound, which "
            "is applied to the non-base notes only",
            "result": "pass",
        },
        "inputs_sha256": {str(p): v1.sha256_bytes((root / p).read_bytes()) for p in INPUTS},
        "outputs_sha256": {"train.jsonl": v1.sha256_bytes(body)},
        "evaluation_sets": (
            "validation.jsonl (in-sample, logging only) and test.jsonl are symlinks to the "
            "frozen annotation-v1 splits"
        ),
    }
    return body, manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    root, out = args.root, args.root / OUT
    try:
        body, manifest = build(root)
    except (ValueError, OSError, KeyError) as exc:
        print(f"{NAME}: {exc}", file=sys.stderr)
        return 1
    text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    targets = {
        n: Path("..") / ".." / "annotation-v1" / "splits" / f"{n}.jsonl"
        for n in ("validation", "test")
    }
    if args.check:
        problems = []
        if not (out / "train.jsonl").is_file() or (out / "train.jsonl").read_bytes() != body:
            problems.append("train.jsonl differs from the rebuild")
        if (
            not (out / "manifest.json").is_file()
            or (out / "manifest.json").read_text("utf-8") != text
        ):
            problems.append("manifest.json differs from the rebuild")
        for name in targets:
            link = out / f"{name}.jsonl"
            if link.resolve() != (root / v1.SPLITS / f"{name}.jsonl").resolve():
                problems.append(f"{link} is not a link to the frozen {name} split")
        if problems:
            print("\n".join(problems), file=sys.stderr)
            return 1
        print(f"{NAME} check OK: {manifest['counts']}")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.jsonl").write_bytes(body)
    for name, target in targets.items():
        link = out / f"{name}.jsonl"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target)
    (out / "manifest.json").write_text(text, encoding="utf-8")
    print(f"wrote {out}: {manifest['counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
