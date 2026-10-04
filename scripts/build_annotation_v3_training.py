#!/usr/bin/env python
"""Build ``datasets/annotation-v3/training-v1``: the encoder-1 retrain set for annotation-v3.

Usage:
    uv run python scripts/build_annotation_v3_training.py [--root R] [--check]

``train.jsonl`` = the 723 byte-identical records of ``datasets/annotation-v1/distillation-v1``
(the set the deployed encoder was trained on) + the ``complete`` human labels of
``datasets/annotation-v3/debt-01`` (debt-only notes: borrow when the user owes, lend when the
other party owes), converted to the same record format (``provenance.annotator`` human,
``source_batch`` debt-01). ``uncertain`` and ``skipped`` labels are never trainable.
``validation.jsonl`` and ``test.jsonl`` are symlinks to the frozen annotation-v1 splits, as in
distillation-v1 (validation is in-sample, test stays held out).

Leakage gate (raises, writes nothing): no id and no exact text (verbatim or NFC + strip +
lowercase) of the final train set may occur in the frozen test split, probe-v1 or
``datasets/annotation-v2/human-value-01``; the 9 human-value-01 ids named in the debt-01 manifest
are asserted absent. ``--check`` rebuilds in memory and compares the files on disk.
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

NAME = "training-v1"
OUT = Path("datasets/annotation-v3/training-v1")
DISTILLATION = Path("datasets/annotation-v1/distillation-v1")
SPLITS = Path("datasets/annotation-v1/splits")
DEBT = Path("datasets/annotation-v3/debt-01")
HUMAN = Path("datasets/annotation-v2/human-value-01")
PROBE = Path("datasets/probe-v1")
SOURCE_BATCH = "debt-01"
TYPES = (
    "expense",
    "income",
    "borrow",
    "lend",
    "repayment_in",
    "repayment_out",
    "transfer",
    "refund",
)
INPUTS = (
    DISTILLATION / "train.jsonl",
    DISTILLATION / "manifest.json",
    SPLITS / "validation.jsonl",
    SPLITS / "test.jsonl",
    DEBT / "queue.jsonl",
    DEBT / "labels.jsonl",
    DEBT / "provenance.jsonl",
    DEBT / "manifest.json",
    HUMAN / "labels.jsonl",
    HUMAN / "review-queue.jsonl",
    HUMAN / "freeze.json",
    PROBE / "notes.jsonl",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def exact_key(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip().lower()


def accented(text: str) -> bool:
    return any(ord(c) > 127 for c in text)


def debt_records(root: Path) -> list[dict[str, Any]]:
    """The ``complete`` debt-01 labels as distillation-v1-format records, in labels.jsonl order."""
    text_of = {r["id"]: r["text"] for r in read_jsonl(root / DEBT / "queue.jsonl")}
    records = []
    for label in read_jsonl(root / DEBT / "labels.jsonl"):
        if label["annotation_status"] != "complete":
            continue
        rid, text = label["id"], text_of[label["id"]]
        if label["type"] not in ("borrow", "lend"):
            raise ValueError(f"{rid}: debt-01 complete label has type {label['type']!r}")
        target = label["target"]
        if target is not None:
            span = {k: target[k] for k in ("text", "start", "end")}
            if text[span["start"] : span["end"]] != span["text"]:
                raise ValueError(f"{rid}: target span does not slice the note text")
            target = span
        records.append(
            {
                "id": rid,
                "text": text,
                "type": label["type"],
                "target": target,
                "source_batch": SOURCE_BATCH,
                "accented": accented(text),
                "provenance": {"annotator": "human", "source_batch": SOURCE_BATCH},
                "group": f"{SOURCE_BATCH}:{rid}",
            }
        )
    return records


def held_out(root: Path) -> dict[str, list[dict[str, Any]]]:
    return {
        "frozen test": read_jsonl(root / SPLITS / "test.jsonl"),
        "probe-v1": read_jsonl(root / PROBE / "notes.jsonl"),
        "human-value-01": read_jsonl(root / HUMAN / "review-queue.jsonl"),
    }


def assert_no_leakage(
    records: list[dict[str, Any]], sets: dict[str, list[dict[str, Any]]], excluded_ids: list[str]
) -> None:
    """Raise ``ValueError`` listing every train record that is held out, by id or exact text."""
    ids = [r["id"] for r in records]
    dupes = sorted(i for i, n in Counter(ids).items() if n > 1)
    if dupes:
        raise ValueError(f"duplicate ids in the train set: {dupes}")
    bad = []
    for name, rows in sets.items():
        held_ids = {r["id"] for r in rows}
        held_texts = {r["text"] for r in rows} | {exact_key(r["text"]) for r in rows}
        for r in records:
            if r["id"] in held_ids:
                bad.append(f"{r['id']}: id in {name}")
            if r["text"] in held_texts or exact_key(r["text"]) in held_texts:
                bad.append(f"{r['id']}: text in {name}: {r['text']!r}")
    missing_from_held = [
        i for i in excluded_ids if i not in {r["id"] for r in sets["human-value-01"]}
    ]
    in_train = sorted(set(excluded_ids) & set(ids))
    if bad or in_train or missing_from_held:
        raise ValueError(
            "held-out leakage: "
            + "; ".join(bad + [f"excluded id in train: {i}" for i in in_train])
            + (
                f"; excluded ids not in human-value-01: {missing_from_held}"
                if missing_from_held
                else ""
            )
        )


def type_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    return {t: sum(r["type"] == t for r in records) for t in TYPES}


def build(root: Path) -> tuple[bytes, dict[str, Any]]:
    """Verify every precondition and return ``(train.jsonl bytes, manifest)``; writes nothing."""
    base_bytes = (root / DISTILLATION / "train.jsonl").read_bytes()
    base = [json.loads(line) for line in base_bytes.decode("utf-8").splitlines()]
    debt = debt_records(root)
    debt_manifest = json.loads((root / DEBT / "manifest.json").read_text("utf-8"))
    excluded = debt_manifest["excluded_human_value_01_ids"]
    if len(base) != 723 or len(debt) != 79:
        raise ValueError(f"expected 723 + 79 records, got {len(base)} + {len(debt)}")
    records = base + debt
    assert_no_leakage(records, held_out(root), excluded)

    body = base_bytes + b"".join(
        (json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8") for r in debt
    )
    manifest = {
        "name": NAME,
        "annotation_version": "annotation-v3",
        "purpose": (
            "encoder-1 retrain set: distillation-v1 train (723, byte-identical) plus the complete "
            "debt-01 human labels (debt-only notes as borrow/lend); validation is in-sample, the "
            "frozen test split, probe-v1 and human-value-01 stay held out"
        ),
        "counts": {
            "distillation_v1_train": len(base),
            "debt_01_complete": len(debt),
            "debt_01_uncertain_dropped": sum(
                r["annotation_status"] != "complete"
                for r in read_jsonl(root / DEBT / "labels.jsonl")
            ),
            "train": len(records),
        },
        "type_counts": type_counts(records),
        "type_counts_by_source": {
            "distillation_v1_train": type_counts(base),
            "debt_01": type_counts(debt),
        },
        "source_counts": dict(Counter(r["source_batch"] for r in records)),
        "debt_01_source_counts": dict(
            Counter(
                r["source"]
                for r in read_jsonl(root / DEBT / "provenance.jsonl")
                if r["id"] in {d["id"] for d in debt}
            )
        ),
        "excluded_human_value_01_ids": excluded,
        "leakage_gate": {
            "sets": {name: len(rows) for name, rows in held_out(root).items()},
            "rule": "no id, no verbatim or NFC+strip+lowercase text of any train record in a set",
            "result": "pass",
        },
        "inputs_sha256": {str(p): sha256_bytes((root / p).read_bytes()) for p in INPUTS},
        "outputs_sha256": {"train.jsonl": sha256_bytes(body)},
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
            if link.resolve() != (root / SPLITS / f"{name}.jsonl").resolve():
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
