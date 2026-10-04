#!/usr/bin/env python
"""Build ``datasets/annotation-v3/training-v2``: training-v1 + the LLM-composed contrast notes.

Usage:
    uv run python scripts/build_annotation_v3_training_v2.py [--root R] [--check]

``train.jsonl`` = the 802 byte-identical records of ``datasets/annotation-v3/training-v1`` (723
distillation-v1 + 79 debt-01 human labels) + the ``complete`` notes of
``datasets/annotation-v3/contrast-01`` (repayment / refund / transfer / expense notes that reuse
debt words; labels are LLM-made, ``provenance.annotator`` ``llm``, ``source_batch`` contrast-01).
``validation.jsonl`` and ``test.jsonl`` are symlinks to the frozen annotation-v1 splits, as in
training-v1 (validation is in-sample, test stays held out).

Leakage gate (raises, writes nothing): no id and no exact text (verbatim or NFC + strip +
lowercase) of the final train set may occur in the frozen test split, probe-v1 or
``datasets/annotation-v2/human-value-01``, and no contrast note may be a near duplicate (the
rules of ``scripts/build_human_value_queue.py``) of any note of those three sets. ``--check``
rebuilds in memory and compares the files on disk.
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
hv = _load("build_human_value_queue")

NAME = "training-v2"
OUT = Path("datasets/annotation-v3/training-v2")
BASE = Path("datasets/annotation-v3/training-v1")
CONTRAST = Path("datasets/annotation-v3/contrast-01")
SOURCE_BATCH = "contrast-01"
INPUTS = (
    BASE / "train.jsonl",
    BASE / "manifest.json",
    CONTRAST / "queue.jsonl",
    CONTRAST / "labels.jsonl",
    CONTRAST / "provenance.jsonl",
    CONTRAST / "manifest.json",
    *v1.INPUTS[2:4],
    v1.DEBT / "manifest.json",
    v1.HUMAN / "labels.jsonl",
    v1.HUMAN / "review-queue.jsonl",
    v1.HUMAN / "freeze.json",
    v1.PROBE / "notes.jsonl",
)


def contrast_records(root: Path) -> list[dict[str, Any]]:
    """The ``complete`` contrast-01 labels as training records, in labels.jsonl order."""
    text_of = {r["id"]: r["text"] for r in v1.read_jsonl(root / CONTRAST / "queue.jsonl")}
    annotator = {
        r["id"]: r["annotator"] for r in v1.read_jsonl(root / CONTRAST / "provenance.jsonl")
    }
    records = []
    for label in v1.read_jsonl(root / CONTRAST / "labels.jsonl"):
        if label["annotation_status"] != "complete":
            continue
        rid, text = label["id"], text_of[label["id"]]
        if label["type"] not in v1.TYPES:
            raise ValueError(f"{rid}: contrast-01 label has type {label['type']!r}")
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
                "source_batch": SOURCE_BATCH,
                "accented": v1.accented(text),
                "provenance": {"annotator": annotator[rid], "source_batch": SOURCE_BATCH},
                "group": f"{SOURCE_BATCH}:{rid}",
            }
        )
    return records


def assert_no_near_duplicates(
    records: list[dict[str, Any]], sets: dict[str, list[dict[str, Any]]]
) -> dict[str, int]:
    """Raise when a record is a near duplicate of a held-out note; returns the set sizes."""
    bad = []
    for name, rows in sets.items():
        ref_set = hv.ReferenceSet([r["text"] for r in rows])
        for record in records:
            rule = ref_set.match(hv.Reference(record["text"]))
            if rule:
                bad.append(f"{record['id']}: {rule} near duplicate in {name}: {record['text']!r}")
    if bad:
        raise ValueError("held-out near duplicates: " + "; ".join(bad))
    return {name: len(rows) for name, rows in sets.items()}


def build(root: Path) -> tuple[bytes, dict[str, Any]]:
    """Verify every precondition and return ``(train.jsonl bytes, manifest)``; writes nothing."""
    base_bytes = (root / BASE / "train.jsonl").read_bytes()
    base = [json.loads(line) for line in base_bytes.decode("utf-8").splitlines()]
    contrast = contrast_records(root)
    excluded = json.loads((root / v1.DEBT / "manifest.json").read_text("utf-8"))[
        "excluded_human_value_01_ids"
    ]
    if len(base) != 802 or not contrast:
        raise ValueError(f"expected 802 base records and some contrast notes, got {len(base)}")
    records = base + contrast
    sets = v1.held_out(root)
    v1.assert_no_leakage(records, sets, excluded)
    sizes = assert_no_near_duplicates(contrast, sets)

    body = base_bytes + b"".join(
        (json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8") for r in contrast
    )
    contrast_manifest = json.loads((root / CONTRAST / "manifest.json").read_text("utf-8"))
    manifest = {
        "name": NAME,
        "annotation_version": "annotation-v3",
        "purpose": (
            "encoder-1 retrain set v2: training-v1 train (802, byte-identical) plus the "
            "LLM-composed contrast-01 notes (debt words in repayment / refund / transfer / "
            "expense notes); validation is in-sample, the frozen test split, probe-v1 and "
            "human-value-01 stay held out"
        ),
        "counts": {
            "training_v1_train": len(base),
            "contrast_01_complete": len(contrast),
            "train": len(records),
        },
        "type_counts": v1.type_counts(records),
        "type_counts_by_source": {
            "training_v1_train": v1.type_counts(base),
            "contrast_01": v1.type_counts(contrast),
        },
        "source_counts": dict(Counter(r["source_batch"] for r in records)),
        "annotator_counts": dict(Counter(r["provenance"]["annotator"] for r in records)),
        "contrast_01_dropped_by_group": contrast_manifest["leakage"]["dropped_by_group"],
        "excluded_human_value_01_ids": excluded,
        "leakage_gate": {
            "sets": sizes,
            "rule": "no id, no verbatim or NFC+strip+lowercase text of any train record in a "
            "set; no contrast-01 note is a near duplicate (exact, folded, sequence >= 0.9, "
            "token, char3: scripts/build_human_value_queue.py) of any note of a set",
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
