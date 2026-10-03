#!/usr/bin/env python
"""Build a training set: the frozen annotation-v1 train split plus complete targeted records.

Usage:
    uv run python scripts/build_training.py --version {v2,v3} [--root R] [--check]

Versions:

* ``v2`` -> ``datasets/annotation-v1/training-v2/``: frozen train + targeted-02;
* ``v3`` -> ``datasets/annotation-v1/training-v3/``: frozen train + targeted-02 + targeted-03.

Writes the output directory:

* ``train.jsonl``: the frozen ``splits/train.jsonl`` lines byte-for-byte, then every
  ``annotation_status == "complete"`` record of each augment batch (in batch order) in the split
  record format, carrying its minimal-pair ``group`` and probe ``patterns``;
* ``validation.jsonl`` and ``test.jsonl``: relative symlinks to the frozen split files, so the
  evaluation sets are the same bytes by construction;
* ``manifest.json``: counts, distributions, leakage diagnostics and sha256 hashes, including the
  untouched evaluation files (frozen validation/test and every ``datasets/probe-*`` file).

Fails loudly (exit 1, nothing written) when an augmented record shares an id with frozen
validation/test, a probe set, frozen train or another augment batch; is template/similarity
leakage against frozen validation, frozen test or a probe set (``gidi.annotation.leakage``); or
exactly duplicates (after normalization) a frozen train note or a record of an earlier augment
batch. ``--check`` rebuilds in memory, re-runs the leakage gate, and verifies the files on disk
and the evaluation hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from gidi.annotation.leakage import EVAL_REJECTS, check, load_references, reference
from gidi.annotation.split import accented
from gidi.corpus.jsonl import read_jsonl
from gidi.training.train import TYPES

SPLITS = Path("datasets/annotation-v1/splits")
ANNOTATION = Path("datasets/annotation-v1")
VERSIONS = {
    "v2": {
        "out": ANNOTATION / "training-v2",
        "batches": ["targeted-02"],
        "purpose": "controlled experiment: frozen train + targeted-02, same frozen evaluation sets",
    },
    "v3": {
        "out": ANNOTATION / "training-v3",
        "batches": ["targeted-02", "targeted-03"],
        "purpose": (
            "controlled experiment: frozen train + targeted-02 + targeted-03, "
            "same frozen evaluation sets"
        ),
    },
}
FAIL_ON = EVAL_REJECTS | {"train_duplicate"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def batch_key(batch: str) -> str:
    return batch.replace("-", "_")


def source_batch(batch: str) -> str:
    return f"targeted-annotation-v1-{batch.rsplit('-', 1)[1]}"


def augmented_records(root: Path, batch: str) -> list[dict]:
    directory = root / ANNOTATION / batch
    notes = {n["id"]: n for n in read_jsonl(directory / "notes.jsonl")}
    labels = {lab["id"]: lab for lab in read_jsonl(directory / "labels.jsonl")}
    prov = {p["id"]: p for p in read_jsonl(directory / "provenance.jsonl")}
    source = source_batch(batch)
    records = []
    for q in read_jsonl(directory / "queue.jsonl"):
        lab = labels[q["id"]]
        if lab["annotation_status"] != "complete":
            continue
        p, n = prov[q["id"]], notes[q["id"]]
        records.append(
            {
                "id": q["id"],
                "text": q["text"],
                "type": lab["type"],
                "target": lab["target"],
                "source_batch": source,
                "accented": accented(q["text"]),
                "provenance": {
                    "annotator": p["annotator"],
                    "source_batch": source,
                    "review_state": p["review_state"],
                },
                "group": n["group"],
                "patterns": n["patterns"],
            }
        )
    return records


def leakage_gate(root: Path, name: str, batches: dict[str, list[dict]]) -> dict:
    eval_ids = {
        r["id"] for s in ("validation", "test") for r in read_jsonl(root / SPLITS / f"{s}.jsonl")
    }
    for probe in sorted(root.glob("datasets/probe-*")):
        eval_ids |= {r["id"] for r in read_jsonl(probe / "queue.jsonl")}
    train_ids = {r["id"] for r in read_jsonl(root / SPLITS / "train.jsonl")}
    records = [r for recs in batches.values() for r in recs]
    ids = [r["id"] for r in records]
    errors = [f"{i}: id is in an evaluation set" for i in ids if i in eval_ids]
    errors += [f"{i}: id already in frozen train" for i in ids if i in train_ids]
    # ids must be unique within and across augment batches
    errors += [f"{i}: duplicate id" for i, c in Counter(ids).items() if c > 1]

    eval_refs, train_refs = load_references(root)
    results = []
    for batch, recs in batches.items():
        candidates = [
            {
                "text": r["text"],
                "intended_target": r["target"]["text"] if r["target"] else None,
                "group": r["group"],
            }
            for r in recs
        ]
        batch_results = check(candidates, eval_refs, train_refs)
        for rec, res in zip(recs, batch_results, strict=True):
            bad = sorted(set(res["reject"]) & FAIL_ON)
            if bad:
                errors.append(
                    f"{rec['id']} {rec['text']!r}: {', '.join(bad)} (eval {res['eval_sim']})"
                )
        results += batch_results
        # later batches treat this one as training data (exact duplicates are rejected)
        train_refs += [reference(c["text"], c["intended_target"], batch) for c in candidates]
    if errors:
        print(f"LEAKAGE: {name} not built", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        raise SystemExit(1)
    sims = sorted(r["eval_sim"] for r in results)
    return {
        "checked": len(records),
        "eval_refs": len(eval_refs),
        "fail_on": sorted(FAIL_ON),
        "failures": 0,
        "max_eval_similarity": sims[-1],
        "median_eval_similarity": sims[len(sims) // 2],
    }


def build(root: Path, version: str) -> tuple[bytes, dict]:
    spec = VERSIONS[version]
    name = f"training-{version}"
    frozen_train = (root / SPLITS / "train.jsonl").read_bytes()
    original = read_jsonl(root / SPLITS / "train.jsonl")
    batches = {b: augmented_records(root, b) for b in spec["batches"]}
    leakage = leakage_gate(root, name, batches)
    records = [r for recs in batches.values() for r in recs]
    body = (
        frozen_train + "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode()
    )
    combined = original + records
    eval_files = [SPLITS / "validation.jsonl", SPLITS / "test.jsonl"] + sorted(
        p.relative_to(root) for p in root.glob("datasets/probe-*/*") if p.is_file()
    )
    keys = {b: batch_key(b) for b in batches}
    manifest = {
        "name": name,
        "annotation_version": "annotation-v1",
        "purpose": spec["purpose"],
        "counts": {
            "original_train": len(original),
            **{f"{keys[b]}_trainable": len(recs) for b, recs in batches.items()},
            "train": len(combined),
        },
        "source_batch_counts": dict(Counter(r["source_batch"] for r in combined)),
        "type_counts": {t: sum(r["type"] == t for r in combined) for t in TYPES},
        **{
            f"type_counts_{keys[b]}": {t: sum(r["type"] == t for r in recs) for t in TYPES}
            for b, recs in batches.items()
        },
        "accented_counts": {
            "accented": sum(r["accented"] for r in combined),
            "unaccented": sum(not r["accented"] for r in combined),
        },
    }
    for b, recs in batches.items():
        manifest[f"{keys[b]}_patterns"] = dict(
            sorted(Counter(p for r in recs for p in r["patterns"]).items())
        )
        manifest[f"{keys[b]}_groups"] = {
            "groups": len({r["group"] for r in recs}),
            "minimal_pair_groups": sum(c == 2 for c in Counter(r["group"] for r in recs).values()),
            "note": "a group is one unit: never split it across train/validation/test",
        }
    manifest |= {
        "evaluation_sets": "validation.jsonl and test.jsonl are symlinks to the frozen splits",
        "leakage_gate": leakage,
        "files": {"train.jsonl": hashlib.sha256(body).hexdigest()},
        "evaluation_files": {str(p): sha256(root / p) for p in eval_files},
    }
    return body, manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", choices=sorted(VERSIONS), required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    root, out = args.root, args.root / VERSIONS[args.version]["out"]

    body, manifest = build(root, args.version)
    text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    if args.check:
        problems = []
        if (out / "train.jsonl").read_bytes() != body:
            problems.append("train.jsonl differs from the rebuild")
        if (out / "manifest.json").read_text(encoding="utf-8") != text:
            problems.append("manifest.json differs from the rebuild (evaluation files changed?)")
        for name in ("validation", "test"):
            link = out / f"{name}.jsonl"
            if link.resolve() != (root / SPLITS / f"{name}.jsonl").resolve():
                problems.append(f"{link} is not a link to the frozen {name} split")
        if problems:
            print("\n".join(problems), file=sys.stderr)
            return 1
        print(f"{manifest['name']} check OK: {manifest['counts']}")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    (out / "train.jsonl").write_bytes(body)
    for name in ("validation", "test"):
        link = out / f"{name}.jsonl"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(Path("..") / "splits" / f"{name}.jsonl")
    (out / "manifest.json").write_text(text, encoding="utf-8")
    print(f"wrote {out}: {manifest['counts']}; leakage gate: 0 failures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
