#!/usr/bin/env python
"""Build the distillation-v1 data: targeted-v2 training data plus the frozen validation split.

Usage:
    uv run python scripts/build_distillation_data.py [--root R] [--check]

Writes ``datasets/annotation-v1/distillation-v1/``:

* ``train.jsonl``: the bytes of ``training-v2/train.jsonl`` (frozen train + targeted-02) followed
  by the bytes of the frozen ``splits/validation.jsonl``;
* ``validation.jsonl`` -> ``../splits/validation.jsonl``: now in-sample, per-epoch logging only;
* ``test.jsonl`` -> ``../splits/test.jsonl``: evaluation-only, never trained on;
* ``manifest.json``: counts by source, type counts, sha256 of the output and of every source,
  the untouched evaluation files (frozen test and every ``datasets/probe-*`` file) and the
  explicit exclusions.

Fails loudly (exit 1, nothing written) unless: ``build_training.py --version v2 --check`` passes;
the frozen validation/test files match the sha256 in ``splits/manifest.json``; every record
passes ``gidi.distillation.guard.assert_distillation_trainable``; ids are unique and disjoint
from the frozen test split and every probe set. ``--check`` rebuilds in memory and compares with
the files on disk.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

from gidi.corpus.jsonl import read_jsonl
from gidi.distillation.guard import assert_distillation_trainable
from gidi.training.train import TYPES

ANNOTATION = Path("datasets/annotation-v1")
SPLITS = ANNOTATION / "splits"
TRAINING_V2 = ANNOTATION / "training-v2"
OUT = ANNOTATION / "distillation-v1"
NAME = "distillation-v1"
TARGETED_02_BATCH = "targeted-annotation-v1-02"

EXCLUDED = [
    {
        "path": "datasets/annotation-v1/splits/test.jsonl",
        "reason": "frozen held-out test split: evaluation-only",
    },
    {
        "path": "datasets/probe-v1/",
        "reason": "diagnostic probe set: evaluation-only, never trained on or augmented from",
    },
    {
        "path": "datasets/annotation-v1/targeted-03/",
        "reason": "targeted-03 batch: experiment artifact rejected as teacher data",
    },
    {
        "path": "datasets/annotation-v1/training-v3/",
        "reason": "training-v3 (frozen train + targeted-02 + targeted-03): rejected as teacher",
    },
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_build_training():
    spec = importlib.util.spec_from_file_location(
        "build_training", Path(__file__).resolve().with_name("build_training.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_training_v2(root: Path) -> None:
    """Run ``build_training.py --version v2 --check``; raise if training-v2 is not intact."""
    code = _load_build_training().main(["--version", "v2", "--root", str(root), "--check"])
    if code != 0:
        raise ValueError("training-v2 is not intact (build_training.py --version v2 --check)")


def eval_files(root: Path) -> list[Path]:
    return [SPLITS / "validation.jsonl", SPLITS / "test.jsonl"] + sorted(
        p.relative_to(root) for p in root.glob("datasets/probe-*/*") if p.is_file()
    )


def probe_ids(root: Path) -> set[str]:
    ids: set[str] = set()
    for path in root.glob("datasets/probe-*/*.jsonl"):
        ids |= {r["id"] for r in read_jsonl(path) if "id" in r}
    return ids


def verify_frozen_hashes(root: Path) -> None:
    expected = json.loads((root / SPLITS / "manifest.json").read_text(encoding="utf-8"))["sha256"]
    bad = [
        f"{name}: {sha256(root / SPLITS / name)} != {expected[name]}"
        for name in ("validation.jsonl", "test.jsonl")
        if sha256(root / SPLITS / name) != expected[name]
    ]
    if bad:
        raise ValueError(
            "frozen split hashes differ from splits/manifest.json:\n  " + "\n  ".join(bad)
        )


def type_counts(records: list[dict]) -> dict[str, int]:
    return {t: sum(r["type"] == t for r in records) for t in TYPES}


def build(root: Path) -> tuple[bytes, dict]:
    """Verify every precondition and return ``(train.jsonl bytes, manifest)``; writes nothing."""
    verify_training_v2(root)
    verify_frozen_hashes(root)

    v2_bytes = (root / TRAINING_V2 / "train.jsonl").read_bytes()
    validation_bytes = (root / SPLITS / "validation.jsonl").read_bytes()
    body = v2_bytes + validation_bytes

    frozen_train_ids = {r["id"] for r in read_jsonl(root / SPLITS / "train.jsonl")}
    v2_records = read_jsonl(root / TRAINING_V2 / "train.jsonl")
    validation = read_jsonl(root / SPLITS / "validation.jsonl")
    original = [r for r in v2_records if r["id"] in frozen_train_ids]
    targeted_02 = [r for r in v2_records if r["id"] not in frozen_train_ids]
    bad_batch = [r["id"] for r in targeted_02 if r.get("source_batch") != TARGETED_02_BATCH]
    if bad_batch:
        raise ValueError(f"training-v2 records outside train/targeted-02: {bad_batch}")
    records = v2_records + validation

    assert_distillation_trainable(records, root)
    ids = [r["id"] for r in records]
    duplicates = sorted(i for i, n in Counter(ids).items() if n > 1)
    if duplicates:
        raise ValueError(f"duplicate ids in the distillation train set: {duplicates}")
    test_ids = {r["id"] for r in read_jsonl(root / SPLITS / "test.jsonl")}
    overlap = sorted((set(ids) & test_ids) | (set(ids) & probe_ids(root)))
    if overlap:
        raise ValueError(f"ids overlap with the frozen test split or a probe set: {overlap}")

    v2_manifest = json.loads((root / TRAINING_V2 / "manifest.json").read_text(encoding="utf-8"))
    if (len(original), len(targeted_02)) != (
        v2_manifest["counts"]["original_train"],
        v2_manifest["counts"]["targeted_02_trainable"],
    ):
        raise ValueError("source counts differ from training-v2/manifest.json")

    manifest = {
        "name": NAME,
        "annotation_version": "annotation-v1",
        "purpose": (
            "final teacher training data: targeted-v2 training data (frozen train + targeted-02) "
            "plus the frozen validation split; validation is in-sample from here on, so the "
            "frozen test split and probe-v1 stay the only held-out evaluation"
        ),
        "counts": {
            "original_train": len(original),
            "targeted_02": len(targeted_02),
            "frozen_validation": len(validation),
            "train": len(records),
        },
        "type_counts": type_counts(records),
        "type_counts_by_source": {
            "original_train": type_counts(original),
            "targeted_02": type_counts(targeted_02),
            "frozen_validation": type_counts(validation),
        },
        "sources": {
            "datasets/annotation-v1/training-v2/train.jsonl": sha256(
                root / TRAINING_V2 / "train.jsonl"
            ),
            "datasets/annotation-v1/training-v2/manifest.json": sha256(
                root / TRAINING_V2 / "manifest.json"
            ),
            "datasets/annotation-v1/splits/validation.jsonl": sha256(
                root / SPLITS / "validation.jsonl"
            ),
        },
        "files": {"train.jsonl": hashlib.sha256(body).hexdigest()},
        "evaluation_sets": (
            "validation.jsonl (in-sample, logging only) and test.jsonl are symlinks to the "
            "frozen splits"
        ),
        "evaluation_files": {str(p): sha256(root / p) for p in eval_files(root)},
        "excluded": EXCLUDED,
        "guard": "gidi.distillation.guard.assert_distillation_trainable",
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
    if args.check:
        problems = []
        if not (out / "train.jsonl").is_file() or (out / "train.jsonl").read_bytes() != body:
            problems.append("train.jsonl differs from the rebuild")
        if not (out / "manifest.json").is_file() or (
            (out / "manifest.json").read_text(encoding="utf-8") != text
        ):
            problems.append(
                "manifest.json differs from the rebuild (sources or evaluation changed?)"
            )
        for name in ("validation", "test"):
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
    for name in ("validation", "test"):
        link = out / f"{name}.jsonl"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(Path("..") / "splits" / f"{name}.jsonl")
    (out / "manifest.json").write_text(text, encoding="utf-8")
    print(f"wrote {out}: {manifest['counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
