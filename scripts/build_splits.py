#!/usr/bin/env python
"""Build the deterministic, leakage-safe train/validation/test split of annotation-v1.

Usage:
    uv run python scripts/build_splits.py [--combined-dir ...] [--out-dir ...] [--seed ...]

Reads the frozen combined dataset (``queue``, ``labels``, ``provenance``; never modified), keeps
only trainable (complete) records, groups template/mirror/near-duplicate records, and writes
``train|validation|test.jsonl`` plus ``manifest.json``. Existing outputs are refused unless
``--overwrite`` is passed.

Split record: ``{id, text, type, target, source_batch, accented, provenance, group}`` where
``provenance`` is the combined provenance entry minus ``id`` and ``group`` indexes the leakage
group (groups sorted by first id). Target offsets are code points into the NFC text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from gidi.annotation.schema import load_config, trainable
from gidi.annotation.split import (
    MAX_LENGTH_GAP,
    MIN_GROUPS_FOR_COVERAGE,
    RATIOS,
    SIMILARITY_THRESHOLD,
    SPLITS,
    STOPWORDS,
    TARGET_MASK,
    accented,
    assign_splits,
    leakage_groups,
)
from gidi.corpus.jsonl import read_jsonl, write_jsonl

COMBINED = Path("datasets/annotation-v1/combined")
SPLITS_DIR = Path("datasets/annotation-v1/splits")
MANIFEST = "manifest.json"
SEED = "annotation-v1-split-02"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(combined_dir: Path) -> list[dict[str, Any]]:
    """Trainable records of the combined dataset, in queue order, ready for grouping."""
    queue = read_jsonl(combined_dir / "queue.jsonl")
    labels = {
        r["id"]: r for r in trainable(read_jsonl(combined_dir / "labels.jsonl"), load_config())
    }
    provenance = {r["id"]: r for r in read_jsonl(combined_dir / "provenance.jsonl")}
    rows = []
    for q in queue:
        label = labels.get(q["id"])
        if label is None:
            continue
        text = unicodedata.normalize("NFC", q["text"])
        rows.append(
            {
                "id": q["id"],
                "text": text,
                "type": label["type"],
                "target": label["target"],
                "source_batch": q["source_batch"],
                "accented": accented(text),
                "provenance": {k: v for k, v in provenance[q["id"]].items() if k != "id"},
            }
        )
    return rows


def count_table(
    by_split: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, dict[str, int]]]:
    counts: dict[str, dict[str, dict[str, int]]] = {}
    for split, records in by_split.items():
        tally = Counter((r["type"], r["accented"]) for r in records)
        counts[split] = {
            t: {"accented": tally[(t, True)], "unaccented": tally[(t, False)]}
            for t in sorted({t for t, _ in tally})
        }
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--combined-dir", type=Path, default=COMBINED)
    parser.add_argument("--out-dir", type=Path, default=SPLITS_DIR)
    parser.add_argument("--seed", default=SEED)
    parser.add_argument("--overwrite", action="store_true", help="replace existing split files")
    args = parser.parse_args(argv)

    outputs = [args.out_dir / f"{s}.jsonl" for s in SPLITS] + [args.out_dir / MANIFEST]
    existing = [p for p in outputs if p.exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"refusing to overwrite existing file: {existing[0]} (pass --overwrite)")

    rows = load_rows(args.combined_dir)
    groups = leakage_groups(rows)
    group_index = {rid: gi for gi, g in enumerate(groups) for rid in g}
    split_of = assign_splits(rows, seed=args.seed, groups=groups)

    by_split: dict[str, list[dict[str, Any]]] = {s: [] for s in SPLITS}
    for row in rows:
        by_split[split_of[row["id"]]].append({**row, "group": group_index[row["id"]]})

    for split, records in by_split.items():
        write_jsonl(args.out_dir / f"{split}.jsonl", records, overwrite=args.overwrite)

    manifest = {
        "annotation_version": "annotation-v1",
        "seed": args.seed,
        "ratios": dict(zip(SPLITS, RATIOS, strict=True)),
        "grouping": {
            "rules": [
                "template (target masked, amounts masked)",
                "mirror (bag of tokens)",
                "near-duplicate",
            ],
            "target_mask": TARGET_MASK,
            "stopwords": sorted(STOPWORDS),
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "max_length_gap": MAX_LENGTH_GAP,
            "min_groups_for_type_coverage": MIN_GROUPS_FOR_COVERAGE,
        },
        "records": len(rows),
        "n_groups": len(groups),
        "n_groups_per_split": {
            s: len({r["group"] for r in records}) for s, records in by_split.items()
        },
        "counts": count_table(by_split),
        "source_batch_counts": {
            s: dict(sorted(Counter(r["source_batch"] for r in records).items()))
            for s, records in by_split.items()
        },
        "combined_sha256": {
            name: sha256_file(args.combined_dir / f"{name}.jsonl")
            for name in ("queue", "labels", "provenance")
        },
        "sha256": {f"{s}.jsonl": sha256_file(args.out_dir / f"{s}.jsonl") for s in SPLITS},
    }
    (args.out_dir / MANIFEST).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"{len(rows)} trainable records in {len(groups)} groups -> {args.out_dir}")
    for split, by_type in manifest["counts"].items():
        print(f"{split}: {len(by_split[split])} records")
        for t, c in by_type.items():
            print(f"  {t:<14} accented {c['accented']:>3}  unaccented {c['unaccented']:>3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
