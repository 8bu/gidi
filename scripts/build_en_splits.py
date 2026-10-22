#!/usr/bin/env python
"""gidi-en data prep: near-duplicate removal and the deterministic test/dev/seed/train split.

Usage:
    uv run python scripts/build_en_splits.py dedupe GENERATED... --out /tmp/en-dedup.jsonl
    uv run python scripts/build_en_splits.py split [--corpus corpus/raw/en-baseline-01.jsonl]

``dedupe`` drops exact (normalised, amounts masked) and near duplicates (SequenceMatcher ratio
>= 0.90, ``gidi.annotation.leakage.NEAR_DUP_THRESHOLD``) from the generated notes, keeping the
first in file order. Its output is then ingested with ``scripts/ingest_generated_corpus.py``.

``split`` orders the corpus by sha256(seed:id) and assigns test, dev, seed, then train. A note is
dropped when it is too similar to a note already placed in another split (ratio >=
``EVAL_THRESHOLD`` 0.85, or the same name/amount skeleton), so train never leaks into the held-out
sets. Output notes only: ``datasets/annotation-en-v1/splits/{test,dev,seed,train}.jsonl`` plus
``manifest.json``. All labels are silver (see docs/annotation-en-v1.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from gidi.annotation.leakage import EVAL_THRESHOLD, NEAR_DUP_THRESHOLD, nearest, skeleton
from gidi.annotation.split import normalize
from gidi.corpus.jsonl import read_jsonl, write_jsonl

SEED = "annotation-en-v1-split-01"
SIZES = {"test": 300, "dev": 200, "seed": 300}
OUT_DIR = Path("datasets/annotation-en-v1/splits")
CORPUS = Path("corpus/raw/en-baseline-01.jsonl")


def dedupe(paths: list[Path], out: Path) -> int:
    kept: list[dict] = []
    refs: list[dict] = []
    seen: set[str] = set()
    dropped: Counter[str] = Counter()
    for path in paths:
        for rec in read_jsonl(path):
            norm = normalize(rec["text"])
            if norm in seen:
                dropped["exact"] += 1
                continue
            sim, _ = nearest(norm, refs)
            if sim >= NEAR_DUP_THRESHOLD:
                dropped["near"] += 1
                continue
            kept.append({"text": rec["text"]})
            refs.append({"norm": norm})
            seen.add(norm)
    write_jsonl(out, kept, overwrite=True)
    print(f"kept {len(kept)}; dropped {dict(dropped)}")
    return 0


def split(corpus: Path, out_dir: Path) -> int:
    rows = read_jsonl(corpus)
    rows.sort(key=lambda r: hashlib.sha256(f"{SEED}:{r['id']}".encode()).hexdigest())
    placed: dict[str, list[dict]] = {"test": [], "dev": [], "seed": [], "train": []}
    refs: dict[str, list[dict]] = {k: [] for k in placed}
    skels: dict[str, set[str]] = {k: set() for k in placed}
    leaked = 0
    order = [name for name, n in SIZES.items() for _ in range(n)] + ["train"] * len(rows)
    cursor = 0
    for row in rows:
        name = order[cursor]
        norm, skel = normalize(row["text"]), skeleton(row["text"], None)
        clash = False
        for other in placed:
            if other == name:
                continue
            if skel in skels[other] or nearest(norm, refs[other])[0] >= EVAL_THRESHOLD:
                clash = True
                break
        if clash:
            leaked += 1
            continue
        placed[name].append(row)
        refs[name].append({"norm": norm})
        skels[name].add(skel)
        cursor += 1
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": "annotation-en-v1",
        "seed": SEED,
        "order": "sha256(seed:id) ascending; test, dev, seed, then train",
        "corpus": str(corpus),
        "corpus_records": len(rows),
        "leak_threshold": EVAL_THRESHOLD,
        "dropped_as_leak": leaked,
        "splits": {},
    }
    for name, recs in placed.items():
        recs.sort(key=lambda r: r["id"])
        write_jsonl(out_dir / f"{name}.jsonl", recs, overwrite=True)
        manifest["splits"][name] = {
            "n": len(recs),
            "sha256": hashlib.sha256((out_dir / f"{name}.jsonl").read_bytes()).hexdigest(),
        }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dedupe")
    d.add_argument("generated", nargs="+", type=Path)
    d.add_argument("--out", required=True, type=Path)
    s = sub.add_parser("split")
    s.add_argument("--corpus", type=Path, default=CORPUS)
    s.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    if args.cmd == "dedupe":
        return dedupe(args.generated, args.out)
    return split(args.corpus, args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
