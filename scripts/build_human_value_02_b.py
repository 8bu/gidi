#!/usr/bin/env python
r"""Build batch B of the test set ``human-value-02``: 50 more LLM-written notes (queue only).

Usage:
    uv run python scripts/build_human_value_02_b.py [--root R] [--check]

``datasets/annotation-v3/human-value-02/candidates-b.txt`` holds the candidate notes in sections
(``weak``, ``gift``, ``debt``, ``shop``, ``other``; ``<section>-reserve`` only tops up a section
that lost candidates to the gate), written from the annotation rules and the
style of the reviewed corpus only, never from training data. They are gated with the
human-value-02 leakage gate (``scripts/build_human_value_02.py``: exact, folded, sequence, token,
char3 rules of ``scripts/build_human_value_queue.py``) against every reference group of batch A
(human-value-01, frozen test splits, probe-v1, annotation-v1 / v2 datasets, ``corpus/``) **and**
against every annotation-v3 dataset on disk (``debt-01``, ``contrast-*``, ``training-v*``, so
also ``contrast-04`` once it exists), the held-out sets and the 150 batch-A notes.

Output, deterministic (``--check`` compares the files on disk byte for byte):

* ``review-queue-b.jsonl``: ``{id, text, strata, review_group}`` for the 50 notes, order = seeded
  hash, ``id`` = ``hv02-<sha256(NFC text)[:12]>``;
* ``review-queue-all.jsonl``: the 150 notes of ``review-queue.jsonl`` (bytes unchanged) followed
  by the 50 of ``review-queue-b.jsonl``;
* ``manifest-b.json``: counts, strata, leakage drops, hashes.

Batch A's ``review-queue.jsonl`` and ``manifest.json`` are never touched. The set is never
trained on.
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


a = _load("build_human_value_02")
hv = a.hv

NAME = "human-value-02-b"
SEED = "human-value-02:b:v1"
OUT_DIR = Path("datasets/annotation-v3/human-value-02")
V3_DIR = Path("datasets/annotation-v3")
CANDIDATES_FILE = "candidates-b.txt"
QUEUE_A = "review-queue.jsonl"
QUEUE_B = "review-queue-b.jsonl"
QUEUE_ALL = "review-queue-all.jsonl"
MANIFEST_B = "manifest-b.json"
SECTION_TARGETS = {"weak": 22, "gift": 8, "debt": 9, "shop": 4, "other": 7}

# Leakage groups in priority order; the human-value-02 batch A queue is its own group and every
# annotation-v3 directory except human-value-02 (incl. contrast-04 once built) is a reference.
EXTRA_FROZEN = ("datasets/annotation-v3/training-v4/test.jsonl",)


def _v3_dirs(root: Path) -> tuple[str, ...]:
    return tuple(
        str(p.relative_to(root))
        for p in sorted((root / V3_DIR).iterdir())
        if p.is_dir() and p.name != OUT_DIR.name
    )


def groups(root: Path) -> tuple[tuple[str, tuple[str, ...]], ...]:
    out: list[tuple[str, tuple[str, ...]]] = []
    for name, entries in a.GROUPS:
        if name == "annotation-v3":
            entries = _v3_dirs(root)
        elif name == "frozen-test":
            entries = entries + EXTRA_FROZEN
        out.append((name, entries))
        if name == "human-value-01":
            out.append(("human-value-02-a", (str(OUT_DIR / QUEUE_A),)))
    return tuple(out)


def read_candidates(path: Path) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    section = None
    seen: set[str] = set()
    for raw in path.read_text("utf-8").splitlines():
        if raw.startswith("## "):
            section = raw[3:].strip()
            if section.removesuffix("-reserve") not in SECTION_TARGETS:
                raise ValueError(f"unknown section {section!r}")
            continue
        if not raw.strip() or section is None:
            raise ValueError(f"{path.name}: blank line or note before a section header")
        text = a._nfc(raw)
        if text != text.strip():
            raise ValueError(f"{path.name}: leading/trailing whitespace in {raw!r}")
        if text.lower() in seen:
            raise ValueError(f"{path.name}: duplicate note {raw!r}")
        seen.add(text.lower())
        out.append((section, text))
    return out


def order_key(record_id: str) -> str:
    return a._sha256(f"{SEED}:{record_id}".encode())


def build(root: Path) -> dict[str, bytes]:
    base = root / OUT_DIR
    candidates = read_candidates(base / CANDIDATES_FILE)
    gs = groups(root)
    saved = a.GROUPS
    a.GROUPS = gs  # leakage_filter / load_reference_groups walk the module-level group list
    try:
        # a nonexistent out_dir: no file is skipped as "own output"
        texts, files = a.load_reference_groups(root, Path("/nonexistent-human-value-02-b"))
        survivors, dropped, ref_sizes = a.leakage_filter(candidates, texts)
    finally:
        a.GROUPS = saved

    chosen: list[tuple[str, str]] = []
    surplus: list[str] = []
    for section, target in SECTION_TARGETS.items():
        # the main pool first (seeded hash order); a ``<section>-reserve`` pool only tops up a
        # section that lost candidates to the gate, so adding reserves never moves a chosen note
        pool = []
        for name in (section, f"{section}-reserve"):
            pool += sorted(
                (c for c in survivors if c[0] == name), key=lambda c: order_key(a.note_id(c[1]))
            )
        if len(pool) < target:
            raise SystemExit(f"section {section}: {len(pool)} survivors < {target}")
        chosen += pool[:target]
        surplus += [c[1] for c in pool[target:]]

    rows: list[dict[str, Any]] = []
    section_of: dict[str, str] = {}
    for section, text in chosen:
        strata, _ = hv.assign_strata(text)
        rid = a.note_id(text)
        section_of[rid] = section
        rows.append(
            {
                "id": rid,
                "text": text,
                "strata": strata or [a.FILL_STRATUM],
                "review_group": a.REVIEW_GROUP,
            }
        )
    if len({r["id"] for r in rows}) != len(rows):
        raise SystemExit("duplicate queue ids")
    rows.sort(key=lambda r: order_key(r["id"]))
    queue_b = a._dump_jsonl(rows)
    queue_a_bytes = (base / QUEUE_A).read_bytes()
    a_ids = {json.loads(line)["id"] for line in queue_a_bytes.decode("utf-8").splitlines()}
    if a_ids & {r["id"] for r in rows}:
        raise SystemExit("batch B overlaps batch A")
    queue_all = queue_a_bytes + queue_b

    manifest = {
        "name": NAME,
        "annotation_version": "annotation-v3",
        "role": "batch B of the human-value-02 TEST set (50 more notes); labelled by two "
        "independent LLM labellers, not by humans (see manifest-labels.json)",
        "never_train": True,
        "provenance": {
            "source": "llm-written",
            "writer": "an LLM (the assistant), by hand, from the annotation rules and the "
            "reviewed-corpus style only; no training-data file was read when writing",
            "candidates_file": CANDIDATES_FILE,
        },
        "seed": SEED,
        "id_rule": "hv02-<sha256(NFC text)[:12]>; order = sha256(seed:id)",
        "section_targets": SECTION_TARGETS,
        "counts": {
            "candidates": len(candidates),
            "queue": len(rows),
            "dropped_by_leakage_gate": len(dropped),
            "survived_gate": len(survivors),
            "not_selected_surplus": len(surplus),
            "queue_by_section": dict(sorted(Counter(section_of.values()).items())),
            "combined_queue": len(rows) + len(a_ids),
        },
        "leakage": {
            "rules": {
                "exact": "NFC + strip + lowercase",
                "folded": "NFC, lowercase, accents stripped, digits masked, whitespace collapsed",
                "sequence": f"difflib ratio of folded texts >= {hv.SEQ_THRESHOLD}",
                "token": f"folded token Jaccard >= {hv.TOKEN_THRESHOLD} (>= {hv.TOKEN_MIN} tokens)",
                "char3": f"folded char-3-gram Jaccard >= {hv.CHAR3_THRESHOLD}",
            },
            "reference_groups_in_priority_order": [name for name, _ in gs],
            "reference_unique_texts": ref_sizes,
            "reference_files": {name: len(f) for name, f in files.items()},
            "annotation_v3_dirs": list(_v3_dirs(root)),
            "dropped_by_group": dict(Counter(d["group"] for d in dropped)),
            "dropped_by_rule": dict(Counter(d["rule"] for d in dropped)),
            "dropped": dropped,
            "surplus_not_selected": sorted(surplus),
        },
        "strata": dict(sorted(Counter(s for r in rows for s in r["strata"]).items())),
        "outputs": {
            QUEUE_B: a._sha256(queue_b),
            QUEUE_ALL: a._sha256(queue_all),
        },
        "inputs": {
            CANDIDATES_FILE: a._sha256((base / CANDIDATES_FILE).read_bytes()),
            QUEUE_A: a._sha256(queue_a_bytes),
        },
    }
    return {
        QUEUE_B: queue_b,
        QUEUE_ALL: queue_all,
        MANIFEST_B: (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--check", action="store_true", help="verify files on disk, write nothing")
    args = parser.parse_args(argv)

    files = build(args.root)
    out = args.root / OUT_DIR
    if args.check:
        bad = [
            n
            for n, data in files.items()
            if not (out / n).exists() or (out / n).read_bytes() != data
        ]
        if bad:
            print(f"MISMATCH: {', '.join(bad)}", file=sys.stderr)
            return 1
        print(f"{NAME}: files match a fresh build")
        return 0
    for name, data in files.items():
        (out / name).write_bytes(data)
    manifest = json.loads(files[MANIFEST_B])
    summary = {k: manifest[k] for k in ("counts", "strata")}
    summary["dropped"] = [
        (d["text"], d["group"], d["rule"]) for d in manifest["leakage"]["dropped"]
    ]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
