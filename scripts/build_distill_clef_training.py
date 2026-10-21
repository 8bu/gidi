#!/usr/bin/env python
"""Build ``training-v6a`` / ``training-v6b``: training-v5 train + the Clef silver notes (A / B).

Silver notes are model labels (``provenance.annotator = "clef"``), appended unchanged after the
byte-identical training-v5 train records; exact duplicates of a training-v5 record (id or
NFC-lowercase text) are dropped and reported. A silver note that matches the frozen test,
probe-v1, human-value-01 or human-value-02 by id, verbatim or NFC-lowercase text, is a near
duplicate under the rules of ``build_annotation_v3_training_v5.py``, or reaches folded char-3
Jaccard >= 0.6 to a human-value-01 or -02 note is dropped and reported with its reason (the
score only; the held-out note is not read), as training-v5 did for inherited notes. After that
the full gate runs on the result and raises, writing nothing, on any hit. No other filtering.
``validation.jsonl`` and ``test.jsonl`` are
symlinks to the frozen annotation-v1 splits. ``--check`` rebuilds in memory and compares.
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
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v1 = _load("build_annotation_v3_training")
v2 = _load("build_annotation_v3_training_v2")
v4 = _load("build_annotation_v3_training_v4")
v5 = _load("build_annotation_v3_training_v5")
hv = v2.hv

SILVER = Path("datasets/annotation-v3/distill-clef-01")
PREV = Path("datasets/annotation-v3/training-v5")
ARMS = {"A": "training-v6a", "B": "training-v6b"}
PREV_RECORDS = 1123
SILVER_RECORDS = {"A": 952, "B": 1344}
KEYS = ("id", "text", "type", "target", "source_batch", "accented", "provenance", "group")


def check_record(record: dict[str, Any]) -> None:
    rid = record["id"]
    if set(record) != set(KEYS):
        raise ValueError(f"{rid}: unexpected keys {sorted(record)}")
    if record["type"] not in v1.TYPES:
        raise ValueError(f"{rid}: unknown type {record['type']!r}")
    if record["provenance"].get("annotator") != "clef":
        raise ValueError(f"{rid}: provenance.annotator is not 'clef'")
    target = record["target"]
    if target is not None and record["text"][target["start"] : target["end"]] != target["text"]:
        raise ValueError(f"{rid}: target span does not slice the note text")


def build_arm(root: Path, arm: str, prev_bytes: bytes, prev: list[dict[str, Any]], sets: dict):
    silver = v1.read_jsonl(root / SILVER / f"silver-{arm}.jsonl")
    if len(silver) != SILVER_RECORDS[arm]:
        raise ValueError(f"silver-{arm}: expected {SILVER_RECORDS[arm]} records, got {len(silver)}")
    for record in silver:
        check_record(record)
    prev_ids = {r["id"] for r in prev}
    prev_keys = {v1.exact_key(r["text"]) for r in prev}
    gate = {}
    for name, rows in sets.items():
        gate[name] = {
            "ids": {r["id"] for r in rows},
            "texts": {r["text"] for r in rows} | {v1.exact_key(r["text"]) for r in rows},
            "near": hv.ReferenceSet([r["text"] for r in rows]),
            "refs": [hv.Reference(r["text"]) for r in rows] if name.startswith("human") else [],
        }
    kept, dropped, seen = [], [], set()
    for record in silver:
        key = v1.exact_key(record["text"])
        reasons = []
        if record["id"] in prev_ids or key in prev_keys:
            reasons.append("duplicate of a training-v5 train record")
        if record["id"] in seen or key in seen:
            reasons.append("duplicate inside silver")
        for name, g in gate.items():
            if record["id"] in g["ids"]:
                reasons.append(f"id in {name}")
            if record["text"] in g["texts"] or key in g["texts"]:
                reasons.append(f"verbatim text in {name}")
            rule = g["near"].match(hv.Reference(record["text"]))
            if rule:
                reasons.append(f"near-duplicate ({rule}) of {name}")
            if g["refs"] and v4.max_char3(record, g["refs"]) >= v5.HV_CHAR3_LIMIT:
                reasons.append(f"char3 >= {v5.HV_CHAR3_LIMIT} to {name}")
        if reasons:
            dropped.append({"id": record["id"], "text": record["text"], "reasons": reasons})
            continue
        kept.append(record)
        seen.update((record["id"], key))
    records = prev + kept
    excluded = json.loads((root / v1.DEBT / "manifest.json").read_text("utf-8"))[
        "excluded_human_value_01_ids"
    ]
    v1.assert_no_leakage(records, sets, excluded)
    sizes = v2.assert_no_near_duplicates(kept, sets)
    char3 = {}
    for name in ("human-value-01", "human-value-02"):
        refs = gate[name]["refs"]
        scores = {r["id"]: v4.max_char3(r, refs) for r in kept}
        over = [f"{i}: {s:.2f}" for i, s in scores.items() if s >= v5.HV_CHAR3_LIMIT]
        if over:
            raise ValueError(f"{name} char3 bound: " + "; ".join(over))
        char3[name] = round(max(scores.values()), 4)
    body = prev_bytes + b"".join(
        (json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8") for r in kept
    )
    manifest = {
        "name": ARMS[arm],
        "annotation_version": "annotation-v3",
        "purpose": (
            f"clef-distill-v1 arm {arm}: training-v5 train + Clef silver-{arm} (model labels, "
            "provenance.annotator = clef, NOT human labels); validation is in-sample, the "
            "frozen test split, probe-v1, human-value-01 and human-value-02 stay held out"
        ),
        "counts": {
            "training_v5_train": len(prev),
            f"silver_{arm}_records": len(silver),
            "silver_dropped": len(dropped),
            "silver_added": len(kept),
            "train": len(records),
        },
        "silver_share": round(len(kept) / len(records), 4),
        "type_counts": v1.type_counts(records),
        "type_counts_by_source": {
            "training_v5_train": v1.type_counts(prev),
            f"silver_{arm}": v1.type_counts(kept),
        },
        "source_counts": dict(Counter(r["source_batch"] for r in records)),
        "annotator_counts": dict(Counter(r["provenance"]["annotator"] for r in records)),
        "silver_dropped": dropped,
        "leakage_gate": {
            "sets": sizes,
            "rule": "no id, no verbatim or NFC+strip+lowercase text of any train record in a "
            "set; no silver note a near duplicate (scripts/build_human_value_queue.py rules) of "
            "any note of a set; no silver note at folded char-3 Jaccard >= "
            f"{v5.HV_CHAR3_LIMIT} to a human-value-01 or human-value-02 note. The training-v5 "
            "records are byte-identical to training-v5 (gated there). Silver notes failing any "
            "rule (or an exact duplicate) are dropped, see silver_dropped",
            "max_char3_silver_notes": char3,
            "result": "pass",
        },
        "inputs_sha256": {
            str(p): v1.sha256_bytes((root / p).read_bytes())
            for p in (
                PREV / "train.jsonl",
                PREV / "manifest.json",
                SILVER / f"silver-{arm}.jsonl",
                SILVER / "manifest.json",
                *v5.HV02_QUEUES,
            )
        },
        "outputs_sha256": {"train.jsonl": v1.sha256_bytes(body)},
        "evaluation_sets": (
            "validation.jsonl (in-sample, logging only) and test.jsonl are symlinks to the "
            "frozen annotation-v1 splits"
        ),
    }
    return body, manifest


def build(root: Path) -> dict[str, tuple[bytes, dict[str, Any]]]:
    """Verify every precondition and return ``{arm: (train bytes, manifest)}``; writes nothing."""
    for rel in v5.HV02_QUEUES:
        if not (root / rel).is_file():
            raise ValueError(f"{rel} is missing: the leakage gate needs human-value-02")
    prev_bytes = (root / PREV / "train.jsonl").read_bytes()
    prev = [json.loads(line) for line in prev_bytes.decode("utf-8").splitlines()]
    if len(prev) != PREV_RECORDS:
        raise ValueError("unexpected training-v5 size")
    sets = v1.held_out(root)
    sets["human-value-02"] = [r for rel in v5.HV02_QUEUES for r in v1.read_jsonl(root / rel)]
    if len(sets["human-value-02"]) < 190:
        raise ValueError("human-value-02 queues look truncated")
    return {arm: build_arm(root, arm, prev_bytes, prev, sets) for arm in ARMS}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    root = args.root
    try:
        built = build(root)
    except (ValueError, OSError, KeyError) as exc:
        print(f"distill-clef: {exc}", file=sys.stderr)
        return 1
    links = {
        n: Path("..") / ".." / "annotation-v1" / "splits" / f"{n}.jsonl"
        for n in ("validation", "test")
    }
    problems = []
    for arm, (body, manifest) in built.items():
        out = root / "datasets/annotation-v3" / ARMS[arm]
        text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        if args.check:
            if not (out / "train.jsonl").is_file() or (out / "train.jsonl").read_bytes() != body:
                problems.append(f"{ARMS[arm]}: train.jsonl differs from the rebuild")
            mf = out / "manifest.json"
            if not mf.is_file() or mf.read_text("utf-8") != text:
                problems.append(f"{ARMS[arm]}: manifest.json differs from the rebuild")
            for name in links:
                link = out / f"{name}.jsonl"
                if link.resolve() != (root / v1.SPLITS / f"{name}.jsonl").resolve():
                    problems.append(f"{link} is not a link to the frozen {name} split")
            continue
        out.mkdir(parents=True, exist_ok=True)
        (out / "train.jsonl").write_bytes(body)
        for name, target in links.items():
            link = out / f"{name}.jsonl"
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(target)
        (out / "manifest.json").write_text(text, encoding="utf-8")
        print(f"wrote {out}: {manifest['counts']}")
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    if args.check:
        print(
            "distill-clef check OK: " + str({ARMS[a]: m["counts"] for a, (_, m) in built.items()})
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
