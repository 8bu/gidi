#!/usr/bin/env python
"""Build ``datasets/annotation-v3/training-v4``: training-v2 + most of contrast-02 + contrast-03.

Usage:
    uv run python scripts/build_annotation_v3_training_v4.py [--root R] [--check]

``train.jsonl`` = the 905 byte-identical records of ``datasets/annotation-v3/training-v2`` + the
``contrast-02`` notes of ``training-v3`` **except** the ones the retrain-v3 report suspects of
shortening targets or pulling shop expenses to income (``REMOVE_RULES``, decided from the
training notes only; the removed ids and the reason are listed in ``manifest.json``) + the
``complete`` notes of ``datasets/annotation-v3/contrast-03`` (targets: shops, no target for gift
receivers / product brands / a bank used as channel, two-word kinship+role targets).
``validation.jsonl`` and ``test.jsonl`` are symlinks to the frozen annotation-v1 splits, as in
training-v2 / -v3.

Leakage gate (raises, writes nothing) against the frozen test split, probe-v1,
``datasets/annotation-v2/human-value-01`` **and ``datasets/annotation-v3/human-value-02``**:
no id and no exact text (verbatim or NFC + strip + lowercase) of any train record in a set; no
train record a near duplicate (the rules of ``scripts/build_human_value_queue.py``) of any note
of a set; no contrast-02 / contrast-03 note at folded char-3-gram Jaccard >= ``HV_CHAR3_LIMIT``
to a human-value-01 or human-value-02 note. human-value-02 is a test set: its texts are only
scored by this gate (never printed or stored); the manifest keeps its file hash. ``--check``
rebuilds in memory and compares the files on disk.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import unicodedata
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
v3 = _load("build_annotation_v3_training_v3")
c3 = _load("build_contrast_03")
hv = v2.hv

NAME = "training-v4"
OUT = Path("datasets/annotation-v3/training-v4")
BASE = Path("datasets/annotation-v3/training-v2")
PREV = Path("datasets/annotation-v3/training-v3")
CONTRAST_02 = Path("datasets/annotation-v3/contrast-02")
CONTRAST_03 = Path("datasets/annotation-v3/contrast-03")
HV02_QUEUE = Path("datasets/annotation-v3/human-value-02/review-queue.jsonl")
BASE_RECORDS = 905
PREV_RECORDS = 1005
HV_CHAR3_LIMIT = 0.6
INPUTS = (
    BASE / "train.jsonl",
    BASE / "manifest.json",
    PREV / "train.jsonl",
    PREV / "manifest.json",
    CONTRAST_03 / "queue.jsonl",
    CONTRAST_03 / "labels.jsonl",
    CONTRAST_03 / "provenance.jsonl",
    CONTRAST_03 / "manifest.json",
    v1.DEBT / "manifest.json",
    v1.HUMAN / "labels.jsonl",
    v1.HUMAN / "review-queue.jsonl",
    v1.HUMAN / "freeze.json",
    v1.PROBE / "notes.jsonl",
    HV02_QUEUE,
)

# contrast-02 notes removed from training-v3 (decided from the training notes only):
BRAND_INCOME_TARGETS = {"grab", "shopee", "cho tot", "chotot", "lazada", "tiki", "sendo", "tiktok"}
GENERIC_PEER_TARGETS = {"ban", "dong nghiep"}
REMOVE_RULES = {
    "brand-income": "an income note whose target is a brand / platform name (grab, shopee, "
    "chợ tốt, ...): the pattern `brand + amount -> income` that turned shop expense notes into "
    "income in retrain-v3",
    "generic-peer-noun": "the whole target is a bare generic peer noun (bạn, đồng nghiệp) that "
    "also prefixes a given name: it taught the encoder to stop at the first word of a role "
    "phrase / to tag `bạn` instead of the name",
    "char3-human-value-02": f"folded char-3-gram Jaccard >= {HV_CHAR3_LIMIT} to a "
    "human-value-02 note (score only; the note itself was not read)",
}


def fold(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower().replace("đ", "d")
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def contrast_03_records(root: Path) -> list[dict[str, Any]]:
    """The ``complete`` contrast-03 labels as training records, in labels.jsonl order."""
    text_of = {r["id"]: r["text"] for r in v1.read_jsonl(root / CONTRAST_03 / "queue.jsonl")}
    annotator = {
        r["id"]: r["annotator"] for r in v1.read_jsonl(root / CONTRAST_03 / "provenance.jsonl")
    }
    records = []
    for label in v1.read_jsonl(root / CONTRAST_03 / "labels.jsonl"):
        if label["annotation_status"] != "complete":
            continue
        rid, text = label["id"], text_of[label["id"]]
        if label["type"] not in v1.TYPES:
            raise ValueError(f"{rid}: contrast-03 label has type {label['type']!r}")
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
                "source_batch": "contrast-03",
                "accented": v1.accented(text),
                "provenance": {"annotator": annotator[rid], "source_batch": "contrast-03"},
                "group": f"contrast-03:{rid}",
            }
        )
    return records


def max_char3(record: dict[str, Any], refs: list[Any]) -> float:
    ref = hv.Reference(record["text"])
    return max(hv._jaccard(ref.char3, other.char3) for other in refs)


def removal_reason(record: dict[str, Any], hv02_refs: list[Any]) -> str | None:
    """Why a contrast-02 record is dropped from training-v4 (a key of ``REMOVE_RULES``), if so."""
    target = fold(record["target"]["text"]) if record["target"] else None
    if record["type"] == "income" and target in BRAND_INCOME_TARGETS:
        return "brand-income"
    if target in GENERIC_PEER_TARGETS:
        return "generic-peer-noun"
    if hv02_refs and max_char3(record, hv02_refs) >= HV_CHAR3_LIMIT:
        return "char3-human-value-02"
    return None


def build(root: Path) -> tuple[bytes, dict[str, Any]]:
    """Verify every precondition and return ``(train.jsonl bytes, manifest)``; writes nothing."""
    hv02_path = root / HV02_QUEUE
    if not hv02_path.is_file():
        raise ValueError(f"{HV02_QUEUE} is missing: the leakage gate needs human-value-02")
    base_bytes = (root / BASE / "train.jsonl").read_bytes()
    base = [json.loads(line) for line in base_bytes.decode("utf-8").splitlines()]
    prev = [
        json.loads(line) for line in (root / PREV / "train.jsonl").read_text("utf-8").splitlines()
    ]
    if len(base) != BASE_RECORDS or len(prev) != PREV_RECORDS or prev[:BASE_RECORDS] != base:
        raise ValueError("training-v3 must be training-v2 train + 100 contrast-02 records")
    contrast_02 = prev[BASE_RECORDS:]
    contrast_03 = contrast_03_records(root)
    excluded = json.loads((root / v1.DEBT / "manifest.json").read_text("utf-8"))[
        "excluded_human_value_01_ids"
    ]
    if not contrast_03:
        raise ValueError("contrast-03 has no complete note")

    sets = v1.held_out(root)
    sets["human-value-02"] = v1.read_jsonl(hv02_path)
    if len(sets["human-value-02"]) < 100:
        raise ValueError("human-value-02 queue looks truncated")
    hv02_refs = [hv.Reference(r["text"]) for r in sets["human-value-02"]]

    # removal rules for contrast-02, from the training notes (and the human-value-02 gate)
    kept_02, removed = [], []
    for record in contrast_02:
        reason = removal_reason(record, hv02_refs)
        if reason:
            removed.append(
                {
                    "id": record["id"],
                    "text": record["text"],
                    "type": record["type"],
                    "target": record["target"]["text"] if record["target"] else None,
                    "reason": reason,
                }
            )
        else:
            kept_02.append(record)

    records = base + kept_02 + contrast_03
    v1.assert_no_leakage(records, sets, excluded)
    new = kept_02 + contrast_03
    sizes = v2.assert_no_near_duplicates(new, sets)  # new notes, every set
    # the base set is templated corpus text: it is near-duplicated by earlier held-out notes
    # (as in every earlier run; informational) but must not be by human-value-02
    v2.assert_no_near_duplicates(base, {"human-value-02": sets["human-value-02"]})
    base_near_dup = {}
    for name, rows in sets.items():
        ref_set = hv.ReferenceSet([r["text"] for r in rows])
        base_near_dup[name] = sum(bool(ref_set.match(hv.Reference(r["text"]))) for r in base)
    char3 = {}
    for name in ("human-value-01", "human-value-02"):
        refs = [hv.Reference(r["text"]) for r in sets[name]]
        scores = {r["id"]: max_char3(r, refs) for r in new}
        over = [f"{i}: {s:.2f}" for i, s in scores.items() if s >= HV_CHAR3_LIMIT]
        if over:
            raise ValueError(f"{name} char3 bound: " + "; ".join(over))
        char3[name] = round(max(scores.values()), 4)
    base_char3 = {}
    for name in ("human-value-01", "human-value-02"):
        refs = [hv.Reference(r["text"]) for r in sets[name]]
        base_char3[name] = sum(max_char3(r, refs) >= HV_CHAR3_LIMIT for r in base)

    body = base_bytes + b"".join(
        (json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8") for r in new
    )
    manifest = {
        "name": NAME,
        "annotation_version": "annotation-v3",
        "purpose": (
            "encoder-1 retrain set v4: training-v2 train (905, byte-identical) + the contrast-02 "
            "notes of training-v3 minus the suspect ones + contrast-03 (target-span notes: "
            "shops, gift receivers / product brands / bank channel without target, two-word "
            "kinship+role targets); validation is in-sample, the frozen test split, probe-v1, "
            "human-value-01 and human-value-02 stay held out"
        ),
        "counts": {
            "training_v2_train": len(base),
            "contrast_02_in_training_v3": len(contrast_02),
            "contrast_02_removed": len(removed),
            "contrast_02_kept": len(kept_02),
            "contrast_03_complete": len(contrast_03),
            "train": len(records),
        },
        "type_counts": v1.type_counts(records),
        "type_counts_by_source": {
            "training_v2_train": v1.type_counts(base),
            "contrast_02_kept": v1.type_counts(kept_02),
            "contrast_03": v1.type_counts(contrast_03),
        },
        "source_counts": dict(Counter(r["source_batch"] for r in records)),
        "annotator_counts": dict(Counter(r["provenance"]["annotator"] for r in records)),
        "contrast_02_removed_rules": REMOVE_RULES,
        "contrast_02_removed": removed,
        "excluded_human_value_01_ids": excluded,
        "leakage_gate": {
            "sets": sizes,
            "human_value_02_review_queue_sha256": v1.sha256_bytes(hv02_path.read_bytes()),
            "rule": "no id, no verbatim or NFC+strip+lowercase text of any train record in a "
            "set; no new (contrast) note is a near duplicate (exact, folded, sequence >= 0.9, "
            "token, char3 >= 0.8: scripts/build_human_value_queue.py) of any note of a set, and "
            "no base note of a human-value-02 note; no "
            f"contrast-02 / contrast-03 note has folded char-3-gram Jaccard >= {HV_CHAR3_LIMIT} "
            "to a human-value-01 or human-value-02 note",
            "max_char3_new_notes": char3,
            "base_notes_near_duplicate_of_set": base_near_dup,
            "base_notes_at_char3_ge_limit": base_char3,
            "base_note": "the 905 training-v2 notes are byte-identical to the earlier runs (same "
            "templated corpus text, not edited here): earlier held-out notes have near "
            "duplicates among them (counts above, informational, as in every earlier run); none "
            "is a near duplicate of human-value-02; some reach the stricter char-3 bound (counts "
            "above), which is applied to the new notes only",
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
