#!/usr/bin/env python
r"""Build ``datasets/annotation-v3/contrast-02``: LLM-composed notes for the confused type pairs.

Usage:
    uv run python scripts/build_contrast_02.py [--root R] [--out-dir D] [--check]

The retrain-v2 encoder still misreads five families of notes on human-value-01: income read as
expense / refund (allowance, bonus, writing fee, customer pays), purchases at shops / apps /
restaurants read as transfer / refund, installment and app-loan repayments read as expense or
repayment_in, paying for someone (``mua hộ / trả hộ / ứng cho X``, ``lend``) read as
repayment_out, and debt-only notes (``X còn thiếu ...``, ``chưa trả ...``) read as expense.
``source.jsonl`` is written by hand (an LLM, user-approved) from ``docs/annotation-v1.md``,
``docs/annotation-v3.md`` and ``configs/annotation-v3.yaml``, never from a held-out note:
``{text, type, target, value, family, cue}`` with ``target`` / ``value`` the exact substrings of
``text`` (first occurrence, on a word boundary). Labels are LLM-made: ``provenance.annotator`` is
``llm``.

Candidates are dropped, in this order and each counted once, when they are
(1) an exact / near duplicate of a held-out note, the existing corpus, ``training-v2`` or an
earlier candidate (``scripts/build_debt_relabel_queue.py`` leakage groups, rules of
``scripts/build_human_value_queue.py``), (2) at folded char-3-gram Jaccard >=
``HV_CHAR3_LIMIT`` of a human-value-01 note (a stricter bound than the near-duplicate rules), or
(3) above the per-family quota (round-robin over the ``cue`` tags). Outputs, all deterministic:
``queue.jsonl``, ``labels.jsonl`` (annotation-v3 combined labels, ``complete``),
``provenance.jsonl`` and ``manifest.json``. ``--check`` rebuilds in memory and compares the
files on disk byte for byte.
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


debt = _load("build_debt_relabel_queue")
c1 = _load("build_contrast_01")

NAME = "contrast-02"
DEFAULT_OUT_DIR = Path("datasets/annotation-v3/contrast-02")
SCHEMA = debt.SCHEMA
CONTRACT = debt.CONTRACT
SOURCE_FILE = "source.jsonl"
QUEUE_FILE = "queue.jsonl"
LABELS_FILE = "labels.jsonl"
PROVENANCE_FILE = "provenance.jsonl"
MANIFEST_FILE = "manifest.json"
SOURCE_TAG = "contrast-02-generated"
GENERATOR = "composed by the assistant (LLM) from docs/annotation-v1.md + docs/annotation-v3.md"
TRAIN_V2 = Path("datasets/annotation-v3/training-v2/train.jsonl")

# Per-family quota of kept notes; filled round-robin over the ``cue`` tags (source order inside
# a cue), so no single phrasing takes a family.
QUOTA = {"income": 25, "expense": 25, "repayment_out": 20, "lend": 15, "debt": 15}
FAMILY_TYPES = {
    "income": {"income"},
    "expense": {"expense"},
    "repayment_out": {"repayment_out"},
    "lend": {"lend"},
    "debt": {"borrow", "lend"},
}
# Extra human-value-01 bound on top of the near-duplicate rules (char-3-gram Jaccard).
HV_CHAR3_LIMIT = 0.6


def note_id(text: str) -> str:
    return f"{NAME}-{debt._sha256(debt._nfc(text).encode('utf-8'))[:12]}"


def load_source(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
    seen: set[str] = set()
    for row in rows:
        extra = set(row) - {"text", "type", "target", "value", "family", "cue"}
        if (
            extra
            or row.get("family") not in QUOTA
            or row.get("type") not in FAMILY_TYPES[row.get("family")]
        ):
            raise ValueError(f"{SOURCE_FILE}: bad row {row!r}")
        text = debt._nfc(row["text"])
        if text != row["text"] or text != text.strip() or not text:
            raise ValueError(f"{SOURCE_FILE}: text must be NFC and stripped: {row['text']!r}")
        if text in seen:
            raise ValueError(f"{SOURCE_FILE}: duplicate text {text!r}")
        seen.add(text)
    return rows


def pick_by_cue(texts: list[str], by_text: dict[str, dict[str, Any]]) -> set[str]:
    """Fill ``QUOTA`` per family, taking one note per ``cue`` in turn (source order in a cue)."""
    chosen: set[str] = set()
    for family, quota in QUOTA.items():
        by_cue: dict[str, list[str]] = {}
        for text in texts:
            if by_text[text]["family"] == family:
                by_cue.setdefault(by_text[text]["cue"], []).append(text)
        queues = list(by_cue.values())
        taken = 0
        while taken < quota and any(queues):
            for queue in queues:
                if queue and taken < quota:
                    chosen.add(queue.pop(0))
                    taken += 1
    return chosen


def max_char3(hv: Any, text: str, refs: list[Any]) -> float:
    ref = hv.Reference(text)
    return max((hv._jaccard(ref.char3, other.char3) for other in refs), default=0.0)


def build(root: Path, out_dir: Path) -> dict[str, str]:
    """All output files of the batch as ``{file name: content}`` (nothing is written)."""
    from gidi.annotation.combined import load_combined_config, validate_combined_label

    hv = debt._load_hv()
    source = load_source(out_dir / SOURCE_FILE)
    by_text = {r["text"]: r for r in source}
    groups = debt.load_reference_groups(root, out_dir, [])
    survivors, dropped, ref_sizes = debt.leakage_filter(hv, list(by_text), groups)

    # training-v2 is a derived set (not in the corpus references): drop duplicates of it too.
    train_v2 = hv.ReferenceSet(debt._texts(root / TRAIN_V2))
    ref_sizes["training-v2"] = len(train_v2.refs)
    fresh = []
    for text in survivors:
        rule = train_v2.match(hv.Reference(text))
        if rule:
            dropped.append({"text": text, "group": "training-v2", "rule": rule})
        else:
            fresh.append(text)

    hv_refs = [hv.Reference(t) for t in groups["human-value-01"]]
    hv_max: dict[str, float] = {}
    eligible: list[str] = []
    for text in fresh:
        score = max_char3(hv, text, hv_refs)
        hv_max[text] = score
        if score >= HV_CHAR3_LIMIT:
            dropped.append(
                {"text": text, "group": "human-value-01-char3", "rule": f"char3 {score:.2f}"}
            )
        else:
            eligible.append(text)
    chosen = pick_by_cue(eligible, by_text)
    kept = [text for text in eligible if text in chosen]
    dropped += [
        {"text": text, "group": "over-quota", "rule": by_text[text]["family"]}
        for text in eligible
        if text not in chosen
    ]

    config = load_combined_config(root / debt.SCHEMA)
    queue, labels, provenance, problems = [], [], [], []
    for text in kept:
        row, rid = by_text[text], note_id(text)
        label = {
            "id": rid,
            "annotation_status": "complete",
            "type": row["type"],
            "target": c1.span(text, row["target"], "target"),
            "value": c1.span(text, row["value"], "value"),
            "span_status": {"value": "complete"},
        }
        problems += [f"{rid}: {p}" for p in validate_combined_label(label, text, config)]
        queue.append({"id": rid, "text": text, "source": SOURCE_TAG})
        labels.append(label)
        provenance.append(
            {
                "id": rid,
                "source": SOURCE_TAG,
                "annotator": "llm",
                "generator": GENERATOR,
                "family": row["family"],
                "cue": row["cue"],
            }
        )
    if problems:
        raise ValueError("labels invalid:\n" + "\n".join(problems))
    if len({q["id"] for q in queue}) != len(queue):
        raise ValueError("duplicate queue ids")

    files = {
        QUEUE_FILE: debt._dump(queue),
        LABELS_FILE: debt._dump(labels),
        PROVENANCE_FILE: debt._dump(provenance),
    }
    files[MANIFEST_FILE] = (
        json.dumps(
            manifest(
                root, out_dir, files, source, queue, labels, provenance, hv_max, dropped, ref_sizes
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    return files


def manifest(
    root: Path,
    out_dir: Path,
    files: dict[str, str],
    source: list[dict[str, Any]],
    queue: list[dict[str, str]],
    labels: list[dict[str, Any]],
    provenance: list[dict[str, Any]],
    hv_max: dict[str, float],
    dropped: list[dict[str, str]],
    ref_sizes: dict[str, int],
) -> dict[str, Any]:
    accented = [any(ord(c) > 127 for c in q["text"]) for q in queue]
    sha = {
        name: debt._sha256(files[name].encode("utf-8"))
        for name in (QUEUE_FILE, LABELS_FILE, PROVENANCE_FILE)
    }
    sha[SOURCE_FILE] = debt._sha256((out_dir / SOURCE_FILE).read_bytes())
    return {
        "name": NAME,
        "annotation_version": "annotation-v3",
        "role": "LLM-composed notes for the confused type pairs of retrain-v2: income, shop / "
        "app / restaurant expenses (incl. jewelry as a gift), installment and app-loan "
        "repayment_out, lend on behalf of someone, debt-only notes in both directions",
        "annotator": "llm",
        "annotator_note": "labels are LLM-made (user-approved), not human labels; "
        "provenance.jsonl marks every note `annotator: llm`",
        "schema": str(SCHEMA),
        "schema_sha256": debt._sha256((root / SCHEMA).read_bytes()),
        "contract": str(CONTRACT),
        "contract_sha256": debt._sha256((root / CONTRACT).read_bytes()),
        "counts": {
            "candidates": len(source),
            "kept": len(queue),
            "dropped": len(dropped),
            "type": dict(sorted(Counter(label["type"] for label in labels).items())),
            "family": dict(sorted(Counter(p["family"] for p in provenance).items())),
            "quota": QUOTA,
            "cue": dict(sorted(Counter(p["cue"] for p in provenance).items())),
            "target_null": sum(label["target"] is None for label in labels),
            "value_null": sum(label["value"] is None for label in labels),
            "accented": sum(accented),
            "unaccented": len(accented) - sum(accented),
            "unaccented_share": round(1 - sum(accented) / len(accented), 4),
        },
        "leakage": {
            "rules": {
                "exact": "NFC + strip + lowercase",
                "folded": "NFC, lowercase, accents stripped, digits masked, whitespace collapsed",
                "sequence": "difflib ratio of folded texts >= 0.9",
                "token": "folded token Jaccard >= 0.8 (>= 4 tokens)",
                "char3": "folded char-3-gram Jaccard >= 0.8",
                "hv_char3_limit": f"folded char-3-gram Jaccard >= {HV_CHAR3_LIMIT} to any "
                "human-value-01 note: dropped",
                "source": "scripts/build_human_value_queue.py (via build_debt_relabel_queue.py)",
            },
            "group_order": [
                *debt.GROUPS,
                "earlier-generated",
                "training-v2",
                "human-value-01-char3",
                "over-quota",
            ],
            "reference_texts": ref_sizes,
            "dropped_by_group": dict(sorted(Counter(d["group"] for d in dropped).items())),
            "dropped_by_rule": dict(sorted(Counter(d["rule"] for d in dropped).items())),
            "dropped": dropped,
            "max_hv_char3_kept": round(max(hv_max[q["text"]] for q in queue), 4),
        },
        "files_sha256": sha,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--check", action="store_true", help="compare, do not write")
    args = parser.parse_args(argv)
    out_dir = args.root / args.out_dir
    try:
        files = build(args.root, out_dir)
    except (ValueError, OSError) as exc:
        print(f"{NAME}: {exc}", file=sys.stderr)
        return 1
    if args.check:
        stale = [
            name
            for name, content in files.items()
            if not (out_dir / name).is_file() or (out_dir / name).read_text("utf-8") != content
        ]
        if stale:
            print(f"STALE: {', '.join(stale)}", file=sys.stderr)
            return 1
        print(f"ok: {len(files)} files match ({NAME})")
        return 0
    for name, content in files.items():
        (out_dir / name).write_text(content, encoding="utf-8")
    manifest_json = json.loads(files[MANIFEST_FILE])
    print(json.dumps(manifest_json["counts"], ensure_ascii=False, indent=2))
    print("dropped:", json.dumps(manifest_json["leakage"]["dropped_by_group"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
