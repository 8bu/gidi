#!/usr/bin/env python
r"""Build ``datasets/annotation-v3/contrast-01``: LLM-composed debt-word contrast notes.

Usage:
    uv run python scripts/build_contrast_01.py [--root R] [--out-dir D] [--check]

The annotation-v3 retrain (``experiments/annotation-v3-retrain``) confused debt state with
repayment / refund / expense. This batch adds notes where the same ``nợ / trả / hoàn / để riêng``
words carry the *other* meaning, written from the annotation-v1 / v3 rules (never from a test,
probe-v1 or human-value-01 note). ``source.jsonl`` is written by hand (an LLM, user-approved):
``{text, type, target, value, cue}`` with ``target`` / ``value`` the exact substrings of ``text``
(first occurrence, on a word boundary). Labels are LLM-made: ``provenance.annotator`` is ``llm``.

Candidates are dropped, in this order and each counted once, when they are
(1) an exact / near duplicate of any held-out note, the existing corpus or an earlier candidate
(``scripts/build_debt_relabel_queue.py`` leakage groups and the rules of
``scripts/build_human_value_queue.py``), (2) too close to one of the 4 regression notes of the
retrain report (folded difflib ratio >= ``REGRESSION_RATIO``), or (3) above the per-type quota
(source order). Outputs, all deterministic: ``queue.jsonl`` ``{id, text, source}``,
``labels.jsonl`` (annotation-v3 combined labels, ``complete``), ``provenance.jsonl`` and
``manifest.json``. ``--check`` rebuilds in memory and compares the files on disk byte for byte.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
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

NAME = "contrast-01"
DEFAULT_OUT_DIR = Path("datasets/annotation-v3/contrast-01")
SCHEMA = debt.SCHEMA
CONTRACT = debt.CONTRACT
SOURCE_FILE = "source.jsonl"
QUEUE_FILE = "queue.jsonl"
LABELS_FILE = "labels.jsonl"
PROVENANCE_FILE = "provenance.jsonl"
MANIFEST_FILE = "manifest.json"
SOURCE_TAG = "contrast-01-generated"
GENERATOR = "composed by the assistant (LLM) from docs/annotation-v1.md + docs/annotation-v3.md"

# Per-type quota of kept notes: repayment_out / repayment_in ~30, refund ~15, transfer ~18 (the
# set-aside / withdraw cues are the weakest class) and 10 expense notes that use a trả-word
# without a debt (trả tiền điện, trả tiền ăn). Within a type the quota is filled round-robin
# over the ``cue`` tags (source order inside a cue), so no single phrasing takes it all.
QUOTA = {
    "repayment_out": 30,
    "repayment_in": 30,
    "refund": 15,
    "transfer": 18,
    "expense": 10,
}
# Frozen-test notes whose type flipped in the retrain: candidates must not paraphrase them.
REGRESSION_ID_SUFFIXES = ("b590dd3df660", "07ca1f56011d", "84622e5a18da", "9d98b78854de")
REGRESSION_RATIO = 0.75
TEST_SPLIT = Path("datasets/annotation-v1/splits/test.jsonl")
_WORD = re.compile(r"\w", re.UNICODE)


def note_id(text: str) -> str:
    return f"{NAME}-{debt._sha256(debt._nfc(text).encode('utf-8'))[:12]}"


def span(text: str, piece: str | None, what: str) -> dict[str, Any] | None:
    """``{text, start, end}`` of the first occurrence of ``piece``, on a word boundary."""
    if piece is None:
        return None
    start = text.find(piece)
    if start < 0:
        raise ValueError(f"{what} {piece!r} not found in {text!r}")
    end = start + len(piece)
    if (start and _WORD.match(text[start - 1]) and _WORD.match(piece[0])) or (
        end < len(text) and _WORD.match(text[end]) and _WORD.match(piece[-1])
    ):
        raise ValueError(f"{what} {piece!r} is not on a word boundary in {text!r}")
    return {"text": piece, "start": start, "end": end}


def load_source(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
    seen: set[str] = set()
    for row in rows:
        extra = set(row) - {"text", "type", "target", "value", "cue"}
        if extra or row.get("type") not in QUOTA:
            raise ValueError(f"{SOURCE_FILE}: bad row {row!r}")
        text = debt._nfc(row["text"])
        if text != row["text"] or text != text.strip() or not text:
            raise ValueError(f"{SOURCE_FILE}: text must be NFC and stripped: {row['text']!r}")
        if text in seen:
            raise ValueError(f"{SOURCE_FILE}: duplicate text {text!r}")
        seen.add(text)
    return rows


def regression_texts(root: Path) -> list[str]:
    rows = [json.loads(line) for line in (root / TEST_SPLIT).read_text("utf-8").splitlines()]
    texts = [r["text"] for r in rows if r["id"].endswith(REGRESSION_ID_SUFFIXES)]
    if len(texts) != len(REGRESSION_ID_SUFFIXES):
        raise ValueError(
            f"expected {len(REGRESSION_ID_SUFFIXES)} regression notes in the test split"
        )
    return texts


def regression_ratio(hv: Any, text: str, refs: list[str]) -> float:
    folded = hv.fold(text)
    return max(SequenceMatcher(None, folded, hv.fold(ref)).ratio() for ref in refs)


def pick_by_cue(texts: list[str], by_text: dict[str, dict[str, Any]]) -> set[str]:
    """Fill ``QUOTA`` per type, taking one note per ``cue`` in turn (source order within a cue)."""
    chosen: set[str] = set()
    for kind, quota in QUOTA.items():
        by_cue: dict[str, list[str]] = {}
        for text in texts:
            if by_text[text]["type"] == kind:
                by_cue.setdefault(by_text[text]["cue"], []).append(text)
        queues = list(by_cue.values())
        taken = 0
        while taken < quota and any(queues):
            for queue in queues:
                if queue and taken < quota:
                    chosen.add(queue.pop(0))
                    taken += 1
    return chosen


def build(root: Path, out_dir: Path) -> dict[str, str]:
    """All output files of the batch as ``{file name: content}`` (nothing is written)."""
    from gidi.annotation.combined import load_combined_config, validate_combined_label

    hv = debt._load_hv()
    source = load_source(out_dir / SOURCE_FILE)
    by_text = {r["text"]: r for r in source}
    groups = debt.load_reference_groups(root, out_dir, [])
    survivors, dropped, ref_sizes = debt.leakage_filter(hv, list(by_text), groups)

    refs = regression_texts(root)
    ratios: dict[str, float] = {}
    eligible: list[str] = []
    for text in survivors:
        ratio = regression_ratio(hv, text, refs)
        ratios[text] = ratio
        if ratio >= REGRESSION_RATIO:
            dropped.append(
                {"text": text, "group": "regression-notes", "rule": f"ratio {ratio:.2f}"}
            )
        else:
            eligible.append(text)
    chosen = pick_by_cue(eligible, by_text)
    kept = [text for text in eligible if text in chosen]
    dropped += [
        {"text": text, "group": "over-quota", "rule": by_text[text]["type"]}
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
            "target": span(text, row["target"], "target"),
            "value": span(text, row["value"], "value"),
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
                root, out_dir, files, source, queue, labels, provenance, ratios, dropped, ref_sizes
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
    ratios: dict[str, float],
    dropped: list[dict[str, str]],
    ref_sizes: dict[str, int],
) -> dict[str, Any]:
    accented = [any(ord(c) > 127 for c in q["text"]) for q in queue]
    sha = {
        name: debt._sha256(files[name].encode("utf-8")) for name in (QUEUE_FILE, LABELS_FILE)
    } | {PROVENANCE_FILE: debt._sha256(files[PROVENANCE_FILE].encode("utf-8"))}
    sha[SOURCE_FILE] = debt._sha256((out_dir / SOURCE_FILE).read_bytes())
    kept_ratios = [ratios[q["text"]] for q in queue]
    return {
        "name": NAME,
        "annotation_version": "annotation-v3",
        "role": "LLM-composed contrast notes: debt words (nợ / trả / hoàn / để riêng) in "
        "repayment_out / repayment_in / refund / transfer / expense notes that are not debt-only",
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
                "source": "scripts/build_human_value_queue.py (via build_debt_relabel_queue.py)",
            },
            "group_order": [*debt.GROUPS, "earlier-generated", "regression-notes", "over-quota"],
            "reference_texts": ref_sizes,
            "dropped_by_group": dict(sorted(Counter(d["group"] for d in dropped).items())),
            "dropped_by_rule": dict(sorted(Counter(d["rule"] for d in dropped).items())),
            "dropped": dropped,
            "regression_notes": {
                "rule": f"folded difflib ratio >= {REGRESSION_RATIO}: dropped",
                "max_ratio_kept": round(max(kept_ratios), 4),
            },
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
    counts = json.loads(files[MANIFEST_FILE])["counts"]
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    leakage = json.loads(files[MANIFEST_FILE])["leakage"]
    print("dropped:", json.dumps(leakage["dropped_by_group"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
