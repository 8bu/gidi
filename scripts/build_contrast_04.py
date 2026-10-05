#!/usr/bin/env python
r"""Build ``datasets/annotation-v3/contrast-04``: shops dropped by the contrast-03 quota + gifts.

Usage:
    uv run python scripts/build_contrast_04.py [--root R] [--out-dir D] [--check]

Run 4 (training-v4) found that the contrast-03 round-robin quota over ``cue`` tags dropped every
``go! big c``, ``big c``, ``lotte mart``, ``mm mega market``, ``aeon mall`` and ``emart``
candidate. contrast-04 draws **per shop name** instead (``PER_CUE_CAP``: at most 3 notes of one
name, every name competes for the quota in turn) and adds the gift notes of the user decision
"a gift receiver is the target" (``docs/annotation-v3.md`` "Gift receivers": ``quà sinh nhật
Nhi 400k`` -> ``Nhi``) plus a few gift notes without a named receiver (target null).
``source.jsonl`` is written by hand (an LLM, user-approved) from ``docs/annotation-v1.md``,
``docs/annotation-v3.md`` and ``configs/annotation-v3.yaml``, never from a held-out note:
``{text, type, target, value, family, cue}`` with ``target`` / ``value`` the exact substrings of
``text`` (first occurrence, on a word boundary). Labels are LLM-made (``provenance.annotator`` is
``llm``). Families: ``shop`` (multi-word shop / supermarket names, the shop is the target),
``gift`` (receiver named, the receiver is the target; kinship prefix dropped), ``gift_null`` (gift
or ceremony money without a receiver: no target).

Candidates are dropped, in this order and each counted once, when they are (1) an exact / near
duplicate of a held-out note (human-value-01, the frozen test split, probe-v1, **human-value-02
batches a and b**), the existing corpus, ``training-v2`` train, ``contrast-02``, ``contrast-03`` or
an earlier candidate (``scripts/build_debt_relabel_queue.py`` leakage groups, rules of
``scripts/build_human_value_queue.py``), (2) at folded char-3-gram Jaccard >= ``HV_CHAR3_LIMIT``
of a human-value-01 or human-value-02 note, or (3) above the per-family quota (round-robin over the
``cue`` tags, at most ``PER_CUE_CAP`` per cue). human-value-02 is a test set: its texts are used
only by this gate (scored, never printed or stored; only the file hashes are recorded). The build
raises when one of ``REQUIRED_SHOPS`` is missing from the kept notes. Outputs, all deterministic:
``queue.jsonl``, ``labels.jsonl`` (annotation-v3 combined labels, ``complete``),
``provenance.jsonl`` and ``manifest.json``. ``--check`` rebuilds in memory and compares the files
on disk byte for byte.
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
c3 = _load("build_contrast_03")

NAME = "contrast-04"
DEFAULT_OUT_DIR = Path("datasets/annotation-v3/contrast-04")
SCHEMA = debt.SCHEMA
CONTRACT = debt.CONTRACT
SOURCE_FILE = "source.jsonl"
QUEUE_FILE = "queue.jsonl"
LABELS_FILE = "labels.jsonl"
PROVENANCE_FILE = "provenance.jsonl"
MANIFEST_FILE = "manifest.json"
SOURCE_TAG = "contrast-04-generated"
GENERATOR = c3.GENERATOR
TRAIN_V2 = c3.TRAIN_V2
CONTRAST_02 = c3.CONTRAST_02
CONTRAST_03 = Path("datasets/annotation-v3/contrast-03/queue.jsonl")
HV02_DIR = Path("datasets/annotation-v3/human-value-02")
HV02_QUEUES = (HV02_DIR / "review-queue.jsonl", HV02_DIR / "review-queue-b.jsonl")
HV_CHAR3_LIMIT = c3.HV_CHAR3_LIMIT

# Per-family quota of kept notes, filled round-robin over the ``cue`` tags (a shop name for
# ``shop``), at most ``PER_CUE_CAP`` notes per cue.
QUOTA = {"shop": 25, "gift": 15, "gift_null": 3}
PER_CUE_CAP = {"shop": 3, "gift": 2, "gift_null": 3}
FAMILY_TYPES = {"shop": {"expense"}, "gift": {"expense"}, "gift_null": {"expense"}}
# The names the contrast-03 quota dropped: each must be in the kept notes.
REQUIRED_SHOPS = ("go! big c", "big c", "lotte mart", "mm mega market", "aeon mall", "emart")


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
            or (row["family"] == "gift_null") != (row.get("target") is None)
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
    """Fill ``QUOTA`` per family: one note per cue in turn, at most ``PER_CUE_CAP`` per cue."""
    chosen: set[str] = set()
    for family, quota in QUOTA.items():
        by_cue: dict[str, list[str]] = {}
        for text in texts:
            if by_text[text]["family"] == family:
                by_cue.setdefault(by_text[text]["cue"], []).append(text)
        queues = [queue[: PER_CUE_CAP[family]] for queue in by_cue.values()]
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
    for rel in HV02_QUEUES:
        if not (root / rel).is_file():
            raise ValueError(f"{rel} is missing: the leakage gate needs it")
    source = load_source(out_dir / SOURCE_FILE)
    by_text = {r["text"]: r for r in source}
    groups = debt.load_reference_groups(root, out_dir, [])
    hv02_texts = [t for rel in HV02_QUEUES for t in debt._texts(root / rel)]
    if len(hv02_texts) < 190:
        raise ValueError("human-value-02 queues look truncated")
    groups["human-value-02"] = hv02_texts

    order = ("human-value-01", "human-value-02", *debt.GROUPS[1:])
    sets = {name: hv.ReferenceSet(groups[name]) for name in order}
    intra = hv.ReferenceSet([])
    survivors: list[str] = []
    dropped: list[dict[str, str]] = []
    for text in by_text:
        ref = hv.Reference(text)
        hit = next(((n, sets[n].match(ref)) for n in order if sets[n].match(ref)), None)
        if hit is None and (rule := intra.match(ref)):
            hit = ("earlier-generated", rule)
        if hit:
            dropped.append({"text": text, "group": hit[0], "rule": hit[1]})
        else:
            survivors.append(text)
            intra.add_ref(ref)
    ref_sizes = {name: len(ref_set.refs) for name, ref_set in sets.items()}

    # derived training sets / earlier batches are not corpus references: gate them here.
    earlier = {
        "training-v2": hv.ReferenceSet(debt._texts(root / TRAIN_V2)),
        "contrast-02": hv.ReferenceSet(debt._texts(root / CONTRAST_02)),
        "contrast-03": hv.ReferenceSet(debt._texts(root / CONTRAST_03)),
    }
    for name, ref_set in earlier.items():
        ref_sizes[name] = len(ref_set.refs)
    fresh = []
    for text in survivors:
        ref = hv.Reference(text)
        hit = next(((n, r.match(ref)) for n, r in earlier.items() if r.match(ref)), None)
        if hit:
            dropped.append({"text": text, "group": hit[0], "rule": hit[1]})
        else:
            fresh.append(text)

    held = {
        "human-value-01": [hv.Reference(t) for t in groups["human-value-01"]],
        "human-value-02": [hv.Reference(t) for t in hv02_texts],
    }
    char3_max: dict[str, dict[str, float]] = {}
    eligible: list[str] = []
    for text in fresh:
        scores = {name: c3.max_char3(hv, text, refs) for name, refs in held.items()}
        char3_max[text] = scores
        over = [n for n, s in scores.items() if s >= HV_CHAR3_LIMIT]
        if over:
            dropped.append(
                {"text": text, "group": f"{over[0]}-char3", "rule": f"char3 {scores[over[0]]:.2f}"}
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
    kept_targets = {by_text[t]["target"] for t in kept if by_text[t]["family"] == "shop"}
    missing = [s for s in REQUIRED_SHOPS if s not in kept_targets]
    if missing:
        raise ValueError(f"required shop names missing from the kept notes: {missing}")

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
    hv02_sha = {str(rel): debt._sha256((root / rel).read_bytes()) for rel in HV02_QUEUES}
    files[MANIFEST_FILE] = (
        json.dumps(
            manifest(
                root,
                out_dir,
                files,
                source,
                queue,
                labels,
                provenance,
                char3_max,
                dropped,
                ref_sizes,
                hv02_sha,
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
    char3_max: dict[str, dict[str, float]],
    dropped: list[dict[str, str]],
    ref_sizes: dict[str, int],
    hv02_sha: dict[str, str],
) -> dict[str, Any]:
    accented = [any(ord(c) > 127 for c in q["text"]) for q in queue]
    sha = {
        name: debt._sha256(files[name].encode("utf-8"))
        for name in (QUEUE_FILE, LABELS_FILE, PROVENANCE_FILE)
    }
    sha[SOURCE_FILE] = debt._sha256((out_dir / SOURCE_FILE).read_bytes())
    worst = {
        name: round(max(char3_max[q["text"]][name] for q in queue), 4)
        for name in ("human-value-01", "human-value-02")
    }
    shops = Counter(
        label["target"]["text"]
        for label, prov in zip(labels, provenance, strict=True)
        if prov["family"] == "shop"
    )
    return {
        "name": NAME,
        "annotation_version": "annotation-v3",
        "role": "LLM-composed notes: multi-word shop names the contrast-03 quota dropped (go! big "
        "c, big c, lotte mart, mm mega market, aeon mall, emart, ...) drawn per shop name, gift "
        "notes whose named receiver is the target (user decision), gift notes without a receiver "
        "(no target)",
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
            "per_cue_cap": PER_CUE_CAP,
            "shop_notes_by_name": dict(sorted(shops.items())),
            "cue": dict(sorted(Counter(p["cue"] for p in provenance).items())),
            "target_null": sum(label["target"] is None for label in labels),
            "target_multi_word": sum(
                label["target"] is not None and " " in label["target"]["text"] for label in labels
            ),
            "value_null": sum(label["value"] is None for label in labels),
            "accented": sum(accented),
            "unaccented": len(accented) - sum(accented),
            "unaccented_share": round(1 - sum(accented) / len(accented), 4),
        },
        "required_shops": list(REQUIRED_SHOPS),
        "leakage": {
            "rules": {
                "exact": "NFC + strip + lowercase",
                "folded": "NFC, lowercase, accents stripped, digits masked, whitespace collapsed",
                "sequence": "difflib ratio of folded texts >= 0.9",
                "token": "folded token Jaccard >= 0.8 (>= 4 tokens)",
                "char3": "folded char-3-gram Jaccard >= 0.8",
                "hv_char3_limit": f"folded char-3-gram Jaccard >= {HV_CHAR3_LIMIT} to any "
                "human-value-01 or human-value-02 note: dropped",
                "source": "scripts/build_human_value_queue.py (via build_debt_relabel_queue.py)",
            },
            "human_value_02": {
                "gated": True,
                "review_queue_sha256": hv02_sha,
                "note": "batches a and b; texts used only by the gate, never printed or stored",
            },
            "group_order": [
                "human-value-01",
                "human-value-02",
                *debt.GROUPS[1:],
                "earlier-generated",
                "training-v2",
                "contrast-02",
                "contrast-03",
                "human-value-01-char3",
                "human-value-02-char3",
                "over-quota",
            ],
            "reference_texts": ref_sizes,
            "dropped_by_group": dict(sorted(Counter(d["group"] for d in dropped).items())),
            "dropped_by_rule": dict(sorted(Counter(d["rule"] for d in dropped).items())),
            "dropped": dropped,
            "max_char3_kept": worst,
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
