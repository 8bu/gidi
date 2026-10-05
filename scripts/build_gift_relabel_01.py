#!/usr/bin/env python
"""Build ``datasets/annotation-v3/gift-relabel-01``: the gift-receiver relabel map for training.

Usage:
    uv run python scripts/build_gift_relabel_01.py [--root R] [--check]

User decision (``docs/annotation-v3.md`` "Gift receivers"): the receiver of a gift or ceremony
money is the target. ``training-v4/train.jsonl`` (905 base records incl. the human annotation-v1
labels, contrast-02 kept, contrast-03) is scanned for gift wording (``GIFT_CUES``, folded text).
Every hit was read once; the decisions are fixed in ``RELABEL`` below (LLM notes of ``contrast-*``
whose gift receiver is named but whose target is null, or a gift note with a second plausible
counterparty that has to leave the training set). ``map.jsonl`` holds one row per change,
``{id, text, old, new, action, reason}`` (``old`` / ``new`` = target text or null; ``action`` is
``relabel`` or ``drop`` (``new`` is null: the note is uncertain under the rule and leaves the
training set)); ``training-v5`` applies it. ``manifest.json`` records the scan, the human
annotation-v1 labels that already follow the rule (kept) and the human labels that conflict with
it (``human_conflicts``, listed and never changed). ``--check`` rebuilds in memory and compares.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

NAME = "gift-relabel-01"
OUT = Path("datasets/annotation-v3/gift-relabel-01")
TRAIN_V4 = Path("datasets/annotation-v3/training-v4/train.jsonl")
GIFT_CUES = re.compile(
    r"\b(qua|tang|bieu|mung|li xi|lixi|phong bi|cuoi|dam cuoi|dam ma|dam hoi|phung|vieng|"
    r"sinh nhat|sn|tan gia|day thang|thoi noi)\b"
)
# id -> (action, new target text, reason); old target is read from training-v4
RELABEL = {
    "contrast-03-202ee7b034e9": (
        "relabel",
        "Hanh",
        "gift (`mua quà ... cho Hanh`): the named receiver is the target",
    ),
    "contrast-03-5e8eb16c75e6": (
        "relabel",
        "Duc",
        "gift (`quà sinh nhật Duc`): the named receiver is the target",
    ),
    "contrast-03-88c14150f4cf": (
        "relabel",
        "Lan",
        "gift (`quà 20/10 cho cô Lan`): the named receiver is the target, prefix `cô` dropped",
    ),
    "contrast-03-24eb1714b7a3": (
        "relabel",
        "sếp mới",
        "gift (`gói quà tặng sếp mới`): the receiver (role term + modifier) is the target",
    ),
    "contrast-03-7813f886774e": (
        "drop",
        None,
        "gift for a named receiver bought on a marketplace (`cho bạn Minh ở shopee`): two "
        "plausible counterparties, so uncertain under the one-target rule; not trainable",
    ),
}
# human-labelled records with a gift cue and a null target, reviewed: none conflicts with the rule
HUMAN_NULL_REVIEWED = {
    "baseline-01-ffe59df773f6": "`qua` = via (`chuyển qua tk tiết kiệm`), not a gift",
    "baseline-01-c0c3b1551f40": "`qua` = yesterday evening (`ốc tối qua vs Hiếu`), not a gift",
    "baseline-01-d432338e650a": "`qua` = yesterday (`hôm qua`), not a gift",
    "targeted-annotation-v1-01-38ce17a93a13": "`qua` = via (`vay qua app`), not a gift",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fold(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower().replace("đ", "d")
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text("utf-8").splitlines() if x.strip()]


def build(root: Path) -> dict[str, str]:
    records = read_jsonl(root / TRAIN_V4)
    by_id = {r["id"]: r for r in records}
    hits = [r for r in records if GIFT_CUES.search(fold(r["text"]))]
    hit_ids = {r["id"] for r in hits}
    rows = []
    for rid, (action, new, reason) in RELABEL.items():
        record = by_id[rid]
        if rid not in hit_ids:
            raise ValueError(f"{rid}: not found by the gift scan")
        if record["provenance"]["annotator"] != "llm" or not rid.startswith("contrast-"):
            raise ValueError(f"{rid}: only LLM contrast notes are relabelled")
        if new is not None and new not in record["text"]:
            raise ValueError(f"{rid}: new target {new!r} not in the text")
        old = record["target"]["text"] if record["target"] else None
        rows.append(
            {"id": rid, "text": record["text"], "old": old, "new": new, "action": action,
             "reason": reason}
        )  # fmt: skip
    human = [r for r in hits if r["provenance"]["annotator"] == "human"]
    human_null = {r["id"]: r["text"] for r in human if r["target"] is None}
    if set(human_null) != set(HUMAN_NULL_REVIEWED):
        raise ValueError(f"human gift-cue records with a null target changed: {human_null}")
    train_bytes = (root / TRAIN_V4).read_bytes()
    map_text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    manifest = {
        "name": NAME,
        "annotation_version": "annotation-v3",
        "rule": "docs/annotation-v3.md 'Gift receivers': the receiver of a gift or ceremony money "
        "is the target; buying a concrete item for someone (`mua hoa tặng mẹ`) has no target",
        "scan": {
            "source": str(TRAIN_V4),
            "source_sha256": sha256(train_bytes),
            "cue_regex_folded": GIFT_CUES.pattern,
            "records": len(records),
            "cue_hits": len(hits),
            "cue_hits_by_annotator": dict(
                sorted(Counter(r["provenance"]["annotator"] for r in hits).items())
            ),
            "cue_hits_by_source": dict(sorted(Counter(r["source_batch"] for r in hits).items())),
        },
        "counts": {
            "relabel": sum(r["action"] == "relabel" for r in rows),
            "drop": sum(r["action"] == "drop" for r in rows),
            "map_rows": len(rows),
        },
        "human_cue_hits_with_target": sorted(r["id"] for r in human if r["target"] is not None),
        "human_cue_hits_with_target_note": "human annotation-v1 labels with a cue hit and a target "
        "(the cue also matches non-gift words): kept; the gift notes among them follow the rule "
        "(e.g. `qua sinh nhat bé Na 300k` -> `Na`, `quà 20/10 cho mẹ` -> `mẹ`, "
        "`mung dam cuoi Tuan 500k` -> `Tuan`)",
        "human_conflicts": [],
        "human_conflicts_note": "no human label conflicts with the rule; the human records with a "
        "gift cue and a null target are listed in `human_null_reviewed`",
        "human_null_reviewed": HUMAN_NULL_REVIEWED,
        "ai_labels_note": "annotator `ai` baseline / targeted labels already follow the rule "
        "(gift receivers are targets, item purchases `mua hoa tặng mẹ` are null); not changed",
        "files_sha256": {"map.jsonl": sha256(map_text.encode("utf-8"))},
    }
    return {
        "map.jsonl": map_text,
        "manifest.json": json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    out = args.root / OUT
    try:
        files = build(args.root)
    except (ValueError, KeyError, OSError) as exc:
        print(f"{NAME}: {exc}", file=sys.stderr)
        return 1
    if args.check:
        stale = [
            n
            for n, c in files.items()
            if not (out / n).is_file() or (out / n).read_text("utf-8") != c
        ]
        if stale:
            print(f"STALE: {', '.join(stale)}", file=sys.stderr)
            return 1
        print(f"ok: {len(files)} files match ({NAME})")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (out / name).write_text(content, encoding="utf-8")
    print(json.loads(files["manifest.json"])["counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
