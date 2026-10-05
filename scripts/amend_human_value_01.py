#!/usr/bin/env python
"""Apply ``amendment-01`` to the frozen set ``datasets/annotation-v2/human-value-01``.

Usage:
    uv run python scripts/amend_human_value_01.py [--root R] [--check]

User decision (annotation-v3 clarification, ``docs/annotation-v3.md`` "Gift receivers"): the
receiver of a gift or ceremony money is the counterparty, so a gift note with a named receiver
has that receiver as ``target`` (kinship/title prefix dropped before a proper name). This agrees
with ``docs/annotation-v1.md`` ("Beneficiary is not payee"). ``AMENDMENTS`` lists every
human-value-01 note that the rule changes (found by reading all 150 notes; every other gift-like
note, e.g. ``mừng cưới Hoa 1 triệu`` -> ``Hoa``, ``mua nửa chỉ vàng cho con``, already follows
the rule). The user approved changing these labels; the original labels are kept in
``amendment-01.json`` (id, text, old target, new target, reason).

The script rewrites ``labels.jsonl`` (only the listed targets), writes ``amendment-01.json`` and
adds an ``amendments`` entry to ``freeze.json`` (new labels hash, stats). It refuses to run twice;
``--check`` verifies that the files on disk are exactly the amended state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

HV = Path("datasets/annotation-v2/human-value-01")
LABELS = "labels.jsonl"
QUEUE = "review-queue.jsonl"
FREEZE = "freeze.json"
AMENDMENT = "amendment-01.json"
RULE = (
    "gift receiver is the target (user decision, docs/annotation-v3.md 'Gift receivers'; "
    "docs/annotation-v1.md 'Beneficiary is not payee'): a gift / ceremony-money note with a "
    "named receiver has the receiver as target, a kinship prefix before a proper name dropped"
)
# id -> (old target text, new target text, reason)
AMENDMENTS = {
    "baseline-01-94b4a287f26a": (
        None,
        "Vy",
        "gift note `quà sinh nhật Vy`: the gift receiver Vy is the counterparty",
    ),
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text("utf-8").splitlines() if x.strip()]


def dump_jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def span(text: str, sub: str) -> dict[str, Any]:
    start = text.index(sub)
    return {"text": sub, "start": start, "end": start + len(sub)}


def stats(labels: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "labels": len(labels),
        "status": dict(Counter(x["annotation_status"] for x in labels)),
        "type": dict(Counter(x["type"] for x in labels)),
        "value_null": sum(x["value"] is None for x in labels),
        "target_null": sum(x["target"] is None for x in labels),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    hv = args.root / HV
    freeze = json.loads((hv / FREEZE).read_text("utf-8"))
    texts = {r["id"]: r["text"] for r in read_jsonl(hv / QUEUE)}
    labels = read_jsonl(hv / LABELS)
    if args.check:
        amendment = json.loads((hv / AMENDMENT).read_text("utf-8"))
        entry = freeze["amendments"][0]
        problems = []
        by_id = {x["id"]: x for x in labels}
        for change in amendment["changes"]:
            target = by_id[change["id"]]["target"]
            if target is None or target["text"] != change["new_target"]:
                problems.append(f"{change['id']}: label is not the amended target")
            if texts[change["id"]] != change["text"]:
                problems.append(f"{change['id']}: text differs from the amendment record")
        if entry["labels_sha256_after"] != sha256((hv / LABELS).read_bytes()):
            problems.append("labels.jsonl hash differs from freeze.json amendments")
        if freeze["files"][LABELS] != entry["labels_sha256_after"]:
            problems.append("freeze.json files.labels.jsonl is not the amended hash")
        if entry["amendment_sha256"] != sha256((hv / AMENDMENT).read_bytes()):
            problems.append("amendment-01.json hash differs from freeze.json")
        now = stats(labels)
        if (freeze["stats"]["labels"], freeze["stats"]["target_null"]) != (
            now["labels"],
            now["target_null"],
        ):
            problems.append("freeze.json stats differ from the labels")
        if problems:
            print("\n".join(problems), file=sys.stderr)
            return 1
        print(f"ok: amendment-01 applied ({len(amendment['changes'])} change(s))")
        return 0
    if "amendments" in freeze:
        raise SystemExit("amendment-01 is already applied")
    before = sha256((hv / LABELS).read_bytes())
    changes = []
    for label in labels:
        if label["id"] not in AMENDMENTS:
            continue
        old, new, reason = AMENDMENTS[label["id"]]
        text = texts[label["id"]]
        current = label["target"]["text"] if label["target"] else None
        if current != old:
            raise SystemExit(f"{label['id']}: expected old target {old!r}, found {current!r}")
        label["target"] = span(text, new)
        if text[label["target"]["start"] : label["target"]["end"]] != new:
            raise SystemExit(f"{label['id']}: bad span")
        changes.append(
            {
                "id": label["id"],
                "text": text,
                "old_target": old,
                "new_target": new,
                "new_span": label["target"],
                "reason": reason,
            }
        )
    if len(changes) != len(AMENDMENTS):
        raise SystemExit("some amended ids are not in the labels file")
    labels_text = dump_jsonl(labels)
    amendment = {
        "set": "human-value-01",
        "amendment": "amendment-01",
        "date": "2026-10-05",
        "approved_by": "user",
        "rule": RULE,
        "labels_sha256_before": before,
        "n_changes": len(changes),
        "changes": changes,
        "scan": "all 150 notes read for gift / ceremony-money wording (quà, mừng, lì xì, tặng, "
        "biếu, cưới, sinh nhật, ...); the others already follow the rule or name no receiver",
    }
    amendment_text = json.dumps(amendment, ensure_ascii=False, indent=2) + "\n"
    (hv / LABELS).write_text(labels_text, encoding="utf-8")
    (hv / AMENDMENT).write_text(amendment_text, encoding="utf-8")
    freeze["files"][LABELS] = sha256(labels_text.encode("utf-8"))
    freeze["stats"]["target_null"] = stats(labels)["target_null"]
    freeze["amendments"] = [
        {
            "id": "amendment-01",
            "file": AMENDMENT,
            "amendment_sha256": sha256(amendment_text.encode("utf-8")),
            "date": "2026-10-05",
            "approved_by": "user",
            "rule": RULE,
            "n_changes": len(changes),
            "labels_sha256_before": before,
            "labels_sha256_after": freeze["files"][LABELS],
        }
    ]
    (hv / FREEZE).write_text(json.dumps(freeze, ensure_ascii=False, indent=2) + "\n", "utf-8")
    after = freeze["files"][LABELS]
    print(f"amended {len(changes)} label(s); labels sha256 {before[:12]} -> {after[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
