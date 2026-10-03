"""Build a Quet re-check queue from the candidates table of a pilot analysis report.

The ids come from one candidates table (``--section``, a ``## `` heading) in the report. Each one
is looked up in the annotation queue, and its entry is copied verbatim (``id``, ``text``,
``corpus``, ``position``) in queue-position order. Labels are never read or copied: Quet re-opens
the records against the existing labels file.

    uv run python scripts/build_recheck_queue.py \
        --report experiments/annotation-v1-pilot.md \
        --out datasets/annotation-v1/recheck-pilot-01.jsonl
    uv run python scripts/build_recheck_queue.py \
        --section "Frozen-v1 re-check candidates" \
        --out datasets/annotation-v1/recheck-frozen-v1-02.jsonl

Open the result in Quet's merge mode, which edits only the subset ids and keeps every other
label (``--out`` is strict and would reject the out-of-queue labels):

    quet annotate datasets/annotation-v1/recheck-pilot-01.jsonl \
        --schema configs/annotation-v1.yaml \
        --labels datasets/annotation-v1/labels.jsonl
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from gidi.corpus.jsonl import read_jsonl, write_jsonl

DEFAULT_SECTION = "Manual re-check candidates"
ROW_ID = re.compile(r"^\|\s*\d+\s*\|\s*([A-Za-z0-9][A-Za-z0-9._-]*-[0-9a-f]{12})\s*\|")


def candidate_ids(report: str, section: str = DEFAULT_SECTION) -> list[str]:
    heading = f"## {section}\n"
    if heading not in report:
        raise ValueError(f"section not found: {heading.strip()!r}")
    section = report.split(heading, 1)[1].split("\n## ", 1)[0]
    ids = [m.group(1) for line in section.splitlines() if (m := ROW_ID.match(line))]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate ids in re-check table")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--report", type=Path, default=Path("experiments/annotation-v1-pilot.md"))
    parser.add_argument("--queue", type=Path, default=Path("datasets/annotation-v1/queue.jsonl"))
    parser.add_argument(
        "--out", type=Path, default=Path("datasets/annotation-v1/recheck-pilot-01.jsonl")
    )
    parser.add_argument("--section", default=DEFAULT_SECTION, help="candidates table heading")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    ids = candidate_ids(args.report.read_text(encoding="utf-8"), args.section)
    queue = {entry["id"]: entry for entry in read_jsonl(args.queue)}
    missing = [i for i in ids if i not in queue]
    if missing:
        raise SystemExit(f"ids not in queue: {missing}")
    entries = sorted((queue[i] for i in ids), key=lambda e: e["position"])
    write_jsonl(args.out, entries, overwrite=args.overwrite)
    print(f"{args.out}: {len(entries)} record(s)")


if __name__ == "__main__":
    main()
