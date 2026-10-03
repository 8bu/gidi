#!/usr/bin/env python
"""Check candidate training notes for overlap with evaluation sets, training data, and each other.

Usage:
    uv run python scripts/check_note_similarity.py CANDIDATES.jsonl [--out report.json]
        [--show-eval]

Each candidate line is ``{"text": ..., "intended_target": str | null, "group": str, ...}``.
``intended_target`` is weak generation metadata, used only to mask the counterparty for
template matching. Reject/warn rules: ``gidi.annotation.leakage``. Evaluation neighbour texts are
hidden unless ``--show-eval`` is passed, so that rewriting a rejected note cannot copy them.
Exits 1 when any candidate is rejected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gidi.annotation.leakage import check, load_references
from gidi.corpus.jsonl import read_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("candidates", type=Path)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--show-eval", action="store_true")
    args = parser.parse_args(argv)

    candidates = read_jsonl(args.candidates)
    eval_refs, train_refs = load_references(args.root)
    results = check(candidates, eval_refs, train_refs, args.show_eval)
    if args.out:
        args.out.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), encoding="utf-8"
        )
    rejected = [r for r in results if r["reject"]]
    for r in rejected:
        print(f"REJECT {r['text']!r}: {', '.join(r['reject'])} (eval {r['eval_sim']})")
    for r in results:
        if r["warn"] and not r["reject"]:
            print(f"warn   {r['text']!r}: {', '.join(r['warn'])}")
    print(
        f"{len(results)} candidate(s): {len(rejected)} rejected, "
        f"{sum(bool(r['warn']) and not r['reject'] for r in results)} warned; "
        f"eval refs {len(eval_refs)}, train refs {len(train_refs)}"
    )
    return 1 if rejected else 0


if __name__ == "__main__":
    raise SystemExit(main())
