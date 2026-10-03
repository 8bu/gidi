#!/usr/bin/env python
"""Build the deterministic annotation queue described in an annotation config.

Usage:
    uv run python scripts/build_annotation_queue.py [--config configs/annotation-v1.yaml]

Reads the reviewed corpus named in the config (never modifies it) and writes the queue JSONL.
An existing queue is refused unless --overwrite is passed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gidi.annotation.queue import build_queue
from gidi.annotation.schema import DEFAULT_CONFIG, load_config
from gidi.corpus.jsonl import iter_jsonl, write_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--overwrite", action="store_true", help="replace an existing queue")
    args = parser.parse_args(argv)

    q = load_config(args.config).queue
    out = Path(q["out"])
    if out.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite existing file: {out} (pass --overwrite)")
    queue = build_queue(
        iter_jsonl(q["corpus"]),
        size=int(q["size"]),
        seed=str(q["seed"]),
        review_status=str(q["review_status"]),
        corpus=str(q["corpus"]),
    )
    write_jsonl(out, queue, overwrite=args.overwrite)
    print(f"wrote {len(queue)} queue entries to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
