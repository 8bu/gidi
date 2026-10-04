#!/usr/bin/env python
"""Validate annotation JSONL files against the annotation-v1 or combined annotation-v2 contract.

Usage:
    uv run python scripts/validate_annotations.py PATH [PATH ...] \
        [--queue datasets/annotation-v1/queue.jsonl] [--config configs/annotation-v1.yaml]

    # combined pass (type + target + value, Quet multi-span): --config selects it; --queue required
    uv run python scripts/validate_annotations.py \
        datasets/annotation-v2/human-value-01/labels.jsonl \
        --config configs/annotation-v2.quet.yaml \
        --queue datasets/annotation-v2/human-value-01/review-queue.jsonl

Span offsets are checked against the queue texts. Exits 1 if any file has errors.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gidi.annotation.combined import (
    is_combined_schema,
    load_combined_config,
    validate_combined_file,
)
from gidi.annotation.schema import (
    DEFAULT_CONFIG,
    load_config,
    load_texts_by_id,
    render_report,
    validate_file,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", metavar="PATH", nargs="+", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="annotation-v1 contract (default) or a Quet multi-span schema "
        "(configs/annotation-v2.quet.yaml: combined type + target + value labels)",
    )
    parser.add_argument("--queue", type=Path, default=None, help="default: queue.out in config")
    args = parser.parse_args(argv)

    if is_combined_schema(args.config):
        if args.queue is None:
            parser.error("--queue is required with a combined (multi-span) --config")
        config = load_combined_config(args.config)
        validate = validate_combined_file
    else:
        config = load_config(args.config)
        validate = validate_file
    texts = load_texts_by_id(args.queue or config.queue["out"])
    failed = False
    for path in args.paths:
        report = validate(path, texts, config)
        print(render_report(report))
        failed = failed or not report.ok
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
