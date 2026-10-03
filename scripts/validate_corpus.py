#!/usr/bin/env python
"""Validate raw corpus JSONL files against the record contract.

Usage:
    uv run python scripts/validate_corpus.py PATH [PATH ...]

Prints a per-file summary and exits 1 if any file has errors. Warnings (exact duplicate
texts) alone exit 0. Logic lives in ``gidi.corpus.schema``; this is a thin wrapper.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gidi.corpus.schema import render_report, validate_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate raw corpus JSONL files against the record contract."
    )
    parser.add_argument(
        "paths",
        metavar="PATH",
        nargs="+",
        type=Path,
        help="JSONL corpus file(s) to validate",
    )
    args = parser.parse_args(argv)

    failed = False
    for path in args.paths:
        try:
            report = validate_file(path)
        except OSError as exc:
            print(f"{path}: cannot read: {exc.strerror or exc}", file=sys.stderr)
            failed = True
            continue
        print(render_report(report))
        failed = failed or not report.ok
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
