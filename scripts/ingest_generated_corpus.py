#!/usr/bin/env python
"""Ingest generated model output into a canonical raw-corpus JSONL file.

Usage:
    uv run python scripts/ingest_generated_corpus.py generated.jsonl \\
        --source claude --prompt-id baseline-01 --out corpus/raw/baseline-01.jsonl

Each JSONL input line is one JSON object with a single key, ``text``; ``--format text`` instead
reads one note per line. ``id``, ``source``, and ``prompt_id`` are assigned here, and any other
input key (including an input ``id``) is discarded and reported. Ingestion is all-or-nothing:
if any input has an error, nothing is written and the command exits 1, so ``corpus/raw/`` never
receives a partially malformed batch. Warnings (duplicate texts, NFC normalization, dropped
keys) still exit 0. Logic lives in ``gidi.corpus.ingest``; this is a thin wrapper.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from gidi.corpus.ingest import ingest_file, merge_reports, render_report
from gidi.corpus.jsonl import write_jsonl

_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest generated output into a canonical raw corpus JSONL file."
    )
    parser.add_argument(
        "generated",
        metavar="GENERATED",
        nargs="+",
        type=Path,
        help="generated output file(s) to ingest",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="corpus source label, e.g. claude (slug: letters, digits, '.', '_', '-')",
    )
    parser.add_argument(
        "--prompt-id",
        required=True,
        dest="prompt_id",
        help="generation prompt id (slug: letters, digits, '.', '_', '-')",
    )
    parser.add_argument("--out", required=True, type=Path, help="output raw corpus JSONL file")
    parser.add_argument(
        "--format",
        choices=("auto", "jsonl", "text"),
        default="auto",
        help="input format (default: auto, decided from the first line with content)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace --out if it already exists",
    )
    args = parser.parse_args(argv)
    _check_slug(parser, "--source", args.source)
    _check_slug(parser, "--prompt-id", args.prompt_id)
    return args


def _check_slug(parser: argparse.ArgumentParser, flag: str, value: str) -> None:
    if not _SLUG.match(value):
        parser.error(f"argument {flag}: invalid slug {value!r} (expected {_SLUG.pattern})")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    out_resolved = args.out.resolve()
    for path in args.generated:
        if path.resolve() == out_resolved:
            print(
                f"--out {args.out} resolves to input {path}; refusing to read and write "
                "the same file",
                file=sys.stderr,
            )
            return 1

    reports = []
    for path in args.generated:
        try:
            report = ingest_file(
                path,
                source=args.source,
                prompt_id=args.prompt_id,
                input_format=args.format,
            )
        except OSError as exc:
            print(f"{path}: cannot read: {exc.strerror or exc}", file=sys.stderr)
            return 1
        print(render_report(report))
        reports.append(report)

    combined = merge_reports(reports)
    if not combined.ok:
        print(
            f"{len(combined.errors)} error(s) in input; no output written",
            file=sys.stderr,
        )
        return 1

    try:
        written = write_jsonl(args.out, combined.records, overwrite=args.overwrite)
    except FileExistsError as exc:
        print(f"{exc}; pass --overwrite to replace it", file=sys.stderr)
        return 1

    print(f"wrote {written} record(s) to {args.out}")
    print(f"validate with: uv run python scripts/validate_corpus.py {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
