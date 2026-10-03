#!/usr/bin/env python
"""Audit pretrained tokenizer candidates against the Vietnamese finance corpus.

Usage:
    uv run python scripts/audit_tokenizers.py \
        --corpus corpus/reviewed/notes.jsonl \
        --tokenizer vinai/phobert-base FPTAI/vibert-base-cased --out experiments/xxx/tokenizers.json

Every ``--tokenizer`` is passed to ``transformers.AutoTokenizer.from_pretrained``, so a Hub name
downloads from the Hub (the only network access in this workflow) while a local directory path
is used as is. Candidate names must be verified on the Hub before use; these were the intended
starting points when the audit was written:

    vinai/phobert-base              Vietnamese BPE. Caveat: expects word-segmented input
                                    (``"ăn phở"`` -> ``"ăn phở"`` with underscore-joined
                                    syllables); raw Gidi notes are not word-segmented, so this
                                    audit measures it on unsegmented text (a lower bound).
    FPTAI/vibert-base-cased         Vietnamese BERT, cased, no word segmentation required.
    xlm-roberta-base                multilingual SentencePiece, 250k vocabulary.
    google/mt5-small                SentencePiece (mT5), multilingual.
    bert-base-multilingual-cased    mBERT, 110k vocabulary.
    microsoft/Multilingual-MiniLM-L12-H384   small multilingual encoder (on-device candidate).

The report is written with the same safety as the corpus writer: an existing ``--out`` file is
refused unless ``--overwrite`` is passed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from gidi.tokenizer.audit import TokenizerAuditReport, audit_tokenizer, load_texts

_HEADERS = (
    "tokenizer",
    "vocab",
    "texts",
    "words",
    "tok/txt",
    "p50",
    "p95",
    "max",
    "fert",
    "%split",
    "unk%",
    "rt%",
    "amt",
    "amt/txt",
    "slang/w",
    "%slsplit",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--corpus",
        type=Path,
        nargs="+",
        required=True,
        metavar="PATH",
        help="JSONL corpus file(s); the 'text' field of each record is audited",
    )
    parser.add_argument(
        "--tokenizer",
        nargs="+",
        required=True,
        metavar="NAME_OR_PATH",
        help="Hub name or local directory for AutoTokenizer.from_pretrained",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write the full reports as JSON here (refuses to overwrite unless --overwrite)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacing an existing --out file",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=20,
        metavar="N",
        help="number of round-trip failure examples to keep per tokenizer (default: 20)",
    )
    return parser.parse_args(argv)


def format_table(reports: list[TokenizerAuditReport]) -> str:
    """Compact one-row-per-tokenizer comparison table."""
    rows = [_row(report) for report in reports]
    widths = [
        max(len(headers), *(len(row[column]) for row in rows))
        for column, headers in enumerate(_HEADERS)
    ]
    lines = [_join(_HEADERS, widths), _join(["-" * width for width in widths], widths)]
    lines.extend(_join(row, widths) for row in rows)
    return "\n".join(lines)


def _row(report: TokenizerAuditReport) -> list[str]:
    name = report.name if len(report.name) <= 40 else report.name[:37] + "..."
    return [
        name,
        str(report.vocab_size),
        str(report.n_texts),
        str(report.n_words),
        f"{report.mean_tokens_per_text:.2f}",
        f"{report.p50_tokens_per_text:.1f}",
        f"{report.p95_tokens_per_text:.1f}",
        str(report.max_tokens_per_text),
        f"{report.fertility:.2f}",
        f"{report.pct_words_split:.1f}",
        f"{100 * report.unk_rate:.1f}",
        f"{100 * report.roundtrip_exact_match_rate:.1f}",
        str(report.n_amounts),
        f"{report.mean_amount_tokens:.2f}",
        f"{report.mean_shorthand_tokens:.2f}",
        f"{report.pct_shorthand_split:.1f}",
    ]


def _join(cells: list[str], widths: list[int]) -> str:
    left = cells[0].ljust(widths[0])
    rest = [cell.rjust(width) for cell, width in zip(cells[1:], widths[1:], strict=True)]
    return "  ".join([left, *rest]).rstrip()


def print_notes(reports: list[TokenizerAuditReport]) -> None:
    """Per-tokenizer findings the table cannot show: unk characters and round-trip failures."""
    for report in reports:
        if report.unk_characters:
            characters = "".join(report.unk_characters)
            print(f"{report.name}: characters mapping to {report.unk_token}: {characters}")
        for failure in report.roundtrip_failures[:3]:
            print(f"{report.name}: round trip {failure['text']!r} -> {failure['roundtrip']!r}")


def print_probes(reports: list[TokenizerAuditReport]) -> None:
    """Exact tokenization of the fixed finance/slang probes, one block per tokenizer."""
    for report in reports:
        print(f"\n{report.name} probes:")
        for probe in report.probes:
            unk = "  UNK" if probe["has_unk"] else ""
            print(
                f"  {probe['text']!r:18} {probe['n_tokens']:>2}  {' | '.join(probe['tokens'])}{unk}"
            )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    texts = load_texts(args.corpus)
    if not texts:
        raise SystemExit(f"no texts found in: {', '.join(str(p) for p in args.corpus)}")

    reports = []
    for name in args.tokenizer:
        print(f"loading {name} ...")
        tokenizer = AutoTokenizer.from_pretrained(name)
        reports.append(audit_tokenizer(tokenizer, texts, max_examples=args.examples))

    print(format_table(reports))
    print_notes(reports)
    print_probes(reports)

    if args.out is not None:
        if args.out.exists() and not args.overwrite:
            raise SystemExit(
                f"refusing to overwrite existing file: {args.out} (pass --overwrite to replace it)"
            )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "corpus": [str(path) for path in args.corpus],
            "examples": args.examples,
            "reports": [report.to_dict() for report in reports],
        }
        args.out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
