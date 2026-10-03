"""Validation for raw corpus records (the "Raw corpus format" contract).

A raw corpus file is JSONL: one JSON object per non-blank line.

Required keys:

- ``id``: str, non-empty, unique within the file
- ``text``: str, non-empty, Unicode NFC, no leading/trailing whitespace, no newlines
- ``source``: str, non-empty (e.g. ``synthetic:<generator>`` or ``manual``)

Optional keys: ``prompt_id`` (str) plus any other keys, which are preserved and ignored by
validation. Only ``text`` is consumed by the tokenizer audit.

``validate_record`` checks a single record; ``validate_file`` adds the whole-file checks
(duplicate ids, exact duplicate texts) and reports physical line numbers, so findings point
back at the file. Malformed JSON is reported as an error instead of raising, because a broken
line is exactly what this validator exists to surface.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

REQUIRED_FIELDS: tuple[str, ...] = ("id", "text", "source")
"""Keys every raw corpus record must carry."""


class Issue(NamedTuple):
    """One finding: 1-based physical file line, record id when known, message.

    Unpacks as ``(line, id, message)``.
    """

    line: int
    record_id: str | None
    message: str


class DuplicateGroup(NamedTuple):
    """A value seen more than once, with the lines and record ids of each occurrence."""

    value: str
    lines: tuple[int, ...]
    ids: tuple[str | None, ...] = ()


@dataclass
class ValidationReport:
    """Result of validating one raw corpus file."""

    path: Path
    n_records: int = 0
    errors: list[Issue] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)
    duplicate_ids: list[DuplicateGroup] = field(default_factory=list)
    duplicate_texts: list[DuplicateGroup] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when the file has no errors (warnings alone still pass)."""
        return not self.errors


def validate_record(record: Any) -> list[str]:
    """Return the contract violations in ``record``; an empty list means valid.

    Uniqueness of ``id`` and of ``text`` is a property of a whole file rather than of a single
    record, so it is checked by :func:`validate_file`.
    """
    if not isinstance(record, dict):
        return ["record: expected a JSON object"]

    problems: list[str] = []
    for name in REQUIRED_FIELDS:
        if name not in record:
            problems.append(f"{name}: missing")
        elif not isinstance(record[name], str):
            problems.append(f"{name}: must be a string")
        elif not record[name]:
            problems.append(f"{name}: must not be empty")

    if "prompt_id" in record and not isinstance(record["prompt_id"], str):
        problems.append("prompt_id: must be a string")

    text = record.get("text")
    if isinstance(text, str) and text:
        if unicodedata.normalize("NFC", text) != text:
            problems.append("text: must be Unicode NFC")
        if text != text.strip():
            problems.append("text: must not have leading or trailing whitespace")
        if "\n" in text or "\r" in text:
            problems.append("text: must not contain newlines")
    return problems


def validate_file(path: str | Path) -> ValidationReport:
    """Validate every record in the JSONL file at ``path``.

    Blank lines are skipped, but reported line numbers are physical lines of the file, and the
    first occurrence of a value is the one later occurrences point back to. Duplicate ids are
    errors; exact duplicate texts are warnings, since a repeat may be a legitimate note.
    """
    path = Path(path)
    report = ValidationReport(path=path)
    id_lines: dict[str, list[int]] = {}
    text_lines: dict[str, list[int]] = {}
    text_ids: dict[str, list[str | None]] = {}

    with path.open(encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                report.errors.append(Issue(lineno, None, f"invalid JSON: {exc.msg}"))
                continue
            if not isinstance(record, dict):
                report.errors.append(Issue(lineno, None, "record: expected a JSON object"))
                continue

            report.n_records += 1
            raw_id = record.get("id")
            record_id = raw_id if isinstance(raw_id, str) else None
            for message in validate_record(record):
                report.errors.append(Issue(lineno, record_id, message))

            if record_id:
                id_lines.setdefault(record_id, []).append(lineno)
            text = record.get("text")
            if isinstance(text, str) and text:
                text_lines.setdefault(text, []).append(lineno)
                text_ids.setdefault(text, []).append(record_id)

    for value, lines in id_lines.items():
        if len(lines) < 2:
            continue
        report.duplicate_ids.append(DuplicateGroup(value, tuple(lines), (value,) * len(lines)))
        for lineno in lines[1:]:
            report.errors.append(
                Issue(lineno, value, f"duplicate id {value!r} (first seen at line {lines[0]})")
            )

    for value, lines in text_lines.items():
        if len(lines) < 2:
            continue
        ids = tuple(text_ids[value])
        report.duplicate_texts.append(DuplicateGroup(value, tuple(lines), ids))
        first_id, first_line = ids[0], lines[0]
        for position, lineno in enumerate(lines[1:], start=1):
            report.warnings.append(
                Issue(
                    lineno,
                    ids[position],
                    f"duplicate text {_preview(value)!r} (id {first_id!r}, line {first_line})",
                )
            )

    report.errors.sort(key=lambda issue: issue.line)
    report.warnings.sort(key=lambda issue: issue.line)
    return report


def render_report(report: ValidationReport) -> str:
    """Render a report as one summary line plus one line per finding."""
    lines = [
        f"{report.path}: {report.n_records} record(s), "
        f"{len(report.errors)} error(s), {len(report.warnings)} warning(s)"
    ]
    for issue in report.errors:
        lines.append(f"  error line {issue.line} [{issue.record_id or '-'}]: {issue.message}")
    for issue in report.warnings:
        lines.append(f"  warning line {issue.line} [{issue.record_id or '-'}]: {issue.message}")
    return "\n".join(lines)


def _preview(text: str, limit: int = 40) -> str:
    """Flatten and shorten a text for use inside a message."""
    flat = text.replace("\n", "\\n").replace("\r", "\\r")
    return flat if len(flat) <= limit else flat[: limit - 1] + "\u2026"
