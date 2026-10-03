"""Ingest generated model output into canonical raw-corpus JSONL records.

Generation emits one JSON object per line carrying a single key, ``text`` (or, with the
``text`` input format, one note per line). This module attaches the bookkeeping the raw corpus
contract requires -- a deterministic ``id`` plus the ``source`` and ``prompt_id`` of the run --
and normalizes the text to Unicode NFC. Anything else found in the input, including an ``id`` a
generator may have produced, is discarded and reported.

Ids are content-derived, never random: ``{prompt_id}-{digest12}`` where ``digest12`` is the
first 12 hex characters of ``sha256(f"{source}\\x00{prompt_id}\\x00{text}")`` over the *final*
NFC-normalized text. Re-running the same ingestion on the same input therefore reproduces the
same ids byte for byte. The ``k``-th further occurrence of an identical text (``k`` = 2, 3,
...) gets a ``-d{k}`` suffix so ids stay unique within the run; a genuine digest collision
between two different texts is reported as an error rather than emitted silently.

Strictness is all-or-nothing. ``IngestReport.records`` holds the canonical records that were
produced, but the module never writes anything itself: when there is *any* error the caller
must write nothing, so ``corpus/raw/`` never receives a partially malformed batch.
``scripts/ingest_generated_corpus.py`` enforces that and prints "no output written".
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gidi.corpus.schema import DuplicateGroup, Issue, validate_record

_INPUT_FORMATS = ("auto", "jsonl", "text")
_FORBIDDEN_CHARS = ("\n", "\r", "\t")


@dataclass
class IngestReport:
    """Outcome of ingesting one or more generated inputs.

    ``records`` are canonical (``id``, ``text``, ``source``, ``prompt_id``) in input order.
    ``input_format`` is the format actually used, with ``auto`` already resolved. ``n_lines``
    counts physical input lines, ``n_records`` the records produced. ``ok`` is True when there
    are no errors; warnings alone still pass.
    """

    sources: list[str] = field(default_factory=list)
    input_format: str = ""
    n_lines: int = 0
    n_records: int = 0
    records: list[dict[str, Any]] = field(default_factory=list)
    errors: list[Issue] = field(default_factory=list)
    warnings: list[Issue] = field(default_factory=list)
    duplicate_texts: list[DuplicateGroup] = field(default_factory=list)
    normalized_lines: list[int] = field(default_factory=list)
    dropped_keys: dict[str, int] = field(default_factory=dict)
    dropped_id_lines: list[int] = field(default_factory=list)
    path: Path | None = None

    @property
    def ok(self) -> bool:
        """True when no error was found (warnings alone still pass)."""
        return not self.errors


def derive_id(*, source: str, prompt_id: str, text: str, occurrence: int = 1) -> str:
    """Derive a stable id for ``text`` from its source, prompt, and content.

    ``occurrence`` numbers identical texts: 1 yields ``{prompt_id}-{digest12}``, and each
    further occurrence ``k`` yields ``{prompt_id}-{digest12}-d{k}``.
    """
    digest = hashlib.sha256(f"{source}\x00{prompt_id}\x00{text}".encode()).hexdigest()[:12]
    base = f"{prompt_id}-{digest}"
    return base if occurrence <= 1 else f"{base}-d{occurrence}"


def ingest_file(
    path: str | Path,
    *,
    source: str,
    prompt_id: str,
    input_format: str = "auto",
) -> IngestReport:
    """Read ``path`` as bytes and ingest it with strict per-line UTF-8 decoding."""
    path = Path(path)
    return ingest_lines(
        path,
        _numbered_lines(path.read_bytes()),
        source=source,
        prompt_id=prompt_id,
        input_format=input_format,
    )


def ingest_lines(
    path: str | Path,
    lines_with_numbers: Iterable[tuple[int, bytes]],
    *,
    source: str,
    prompt_id: str,
    input_format: str = "auto",
) -> IngestReport:
    """Ingest ``(lineno, raw bytes)`` pairs; the core behind :func:`ingest_file`."""
    if input_format not in _INPUT_FORMATS:
        raise ValueError(f"unknown input format {input_format!r}; expected one of {_INPUT_FORMATS}")
    lines = list(lines_with_numbers)
    report = IngestReport(sources=[source], path=Path(path))
    report.n_lines = len(lines)
    report.input_format = _resolve_format(lines, input_format)

    counts: dict[str, int] = {}
    id_texts: dict[str, str] = {}
    text_lines: dict[str, list[int]] = {}
    text_ids: dict[str, list[str]] = {}

    for lineno, chunk in lines:
        text, decode_error = _decode(chunk, lineno)
        if decode_error is not None:
            report.errors.append(decode_error)
            continue
        assert text is not None
        if report.input_format == "jsonl":
            _ingest_jsonl_line(
                text, lineno, report, source, prompt_id, counts, id_texts, text_lines, text_ids
            )
        else:
            _ingest_text_line(
                text, lineno, report, source, prompt_id, counts, id_texts, text_lines, text_ids
            )

    for value, lines_seen in text_lines.items():
        if len(lines_seen) < 2:
            continue
        ids = tuple(text_ids[value])
        report.duplicate_texts.append(DuplicateGroup(value, tuple(lines_seen), ids))
    report.duplicate_texts.sort(key=lambda group: group.lines[0])
    for group in report.duplicate_texts:
        lines_txt = ", ".join(str(line) for line in group.lines)
        ids_txt = ", ".join(repr(record_id) for record_id in group.ids)
        report.warnings.append(
            Issue(
                group.lines[0],
                group.ids[0],
                f"duplicate text {_preview(group.value)!r} at lines {lines_txt} (ids {ids_txt})",
            )
        )

    report.errors.sort(key=lambda issue: issue.line)
    report.warnings.sort(key=lambda issue: issue.line)
    report.n_records = len(report.records)
    return report


def merge_reports(reports: Iterable[IngestReport]) -> IngestReport:
    """Combine per-input reports into one, concatenating records in input order."""
    reports = list(reports)
    sources: list[str] = []
    for report in reports:
        for source in report.sources:
            if source not in sources:
                sources.append(source)
    formats = {report.input_format for report in reports}
    resolved = formats.pop() if len(formats) == 1 else "mixed"
    merged = IngestReport(sources=sources, input_format=resolved or "auto", path=None)
    for report in reports:
        merged.n_lines += report.n_lines
        merged.records.extend(report.records)
        merged.errors.extend(report.errors)
        merged.warnings.extend(report.warnings)
        merged.duplicate_texts.extend(report.duplicate_texts)
        merged.normalized_lines.extend(report.normalized_lines)
        merged.dropped_id_lines.extend(report.dropped_id_lines)
        for key, count in report.dropped_keys.items():
            merged.dropped_keys[key] = merged.dropped_keys.get(key, 0) + count
    merged.n_records = len(merged.records)
    return merged


def render_report(report: IngestReport) -> str:
    """Render an ingest report as a summary line, findings, and dropped-key summary."""
    label = str(report.path) if report.path is not None else "-"
    lines = [
        f"{label}: {report.input_format} input, {report.n_lines} line(s), "
        f"{report.n_records} record(s), {len(report.errors)} error(s), "
        f"{len(report.warnings)} warning(s)"
    ]
    for issue in report.errors:
        lines.append(f"  error line {issue.line} [{issue.record_id or '-'}]: {issue.message}")
    for issue in report.warnings:
        lines.append(f"  warning line {issue.line} [{issue.record_id or '-'}]: {issue.message}")
    if report.dropped_keys:
        summary = ", ".join(f"{key} x{count}" for key, count in report.dropped_keys.items())
        lines.append(f"  dropped input keys: {summary}")
    if report.dropped_id_lines:
        dropped = ", ".join(str(line) for line in report.dropped_id_lines)
        lines.append(f"  discarded input id on line(s): {dropped}")
    return "\n".join(lines)


def _ingest_jsonl_line(
    text: str,
    lineno: int,
    report: IngestReport,
    source: str,
    prompt_id: str,
    counts: dict[str, int],
    id_texts: dict[str, str],
    text_lines: dict[str, list[int]],
    text_ids: dict[str, list[str]],
) -> None:
    """Parse one JSONL line, dropping unknown keys and deriving the record id."""
    if not text.strip():
        return
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        report.errors.append(Issue(lineno, None, f"invalid JSON: {exc.msg}"))
        return
    if not isinstance(obj, dict):
        report.errors.append(Issue(lineno, None, "record: expected a JSON object"))
        return
    if "text" not in obj:
        report.errors.append(Issue(lineno, None, "text: missing"))
        return
    value = obj["text"]
    if not isinstance(value, str):
        report.errors.append(Issue(lineno, None, "text: must be a string"))
        return
    if not value:
        report.errors.append(Issue(lineno, None, "text: must not be empty"))
        return

    pending: list[str] = []
    if "id" in obj:
        report.dropped_id_lines.append(lineno)
        pending.append("input field 'id' ignored; id is derived from the text")
    for key in obj:
        if key in ("text", "id"):
            continue
        report.dropped_keys[key] = report.dropped_keys.get(key, 0) + 1
        pending.append(f"dropped input key {key!r}; only 'text' is read")

    shape_error = _shape_error(value, lineno)
    if shape_error is not None:
        report.errors.append(shape_error)
        return
    _accept_text(
        value, lineno, report, source, prompt_id, pending, counts, id_texts, text_lines, text_ids
    )


def _ingest_text_line(
    text: str,
    lineno: int,
    report: IngestReport,
    source: str,
    prompt_id: str,
    counts: dict[str, int],
    id_texts: dict[str, str],
    text_lines: dict[str, list[int]],
    text_ids: dict[str, list[str]],
) -> None:
    """Ingest one line of the plain-text format."""
    if text == "":
        return
    if not text.strip():
        report.warnings.append(Issue(lineno, None, "whitespace-only line skipped"))
        return
    shape_error = _shape_error(text, lineno)
    if shape_error is not None:
        report.errors.append(shape_error)
        return
    _accept_text(
        text, lineno, report, source, prompt_id, [], counts, id_texts, text_lines, text_ids
    )


def _accept_text(
    text: str,
    lineno: int,
    report: IngestReport,
    source: str,
    prompt_id: str,
    pending: list[str],
    counts: dict[str, int],
    id_texts: dict[str, str],
    text_lines: dict[str, list[int]],
    text_ids: dict[str, list[str]],
) -> None:
    """Normalize, identify, and record one accepted text (or report a digest collision)."""
    final = unicodedata.normalize("NFC", text)
    if final != text:
        report.normalized_lines.append(lineno)
        pending.append(f"text normalized to NFC: {_preview(text)!r} -> {_preview(final)!r}")

    occurrence = counts.get(final, 0) + 1
    counts[final] = occurrence
    record_id = derive_id(source=source, prompt_id=prompt_id, text=final, occurrence=occurrence)

    previous = id_texts.get(record_id)
    if previous is not None and previous != final:
        message = (
            f"id collision: {_preview(previous)!r} and {_preview(final)!r} share id {record_id!r}"
        )
        report.errors.append(Issue(lineno, record_id, message))
        return
    id_texts[record_id] = final
    text_lines.setdefault(final, []).append(lineno)
    text_ids.setdefault(final, []).append(record_id)

    record = {"id": record_id, "text": final, "source": source, "prompt_id": prompt_id}
    problems = validate_record(record)
    assert not problems, f"ingest produced an invalid record {record!r}: {problems}"
    report.records.append(record)
    for message in pending:
        report.warnings.append(Issue(lineno, record_id, message))


def _shape_error(text: str, lineno: int) -> Issue | None:
    """Return the contract violation in ``text``, if any (issues never carry an id)."""
    if text != text.strip():
        return Issue(lineno, None, f"text must not have leading or trailing whitespace: {text!r}")
    bad = next((char for char in _FORBIDDEN_CHARS if char in text), None)
    if bad is not None:
        return Issue(lineno, None, f"text must not contain {bad!r}: {_preview(text)!r}")
    return None


def _decode(chunk: bytes, lineno: int) -> tuple[str | None, Issue | None]:
    """Drop one trailing CR, then decode ``chunk`` as strict UTF-8."""
    if chunk.endswith(b"\r"):
        chunk = chunk[:-1]
    try:
        return chunk.decode("utf-8"), None
    except UnicodeDecodeError as exc:
        return None, Issue(lineno, None, f"invalid UTF-8 at byte offset {exc.start}: {exc.reason}")


def _resolve_format(lines: Sequence[tuple[int, bytes]], requested: str) -> str:
    """Resolve ``auto`` from the first line with content: ``{`` means JSONL, else text."""
    if requested != "auto":
        return requested
    for _lineno, chunk in lines:
        probe = chunk[:-1] if chunk.endswith(b"\r") else chunk
        text = probe.decode("utf-8", "replace")
        if not text.strip():
            continue
        return "jsonl" if text.lstrip().startswith("{") else "text"
    return "jsonl"


def _numbered_lines(data: bytes) -> list[tuple[int, bytes]]:
    """Split raw file bytes into 1-based physical lines (no trailing empty line)."""
    if not data:
        return []
    chunks = data.split(b"\n")
    if chunks and chunks[-1] == b"":
        chunks.pop()
    return list(enumerate(chunks, start=1))


def _preview(text: str, limit: int = 40) -> str:
    """Flatten and shorten a text for use inside a message."""
    flat = text.replace("\n", "\\n").replace("\r", "\\r")
    return flat if len(flat) <= limit else flat[: limit - 1] + "\u2026"
