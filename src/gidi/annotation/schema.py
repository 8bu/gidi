"""Validation for annotation-v1 records (the contract in ``configs/annotation-v1.yaml``).

An annotation labels one approved corpus record::

    {"id": "baseline-01-…", "annotation_status": "complete",
     "type": "lend", "target": {"text": "Nam", "start": 4, "end": 7}}

``target`` may be ``null``. Offsets are Unicode code points into the NFC corpus text, start
inclusive, end exclusive, and ``text[start:end]`` must reproduce ``target.text`` exactly.
``uncertain`` records may leave ``type``/``target`` null; any value they do carry is still
validated. ``skipped`` records must have both null. The taxonomy is read from the config, never
hard-coded here.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from gidi.corpus.jsonl import iter_jsonl
from gidi.corpus.schema import Issue

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "annotation-v1.yaml"

REQUIRED_FIELDS: tuple[str, ...] = ("id", "annotation_status", "type", "target")
OPTIONAL_FIELDS: tuple[str, ...] = ("note",)
TARGET_FIELDS: tuple[str, ...] = ("text", "start", "end")


@dataclass(frozen=True)
class AnnotationConfig:
    """The parts of ``annotation-v1.yaml`` the validator and queue builder use."""

    version: str
    types: tuple[str, ...]
    null_target_types: frozenset[str]
    statuses: tuple[str, ...]
    trainable_statuses: frozenset[str]
    note_required_statuses: frozenset[str]
    null_label_statuses: frozenset[str]
    queue: dict[str, Any]


def load_config(path: str | Path = DEFAULT_CONFIG) -> AnnotationConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return AnnotationConfig(
        version=str(raw["version"]),
        types=tuple(raw["types"]),
        null_target_types=frozenset(raw.get("null_target_types", ())),
        statuses=tuple(raw["statuses"]),
        trainable_statuses=frozenset(raw["trainable_statuses"]),
        note_required_statuses=frozenset(raw.get("note_required_statuses", ())),
        null_label_statuses=frozenset(raw.get("null_label_statuses", ())),
        queue=dict(raw["queue"]),
    )


def validate_annotation(record: Any, text: str | None, config: AnnotationConfig) -> list[str]:
    """Return the contract violations in ``record``; empty means valid.

    ``text`` is the NFC corpus text of the referenced record, or ``None`` when the id is unknown
    (reported by the caller); span offsets are only checked against a known text.
    """
    if not isinstance(record, dict):
        return ["record: expected a JSON object"]
    problems = [f"missing field {key!r}" for key in REQUIRED_FIELDS if key not in record]
    allowed = set(REQUIRED_FIELDS) | set(OPTIONAL_FIELDS)
    problems += [f"unknown field {key!r}" for key in record if key not in allowed]

    record_id = record.get("id")
    if "id" in record and (not isinstance(record_id, str) or not record_id):
        problems.append("id: expected a non-empty string")
    if "note" in record and not isinstance(record["note"], str):
        problems.append("note: expected a string")

    status = record.get("annotation_status")
    if "annotation_status" in record and status not in config.statuses:
        problems.append(f"annotation_status: {status!r} is not one of {list(config.statuses)}")
    note = record.get("note")
    if status in config.note_required_statuses and not (isinstance(note, str) and note.strip()):
        problems.append(f"note: required (non-empty) when annotation_status is {status!r}")
    if status in config.null_label_statuses:
        for field in ("type", "target"):
            if record.get(field) is not None:
                problems.append(f"{field}: must be null when annotation_status is {status!r}")

    kind = record.get("type")
    if kind is None:
        if status == "complete":
            problems.append("type: required when annotation_status is 'complete'")
    elif kind not in config.types:
        problems.append(f"type: {kind!r} is not one of {list(config.types)}")

    target = record.get("target")
    if target is not None:
        problems += _validate_target(target, text)
        if kind in config.null_target_types:
            problems.append(f"target: must be null for type {kind!r}")
    return problems


def _validate_target(target: Any, text: str | None) -> list[str]:
    if not isinstance(target, dict):
        return ["target: expected an object or null"]
    problems = [f"target: missing field {key!r}" for key in TARGET_FIELDS if key not in target]
    problems += [f"target: unknown field {key!r}" for key in target if key not in TARGET_FIELDS]
    if problems:
        return problems

    span, start, end = target["text"], target["start"], target["end"]
    if not isinstance(span, str) or not span:
        problems.append("target.text: expected a non-empty string")
    elif span != span.strip():
        problems.append("target.text: has leading or trailing whitespace")
    elif span != unicodedata.normalize("NFC", span):
        problems.append("target.text: is not NFC-normalized")
    for name, value in (("start", start), ("end", end)):
        # bool is an int subclass; reject it explicitly.
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            problems.append(f"target.{name}: expected a non-negative integer")
    if problems:
        return problems
    if start >= end:
        return [f"target: start {start} must be less than end {end}"]
    if text is None:
        return []
    if end > len(text):
        return [f"target: end {end} is beyond the text length {len(text)}"]
    if text[start:end] != span:
        return [f"target: text[{start}:{end}] is {text[start:end]!r}, not {span!r}"]
    return []


@dataclass
class AnnotationReport:
    path: Path
    n_records: int = 0
    status_counts: dict[str, int] = field(default_factory=dict)
    errors: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def load_texts_by_id(path: str | Path) -> dict[str, str]:
    """``id -> NFC text`` from a JSONL file of records carrying ``id`` and ``text``."""
    return {
        record["id"]: unicodedata.normalize("NFC", record["text"]) for record in iter_jsonl(path)
    }


def validate_file(
    path: str | Path, texts: dict[str, str], config: AnnotationConfig
) -> AnnotationReport:
    """Validate an annotation JSONL file against the texts of the records it references."""
    path = Path(path)
    report = AnnotationReport(path=path)
    first_line: dict[str, int] = {}
    with path.open(encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                report.errors.append(Issue(lineno, None, f"invalid JSON: {exc.msg}"))
                continue
            report.n_records += 1
            record_id = record.get("id") if isinstance(record, dict) else None
            record_id = record_id if isinstance(record_id, str) and record_id else None
            text = texts.get(record_id) if record_id else None
            if record_id and text is None:
                report.errors.append(Issue(lineno, record_id, "id: not in the annotation queue"))
            if record_id in first_line:
                report.errors.append(
                    Issue(
                        lineno,
                        record_id,
                        f"duplicate id (first seen at line {first_line[record_id]})",
                    )
                )
            elif record_id:
                first_line[record_id] = lineno
            for message in validate_annotation(record, text, config):
                report.errors.append(Issue(lineno, record_id, message))
            if isinstance(record, dict):
                status = str(record.get("annotation_status"))
                report.status_counts[status] = report.status_counts.get(status, 0) + 1
    return report


def trainable(records: list[dict[str, Any]], config: AnnotationConfig) -> list[dict[str, Any]]:
    """Records allowed into the initial training split (``complete`` only)."""
    return [r for r in records if r.get("annotation_status") in config.trainable_statuses]


def render_report(report: AnnotationReport) -> str:
    counts = ", ".join(f"{k} {v}" for k, v in sorted(report.status_counts.items()))
    lines = [
        f"{report.path}: {report.n_records} annotation(s), {len(report.errors)} error(s)"
        + (f" [{counts}]" if counts else "")
    ]
    for issue in report.errors:
        lines.append(f"  error line {issue.line} [{issue.record_id or '-'}]: {issue.message}")
    return "\n".join(lines)
