"""Validation of combined annotation-v2 labels (type + target span + value span in one Quet pass).

The contract is ``configs/annotation-v2.quet.yaml`` (a Quet multi-span schema). A label line is::

    {"id": "baseline-01-…", "annotation_status": "complete", "type": "expense",
     "target": {"text": "highlands", "start": 4, "end": 13},
     "value": {"text": "45k", "start": 14, "end": 17},
     "span_status": {"value": "complete"}, "note": "optional"}

Type and target follow annotation-v1 (``gidi.annotation.schema.validate_annotation``), the value
span follows the annotation-v2 span mechanics (``gidi.annotation.value_span.validate_span``).
``span_status`` only carries the ``value`` span; a value marked ``uncertain`` needs a non-empty
``note`` (Gidi's rule, like ``uncertain`` for the whole annotation). ``skipped`` labels have a null
type, target and value and no ``span_status``. The rules live in the sibling modules; this one
only combines them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from gidi.annotation.schema import (
    AnnotationConfig,
    AnnotationReport,
    load_config,
    validate_annotation,
    validate_file,
)
from gidi.annotation.value_span import DEFAULT_VALUE_CONFIG, validate_span

DEFAULT_COMBINED_SCHEMA = (
    Path(__file__).resolve().parents[3] / "configs" / "annotation-v2.quet.yaml"
)

COMBINED_SPANS: tuple[str, ...] = ("target", "value")
COMBINED_FIELDS: tuple[str, ...] = ("id", "annotation_status", "type", *COMBINED_SPANS)
COMBINED_OPTIONAL: tuple[str, ...] = ("span_status", "note")
#: spans of the combined pass that carry a status (``span_status`` keys)
STATUS_SPAN = "value"


@dataclass(frozen=True)
class CombinedConfig:
    """The annotation-v1 rules (types, statuses, null rules) plus the value span's statuses."""

    annotation: AnnotationConfig
    value_statuses: tuple[str, ...]
    value_null_types: frozenset[str]


def is_combined_schema(path: str | Path) -> bool:
    """Whether ``path`` is a Quet multi-span schema (has a ``spans:`` mapping)."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return isinstance(raw, dict) and "spans" in raw


def load_combined_config(
    path: str | Path = DEFAULT_COMBINED_SCHEMA,
    contract: str | Path = DEFAULT_VALUE_CONFIG,
) -> CombinedConfig:
    """Types, statuses and span rules from the Quet schema.

    Rules Quet has no field for (``uncertain`` needs a note) come from the annotation-v2
    ``contract``.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    spans = raw.get("spans")
    if not isinstance(spans, dict) or set(spans) != set(COMBINED_SPANS):
        raise ValueError(f"{path}: expected spans {list(COMBINED_SPANS)}, got {spans!r}")
    base = load_config(contract)
    annotation = replace(
        base,
        version=str(raw["version"]),
        types=tuple(raw["types"]),
        statuses=tuple(raw["statuses"]),
        null_target_types=frozenset(spans["target"].get("null_for_types", ())),
        null_label_statuses=frozenset(raw.get("null_label_statuses", ())),
    )
    value = spans["value"]
    return CombinedConfig(
        annotation=annotation,
        value_statuses=tuple(value["statuses"]),
        value_null_types=frozenset(value.get("null_for_types", ())),
    )


def validate_combined_label(record: Any, text: str | None, config: CombinedConfig) -> list[str]:
    """Contract violations of one combined label; empty means valid.

    ``text`` is the NFC note of the referenced record, or ``None`` when the id is unknown
    (offsets are then only checked structurally).
    """
    if not isinstance(record, dict):
        return ["record: expected a JSON object"]
    ann = config.annotation
    allowed = set(COMBINED_FIELDS) | set(COMBINED_OPTIONAL)
    problems = [f"unknown field {key!r}" for key in record if key not in allowed]
    if "value" not in record:
        problems.append("missing field 'value'")
    # type/target/status/note/id: the unchanged annotation-v1 rules (also reports missing fields)
    v1_view = {key: val for key, val in record.items() if key in set(COMBINED_FIELDS) - {"value"}}
    v1_view |= {"note": record["note"]} if "note" in record else {}
    problems += validate_annotation(v1_view, text, ann)

    status = record.get("annotation_status")
    kind = record.get("type")
    value = record.get("value")
    if value is not None:
        problems += validate_span(value, text, "value")
        if kind in config.value_null_types:
            problems.append(f"value: must be null for type {kind!r}")
    if status in ann.null_label_statuses and value is not None:
        problems.append(f"value: must be null when annotation_status is {status!r}")

    problems += _validate_span_status(record, status, config)
    return problems


def _validate_span_status(record: dict[str, Any], status: Any, config: CombinedConfig) -> list[str]:
    if "span_status" not in record:
        return []
    span_status = record["span_status"]
    if not isinstance(span_status, dict):
        return ["span_status: expected an object"]
    if status in config.annotation.null_label_statuses:
        return [f"span_status: must be absent when annotation_status is {status!r}"]
    problems = [f"span_status: unknown span {key!r}" for key in span_status if key != STATUS_SPAN]
    value_status = span_status.get(STATUS_SPAN)
    if STATUS_SPAN in span_status:
        if value_status not in config.value_statuses:
            problems.append(
                f"span_status.{STATUS_SPAN}: {value_status!r} is not one of "
                f"{list(config.value_statuses)}"
            )
        elif value_status in config.annotation.note_required_statuses:
            note = record.get("note")
            if not (isinstance(note, str) and note.strip()):
                problems.append(
                    f"note: required (non-empty) when span_status.{STATUS_SPAN} is {value_status!r}"
                )
    return problems


def validate_combined_file(
    path: str | Path, texts: dict[str, str], config: CombinedConfig
) -> AnnotationReport:
    """Validate a combined labels JSONL against the queue texts (also rejects duplicate ids)."""
    return validate_file(path, texts, config, validator=validate_combined_label)
