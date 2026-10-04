from __future__ import annotations

import json
from pathlib import Path

import yaml

from gidi.annotation.combined import (
    DEFAULT_COMBINED_SCHEMA,
    is_combined_schema,
    load_combined_config,
    validate_combined_file,
    validate_combined_label,
)
from gidi.annotation.value_span import DEFAULT_VALUE_CONFIG

CONFIG = load_combined_config()
TEXT = "ăn trưa highlands 45k"


def span(text: str, sub: str) -> dict:
    start = text.index(sub)
    return {"text": sub, "start": start, "end": start + len(sub)}


def label(text: str = TEXT, **overrides) -> dict:
    record = {
        "id": "r1",
        "annotation_status": "complete",
        "type": "expense",
        "target": span(text, "highlands"),
        "value": span(text, "45k"),
        "span_status": {"value": "complete"},
    }
    record.update(overrides)
    return record


def test_expense_with_target_and_value_is_valid():
    assert validate_combined_label(label(), TEXT, CONFIG) == []


def test_null_spans_and_missing_span_status_are_valid():
    # no counterparty, no amount stated; Quet may omit span_status
    record = label(target=None, value=None)
    del record["span_status"]
    assert validate_combined_label(record, TEXT, CONFIG) == []


def test_transfer_with_a_target_is_rejected():
    problems = validate_combined_label(label(type="transfer"), TEXT, CONFIG)
    assert problems == ["target: must be null for type 'transfer'"]


def test_value_text_must_equal_the_slice():
    value = {"text": "50k", "start": 18, "end": 21}
    problems = validate_combined_label(label(value=value), TEXT, CONFIG)
    assert problems == ["value: text[18:21] is '45k', not '50k'"]


def test_value_offsets_must_stay_inside_the_note():
    value = {"text": "45k", "start": 20, "end": 23}
    (problem,) = validate_combined_label(label(value=value), TEXT, CONFIG)
    assert problem.startswith("value: end 23 is beyond the text length")


def test_offsets_are_code_points_on_vietnamese_text():
    text = "đóng tiền điện Điện lực HCM 612.000đ"
    record = {
        "id": "r2",
        "annotation_status": "complete",
        "type": "expense",
        "target": span(text, "Điện lực HCM"),
        "value": span(text, "612.000đ"),
    }
    assert record["value"]["start"] == 28  # code points; UTF-8 byte offsets would be larger
    assert validate_combined_label(record, text, CONFIG) == []
    record["value"]["start"] += 1
    assert validate_combined_label(record, text, CONFIG)


def test_value_uncertain_needs_a_note():
    record = label(span_status={"value": "uncertain"})
    assert validate_combined_label(record, TEXT, CONFIG) == [
        "note: required (non-empty) when span_status.value is 'uncertain'"
    ]
    assert validate_combined_label(
        label(span_status={"value": "uncertain"}, note=" "), TEXT, CONFIG
    )
    ok = label(span_status={"value": "uncertain"}, note="45k or the 2 nights?")
    assert validate_combined_label(ok, TEXT, CONFIG) == []


def test_span_status_rejects_other_spans_and_statuses():
    assert validate_combined_label(label(span_status={"target": "complete"}), TEXT, CONFIG) == [
        "span_status: unknown span 'target'"
    ]
    (problem,) = validate_combined_label(label(span_status={"value": "skipped"}), TEXT, CONFIG)
    assert problem.startswith("span_status.value: 'skipped' is not one of")


def test_skipped_requires_every_span_null_and_no_span_status():
    skipped = label(annotation_status="skipped", type=None, target=None, value=None)
    del skipped["span_status"]
    assert validate_combined_label(skipped, TEXT, CONFIG) == []

    with_value = dict(skipped, value=span(TEXT, "45k"))
    assert validate_combined_label(with_value, TEXT, CONFIG) == [
        "value: must be null when annotation_status is 'skipped'"
    ]
    with_status = dict(skipped, span_status={"value": "complete"})
    assert validate_combined_label(with_status, TEXT, CONFIG) == [
        "span_status: must be absent when annotation_status is 'skipped'"
    ]


def test_unknown_and_missing_fields_are_errors():
    problems = validate_combined_label(label(price=1), TEXT, CONFIG)
    assert problems == ["unknown field 'price'"]
    record = label()
    del record["value"]
    assert validate_combined_label(record, TEXT, CONFIG) == ["missing field 'value'"]


def test_v1_rules_still_apply():
    assert validate_combined_label(label(annotation_status="uncertain"), TEXT, CONFIG) == [
        "note: required (non-empty) when annotation_status is 'uncertain'"
    ]
    assert validate_combined_label(label(type="other"), TEXT, CONFIG)


def test_file_validation_reports_line_and_duplicate_ids(tmp_path: Path):
    path = tmp_path / "labels.jsonl"
    rows = [label(), label(value={"text": "x", "start": 0, "end": 1})]
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    report = validate_combined_file(path, {"r1": TEXT}, CONFIG)
    assert [(i.line, i.message) for i in report.errors] == [
        (2, "duplicate id (first seen at line 1)"),
        (2, "value: text[0:1] is 'ă', not 'x'"),
    ]


def test_quet_schema_is_a_multi_span_schema_in_ui_order():
    raw = yaml.safe_load(DEFAULT_COMBINED_SCHEMA.read_text(encoding="utf-8"))
    assert list(raw["spans"]) == ["target", "value"]
    assert "null_target_types" not in raw  # Quet rejects it together with spans:
    assert is_combined_schema(DEFAULT_COMBINED_SCHEMA)
    assert not is_combined_schema(DEFAULT_VALUE_CONFIG.parent / "annotation-v1.yaml")


def test_quet_schema_types_and_statuses_match_annotation_v2_contract():
    quet = yaml.safe_load(DEFAULT_COMBINED_SCHEMA.read_text(encoding="utf-8"))
    contract = yaml.safe_load(DEFAULT_VALUE_CONFIG.read_text(encoding="utf-8"))
    assert list(quet["types"]) == list(contract["types"]) and len(quet["types"]) == 8
    assert list(quet["statuses"]) == list(contract["statuses"])
    assert quet["null_label_statuses"] == contract["null_label_statuses"]
    assert quet["spans"]["target"]["null_for_types"] == contract["null_target_types"]
    assert quet["version"] == contract["version"]
