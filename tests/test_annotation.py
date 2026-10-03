from __future__ import annotations

import json

import pytest

from gidi.annotation.queue import build_queue
from gidi.annotation.schema import load_config, trainable, validate_annotation, validate_file

CONFIG = load_config()
TEXT = "cho Nam vay 500k"


def ann(**overrides):
    record = {
        "id": "r1",
        "annotation_status": "complete",
        "type": "lend",
        "target": {"text": "Nam", "start": 4, "end": 7},
    }
    record.update(overrides)
    return record


def test_config_holds_the_canonical_taxonomy_without_other():
    assert CONFIG.types == (
        "expense", "income", "borrow", "lend",
        "repayment_in", "repayment_out", "transfer", "refund",
    )  # fmt: skip
    assert CONFIG.trainable_statuses == {"complete"}


def test_valid_span_and_null_target_pass():
    assert validate_annotation(ann(), TEXT, CONFIG) == []
    assert validate_annotation(ann(type="expense", target=None), "ăn trưa 80k", CONFIG) == []


def test_code_point_offsets_on_accented_text():
    text = "ăn với Nam ở Pizza 4P 800k"
    start = text.index("Pizza 4P")
    target = {"text": "Pizza 4P", "start": start, "end": start + 8}
    assert validate_annotation(ann(type="expense", target=target), text, CONFIG) == []


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ({"text": "Nam", "start": 3, "end": 6}, "text[3:6] is ' Na', not 'Nam'"),
        ({"text": "Hùng", "start": 4, "end": 8}, "not 'Hùng'"),
        ({"text": "Nam", "start": 7, "end": 4}, "start 7 must be less than end 4"),
        ({"text": "Nam", "start": 4, "end": 99}, "beyond the text length"),
        ({"text": " Nam", "start": 3, "end": 7}, "leading or trailing whitespace"),
        ({"text": "Nam", "start": True, "end": 7}, "target.start: expected a non-negative"),
        ({"text": "Nam", "start": 4}, "missing field 'end'"),
    ],
)
def test_invalid_spans_are_reported(target, expected):
    problems = validate_annotation(ann(target=target), TEXT, CONFIG)
    assert any(expected in p for p in problems), problems


def test_decomposed_target_text_is_rejected():
    text = "mượn Hùng 500k"
    nfd = "Hu\u0300ng"
    problems = validate_annotation(
        ann(type="borrow", target={"text": nfd, "start": 5, "end": 10}), text, CONFIG
    )
    assert any("NFC" in p for p in problems)


def test_unknown_type_and_status_and_fields_are_rejected():
    problems = validate_annotation(
        ann(type="other", annotation_status="done", extra=1), TEXT, CONFIG
    )
    assert any("type: 'other'" in p for p in problems)
    assert any("annotation_status: 'done'" in p for p in problems)
    assert "unknown field 'extra'" in problems


def test_complete_requires_type_but_uncertain_and_skipped_may_be_null():
    assert validate_annotation(ann(type=None), TEXT, CONFIG) == [
        "type: required when annotation_status is 'complete'"
    ]
    for status in ("uncertain", "skipped"):
        record = ann(annotation_status=status, type=None, target=None, note="why")
        assert validate_annotation(record, TEXT, CONFIG) == []


@pytest.mark.parametrize("note", [None, "", "   ", 3])
def test_uncertain_requires_a_non_empty_note(note):
    record = ann(annotation_status="uncertain", type=None, target=None)
    if note is not None:
        record["note"] = note
    assert "note: required (non-empty) when annotation_status is 'uncertain'" in (
        validate_annotation(record, TEXT, CONFIG)
    )


def test_note_is_optional_for_complete_and_skipped():
    assert validate_annotation(ann(), TEXT, CONFIG) == []
    skipped = ann(annotation_status="skipped", type=None, target=None)
    assert validate_annotation(skipped, TEXT, CONFIG) == []


def test_debt_state_note_is_skipped_and_never_trainable():
    record = ann(annotation_status="skipped", type=None, target=None)
    assert validate_annotation(record, "còn nợ Hùng 300k tiền ăn", CONFIG) == []
    assert trainable([record], CONFIG) == []


def test_skipped_rejects_a_retained_type_or_target():
    target = {"text": "Nam", "start": 4, "end": 7}
    record = ann(annotation_status="skipped", type="lend", target=target)
    assert validate_annotation(record, TEXT, CONFIG) == [
        "type: must be null when annotation_status is 'skipped'",
        "target: must be null when annotation_status is 'skipped'",
    ]
    assert validate_annotation(ann(annotation_status="skipped", target=None), TEXT, CONFIG) == [
        "type: must be null when annotation_status is 'skipped'"
    ]


def test_uncertain_values_are_still_validated():
    record = ann(annotation_status="uncertain", target={"text": "Nam", "start": 0, "end": 3})
    assert validate_annotation(record, TEXT, CONFIG)


def test_uncertain_may_record_a_clear_target_without_a_type():
    # "Nam gửi tao 500k": counterparty clear, type not established by the note.
    target = {"text": "Nam", "start": 0, "end": 3}
    record = ann(annotation_status="uncertain", type=None, target=target, note="purpose unstated")
    assert validate_annotation(record, "Nam gửi tao 500k", CONFIG) == []
    assert trainable([record], CONFIG) == []


def span(text, sub):
    start = text.index(sub)
    return {"text": sub, "start": start, "end": start + len(sub)}


@pytest.mark.parametrize(
    ("text", "kind", "target"),
    [
        ("trả chủ nhà 6tr", "expense", "chủ nhà"),  # multi-word role noun
        ("cty hoàn tiền công tác 780k", "refund", "cty"),  # employer reimbursement
        ("chị Linh trả nợ 1tr", "repayment_in", "Linh"),  # prefix excluded, implied user
        ("ck momo cho Nam 200k", None, "Nam"),  # channel not target; purpose unstated
        ("trả KFC qua momo 200k", "expense", "KFC"),
        ("momo hoàn 50k", "refund", "momo"),  # platform as counterparty
    ],
)
def test_resolved_decision_examples_validate(text, kind, target):
    status = "complete" if kind else "uncertain"
    record = ann(annotation_status=status, type=kind, target=span(text, target), note="n")
    assert validate_annotation(record, text, CONFIG) == []


@pytest.mark.parametrize(
    ("text", "kind", "target"),
    [
        ("trả spaylater 560k", "repayment_out", "spaylater"),  # BNPL
        ("home credit kỳ này 1tr250", "repayment_out", "home credit"),
        ("thanh toán dư nợ thẻ 4tr", "repayment_out", None),  # issuer not named
        ("mua ccq 5tr", "transfer", None),  # investment
        ("đổi 10tr sang usd", "transfer", None),
        ("mua nhẫn vàng tặng vợ", "expense", None),  # consumption; vợ is beneficiary
        ("mua nhẫn PNJ nửa chỉ", "transfer", None),  # vi-VN quantified precious asset
        ("mua nhẫn vàng 1 chỉ tặng vợ", "expense", None),  # gift intent overrides quantity
        ("momo cashback 50k", "refund", "momo"),
        ("cf highlands 59k", "expense", "highlands"),  # named merchant
        ("tra sua vs Linh 55k", "expense", None),  # vs = companion
    ],
)
def test_frozen_v1_examples_validate(text, kind, target):
    record = ann(type=kind, target=span(text, target) if target else None)
    assert validate_annotation(record, text, CONFIG) == []


def test_own_wallet_top_up_is_transfer_with_null_target():
    text = "nạp momo 200k"
    assert validate_annotation(ann(type="transfer", target=None), text, CONFIG) == []
    assert validate_annotation(ann(type="transfer", target=span(text, "momo")), text, CONFIG) == [
        "target: must be null for type 'transfer'"
    ]


def test_transfer_requires_null_target():
    text = "ck 5tr qua tk tiết kiệm"
    target = {"text": "tk tiết kiệm", "start": 11, "end": 23}
    assert validate_annotation(ann(type="transfer", target=target), text, CONFIG) == [
        "target: must be null for type 'transfer'"
    ]


def test_missing_fields_are_required_explicitly():
    problems = validate_annotation({"id": "r1", "annotation_status": "skipped"}, TEXT, CONFIG)
    assert problems == ["missing field 'type'", "missing field 'target'"]


def test_trainable_keeps_only_complete():
    records = [ann(), ann(annotation_status="uncertain"), ann(annotation_status="skipped")]
    assert trainable(records, CONFIG) == [records[0]]


def test_validate_file_flags_unknown_and_duplicate_ids(tmp_path):
    path = tmp_path / "labels.jsonl"
    lines = [ann(), ann(), ann(id="zz", type="expense", target=None), "{bad"]
    path.write_text(
        "\n".join(x if isinstance(x, str) else json.dumps(x) for x in lines) + "\n",
        encoding="utf-8",
    )
    report = validate_file(path, {"r1": TEXT}, CONFIG)

    messages = [(i.line, i.message) for i in report.errors]
    assert (2, "duplicate id (first seen at line 1)") in messages
    assert (3, "id: not in the annotation queue") in messages
    assert any(line == 4 and m.startswith("invalid JSON") for line, m in messages)
    assert report.status_counts == {"complete": 3}


def corpus_records(n, status="approved"):
    return [{"id": f"c{i}", "text": f"note {i}", "quet": {"status": status}} for i in range(n)]


def test_queue_is_deterministic_and_independent_of_input_order():
    records = corpus_records(20)
    kwargs = dict(size=10, seed="s", review_status="approved", corpus="c.jsonl")
    first = build_queue(records, **kwargs)
    again = build_queue(list(reversed(records)), **kwargs)

    assert first == again
    assert [q["position"] for q in first] == list(range(1, 11))
    assert all(q["text"] == f"note {q['id'][1:]}" for q in first)
    assert build_queue(records, **{**kwargs, "seed": "other"}) != first


def test_queue_only_takes_approved_and_fails_when_short():
    records = corpus_records(3) + [
        {"id": "x", "text": "t", "quet": {"status": "rejected"}},
        {"id": "y", "text": "t"},
    ]
    queue = build_queue(records, size=3, seed="s", review_status="approved", corpus="c")
    assert {q["id"] for q in queue} == {"c0", "c1", "c2"}
    with pytest.raises(ValueError, match="only 3 'approved'"):
        build_queue(records, size=4, seed="s", review_status="approved", corpus="c")
