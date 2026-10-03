import json

import pytest

from gidi.evaluation.metrics import evaluate, span_metrics, type_metrics
from gidi.modeling.preprocessing import TYPES


def span(start, end, text="x"):
    return {"text": text, "start": start, "end": end}


def test_type_metrics_hand_built():
    gold = ["expense", "expense", "expense", "income", "income", "lend"]
    pred = ["expense", "expense", "income", "income", "expense", "lend"]
    m = type_metrics(gold, pred)
    assert m["n"] == 6
    assert m["accuracy"] == pytest.approx(4 / 6)
    expense, income, lend = (m["per_class"][t] for t in ("expense", "income", "lend"))
    assert (expense["precision"], expense["recall"]) == (pytest.approx(2 / 3), pytest.approx(2 / 3))
    assert (income["precision"], income["recall"]) == (pytest.approx(1 / 2), pytest.approx(1 / 2))
    assert lend["f1"] == 1.0
    assert [m["per_class"][t]["support"] for t in TYPES] == [3, 2, 0, 1, 0, 0, 0, 0]
    # macro over supported classes only; the all-class variant counts absent classes as 0
    assert m["macro_f1"] == pytest.approx((2 / 3 + 1 / 2 + 1) / 3)
    assert m["macro_f1_all_classes"] == pytest.approx((2 / 3 + 1 / 2 + 1) / 8)
    cm = m["confusion_matrix"]
    assert len(cm) == 8 and all(len(row) == 8 for row in cm)
    assert cm[0][0] == 2 and cm[0][1] == 1 and cm[1][0] == 1 and cm[1][1] == 1
    assert cm[TYPES.index("lend")][TYPES.index("lend")] == 1
    assert sum(map(sum, cm)) == 6


def test_type_metrics_rejects_unknown_label():
    with pytest.raises(ValueError, match="unknown"):
        type_metrics(["expense"], ["gift"])


def test_span_exact_vs_partial_vs_null():
    gold = [span(0, 4), span(5, 9), span(2, 6), None, None, span(1, 3)]
    pred = [span(0, 4), span(5, 8), None, None, span(0, 2), span(0, 3)]
    m = span_metrics(gold, pred)
    # exact: #0 and the null==null #3 count; partial overlap #1/#5 and misses do not
    assert m["n"] == 6
    assert m["exact_match"] == pytest.approx(2 / 6)
    assert (m["tp"], m["n_pred"], m["n_gold"]) == (1, 4, 4)
    assert m["precision"] == pytest.approx(1 / 4)
    assert m["recall"] == pytest.approx(1 / 4)
    assert m["f1"] == pytest.approx(1 / 4)

    target = m["gold_target"]
    assert target["n"] == 4
    assert target["exact_match"] == pytest.approx(1 / 4)  # null==null does not inflate this
    assert (target["tp"], target["n_pred"]) == (1, 3)
    assert target["precision"] == pytest.approx(1 / 3)
    assert target["recall"] == pytest.approx(1 / 4)

    null = m["gold_null"]
    assert null["n"] == 2
    assert null["null_accuracy"] == pytest.approx(1 / 2)
    assert null["false_span_rate"] == pytest.approx(1 / 2)
    assert null["n_false_span"] == 1


def test_span_compares_offsets_not_text():
    assert span_metrics([span(0, 3, "Nam")], [span(4, 7, "Nam")])["exact_match"] == 0.0


def test_span_metrics_edge_cases():
    all_null = span_metrics([None, None], [None, None])
    assert all_null["exact_match"] == 1.0
    assert all_null["f1"] == 0.0  # no non-null spans anywhere: nothing to score
    assert all_null["gold_target"]["n"] == 0 and all_null["gold_target"]["exact_match"] is None
    only_targets = span_metrics([span(0, 1)], [span(0, 1)])
    assert only_targets["f1"] == 1.0
    assert only_targets["gold_null"] == {
        "n": 0,
        "null_accuracy": None,
        "false_span_rate": None,
        "n_false_span": 0,
    }
    silent = span_metrics([span(0, 1)], [None])
    assert (silent["precision"], silent["recall"], silent["f1"]) == (0.0, 0.0, 0.0)


def test_evaluate_overall_and_slices_report_counts():
    records = [
        {"type": "expense", "target": span(0, 3), "accented": False, "source_batch": "a"},
        {"type": "income", "target": None, "accented": True, "source_batch": "a"},
        {"type": "lend", "target": span(2, 5), "accented": True, "source_batch": "b"},
    ]
    types = ["expense", "expense", "lend"]
    spans = [span(0, 3), None, span(2, 4)]
    result = evaluate(records, types, spans)
    assert result["n"] == 3
    assert result["overall"]["type"]["accuracy"] == pytest.approx(2 / 3)
    assert result["overall"]["target"]["exact_match"] == pytest.approx(2 / 3)

    accented = result["slices"]["accented"]
    assert set(accented) == {"true", "false"}
    assert accented["true"]["n"] == 2 and accented["false"]["n"] == 1
    assert accented["false"]["type"]["accuracy"] == 1.0
    assert accented["true"]["target"]["exact_match"] == pytest.approx(1 / 2)
    batches = result["slices"]["source_batch"]
    assert {k: v["n"] for k, v in batches.items()} == {"a": 2, "b": 1}
    assert sum(v["n"] for v in batches.values()) == result["n"]
    json.dumps(result)  # JSON-able


def test_evaluate_derives_accented_from_text_and_checks_lengths():
    records = [
        {"text": "tra no Nam", "type": "repayment_out", "target": None},
        {"text": "trả nợ Nam", "type": "repayment_out", "target": None},
    ]
    result = evaluate(records, ["repayment_out"] * 2, [None, None], slices=("accented",))
    assert {k: v["n"] for k, v in result["slices"]["accented"].items()} == {"false": 1, "true": 1}
    with pytest.raises(ValueError):
        evaluate(records, ["repayment_out"], [None, None])
