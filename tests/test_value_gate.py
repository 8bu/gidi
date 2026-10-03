"""value-span-v1 gate (``gidi.evaluation.value_gate``): verdicts, thresholds, slice membership."""

from __future__ import annotations

import copy

import pytest

from gidi.evaluation import value_gate as vg
from gidi.evaluation.value_gate import (
    VALUE_SLICES,
    changes_vs_v1,
    evaluate_gate,
    multi_number_errors,
    summarize_set,
    value_slice_names,
)

SEEDS = (1, 2, 3)


def v2_seed(*, exact=0.96, human=0.92, present=0.98, multi=0.95, slices=None, joint=(90, 90, 80)):
    """Synthetic v2 summaries of one seed: test (value gate), test-v1 / probe note counts."""
    test_slices = {name: {"n": 20, "exact": 0.97} for name in VALUE_SLICES}
    test_slices["multi_number"] = {"n": 10, "exact": multi}
    test_slices.update(slices or {})
    type_c, target_c, joint_c = joint
    return {
        "test": {
            "value": {
                "n": 139,
                "exact": exact,
                "present_accuracy": present,
                "human": {"n": 12, "exact": human},
            },
            "value_slices": test_slices,
        },
        "test-v1": {
            "counts": {
                "n": 105,
                "type_correct": type_c,
                "target_correct": target_c,
                "joint_correct": joint_c,
            }
        },
        "probe": {
            "counts": {"n": 81, "type_correct": 60, "target_correct": 60, "joint_correct": 50},
            "regression_slices": {"lender_first": {"n": 9, "type_correct": 6}},
        },
    }


def v1_seed(*, joint=(90, 90, 80), probe_joint=50, lender_first_type=6):
    type_c, target_c, joint_c = joint
    return {
        "test-v1": {
            "counts": {
                "n": 105,
                "type_correct": type_c,
                "target_correct": target_c,
                "joint_correct": joint_c,
            }
        },
        "probe": {
            "counts": {
                "n": 81,
                "type_correct": 60,
                "target_correct": 60,
                "joint_correct": probe_joint,
            },
            "regression_slices": {"lender_first": {"n": 9, "type_correct": lender_first_type}},
        },
    }


def gate(v2=None, v1=None):
    v2 = v2 or {s: v2_seed() for s in SEEDS}
    v1 = v1 or {s: v1_seed() for s in SEEDS}
    return evaluate_gate(v2, v1, seeds=SEEDS)


def failed(result, group):
    return {f"{c['id']}/{c['scope']}" for c in result[group]["criteria"] if not c["pass"]}


def test_all_criteria_pass_is_verdict_a():
    result = gate()
    assert result["verdict"] == "A"
    assert not failed(result, "value_quality") and not failed(result, "regression")
    # five value criteria on the mean and on seed 1, four type/target checks on both scopes
    assert len(result["value_quality"]["criteria"]) == 10
    assert len(result["regression"]["criteria"]) == 9


@pytest.mark.parametrize(
    "kwargs, criterion",
    [
        ({"exact": 0.9499}, "test_value_exact"),
        ({"human": 0.8999}, "test_value_exact_human"),
        ({"present": 0.9699}, "test_value_present_accuracy"),
        ({"multi": 0.8999}, "test_multi_number_exact"),
    ],
)
def test_value_threshold_just_below_fails_and_gives_c(kwargs, criterion):
    result = gate(v2={s: v2_seed(**kwargs) for s in SEEDS})
    assert result["verdict"] == "C"
    assert failed(result, "value_quality") == {f"{criterion}/mean", f"{criterion}/seed1"}


def test_value_thresholds_are_inclusive():
    exactly = v2_seed(exact=0.95, human=0.90, present=0.97, multi=0.90)
    assert gate(v2={s: exactly for s in SEEDS})["verdict"] == "A"


def test_seed1_must_meet_thresholds_even_when_the_mean_does():
    v2 = {1: v2_seed(exact=0.94), 2: v2_seed(exact=0.99), 3: v2_seed(exact=0.99)}
    result = gate(v2=v2)
    assert result["verdict"] == "C"
    assert failed(result, "value_quality") == {"test_value_exact/seed1"}


def test_mean_fails_when_seed1_passes():
    v2 = {1: v2_seed(exact=0.96), 2: v2_seed(exact=0.92), 3: v2_seed(exact=0.92)}
    assert failed(gate(v2=v2), "value_quality") == {"test_value_exact/mean"}


def test_slice_floor_only_counts_slices_with_at_least_five_notes():
    small = v2_seed(slices={"slang": {"n": 4, "exact": 0.25}, "no_amount": {"n": 0, "exact": None}})
    assert gate(v2={s: small for s in SEEDS})["verdict"] == "A"

    five = v2_seed(slices={"slang": {"n": 5, "exact": 0.79}})
    result = gate(v2={s: five for s in SEEDS})
    assert result["verdict"] == "C"
    floor = next(c for c in result["value_quality"]["criteria"] if c["id"] == "test_slice_floor")
    assert floor["slices_below_threshold"] == ["slang"]
    assert floor["value"] == pytest.approx(0.79)

    edge = v2_seed(slices={"slang": {"n": 5, "exact": 0.80}})
    assert gate(v2={s: edge for s in SEEDS})["verdict"] == "A"


def test_missing_multi_number_data_fails_instead_of_passing_vacuously():
    empty = v2_seed(slices={"multi_number": {"n": 0, "exact": None}})
    result = gate(v2={s: empty for s in SEEDS})
    assert "test_multi_number_exact/mean" in failed(result, "value_quality")


def test_regression_with_good_value_quality_is_verdict_b():
    # joint -3 notes per seed: mean drop 3 > 2 and seed 1 drop 3 is allowed (not > 3)
    v2 = {s: v2_seed(joint=(90, 90, 77)) for s in SEEDS}
    result = gate(v2=v2)
    assert result["verdict"] == "B"
    assert failed(result, "regression") == {"test-v1_joint_mean/mean"}


def test_value_failure_dominates_regression():
    v2 = {s: v2_seed(exact=0.5, joint=(70, 70, 60)) for s in SEEDS}
    assert gate(v2=v2)["verdict"] == "C"


def test_mean_drop_of_exactly_two_notes_is_not_material_but_is_reported():
    v2 = {s: v2_seed(joint=(90, 90, 78)) for s in SEEDS}  # every seed loses 2 notes
    result = gate(v2=v2)
    crit = next(c for c in result["regression"]["criteria"] if c["id"] == "test-v1_joint_mean")
    assert result["verdict"] == "A" and crit["pass"] and crit["value"] == 2
    assert crit["rate_shorthand_flagged"] is True  # 2/105 = 0.01905 > 0.019
    assert result["regression"]["rate_shorthand_only"] == ["test-v1_joint_mean"]


def test_mean_drop_is_averaged_over_seeds():
    # seed drops 2, 2, 3 -> mean 7/3 > 2
    v2 = {
        1: v2_seed(joint=(90, 90, 78)),
        2: v2_seed(joint=(90, 90, 78)),
        3: v2_seed(joint=(90, 90, 77)),
    }
    assert failed(gate(v2=v2), "regression") == {"test-v1_joint_mean/mean"}


def test_probe_joint_mean_threshold_is_two_notes():
    v1 = {s: v1_seed(probe_joint=50) for s in SEEDS}

    def probe_v2(joint):
        out = v2_seed()
        out["probe"]["counts"]["joint_correct"] = joint
        return out

    assert gate(v2={s: probe_v2(48) for s in SEEDS}, v1=v1)["verdict"] == "A"
    assert failed(gate(v2={s: probe_v2(47) for s in SEEDS}, v1=v1), "regression") == {
        "probe_joint_mean/mean"
    }


def test_seed1_alone_regression_threshold_is_three_notes():
    base = {s: v2_seed() for s in SEEDS}
    three = copy.deepcopy(base)
    three[1]["test-v1"]["counts"]["type_correct"] = 87  # -3: allowed
    assert gate(v2=three)["verdict"] == "A"
    four = copy.deepcopy(base)
    four[1]["test-v1"]["counts"]["type_correct"] = 86  # -4: material (mean drop only 1.33)
    result = gate(v2=four)
    assert result["verdict"] == "B"
    assert failed(result, "regression") == {"test-v1_type_seed1/seed1"}


def test_improvements_never_count_as_regression():
    v2 = {s: v2_seed(joint=(99, 99, 99)) for s in SEEDS}
    assert gate(v2=v2)["verdict"] == "A"


def test_lender_first_probe_slice_loses_more_than_one_note_on_mean_type():
    def with_lender(correct):
        out = v2_seed()
        out["probe"]["regression_slices"]["lender_first"]["type_correct"] = correct
        return out

    assert gate(v2={s: with_lender(5) for s in SEEDS})["verdict"] == "A"  # -1 note exactly
    # seeds lose 1, 1, 2 notes: mean 4/3 > 1
    v2 = {1: with_lender(5), 2: with_lender(5), 3: with_lender(4)}
    result = gate(v2=v2)
    assert result["verdict"] == "B"
    assert failed(result, "regression") == {"probe_lender_first_type_mean/mean"}


def test_slices_with_different_n_across_seeds_are_an_error():
    v2 = {s: v2_seed() for s in SEEDS}
    v2[2]["test"]["value_slices"]["slang"]["n"] = 7
    with pytest.raises(ValueError, match="slang"):
        gate(v2=v2)


# --------------------------------------------------------------------------- slice membership


def test_value_slice_names_combine_categories_record_flags_and_span_shape():
    seen = {"50k"}
    row = {"categories": ["explicit_unit", "multi_number", "unaccented"], "gold_value_tokens": 1}
    rec = {"accented": False, "value": {"text": "2tr"}}
    assert value_slice_names(rec, row, seen) == [
        "explicit_unit",
        "multi_number",
        "unaccented",
        "unseen_span",
    ]
    # exact string match: the same amount in another case is unseen
    assert "unseen_span" in value_slice_names(
        {"accented": True, "value": {"text": "50K"}}, {"categories": []}, seen
    )
    assert "unseen_span" not in value_slice_names(
        {"accented": True, "value": {"text": "50k"}}, {"categories": []}, seen
    )


@pytest.mark.parametrize(
    "text, tokens, long",
    [("1tr5", 2, False), ("1tr5", 3, True), ("5 triệu", 1, True)],
)
def test_long_multi_token_is_more_than_two_tokens_or_a_space(text, tokens, long):
    names = value_slice_names(
        {"accented": True, "value": {"text": text}}, {"gold_value_tokens": tokens}, {text}
    )
    assert ("long_multi_token" in names) is long


def test_no_amount_slice_is_a_null_gold_value():
    names = value_slice_names({"accented": True, "value": None}, {"categories": ["none"]}, set())
    assert names == ["no_amount"]


# --------------------------------------------------------------------------- summaries / lists


def row(rid, *, type_ok=True, target_ok=True, value_ok=True, **extra):
    base = {
        "id": rid,
        "text": f"text {rid}",
        "type_ok": type_ok,
        "target_ok": target_ok,
        "value_ok": value_ok,
        "full_joint_ok": type_ok and target_ok and value_ok,
        "gold_value": {"text": "5", "start": 0, "end": 1},
        "pred_value": {"text": "5", "start": 0, "end": 1} if value_ok else None,
        "categories": [],
        "gold_value_tokens": 1,
    }
    return base | extra


def test_summarize_set_counts_notes_per_regression_slice_and_value_provenance():
    records = [
        {"text": "a", "accented": True, "value_status": "complete", "value_provenance": "human",
         "value": {"text": "5", "start": 0, "end": 1}},
        {"text": "b", "accented": False, "value_status": "complete", "value_provenance": "rule",
         "value": {"text": "5", "start": 0, "end": 1}},
        {"text": "c", "accented": True, "value_status": "uncertain", "value_provenance": "human",
         "value": {"text": "5", "start": 0, "end": 1}},
    ]  # fmt: skip
    rows = [row("a"), row("b", value_ok=False), row("c", type_ok=False, value_ok=False)]
    metrics = {
        "n": 3,
        "overall": {
            "type": {"accuracy": 2 / 3, "macro_f1": 0.5},
            "target": {"f1": 1.0, "exact_match": 1.0},
            "type_target_joint": 2 / 3,
        },
        "value": {"token": None},
    }
    flags = [{"lender_first"}, {"unaccented"}, {"lender_first", "unaccented"}]
    s = summarize_set(metrics, rows, records, with_value=True, seen_texts={"5"},
                      regression_flags=flags)  # fmt: skip
    assert s["counts"] == {"n": 3, "type_correct": 2, "target_correct": 3, "joint_correct": 2}
    assert s["regression_slices"]["lender_first"] == {
        "n": 2, "type_correct": 1, "target_correct": 2, "joint_correct": 1,
    }  # fmt: skip
    assert s["regression_slices"]["loan_installment"]["n"] == 0
    # the uncertain record is masked from every value metric
    assert s["value"]["n"] == 2 and s["value"]["exact"] == 0.5
    assert s["value"]["human"]["n"] == 1 and s["value"]["human"]["exact"] == 1.0
    assert s["value_slices"]["unaccented"]["n"] == 1
    assert s["value_slices_human"]["unaccented"]["n"] == 0
    assert s["value_slices"]["no_amount"]["exact"] is None


def test_multi_number_errors_lists_only_complete_wrong_multi_number_notes():
    rows = [
        row("ok", categories=["multi_number"], value_status="complete"),
        row("bad", categories=["multi_number"], value_status="complete", value_ok=False,
            gold_type="expense", pred_type="expense", value_confidence=0.9),
        row("other", categories=["explicit_unit"], value_status="complete", value_ok=False),
        row("masked", categories=["multi_number"], value_status="uncertain", value_ok=False),
    ]  # fmt: skip
    listed = multi_number_errors("test", 2, rows)
    assert [(e["id"], e["seed"], e["set"]) for e in listed] == [("bad", 2, "test")]
    assert listed[0]["gold_value"]["text"] == "5" and listed[0]["pred_value"] is None


def test_changes_vs_v1_classifies_regressions_improvements_and_moves():
    def pred(rid, ptype, ok_type, target=None, ok_target=True):
        return {
            "id": rid, "text": rid, "gold_type": "lend", "pred_type": ptype,
            "gold_target": None, "pred_target": target, "type_ok": ok_type,
            "target_ok": ok_target,
        }  # fmt: skip

    span = {"text": "x", "start": 0, "end": 1}
    v1 = [
        pred("same", "lend", True),
        pred("reg", "lend", True),
        pred("imp", "borrow", False),
        pred("moved", "borrow", False),
        pred("tgt", "lend", True, ok_target=True),
    ]
    v2 = [
        pred("same", "lend", True),
        pred("reg", "borrow", False),
        pred("imp", "lend", True),
        pred("moved", "income", False),
        pred("tgt", "lend", True, target=span, ok_target=False),
    ]
    changes = {c["id"]: c for c in changes_vs_v1(v1, v2)}
    assert set(changes) == {"reg", "imp", "moved", "tgt"}
    assert changes["reg"]["kind"] == "regression" and changes["reg"]["changed"] == ["type"]
    assert changes["imp"]["kind"] == "improvement"
    assert changes["moved"]["kind"] == "changed"
    assert changes["tgt"]["kind"] == "regression" and changes["tgt"]["changed"] == ["target"]


@pytest.mark.parametrize(
    ("text", "gold", "pred", "category"),
    [
        ("cho Nam 1 triệu 20/10", (8, 15), (8, 15), None),
        ("cho Nam 1 triệu 20/10", (8, 15), (8, 21), "right_overrun_date_time"),
        ("vay 5 củ, hẹn t10 trả", (4, 8), (4, 17), "right_overrun_date_time"),
        ("cho Lợi 1tr, hẹn", (8, 11), (8, 12), "right_overrun_punctuation"),
        ("gia han goi 4g 120k", (15, 19), (12, 19), "left_overrun"),
        ("gia han goi 4g 120k", (15, 19), (12, 14), "wrong_number_selection"),
        ("điện 540k", (5, 9), (8, 9), "boundary_truncation"),
        ("điện 540k", (5, 9), None, "missed_value"),
        ("ăn trưa", None, (0, 2), "spurious_value"),
    ],
)
def test_value_error_category(text, gold, pred, category):
    def span(s):
        return None if s is None else {"text": text[s[0] : s[1]], "start": s[0], "end": s[1]}

    assert vg.value_error_category(text, span(gold), span(pred)) == category
