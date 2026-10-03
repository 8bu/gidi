import pytest

from gidi.evaluation.metrics import evaluate
from gidi.evaluation.summary import (
    aggregate_config,
    best_config_per_model,
    best_run,
    failure_cases,
    mean_spread,
    per_class_mean,
    selection_score,
    shared_failures,
    slice_means,
    summed_confusion,
)
from gidi.modeling.preprocessing import TYPES


def span(start, end, text="x"):
    return {"text": text, "start": start, "end": end}


RECORDS = [
    {"id": "a", "text": "an co", "type": "expense", "target": span(0, 2), "source_batch": "b1"},
    {"id": "b", "text": "an com", "type": "expense", "target": None, "source_batch": "b1"},
    {"id": "c", "text": "ăn cơ", "type": "income", "target": span(0, 2), "source_batch": "b2"},
    {"id": "d", "text": "luong", "type": "income", "target": None, "source_batch": "b2"},
]


def make_run(model, lr, seed, val, test=None, best_epoch=2, epochs_run=5):
    """``val``/``test`` are ``(type_preds, span_preds)`` scored on RECORDS with real evaluate."""
    test = test or val
    return {
        "model": model,
        "lr": lr,
        "seed": seed,
        "dir": f"runs/{model}/lr{lr:g}-seed{seed}",
        "metrics": {
            "validation": evaluate(RECORDS, *val),
            "test": evaluate(RECORDS, *test),
        },
        "checkpoint": {
            "best_epoch": best_epoch,
            "epochs_run": epochs_run,
            "wall_time_sec": 10.0 * seed,
            "sec_per_epoch": 1.0,
            "param_count": 7,
        },
    }


PERFECT = (["expense", "expense", "income", "income"], [span(0, 2), None, span(0, 2), None])
# one wrong type (b -> income), one wrong span (a -> [0,1]), one false span on null record d
BAD = (["expense", "income", "income", "income"], [span(0, 1), None, span(0, 2), span(0, 1)])
WORST = (["income", "income", "expense", "expense"], [None, span(0, 1), None, span(0, 1)])
# all types right, spans all null (gold spans missed, nulls right)
NO_SPANS = (["expense", "expense", "income", "income"], [None, None, None, None])


def test_mean_spread_sample_std_and_none_handling():
    s = mean_spread([1.0, 2.0, None, 4.0])
    assert s["n"] == 3
    assert s["mean"] == pytest.approx(7 / 3)
    assert s["std"] == pytest.approx(1.5275252316519465)  # sample (n-1) std
    assert (s["min"], s["max"]) == (1.0, 4.0)
    assert mean_spread([3.0])["std"] is None
    assert mean_spread([None])["n"] == 0


def test_selection_score_is_mean_of_type_macro_f1_and_span_f1():
    m = evaluate(RECORDS, *NO_SPANS)
    assert m["overall"]["type"]["macro_f1"] == 1.0 and m["overall"]["target"]["f1"] == 0.0
    assert selection_score(m) == 0.5
    assert selection_score(evaluate(RECORDS, *PERFECT)) == 1.0


def test_aggregate_config_headline_mean_std_and_cost():
    runs = [
        make_run("m", 2e-5, 2, PERFECT, best_epoch=3, epochs_run=6),
        make_run("m", 2e-5, 1, BAD, best_epoch=2, epochs_run=5),
    ]
    cfg = aggregate_config(runs)
    assert cfg["n_seeds"] == 2 and cfg["seeds"] == [1, 2]
    assert [p["best_epoch"] for p in cfg["per_seed"]] == [2, 3]  # sorted by seed
    acc = cfg["validation"]["type_accuracy"]
    assert acc["mean"] == pytest.approx((0.75 + 1.0) / 2)
    assert acc["std"] == pytest.approx(0.25 / 2**0.5)
    assert (acc["min"], acc["max"]) == (0.75, 1.0)
    # null accuracy: perfect 2/2, BAD 1/2 (false span on d)
    assert cfg["test"]["null_accuracy"]["mean"] == pytest.approx(0.75)
    assert cfg["test"]["false_span_rate"]["mean"] == pytest.approx(0.25)
    assert cfg["split_n"]["test"] == {"n": 4, "n_gold_target": 2, "n_gold_null": 2}
    assert cfg["wall_time_sec"]["mean"] == pytest.approx(15.0)
    assert cfg["param_count"] == 7


def test_best_run_and_best_config_selection_use_validation_only():
    # seed 1 is worse on validation but perfect on test: selection must not look at test
    r1 = make_run("m", 1e-5, 1, WORST, test=PERFECT)
    r2 = make_run("m", 1e-5, 2, NO_SPANS, test=BAD)
    r3 = make_run("m", 1e-5, 3, NO_SPANS, test=BAD)
    assert best_run([r1, r2, r3])["seed"] == 2  # tie between 2 and 3 -> lowest seed
    good = [make_run("m", 3e-5, s, PERFECT) for s in (1, 2)]
    weak = [make_run("m", 1e-5, s, NO_SPANS) for s in (1, 2)]
    other = [make_run("n", 1e-5, 1, BAD)]
    configs = [aggregate_config(weak), aggregate_config(good), aggregate_config(other)]
    best = best_config_per_model(configs)
    assert best["m"]["lr"] == 3e-5 and best["n"]["lr"] == 1e-5


def test_best_config_tie_prefers_lower_lr():
    configs = [aggregate_config([make_run("m", lr, 1, PERFECT)]) for lr in (5e-5, 2e-5)]
    assert best_config_per_model(configs)["m"]["lr"] == 2e-5


def test_per_class_mean_confusion_and_slices():
    runs = [make_run("m", 1e-5, 1, PERFECT), make_run("m", 1e-5, 2, BAD)]
    pc = per_class_mean(runs)
    assert list(pc) == list(TYPES)
    # expense: perfect P/R=1; BAD predicts b as income -> recall 1/2, precision 1
    assert pc["expense"]["recall"] == pytest.approx(0.75)
    assert pc["expense"]["support"] == 2 and pc["borrow"]["support"] == 0
    cm = summed_confusion(runs)
    e, i = TYPES.index("expense"), TYPES.index("income")
    assert cm[e][e] == 3 and cm[e][i] == 1 and cm[i][i] == 4 and cm[i][e] == 0
    assert sum(map(sum, cm)) == 8
    sl = slice_means(runs)
    assert sl["accented"]["true"]["n"] == 1 and sl["accented"]["false"]["n"] == 3
    assert sl["source_batch"]["b1"]["n"] == 2 and sl["source_batch"]["b2"]["n"] == 2
    # b1 holds a (span wrong in BAD) and b (type wrong in BAD)
    assert sl["source_batch"]["b1"]["type_accuracy"] == pytest.approx((1.0 + 0.5) / 2)
    assert sl["source_batch"]["b2"]["type_accuracy"] == 1.0
    assert sl["source_batch"]["b1"]["n_runs"] == 2


def _predictions(types, spans):
    return [
        {
            "id": r["id"],
            "text": r["text"],
            "gold_type": r["type"],
            "pred_type": t,
            "gold_span": r["target"],
            "pred_span": s,
        }
        for r, t, s in zip(RECORDS, types, spans, strict=True)
    ]


def test_failure_cases_and_shared_failures():
    bad = _predictions(*BAD)
    f = failure_cases(bad)
    assert [e["id"] for e in f["type_errors"]] == ["b"]
    assert f["type_errors"][0]["pred_type"] == "income"
    # a: wrong bounds; d: false span on gold-null. b (null == null) is correct.
    assert [e["id"] for e in f["span_errors"]] == ["a", "d"]
    assert f["span_errors"][1]["gold_span"] is None
    assert failure_cases(_predictions(*PERFECT))["span_errors"] == []

    other = _predictions(["expense", "income", "expense", "income"], [span(0, 1), None, None, None])
    shared = shared_failures({"m1": bad, "m2": other})
    assert [e["id"] for e in shared["type_errors"]] == ["b"]
    assert shared["type_errors"][0]["pred_type"] == {"m1": "income", "m2": "income"}
    assert [e["id"] for e in shared["span_errors"]] == ["a"]  # d is only wrong in m1
    assert shared["span_errors"][0]["pred_span"]["m2"] == span(0, 1)
    assert shared["n_common"] == 4 and shared["n_unmatched"] == 0

    shared = shared_failures({"m1": bad, "m2": other[:3]})
    assert shared["n_common"] == 3 and shared["n_unmatched"] == 1
