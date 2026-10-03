"""value-span-v1 adoption gate: per-set summaries, regression slices, and the A/B/C verdict.

Everything here is pure computation over the outputs of :func:`gidi.evaluation.value_eval.
evaluate_records`; ``scripts/compare_value_v1.py`` runs the models and writes the files. The
criteria and thresholds are those of ``experiments/value-span-v1/protocol.json`` (frozen); this
module adds none.

Set summary (``summarize_set``), one per (model, seed, set)::

    {"n", "type_accuracy", "type_macro_f1", "target_f1", "target_exact", "type_target_joint",
     "counts": {"n", "type_correct", "target_correct", "joint_correct"},         # notes
     "regression_slices": {name: {"n", "type_correct", "target_correct", "joint_correct"}},
     "value": {"n", "exact", "present_accuracy", "span_*", "token_f1", "full_joint",
               "human": {...same block...}},                  # v2 only
     "value_slices": {name: block}}                           # v2 only

Value metrics are over ``value_status == complete`` records. ``joint`` is type correct AND
target exact (the v1 ``type+span joint``); note counts are what the regression criteria use.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from typing import Any

from gidi.annotation.value_span import fold
from gidi.evaluation.value_metrics import value_span_metrics

# --- protocol value slices (value_slices) --------------------------------------------------------
VALUE_SLICES: tuple[str, ...] = (
    "explicit_unit",  # category of gidi.annotation.value_span.propose_value
    "bare_number",  # category
    "multi_number",  # category
    "slang",  # category (xi, chai, lit, cu, ty ...)
    "unaccented",  # record ``accented`` is False
    "unseen_span",  # gold value text (exact string) in no complete train value
    "long_multi_token",  # gold value covers > 2 tokens of the model tokenizer, or contains a space
    "no_amount",  # complete record with value null
)
# --- regression slices (regression_vs_v1.checks) -------------------------------------------------
REGRESSION_SLICES: tuple[str, ...] = (
    "lender_first",
    "title_name",
    "insurance",
    "loan_installment",
    "unaccented",
)
HEADLINE: tuple[str, ...] = (
    "type_accuracy",
    "type_macro_f1",
    "target_f1",
    "target_exact",
    "type_target_joint",
)
VALUE_HEADLINE: tuple[str, ...] = (
    "exact",
    "span_f1",
    "span_precision",
    "span_recall",
    "token_f1",
    "present_accuracy",
    "full_joint",
)

# --- thresholds, verbatim from protocol.json adoption_criteria / regression_vs_v1 ---------------
VALUE_EXACT_MIN = 0.95
HUMAN_EXACT_MIN = 0.90
PRESENT_ACCURACY_MIN = 0.97
MULTI_NUMBER_EXACT_MIN = 0.90
SLICE_MIN_N = 5
SLICE_EXACT_MIN = 0.80
# mean over seeds: test v1 subset (type accuracy / target exact / joint) and probe joint
MEAN_DROP_MAX_NOTES = 2
SEED1_DROP_MAX_NOTES = 3  # seed 1 alone, same four checks
LENDER_FIRST_DROP_MAX_NOTES = 1  # probe lender-first slice, mean type accuracy
# The protocol writes the mean-drop threshold as a rate next to the note count. 2 notes is
# 0.01905 of 105 but 0.02469 of 81, so the rates are rounded forms of "2 notes"; the note count
# decides, the rate form is reported beside it (``rate_shorthand_flagged``).
RATE_SHORTHAND = {"test-v1": 0.019, "probe": 0.025}
_EPS = 1e-9

Row = Mapping[str, Any]


def _ratio(num: float, den: float) -> float | None:
    return num / den if den else None


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def sample_std(values: Sequence[float]) -> float | None:
    """Sample standard deviation (ddof 1), as the compression-v3 report; ``None`` for one value."""
    return statistics.stdev(values) if len(values) > 1 else None


# --------------------------------------------------------------------------- value slices


def value_slice_names(record: Mapping[str, Any], row: Row, seen_texts: set[str]) -> list[str]:
    """Protocol value slices a ``complete`` record belongs to (a note can be in several).

    ``row`` is its ``evaluate_records`` row (``categories``, ``gold_value_tokens``).
    """
    cats = set(row.get("categories") or [])
    value = record.get("value")
    names = [c for c in ("explicit_unit", "bare_number", "multi_number", "slang") if c in cats]
    if not record.get("accented", True):
        names.append("unaccented")
    if value is None:
        names.append("no_amount")
    else:
        if value["text"] not in seen_texts:
            names.append("unseen_span")
        if int(row.get("gold_value_tokens", 0)) > 2 or " " in value["text"]:
            names.append("long_multi_token")
    return names


def seen_value_texts(train_records: Sequence[Mapping[str, Any]]) -> set[str]:
    """Exact value strings of the ``complete`` training records (the ``unseen_span`` reference)."""
    return {
        t["value"]["text"]
        for t in train_records
        if t.get("value_status") == "complete" and t.get("value") is not None
    }


def value_block(rows: Sequence[Row]) -> dict[str, Any]:
    """Value metrics of the (complete) ``rows``: exact, present/null accuracy, span P/R/F1."""
    n = len(rows)
    block: dict[str, Any] = {
        "n": n,
        "exact": None,
        "present_accuracy": None,
        "span_precision": None,
        "span_recall": None,
        "span_f1": None,
        "full_joint": None,
        "n_errors": 0,
    }
    if not n:
        return block
    metrics = value_span_metrics(
        [{"text": r["text"], "value": r["gold_value"]} for r in rows],
        [r["pred_value"] for r in rows],
    )
    block |= {
        "exact": metrics["exact_match"],
        "present_accuracy": metrics["present_accuracy"],
        "span_precision": metrics["span"]["precision"],
        "span_recall": metrics["span"]["recall"],
        "span_f1": metrics["span"]["f1"],
        "full_joint": _ratio(sum(bool(r["full_joint_ok"]) for r in rows), n),
        "n_errors": sum(not r["value_ok"] for r in rows),
    }
    return block


# --------------------------------------------------------------------------- set summaries


def _counts(rows: Sequence[Row]) -> dict[str, int]:
    return {
        "n": len(rows),
        "type_correct": sum(bool(r["type_ok"]) for r in rows),
        "target_correct": sum(bool(r["target_ok"]) for r in rows),
        "joint_correct": sum(bool(r["type_ok"] and r["target_ok"]) for r in rows),
    }


def summarize_set(
    metrics: Mapping[str, Any],
    rows: Sequence[Row],
    records: Sequence[Mapping[str, Any]],
    *,
    with_value: bool,
    seen_texts: set[str] | None = None,
    regression_flags: Sequence[set[str]] | None = None,
) -> dict[str, Any]:
    """Flat summary of one scored set (see the module docstring).

    ``metrics`` / ``rows`` are ``evaluate_records`` results for ``records``; ``regression_flags[i]``
    are the regression-slice names of record ``i`` (omit for sets the gate does not slice).
    """
    if len(rows) != len(records):
        raise ValueError(f"{len(records)} records but {len(rows)} rows")
    overall = metrics["overall"]
    out: dict[str, Any] = {
        "n": metrics["n"],
        "type_accuracy": overall["type"]["accuracy"],
        "type_macro_f1": overall["type"]["macro_f1"],
        "target_f1": overall["target"]["f1"],
        "target_exact": overall["target"]["exact_match"],
        "type_target_joint": overall["type_target_joint"],
        "counts": _counts(rows),
    }
    if regression_flags is not None:
        out["regression_slices"] = {
            name: _counts([r for r, f in zip(rows, regression_flags, strict=True) if name in f])
            for name in REGRESSION_SLICES
        }
    if with_value:
        if seen_texts is None:
            raise ValueError("value summaries need seen_texts (the training value strings)")
        complete = [
            (rec, row)
            for rec, row in zip(records, rows, strict=True)
            if rec.get("value_status") == "complete"
        ]
        token = metrics["value"].get("token")
        out["value"] = value_block([row for _, row in complete]) | {
            "token_f1": token["f1"] if token else None,
            "human": value_block(
                [row for rec, row in complete if rec.get("value_provenance") == "human"]
            ),
        }
        members = [value_slice_names(rec, row, seen_texts) for rec, row in complete]

        def slice_blocks(human_only: bool) -> dict[str, Any]:
            return {
                name: value_block(
                    [
                        row
                        for (rec, row), m in zip(complete, members, strict=True)
                        if name in m and (not human_only or rec.get("value_provenance") == "human")
                    ]
                )
                for name in VALUE_SLICES
            }

        out["value_slices"] = slice_blocks(False)
        out["value_slices_human"] = slice_blocks(True)
    return out


def headline_stats(per_seed: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """Per-seed values, mean and sample std of the headline metrics of one set, across seeds."""
    seeds = sorted(per_seed)

    def stat(values: list[float | None]) -> dict[str, Any]:
        present = [v for v in values if v is not None]
        complete = len(present) == len(values)
        return {
            "per_seed": dict(zip(seeds, values, strict=True)),
            "mean": mean(present) if present and complete else None,
            "std": sample_std(present) if complete else None,
        }

    out: dict[str, Any] = {
        "n": per_seed[seeds[0]]["n"],
        **{k: stat([per_seed[s][k] for s in seeds]) for k in HEADLINE},
    }
    if "value" in per_seed[seeds[0]]:
        out["value"] = {
            "n": per_seed[seeds[0]]["value"]["n"],
            **{k: stat([per_seed[s]["value"][k] for s in seeds]) for k in VALUE_HEADLINE},
            "human_n": per_seed[seeds[0]]["value"]["human"]["n"],
            "human_exact": stat([per_seed[s]["value"]["human"]["exact"] for s in seeds]),
        }
    return out


def value_slice_table(
    per_seed: Mapping[int, Mapping[str, Any]], key: str = "value_slices"
) -> dict[str, Any]:
    """Per value slice: ``n`` and value exact per seed, mean, sample std (one set, v2 seeds)."""
    seeds = sorted(per_seed)
    table = {}
    for name in VALUE_SLICES:
        exact = [per_seed[s][key][name]["exact"] for s in seeds]
        have = [e for e in exact if e is not None]
        table[name] = {
            "n": per_seed[seeds[0]][key][name]["n"],
            "exact_per_seed": dict(zip(seeds, exact, strict=True)),
            "exact_mean": mean(have) if have and len(have) == len(exact) else None,
            "exact_std": sample_std(have) if len(have) == len(exact) else None,
        }
    return table


def regression_table(
    v2: Mapping[int, Mapping[str, Any]],
    v1: Mapping[int, Mapping[str, Any]],
    seeds: Sequence[int],
    set_name: str,
) -> dict[str, Any]:
    """Note counts (type / target / joint correct) per regression slice, v1 vs v2, per seed.

    ``all`` is the whole set. ``delta_mean`` is mean(v2) - mean(v1) in notes (negative = loss).
    """
    names = ("all", *REGRESSION_SLICES)
    table: dict[str, Any] = {}
    for name in names:

        def counts(summary: Mapping[str, Any], name: str = name) -> Mapping[str, int]:
            return summary["counts"] if name == "all" else summary["regression_slices"][name]

        entry: dict[str, Any] = {"n": counts(v2[seeds[0]][set_name])["n"]}
        for key in ("type_correct", "target_correct", "joint_correct"):
            a = {s: counts(v1[s][set_name])[key] for s in seeds}
            b = {s: counts(v2[s][set_name])[key] for s in seeds}
            entry[key] = {
                "v1_per_seed": a,
                "v2_per_seed": b,
                "v1_mean": mean(list(a.values())),
                "v2_mean": mean(list(b.values())),
                "delta_mean": mean(list(b.values())) - mean(list(a.values())),
            }
        table[name] = entry
    return table


# --------------------------------------------------------------------------- gate


def _criterion(
    cid: str,
    scope: str,
    metric: str,
    value: float | None,
    comparison: str,
    threshold: float,
    passed: bool,
    **detail: Any,
) -> dict[str, Any]:
    return {
        "id": cid,
        "scope": scope,
        "metric": metric,
        "value": value,
        "comparison": comparison,
        "threshold": threshold,
        "pass": bool(passed),
        **detail,
    }


def _at_least(value: float | None, threshold: float) -> bool:
    return value is not None and value >= threshold - _EPS


def _scope_mean(values: Sequence[float | None]) -> float | None:
    return None if any(v is None for v in values) else mean(values)  # type: ignore[arg-type]


def value_quality_criteria(
    v2: Mapping[int, Mapping[str, Any]], seeds: Sequence[int], scope: str
) -> list[dict[str, Any]]:
    """The five ``value_quality`` criteria over ``seeds`` (all seeds: ``mean``; one: ``seed``).

    ``v2[seed]["test"]`` is the full 143-note test summary.
    """
    tests = [v2[s]["test"] for s in seeds]
    out: list[dict[str, Any]] = []

    def block_mean(key: str, *path: str) -> float | None:
        vals = []
        for t in tests:
            v: Any = t["value"]
            for p in (*path, key):
                v = v[p]
            vals.append(v)
        return _scope_mean(vals)

    exact = block_mean("exact")
    out.append(
        _criterion(
            "test_value_exact",
            scope,
            "test value exact (complete labels)",
            exact,
            ">=",
            VALUE_EXACT_MIN,
            _at_least(exact, VALUE_EXACT_MIN),
            n=tests[0]["value"]["n"],
        )
    )
    human = block_mean("exact", "human")
    out.append(
        _criterion(
            "test_value_exact_human",
            scope,
            "test value exact (human-provenance labels)",
            human,
            ">=",
            HUMAN_EXACT_MIN,
            _at_least(human, HUMAN_EXACT_MIN),
            n=tests[0]["value"]["human"]["n"],
        )
    )
    present = block_mean("present_accuracy")
    out.append(
        _criterion(
            "test_value_present_accuracy",
            scope,
            "test value present/null accuracy",
            present,
            ">=",
            PRESENT_ACCURACY_MIN,
            _at_least(present, PRESENT_ACCURACY_MIN),
            n=tests[0]["value"]["n"],
        )
    )
    multi = _scope_mean([t["value_slices"]["multi_number"]["exact"] for t in tests])
    out.append(
        _criterion(
            "test_multi_number_exact",
            scope,
            "multi_number test slice value exact",
            multi,
            ">=",
            MULTI_NUMBER_EXACT_MIN,
            _at_least(multi, MULTI_NUMBER_EXACT_MIN),
            n=tests[0]["value_slices"]["multi_number"]["n"],
        )
    )
    slices: dict[str, dict[str, Any]] = {}
    for name in VALUE_SLICES:
        ns = {t["value_slices"][name]["n"] for t in tests}
        if len(ns) != 1:
            raise ValueError(f"slice {name!r} has different n across seeds: {sorted(ns)}")
        slices[name] = {
            "n": ns.pop(),
            "exact": _scope_mean([t["value_slices"][name]["exact"] for t in tests]),
        }
    eligible = {k: v for k, v in slices.items() if v["n"] >= SLICE_MIN_N}
    below = sorted(k for k, v in eligible.items() if v["exact"] < SLICE_EXACT_MIN - _EPS)
    out.append(
        _criterion(
            "test_slice_floor",
            scope,
            f"lowest value exact over test value slices with n >= {SLICE_MIN_N}",
            min((v["exact"] for v in eligible.values()), default=None),
            ">=",
            SLICE_EXACT_MIN,
            not below,
            slices=slices,
            slices_below_threshold=below,
        )
    )
    return out


def _drop_notes(
    v1: Mapping[int, Mapping[str, Any]],
    v2: Mapping[int, Mapping[str, Any]],
    seeds: Sequence[int],
    getter: Callable[[Mapping[str, Any]], int] | str,
) -> tuple[Fraction, list[int], list[int]]:
    """Mean over ``seeds`` of ``v1 - v2`` note counts, and the per-seed counts."""
    get = (lambda summary: summary["counts"][getter]) if isinstance(getter, str) else getter
    a = [get(v1[s]) for s in seeds]
    b = [get(v2[s]) for s in seeds]
    return Fraction(sum(a) - sum(b), len(seeds)), a, b


def regression_criteria(
    v2: Mapping[int, Mapping[str, Any]],
    v1: Mapping[int, Mapping[str, Any]],
    seeds: Sequence[int],
    export_seed: int,
) -> list[dict[str, Any]]:
    """The ``material_regression`` checks; ``pass`` means *no* material regression on that check.

    ``v1[seed]`` / ``v2[seed]`` map set name (``test-v1``, ``probe``) to its summary.
    """
    checks = (
        ("test-v1", "type_correct", "type accuracy"),
        ("test-v1", "target_correct", "target exact"),
        ("test-v1", "joint_correct", "type+target joint"),
        ("probe", "joint_correct", "type+target joint"),
    )
    out: list[dict[str, Any]] = []
    for set_name, key, label in checks:
        n = v2[seeds[0]][set_name]["counts"]["n"]
        v1_sets = {s: v1[s][set_name] for s in seeds}
        v2_sets = {s: v2[s][set_name] for s in seeds}
        for scope, scope_seeds, limit in (
            ("mean", list(seeds), MEAN_DROP_MAX_NOTES),
            (f"seed{export_seed}", [export_seed], SEED1_DROP_MAX_NOTES),
        ):
            drop, a, b = _drop_notes(v1_sets, v2_sets, scope_seeds, key)
            extra: dict[str, Any] = {}
            if scope == "mean":
                rate = float(drop) / n
                extra["rate_shorthand"] = RATE_SHORTHAND[set_name]
                extra["rate_shorthand_flagged"] = rate > RATE_SHORTHAND[set_name]
                extra["drop_rate"] = rate
            out.append(
                _criterion(
                    f"{set_name}_{key.removesuffix('_correct')}_{scope}",
                    scope,
                    f"{label} drop vs v1 on {set_name} (notes of {n})",
                    float(drop),
                    "<=",
                    limit,
                    drop <= limit,
                    set=set_name,
                    n=n,
                    unit="notes",
                    v1_correct_per_seed=dict(zip(scope_seeds, a, strict=True)),
                    v2_correct_per_seed=dict(zip(scope_seeds, b, strict=True)),
                    **extra,
                )
            )

    def lender(summary: Mapping[str, Any]) -> int:
        return summary["regression_slices"]["lender_first"]["type_correct"]

    v1_probe = {s: v1[s]["probe"] for s in seeds}
    v2_probe = {s: v2[s]["probe"] for s in seeds}
    drop, a, b = _drop_notes(v1_probe, v2_probe, list(seeds), lender)
    out.append(
        _criterion(
            "probe_lender_first_type_mean",
            "mean",
            "probe lender-first slice type accuracy loss vs v1 (notes)",
            float(drop),
            "<=",
            LENDER_FIRST_DROP_MAX_NOTES,
            drop <= LENDER_FIRST_DROP_MAX_NOTES,
            set="probe",
            n=v2[seeds[0]]["probe"]["regression_slices"]["lender_first"]["n"],
            unit="notes",
            v1_correct_per_seed=dict(zip(seeds, a, strict=True)),
            v2_correct_per_seed=dict(zip(seeds, b, strict=True)),
        )
    )
    return out


def evaluate_gate(
    v2: Mapping[int, Mapping[str, Any]],
    v1: Mapping[int, Mapping[str, Any]],
    *,
    seeds: Sequence[int] = (1, 2, 3),
    export_seed: int = 1,
) -> dict[str, Any]:
    """Adoption verdict (A/B/C) exactly as ``protocol.json`` defines it.

    ``v2[seed][set]`` and ``v1[seed][set]`` are :func:`summarize_set` results; needed sets:
    ``v2``: ``test``, ``test-v1``, ``probe``; ``v1``: ``test-v1``, ``probe``.
    """
    seeds = list(seeds)
    quality = [
        *value_quality_criteria(v2, seeds, "mean"),
        *value_quality_criteria(v2, [export_seed], f"seed{export_seed}"),
    ]
    regression = regression_criteria(v2, v1, seeds, export_seed)
    quality_ok = all(c["pass"] for c in quality)
    regression_ok = all(c["pass"] for c in regression)
    verdict = "C" if not quality_ok else ("A" if regression_ok else "B")
    meaning = {
        "A": "value_quality and regression both pass: adopt gidi-finance-v2 with value-span head",
        "B": "value_quality passes, regression fails: value-span works but type/target "
        "regressions need fixing",
        "C": "value_quality fails: value-span quality is not sufficient yet",
    }
    return {
        "verdict": verdict,
        "meaning": meaning[verdict],
        "seeds": seeds,
        "export_seed": export_seed,
        "value_quality": {
            "pass": quality_ok,
            "failed": [f"{c['id']}[{c['scope']}]" for c in quality if not c["pass"]],
            "criteria": quality,
        },
        "regression": {
            "material_regression": not regression_ok,
            "flagged": [f"{c['id']}" for c in regression if not c["pass"]],
            "rate_shorthand_only": [
                c["id"] for c in regression if c["pass"] and c.get("rate_shorthand_flagged")
            ],
            "criteria": regression,
        },
    }


# --------------------------------------------------------------------------- inspection lists


def multi_number_errors(set_name: str, seed: int, rows: Sequence[Row]) -> list[dict[str, Any]]:
    """Value errors on ``complete`` notes tagged ``multi_number``: text, gold, predicted."""
    return [
        {
            "seed": seed,
            "set": set_name,
            "id": r["id"],
            "text": r["text"],
            "gold_value": r["gold_value"],
            "pred_value": r["pred_value"],
            "value_confidence": r["value_confidence"],
            "value_provenance": r.get("value_provenance"),
            "gold_type": r["gold_type"],
            "pred_type": r["pred_type"],
            "categories": r.get("categories"),
        }
        for r in rows
        if r.get("value_status") == "complete"
        and "multi_number" in (r.get("categories") or [])
        and not r["value_ok"]
    ]


VALUE_ERROR_CATEGORIES = (
    "wrong_number_selection",
    "boundary_truncation",
    "left_overrun",
    "right_overrun_date_time",
    "right_overrun_punctuation",
    "right_overrun_other",
    "partial_overlap",
    "missed_value",
    "spurious_value",
)
# Date/time markers in the text a right overrun absorbs (accent-folded, lowercase):
# ``20/10``, ``t10``, ``10h``, ``thang``, ``hom``, ``ngay``, ``tuan``, ``mai``, ``toi``, ``sang``.
_DATE_TIME = re.compile(
    r"\d{1,2}/\d{1,2}|\bt\d{1,2}\b|\b\d{1,2}h\d{0,2}\b|"
    r"\b(thang|hom|ngay|tuan|mai|toi|sang|chieu|nay|qua|sau)\b"
)


def value_error_category(
    text: str, gold: Mapping[str, Any] | None, pred: Mapping[str, Any] | None
) -> str | None:
    """Position-based category of a wrong value prediction (``None`` when it is exact).

    Right overruns absorbing a date/time marker are ``right_overrun_date_time``; those adding
    only punctuation/whitespace are ``right_overrun_punctuation``. An overrun on both sides is
    ``left_overrun``.
    """
    if _span_key(gold) == _span_key(pred):
        return None
    if pred is None:
        return "missed_value"
    if gold is None:
        return "spurious_value"
    gs, ge, ps, pe = gold["start"], gold["end"], pred["start"], pred["end"]
    if pe <= gs or ps >= ge:
        return "wrong_number_selection"
    if ps >= gs and pe <= ge:
        return "boundary_truncation"
    if ps <= gs and pe >= ge:
        if ps < gs:
            return "left_overrun"
        extra = text[ge:pe]
        if not any(c.isalnum() for c in extra):
            return "right_overrun_punctuation"
        if _DATE_TIME.search(fold(extra).lower()):
            return "right_overrun_date_time"
        return "right_overrun_other"
    return "partial_overlap"


def value_error_table(rows_by_seed: Mapping[int, Mapping[str, Sequence[Row]]], sets: Sequence[str]):
    """``{set: {category: {seed: n}}}`` plus every error row, over ``complete`` labels."""
    counts: dict[str, dict[str, dict[int, int]]] = {
        s: {c: dict.fromkeys(rows_by_seed, 0) for c in VALUE_ERROR_CATEGORIES} for s in sets
    }
    errors = []
    for seed, by_set in rows_by_seed.items():
        for name in sets:
            for r in by_set[name]:
                if r.get("value_status") != "complete":
                    continue
                cat = value_error_category(r["text"], r["gold_value"], r["pred_value"])
                if cat is None:
                    continue
                counts[name][cat][seed] += 1
                errors.append(
                    {
                        "seed": seed,
                        "set": name,
                        "id": r["id"],
                        "text": r["text"],
                        "category": cat,
                        "gold_value": r["gold_value"],
                        "pred_value": r["pred_value"],
                        "multi_number": "multi_number" in (r.get("categories") or []),
                    }
                )
    return counts, errors


def _span_key(span: Mapping[str, Any] | None) -> tuple[int, int] | None:
    return None if span is None else (span["start"], span["end"])


def changes_vs_v1(v1_rows: Sequence[Row], v2_rows: Sequence[Row]) -> list[dict[str, Any]]:
    """Notes whose predicted type or target differs between v1 and v2 (same records, same order).

    ``kind`` is ``regression`` (v1 right, v2 wrong on type, target or both), ``improvement``
    (v1 wrong, v2 right), or ``changed`` (the prediction moved but correctness did not).
    """
    if len(v1_rows) != len(v2_rows):
        raise ValueError("v1 and v2 rows differ in length")
    out = []
    for a, b in zip(v1_rows, v2_rows, strict=True):
        if a["id"] != b["id"]:
            raise ValueError(f"row order differs: {a['id']} vs {b['id']}")
        type_changed = a["pred_type"] != b["pred_type"]
        target_changed = _span_key(a["pred_target"]) != _span_key(b["pred_target"])
        if not (type_changed or target_changed):
            continue
        changed = [
            name for name, did in (("type", type_changed), ("target", target_changed)) if did
        ]
        a_ok, b_ok = a["type_ok"] and a["target_ok"], b["type_ok"] and b["target_ok"]
        kind = "changed"
        if a_ok and not b_ok:
            kind = "regression"
        elif b_ok and not a_ok:
            kind = "improvement"
        out.append(
            {
                "id": a["id"],
                "text": a["text"],
                "kind": kind,
                "changed": changed,
                "gold_type": a["gold_type"],
                "v1_type": a["pred_type"],
                "v2_type": b["pred_type"],
                "gold_target": a["gold_target"],
                "v1_target": a["pred_target"],
                "v2_target": b["pred_target"],
                "v1_type_ok": a["type_ok"],
                "v2_type_ok": b["type_ok"],
                "v1_target_ok": a["target_ok"],
                "v2_target_ok": b["target_ok"],
            }
        )
    return out
