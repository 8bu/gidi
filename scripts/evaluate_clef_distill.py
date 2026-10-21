#!/usr/bin/env python
"""Score clef-distill-v1 once: arms A / B seeds 1-3 + the run 4 / run 5 seed 1 controls.

Usage:
    uv run python scripts/evaluate_clef_distill.py

Every system is the INT8 encoder (type + target, ``snap_words=True``) + the unchanged rule value
parser ``gidi.value_parser.parse_value``. Sets: frozen test (105), probe-v1 (81 complete),
human-value-01 (147 complete, amended labels), human-value-02 (190 complete, LLM labels, a reused
test set). Reported per set: type accuracy and macro-F1, target exact and, on the two human sets,
value exact and end-to-end exact (type + target + value). The control rows are rescored and
asserted equal to the stored results (``annotation-v3-retrain-v5/results.json`` for run 4 / run 5
on test, probe-v1 and human-value-01; ``results-hv02.json`` on human-value-02) before anything is
written. The success criterion of ``protocol.json`` is evaluated verbatim. Refuses to overwrite
``results.json`` (the sets are scored once).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

import evaluate_encoder_retrain as base  # noqa: E402
import evaluate_encoder_retrain_v3 as v3e  # noqa: E402

from gidi.inference import GidiPredictor  # noqa: E402

EXP = ROOT / "experiments" / "clef-distill-v1"
PROTOCOL = EXP / "protocol.json"
DEFAULT_OUT = EXP / "results.json"
V5 = ROOT / "experiments" / "annotation-v3-retrain-v5"
V4_ONNX = ROOT / "experiments" / "annotation-v3-retrain-v4" / "onnx" / "seed1" / "model.int8.onnx"
V5_ONNX = V5 / "onnx" / "seed1" / "model.int8.onnx"
HV02 = ROOT / "datasets" / "annotation-v3" / "human-value-02"
SEEDS = (1, 2, 3)
SETS = ("test", "probe", "hv01", "hv02")
METRICS = ("type_accuracy", "type_macro_f1", "target_exact", "value_exact", "end_to_end_exact")
CRITERION = {"hv02_end_to_end_at_least": 0.8105, "max_regression_vs_run4": 0.02}


def block_of(records, types, spans, values, with_value: bool) -> dict[str, Any]:
    if with_value:
        raw = base.human_block(records, types, spans, values)
    else:
        raw = base.encoder_block(records, types, spans)
    return {k: raw.get(k) for k in ("n", *METRICS, "type_target_exact", "span_f1")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists: held-out scoring runs once")
    if not PROTOCOL.is_file():
        raise SystemExit(f"{PROTOCOL} must exist before scoring")

    base_sets, _ = base.load_sets(False)
    hv02, hv02_excluded = base.load_human(HV02 / "review-queue-all.jsonl", HV02 / "labels.jsonl")
    sets = {"test": base_sets["test"], "probe": base_sets["probe"], "hv01": base_sets["human"]}
    sets["hv02"] = hv02
    values = {name: base.value_spans(sets[name]) for name in ("hv01", "hv02")}

    models: dict[str, Path] = {
        "run4_seed1": V4_ONNX,
        "run5_seed1": V5_ONNX,
        **{
            f"{arm}_seed{s}": EXP / f"arm{arm}" / "onnx" / f"seed{s}" / "model.int8.onnx"
            for arm in ("A", "B")
            for s in SEEDS
        },
    }
    for name, path in models.items():
        if not path.is_file():
            raise SystemExit(f"missing {name}: {path}")

    systems: dict[str, Any] = {}
    preds: dict[str, dict[str, Any]] = {}
    for name, path in models.items():
        predictor = GidiPredictor.from_bundle(
            base.OLD_BUNDLE, model_path=path, intra_op_threads=1, snap_words=True
        )
        preds[name] = {s: base.predict_all(predictor, sets[s]) for s in SETS}
        systems[name] = {
            s: block_of(sets[s], *preds[name][s], values.get(s), s in values) for s in SETS
        }

    stored = json.loads((V5 / "results.json").read_text("utf-8"))["systems"]
    stored_hv02 = json.loads((V5 / "results-hv02.json").read_text("utf-8"))["systems"]
    for name, key, hv02_key in (
        ("run4_seed1", "v4_seed1_snap", "run4"),
        ("run5_seed1", "v5_seed1_snap", "candidate"),
    ):
        for s, ref in (
            ("test", stored[key]["test"]),
            ("probe", stored[key]["probe"]),
            ("hv01", stored[key]["human"]["all_complete"]),
            ("hv02", stored_hv02[hv02_key]),
        ):
            keys = v3e.KEYS_ENC if s in ("test", "probe") else v3e.KEYS_HUMAN
            v3e.assert_equal(f"{name} {s}", systems[name][s], ref, keys)

    def mean_sd(arm: str) -> dict[str, Any]:
        out = {}
        for s in SETS:
            out[s] = {}
            for m in METRICS:
                vals = [systems[f"{arm}_seed{k}"][s][m] for k in SEEDS]
                if vals[0] is None:
                    out[s][m] = None
                    continue
                out[s][m] = {"mean": statistics.fmean(vals), "sd": statistics.stdev(vals)}
        return out

    aggregates = {arm: mean_sd(arm) for arm in ("A", "B")}

    # success criterion, verbatim from protocol.json
    cand, ctrl = systems["A_seed1"], systems["run4_seed1"]
    hv02_e2e = cand["hv02"]["end_to_end_exact"]
    checks = {
        "hv02_end_to_end_exact": {
            "arm_A_seed1": hv02_e2e,
            "run4_seed1": ctrl["hv02"]["end_to_end_exact"],
            "need_at_least": CRITERION["hv02_end_to_end_at_least"],
            "pass": hv02_e2e >= CRITERION["hv02_end_to_end_at_least"] - 1e-9,
        }
    }
    for label, s, m in (
        ("frozen_test_type_macro_f1", "test", "type_macro_f1"),
        ("frozen_test_target_exact", "test", "target_exact"),
        ("probe_v1_type_macro_f1", "probe", "type_macro_f1"),
        ("human_value_01_end_to_end_exact", "hv01", "end_to_end_exact"),
    ):
        drop = ctrl[s][m] - cand[s][m]
        checks[label] = {
            "arm_A_seed1": cand[s][m],
            "run4_seed1": ctrl[s][m],
            "regression": drop,
            "max_allowed": CRITERION["max_regression_vs_run4"],
            "pass": drop <= CRITERION["max_regression_vs_run4"] + 1e-9,
        }
    verdict = "PASS" if all(c["pass"] for c in checks.values()) else "FAIL"

    # human-value-02: notes where arm A seed 1 differs from run 4 seed 1 (ids and labels only)
    def note_row(i: int, system: str) -> dict[str, Any]:
        types, spans = preds[system]["hv02"]
        return {
            "type": types[i],
            "target": None if spans[i] is None else [spans[i]["start"], spans[i]["end"]],
        }

    def e2e(i: int, system: str) -> bool:
        types, spans = preds[system]["hv02"]
        r, v = hv02[i], values["hv02"][i]
        return (
            types[i] == r["type"]
            and base.key(spans[i]) == base.key(r["target"])
            and base.key(v) == base.key(r["value"])
        )

    changed = []
    for i, r in enumerate(hv02):
        a, b = note_row(i, "A_seed1"), note_row(i, "run4_seed1")
        if a == b and e2e(i, "A_seed1") == e2e(i, "run4_seed1"):
            continue
        changed.append(
            {
                "id": r["id"],
                "gold": {
                    "type": r["type"],
                    "target": None
                    if r["target"] is None
                    else [r["target"]["start"], r["target"]["end"]],
                },
                "run4_seed1": b | {"end_to_end_ok": e2e(i, "run4_seed1")},
                "A_seed1": a | {"end_to_end_ok": e2e(i, "A_seed1")},
                "outcome": (
                    "fixed"
                    if e2e(i, "A_seed1") and not e2e(i, "run4_seed1")
                    else "broke"
                    if e2e(i, "run4_seed1") and not e2e(i, "A_seed1")
                    else "changed, same end-to-end result"
                ),
            }
        )

    results = {
        "experiment": "clef-distill-v1",
        "protocol_sha256": base.sha256_file(PROTOCOL),
        "release_candidate": "A_seed1",
        "sets": {s: len(r) for s, r in sets.items()},
        "hv02_not_complete_excluded": hv02_excluded,
        "models": {
            n: {"path": str(p.relative_to(ROOT)), "sha256": base.sha256_file(p)}
            for n, p in models.items()
        },
        "systems": systems,
        "aggregate_seeds": aggregates,
        "success_criterion": {"text": json.loads(PROTOCOL.read_text("utf-8"))["success_criterion"]},
        "criterion_checks": checks,
        "verdict": verdict,
        "hv02_changed_notes_arm_A_seed1_vs_run4_seed1": changed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(results, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")
    for name in models:
        row = " | ".join(
            f"{s} F1 {systems[name][s]['type_macro_f1']:.4f} e2e "
            f"{(systems[name][s]['end_to_end_exact'] or 0):.4f}"
            for s in SETS
        )
        print(f"{name:12s} {row}")
    print(f"verdict: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
