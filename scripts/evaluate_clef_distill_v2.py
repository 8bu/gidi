#!/usr/bin/env python
"""Score clef-distill-v2 once: D1 / D2 / D3 seeds 1-3 + the v3 and clef-distill-v1 arm A controls.

Usage:
    uv run python scripts/evaluate_clef_distill_v2.py          # D1-D3  -> results.json
    uv run python scripts/evaluate_clef_distill_v2.py --d4     # D4     -> results-d4.json

Every system is the INT8 encoder (type + target, ``snap_words=True``) + the unchanged rule value
parser ``gidi.value_parser.parse_value``. Sets: frozen test (105), probe-v1 (81 complete),
human-value-01 (147 complete, amended labels), human-value-02 (190 complete, LLM labels) and
probe-rules-v1 (64 LLM-labelled notes; no value labels, so type / target only plus a per-rule
R1..R4 breakdown). The 17 field-case notes (dev only, reported, never gated) are predicted per
system. The control rows are rescored and asserted equal to their stored results before
anything is written (``clef-distill-v1/results.json`` for arm A seed 1;
``annotation-v3-retrain-v5/results*.json`` for the v3 model = run 4 seed 1) on the four old
sets. The success criterion of ``protocol.json`` is evaluated verbatim. Refuses to overwrite
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

EXP = ROOT / "experiments" / "clef-distill-v2"
V1 = ROOT / "experiments" / "clef-distill-v1"
PROTOCOL = EXP / "protocol.json"
V5 = ROOT / "experiments" / "annotation-v3-retrain-v5"
V3_ONNX = ROOT / "models" / "gidi-finance-v3" / "model.int8.onnx"
A_ONNX = V1 / "armA" / "onnx" / "seed1" / "model.int8.onnx"
HV02 = ROOT / "datasets" / "annotation-v3" / "human-value-02"
PROBE_RULES = ROOT / "datasets" / "probe-rules-v1"
SEEDS = (1, 2, 3)
ARMS = ("D1", "D2", "D3")
SETS = ("test", "probe", "hv01", "hv02", "rules")
METRICS = ("type_accuracy", "type_macro_f1", "target_exact", "value_exact", "end_to_end_exact")
MAX_DROP = 0.02
MIN_GAIN = 0.10
FIELD_CASES = [  # (note, expected type, expected target) -- experiments/v3-field-issues.md
    ("Chuyển cho họ hàng 2 củ", "expense", None),
    ("trả giùm tiền cơm Khoa 50k", "lend", "Khoa"),
    ("moi nhan luong 26cu", "income", None),
    ("moi lanh luong 26tr", "income", None),
    ("long tra 5 lit", "repayment_in", "long"),
    ("Long tra 5 lit", "repayment_in", "Long"),
    ("chị Linh trả 2tr", "repayment_in", "Linh"),
    ("cong ty tra tien t10 5 lit", "income", None),
    ("khach tra tien 500k", "income", "khach"),
    ("chuyển sang ví momo 1tr", "transfer", None),
    ("chuyển cho mẹ 2 củ", "expense", "mẹ"),
    ("mới nhận lương 26cu", "income", None),
    ("lãnh lương 18tr", "income", None),
    ("ứng lương 3tr", "borrow", None),
    ("ứng giùm Lan tiền điện 400k", "lend", "Lan"),
    ("mua hộ Tú 200k", "lend", "Tú"),
    ("Khoa trả lại tiền mình trả giùm 50k", "repayment_in", "Khoa"),
]


def block_of(records, types, spans, values, with_value: bool) -> dict[str, Any]:
    raw = (
        base.human_block(records, types, spans, values)
        if with_value
        else base.encoder_block(records, types, spans)
    )
    return {k: raw.get(k) for k in ("n", *METRICS, "type_target_exact", "span_f1")}


def rules_block(records, types, spans) -> dict[str, Any]:
    out = block_of(records, types, spans, None, False)
    ok = [
        t == r["type"] and base.key(s) == base.key(r["target"])
        for t, s, r in zip(types, spans, records, strict=True)
    ]
    out["type_target_exact"] = sum(ok) / len(ok)
    out["per_rule"] = {}
    for rule in ("R1", "R2", "R3", "R4"):
        idx = [i for i, r in enumerate(records) if r["rule"] == rule]
        out["per_rule"][rule] = {
            "n": len(idx),
            "type_accuracy": sum(types[i] == records[i]["type"] for i in idx) / len(idx),
            "target_exact": sum(base.key(spans[i]) == base.key(records[i]["target"]) for i in idx)
            / len(idx),
            "type_target_exact": sum(ok[i] for i in idx) / len(idx),
        }
    return out


def load_rules() -> list[dict[str, Any]]:
    text_of = {r["id"]: r["text"] for r in base.read_jsonl(PROBE_RULES / "queue.jsonl")}
    rule_of = {r["id"]: r["rule"] for r in base.read_jsonl(PROBE_RULES / "provenance.jsonl")}
    return [
        {
            "id": lab["id"],
            "text": text_of[lab["id"]],
            "type": lab["type"],
            "target": lab["target"],
            "rule": rule_of[lab["id"]],
        }
        for lab in base.read_jsonl(PROBE_RULES / "labels.jsonl")
        if lab["annotation_status"] == "complete"
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--d4", action="store_true", help="score D4 (protocol.json: d4) instead")
    args = parser.parse_args(argv)
    arms = ("D4",) if args.d4 else ARMS
    cand_name = "D4_seed1" if args.d4 else "D3_seed1"
    if args.out is None:
        args.out = EXP / ("results-d4.json" if args.d4 else "results.json")
    if args.out.exists():
        raise SystemExit(f"{args.out} exists: held-out scoring runs once")
    if not PROTOCOL.is_file():
        raise SystemExit(f"{PROTOCOL} must exist before scoring")

    base_sets, _ = base.load_sets(False)
    hv02, hv02_excluded = base.load_human(HV02 / "review-queue-all.jsonl", HV02 / "labels.jsonl")
    sets = {"test": base_sets["test"], "probe": base_sets["probe"], "hv01": base_sets["human"]}
    sets["hv02"] = hv02
    sets["rules"] = load_rules()
    values = {name: base.value_spans(sets[name]) for name in ("hv01", "hv02")}

    models: dict[str, Path] = {
        "v3": V3_ONNX,
        "A_seed1": A_ONNX,
        **{
            f"{arm}_seed{s}": EXP / arm.lower() / "onnx" / f"seed{s}" / "model.int8.onnx"
            for arm in arms
            for s in SEEDS
        },
    }
    for name, path in models.items():
        if not path.is_file():
            raise SystemExit(f"missing {name}: {path}")

    systems: dict[str, Any] = {}
    preds: dict[str, dict[str, Any]] = {}
    field: dict[str, Any] = {}
    for name, path in models.items():
        predictor = GidiPredictor.from_bundle(
            base.OLD_BUNDLE, model_path=path, intra_op_threads=1, snap_words=True
        )
        preds[name] = {s: base.predict_all(predictor, sets[s]) for s in SETS}
        systems[name] = {
            s: (
                rules_block(sets[s], *preds[name][s])
                if s == "rules"
                else block_of(sets[s], *preds[name][s], values.get(s), s in values)
            )
            for s in SETS
        }
        rows = []
        for note, et, eg in FIELD_CASES:
            o = predictor.predict(note)
            rows.append(
                {
                    "note": note,
                    "expected": [et, eg],
                    "type": o.type,
                    "target": o.target or None,
                    "ok": o.type == et and (o.target or None) == (eg if eg else None),
                }
            )
        field[name] = {"correct": sum(r["ok"] for r in rows), "n": len(rows), "rows": rows}

    stored = json.loads((V5 / "results.json").read_text("utf-8"))["systems"]
    stored_hv02 = json.loads((V5 / "results-hv02.json").read_text("utf-8"))["systems"]
    stored_v1 = json.loads((V1 / "results.json").read_text("utf-8"))["systems"]
    for s, ref in (
        ("test", stored["v4_seed1_snap"]["test"]),
        ("probe", stored["v4_seed1_snap"]["probe"]),
        ("hv01", stored["v4_seed1_snap"]["human"]["all_complete"]),
        ("hv02", stored_hv02["run4"]),
    ):
        keys = v3e.KEYS_ENC if s in ("test", "probe") else v3e.KEYS_HUMAN
        v3e.assert_equal(f"v3 {s}", systems["v3"][s], ref, keys)
    for s in ("test", "probe", "hv01", "hv02"):
        keys = v3e.KEYS_ENC if s in ("test", "probe") else v3e.KEYS_HUMAN
        v3e.assert_equal(f"A_seed1 {s}", systems["A_seed1"][s], stored_v1["A_seed1"][s], keys)

    aggregates = {}
    for arm in arms:
        aggregates[arm] = {}
        for s in SETS:
            aggregates[arm][s] = {}
            for m in (*METRICS, "type_target_exact"):
                vals = [systems[f"{arm}_seed{k}"][s][m] for k in SEEDS]
                aggregates[arm][s][m] = (
                    None
                    if vals[0] is None
                    else {"mean": statistics.fmean(vals), "sd": statistics.stdev(vals)}
                )

    # success criterion, verbatim from protocol.json
    protocol = json.loads(PROTOCOL.read_text("utf-8"))
    criterion_text = (protocol["d4"] if args.d4 else protocol)["success_criterion"]
    cand, ctrl_a, ctrl_v3 = systems[cand_name], systems["A_seed1"], systems["v3"]
    c_rules = cand["rules"]["type_target_exact"]
    checks: dict[str, Any] = {
        "probe_rules_v1_type_target_exact_gain_vs_arm_A_seed1": {
            cand_name: c_rules,
            "arm_A_seed1": ctrl_a["rules"]["type_target_exact"],
            "gain": c_rules - ctrl_a["rules"]["type_target_exact"],
            "need_at_least": MIN_GAIN,
            "pass": c_rules - ctrl_a["rules"]["type_target_exact"] >= MIN_GAIN - 1e-9,
        },
        "probe_rules_v1_type_target_exact_gain_vs_v3": {
            cand_name: c_rules,
            "v3": ctrl_v3["rules"]["type_target_exact"],
            "gain": c_rules - ctrl_v3["rules"]["type_target_exact"],
            "need_at_least": MIN_GAIN,
            "pass": c_rules - ctrl_v3["rules"]["type_target_exact"] >= MIN_GAIN - 1e-9,
        },
    }
    for label, s, m in (
        ("frozen_test_type_macro_f1", "test", "type_macro_f1"),
        ("frozen_test_target_exact", "test", "target_exact"),
        ("probe_v1_type_macro_f1", "probe", "type_macro_f1"),
        ("human_value_01_end_to_end_exact", "hv01", "end_to_end_exact"),
        ("human_value_02_end_to_end_exact", "hv02", "end_to_end_exact"),
    ):
        drop = ctrl_a[s][m] - cand[s][m]
        checks[label] = {
            cand_name: cand[s][m],
            "arm_A_seed1": ctrl_a[s][m],
            "drop": drop,
            "max_allowed": MAX_DROP,
            "pass": drop <= MAX_DROP + 1e-9,
        }
    verdict = "PASS" if all(c["pass"] for c in checks.values()) else "FAIL"

    results = {
        "experiment": "clef-distill-v2",
        "protocol_sha256": base.sha256_file(PROTOCOL),
        "release_candidate": cand_name,
        "sets": {s: len(r) for s, r in sets.items()},
        "hv02_not_complete_excluded": hv02_excluded,
        "models": {
            n: {"path": str(p.relative_to(ROOT)), "sha256": base.sha256_file(p)}
            for n, p in models.items()
        },
        "systems": systems,
        "aggregate_seeds": aggregates,
        "success_criterion": {"text": criterion_text},
        "criterion_checks": checks,
        "verdict": verdict,
        "field_cases_dev_only": field,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(results, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")
    for name in models:
        parts = []
        for s in SETS:
            b = systems[name][s]
            last = b["end_to_end_exact"] or b["type_target_exact"] or 0
            parts.append(f"{s} F1 {b['type_macro_f1']:.4f} e2e {last:.4f}")
        row = " | ".join(parts)
        print(f"{name:9s} {row}")
    print(f"verdict: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
