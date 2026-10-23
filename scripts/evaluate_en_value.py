#!/usr/bin/env python
"""Score a gidi-en value-head model (int8 + fp32 ONNX) on every evaluation set, vs the v2 model
and vs the playground's regex scanner.

Usage:
    uv run python scripts/evaluate_en_value.py --onnx-dir models/gidi-en-v3-onnx/seed1 \\
        --checkpoint models/gidi-en-v3/seed1 --out experiments/gidi-en-student-v3/results.json

Sets (all silver labels, never human): test ``primary`` (consensus type/target labels) and
``secondary`` (all complete notes), ``dev``, the slang probe and the Gen Z probe. Per set: type
macro-F1, target exact and joint (target with whole-word snapping, as v2), value exact with
snapping (declared) and without, present/null accuracy, value errors by category, and the same
value metrics for the playground scanner ``findEnglishValue`` (run through bun,
``experiments/gidi-en-student-v3/app_scanner.ts``): ``strict`` (span equals the gold span,
currency marker included) and ``core`` (currency symbols and money words stripped from both).
The pass/fail verdict is computed from ``experiments/gidi-en-student-v3/protocol.json``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from evaluate_en_probe import load_probe
from evaluate_en_student import evaluate_set
from evaluate_onnx import snap_span, span_key

from gidi.evaluation.value_predict import decode_predictions, onnx_logits
from gidi.export.onnx_export import make_session
from gidi.modeling.value import load_value_checkpoint, prepare_value
from gidi.training.english import attach_values, load_labeled_values, primary_ids
from gidi.value_parser.english import amount_core, loose_candidates

PROTOCOL = Path("experiments/gidi-en-student-v3/protocol.json")
SCANNER = Path("experiments/gidi-en-student-v3/app_scanner.ts")
V2 = Path("experiments/gidi-en-student-v2")

_SCALED = re.compile(
    r"\d\s?(?:k|m|mm|bn|b|cr|lac|lacs|grand|lakhs?|crores?|mil|million|thousand)\b", re.I
)
_CURRENCY = re.compile(
    r"[$£€₹¥₱]|\d\s?(?:us|s|a|c)?\$|\b(?:bucks?|bux|dollars?|usd|sgd|gbp|eur|inr|quid|pounds?|"
    r"euros?|rupees?|rs|aed|cents?)\b",
    re.I,
)
CATEGORIES = (
    "no_amount",
    "two_numbers",
    "scaled_k_grand",
    "decimals",
    "currency_marker",
    "word_amount",
    "grouped_thousands",
)


def categories(record: dict[str, Any]) -> list[str]:
    """Value categories of a note (multi-label); ``plain`` when none of the others applies."""
    gold = record["value"]
    tags: list[str] = []
    if gold is None:
        tags.append("no_amount")
    if len(loose_candidates(record["text"])) >= 2:
        tags.append("two_numbers")
    if gold is not None:
        g = gold["text"]
        if _SCALED.search(g):
            tags.append("scaled_k_grand")
        if re.search(r"\d\.\d", g):
            tags.append("decimals")
        if _CURRENCY.search(g):
            tags.append("currency_marker")
        if not re.search(r"\d", g):
            tags.append("word_amount")
        if re.search(r"\d,\d", g):
            tags.append("grouped_thousands")
    return tags or ["plain"]


def run_scanner(texts: list[str]) -> list[dict[str, Any] | None]:
    """The playground ``findEnglishValue`` spans of ``texts`` (code points = Python indices)."""
    out = subprocess.run(
        ["bun", str(SCANNER)], input=json.dumps(texts), capture_output=True, text=True, check=True
    )
    spans = json.loads(out.stdout)
    return [
        None if s is None else {"start": s[0], "end": s[1], "text": t[s[0] : s[1]]}
        for s, t in zip(spans, texts, strict=True)
    ]


def _same(gold: dict[str, Any] | None, pred: dict[str, Any] | None, *, core: bool) -> bool:
    if gold is None or pred is None:
        return gold is None and pred is None
    if core:
        return amount_core(gold["text"]) == amount_core(pred["text"])
    return span_key(gold) == span_key(pred)


def value_block(
    records: list[dict[str, Any]], preds: list[dict[str, Any] | None], conf: list[float] | None
) -> dict[str, Any]:
    """Value metrics of ``preds`` over the notes whose value label is complete."""
    idx = [i for i, r in enumerate(records) if r["value_status"] == "complete"]
    golds = [records[i]["value"] for i in idx]
    ps = [preds[i] for i in idx]
    exact = [_same(g, p, core=False) for g, p in zip(golds, ps, strict=True)]
    core = [_same(g, p, core=True) for g, p in zip(golds, ps, strict=True)]
    present = [(g is None) == (p is None) for g, p in zip(golds, ps, strict=True)]
    cats: dict[str, dict[str, Any]] = {}
    for i, ok in zip(idx, exact, strict=True):
        for c in categories(records[i]):
            slot = cats.setdefault(c, {"n": 0, "exact": 0})
            slot["n"] += 1
            slot["exact"] += ok
    errors = []
    for i, ok in zip(idx, exact, strict=True):
        if ok:
            continue
        r, p = records[i], preds[i]
        errors.append(
            {
                "id": r["id"],
                "text": r["text"],
                "gold": r["value"] and r["value"]["text"],
                "pred": p and p["text"],
                "categories": categories(r),
                "confidence": None if conf is None else round(conf[i], 3),
            }
        )
    n = len(idx)
    return {
        "n": n,
        "exact": sum(exact) / n,
        "core_exact": sum(core) / n,
        "present_null_accuracy": sum(present) / n,
        "by_category": {
            c: {**v, "exact_rate": v["exact"] / v["n"]} for c, v in sorted(cats.items())
        },
        "errors": errors,
    }


def score_model(
    session: Any, tokenizer: Any, records: list[dict[str, Any]], max_length: int
) -> dict[str, Any]:
    data = prepare_value(tokenizer, records, max_length, strict=False)
    preds = decode_predictions(data, onnx_logits(session, data))
    types = [p["type"] for p in preds]
    texts = [r["text"] for r in records]
    snapped_t = [snap_span(p["target"], t) for p, t in zip(preds, texts, strict=True)]
    values_raw = [p["value"] for p in preds]
    values_snap = [snap_span(v, t) for v, t in zip(values_raw, texts, strict=True)]
    conf = [p["value_confidence"] for p in preds]
    base = evaluate_set(records, types, snapped_t)
    snap = value_block(records, values_snap, conf)
    raw = value_block(records, values_raw, conf)
    joint = [
        t == r["type"] and span_key(s) == span_key(r["target"]) and _same(r["value"], v, core=False)
        for r, t, s, v in zip(records, types, snapped_t, values_snap, strict=True)
    ]
    return {
        **base,
        "value": snap,
        "value_unsnapped": {k: raw[k] for k in ("exact", "core_exact", "present_null_accuracy")},
        "full_joint": sum(joint) / len(joint),
        "predictions": [
            {
                "id": r["id"],
                "text": r["text"],
                "gold": [
                    r["type"],
                    r["target"] and r["target"]["text"],
                    r["value"] and r["value"]["text"],
                ],
                "pred": [t, s and s["text"], v and v["text"]],
            }
            for r, t, s, v in zip(records, types, snapped_t, values_snap, strict=True)
        ],
    }


def eval_sets() -> dict[str, list[dict[str, Any]]]:
    test = load_labeled_values("test")
    keep = primary_ids("test")
    slang, _ = load_probe(Path("datasets/probe-en-slang"))
    genz, _ = load_probe(Path("datasets/probe-en-genz"))
    return {
        "primary": [r for r in test if r["id"] in keep],
        "secondary": test,
        "dev": load_labeled_values("dev"),
        "slang_probe": attach_values(slang, "probe-slang"),
        "genz_probe": attach_values(genz, "probe-genz"),
    }


def v2_reference() -> dict[str, dict[str, float]]:
    results = json.loads((V2 / "results.json").read_text(encoding="utf-8"))["int8_snap"]
    slang = json.loads((V2 / "slang-probe-snap.json").read_text(encoding="utf-8"))["all"]
    genz = json.loads((V2 / "genz-probe-snap.json").read_text(encoding="utf-8"))["all"]
    pick = ("type_macro_f1", "span_exact", "joint")
    return {
        "primary": {k: results["primary"][k] for k in pick},
        "secondary": {k: results["secondary"][k] for k in pick},
        "dev": {k: results["dev"][k] for k in pick},
        "slang_probe": {k: slang[k] for k in pick},
        "genz_probe": {k: genz[k] for k in pick},
    }


def verdict(results: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    int8 = results["int8"]
    v2 = results["v2_reference"]
    succ = protocol["success"]
    checks: dict[str, Any] = {}
    for name, key in (("value_test_primary", "primary"), ("value_test_secondary", "secondary")):
        v = int8[key]["value"]["exact"]
        checks[name] = {
            "value": v,
            "min": succ["value_exact_min"]["test"],
            "pass": v >= succ["value_exact_min"]["test"],
        }
    v = int8["slang_probe"]["value"]["exact"]
    m = succ["value_exact_min"]["slang_probe"]
    checks["value_slang_probe"] = {"value": v, "min": m, "pass": v >= m}
    drop = succ["max_regression_vs_v2"]
    for key in ("primary", "secondary", "slang_probe"):
        for metric, label in (("type_macro_f1", "type_macro_f1"), ("span_exact", "target_exact")):
            now, ref = int8[key][metric], v2[key][metric]
            checks[f"{label}_{key}"] = {
                "value": now,
                "v2": ref,
                "min": ref - drop,
                "pass": now >= ref - drop - 1e-12,
            }
    for key in ("primary", "secondary", "slang_probe"):
        model, scanner = int8[key]["value"]["core_exact"], results["scanner"][key]["core_exact"]
        checks[f"beats_scanner_core_{key}"] = {
            "model": model,
            "scanner": scanner,
            "pass": model >= scanner,
        }
    size = results["int8_bytes"]
    checks["int8_size"] = {
        "bytes": size,
        "max": succ["int8_max_bytes"],
        "pass": size <= succ["int8_max_bytes"],
    }
    return {"checks": checks, "pass": all(c["pass"] for c in checks.values())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--onnx-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--protocol", type=Path, default=PROTOCOL)
    args = ap.parse_args(argv)

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    _, tokenizer, meta = load_value_checkpoint(args.checkpoint)
    max_length = int(meta["max_length"])
    sets = eval_sets()
    int8_path = args.onnx_dir / "model.int8.onnx"
    results: dict[str, Any] = {
        "int8_file": str(int8_path),
        "int8_bytes": int8_path.stat().st_size,
        "fp32_bytes": (args.onnx_dir / "model.onnx").stat().st_size,
        "n": {k: len(v) for k, v in sets.items()},
        "v2_reference": v2_reference(),
    }
    for kind, name in (("int8", "model.int8.onnx"), ("fp32", "model.onnx")):
        session = make_session(args.onnx_dir / name, threads=1)
        results[kind] = {k: score_model(session, tokenizer, v, max_length) for k, v in sets.items()}
    scanner: dict[str, Any] = {}
    for key, records in sets.items():
        spans = run_scanner([r["text"] for r in records])
        scanner[key] = value_block(records, spans, None)
    results["scanner"] = scanner
    results["verdict"] = verdict(results, protocol)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for key in sets:
        m = results["int8"][key]
        print(
            key,
            m["n"],
            f"type {m['type_macro_f1']:.3f} target {m['span_exact']:.3f} "
            f"value {m['value']['exact']:.3f} (raw {m['value_unsnapped']['exact']:.3f}) "
            f"scanner strict {scanner[key]['exact']:.3f} core {scanner[key]['core_exact']:.3f}",
        )
    print(json.dumps(results["verdict"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
