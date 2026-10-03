#!/usr/bin/env python
"""Experiment value-span-v8-deterministic-parser: develop on train, freeze, score once, cost.

Stages (protocol: experiments/value-span-v8-deterministic-parser/protocol.json):

    train     score the parser on train.jsonl only (development numbers); writes nothing.
    freeze    train stage + write parser-freeze.json (parser source sha256, protocol sha256, train
              numbers) and train-errors.jsonl. Run once, before any held-out scoring.
    heldout   score validation, test, test-v1, test-targeted, probe ONCE with the frozen parser:
              v8 = gidi-finance-v1 INT8 (type, target) + parser (value), against gidi-finance-v2
              INT8 and the rule proposer diagnostic. Refuses to run unless parser-freeze.json
              matches the parser source, and refuses to overwrite results.
    cost      size, CPU latency and RSS of v2 INT8, v1 INT8, parser and v1 + parser.
    verdict   combine results.json and cost.json with the declared verdict rule.

Scoring reuses the frozen V7 code (gidi.evaluation.value_metrics / value_gate / value_predict and
scripts/compare_value_v1.load_sets); no metric is re-implemented here. Predictions come from
``GidiPredictor`` (caller-string offsets) and from ``parse_value`` instead of from logits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

from gidi.corpus.jsonl import read_jsonl, write_jsonl  # noqa: E402
from gidi.inference import GidiPredictor  # noqa: E402
from gidi.value_parser import parse_value  # noqa: E402

EXP = ROOT / "experiments" / "value-span-v8-deterministic-parser"
DATA = ROOT / "datasets" / "annotation-v2" / "training-v1"
V1_BUNDLE = ROOT / "models" / "gidi-finance-v1"
V2_BUNDLE = ROOT / "models" / "gidi-finance-v2"
TOKENIZER_CHECKPOINT = ROOT / "models" / "value-span-v7-dual-encoder" / "seed1"
PROTOCOL = EXP / "protocol.json"
FREEZE = EXP / "parser-freeze.json"
PARSER_DIR = ROOT / "src" / "gidi" / "value_parser"
SCORING_FILES = (
    "src/gidi/evaluation/value_eval.py",
    "src/gidi/evaluation/value_metrics.py",
    "src/gidi/evaluation/value_gate.py",
    "src/gidi/evaluation/value_predict.py",
    "scripts/score_web_parity_quality.py",
    "scripts/compare_value_v1.py",
)
EVAL_SETS = ("test", "test-v1", "test-targeted", "probe", "validation", "validation-new")
SYSTEMS = ("v8", "v2", "proposer")
EXPECTED_SHA = {
    "train": "0a03db64238e6f202918a30e3727178c437045ce4703d38f14a7dda9883589e5",
    "validation": "eef077c97a519dc532112683f3ab35daeaa6e55187c3c2fb01f53a0ee37f629c",
    "test": "acaca8d767828e48954e91871886ac191a45110af695af7df7573a9c0be6f91d",
    "probe": "638aeff00243f20af2dbeb2b340da0142631566057aef57d75005b75aedece87",
}


# --------------------------------------------------------------------------- small helpers


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parser_source_sha256() -> str:
    """One hash over every ``*.py`` of the parser package (names and bytes, sorted)."""
    digest = hashlib.sha256()
    for path in sorted(PARSER_DIR.glob("*.py")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def key(span: dict[str, Any] | None) -> tuple[int, int] | None:
    return None if span is None else (span["start"], span["end"])


def span_dict(text: str, start: int, end: int) -> dict[str, Any]:
    return {"text": text[start:end], "start": start, "end": end}


def write_json(path: Path, payload: Any, *, exclusive: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# --------------------------------------------------------------------------- scoring context


class Scorer:
    """Frozen V7 scoring for a list of (records, predictions) pairs."""

    def __init__(self, train_records: list[dict[str, Any]]) -> None:
        from gidi.evaluation.value_eval import value_span_categories
        from gidi.evaluation.value_gate import seen_value_texts
        from gidi.modeling.value import load_value_checkpoint

        _, self.tokenizer, meta = load_value_checkpoint(TOKENIZER_CHECKPOINT)
        self.max_length = int(meta["max_length"])
        self.train_records = train_records
        self.seen = seen_value_texts(train_records)
        self.categories = value_span_categories()

    def score(
        self, records: list[dict[str, Any]], preds: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """``(set summary, per-record rows)``, as ``evaluate_records`` + ``summarize_set``."""
        from gidi.evaluation.value_gate import summarize_set
        from gidi.evaluation.value_metrics import evaluate_value, gold_token_count, surface_pattern
        from gidi.evaluation.value_predict import prediction_rows
        from gidi.modeling.value import prepare_value

        data = prepare_value(self.tokenizer, records, self.max_length, strict=False)
        offsets = data.encoded.offsets
        metrics = evaluate_value(
            records,
            preds,
            offsets=offsets,
            train_records=self.train_records,
            categories=self.categories,
        )
        seen_patterns = {
            surface_pattern(t["value"]["text"])
            for t in self.train_records
            if t.get("value_status") == "complete" and t.get("value") is not None
        }
        extra = []
        for record, offs in zip(records, offsets, strict=True):
            row: dict[str, Any] = {"categories": list(self.categories(record["text"]))}
            if record.get("value") is not None:
                pattern = surface_pattern(record["value"]["text"])
                row["gold_value_pattern"] = pattern
                row["pattern_seen_in_train"] = pattern in seen_patterns
                row["gold_value_tokens"] = gold_token_count(record, offs)
            extra.append(row)
        rows = prediction_rows(records, preds, extra)
        summary = summarize_set(
            metrics, rows, records, with_value=True, seen_texts=self.seen, regression_flags=None
        )
        return summary, rows


def pred_dict(
    record: dict[str, Any], model_pred: Any, value: dict[str, Any] | None
) -> dict[str, Any]:
    """Prediction dict of ``decode_predictions``' shape from a ``GidiPredictor`` result."""
    target = None
    if model_pred.target_span is not None:
        target = span_dict(record["text"], *model_pred.target_span)
    return {
        "type": model_pred.type,
        "type_confidence": model_pred.type_confidence,
        "target": target,
        "target_confidence": model_pred.target_confidence,
        "value": value,
        "value_confidence": None,
    }


def v2_value(record: dict[str, Any], model_pred: Any) -> dict[str, Any] | None:
    if model_pred.value is None or model_pred.value.span is None:
        return None
    return span_dict(record["text"], *model_pred.value.span)


def parser_value(text: str) -> dict[str, Any] | None:
    span = parse_value(text)
    return None if span is None else span_dict(text, span.start, span.end)


def proposer_value(text: str) -> dict[str, Any] | None:
    from gidi.annotation.value_span import propose_value

    chosen = propose_value(text).chosen
    return None if chosen is None else span_dict(text, chosen.start, chosen.end)


# --------------------------------------------------------------------------- train / freeze


def train_stage(write: bool) -> None:
    train = read_jsonl(DATA / "train.jsonl")
    if sha256_file(DATA / "train.jsonl") != EXPECTED_SHA["train"]:
        raise SystemExit("train.jsonl sha256 differs from the protocol")
    scorer = Scorer(train)
    preds = [
        {
            "type": r["type"],
            "type_confidence": 1.0,
            "target": r["target"],
            "target_confidence": 1.0,
            "value": parser_value(r["text"]),
            "value_confidence": None,
        }
        for r in train
    ]
    summary, rows = scorer.score(train, preds)
    value = summary["value"]
    errors = [
        {k: r[k] for k in ("id", "text", "value_provenance", "gold_value", "pred_value")}
        for r in rows
        if r["value_status"] == "complete" and not r["value_ok"]
    ]
    print(
        f"train value exact {value['exact']:.4f} (n={value['n']}, errors={value['n_errors']}) "
        f"human {value['human']['exact']:.4f} (n={value['human']['n']}) "
        f"present/null {value['present_accuracy']:.4f} token F1 {value['token_f1']:.4f}"
    )
    for slice_name, block in summary["value_slices"].items():
        if block["n"]:
            print(f"  {slice_name:18s} n={block['n']:4d} exact {block['exact']:.4f}")
    for error in errors:
        print(
            "  ERR",
            error["value_provenance"],
            error["text"],
            error["gold_value"],
            error["pred_value"],
        )
    if write:
        write_json(
            FREEZE,
            {
                "experiment": "value-span-v8-deterministic-parser",
                "frozen_at_utc": utc_now(),
                "protocol_sha256": sha256_file(PROTOCOL),
                "parser_source_sha256": parser_source_sha256(),
                "parser_files": {p.name: sha256_file(p) for p in sorted(PARSER_DIR.glob("*.py"))},
                "run_script_sha256": sha256_file(Path(__file__)),
                "python": platform.python_version(),
                "train_sha256": EXPECTED_SHA["train"],
                "train_dev_summary": summary,
                "statement": (
                    "Parser frozen after development on train.jsonl only. Held-out scoring "
                    "(validation, test, probe) runs once after this file exists and is "
                    "refused when the parser source differs."
                ),
            },
        )
        write_jsonl(EXP / "train-errors.jsonl", errors)
        FREEZE.chmod(0o444)
        print(f"wrote {FREEZE} (parser sha256 {parser_source_sha256()})")


# --------------------------------------------------------------------------- held-out


def check_frozen() -> dict[str, Any]:
    if not FREEZE.exists():
        raise SystemExit("parser-freeze.json is missing: run --stage freeze first")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if freeze["parser_source_sha256"] != parser_source_sha256():
        raise SystemExit("parser source changed after the freeze; held-out scoring refused")
    if freeze["protocol_sha256"] != sha256_file(PROTOCOL):
        raise SystemExit("protocol.json changed after the freeze")
    return freeze


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def heldout_stage() -> None:
    from gidi.evaluation.value_gate import (
        multi_number_errors,
        value_error_table,
        value_quality_criteria,
    )

    freeze = check_frozen()
    results_path = EXP / "results.json"
    if results_path.exists():
        raise SystemExit("results.json exists: held-out scoring runs once (protocol)")
    for name, expected in EXPECTED_SHA.items():
        path = DATA / ("probe-v1-eval-only.jsonl" if name == "probe" else f"{name}.jsonl")
        if sha256_file(path) != expected:
            raise SystemExit(f"{path.name} sha256 differs from the protocol")

    from compare_value_v1 import load_sets

    sets = load_sets(DATA)
    train = read_jsonl(DATA / "train.jsonl")
    train_ids = {r["id"] for r in train}
    sets["validation-new"] = [r for r in sets["validation"] if r["id"] not in train_ids]
    scorer = Scorer(train)
    p1 = GidiPredictor.from_bundle(V1_BUNDLE, intra_op_threads=1)
    p2 = GidiPredictor.from_bundle(V2_BUNDLE, intra_op_threads=1)

    summaries: dict[str, dict[str, Any]] = {s: {} for s in SYSTEMS}
    rows_by: dict[str, dict[str, list[dict[str, Any]]]] = {s: {} for s in SYSTEMS}
    identity = {"notes": 0, "type_diff": 0, "target_diff": 0, "confidence_diff": 0, "diffs": []}
    determinism = {"notes": 0, "mismatch": 0, "errors": 0}
    cache: dict[str, tuple[Any, Any]] = {}
    for name in EVAL_SETS:
        records = sets[name]
        pairs = []
        for record in records:
            text = record["text"]
            if text not in cache:
                cache[text] = (p1.predict(text), p2.predict(text))
            pairs.append(cache[text])
        per_system = {
            "v8": [
                pred_dict(r, a, parser_value(r["text"]))
                for r, (a, _) in zip(records, pairs, strict=True)
            ],
            "v2": [
                pred_dict(r, b, v2_value(r, b)) for r, (_, b) in zip(records, pairs, strict=True)
            ],
            "proposer": [
                pred_dict(r, a, proposer_value(r["text"]))
                for r, (a, _) in zip(records, pairs, strict=True)
            ],
        }
        if name in ("validation", "test", "probe"):
            for record, (a, b) in zip(records, pairs, strict=True):
                identity["notes"] += 1
                same_type = a.type == b.type
                same_target = a.target_span == b.target_span
                same_conf = (a.type_confidence, a.target_confidence) == (
                    b.type_confidence,
                    b.target_confidence,
                )
                identity["type_diff"] += not same_type
                identity["target_diff"] += not same_target
                identity["confidence_diff"] += not same_conf
                if not (same_type and same_target):
                    identity["diffs"].append(record["id"])
        for record in records:
            determinism["notes"] += 1
            try:
                determinism["mismatch"] += parser_value(record["text"]) != parser_value(
                    record["text"]
                )
            except Exception:  # noqa: BLE001 - counted, never raised
                determinism["errors"] += 1
        for system in SYSTEMS:
            summary, rows = scorer.score(records, per_system[system])
            summaries[system][name] = summary
            rows_by[system][name] = rows
            write_json(EXP / "eval" / f"{system}-{name}.json", summary)
            write_jsonl(EXP / "eval" / f"{system}-{name}-predictions.jsonl", rows)
        print(
            f"{name:14s} n={len(records):3d} value exact: "
            + " ".join(
                f"{s} {pct(summaries[s][name]['value']['exact'])}"
                f" (human {pct(summaries[s][name]['value']['human']['exact'])})"
                for s in SYSTEMS
            )
        )

    adversarial = ["", " ", "\n", "1" * 50_000, "5 củ " * 10_000, "\u0300" * 100, "k" * 50_000]
    determinism["adversarial_errors"] = 0
    for text in adversarial:
        try:
            if parse_value(text) != parse_value(text):
                determinism["mismatch"] += 1
        except Exception:  # noqa: BLE001
            determinism["adversarial_errors"] += 1

    # ---- criteria
    def test_value(system: str) -> dict[str, Any]:
        return summaries[system]["test"]["value"]

    def notes(block: dict[str, Any]) -> int:
        return round(block["exact"] * block["n"])

    quality = {
        s: value_quality_criteria({1: {"test": summaries[s]["test"]}}, [1], s) for s in SYSTEMS
    }
    t8, t2 = summaries["v8"]["test"], summaries["v2"]["test"]
    multi8, multi2 = t8["value_slices"]["multi_number"], t2["value_slices"]["multi_number"]
    gates = {
        "G1_test_value_exact": quality["v8"][0],
        "G2_test_value_exact_human": quality["v8"][1],
        "G3_test_present_null": quality["v8"][2],
        "G4_test_multi_number": quality["v8"][3],
        "G5_slice_floor": quality["v8"][4],
        "G6_non_inferior_test_exact": {
            "v8": t8["value"]["exact"],
            "v2": t2["value"]["exact"],
            "threshold": t2["value"]["exact"] - 0.02,
            "pass": t8["value"]["exact"] >= t2["value"]["exact"] - 0.02 - 1e-9,
        },
        "G7_non_inferior_present_null": {
            "v8": t8["value"]["present_accuracy"],
            "v2": t2["value"]["present_accuracy"],
            "threshold": t2["value"]["present_accuracy"] - 0.01,
            "pass": t8["value"]["present_accuracy"]
            >= t2["value"]["present_accuracy"] - 0.01 - 1e-9,
        },
        "G8_non_inferior_human_exact": {
            "v8_notes": notes(t8["value"]["human"]),
            "v2_notes": notes(t2["value"]["human"]),
            "n": t8["value"]["human"]["n"],
            "pass": notes(t8["value"]["human"]) >= notes(t2["value"]["human"]) - 1,
        },
        "G9_non_inferior_multi_number": {
            "v8_notes": notes(multi8),
            "v2_notes": notes(multi2),
            "n": multi8["n"],
            "pass": notes(multi8) >= notes(multi2) - 1,
        },
        "G10_type_target_unchanged": {
            **{k: v for k, v in identity.items() if k != "diffs"},
            "diff_ids": identity["diffs"],
            "pass": identity["type_diff"] == 0 and identity["target_diff"] == 0,
        },
    }
    pooled = pooled_human(sets, rows_by)
    secondary = {
        "S1_probe_value_exact": {
            "v8": summaries["v8"]["probe"]["value"]["exact"],
            "v2": summaries["v2"]["probe"]["value"]["exact"],
            "threshold": 0.95,
            "pass": summaries["v8"]["probe"]["value"]["exact"] >= 0.95 - 1e-9,
        },
        "S2_pooled_human_heldout_exact": {
            **pooled,
            "threshold": 0.90,
            "pass": pooled["v8"]["exact"] >= 0.90 - 1e-9,
        },
        "S3_no_crash_and_deterministic": {
            **determinism,
            "pass": determinism["mismatch"] == 0
            and determinism["errors"] == 0
            and determinism["adversarial_errors"] == 0,
        },
    }

    # ---- errors
    error_counts: dict[str, Any] = {}
    error_rows: list[dict[str, Any]] = []
    multi_rows: list[dict[str, Any]] = []
    for system in ("v8", "v2"):
        by_set = {1: {n: rows_by[system][n] for n in EVAL_SETS}}
        counts, errors = value_error_table(by_set, EVAL_SETS)
        error_counts[system] = {
            n: {c: v[1] for c, v in cats.items() if v[1]} for n, cats in counts.items()
        }
        error_rows += [{"system": system, **e} for e in errors]
        for name in EVAL_SETS:
            multi_rows += [
                {"system": system, **m} for m in multi_number_errors(name, 1, rows_by[system][name])
            ]
    write_jsonl(EXP / "value-errors.jsonl", error_rows)
    write_jsonl(EXP / "multi-number-errors.jsonl", multi_rows)

    changes = note_changes(sets, rows_by)
    results = {
        "experiment": "value-span-v8-deterministic-parser",
        "scored_at_utc": utc_now(),
        "protocol_sha256": sha256_file(PROTOCOL),
        "parser_freeze_sha256": sha256_file(FREEZE),
        "parser_source_sha256": freeze["parser_source_sha256"],
        "frozen_scoring_code_sha256": {f: sha256_file(ROOT / f) for f in SCORING_FILES},
        "data_sha256": EXPECTED_SHA,
        "models": {
            "v1_int8_sha256": sha256_file(V1_BUNDLE / "model.int8.onnx"),
            "v2_int8_sha256": sha256_file(V2_BUNDLE / "model.int8.onnx"),
        },
        "sets": {n: len(sets[n]) for n in EVAL_SETS},
        "systems": {
            "v8": "gidi-finance-v1 INT8 (type, target) + value_parser.parse_value",
            "v2": "gidi-finance-v2 INT8 as shipped (V7 seed 1)",
            "proposer": "diagnostic only: gidi.annotation.value_span.propose_value chosen span "
            "(refuses ambiguous notes); type/target from the v1 path",
        },
        "headline": {
            s: {
                n: {
                    "type_accuracy": summaries[s][n]["type_accuracy"],
                    "target_exact": summaries[s][n]["target_exact"],
                    "value": {
                        k: summaries[s][n]["value"][k]
                        for k in (
                            "n",
                            "exact",
                            "present_accuracy",
                            "span_precision",
                            "span_recall",
                            "span_f1",
                            "token_f1",
                            "n_errors",
                        )
                    },
                    "human": summaries[s][n]["value"]["human"],
                }
                for n in EVAL_SETS
            }
            for s in SYSTEMS
        },
        "slices": {
            s: {
                n: {
                    "all": {k: v["exact"] for k, v in summaries[s][n]["value_slices"].items()}
                    | {"_n": {k: v["n"] for k, v in summaries[s][n]["value_slices"].items()}},
                    "human": {
                        k: v["exact"] for k, v in summaries[s][n]["value_slices_human"].items()
                    }
                    | {"_n": {k: v["n"] for k, v in summaries[s][n]["value_slices_human"].items()}},
                }
                for n in EVAL_SETS
            }
            for s in SYSTEMS
        },
        "provenance_split": provenance_split(sets, rows_by),
        "frozen_v7_criteria": {s: quality[s] for s in ("v8", "v2")},
        "gates": gates,
        "secondary": secondary,
        "error_counts": error_counts,
        "note_changes_v8_vs_v2": changes,
    }
    write_json(results_path, results)
    results_path.chmod(0o444)
    for gate, body in gates.items():
        print(f"{gate:34s} {'PASS' if body['pass'] else 'FAIL'}")
    for gate, body in secondary.items():
        print(f"{gate:34s} {'PASS' if body['pass'] else 'FAIL'}")


def pooled_human(
    sets: dict[str, list[dict[str, Any]]], rows_by: dict[str, dict[str, list[dict[str, Any]]]]
) -> dict[str, Any]:
    """Exact on the human-provenance complete rows of test, probe and validation-new."""
    out: dict[str, Any] = {"sets": ["test", "probe", "validation-new"]}
    for system in SYSTEMS:
        ok = n = 0
        for name in out["sets"]:
            for record, row in zip(sets[name], rows_by[system][name], strict=True):
                if (
                    record.get("value_status") == "complete"
                    and record.get("value_provenance") == "human"
                ):
                    n += 1
                    ok += bool(row["value_ok"])
        out[system] = {"n": n, "correct": ok, "exact": ok / n if n else None}
    return out


def provenance_split(
    sets: dict[str, list[dict[str, Any]]], rows_by: dict[str, dict[str, list[dict[str, Any]]]]
) -> dict[str, Any]:
    """Value exact by label provenance (human / rule) per set and system."""
    out: dict[str, Any] = {}
    for name in EVAL_SETS:
        out[name] = {}
        for system in SYSTEMS:
            block = {}
            for prov in ("human", "rule"):
                flags = [
                    bool(row["value_ok"])
                    for record, row in zip(sets[name], rows_by[system][name], strict=True)
                    if record.get("value_status") == "complete"
                    and record.get("value_provenance") == prov
                ]
                block[prov] = {
                    "n": len(flags),
                    "correct": sum(flags),
                    "exact": sum(flags) / len(flags) if flags else None,
                }
            out[name][system] = block
    return out


def note_changes(
    sets: dict[str, list[dict[str, Any]]], rows_by: dict[str, dict[str, list[dict[str, Any]]]]
) -> dict[str, Any]:
    """Notes where v8 and v2 differ in value correctness, per set."""
    out: dict[str, Any] = {}
    for name in ("test", "probe", "validation-new"):
        v8_only, v2_only, both_wrong = [], [], []
        for record, a, b in zip(sets[name], rows_by["v8"][name], rows_by["v2"][name], strict=True):
            if record.get("value_status") != "complete":
                continue
            item = {
                "id": record["id"],
                "text": record["text"],
                "provenance": record.get("value_provenance"),
                "gold": record["value"],
                "v8": a["pred_value"],
                "v2": b["pred_value"],
            }
            if a["value_ok"] and not b["value_ok"]:
                v8_only.append(item)
            elif b["value_ok"] and not a["value_ok"]:
                v2_only.append(item)
            elif not a["value_ok"] and not b["value_ok"]:
                both_wrong.append(item)
        out[name] = {
            "v8_right_v2_wrong": v8_only,
            "v2_right_v8_wrong": v2_only,
            "both_wrong": both_wrong,
        }
    return out


# --------------------------------------------------------------------------- cost


def dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


RSS_CHILD = """
import json, os, resource, subprocess, sys
sys.path.insert(0, {src!r})
system = {system!r}
texts = [json.loads(l)["text"] for l in open({data!r}) if l.strip()] if system != "baseline" else []
import onnxruntime, numpy  # noqa: F401,E401  (the runtime baseline every system shares)
if system in ("v1", "v1+parser"):
    from gidi.inference import GidiPredictor
    predictor = GidiPredictor.from_bundle({v1!r}, intra_op_threads=1)
elif system == "v2":
    from gidi.inference import GidiPredictor
    predictor = GidiPredictor.from_bundle({v2!r}, intra_op_threads=1)
else:
    predictor = None
if system in ("parser", "v1+parser"):
    from gidi.value_parser import parse_value
for text in texts:
    if predictor is not None:
        predictor.predict(text)
    if system in ("parser", "v1+parser"):
        parse_value(text)
rss = int(subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True,
                         text=True, check=True).stdout)
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS
print(json.dumps({{"rss_mb": rss / 1024, "peak_rss_mb": peak / 2**20}}))
"""


def cost_stage() -> None:
    texts = [r["text"] for r in read_jsonl(DATA / "test.jsonl")] + [
        r["text"] for r in read_jsonl(DATA / "probe-v1-eval-only.jsonl")
    ]
    p1 = GidiPredictor.from_bundle(V1_BUNDLE, intra_op_threads=1)
    p2 = GidiPredictor.from_bundle(V2_BUNDLE, intra_op_threads=1)

    def run_v2(text: str) -> None:
        p2.predict(text)

    def run_v1(text: str) -> None:
        p1.predict(text)

    def run_parser(text: str) -> None:
        parse_value(text)

    def run_v1_parser(text: str) -> None:
        p1.predict(text)
        parse_value(text)

    systems = {"v2": run_v2, "v1": run_v1, "parser": run_parser, "v1+parser": run_v1_parser}
    samples: dict[str, list[float]] = {name: [] for name in systems}
    per_round: dict[str, list[float]] = {name: [] for name in systems}
    rounds, passes = 3, 5
    for fn in systems.values():  # warm-up
        for text in texts:
            fn(text)
    for _ in range(rounds):
        for name, fn in systems.items():
            round_samples = []
            for _ in range(passes):
                for text in texts:
                    start = time.perf_counter_ns()
                    fn(text)
                    round_samples.append((time.perf_counter_ns() - start) / 1e6)
            samples[name] += round_samples
            per_round[name].append(statistics.median(round_samples))
    latency = {
        name: {
            "n_calls": len(values),
            "p50_ms": percentile(values, 0.50),
            "p95_ms": percentile(values, 0.95),
            "mean_ms": statistics.fmean(values),
            "p50_ms_per_round": per_round[name],
        }
        for name, values in samples.items()
    }

    rss = {}
    for system in ("baseline", "v2", "v1", "parser", "v1+parser"):
        runs = []
        for _ in range(3):
            code = RSS_CHILD.format(
                src=str(ROOT / "src"),
                system=system,
                data=str(DATA / "test.jsonl"),
                v1=str(V1_BUNDLE),
                v2=str(V2_BUNDLE),
            )
            proc = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                check=True,
                cwd=ROOT,
            )
            runs.append(json.loads(proc.stdout.strip().splitlines()[-1]))
        rss[system] = {
            "rss_mb_median": statistics.median(r["rss_mb"] for r in runs),
            "peak_rss_mb_median": statistics.median(r["peak_rss_mb"] for r in runs),
            "runs": runs,
        }

    parser_bytes = sum(p.stat().st_size for p in PARSER_DIR.glob("*.py"))
    size = {
        "v1_model_int8_bytes": (V1_BUNDLE / "model.int8.onnx").stat().st_size,
        "v2_model_int8_bytes": (V2_BUNDLE / "model.int8.onnx").stat().st_size,
        "v1_bundle_bytes": dir_bytes(V1_BUNDLE),
        "v2_bundle_bytes": dir_bytes(V2_BUNDLE),
        "parser_source_bytes": parser_bytes,
        "v8_system_bytes": dir_bytes(V1_BUNDLE) + parser_bytes,
    }
    size["v8_vs_v2_bytes"] = size["v8_system_bytes"] - size["v2_bundle_bytes"]
    size["v8_over_v2"] = size["v8_system_bytes"] / size["v2_bundle_bytes"]
    cost = {
        "measured_at_utc": utc_now(),
        "machine": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "notes": {"n_texts": len(texts), "source": "test (143) + probe (81) texts"},
        "threads": 1,
        "rounds": rounds,
        "passes_per_round": passes,
        "size": size,
        "latency": latency,
        "rss": rss,
    }
    write_json(EXP / "cost.json", cost)
    print(
        json.dumps(
            {
                "size": size,
                "latency": {k: {m: v[m] for m in ("p50_ms", "p95_ms")} for k, v in latency.items()},
                "rss": {k: v["rss_mb_median"] for k, v in rss.items()},
            },
            indent=1,
        )
    )


# --------------------------------------------------------------------------- verdict


def verdict_stage() -> None:
    results = json.loads((EXP / "results.json").read_text(encoding="utf-8"))
    cost = json.loads((EXP / "cost.json").read_text(encoding="utf-8"))
    gates = {name: body["pass"] for name, body in results["gates"].items()}
    smaller = cost["size"]["v8_system_bytes"] < cost["size"]["v2_bundle_bytes"]
    faster = cost["latency"]["v1+parser"]["p50_ms"] < cost["latency"]["v2"]["p50_ms"]
    gates["G11_cost_lower"] = smaller and faster
    secondary = {name: body["pass"] for name, body in results["secondary"].items()}
    if not all(gates.values()):
        verdict = "REJECT"
    elif all(secondary.values()):
        verdict = "ACCEPT_AS_V3_CANDIDATE"
    else:
        verdict = "INCONCLUSIVE"
    payload = {
        "verdict": verdict,
        "gates": gates,
        "secondary": secondary,
        "cost_gate": {
            "v8_system_bytes": cost["size"]["v8_system_bytes"],
            "v2_bundle_bytes": cost["size"]["v2_bundle_bytes"],
            "p50_ms_v8": cost["latency"]["v1+parser"]["p50_ms"],
            "p50_ms_v2": cost["latency"]["v2"]["p50_ms"],
        },
        "note": "ACCEPT_AS_V3_CANDIDATE means a follow-up validation is justified; it does not "
        "confirm or ship gidi-finance-v3.",
    }
    write_json(EXP / "verdict.json", payload)
    print(json.dumps(payload, indent=1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--stage", required=True, choices=("train", "freeze", "heldout", "cost", "verdict")
    )
    args = parser.parse_args()
    if args.stage == "train":
        train_stage(write=False)
    elif args.stage == "freeze":
        train_stage(write=True)
    elif args.stage == "heldout":
        heldout_stage()
    elif args.stage == "cost":
        cost_stage()
    else:
        verdict_stage()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
