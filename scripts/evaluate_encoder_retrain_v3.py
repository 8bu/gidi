#!/usr/bin/env python
"""Score the three seeds and the soft-vote ensemble of the annotation-v3 retrain v3 once.

Usage:
    uv run python scripts/evaluate_encoder_retrain_v3.py \\
        [--onnx-dir experiments/annotation-v3-retrain-v3/onnx] \\
        [--out experiments/annotation-v3-retrain-v3/results.json] [--smoke]

Systems: the retrained INT8 encoders ``<onnx-dir>/seed{1,2,3}/model.int8.onnx`` (same bundle
config, tokenizer and decoding as ``models/gidi-finance-v1``), the ensemble of the three, and
the unchanged rule value parser ``gidi.value_parser.parse_value`` for the value span. Held-out
sets, scored once and never trained on: the frozen test split (105), probe-v1 (81 complete
labels) and ``datasets/annotation-v2/human-value-01`` (147 complete labels, annotation-v3 debt
rule). The helpers are those of ``scripts/evaluate_encoder_retrain.py`` and ``..._v2.py``.

Ensemble (``ENSEMBLE_METHOD``, fixed before any seed was scored): soft vote. The three models
share one tokenizer, so tokens align. Type = argmax of the mean of the three softmax type
distributions; tags = argmax of the mean of the three per-token softmax tag distributions;
the target span is decoded from those tags with the unchanged ``decode_first_span``.

Rows reused from the previous experiment (``experiments/annotation-v3-retrain-v2/results.json``):
retrain-v2 seed 1 and the retrain-v2 three-seed mean. The old encoder (``models/gidi-finance-v1``)
and retrain-v2 seed 1 are re-run only for per-note comparison; their aggregates are asserted equal
to the stored ones. Refuses to overwrite ``--out``. ``--smoke`` swaps every set for in-sample data
and never writes into ``experiments/``.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

import evaluate_encoder_retrain as base  # noqa: E402
import evaluate_encoder_retrain_v2 as v2e  # noqa: E402

from gidi.inference import GidiPredictor  # noqa: E402
from gidi.inference.decode import decode_first_span, log_softmax  # noqa: E402
from gidi.inference.text import normalize_nfc  # noqa: E402
from gidi.modeling.preprocessing import TYPES  # noqa: E402

V2_RESULTS = ROOT / "experiments" / "annotation-v3-retrain-v2" / "results.json"
V2_ONNX = ROOT / "experiments" / "annotation-v3-retrain-v2" / "onnx"
DEFAULT_ONNX = ROOT / "experiments" / "annotation-v3-retrain-v3" / "onnx"
DEFAULT_OUT = ROOT / "experiments" / "annotation-v3-retrain-v3" / "results.json"
SEEDS = (1, 2, 3)
EPS = 1e-9

ENSEMBLE_METHOD = (
    "Fixed before any seed was scored. Soft vote over the three INT8 models of seeds 1, 2, 3 "
    "(same tokenizer, so tokens align): type = argmax of the mean of the three softmax type "
    "distributions; tag = argmax of the mean of the three per-token softmax tag distributions, "
    "and the target span is decoded from those tags by the unchanged decode_first_span; value = "
    "the unchanged rule parser gidi.value_parser.parse_value. No weights, no tuning."
)
GATE_RULE = (
    "Fixed before any seed was scored; the rule of retrain-v2 (old = the deployed encoder "
    "models/gidi-finance-v1): (a) frozen-test type macro-F1 drop <= 0.01, (b) frozen-test "
    "target exact drop <= 0.02, (c) human-value-01 debt-only slice (n=13, encoder + unchanged "
    "parser) end-to-end exact >= 0.8. Reported for seed 1 (deployment convention), each seed, "
    "the three-seed mean and the ensemble; nothing is selected by it."
)
KEYS_ENC = ("type_accuracy", "type_macro_f1", "target_exact", "span_f1")
KEYS_HUMAN = (*KEYS_ENC, "type_target_exact", "value_exact", "end_to_end_exact")
SLICE_NAMES = ("all_complete", "debt_only", "debt_only_v1_skipped", "non_debt")


class SoftVoteEnsemble:
    """Mean of softmax type / tag distributions of several members, same bundle tokenizer."""

    def __init__(self, members: list[GidiPredictor]) -> None:
        self._members = members

    def predict(self, text: str) -> SimpleNamespace:
        raws = [m.run(text) for m in self._members]
        first = raws[0]
        for raw in raws[1:]:
            if (
                raw.input_ids != first.input_ids
                or raw.normalized_text != first.normalized_text
                or raw.normalized_offsets != first.normalized_offsets
            ):
                raise ValueError("ensemble members tokenize the note differently")
        type_p = np.mean([np.exp(log_softmax(r.type_logits)) for r in raws], axis=0)
        tag_p = np.mean([np.exp(log_softmax(r.tag_logits)) for r in raws], axis=0)
        decoded = decode_first_span(
            first.normalized_offsets, np.argmax(tag_p, axis=-1), first.normalized_text
        )
        span = None
        if decoded is not None:
            span = normalize_nfc(text).span_to_original(decoded.start, decoded.end)
        return SimpleNamespace(type=TYPES[int(np.argmax(type_p))], target_span=span)


def recall_by_type(records: list[dict[str, Any]], types: list[str]) -> dict[str, Any]:
    out = {}
    for kind in TYPES:
        idx = [i for i, r in enumerate(records) if r["type"] == kind]
        out[kind] = {"n": len(idx), "k": sum(types[i] == kind for i in idx)}
    return out


def system_metrics(
    sets: dict[str, list[dict[str, Any]]],
    preds: dict[str, tuple[list[str], list[dict[str, Any] | None]]],
    parser_values: list[dict[str, Any] | None],
    slices: dict[str, set[str]],
) -> dict[str, Any]:
    out: dict[str, Any] = {s: base.encoder_block(sets[s], *preds[s]) for s in ("test", "probe")}
    human = sets["human"]
    out["human"] = {}
    for name, ids in slices.items():
        recs, t, sp, v = base.pick(human, *preds["human"], parser_values, ids=ids)
        out["human"][name] = base.human_block(recs, t, sp, v)
    out["human_recall_by_type"] = recall_by_type(human, preds["human"][0])
    return out


def assert_equal(label: str, new: dict[str, Any], stored: dict[str, Any], keys) -> None:
    for k in keys:
        if stored.get(k) is None and new.get(k) is None:
            continue
        if abs(new[k] - stored[k]) > EPS:
            raise SystemExit(f"{label}: {k} {new[k]} differs from the stored {stored[k]}")


def wrong_rows(
    human: list[dict[str, Any]],
    pred: tuple[list[str], list[dict[str, Any] | None]],
    values: list[dict[str, Any] | None],
    others: dict[str, tuple[list[str], list[dict[str, Any] | None]]],
) -> list[dict[str, Any]]:
    """Every human-value-01 note whose type, target or value is not exactly the gold."""
    rows = []
    for i, r in enumerate(human):
        t, s, v = pred[0][i], pred[1][i], values[i]
        type_ok, target_ok = t == r["type"], base.key(s) == base.key(r["target"])
        value_ok = base.key(v) == base.key(r["value"])
        if type_ok and target_ok and value_ok:
            continue
        text_of = lambda span: None if span is None else span["text"]  # noqa: E731
        row = {
            "id": r["id"],
            "text": r["text"],
            "gold": {
                "type": r["type"],
                "target": text_of(r["target"]),
                "value": text_of(r["value"]),
            },
            "pred": {"type": t, "target": text_of(s), "value": text_of(v)},
            "type_ok": type_ok,
            "target_ok": target_ok,
            "value_ok": value_ok,
            "debt_only": r["id"] in base.DEBT_ONLY_IDS,
        }
        for name, (o_types, o_spans) in others.items():
            row[f"{name}_end_to_end_ok"] = (
                o_types[i] == r["type"]
                and base.key(o_spans[i]) == base.key(r["target"])
                and base.key(v) == base.key(r["value"])
            )
            row[f"{name}_pred"] = {"type": o_types[i], "target": text_of(o_spans[i])}
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--onnx-dir", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--smoke", action="store_true", help="in-sample data only; debug run")
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists: held-out scoring runs once")
    if args.smoke and ROOT / "experiments" in args.out.resolve().parents:
        raise SystemExit("--smoke must not write into experiments/")

    v2 = json.loads(V2_RESULTS.read_text("utf-8"))
    sets, human_meta = base.load_sets(args.smoke)
    human = sets["human"]
    human_ids = {r["id"] for r in human}
    debt_ids = human_ids if args.smoke else set(base.DEBT_ONLY_IDS)
    v1_skipped = debt_ids if args.smoke else set(base.V1_SKIPPED_DEBT_IDS)
    if debt_ids - human_ids:
        raise SystemExit(f"debt-only ids not in the human set: {sorted(debt_ids - human_ids)}")
    slices = {
        "all_complete": human_ids,
        "debt_only": debt_ids,
        "debt_only_v1_skipped": v1_skipped,
        "non_debt": human_ids - debt_ids,
    }
    parser_values = base.value_spans(human)

    def models_of(onnx_dir: Path, seeds=SEEDS) -> dict[int, Path]:
        return {s: onnx_dir / f"seed{s}" / "model.int8.onnx" for s in seeds}

    def load(model: Path | None) -> GidiPredictor:
        return GidiPredictor.from_bundle(base.OLD_BUNDLE, model_path=model, intra_op_threads=1)

    # sanity (in-sample, no scoring): a one-member ensemble is the member's own prediction
    check = load(models_of(args.onnx_dir)[1])
    solo = SoftVoteEnsemble([check])
    for r in base.read_jsonl(base.VALIDATION):
        one, ref = solo.predict(r["text"]), check.predict(r["text"])
        if one.type != ref.type or one.target_span != ref.target_span:
            raise SystemExit(f"ensemble of one differs from the member on {r['text']!r}")

    systems: dict[str, Any] = {}
    preds: dict[str, dict[str, Any]] = {}
    old = load(None)
    old_model = base.OLD_BUNDLE / "model.int8.onnx"
    preds["old"] = {s: base.predict_all(old, recs) for s, recs in sets.items()}
    systems["old"] = system_metrics(sets, preds["old"], parser_values, slices)
    v2_seed1_model = models_of(V2_ONNX, (1,))[1]
    preds["v2_seed1"] = {s: base.predict_all(load(v2_seed1_model), r) for s, r in sets.items()}
    systems["v2_seed1"] = system_metrics(sets, preds["v2_seed1"], parser_values, slices)
    if not args.smoke:  # the reused rows are valid only if the re-run reproduces them
        stored = v2["seeds"]["1"]["metrics"]
        for s in ("test", "probe"):
            assert_equal(f"v2 seed1 {s}", systems["v2_seed1"][s], stored[s], KEYS_ENC)
            assert_equal(f"old {s}", systems["old"][s], v2["old_encoder"][s], KEYS_ENC)
        for name in slices:
            assert_equal(
                f"v2 seed1 human/{name}",
                systems["v2_seed1"]["human"][name],
                stored["human"][name],
                KEYS_HUMAN,
            )
            assert_equal(
                f"old human/{name}",
                systems["old"]["human"][name],
                v2["reused_from_v3_retrain_v1"]["human"][name]["v8_old"],
                KEYS_HUMAN,
            )

    members: list[GidiPredictor] = []
    models = models_of(args.onnx_dir)
    for seed, model in models.items():
        members.append(load(model))
        preds[f"v3_seed{seed}"] = {s: base.predict_all(members[-1], r) for s, r in sets.items()}
        systems[f"v3_seed{seed}"] = system_metrics(
            sets, preds[f"v3_seed{seed}"], parser_values, slices
        )
    ensemble = SoftVoteEnsemble(members)
    preds["v3_ensemble"] = {s: base.predict_all(ensemble, r) for s, r in sets.items()}
    systems["v3_ensemble"] = system_metrics(sets, preds["v3_ensemble"], parser_values, slices)

    # three-seed aggregate of the new seeds
    aggregates: dict[str, Any] = {"seeds": list(SEEDS), "metrics": {}}
    seed_names = [f"v3_seed{k}" for k in SEEDS]
    for s in ("test", "probe"):
        blocks = [systems[n][s] for n in seed_names]
        aggregates["metrics"][s] = v2e.agg_block(blocks, KEYS_ENC) | {
            "per_class_f1": {
                t: v2e.aggregate([b["per_class_f1"][t] for b in blocks]) for t in TYPES
            }
        }
    aggregates["metrics"]["human"] = {
        name: v2e.agg_block([systems[n]["human"][name] for n in seed_names], KEYS_HUMAN)
        for name in slices
    }

    # gate, reported for every candidate
    old_test = systems["old"]["test"]
    mean_test = v2e.mean_block([systems[n]["test"] for n in seed_names], KEYS_ENC)
    mean_debt = aggregates["metrics"]["human"]["debt_only"]["end_to_end_exact"]["mean"]
    gates = {
        n: v2e.gate(
            old_test, systems[n]["test"], systems[n]["human"]["debt_only"]["end_to_end_exact"]
        )
        for n in (*seed_names, "v3_ensemble")
    }
    gates["v3_mean"] = v2e.gate(old_test, mean_test, mean_debt)

    # remaining wrong human-value-01 notes of seed 1 and the ensemble
    others = {"old": preds["old"]["human"], "v2_seed1": preds["v2_seed1"]["human"]}
    wrong = {
        "v3_seed1": wrong_rows(human, preds["v3_seed1"]["human"], parser_values, others),
        "v3_ensemble": wrong_rows(human, preds["v3_ensemble"]["human"], parser_values, others),
    }

    # size and latency (1 thread, systems interleaved per round)
    sample = random.Random(0).sample([r["text"] for r in sets["test"]], min(50, len(sets["test"])))
    latency = base.latency(
        {"single_model": members[0], "ensemble_of_3": ensemble}, sample, args.runs, args.warmup
    )
    sizes = {f"seed{s}_int8_onnx": m.stat().st_size for s, m in models.items()}
    other_files = sum(
        p.stat().st_size for p in base.OLD_BUNDLE.iterdir() if p.name != old_model.name
    )
    size = {
        **sizes,
        "single_model_int8_onnx": sizes["seed1_int8_onnx"],
        "ensemble_int8_onnx": sum(sizes.values()),
        "bundle_other_files": other_files,
        "single_bundle": sizes["seed1_int8_onnx"] + other_files,
        "ensemble_bundle": sum(sizes.values()) + other_files,
    }

    results: dict[str, Any] = {
        "experiment": "annotation-v3-retrain-v3",
        "smoke": args.smoke,
        "ensemble_method": ENSEMBLE_METHOD,
        "gate_rule": GATE_RULE,
        "old_model": {"path": str(old_model), "sha256": base.sha256_file(old_model)},
        "models": {
            f"v3_seed{s}": {"path": str(m), "sha256": base.sha256_file(m)}
            for s, m in models.items()
        },
        "v2_seed1_model": {"path": str(v2_seed1_model), "sha256": base.sha256_file(v2_seed1_model)},
        "sets": {s: len(r) for s, r in sets.items()},
        "human_value_01": {**human_meta, "complete": len(human)},
        "debt_only_ids": sorted(debt_ids),
        "systems": systems,
        "aggregate_v3_seeds": aggregates,
        "mean_test_metrics": mean_test,
        "gate": gates,
        "reused_from_v2": {
            "path": str(V2_RESULTS.relative_to(ROOT)),
            "v2_seed1": {
                "model": v2["seeds"]["1"]["model"],
                "metrics": v2["seeds"]["1"]["metrics"],
            },
            "v2_mean": v2["aggregate"],
            "old_encoder": v2["old_encoder"],
            "human_v7_v8old_v3retrain_v1": v2["reused_from_v3_retrain_v1"]["human"],
        },
        "wrong_human_notes": wrong,
        "size_bytes": size,
        "latency_int8_ms": latency,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "types": list(TYPES),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(results, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")
    for name in ("old", "v2_seed1", *seed_names, "v3_ensemble"):
        m, h = systems[name]["test"], systems[name]["human"]["all_complete"]
        d = systems[name]["human"]["debt_only"]
        print(
            f"{name:12s} test acc {m['type_accuracy']:.4f} F1 {m['type_macro_f1']:.4f} "
            f"tgt {m['target_exact']:.4f} | probe F1 {systems[name]['probe']['type_macro_f1']:.4f}"
            f" | human acc {h['type_accuracy']:.4f} e2e {h['end_to_end_exact']:.4f} "
            f"debt e2e {d['end_to_end_exact']:.4f}"
        )
    for name, g in gates.items():
        print(f"gate {name}: {g['pass']}")
    print(json.dumps(latency, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
