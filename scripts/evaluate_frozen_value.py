#!/usr/bin/env python
"""Score a frozen-v1 value-head experiment: type/target invariance, value metrics, verdict A/B/C.

Usage (after training finished; seeds 1-3 in models/<experiment>/seed{n}):
    uv run python scripts/evaluate_frozen_value.py \\
        --protocol experiments/value-span-v4-nonlinear-head/protocol.json [--overwrite]

Evaluation only; reads the frozen protocol (default value-span-v3-frozen-v1). The experiment
name comes from the protocol; ``--v2-dir`` / ``--out`` default to ``models/<experiment>`` and
``experiments/<experiment>``.

1. Invariance (CPU). The base checkpoint (gidi-finance-v1 seed 1, 2 heads) and every value-head
   seed (3 heads) run on the same inputs of test (143, with test-v1 105 and test-targeted 38)
   and probe (81). Required exactly equal: type logits, target logits (max abs diff 0), type
   prediction, target BIO prediction, decoded target span. The seed's decoded type / target is
   also cross-checked with ``experiments/value-span-v1/eval/v1-seed1-{test,probe}-predictions
   .jsonl`` (the frozen v1 seed-1 predictions), as is the base. The ``frozen_state_sha256`` of
   every checkpoint's metadata, the base's recomputed hash and each loaded seed model's
   recomputed hash must be equal.
2. Value metrics with the value-span-v1 code (``compare_value_v1.score_model`` /
   ``value_eval``) on test, test-v1, test-targeted, probe, validation (monitoring only); value
   slices and ``value_gate.value_quality_criteria`` (mean over seeds and seed 1), unchanged
   thresholds. The old v1 regression gate (mean over v1 seeds) is not applied.
3. Verdict: C if any invariance mismatch, else A if every value criterion passes, else B; the
   meaning is the protocol's ``adoption_criteria.verdict`` text.
4. Every wrong value on complete labels (test, probe) is categorised by span position
   (``value_gate.value_error_category``); the protocol's tracked notes are reported per seed.

Writes under ``--out``:
    invariance.json
    eval/seed{n}-{set}.json, eval/seed{n}-{set}-predictions.jsonl
    multi-number-errors.jsonl   every multi-number value error on test and probe, per seed
    value-errors.jsonl          every value error on test and probe with its category
    comparison.json
Refuses to overwrite any of them without ``--overwrite``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import compare_value_v1 as cmp  # noqa: E402

from gidi.corpus.jsonl import read_jsonl, write_jsonl  # noqa: E402
from gidi.evaluation.value_eval import value_span_categories  # noqa: E402
from gidi.evaluation.value_gate import (  # noqa: E402
    headline_stats,
    multi_number_errors,
    seen_value_texts,
    value_error_category,
    value_error_table,
    value_quality_criteria,
    value_slice_table,
)
from gidi.evaluation.value_predict import decode_predictions, torch_logits  # noqa: E402
from gidi.modeling.checkpoint import WEIGHTS_FILE, load_checkpoint  # noqa: E402
from gidi.modeling.value import load_value_checkpoint, prepare_value, value_crf  # noqa: E402
from gidi.training.train_value_head import frozen_state_sha256  # noqa: E402

ROOT = cmp.ROOT
DEFAULT_EXPERIMENT = "value-span-v3-frozen-v1"
PROTOCOL = ROOT / "experiments" / DEFAULT_EXPERIMENT / "protocol.json"
V1_PREDICTIONS = ROOT / "experiments" / "value-span-v1" / "eval"
INVARIANCE_SETS = ("test", "test-v1", "test-targeted", "probe")
FILE_SETS = ("test", "probe")  # sets with a frozen v1 seed-1 prediction file
ERROR_SETS = ("test", "probe")
SEEDS = cmp.SEEDS
EXPORT_SEED = cmp.EXPORT_SEED
# Held-out notes tracked by text (report only; never used for training or tuning).
TRACKED_NOTES = ("cho a Nam vay 1 triệu 20/10", "Thắng vay 5 củ, hẹn t10 trả")


# --------------------------------------------------------------------------- invariance


def _bio(tags: list[np.ndarray]) -> list[list[int]]:
    return [[int(k) for k in t.argmax(-1)] for t in tags]


def run_model(model: Any, tokenizer: Any, meta: dict[str, Any], records: list[dict[str, Any]]):
    """Encoded inputs, raw logits, decoded predictions of one checkpoint on ``records`` (CPU).

    A v1 (2-head) model has no value logits: a constant ``O`` stands in (``cmp.v1_logits``),
    which the decoder turns into a null value that is never compared.
    """
    data = prepare_value(tokenizer, records, int(meta["max_length"]), strict=False)
    device = torch.device("cpu")
    if hasattr(model, "value_head"):
        logits = torch_logits(model, data, device)
    else:
        logits = cmp.v1_logits(model, data, device)
    return data, logits, decode_predictions(data, logits, value_crf(model))


def per_record_diffs(
    base: tuple[Any, Any, list[dict[str, Any]]],
    other: tuple[Any, Any, list[dict[str, Any]]],
) -> dict[str, list[Any]]:
    """Per-record difference of ``other`` to ``base``: 0 / False means identical."""
    (bdata, (btype, btag, _), bpred), (odata, (otype, otag, _), opred) = base, other
    n = len(bpred)
    ids_ne = [
        not torch.equal(bdata.encoded.input_ids[i], odata.encoded.input_ids[i])
        or not torch.equal(bdata.encoded.attention_mask[i], odata.encoded.attention_mask[i])
        for i in range(n)
    ]
    bbio, obio = _bio(btag), _bio(otag)
    return {
        "input_ids_ne": ids_ne,
        "type_logits_diff": [float(np.abs(btype[i] - otype[i]).max()) for i in range(n)],
        "target_logits_diff": [
            float(np.abs(btag[i] - otag[i]).max()) if btag[i].shape == otag[i].shape else np.inf
            for i in range(n)
        ],
        "type_prediction_ne": [bpred[i]["type"] != opred[i]["type"] for i in range(n)],
        "target_bio_ne": [bbio[i] != obio[i] for i in range(n)],
        "target_span_ne": [bpred[i]["target"] != opred[i]["target"] for i in range(n)],
    }


def file_diffs(
    pred: list[dict[str, Any]], records: list[dict[str, Any]], rows: dict[str, dict[str, Any]]
) -> dict[str, list[bool]]:
    """Per-record difference of decoded type / target to the frozen v1 seed-1 prediction rows."""
    pairs = list(zip(pred, records, strict=True))
    return {
        "type_ne": [p["type"] != rows[r["id"]]["pred_type"] for p, r in pairs],
        "target_ne": [p["target"] != rows[r["id"]]["pred_target"] for p, r in pairs],
    }


def aggregate(diffs: dict[str, list[Any]], idx: list[int]) -> dict[str, Any]:
    """Mismatch counts and max abs diffs of the records ``idx``."""
    out: dict[str, Any] = {"n": len(idx)}
    for key, values in diffs.items():
        sub = [values[i] for i in idx]
        if key.endswith("_diff"):
            out[key.removesuffix("_diff")] = {
                "records_not_equal": sum(1 for v in sub if v != 0.0),
                "max_abs_diff": max(sub, default=0.0),
            }
        else:
            out[key.removesuffix("_ne")] = sum(bool(v) for v in sub)
    return out


def mismatch_total(block: dict[str, Any]) -> int:
    """Records that differ, summed over every check of one aggregated set block."""
    total = 0
    for key, value in block.items():
        if key == "n":
            continue
        if isinstance(value, dict):
            total += (
                value["records_not_equal"]
                if "records_not_equal" in value
                else mismatch_total(value)
            )
        else:
            total += int(value)
    return total


def invariance_ok(report: dict[str, Any]) -> bool:
    return (
        report["checks"]["base_sha_matches_protocol"]
        and report["checks"]["frozen_state_hashes_equal_base"]
        and report["summary"]["total_mismatches"] == 0
        and report["summary"]["max_abs_diff_type_logits"] == 0.0
        and report["summary"]["max_abs_diff_target_logits"] == 0.0
    )


def load_v1_rows(kind: str) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(V1_PREDICTIONS / f"v1-seed1-{kind}-predictions.jsonl")
    by_id = {r["id"]: r for r in rows}
    if len(by_id) != len(rows):
        raise SystemExit(f"v1-seed1-{kind}-predictions.jsonl: duplicate ids")
    return by_id


def invariance(
    base_dir: Path,
    ckpts: dict[int, Path],
    sets: dict[str, list[dict[str, Any]]],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    base_model, base_tok, base_meta = load_checkpoint(base_dir)
    base_state = frozen_state_sha256(base_model)
    base_file_sha = cmp.sha256(base_dir / WEIGHTS_FILE)

    test, probe = sets["test"], sets["probe"]
    ids = [r["id"] for r in (*test, *probe)]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate record ids across test and probe")
    pos = {r["id"]: i for i, r in enumerate(test)}
    indices = {
        "test": list(range(len(test))),
        "test-v1": [pos[r["id"]] for r in sets["test-v1"]],
        "test-targeted": [pos[r["id"]] for r in sets["test-targeted"]],
        "probe": list(range(len(probe))),
    }
    base_runs = {
        "test": run_model(base_model, base_tok, base_meta, test),
        "probe": run_model(base_model, base_tok, base_meta, probe),
    }
    v1_rows = {"test": load_v1_rows("test"), "probe": load_v1_rows("probe")}

    def source(name: str) -> str:
        return "probe" if name == "probe" else "test"

    def block(diffs: dict[str, dict[str, list[Any]]], file: dict[str, dict[str, list[bool]]]):
        out = {}
        for name in INVARIANCE_SETS:
            src = source(name)
            entry = aggregate(diffs[src], indices[name])
            entry["vs_v1_seed1_predictions"] = aggregate(file[src], indices[name])
            out[name] = entry
        return out

    base_file = {s: file_diffs(base_runs[s][2], sets[s], v1_rows[s]) for s in ("test", "probe")}
    seeds_report: dict[str, Any] = {}
    all_blocks = []
    for seed, ckpt in ckpts.items():
        model, tok, meta = load_value_checkpoint(ckpt)
        recomputed = frozen_state_sha256(model)
        runs = {
            "test": run_model(model, tok, meta, test),
            "probe": run_model(model, tok, meta, probe),
        }
        diffs = {s: per_record_diffs(base_runs[s], runs[s]) for s in ("test", "probe")}
        file = {s: file_diffs(runs[s][2], sets[s], v1_rows[s]) for s in ("test", "probe")}
        sets_block = block(diffs, file)
        all_blocks.append(sets_block)
        frozen_base = meta.get("frozen_base", {})
        seeds_report[str(seed)] = {
            "checkpoint": str(ckpt),
            "frozen_state_sha256_meta": meta.get("frozen_state_sha256"),
            "frozen_state_sha256_recomputed": recomputed,
            "meta_equals_base": meta.get("frozen_state_sha256") == base_state,
            "recomputed_equals_base": recomputed == base_state,
            "meta_frozen_base_sha256": frozen_base.get("model_safetensors_sha256"),
            "meta_frozen_base_sha256_equals_base": (
                frozen_base.get("model_safetensors_sha256") == base_file_sha
            ),
            "sets": sets_block,
        }
    base_vs_file = {
        name: aggregate(base_file[source(name)], indices[name]) for name in INVARIANCE_SETS
    }

    def worst(key: str) -> float:
        return max(
            (b[name][key]["max_abs_diff"] for b in all_blocks for name in INVARIANCE_SETS),
            default=0.0,
        )

    hashes_equal = all(
        s["meta_equals_base"]
        and s["recomputed_equals_base"]
        and s["meta_frozen_base_sha256_equals_base"]
        for s in seeds_report.values()
    )
    total = sum(mismatch_total(entry) for b in all_blocks for entry in b.values())
    report = {
        "experiment": protocol["experiment"],
        "device": "cpu",
        "base": {
            "checkpoint": str(base_dir),
            "model_safetensors_sha256": base_file_sha,
            "protocol_model_safetensors_sha256": protocol["base_model"]["model_safetensors_sha256"],
            "frozen_state_sha256": base_state,
        },
        "v1_seed1_prediction_files": {
            s: {
                "path": str(V1_PREDICTIONS / f"v1-seed1-{s}-predictions.jsonl"),
                "sha256": cmp.sha256(V1_PREDICTIONS / f"v1-seed1-{s}-predictions.jsonl"),
            }
            for s in FILE_SETS
        },
        "base_vs_v1_seed1_predictions": base_vs_file,
        "checks": {
            "base_sha_matches_protocol": base_file_sha
            == protocol["base_model"]["model_safetensors_sha256"],
            "frozen_state_hashes_equal_base": hashes_equal,
        },
        "seeds": seeds_report,
        "summary": {
            "total_mismatches": total,
            "max_abs_diff_type_logits": worst("type_logits"),
            "max_abs_diff_target_logits": worst("target_logits"),
        },
    }
    report["summary"]["ok"] = invariance_ok(report)
    return report


# --------------------------------------------------------------------------- checkpoints


def check_seed_checkpoint(path: Path, seed: int, experiment: str) -> dict[str, Any]:
    """Meta of a completed value-head seed dir (the v1 checks plus the frozen-run keys)."""
    meta = cmp.check_v2_checkpoint(path, seed)
    problems = []
    if meta.get("experiment") != experiment:
        problems.append(f"experiment is {meta.get('experiment')!r}")
    trainable = meta.get("trainable_parameters") or []
    if not trainable or not all(n.startswith("value_head.") for n in trainable):
        problems.append(f"trainable_parameters is {trainable!r}")
    if not meta.get("frozen_state_sha256"):
        problems.append("no frozen_state_sha256")
    if "model_safetensors_sha256" not in (meta.get("frozen_base") or {}):
        problems.append("no frozen_base.model_safetensors_sha256")
    if problems:
        raise SystemExit(f"{path}: not a {experiment} checkpoint ({'; '.join(problems)})")
    return meta


def output_paths(out: Path, seeds: tuple[int, ...]) -> list[Path]:
    paths = [
        out / "comparison.json",
        out / "invariance.json",
        out / "multi-number-errors.jsonl",
        out / "value-errors.jsonl",
    ]
    for seed in seeds:
        for name in cmp.SETS:
            stem = out / "eval" / f"seed{seed}-{name}"
            paths += [stem.with_suffix(".json"), Path(f"{stem}-predictions.jsonl")]
    return paths


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--v2-dir", type=Path, default=None, help="default: models/<experiment>")
    parser.add_argument("--out", type=Path, default=None, help="default: experiments/<experiment>")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--data-dir", type=Path, default=cmp.DATA_DIR)
    parser.add_argument("--base", type=Path, default=None, help="default: protocol base checkpoint")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    experiment = protocol["experiment"]
    args.v2_dir = args.v2_dir or Path("models") / experiment
    args.out = args.out or Path("experiments") / experiment
    base_dir = args.base or ROOT / protocol["base_model"]["checkpoint"]
    if not (base_dir / WEIGHTS_FILE).is_file():
        raise SystemExit(f"{base_dir}: no base checkpoint")
    ckpts = {s: args.v2_dir / f"seed{s}" for s in SEEDS}
    meta = {s: check_seed_checkpoint(ckpts[s], s, experiment) for s in SEEDS}
    if not args.overwrite:
        existing = [p for p in output_paths(args.out, SEEDS) if p.exists()]
        if existing:
            raise SystemExit(
                f"refusing to overwrite {existing[0]} (+{len(existing) - 1} more); pass --overwrite"
            )

    sets = cmp.load_sets(args.data_dir)
    train_records = read_jsonl(args.data_dir / "train.jsonl")
    seen_texts = seen_value_texts(train_records)
    categories = value_span_categories()
    files = cmp.describe_files(args.data_dir, sets, train_records)

    print("invariance (CPU): base v1 vs every value-head seed")
    inv = invariance(base_dir, ckpts, sets, protocol)
    cmp.write_json(args.out / "invariance.json", inv)
    _print_invariance(inv)

    summaries: dict[int, dict[str, Any]] = {}
    rows: dict[int, dict[str, list]] = {}
    print(f"value head {experiment}")
    for s in SEEDS:
        summaries[s], rows[s] = cmp.score_model(
            version="v2",
            seed=s,
            ckpt=ckpts[s],
            sets=sets,
            set_names=tuple(cmp.SETS),
            train_records=train_records,
            seen_texts=seen_texts,
            categories=categories,
            flags={},
            out=args.out,
            files=files,
        )

    errors = [
        e
        for s in SEEDS
        for name in ("test", "probe")
        for e in multi_number_errors(name, s, rows[s][name])
    ]
    write_jsonl(args.out / "multi-number-errors.jsonl", errors, overwrite=True)
    error_counts, value_errors = value_error_table(rows, ERROR_SETS)
    write_jsonl(args.out / "value-errors.jsonl", value_errors, overwrite=True)
    tracked = {
        text: {
            f"seed{s}": [
                {
                    "set": name,
                    "gold_value": r["gold_value"],
                    "pred_value": r["pred_value"],
                    "value_ok": r["value_ok"],
                    "category": value_error_category(r["text"], r["gold_value"], r["pred_value"]),
                }
                for name in ERROR_SETS
                for r in rows[s][name]
                if r["text"] == text
            ]
            for s in SEEDS
        }
        for text in TRACKED_NOTES
    }

    criteria = [
        *value_quality_criteria(summaries, list(SEEDS), "mean"),
        *value_quality_criteria(summaries, [EXPORT_SEED], f"seed{EXPORT_SEED}"),
    ]
    value_ok = all(c["pass"] for c in criteria)
    invariant = inv["summary"]["ok"]
    verdict = "C" if not invariant else ("A" if value_ok else "B")
    report = {
        "experiment": experiment,
        "protocol": {"path": str(args.protocol), "sha256": cmp.sha256(args.protocol)},
        "sets": files,
        "checkpoints": {
            "base": str(base_dir),
            "base_model_safetensors_sha256": inv["base"]["model_safetensors_sha256"],
            "value_head": {
                s: {
                    "path": str(ckpts[s]),
                    "epochs": meta[s]["epochs"],
                    "model_sha256": cmp.sha256(ckpts[s] / WEIGHTS_FILE),
                    "frozen_state_sha256": meta[s]["frozen_state_sha256"],
                }
                for s in SEEDS
            },
        },
        "verdict": {
            "verdict": verdict,
            "meaning": protocol["adoption_criteria"]["verdict"][verdict],
            "seeds": list(SEEDS),
            "export_seed": EXPORT_SEED,
            "invariance": {"pass": invariant, **inv["summary"], "file": "invariance.json"},
            "value_quality": {
                "pass": value_ok,
                "failed": [f"{c['id']}[{c['scope']}]" for c in criteria if not c["pass"]],
                "criteria": criteria,
            },
        },
        "invariance": inv["summary"] | {"checks": inv["checks"]},
        "results": {
            name: headline_stats({s: summaries[s][name] for s in SEEDS}) for name in cmp.SETS
        },
        "value_slices": {
            "test": value_slice_table({s: summaries[s]["test"] for s in SEEDS}),
            "test_human_only": value_slice_table(
                {s: summaries[s]["test"] for s in SEEDS}, "value_slices_human"
            ),
            "probe": value_slice_table({s: summaries[s]["probe"] for s in SEEDS}),
            "probe_human_only": value_slice_table(
                {s: summaries[s]["probe"] for s in SEEDS}, "value_slices_human"
            ),
        },
        "slice_definitions": cmp._slice_definitions()["value"],
        "multi_number": {
            "errors_file": str(args.out / "multi-number-errors.jsonl"),
            "n_errors": {
                f"seed{s}": {
                    name: sum(1 for e in errors if e["seed"] == s and e["set"] == name)
                    for name in ("test", "probe")
                }
                for s in SEEDS
            },
            "n_notes": {
                name: summaries[SEEDS[0]][name]["value_slices"]["multi_number"]["n"]
                for name in ("test", "probe")
            },
        },
        "value_errors": {"file": str(args.out / "value-errors.jsonl"), "counts": error_counts},
        "tracked_notes": tracked,
        "per_seed": summaries,
        "monitoring_only": ["validation"],
    }
    cmp.write_json(args.out / "comparison.json", report)
    _print_verdict(report["verdict"], report["results"])
    print("  value errors by category (seeds 1 / 2 / 3)")
    for name, by_cat in error_counts.items():
        for cat, per_seed in by_cat.items():
            if any(per_seed.values()):
                print(f"    {name:6s} {cat:28s} " + " / ".join(str(per_seed[s]) for s in SEEDS))
    print("  tracked notes")
    for text, by_seed in tracked.items():
        for seed, hits in by_seed.items():
            for h in hits:
                pred = (h["pred_value"] or {}).get("text")
                print(f"    {seed} {h['set']}: {text!r} -> {pred!r} ({h['category'] or 'exact'})")
    print(f"comparison -> {args.out / 'comparison.json'}")
    return 0


def _print_invariance(inv: dict[str, Any]) -> None:
    print(
        f"  base sha matches protocol: {inv['checks']['base_sha_matches_protocol']}; "
        f"frozen-state hashes equal base: {inv['checks']['frozen_state_hashes_equal_base']}"
    )
    for seed, entry in inv["seeds"].items():
        for name, block in entry["sets"].items():
            ref = block["vs_v1_seed1_predictions"]
            print(
                f"  seed{seed} {name:<13} n={block['n']:<4} "
                f"type logits {block['type_logits']['max_abs_diff']:g} "
                f"target logits {block['target_logits']['max_abs_diff']:g} "
                f"type pred ne {block['type_prediction']} bio ne {block['target_bio']} "
                f"span ne {block['target_span']} | v1 file type ne {ref['type']} "
                f"target ne {ref['target']}"
            )
    summary = inv["summary"]
    print(
        f"  INVARIANCE {'PASS' if summary['ok'] else 'FAIL'}: {summary['total_mismatches']} "
        f"mismatches, max abs diff type {summary['max_abs_diff_type_logits']:g} "
        f"target {summary['max_abs_diff_target_logits']:g}"
    )


def _print_verdict(verdict: dict[str, Any], results: dict[str, Any]) -> None:
    print(f"\nVERDICT {verdict['verdict']}: {verdict['meaning']}")
    inv = verdict["invariance"]
    print(
        f"  invariance [{'PASS' if inv['pass'] else 'FAIL'}] {inv['total_mismatches']} mismatches"
    )
    print("  value_quality")
    for c in verdict["value_quality"]["criteria"]:
        value = "n/a" if c["value"] is None else f"{c['value']:.4f}"
        print(
            f"    [{'PASS' if c['pass'] else 'FAIL'}] {c['id']:<34} {c['scope']:<6} "
            f"{value} {c['comparison']} {c['threshold']}"
        )
    print("  mean over seeds")
    for name, stats in results.items():
        line = (
            f"    {name:<13} n={stats['n']:<4} type acc {stats['type_accuracy']['mean']:.4f} "
            f"target exact {stats['target_exact']['mean']:.4f}"
        )
        if "value" in stats and stats["value"]["exact"]["mean"] is not None:
            line += f" value exact {stats['value']['exact']['mean']:.4f}"
        print(line)


if __name__ == "__main__":
    raise SystemExit(main())
