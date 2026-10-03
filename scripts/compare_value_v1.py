#!/usr/bin/env python
"""Score value-span-v1 against the frozen protocol: v2 seeds, v1 baseline, gate verdict A/B/C.

Usage (after training finished; seeds 1-3 in models/value-span-v1/seed{n}):
    uv run python scripts/compare_value_v1.py --v2-dir models/value-span-v1 \\
        --out experiments/value-span-v1 [--overwrite]
    # v1 baseline alone, no v2 checkpoints needed (sanity check of the metric code)
    uv run python scripts/compare_value_v1.py --v1-only --out /tmp/value-v1-baseline

Evaluation only; reads ``experiments/value-span-v1/protocol.json`` (frozen) and never trains.

Scores, with the same metric code (``gidi.evaluation.value_eval.evaluate_records``) on the same
records:
    v2  models/value-span-v1/seed{1,2,3}                      type + target + value
    v1  models/compression-v3/supervised/...-K2048/seed{1,2,3}  type + target (no value head)
on test (143), test-v1 (105: source_batch != targeted-value-01), test-targeted (38), probe (81),
and validation (149, v2 only; monitoring only, never gated).

Writes under ``--out``:
    eval/seed{n}-{set}.json, eval/seed{n}-{set}-predictions.jsonl   v2, per seed and set
    eval/v1-seed{n}-{set}.json, eval/v1-seed{n}-{set}-predictions.jsonl
    multi-number-errors.jsonl   every multi-number value error on test and probe, per seed
    comparison.json             summaries, mean/std, slices, regression table, changes, verdict
(``--v1-only``: ``v1-baseline.json`` instead of ``comparison.json``; no gate.)

Regression slices (v1 test subset and probe). Probe notes carry a ``pattern`` tag
(``datasets/probe-v1/notes.jsonl``) and use it. The v1 test records have no tag, so their
slices are the golden-suite rules of ``scripts/build_golden_suite.py`` plus one new regex:
    lender_first      probe ``lender_first_cho_muon``; test: rule ``borrow_with_cho_vay_phrase``
                      (type borrow and text contains ``cho mượn/vay``)
    title_name        probe ``title_name_span``; test: rules ``title_excluded_from_target`` or
                      ``title_included_in_target`` (a title word before / starting the target)
    insurance         probe ``insurance_premium`` | ``insurance_payout``; test: rule ``insurance``
                      (``bảo hiểm|bao hiem|bhyt|bhxh|bhnt``)
    loan_installment  probe ``loan_installment``; test: type repayment_out and the regex
                      ``LOAN_INSTALLMENT`` below (no existing code defines it)
    unaccented        record ``accented`` is false (both sets)
The regexes are also run on the probe and compared with its tags (``regex_vs_probe_tags``).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gidi.corpus.jsonl import read_jsonl, write_jsonl
from gidi.evaluation.value_eval import evaluate_records, value_span_categories
from gidi.evaluation.value_gate import (
    REGRESSION_SLICES,
    changes_vs_v1,
    evaluate_gate,
    headline_stats,
    multi_number_errors,
    regression_table,
    seen_value_texts,
    summarize_set,
    value_slice_table,
)
from gidi.evaluation.value_predict import BATCH, Logits, torch_logits
from gidi.modeling.checkpoint import META_FILE, WEIGHTS_FILE, load_checkpoint
from gidi.modeling.value import load_value_checkpoint, value_crf

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets" / "annotation-v2" / "training-v1"
PROTOCOL = ROOT / "experiments" / "value-span-v1" / "protocol.json"
PROBE_NOTES = ROOT / "datasets" / "probe-v1" / "notes.jsonl"
V1_DIR = (
    ROOT
    / "models"
    / "compression-v3"
    / "supervised"
    / "student-4x768-pretrained-pos32-vocab-B-rank-8000-ffn-combined-K2048"
)
SEEDS = (1, 2, 3)
EXPORT_SEED = 1
EXPECTED_EPOCHS = 40  # protocol.training.recipe
TARGETED_BATCH = "targeted-value-01"
# set name -> (expected n); order is the report order. ``validation`` is monitoring only.
SETS = {"test": 143, "test-v1": 105, "test-targeted": 38, "probe": 81, "validation": 149}
GATED_SETS = ("test", "test-v1", "probe")
REGRESSION_SETS = ("test-v1", "probe")
V1_SETS = ("test", "test-v1", "test-targeted", "probe")  # v1 validation is in-sample: skipped
VALUE_KEYS = (
    "pred_value",
    "gold_value",
    "value_confidence",
    "value_ok",
    "full_joint_ok",
    "value_status",
    "value_provenance",
    "categories",
    "gold_value_pattern",
    "pattern_seen_in_train",
    "gold_value_tokens",
)

# Installment of a named loan product: consumer-finance words / installment words. Applied to
# ``repayment_out`` records only (the probe pattern is all repayment_out).
LOAN_INSTALLMENT = re.compile(
    r"khoản vay|khoan vay|(?<!\w)vay(?!\w)|trả góp|tra gop|(?<!\w)g[óo]p(?!\w)"
    r"|(?<!\w)k[ỳy](?!\w)|home credit|fe credit",
    re.IGNORECASE,
)
PROBE_TAGS = {
    "lender_first": {"lender_first_cho_muon"},
    "title_name": {"title_name_span"},
    "insurance": {"insurance_premium", "insurance_payout"},
    "loan_installment": {"loan_installment"},
}


# --------------------------------------------------------------------------- helpers


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_golden_rules() -> dict[str, Callable[[dict[str, Any]], bool]]:
    """Rules of ``scripts/build_golden_suite.py`` by name (not importable as a package)."""
    spec = importlib.util.spec_from_file_location(
        "build_golden_suite", ROOT / "scripts" / "build_golden_suite.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # @dataclass looks the module up while it executes
    spec.loader.exec_module(module)
    return {rule.name: rule.match for rule in module.RULES}


def regex_flags(record: dict[str, Any], rules: dict[str, Callable[[dict], bool]]) -> set[str]:
    """Regression slices of a record from the golden-suite rules (+ the loan regex)."""
    gold = {"type": record["type"], "target": record["target"]}
    rec = {"text": record["text"], "accented": record["accented"], "gold": gold}
    flags = set()
    if rules["borrow_with_cho_vay_phrase"](rec):
        flags.add("lender_first")
    if rules["title_excluded_from_target"](rec) or rules["title_included_in_target"](rec):
        flags.add("title_name")
    if rules["insurance"](rec):
        flags.add("insurance")
    if record["type"] == "repayment_out" and LOAN_INSTALLMENT.search(record["text"]):
        flags.add("loan_installment")
    if not record["accented"]:
        flags.add("unaccented")
    return flags


def tag_flags(record: dict[str, Any], pattern: str) -> set[str]:
    flags = {name for name, tags in PROBE_TAGS.items() if pattern in tags}
    if not record["accented"]:
        flags.add("unaccented")
    return flags


def load_sets(data_dir: Path) -> dict[str, list[dict[str, Any]]]:
    test = read_jsonl(data_dir / "test.jsonl")
    sets = {
        "test": test,
        "test-v1": [r for r in test if r["source_batch"] != TARGETED_BATCH],
        "test-targeted": [r for r in test if r["source_batch"] == TARGETED_BATCH],
        "probe": read_jsonl(data_dir / "probe-v1-eval-only.jsonl"),
        "validation": read_jsonl(data_dir / "validation.jsonl"),
    }
    for name, expected in SETS.items():
        if len(sets[name]) != expected:
            raise SystemExit(f"{name}: {len(sets[name])} records, protocol says {expected}")
    return sets


@torch.no_grad()
def v1_logits(model: Any, data: Any, device: torch.device, batch_size: int = BATCH) -> Logits:
    """``torch_logits`` for a 2-head v1 model; the value logits are constant ``O`` (unused)."""
    model.eval()
    enc = data.encoded
    types: list[np.ndarray] = []
    tags: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for i in range(0, len(data), batch_size):
        mask = enc.attention_mask[i : i + batch_size]
        width = int(mask.sum(dim=1).max())
        ids = enc.input_ids[i : i + batch_size, :width]
        type_l, tag_l = model(ids.to(device), mask[:, :width].to(device))
        types.append(type_l.float().cpu().numpy())
        tag_np = tag_l.float().cpu().numpy()
        for j in range(ids.shape[0]):
            n = int(mask[j].sum())
            tags.append(tag_np[j, :n])
            values.append(np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (n, 1)))
    return np.concatenate(types), tags, values


def strip_value(metrics: dict[str, Any], rows: list[dict[str, Any]]):
    """Drop everything value-related (v1 has no value head; its value logits are a constant)."""
    kept = {k: metrics[k] for k in ("n", "overall", "alignment") if k in metrics}
    kept["alignment"] = {
        k: v for k, v in kept.get("alignment", {}).items() if not k.startswith("value")
    }
    return kept, [{k: v for k, v in row.items() if k not in VALUE_KEYS} for row in rows]


def check_v2_checkpoint(path: Path, seed: int) -> dict[str, Any]:
    """Meta of a completed v2 seed dir; exit unless the final (last-epoch) checkpoint is there."""
    problems = []
    meta: dict[str, Any] = {}
    for name in (WEIGHTS_FILE, META_FILE, "config.json", "tokenizer.json"):
        if not (path / name).is_file():
            problems.append(f"missing {name}")
    if not problems:
        meta = json.loads((path / META_FILE).read_text(encoding="utf-8"))
        if "value_tags" not in meta or meta.get("heads") != ["type", "target", "value"]:
            problems.append("not a value-span checkpoint (no value head in the metadata)")
        if meta.get("epochs") != EXPECTED_EPOCHS:
            problems.append(f"trained {meta.get('epochs')} epochs, protocol says {EXPECTED_EPOCHS}")
        if meta.get("seed") != seed:
            problems.append(f"checkpoint seed is {meta.get('seed')}")
    if problems:
        raise SystemExit(f"{path}: no completed final checkpoint ({'; '.join(problems)})")
    return meta


def output_paths(out: Path, v2_seeds: tuple[int, ...], v1_seeds: tuple[int, ...], v1_only: bool):
    paths = [out / ("v1-baseline.json" if v1_only else "comparison.json")]
    if not v1_only:
        paths.append(out / "multi-number-errors.jsonl")
    groups = (("v1-", v1_seeds, V1_SETS), ("", () if v1_only else v2_seeds, SETS))
    for prefix, seeds, sets in groups:
        for seed in seeds:
            for name in sets:
                stem = out / "eval" / f"{prefix}seed{seed}-{name}"
                paths += [stem.with_suffix(".json"), Path(f"{stem}-predictions.jsonl")]
    return paths


def describe_files(
    data_dir: Path, sets: dict[str, list[dict[str, Any]]], train_records: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Path / n / sha256 (or filter) of every evaluated set and the training file."""
    files = {name: {"n": len(recs)} for name, recs in sets.items()} | {
        "train": {"path": str(data_dir / "train.jsonl"), "n": len(train_records)}
    }
    files["train"]["sha256"] = sha256(data_dir / "train.jsonl")
    for name, fname in (("test", "test.jsonl"), ("probe", "probe-v1-eval-only.jsonl")):
        files[name] |= {"path": str(data_dir / fname), "sha256": sha256(data_dir / fname)}
    test_path = files["test"]["path"]
    files["test-v1"] |= {"path": test_path, "filter": f"source_batch != {TARGETED_BATCH}"}
    files["test-targeted"] |= {"path": test_path, "filter": f"source_batch == {TARGETED_BATCH}"}
    files["validation"] |= {
        "path": str(data_dir / "validation.jsonl"),
        "sha256": sha256(data_dir / "validation.jsonl"),
    }
    return files


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- scoring


def score_model(
    *,
    version: str,
    seed: int,
    ckpt: Path,
    sets: dict[str, list[dict[str, Any]]],
    set_names: tuple[str, ...],
    train_records: list[dict[str, Any]],
    seen_texts: set[str],
    categories: Callable[[str], Any],
    flags: dict[str, list[set[str]]],
    out: Path,
    files: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Evaluate one checkpoint on ``set_names``; write its eval files; summaries and rows by set."""
    device = torch.device("cpu")
    if version == "v2":
        model, tokenizer, meta = load_value_checkpoint(ckpt)
        crf = value_crf(model)

        def logits_fn(data):
            return torch_logits(model, data, device)

    else:
        model, tokenizer, meta = load_checkpoint(ckpt)
        crf = None

        def logits_fn(data):
            return v1_logits(model, data, device)

    max_length = int(meta["max_length"])
    backend = {
        "version": version,
        "seed": seed,
        "checkpoint": str(ckpt),
        "model_sha256": sha256(ckpt / WEIGHTS_FILE),
        "epochs": meta.get("epochs"),
    }
    summaries: dict[str, dict[str, Any]] = {}
    all_rows: dict[str, list[dict[str, Any]]] = {}
    for name in set_names:
        records = sets[name]
        metrics, rows, _ = evaluate_records(
            records,
            logits_fn,
            tokenizer,
            max_length,
            train_records=train_records,
            categories=categories,
            crf=crf,
        )
        if version == "v1":
            report_metrics, rows = strip_value(metrics, rows)
        else:
            report_metrics = metrics
        summary = summarize_set(
            metrics,
            rows,
            records,
            with_value=version == "v2",
            seen_texts=seen_texts,
            regression_flags=flags.get(name),
        )
        stem = out / "eval" / f"{'v1-' if version == 'v1' else ''}seed{seed}-{name}"
        write_json(
            stem.with_suffix(".json"),
            {
                "backend": backend,
                "set": name,
                "monitoring_only": name == "validation",
                "records": files[name],
                "train": files["train"],
                "summary": summary,
                "metrics": report_metrics,
            },
        )
        write_jsonl(Path(f"{stem}-predictions.jsonl"), rows, overwrite=True)
        summaries[name] = summary
        all_rows[name] = rows
        print(
            f"  {version} seed{seed} {name:<13} n={summary['n']:<4} "
            f"type acc {summary['type_accuracy']:.4f} target exact {summary['target_exact']:.4f} "
            f"joint {summary['type_target_joint']:.4f}"
            + (f" value exact {summary['value']['exact']:.4f}" if "value" in summary else "")
        )
    return summaries, all_rows


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--v2-dir", type=Path, default=Path("models/value-span-v1"))
    parser.add_argument("--experiment", default="value-span-v1", help="label in the reports")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--v1-dir", type=Path, default=V1_DIR)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--probe-notes", type=Path, default=PROBE_NOTES)
    parser.add_argument("--v1-only", action="store_true", help="score the v1 baseline only")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    seeds = SEEDS
    v2_ckpts = {s: args.v2_dir / f"seed{s}" for s in seeds}
    v1_ckpts = {s: args.v1_dir / f"seed{s}" for s in seeds}
    for s in seeds:
        if not (v1_ckpts[s] / WEIGHTS_FILE).is_file():
            raise SystemExit(f"{v1_ckpts[s]}: no v1 checkpoint")
    v2_meta = {} if args.v1_only else {s: check_v2_checkpoint(v2_ckpts[s], s) for s in seeds}
    if not args.overwrite:
        existing = [p for p in output_paths(args.out, seeds, seeds, args.v1_only) if p.exists()]
        if existing:
            raise SystemExit(
                f"refusing to overwrite {existing[0]} (+{len(existing) - 1} more); pass --overwrite"
            )

    sets = load_sets(args.data_dir)
    train_records = read_jsonl(args.data_dir / "train.jsonl")
    seen_texts = seen_value_texts(train_records)
    categories = value_span_categories()
    rules = load_golden_rules()
    patterns = {n["id"]: n["pattern"] for n in read_jsonl(args.probe_notes)}

    flags: dict[str, list[set[str]]] = {}
    for name in REGRESSION_SETS:
        if name == "probe":
            flags[name] = [tag_flags(r, patterns[r["id"]]) for r in sets[name]]
        else:
            flags[name] = [regex_flags(r, rules) for r in sets[name]]
    slice_counts = {
        name: {s: sum(s in f for f in flags[name]) for s in REGRESSION_SLICES}
        for name in REGRESSION_SETS
    }
    probe_regex = [regex_flags(r, rules) for r in sets["probe"]]
    regex_vs_tags = {
        s: {
            "tag_n": sum(s in f for f in flags["probe"]),
            "regex_n": sum(s in f for f in probe_regex),
            "both": sum(
                s in a and s in b for a, b in zip(flags["probe"], probe_regex, strict=True)
            ),
        }
        for s in REGRESSION_SLICES
    }
    files = describe_files(args.data_dir, sets, train_records)

    v1_summaries: dict[int, dict[str, Any]] = {}
    v1_rows: dict[int, dict[str, list]] = {}
    print("v1 baseline (frozen compression-v3 K2048, type + target)")
    for s in seeds:
        v1_summaries[s], v1_rows[s] = score_model(
            version="v1",
            seed=s,
            ckpt=v1_ckpts[s],
            sets=sets,
            set_names=V1_SETS,
            train_records=train_records,
            seen_texts=seen_texts,
            categories=categories,
            flags=flags,
            out=args.out,
            files=files,
        )

    protocol = {"path": str(args.protocol), "sha256": sha256(args.protocol)}
    v1_stats = {name: headline_stats({s: v1_summaries[s][name] for s in seeds}) for name in V1_SETS}
    if args.v1_only:
        report = {
            "experiment": args.experiment,
            "mode": "v1-only",
            "protocol": protocol,
            "sets": files,
            "checkpoints": {s: str(v1_ckpts[s]) for s in seeds},
            "slice_definitions": _slice_definitions(),
            "regression_slice_counts": slice_counts,
            "regex_vs_probe_tags": regex_vs_tags,
            "v1": v1_stats,
            "per_seed": {"v1": v1_summaries},
        }
        write_json(args.out / "v1-baseline.json", report)
        print(f"v1 baseline -> {args.out / 'v1-baseline.json'}")
        return 0

    v2_summaries: dict[int, dict[str, Any]] = {}
    v2_rows: dict[int, dict[str, list]] = {}
    print(f"v2 {args.experiment}")
    for s in seeds:
        v2_summaries[s], v2_rows[s] = score_model(
            version="v2",
            seed=s,
            ckpt=v2_ckpts[s],
            sets=sets,
            set_names=tuple(SETS),
            train_records=train_records,
            seen_texts=seen_texts,
            categories=categories,
            flags=flags,
            out=args.out,
            files=files,
        )

    errors = [
        e
        for s in seeds
        for name in ("test", "probe")
        for e in multi_number_errors(name, s, v2_rows[s][name])
    ]
    write_jsonl(args.out / "multi-number-errors.jsonl", errors, overwrite=True)

    gate = evaluate_gate(v2_summaries, v1_summaries, seeds=seeds, export_seed=EXPORT_SEED)
    report = {
        "experiment": args.experiment,
        "protocol": protocol,
        "sets": files,
        "checkpoints": {
            "v1": {s: str(v1_ckpts[s]) for s in seeds},
            "v2": {s: {"path": str(v2_ckpts[s]), "epochs": v2_meta[s]["epochs"]} for s in seeds},
        },
        "verdict": gate,
        "v2": {name: headline_stats({s: v2_summaries[s][name] for s in seeds}) for name in SETS},
        "v1": v1_stats,
        "value_slices": {
            "test": value_slice_table({s: v2_summaries[s]["test"] for s in seeds}),
            "test_human_only": value_slice_table(
                {s: v2_summaries[s]["test"] for s in seeds}, "value_slices_human"
            ),
            "probe": value_slice_table({s: v2_summaries[s]["probe"] for s in seeds}),
            "probe_human_only": value_slice_table(
                {s: v2_summaries[s]["probe"] for s in seeds}, "value_slices_human"
            ),
        },
        "slice_definitions": _slice_definitions(),
        "regression_slice_counts": slice_counts,
        "regex_vs_probe_tags": regex_vs_tags,
        "regression_table": {
            name: regression_table(v2_summaries, v1_summaries, list(seeds), name)
            for name in REGRESSION_SETS
        },
        "changes_vs_v1": {
            f"seed{s}": {
                name: changes_vs_v1(v1_rows[s][name], v2_rows[s][name]) for name in REGRESSION_SETS
            }
            for s in seeds
        },
        "multi_number": {
            "errors_file": str(args.out / "multi-number-errors.jsonl"),
            "n_errors": {
                f"seed{s}": {
                    name: sum(1 for e in errors if e["seed"] == s and e["set"] == name)
                    for name in ("test", "probe")
                }
                for s in seeds
            },
            "n_notes": {
                name: v2_summaries[seeds[0]][name]["value_slices"]["multi_number"]["n"]
                for name in ("test", "probe")
            },
        },
        "per_seed": {"v1": v1_summaries, "v2": v2_summaries},
        "monitoring_only": ["validation"],
    }
    write_json(args.out / "comparison.json", report)
    _print_verdict(gate)
    print(f"comparison -> {args.out / 'comparison.json'}")
    return 0


def _slice_definitions() -> dict[str, Any]:
    return {
        "value": {
            "explicit_unit": "propose_value category",
            "bare_number": "propose_value category",
            "multi_number": "propose_value category",
            "slang": "propose_value category (xi, chai, lit, cu, ty)",
            "unaccented": "record accented == false",
            "unseen_span": "gold value text (exact string) not among the complete train values",
            "long_multi_token": "gold value covers > 2 model tokens or contains a space",
            "no_amount": "complete record with value null",
            "scope": "value_status == complete; *_human_only: value_provenance == human",
        },
        "regression": {
            "probe": "pattern tag of datasets/probe-v1/notes.jsonl: "
            + "; ".join(f"{k} = {sorted(v)}" for k, v in PROBE_TAGS.items())
            + "; unaccented = record accented == false",
            "test-v1": "golden-suite rules (scripts/build_golden_suite.py): lender_first = "
            "borrow_with_cho_vay_phrase; title_name = title_excluded_from_target or "
            "title_included_in_target; insurance = insurance; loan_installment = type "
            f"repayment_out and /{LOAN_INSTALLMENT.pattern}/i; "
            "unaccented = record accented == false",
        },
    }


def _print_verdict(gate: dict[str, Any]) -> None:
    print(f"\nVERDICT {gate['verdict']}: {gate['meaning']}")
    for group, key in (("value_quality", "criteria"), ("regression", "criteria")):
        print(f"  {group}")
        for c in gate[group][key]:
            value = "n/a" if c["value"] is None else f"{c['value']:.4f}"
            print(
                f"    [{'PASS' if c['pass'] else 'FAIL'}] {c['id']:<34} {c['scope']:<6} "
                f"{value} {c['comparison']} {c['threshold']}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
