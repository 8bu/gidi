#!/usr/bin/env python
"""Build and evaluate vocabulary-pruning policies for the compression-v1 student.

Usage:
    uv run python scripts/prune_vocab.py \\
        [--checkpoints models/distillation-v2/supervised/student-4x768-pretrained/seed1 ...] \\
        [--spec-root models/compression-v1/vocab] \\
        [--out experiments/compression-v1/vocab/policies.json] [--no-eval]

Construction uses only the BamiBERT vocab/merges (merge rank = pretraining-frequency proxy), token
script categories, and the 723 training notes. Frozen-test and probe notes are only measured:
tokenization changes, then a ZERO-TRAINING evaluation (``prune_vocab`` on each trained baseline
checkpoint + the policy tokenizer, no weight update). Writes ``policies.json`` and a spec directory
per policy (``A-conservative``, ``B-rank-*``, ``C-aggressive``; ``T-only`` is reference only and
evaluated from a temporary directory).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from analyze_vocab import (
    MAX_LENGTH,
    load_splits,
    pieces_in_span,
    show,
)
from tokenizers import Tokenizer

from gidi.compression.vocab import (
    BpeVocab,
    closure,
    decode_token,
    load_vocab_spec,
    prune_vocab,
    token_category,
    vietnamese_compatible_ids,
    write_vocab_spec,
)
from gidi.evaluation.metrics import evaluate
from gidi.modeling.checkpoint import load_checkpoint
from gidi.modeling.model import GidiMultiTaskModel
from gidi.training.train import Prepared, predict, prepare

CHECKPOINTS = tuple(
    Path(f"models/distillation-v2/supervised/student-4x768-pretrained/seed{s}") for s in (1, 2, 3)
)
BASELINE_PARAMS = 45_666_059
BASELINE_INT8_BYTES = 45_856_637
POSITION_TRUNCATION_PARAMS = 1_548_288  # 2050 -> 34 position rows
HIDDEN = 768
RANK_GRID = (12000, 10000, 8000, 6000, 4000)
AGGRESSIVE_TOP = 2000
EVAL_SPLITS = ("train", "test", "probe")

# Hand-written OOV probes (NOT from test/probe): unseen names, merchants, unaccented variants,
# amount shorthand. Each is tokenized inside "trả <s> 100k" and the pieces of <s> are shown.
OOV_STRINGS = (
    "Highlands",
    "Phúc Long",
    "Bách Hóa Xanh",
    "Grab",
    "Momo",
    "Tiki",
    "Lazada",
    "Circle K",
    "Thế Giới Di Động",
    "GS25",
    "Winmart",
    "The Coffee House",
    "Trung Nguyên",
    "Gong Cha",
    "KFC",
    "Jollibee",
    "Shopee",
    "Xanh SM",
    "Nguyễn Văn Hùng",
    "Trần Thị Lan",
    "Phạm Quỳnh Anh",
    "Đặng Thuỳ Dương",
    "Khôi",
    "Uyên",
    "Thịnh",
    "highlands coffee",
    "phuc long",
    "bach hoa xanh",
    "the gioi di dong",
    "nguyen van hung",
    "tran thi lan",
    "co Huong",
    "2tr35",
    "7tr8",
    "250k",
    "1.25tr",
    "12,500,000",
    "3 triệu rưỡi",
    "70 ngàn",
    "1tỷ2",
)


# --------------------------------------------------------------------------------------------
# Keep-set construction (train + vocab only)
# --------------------------------------------------------------------------------------------


def policy_keep_sets(bpe: BpeVocab, train_ids: set[int]) -> dict[str, list[int]]:
    mandatory = bpe.mandatory_ids()
    rank = bpe.merge_rank()
    train_closure = closure(bpe, train_ids | mandatory)
    vietnamese = vietnamese_compatible_ids(bpe)
    policies: dict[str, set[int]] = {
        "A-conservative": closure(bpe, train_closure | vietnamese),
    }
    for n in RANK_GRID:
        policies[f"B-rank-{n}"] = closure(
            bpe, train_closure | {i for i in vietnamese if rank.get(i, n) < n}
        )
    top = {i for i, r in rank.items() if r < AGGRESSIVE_TOP}
    policies["C-aggressive"] = train_closure | closure(bpe, top | mandatory)
    policies["T-only"] = train_closure
    return {name: sorted(ids) for name, ids in policies.items()}


def size_estimates(n_rows: int, old_rows: int) -> dict[str, Any]:
    removed = (old_rows - n_rows) * HIDDEN
    params_vocab_only = BASELINE_PARAMS - removed
    params_with_positions = params_vocab_only - POSITION_TRUNCATION_PARAMS
    overhead = BASELINE_INT8_BYTES - BASELINE_PARAMS
    return {
        "embedding_params": n_rows * HIDDEN,
        "total_params_vocab_only": params_vocab_only,
        "total_params_with_34_positions": params_with_positions,
        "est_int8_mb_vocab_only": (params_vocab_only + overhead) / 1e6,
        "est_int8_mb_with_34_positions": (params_with_positions + overhead) / 1e6,
    }


# --------------------------------------------------------------------------------------------
# Tokenization comparison
# --------------------------------------------------------------------------------------------


def _tokens(tokenizer, text: str, max_length: int | None) -> list[str]:
    enc = tokenizer(text, truncation=max_length is not None, max_length=max_length)
    return tokenizer.convert_ids_to_tokens(enc["input_ids"])


def tokenization_report(orig, pruned, records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    unchanged = 0
    span_total = span_unchanged = 0
    ratios, extra = [], []
    trunc_orig = trunc_new = 0
    max_len_new = 0
    changed_examples = []
    changed_ids: list[int] = []
    for i, r in enumerate(records):
        full_o = _tokens(orig, r["text"], None)
        full_n = _tokens(pruned, r["text"], None)
        trunc_orig += len(full_o) > MAX_LENGTH
        trunc_new += len(full_n) > MAX_LENGTH
        max_len_new = max(max_len_new, len(full_n))
        same = _tokens(orig, r["text"], MAX_LENGTH) == _tokens(pruned, r["text"], MAX_LENGTH)
        unchanged += same
        if not same:
            changed_ids.append(i)
            if len(changed_examples) < 8:
                changed_examples.append(
                    {"text": r["text"], "orig": show(full_o), "pruned": show(full_n)}
                )
        ratios.append(len(full_n) / len(full_o))
        extra.append(len(full_n) - len(full_o))
        if r["target"]:
            t = r["target"]
            span_total += 1
            span_unchanged += pieces_in_span(orig, r["text"], t["start"], t["end"]) == (
                pieces_in_span(pruned, r["text"], t["start"], t["end"])
            )
    n_changed = n - unchanged
    return {
        "notes": n,
        "notes_unchanged": unchanged,
        "notes_unchanged_fraction": unchanged / n,
        "target_spans": span_total,
        "target_spans_unchanged": span_unchanged,
        "target_spans_unchanged_fraction": span_unchanged / span_total,
        "mean_token_count_ratio": sum(ratios) / n,
        "mean_extra_tokens_per_note": sum(extra) / n,
        "mean_extra_tokens_per_changed_note": (
            sum(e for e in extra if e) / n_changed if n_changed else 0.0
        ),
        "notes_truncated_at_32": {"orig": trunc_orig, "pruned": trunc_new},
        "truncation_increase": trunc_new - trunc_orig,
        "max_pruned_length_with_specials": max_len_new,
        "changed_note_indices": changed_ids,
        "changed_examples": changed_examples,
    }


def exactness_report(
    orig_tok: Tokenizer, pruned_tok: Tokenizer, kept: Sequence[int], words: Sequence[str]
) -> dict[str, Any]:
    """Words whose original pieces are all kept must segment identically (mapped ids)."""
    new_of = {old: new for new, old in enumerate(kept)}
    unk = orig_tok.token_to_id("<unk>")
    unk_pruned = pruned_tok.token_to_id("<unk>")
    orig = orig_tok.encode_batch(list(words), add_special_tokens=False)
    new = pruned_tok.encode_batch(list(words), add_special_tokens=False)
    identical = mismatched = fallback = fallback_new_unk = fallback_longer = 0
    for o, p in zip(orig, new, strict=True):
        if all(i in new_of for i in o.ids):
            if p.ids == [new_of[i] for i in o.ids]:
                identical += 1
            else:
                mismatched += 1
        else:
            fallback += 1
            fallback_new_unk += p.ids.count(unk_pruned) > o.ids.count(unk)
            fallback_longer += len(p.ids) > len(o.ids)
    return {
        "words": len(words),
        "all_pieces_kept_identical": identical,
        "all_pieces_kept_mismatched": mismatched,
        "fell_back": fallback,
        "fell_back_longer": fallback_longer,
        "fell_back_with_extra_unk": fallback_new_unk,
    }


def word_sets(bpe: BpeVocab, splits: dict[str, list[dict[str, Any]]]) -> dict[str, list[str]]:
    sets: dict[str, list[str]] = {}
    for name, records in splits.items():
        words = sorted({w for r in records for w in r["text"].split()})
        sets[f"{name}_words"] = [*words, *(" " + w for w in words)]
    texts = []
    for i, tok in enumerate(bpe.tokens):
        if i in bpe.special_ids:
            continue
        text = decode_token(tok)
        if "\\x" not in text and text.isprintable():
            texts.append(text)
    sets["vocab_token_strings"] = sorted(set(texts))
    return sets


# --------------------------------------------------------------------------------------------
# Zero-training evaluation
# --------------------------------------------------------------------------------------------


def joint_flags(records, types, spans) -> list[bool]:
    flags = []
    for r, t, s in zip(records, types, spans, strict=True):
        gold = r["target"]
        gold_b = None if gold is None else (gold["start"], gold["end"])
        pred_b = None if s is None else (s["start"], s["end"])
        flags.append(t == r["type"] and gold_b == pred_b)
    return flags


def split_metrics(records, types, spans) -> dict[str, float]:
    m = evaluate(records, types, spans, slices=())["overall"]
    flags = joint_flags(records, types, spans)
    acc = [_accented(r) for r in records]
    return {
        "type_macro_f1": m["type"]["macro_f1"],
        "type_accuracy": m["type"]["accuracy"],
        "span_f1": m["target"]["f1"],
        "span_exact": m["target"]["exact_match"],
        "joint": sum(flags) / len(flags),
        "joint_accented": _rate([f for f, a in zip(flags, acc, strict=True) if a]),
        "joint_unaccented": _rate([f for f, a in zip(flags, acc, strict=True) if not a]),
    }


def _accented(record: dict[str, Any]) -> bool:
    return any(ord(c) > 127 for c in record["text"])


def _rate(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def single_logits(model: GidiMultiTaskModel, data: Prepared, idx: Sequence[int]):
    """Batch-1 raw logits for the notes in ``idx`` (no padding dependence)."""
    out = {}
    enc = data.encoded
    with torch.no_grad():
        for i in idx:
            width = int(enc.attention_mask[i].sum())
            t, g = model(enc.input_ids[i : i + 1, :width], enc.attention_mask[i : i + 1, :width])
            out[i] = (t[0], g[0])
    return out


def mean_std(values: Sequence[float | None]) -> dict[str, float | None]:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "std": None}
    return {"mean": statistics.fmean(vals), "std": statistics.pstdev(vals)}


def run_zero_training(
    checkpoints: Sequence[Path],
    splits: dict[str, list[dict[str, Any]]],
    policy_specs: dict[str, tuple[Path, dict[str, Any]]],
) -> dict[str, Any]:
    """Per policy: metrics per seed/split, prediction changes vs unpruned, logit equality."""
    result: dict[str, Any] = {"baseline": {}, "policies": {}}
    per_seed: dict[str, dict[str, Any]] = {}
    base_models: dict[str, GidiMultiTaskModel] = {}
    for ckpt in checkpoints:
        model, tokenizer, _ = load_checkpoint(ckpt)
        base_models[ckpt.name] = model
        data = {n: prepare(tokenizer, recs, MAX_LENGTH) for n, recs in splits.items()}
        preds = {n: predict(model, data[n], torch.device("cpu")) for n in splits}
        per_seed[ckpt.name] = {
            "metrics": {n: split_metrics(splits[n], *preds[n]) for n in splits},
            "preds": preds,
            "logits": {n: single_logits(model, data[n], range(len(splits[n]))) for n in splits},
        }
    result["baseline"] = {
        "per_seed": {s: v["metrics"] for s, v in per_seed.items()},
        "summary": _summary([v["metrics"] for v in per_seed.values()]),
    }
    for name, (spec_dir, tok_report) in policy_specs.items():
        tokenizer, kept = load_vocab_spec(spec_dir)
        seeds: dict[str, Any] = {}
        for seed, base in per_seed.items():
            pruned = prune_vocab(base_models[seed], kept).eval()
            metrics, changes, logit_diff = {}, {}, {}
            for split, records in splits.items():
                data = prepare(tokenizer, records, MAX_LENGTH)
                types, spans = predict(pruned, data, torch.device("cpu"))
                metrics[split] = split_metrics(records, types, spans)
                b_types, b_spans = base["preds"][split]
                b_flags = joint_flags(records, b_types, b_spans)
                p_flags = joint_flags(records, types, spans)
                changes[split] = {
                    "type_changed": sum(a != b for a, b in zip(types, b_types, strict=True)),
                    "span_changed": sum(a != b for a, b in zip(spans, b_spans, strict=True)),
                    "any_prediction_changed": sum(
                        a != b or c != d
                        for a, b, c, d in zip(types, b_types, spans, b_spans, strict=True)
                    ),
                    "joint_gained": sum(p and not b for p, b in zip(p_flags, b_flags, strict=True)),
                    "joint_lost": sum(b and not p for b, p in zip(b_flags, p_flags, strict=True)),
                }
                same = [
                    i
                    for i in range(len(records))
                    if i not in set(tok_report[split]["changed_note_indices"])
                ]
                logits = single_logits(pruned, data, same)
                diffs = [
                    max(
                        float((logits[i][0] - base["logits"][split][i][0]).abs().max()),
                        float((logits[i][1] - base["logits"][split][i][1]).abs().max()),
                    )
                    for i in same
                ]
                logit_diff[split] = {"notes_compared": len(same), "max_abs_diff": max(diffs)}
            seeds[seed] = {"metrics": metrics, "changes_vs_unpruned": changes, "logits": logit_diff}
            del pruned
        result["policies"][name] = {
            "per_seed": seeds,
            "summary": _summary([s["metrics"] for s in seeds.values()]),
            "changes_total_over_seeds": {
                split: {
                    k: sum(s["changes_vs_unpruned"][split][k] for s in seeds.values())
                    for k in next(iter(seeds.values()))["changes_vs_unpruned"][split]
                }
                for split in splits
            },
            "train_identical_to_unpruned": all(
                s["changes_vs_unpruned"]["train"]["any_prediction_changed"] == 0
                and s["logits"]["train"]["max_abs_diff"] == 0.0
                for s in seeds.values()
            ),
        }
    return result


def _summary(per_seed_metrics: list[dict[str, dict[str, float]]]) -> dict[str, Any]:
    splits = per_seed_metrics[0].keys()
    keys = per_seed_metrics[0]["train"].keys()
    return {
        split: {k: mean_std([m[split][k] for m in per_seed_metrics]) for k in keys}
        for split in splits
    }


# --------------------------------------------------------------------------------------------
# OOV probe
# --------------------------------------------------------------------------------------------


def oov_probe(orig, pruned_tokenizers: dict[str, Any], splits) -> dict[str, Any]:
    corpus = " ".join(r["text"].lower() for n in ("test", "probe") for r in splits[n])
    rows = []
    for s in OOV_STRINGS:
        text = f"trả {s} 100k"
        start = len("trả ")
        row: dict[str, Any] = {
            "string": s,
            "in_test_or_probe_text": s.lower() in corpus,
            "original": show(pieces_in_span(orig, text, start, start + len(s))),
        }
        for name, tok in pruned_tokenizers.items():
            row[name] = show(pieces_in_span(tok, text, start, start + len(s)))
        rows.append(row)
    summary = {}
    for name in pruned_tokenizers:
        changed = [r for r in rows if r[name] != r["original"]]
        summary[name] = {
            "changed": len(changed),
            "of": len(rows),
            "mean_pieces_original": sum(len(r["original"]) for r in rows) / len(rows),
            "mean_pieces_pruned": sum(len(r[name]) for r in rows) / len(rows),
        }
    return {"strings": rows, "summary": summary}


# --------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoints", type=Path, nargs="+", default=list(CHECKPOINTS))
    parser.add_argument("--spec-root", type=Path, default=Path("models/compression-v1/vocab"))
    parser.add_argument(
        "--out", type=Path, default=Path("experiments/compression-v1/vocab/policies.json")
    )
    parser.add_argument("--no-eval", action="store_true", help="skip the zero-training eval")
    args = parser.parse_args(argv)

    from gidi.modeling.tokenization import load_tokenizer

    tok_dir = args.checkpoints[0]
    tokenizer = load_tokenizer(str(tok_dir))
    bpe = BpeVocab.from_json(json.loads((tok_dir / "tokenizer.json").read_text(encoding="utf-8")))
    splits = load_splits()
    train_ids = {
        i
        for r in splits["train"]
        for i in tokenizer(r["text"], truncation=True, max_length=MAX_LENGTH)["input_ids"]
    }
    keep_sets = policy_keep_sets(bpe, train_ids)
    mandatory = bpe.mandatory_ids()
    orig_backend = tokenizer.backend_tokenizer
    words = word_sets(bpe, splits)

    spec_dirs: dict[str, Path] = {}
    tmp = tempfile.TemporaryDirectory()
    for name, kept in keep_sets.items():
        spec_dirs[name] = (Path(tmp.name) if name == "T-only" else args.spec_root) / name
        write_vocab_spec(spec_dirs[name], tok_dir, kept, policy=name)

    policies: dict[str, Any] = {}
    pruned_tokenizers: dict[str, Any] = {}
    for name, kept in keep_sets.items():
        pruned_tok, loaded = load_vocab_spec(spec_dirs[name])
        assert loaded == kept
        pruned_tokenizers[name] = pruned_tok
        cats: dict[str, int] = {}
        for i in kept:
            c = token_category(bpe.tokens[i], special=i in bpe.special_ids, base=i in bpe.base_ids)
            cats[c] = cats.get(c, 0) + 1
        policies[name] = {
            "spec_dir": None if name == "T-only" else str(spec_dirs[name]),
            "vocab_rows": len(kept),
            "removed_rows": bpe.size - len(kept),
            **size_estimates(len(kept), bpe.size),
            "kept_by_category": dict(sorted(cats.items())),
            "mandatory_kept": mandatory <= set(kept),
            "merges_kept": len(
                json.loads((spec_dirs[name] / "tokenizer.json").read_text(encoding="utf-8"))[
                    "model"
                ]["merges"]
            ),
            "tokenization": {
                split: tokenization_report(tokenizer, pruned_tok, splits[split])
                for split in EVAL_SPLITS
            },
            "exactness": {
                label: exactness_report(orig_backend, pruned_tok.backend_tokenizer, kept, ws)
                for label, ws in words.items()
            },
        }
        t = policies[name]["tokenization"]["train"]
        assert t["notes_unchanged"] == t["notes"], f"{name}: train tokenization changed"
        e = policies[name]["exactness"]
        assert all(v["all_pieces_kept_mismatched"] == 0 for v in e.values()), name
        print(
            f"{name}: rows {len(kept)} est INT8 "
            f"{policies[name]['est_int8_mb_with_34_positions']:.2f}"
            f" MB | unchanged notes test "
            f"{policies[name]['tokenization']['test']['notes_unchanged_fraction']:.3f} probe "
            f"{policies[name]['tokenization']['probe']['notes_unchanged_fraction']:.3f}"
        )

    report: dict[str, Any] = {
        "source_tokenizer": str(tok_dir / "tokenizer.json"),
        "construction": {
            "inputs": "BamiBERT vocab+merges (rank), token categories, 723 train notes only",
            "mandatory": "specials + single-byte base tokens present in the vocab "
            f"({len(mandatory)} ids; {len(bpe.missing_bytes())} byte values have no token and "
            "already map to <unk> in the baseline)",
            "train_observed_ids": len(train_ids),
            "orphans_not_produced_by_a_merge": len(bpe.orphans()),
            "orphan_handling": "none exist (every non-special, non-byte token is produced by "
            "exactly one merge); closure would drop such tokens unless a policy selected them",
            "size_model": {
                "baseline_params": BASELINE_PARAMS,
                "baseline_int8_bytes": BASELINE_INT8_BYTES,
                "position_truncation_params_removed": POSITION_TRUNCATION_PARAMS,
                "hidden": HIDDEN,
                "note": "estimate = (params + measured ONNX overhead) bytes, ~1 byte/param",
            },
            "grid": {
                "A-conservative": "M ∪ T ∪ closure(all Vietnamese-compatible tokens)",
                "B-rank-N": "M ∪ T ∪ closure(Vietnamese-compatible tokens with merge rank < N)",
                "C-aggressive": f"M ∪ T ∪ closure(merge rank < {AGGRESSIVE_TOP})",
                "T-only": "M ∪ T (reference)",
            },
        },
        "policies": policies,
    }
    report["oov_probe"] = oov_probe(tokenizer, pruned_tokenizers, splits)

    if not args.no_eval:
        specs = {n: (spec_dirs[n], policies[n]["tokenization"]) for n in keep_sets}
        zt = run_zero_training(args.checkpoints, splits, specs)
        report["zero_training"] = {
            "baseline": zt["baseline"],
            "notes": "metrics: mean/pstd over seeds",
        }
        for name, z in zt["policies"].items():
            policies[name]["zero_training"] = z
            s = z["summary"]
            print(
                f"{name}: joint test {s['test']['joint']['mean']:.3f} "
                f"probe {s['probe']['joint']['mean']:.3f} "
                f"train identical: {z['train_identical_to_unpruned']}"
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    tmp.cleanup()
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
