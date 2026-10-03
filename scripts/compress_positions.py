#!/usr/bin/env python
"""Truncate a checkpoint's position-embedding table and verify it changes nothing.

Usage:
    uv run python scripts/compress_positions.py \\
        --checkpoint models/distillation-v2/supervised/student-4x768-pretrained/seed1 \\
        --out models/compression-v1/positions/student-4x768-pretrained/seed1 \\
        --report experiments/compression-v1/positions/verify-seed1.json

Loads ``--checkpoint``, keeps the first ``max_length + pad_id + 1`` position rows, writes a new
checkpoint to ``--out`` and compares it with the original on train + test + probe (tokenization,
raw logits, predicted types and spans, all float32 CPU). Exits non-zero if predictions differ.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import torch
import transformers
from evaluate_probe import load_probe
from transformers.models.roberta import modeling_roberta

from gidi.compression.positions import required_rows, truncate_positions
from gidi.modeling.checkpoint import WEIGHTS_FILE, load_checkpoint, save_checkpoint
from gidi.modeling.model import GidiMultiTaskModel
from gidi.training.train import Prepared, load_split, predict, prepare

TRAIN = Path("datasets/annotation-v1/distillation-v1/train.jsonl")
TEST = Path("datasets/annotation-v1/distillation-v1/test.jsonl")
PROBE = Path("datasets/probe-v1")
BATCH = 64


def logits(model: GidiMultiTaskModel, data: Prepared) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Type logits ``(N, types)`` and per-record tag logits ``(len_i, tags)`` (padding trimmed)."""
    enc = data.encoded
    type_rows: list[torch.Tensor] = []
    tag_rows: list[torch.Tensor] = []
    with torch.inference_mode():
        for i in range(0, len(data), BATCH):
            ids = enc.input_ids[i : i + BATCH]
            mask = enc.attention_mask[i : i + BATCH]
            width = int(mask.sum(dim=1).max())
            type_logits, tag_logits = model(ids[:, :width], mask[:, :width])
            type_rows.append(type_logits)
            for j in range(ids.shape[0]):
                tag_rows.append(tag_logits[j, : int(mask[j].sum())])
    return torch.cat(type_rows), tag_rows


def specials_counted(tokenizer: Any, max_length: int) -> dict[str, Any]:
    """Tokenizer truncation includes ``<s>``/``</s>``: an over-long text yields max_length ids."""
    ids = tokenizer("tra lai " * 40, truncation=True, max_length=max_length)["input_ids"]
    ok = len(ids) == max_length and ids[0] == tokenizer.bos_token_id
    ok = ok and ids[-1] == tokenizer.eos_token_id
    if not ok:
        raise RuntimeError(
            f"truncated sequence is {len(ids)} ids, expected {max_length} with specials"
        )
    return {"length": len(ids), "first": ids[0], "last": ids[-1]}


def position_evidence(
    max_length: int, pad_id: int, splits: dict[str, Prepared], tokenizer: Any
) -> dict[str, Any]:
    """Where RoBERTa's position indexing lives, and the largest index the data actually uses."""
    fn = modeling_roberta.RobertaEmbeddings.create_position_ids_from_input_ids
    lines, first = inspect.getsourcelines(fn)
    logic = [
        {"line": first + k, "code": line.strip()}
        for k, line in enumerate(lines)
        if "mask =" in line or "incremental_indices" in line
    ]
    max_index = 0
    per_split: dict[str, Any] = {}
    for name, data in splits.items():
        enc = data.encoded
        pos = fn(enc.input_ids, pad_id)
        lengths = enc.attention_mask.sum(dim=1)
        max_index = max(max_index, int(pos.max()))
        per_split[name] = {
            "n": len(data),
            "max_tokens_incl_specials": int(lengths.max()),
            "n_at_max_length": int((lengths == max_length).sum()),
            "n_truncated": int(sum(enc.truncated)),
            "max_position_index": int(pos.max()),
        }
    return {
        "file": str(Path(modeling_roberta.__file__).resolve()),
        "transformers_version": transformers.__version__,
        "function": fn.__qualname__,
        "function_first_line": first,
        "logic": logic,
        "rule": "non-pad tokens: padding_idx + cumsum(non_pad_mask); pad slots: padding_idx",
        "specials_count_toward_max_length": specials_counted(tokenizer, max_length),
        "rows_needed": required_rows(max_length, pad_id),
        "max_index_possible": max_length + pad_id,
        "max_index_in_data": max_index,
        "splits": per_split,
    }


def diagnose(
    texts: list[str], mismatches: list[int], a: torch.Tensor, b: torch.Tensor
) -> list[dict[str, Any]]:
    return [
        {"index": i, "text": texts[i], "type_logit_max_abs_diff": float((a[i] - b[i]).abs().max())}
        for i in mismatches[:20]
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if (args.out.exists() and any(args.out.iterdir()) or args.report.exists()) and (
        not args.overwrite
    ):
        raise SystemExit(f"refusing to overwrite {args.out} / {args.report} (pass --overwrite)")

    model, tokenizer, meta = load_checkpoint(args.checkpoint)
    pad_id = int(model.encoder.config.pad_token_id)
    rows_before = int(model.encoder.config.max_position_embeddings)
    new_model = truncate_positions(model, args.max_length)
    rows_after = int(new_model.encoder.config.max_position_embeddings)
    meta = {
        **meta,
        "compression": {**meta.get("compression", {})}
        | {"positions": {"rows_before": rows_before, "rows_after": rows_after}},
    }
    save_checkpoint(new_model, tokenizer, args.out, meta)
    loaded, loaded_tokenizer, _ = load_checkpoint(args.out)

    records = {
        "train": load_split(TRAIN),
        "test": load_split(TEST),
        "probe": load_probe(PROBE),
    }
    original = {k: prepare(tokenizer, v, args.max_length) for k, v in records.items()}
    reloaded = {k: prepare(loaded_tokenizer, v, args.max_length) for k, v in records.items()}

    report: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "out": str(args.out),
        "max_length": args.max_length,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "position_indexing": position_evidence(args.max_length, pad_id, original, tokenizer),
        "splits": {},
    }
    before_w, after_w = args.checkpoint / WEIGHTS_FILE, args.out / WEIGHTS_FILE
    params_before, params_after = model.param_count(), loaded.param_count()
    report["params_before"] = params_before
    report["params_after"] = params_after
    report["params_saved"] = params_before - params_after
    report["fp32_safetensors_mb_before"] = before_w.stat().st_size / 1e6
    report["fp32_safetensors_mb_after"] = after_w.stat().st_size / 1e6
    report["fp32_safetensors_mb_saved"] = (
        report["fp32_safetensors_mb_before"] - report["fp32_safetensors_mb_after"]
    )

    all_ok = True
    device = torch.device("cpu")
    for name in records:
        a, b = original[name], reloaded[name]
        tok_same = (
            torch.equal(a.encoded.input_ids, b.encoded.input_ids)
            and torch.equal(a.encoded.attention_mask, b.encoded.attention_mask)
            and a.encoded.offsets == b.encoded.offsets
        )
        type_a, tags_a = logits(model, a)
        type_b, tags_b = logits(loaded, b)
        type_diff = float((type_a - type_b).abs().max())
        tag_diff = max(float((x - y).abs().max()) for x, y in zip(tags_a, tags_b, strict=True))
        bitwise = torch.equal(type_a, type_b) and all(
            torch.equal(x, y) for x, y in zip(tags_a, tags_b, strict=True)
        )
        types_a, spans_a = predict(model, a, device)
        types_b, spans_b = predict(loaded, b, device)
        type_mismatch = [i for i, (x, y) in enumerate(zip(types_a, types_b, strict=True)) if x != y]
        span_mismatch = [i for i, (x, y) in enumerate(zip(spans_a, spans_b, strict=True)) if x != y]
        identical = tok_same and not type_mismatch and not span_mismatch
        all_ok &= identical
        report["splits"][name] = {
            "n": len(a),
            "tokenization_identical": tok_same,
            "type_logits_max_abs_diff": type_diff,
            "tag_logits_max_abs_diff": tag_diff,
            "logits_bitwise_equal": bitwise,
            "types_identical": not type_mismatch,
            "spans_identical": not span_mismatch,
            "n_type_mismatch": len(type_mismatch),
            "n_span_mismatch": len(span_mismatch),
        }
        if not identical:
            texts = [r["text"] for r in records[name]]
            report["splits"][name]["diagnosis"] = {
                "type_mismatches": diagnose(texts, type_mismatch, type_a, type_b),
                "span_mismatch_indices": span_mismatch[:20],
            }
    report["n_texts_verified"] = sum(len(v) for v in records.values())
    report["all_identical"] = all_ok
    report["environment"] = {"torch": torch.__version__, "transformers": transformers.__version__}

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    mb_before, mb_after = report["fp32_safetensors_mb_before"], report["fp32_safetensors_mb_after"]
    print(
        f"rows {rows_before} -> {rows_after}; params {params_before:,} -> {params_after:,} "
        f"(-{params_before - params_after:,}); fp32 safetensors "
        f"{mb_before:.2f} -> {mb_after:.2f} MB"
    )
    for name, s in report["splits"].items():
        print(
            f"{name:<6} n={s['n']:<4} tok_identical={s['tokenization_identical']} "
            f"type_diff={s['type_logits_max_abs_diff']:.3g} "
            f"tag_diff={s['tag_logits_max_abs_diff']:.3g} "
            f"bitwise={s['logits_bitwise_equal']} types_same={s['types_identical']} "
            f"spans_same={s['spans_identical']}"
        )
    print(f"report -> {args.report}")
    if not all_ok:
        print("DIAGNOSIS: truncated model is not equivalent to the original:", file=sys.stderr)
        for name, s in report["splits"].items():
            if "diagnosis" in s:
                print(
                    f"  {name}: {json.dumps(s['diagnosis'], ensure_ascii=False)}", file=sys.stderr
                )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
