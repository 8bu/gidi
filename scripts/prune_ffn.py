#!/usr/bin/env python
"""Analyze and build a structured FFN-pruning neuron map (compression-v3).

Usage:
    uv run python scripts/prune_ffn.py --criterion combined --k 2048 \\
        [--vocab-spec models/compression-v1/vocab/B-rank-8000] \\
        [--map-root models/compression-v3/ffn] [--out experiments/compression-v3/ffn/analysis.json]

Builds the compressed pretrained initial model of the compression-v1 student (pretrained
BamiBERT 4x768, position table truncated, vocabulary pruned; the encoder does not depend on the
seed) and scores every FFN neuron with each criterion of :mod:`gidi.compression.ffn` on the 723
TRAINING notes only. Writes:

* ``--out``: the parameter budget, INT8 size estimates per width, and per criterion and width the
  share of FFN-output energy the removed neurons carry on the training tokens (a train-only
  proxy, lower is better) and the overlap between criteria.
* ``<map-root>/<criterion>-K<k>/ffn_map.json``: the kept neuron indices of every layer for the
  chosen ``--criterion`` and ``--k``, plus their scores and provenance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gidi.compression.ffn import CRITERIA, neuron_scores, params_per_neuron, select_neurons
from gidi.compression.vocab import load_vocab_spec
from gidi.distillation.targets import sha256_file
from gidi.distillation.train_student import DEFAULT_SPLITS_DIR, StudentTrainConfig, initial_model
from gidi.training.train import load_split, prepare

VOCAB_SPEC = "models/compression-v1/vocab/B-rank-8000"
WIDTHS = (2816, 2560, 2304, 2048, 1792, 1536)
# Measured compression-v1 4x768 INT8 ONNX (experiments/compression-v1) and the INT8 bytes per
# removed transformer parameter measured by compression-v2 (4x768 -> 3x768).
BASE_INT8_BYTES = 34_966_389
INT8_BYTES_PER_PARAM = (34_966_389 - 27_838_369) / 7_087_872


def budget(model) -> dict:
    groups = {
        "embeddings": 0,
        "attention": 0,
        "ffn": 0,
        "layer_norms": 0,
        "heads": 0,
    }
    for name, p in model.named_parameters():
        n = p.numel()
        if "LayerNorm" in name:
            groups["layer_norms"] += n
        elif ".embeddings." in name:
            groups["embeddings"] += n
        elif ".attention." in name:
            groups["attention"] += n
        elif ".intermediate." in name or ".output.dense" in name:
            groups["ffn"] += n
        else:
            groups["heads"] += n
    total = sum(groups.values())
    return {
        "total": total,
        "groups": {k: {"params": v, "share": v / total} for k, v in groups.items()},
    }


@torch.no_grad()
def removed_energy(model, ids, mask, keeps: dict, batch_size: int = 64) -> dict:
    """Per policy and layer: ‖removed-neuron contribution‖² / ‖full FFN contribution‖²."""
    layers = list(model.encoder.encoder.layer)
    num = {p: [0.0] * len(layers) for p in keeps}
    den = [0.0] * len(layers)
    box: list[torch.Tensor] = []

    def hook(i):
        def fn(_m, _inp, out):
            act = out.reshape(-1, out.shape[-1])[box[0].reshape(-1)]
            w2 = layers[i].output.dense.weight  # [H, I]
            den[i] += float((act @ w2.T).pow(2).sum())
            for policy, dropped in keeps.items():
                drop = dropped[i]
                part = act[:, drop] @ w2[:, drop].T
                num[policy][i] += float(part.pow(2).sum())

        return fn

    handles = [layer.intermediate.register_forward_hook(hook(i)) for i, layer in enumerate(layers)]
    try:
        for s in range(0, len(ids), batch_size):
            box[:] = [mask[s : s + batch_size].bool()]
            m = mask[s : s + batch_size]
            model.encoder(input_ids=ids[s : s + batch_size], attention_mask=m)
    finally:
        for h in handles:
            h.remove()
    return {p: [n / d for n, d in zip(v, den, strict=True)] for p, v in num.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--criterion", required=True, choices=CRITERIA)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--vocab-spec", default=VOCAB_SPEC)
    parser.add_argument("--splits-dir", default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--map-root", type=Path, default=Path("models/compression-v3/ffn"))
    parser.add_argument(
        "--out", type=Path, default=Path("experiments/compression-v3/ffn/analysis.json")
    )
    args = parser.parse_args(argv)
    torch.set_grad_enabled(False)

    cfg = StudentTrainConfig(
        arm="supervised",
        student="student-4x768",
        init="pretrained",
        truncate_positions=True,
        vocab_spec=args.vocab_spec,
        seed=1,
    )
    tokenizer, kept_old_ids = load_vocab_spec(args.vocab_spec)
    model, init_report, compression = initial_model(cfg, kept_old_ids)
    model.eval()
    train_path = Path(args.splits_dir) / "train.jsonl"
    records = load_split(train_path)
    enc = prepare(tokenizer, records, cfg.max_length).encoded
    ids, mask = enc.input_ids, enc.attention_mask

    hidden = int(model.encoder.config.hidden_size)
    size = int(model.encoder.config.intermediate_size)
    n_layers = int(model.encoder.config.num_hidden_layers)
    scores = neuron_scores(model, ids, mask)

    keeps_all = {
        (c, k): [select_neurons(layer[c], k) for layer in scores] for c in CRITERIA for k in WIDTHS
    }
    if (args.criterion, args.k) not in keeps_all:
        keeps_all[(args.criterion, args.k)] = [
            select_neurons(s[args.criterion], args.k) for s in scores
        ]
    dropped = {
        f"{c}-K{k}": [sorted(set(range(size)) - set(kl)) for kl in keep]
        for (c, k), keep in keeps_all.items()
    }
    energy = removed_energy(model, ids, mask, dropped)

    per_neuron = params_per_neuron(hidden)
    widths = {}
    for k in sorted({k for _, k in keeps_all}, reverse=True):
        removed = (size - k) * per_neuron * n_layers
        widths[str(k)] = {
            "ffn_kept_share": k / size,
            "params_removed": removed,
            "params_after": model.param_count() - removed,
            "est_int8_mb": (BASE_INT8_BYTES - removed * INT8_BYTES_PER_PARAM) / 1e6,
        }
    criteria = {}
    for c, k in sorted(keeps_all, key=lambda x: (x[0], -x[1])):
        e = energy[f"{c}-K{k}"]
        criteria.setdefault(c, {})[str(k)] = {
            "removed_energy_per_layer": e,
            "removed_energy_mean": sum(e) / len(e),
        }
    overlap = {}
    for k in WIDTHS:
        for i, a in enumerate(CRITERIA):
            for b in CRITERIA[i + 1 :]:
                ka, kb = keeps_all[(a, k)], keeps_all[(b, k)]
                overlap[f"K{k}:{a}~{b}"] = sum(
                    len(set(x) & set(y)) for x, y in zip(ka, kb, strict=True)
                ) / (k * n_layers)

    analysis = {
        "model": "compression-v1 initial model: pretrained BamiBERT 4x768 + pos34 + vocab",
        "vocab_spec": args.vocab_spec,
        "layer_map": init_report["layer_map"],
        "scored_on": {
            "split": str(train_path),
            "sha256": sha256_file(train_path),
            "n_notes": len(records),
            "n_tokens": int(mask.sum()),
        },
        "budget": budget(model),
        "params_per_neuron_per_layer": per_neuron,
        "int8_estimate": {
            "base_int8_bytes": BASE_INT8_BYTES,
            "bytes_per_removed_param": INT8_BYTES_PER_PARAM,
        },
        "widths": widths,
        "criteria": criteria,
        "overlap_kept_sets": overlap,
        "chosen": {"criterion": args.criterion, "k": args.k},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(analysis, indent=1) + "\n", encoding="utf-8")

    keep = keeps_all[(args.criterion, args.k)]
    map_dir = args.map_root / f"{args.criterion}-K{args.k}"
    map_dir.mkdir(parents=True, exist_ok=True)
    ffn_map = {
        "criterion": args.criterion,
        "k": args.k,
        "intermediate_before": size,
        "base": {
            "student": cfg.student,
            "init": cfg.init,
            "max_length": cfg.max_length,
            "truncate_positions": cfg.truncate_positions,
            "vocab_map_sha256": compression["vocab"]["map_sha256"],
        },
        "layer_map": init_report["layer_map"],
        "scored_on": analysis["scored_on"],
        "keep": keep,
        "kept_scores": [
            [round(float(s[args.criterion][j]), 6) for j in kl]
            for s, kl in zip(scores, keep, strict=True)
        ],
        "removed_energy_per_layer": energy[f"{args.criterion}-K{args.k}"],
    }
    (map_dir / "ffn_map.json").write_text(json.dumps(ffn_map) + "\n", encoding="utf-8")
    for k, w in widths.items():
        print(f"K={k}: params {w['params_after']:,}  est INT8 {w['est_int8_mb']:.2f} MB")
    for c, per_k in criteria.items():
        print(c, {k: round(v["removed_energy_mean"], 4) for k, v in per_k.items()})
    print(f"map -> {map_dir / 'ffn_map.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
