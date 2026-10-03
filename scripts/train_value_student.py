#!/usr/bin/env python
"""Train the value-span model (type + target + value heads), one run per seed.

Usage (real training; seeds 1-3 -> models/value-span-v1/seed{n}):
    uv run python scripts/train_value_student.py --train <merged-train.jsonl> \\
        --validation <merged-validation.jsonl> --seed 1 --seed 2 --seed 3 \\
        --out-dir experiments/value-span-v1/runs --weights-dir models/value-span-v1

The model starts from the compression-v3 / gidi-finance-v1 initial model of each seed (pretrained
BamiBERT 4x768 layers 2,5,8,11, 34 position rows, vocabulary B-rank-8000, FFN 2048, seeded fresh
type/target heads) plus a seeded value head, and trains with the compression-v3 recipe (encoder lr
5e-5, head lr 1e-3, batch 8, 40 epochs, warmup 0.1, weight decay 0.01, dropout 0.1, max grad norm
1.0, last epoch kept). ``--train`` is a merged annotation-v2 records file (v1 fields plus
``value``, ``value_status``, ``value_provenance``); records whose ``value_status`` is not
``complete`` still train type/target and are masked in the value loss. ``--validation`` and
``--test`` are optional evaluation files (never selected on). ``--max-steps`` is for smoke runs.
"""

from __future__ import annotations

import argparse
import json
import logging

from gidi.training.train_value import (
    DEFAULT_FFN_MAP,
    DEFAULT_OUT_DIR,
    DEFAULT_VOCAB_SPEC,
    DEFAULT_WEIGHTS_DIR,
    ValueTrainConfig,
    train_value,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    d = ValueTrainConfig(seed=1, train_file="")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", required=True, help="merged annotation-v2 training records")
    parser.add_argument("--validation", default=None, help="optional, logged per epoch only")
    parser.add_argument("--test", default=None, help="optional, scored once after training")
    parser.add_argument("--seed", type=int, action="append", help="repeatable (default 1)")
    parser.add_argument("--vocab-spec", default=DEFAULT_VOCAB_SPEC)
    parser.add_argument("--ffn-map", default=DEFAULT_FFN_MAP)
    parser.add_argument("--epochs", type=int, default=d.epochs)
    parser.add_argument("--lr", type=float, default=d.lr, help="encoder learning rate")
    parser.add_argument("--head-lr", type=float, default=d.head_lr, help="lr of the three heads")
    parser.add_argument("--batch-size", type=int, default=d.batch_size)
    parser.add_argument("--warmup-ratio", type=float, default=d.warmup_ratio)
    parser.add_argument("--weight-decay", type=float, default=d.weight_decay)
    parser.add_argument("--dropout", type=float, default=d.dropout)
    parser.add_argument("--max-grad-norm", type=float, default=d.max_grad_norm)
    parser.add_argument("--max-length", type=int, default=d.max_length)
    parser.add_argument("--device", default=d.device, choices=("auto", "mps", "cuda", "cpu"))
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--weights-dir", default=DEFAULT_WEIGHTS_DIR)
    parser.add_argument("--max-steps", type=int, default=None, help="smoke check: stop early")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing run dir")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "httpcore", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    summaries = []
    for seed in args.seed or [1]:
        cfg = ValueTrainConfig(
            seed=seed,
            train_file=args.train,
            validation_file=args.validation,
            test_file=args.test,
            lr=args.lr,
            head_lr=args.head_lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            max_grad_norm=args.max_grad_norm,
            max_length=args.max_length,
            vocab_spec=args.vocab_spec,
            ffn_map=args.ffn_map,
            out_dir=args.out_dir,
            weights_dir=args.weights_dir,
            max_steps=args.max_steps,
            device=args.device,
            overwrite=args.overwrite,
        )
        summaries.append(train_value(cfg))
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
