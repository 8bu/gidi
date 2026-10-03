#!/usr/bin/env python
"""Train only a new value BIO head on the frozen gidi-finance-v1 seed-1 model (value-span-v3/v4).

Usage (real training; seeds 1-3 -> models/value-span-v3-frozen-v1/seed{n}):
    uv run python scripts/train_value_head.py \\
        --train datasets/annotation-v2/training-v1/train.jsonl \\
        --validation datasets/annotation-v2/training-v1/validation.jsonl \\
        --seed 1 --seed 2 --seed 3

The base checkpoint (encoder, type head, target head) is frozen and run in eval mode; the seed
only sets the value-head init and the batch order. Refuses to overwrite an existing seed dir
without ``--overwrite``. ``--max-steps`` is for smoke runs.
"""

from __future__ import annotations

import argparse
import json
import logging

from gidi.modeling.value import VALUE_HEAD_ARCHS
from gidi.training.train_value_head import (
    DEFAULT_BASE,
    DEFAULT_OUT_DIR,
    ValueHeadConfig,
    train_value_head,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    d = ValueHeadConfig(seed=1, train_file="")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", required=True, help="merged annotation-v2 training records")
    parser.add_argument("--validation", default=None, help="optional, logged per epoch only")
    parser.add_argument("--base", default=DEFAULT_BASE, help="frozen v1 checkpoint dir")
    parser.add_argument("--seed", type=int, action="append", help="repeatable (default 1)")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--experiment", default=d.experiment, help="label in metrics and meta")
    parser.add_argument("--head-arch", default=d.head_arch, choices=VALUE_HEAD_ARCHS)
    parser.add_argument("--epochs", type=int, default=d.epochs)
    parser.add_argument("--lr", type=float, default=d.lr)
    parser.add_argument(
        "--adapter-lr", type=float, default=None, help="value adapter lr (default: --lr)"
    )
    parser.add_argument(
        "--encoder-lr", type=float, default=None, help="value encoder lr (default: --lr)"
    )
    parser.add_argument("--batch-size", type=int, default=d.batch_size)
    parser.add_argument("--device", default=d.device, choices=("auto", "mps", "cuda", "cpu"))
    parser.add_argument("--max-steps", type=int, default=None, help="smoke check: stop early")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing seed dir")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "httpcore", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    summaries = []
    for seed in args.seed or [1]:
        cfg = ValueHeadConfig(
            seed=seed,
            train_file=args.train,
            validation_file=args.validation,
            base=args.base,
            experiment=args.experiment,
            head_arch=args.head_arch,
            lr=args.lr,
            adapter_lr=args.adapter_lr,
            encoder_lr=args.encoder_lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            out_dir=args.out_dir,
            max_steps=args.max_steps,
            device=args.device,
            overwrite=args.overwrite,
        )
        summaries.append(train_value_head(cfg))
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
