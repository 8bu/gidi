#!/usr/bin/env python
"""Train the shared-encoder multi-task baseline over a (lr x seed) grid, sequentially.

Usage:
    uv run python scripts/train_baseline.py --model Qualcomm-AI-Research/BamiBERT \
        --lr 3e-5 --lr 5e-5 --seed 1 --seed 2

Each run writes ``<out-dir>/<model-slug>/lr<lr>-seed<seed>/`` (config, train log, metrics,
test predictions, checkpoint summary) and, unless ``--no-save-weights``, the best weights to
``models/baseline-v1/<model-slug>/lr<lr>-seed<seed>/``. ``--max-steps`` stops each run after
that many optimizer steps (smoke check only).
"""

from __future__ import annotations

import argparse
import json
import logging

from gidi.training.train import (
    DEFAULT_OUT_DIR,
    DEFAULT_SPLITS_DIR,
    DEFAULT_WEIGHTS_DIR,
    train_grid,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="hub id or local path of the encoder")
    parser.add_argument("--lr", type=float, action="append", help="repeatable (default 5e-5)")
    parser.add_argument("--seed", type=int, action="append", help="repeatable (default 1)")
    parser.add_argument("--head-lr", type=float, default=1e-3, help="lr of the two heads")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--splits-dir", default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--weights-dir", default=DEFAULT_WEIGHTS_DIR)
    parser.add_argument("--max-steps", type=int, default=None, help="smoke check: stop early")
    parser.add_argument(
        "--stop-epoch",
        type=int,
        default=None,
        help="fixed-epoch mode: train exactly N epochs (schedule horizon stays --epochs), no "
        "early stopping, keep the last epoch's weights",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cuda", "cpu"))
    parser.add_argument(
        "--save-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save the best epoch's weights under --weights-dir (default on)",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace an existing run dir")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "httpcore", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    summaries = train_grid(
        args.model,
        args.lr or [5e-5],
        args.seed or [1],
        epochs=args.epochs,
        head_lr=args.head_lr,
        batch_size=args.batch_size,
        patience=args.patience,
        max_length=args.max_length,
        splits_dir=args.splits_dir,
        out_dir=args.out_dir,
        weights_dir=args.weights_dir,
        save_weights=args.save_weights,
        max_steps=args.max_steps,
        stop_epoch=args.stop_epoch,
        device=args.device,
        overwrite=args.overwrite,
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
