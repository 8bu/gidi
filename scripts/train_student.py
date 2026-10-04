#!/usr/bin/env python
"""Train a distillation student (supervised-only or distilled), one run per seed.

Usage:
    uv run python scripts/train_student.py --arm distilled --seed 1 --seed 2 --seed 3
    uv run python scripts/train_student.py --arm supervised --student student-4x768 \\
        --init pretrained --lr 5e-5 --out-dir experiments/distillation-v2/runs \\
        --weights-dir models/distillation-v2 --seed 1

All arms read inputs and hard labels from the teacher-target cache (``--targets``) and differ
only in the loss: ``supervised`` forces ``alpha_hard = 1``; ``distilled`` uses the KD flags
(defaults: T=2, alpha_hard=0.5, weights 1/1). ``--student`` picks the architecture and
``--init`` random or pretrained-BamiBERT initialization (see ``gidi.distillation.student``).
Each run writes ``<out-dir>/<arm>/<variant>/seed<N>/`` and
``<weights-dir>/<arm>/<variant>/seed<N>/``. The epoch budget is fixed and the last epoch's
weights are kept. ``--max-steps`` stops each run after that many optimizer steps (smoke only).
"""

from __future__ import annotations

import argparse
import json
import logging

from gidi.distillation.loss import KDConfig
from gidi.distillation.student import DEFAULT_STUDENT, INITS, STUDENTS
from gidi.distillation.train_student import (
    ARMS,
    DEFAULT_OUT_DIR,
    DEFAULT_SPLITS_DIR,
    DEFAULT_TARGETS_DIR,
    DEFAULT_WEIGHTS_DIR,
    StudentTrainConfig,
    train_student,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    kd = KDConfig()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--seed", type=int, action="append", help="repeatable (default 1)")
    parser.add_argument("--student", default=DEFAULT_STUDENT, choices=sorted(STUDENTS))
    parser.add_argument("--init", default="random", choices=INITS)
    parser.add_argument(
        "--truncate-positions",
        action="store_true",
        help="keep only the max_length + pad + 1 position rows the model can index",
    )
    parser.add_argument(
        "--vocab-spec", default=None, help="pruned vocabulary dir (models/compression-v1/vocab/X)"
    )
    parser.add_argument(
        "--ffn-map", default=None, help="FFN neuron map dir (models/compression-v3/ffn/X)"
    )
    parser.add_argument(
        "--retokenize",
        action="store_true",
        help="encode the notes with the pruned tokenizer, not remapped cache ids (supervised only)",
    )
    parser.add_argument("--temperature", type=float, default=kd.temperature)
    parser.add_argument("--alpha-hard", type=float, default=kd.alpha_hard)
    parser.add_argument("--type-weight", type=float, default=kd.type_weight)
    parser.add_argument("--span-weight", type=float, default=kd.span_weight)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=5e-4, help="encoder learning rate")
    parser.add_argument("--head-lr", type=float, default=1e-3, help="lr of the two heads")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--device", default="mps", choices=("auto", "mps", "cuda", "cpu"))
    parser.add_argument("--targets", default=DEFAULT_TARGETS_DIR, help="teacher target cache")
    parser.add_argument("--splits-dir", default=DEFAULT_SPLITS_DIR)
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
    kd = KDConfig(
        temperature=args.temperature,
        alpha_hard=args.alpha_hard,
        type_weight=args.type_weight,
        span_weight=args.span_weight,
    )
    summaries = []
    for seed in args.seed or [1]:
        cfg = StudentTrainConfig(
            arm=args.arm,
            student=args.student,
            init=args.init,
            truncate_positions=args.truncate_positions,
            vocab_spec=args.vocab_spec,
            ffn_map=args.ffn_map,
            retokenize=args.retokenize,
            seed=seed,
            kd=kd,
            lr=args.lr,
            head_lr=args.head_lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            max_grad_norm=args.max_grad_norm,
            max_length=args.max_length,
            targets_dir=args.targets,
            splits_dir=args.splits_dir,
            out_dir=args.out_dir,
            weights_dir=args.weights_dir,
            max_steps=args.max_steps,
            device=args.device,
            overwrite=args.overwrite,
        )
        summaries.append(train_student(cfg))
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
