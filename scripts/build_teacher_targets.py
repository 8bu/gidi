#!/usr/bin/env python
"""Cache the frozen teacher's raw logits over the distillation training data.

Usage:
    uv run python scripts/build_teacher_targets.py \
        --teacher models/distillation-v1/teacher/bamibert/lr5e-05-seed1 \
        --data datasets/annotation-v1/distillation-v1/train.jsonl \
        --out experiments/distillation-v1/teacher-targets [--check]

Writes ``targets.safetensors``, ``index.jsonl`` and ``manifest.json`` (see
``gidi.distillation.targets``). ``--check`` regenerates the targets in memory and compares them
with the stored cache (exact for ids/labels/index/provenance, within 1e-4 for the logits) without
writing anything. ``--limit N`` uses only the first N records (smoke tests).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from gidi.distillation.targets import build_targets, check_targets, write_targets

log = logging.getLogger("build_teacher_targets")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--teacher", type=Path, required=True, help="teacher checkpoint directory")
    parser.add_argument("--data", type=Path, required=True, help="training JSONL to run through it")
    parser.add_argument("--out", type=Path, required=True, help="target cache directory")
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu", choices=("auto", "mps", "cuda", "cpu"))
    parser.add_argument("--limit", type=int, default=None, help="smoke: first N records only")
    parser.add_argument("--check", action="store_true", help="compare with the stored cache")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing cache")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    targets = build_targets(
        args.teacher,
        args.data,
        limit=args.limit,
        max_length=args.max_length,
        device=args.device,
        batch_size=args.batch_size,
    )
    stats = targets.manifest["teacher_vs_gold"]
    log.info("teacher vs gold on %d records: %s", len(targets), json.dumps(stats))
    if args.check:
        problems = check_targets(args.out, targets)
        if problems:
            log.error("target cache %s does not match regeneration:", args.out)
            for problem in problems:
                log.error("  %s", problem)
            return 1
        log.info("target cache %s matches regeneration", args.out)
        return 0
    write_targets(targets, args.out, overwrite=args.overwrite)
    log.info("wrote %d targets to %s", len(targets), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
