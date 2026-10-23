#!/usr/bin/env python
"""Train the gidi-en student from a ``build_en_student.py`` checkpoint (one seed).

Usage:
    uv run python scripts/train_en_student.py --init models/gidi-en-v1-init \\
        --out-dir experiments/gidi-en-student-v1/runs --weights-dir models/gidi-en-v1 --seed 1

Training data: silver-train + labels-seed (``gidi.training.english.load_train``). Dev is scored
each epoch for monitoring only; the last epoch's weights are kept (no selection). Test is never
loaded. Loss is ``CE(type) + CE(tags)`` as in the Vietnamese supervised arm.

``--value`` (gidi-en v3): the model gets the value-span head of value-span-v1 / gidi-finance-v2
(``GidiValueModel``: a ``Linear(hidden, 3)`` BIO head on the shared token states, seeded from
``VALUE_HEAD_SEED_BASE + seed``) and the loss is ``CE(type) + CE(target) + CE(value)``, equal
weights, the value tags of notes whose silver value label is not ``complete`` being
``IGNORE_INDEX`` (``gidi.modeling.value``). Everything else is the v2 recipe, unchanged.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import get_linear_schedule_with_warmup

from gidi.corpus.jsonl import write_jsonl
from gidi.device import resolve_device
from gidi.modeling.checkpoint import load_checkpoint, save_checkpoint
from gidi.modeling.preprocessing import IGNORE_INDEX
from gidi.modeling.value import GidiValueModel, prepare_value, save_value_checkpoint
from gidi.training.english import (
    load_labeled,
    load_labeled_values,
    load_train,
    load_train_v2,
    load_train_values_v2,
)
from gidi.training.train import (
    TrainConfig,
    _batches,
    _optimizer,
    _sync,
    evaluate_split,
    prepare,
    set_seed,
)
from gidi.training.train_value import _batches as _value_batches
from gidi.training.train_value import _validation_row, multitask_loss

log = logging.getLogger("train_en_student")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--init", type=Path, required=True, help="build_en_student.py output dir")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--weights-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5, help="encoder learning rate")
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--device", default="mps", choices=("auto", "mps", "cuda", "cpu"))
    ap.add_argument(
        "--slang-silver",
        action="store_true",
        help="v2: also train on the consistent notes of silver-slang-train.jsonl",
    )
    ap.add_argument(
        "--value",
        action="store_true",
        help="v3: add the value-span head and its loss (needs --slang-silver)",
    )
    ap.add_argument("--max-steps", type=int, default=None, help="smoke check: stop early")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    set_seed(args.seed)
    device = resolve_device(args.device)
    model, tokenizer, meta = load_checkpoint(args.init)
    max_length = int(meta["max_length"])
    if args.value:
        if not args.slang_silver:
            ap.error("--value is the v3 recipe: it needs --slang-silver (v2 data)")
        model = GidiValueModel.from_v1(model, args.seed)
        train = prepare_value(tokenizer, load_train_values_v2(), max_length)
        dev = prepare_value(tokenizer, load_labeled_values("dev"), max_length)
        value_labelled = sum(train.value_encoded.supervised)
        log.info("value supervision on %d of %d training notes", value_labelled, len(train))
    else:
        train = prepare(
            tokenizer, load_train_v2() if args.slang_silver else load_train(), max_length
        )
        dev = prepare(tokenizer, load_labeled("dev"), max_length)
    log.info(
        "train %d notes (truncated %d, span truncated %d), dev %d",
        len(train),
        train.encoded.n_truncated,
        train.encoded.n_span_truncated,
        len(dev),
    )

    model.to(device)
    opt_cfg = TrainConfig(
        model="gidi-en-student",
        lr=args.lr,
        seed=args.seed,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
    )
    optimizer = _optimizer(model, opt_cfg)
    steps_per_epoch = -(-len(train) // args.batch_size)
    total = args.epochs * steps_per_epoch
    if args.max_steps is not None:
        total = min(total, args.max_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(args.warmup_ratio * total), num_training_steps=total
    )

    rows = []
    step = 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(
            len(train), generator=torch.Generator().manual_seed(args.seed + epoch)
        )
        loss_sum = torch.zeros((), device=device)
        n_batches = 0
        batches = (
            _value_batches(train, args.batch_size, order)
            if args.value
            else _batches(train, args.batch_size, order)
        )
        for batch in batches:
            if args.value:
                ids, mask, tags, values, types = (x.to(device) for x in batch)
                type_logits, tag_logits, value_logits = model(ids, mask)
                parts = multitask_loss(type_logits, tag_logits, value_logits, types, tags, values)
                loss = parts["total"]
            else:
                _, ids, mask, tags, types = batch
                ids, mask, tags, types = (x.to(device) for x in (ids, mask, tags, types))
                type_logits, tag_logits = model(ids, mask)
                loss = F.cross_entropy(type_logits, types) + F.cross_entropy(
                    tag_logits.reshape(-1, tag_logits.shape[-1]),
                    tags.reshape(-1),
                    ignore_index=IGNORE_INDEX,
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            loss_sum += loss.detach()
            n_batches += 1
            step += 1
            if args.max_steps is not None and step >= args.max_steps:
                break
        _sync(device)
        if args.value:
            val = _validation_row(model, dev, device)
            dev_type, dev_span = val["val_type_macro_f1"], val["val_target_exact"]
            extra = {
                "dev_value_exact": val["val_value_exact"],
                "dev_value_loss": val["val_value_loss"],
            }
        else:
            metrics, _, _ = evaluate_split(model, dev, device)
            overall = metrics["overall"]
            dev_type, dev_span = overall["type"]["macro_f1"], overall["target"]["exact_match"]
            extra = {}
        row = {
            "epoch": epoch,
            "step": step,
            "train_loss": float(loss_sum.item() / n_batches),
            "dev_type_macro_f1": dev_type,
            "dev_span_exact": dev_span,
            **extra,
            "wall_sec": time.perf_counter() - started,
        }
        rows.append(row)
        log.info(
            "epoch %d step %d loss %.4f dev type F1 %.3f span exact %.3f%s",
            epoch,
            step,
            row["train_loss"],
            row["dev_type_macro_f1"] or 0.0,
            row["dev_span_exact"] or 0.0,
            f" value exact {extra['dev_value_exact'] or 0.0:.3f}" if args.value else "",
        )
        if args.max_steps is not None and step >= args.max_steps:
            break

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "train_log.jsonl", rows, overwrite=True)
    (args.out_dir / "config.json").write_text(
        json.dumps({**{k: str(v) for k, v in vars(args).items()}, "n_train": len(train)}, indent=2)
        + "\n"
    )
    save = save_value_checkpoint if args.value else save_checkpoint
    save(
        model.cpu(),
        tokenizer,
        args.weights_dir,
        {
            "seed": args.seed,
            "epochs": epoch,
            "lr": args.lr,
            "max_length": max_length,
            "init": str(args.init),
            **({"annotation_version": "annotation-en-v1"} if args.value else {}),
            "labels": "silver-train + labels-seed (annotation-en-v1)"
            + (" + silver value labels (value-*.jsonl)" if args.value else ""),
        },
    )
    log.info("saved %s", args.weights_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
