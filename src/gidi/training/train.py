"""Fine-tune the shared-encoder multi-task model on the annotation-v1 splits.

One call to :func:`train_run` trains one (model, lr, seed) configuration, keeps the best
validation epoch, evaluates it on validation and test, and writes the run directory::

    <out_dir>/<model-slug>/lr<lr>-seed<seed>/
        config.json            resolved configuration + environment
        train_log.jsonl        one row per epoch (losses, validation metrics, wall time)
        metrics.json           best-epoch validation + test metrics (``gidi.evaluation.evaluate``)
        predictions_test.jsonl id, text, gold/pred type, gold/pred span
        checkpoint.json        best epoch, score, parameter count, throughput, weights path

Loss is ``CE(type) + CE(tags, ignore -100)`` with equal weight. The model-selection score is
``mean(type macro-F1, target span F1)`` on validation; training stops after ``patience`` epochs
without a strictly better score.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedTokenizerBase, get_linear_schedule_with_warmup

from gidi.annotation.schema import load_config
from gidi.corpus.jsonl import read_jsonl, write_jsonl
from gidi.device import resolve_device
from gidi.evaluation.metrics import evaluate
from gidi.modeling.checkpoint import save_checkpoint
from gidi.modeling.model import GidiMultiTaskModel
from gidi.modeling.preprocessing import IGNORE_INDEX, TYPES, Encoded, encode, spans_from_tags
from gidi.modeling.tokenization import load_tokenizer

log = logging.getLogger(__name__)

SPLIT_NAMES = ("train", "validation", "test")
DEFAULT_OUT_DIR = "experiments/baseline-v1/runs"
DEFAULT_WEIGHTS_DIR = "models/baseline-v1"
DEFAULT_SPLITS_DIR = "datasets/annotation-v1/splits"
# Diagnostic probe sets (datasets/probe-*) are evaluation-only and must never be trained on.
PROBE_ID_PREFIX = "probe-"


@dataclass
class TrainConfig:
    model: str
    lr: float
    seed: int
    head_lr: float = 1e-3  # randomly initialized heads; the encoder uses ``lr``
    epochs: int = 10
    patience: int = 3
    max_length: int = 32
    batch_size: int = 16
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    dropout: float = 0.1
    max_grad_norm: float = 1.0
    splits_dir: str = DEFAULT_SPLITS_DIR
    out_dir: str = DEFAULT_OUT_DIR
    weights_dir: str = DEFAULT_WEIGHTS_DIR
    save_weights: bool = True
    max_steps: int | None = None  # smoke runs: stop after this many optimizer steps
    # Fixed-epoch mode: train exactly this many epochs (the schedule horizon stays ``epochs``),
    # no early stopping, keep the LAST epoch's weights. Validation is logged only.
    stop_epoch: int | None = None
    device: str = "auto"
    overwrite: bool = False


def model_slug(name: str) -> str:
    """Directory-safe model name: last path component, lowercased."""
    return Path(name.rstrip("/")).name.lower()


def run_name(lr: float, seed: int) -> str:
    return f"lr{lr:g}-seed{seed}"


def run_dir(cfg: TrainConfig) -> Path:
    return Path(cfg.out_dir) / model_slug(cfg.model) / run_name(cfg.lr, cfg.seed)


def weights_dir(cfg: TrainConfig) -> Path:
    return Path(cfg.weights_dir) / model_slug(cfg.model) / run_name(cfg.lr, cfg.seed)


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch and request deterministic kernels where they exist."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


@dataclass
class Prepared:
    """One split, encoded once."""

    records: list[dict[str, Any]]
    encoded: Encoded
    type_ids: torch.Tensor

    def __len__(self) -> int:
        return len(self.records)


def load_split(path: str | Path) -> list[dict[str, Any]]:
    records = read_jsonl(path)
    for r in records:
        if str(r.get("id", "")).startswith(PROBE_ID_PREFIX):
            raise ValueError(f"{path}: {r['id']!r} is a diagnostic probe record; never train on it")
        missing = [k for k in ("id", "text", "type", "target") if k not in r]
        if missing:
            raise ValueError(f"{path}: record {r.get('id')!r} lacks {missing}")
        if r["type"] not in TYPES:
            raise ValueError(f"{path}: record {r['id']!r} has unknown type {r['type']!r}")
    return records


def prepare(
    tokenizer: PreTrainedTokenizerBase, records: list[dict[str, Any]], max_length: int
) -> Prepared:
    encoded = encode(
        tokenizer, [r["text"] for r in records], [r["target"] for r in records], max_length
    )
    type_ids = torch.tensor([TYPES.index(r["type"]) for r in records], dtype=torch.long)
    return Prepared(records, encoded, type_ids)


def _batches(
    data: Prepared, batch_size: int, order: torch.Tensor | None = None
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Yield ``(idx, input_ids, attention_mask, tag_labels, type_ids)``, trimmed to max length."""
    n = len(data)
    order = torch.arange(n) if order is None else order
    enc = data.encoded
    for i in range(0, n, batch_size):
        idx = order[i : i + batch_size]
        mask = enc.attention_mask[idx]
        width = int(mask.sum(dim=1).max())
        yield (
            idx,
            enc.input_ids[idx, :width],
            mask[:, :width],
            enc.tag_labels[idx, :width],
            data.type_ids[idx],
        )


@torch.no_grad()
def predict(
    model: GidiMultiTaskModel, data: Prepared, device: torch.device, batch_size: int = 64
) -> tuple[list[str], list[dict[str, Any] | None]]:
    """Predicted type names and first target span (or ``None``) per record, in record order."""
    model.eval()
    types: list[str] = []
    spans: list[dict[str, Any] | None] = []
    for idx, input_ids, mask, _, _ in _batches(data, batch_size):
        type_logits, tag_logits = model(input_ids.to(device), mask.to(device))
        type_pred = type_logits.argmax(-1).cpu().tolist()
        tag_pred = tag_logits.argmax(-1).cpu().tolist()
        for j, i in enumerate(idx.tolist()):
            types.append(TYPES[type_pred[j]])
            spans.append(
                spans_from_tags(data.encoded.offsets[i], tag_pred[j], data.records[i]["text"])
            )
    return types, spans


def evaluate_split(
    model: GidiMultiTaskModel, data: Prepared, device: torch.device
) -> tuple[dict[str, Any], list[str], list[dict[str, Any] | None]]:
    types, spans = predict(model, data, device)
    return evaluate(data.records, types, spans), types, spans


def selection_score(metrics: dict[str, Any]) -> float:
    """mean(type macro-F1, target span F1) of an ``evaluate`` result's overall block."""
    overall = metrics["overall"]
    return ((overall["type"]["macro_f1"] or 0.0) + (overall["target"]["f1"] or 0.0)) / 2


def _optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """AdamW; the two heads get ``head_lr`` (at the encoder lr the tag head stays all-O)."""
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups: dict[tuple[bool, bool], list[nn.Parameter]] = {}
    for name, param in model.named_parameters():
        key = (name.startswith("encoder."), any(k in name for k in no_decay))
        groups.setdefault(key, []).append(param)
    return torch.optim.AdamW(
        [
            {
                "params": params,
                "lr": cfg.lr if is_encoder else cfg.head_lr,
                "weight_decay": 0.0 if plain else cfg.weight_decay,
            }
            for (is_encoder, plain), params in sorted(groups.items())
        ],
        lr=cfg.lr,
    )


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _write_json(path: Path, obj: Any, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def train_run(cfg: TrainConfig) -> dict[str, Any]:
    """Train one configuration end to end; returns the ``checkpoint.json`` payload."""
    out = run_dir(cfg)
    if out.exists() and any(out.iterdir()) and not cfg.overwrite:
        raise FileExistsError(f"run directory exists (pass overwrite): {out}")
    if load_config().types != TYPES:
        raise RuntimeError("TYPES does not match configs/annotation-v1.yaml")
    if cfg.stop_epoch is not None and not 1 <= cfg.stop_epoch <= cfg.epochs:
        raise ValueError(f"stop_epoch must be in [1, epochs={cfg.epochs}], got {cfg.stop_epoch}")

    device = resolve_device(cfg.device)
    set_seed(cfg.seed)
    if device.type == "mps":
        log.warning(
            "MPS kernels are not bit-reproducible across runs even with fixed seeds; expect "
            "small metric differences between repeated runs of the same seed."
        )

    splits_dir = Path(cfg.splits_dir)
    records = {name: load_split(splits_dir / f"{name}.jsonl") for name in SPLIT_NAMES}
    tokenizer = load_tokenizer(cfg.model)
    data = {name: prepare(tokenizer, recs, cfg.max_length) for name, recs in records.items()}
    truncation = {
        name: {
            "n": len(d),
            "truncated": d.encoded.n_truncated,
            "span_truncated": d.encoded.n_span_truncated,
        }
        for name, d in data.items()
    }
    if any(t["truncated"] for t in truncation.values()):
        log.warning("texts truncated at max_length=%d: %s", cfg.max_length, truncation)

    model = GidiMultiTaskModel.from_encoder(cfg.model, dropout=cfg.dropout).to(device)
    param_count = model.param_count()

    train = data["train"]
    steps_per_epoch = -(-len(train) // cfg.batch_size)
    total_steps = cfg.epochs * steps_per_epoch
    if cfg.max_steps is not None:
        total_steps = min(total_steps, cfg.max_steps)
    optimizer = _optimizer(model, cfg)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    out.mkdir(parents=True, exist_ok=True)
    _write_json(
        out / "config.json",
        {
            **asdict(cfg),
            "resolved_device": str(device),
            "model_slug": model_slug(cfg.model),
            "param_count": param_count,
            "split_sizes": {name: len(d) for name, d in data.items()},
            "truncation": truncation,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "env": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "platform": platform.platform(),
            },
        },
        cfg.overwrite,
    )

    log_rows: list[dict[str, Any]] = []
    best_score = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    step = 0
    train_seconds = 0.0
    trained_examples = 0
    started = time.perf_counter()
    epochs_run = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = torch.randperm(
            len(train), generator=torch.Generator().manual_seed(cfg.seed + epoch)
        )
        type_loss_sum = torch.zeros((), device=device)
        tag_loss_sum = torch.zeros((), device=device)
        n_batches = 0
        _sync(device)
        t0 = time.perf_counter()
        for _, input_ids, mask, tags, types in _batches(train, cfg.batch_size, order):
            input_ids, mask, tags, types = (x.to(device) for x in (input_ids, mask, tags, types))
            type_logits, tag_logits = model(input_ids, mask)
            type_loss = F.cross_entropy(type_logits, types)
            tag_loss = F.cross_entropy(
                tag_logits.reshape(-1, tag_logits.shape[-1]),
                tags.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
            loss = type_loss + tag_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            type_loss_sum += type_loss.detach()
            tag_loss_sum += tag_loss.detach()
            n_batches += 1
            trained_examples += len(types)
            step += 1
            if cfg.max_steps is not None and step >= cfg.max_steps:
                break
        _sync(device)
        epoch_seconds = time.perf_counter() - t0
        train_seconds += epoch_seconds
        epochs_run = epoch
        fixed_last = cfg.stop_epoch is not None and (
            epoch == cfg.stop_epoch or (cfg.max_steps is not None and step >= cfg.max_steps)
        )

        t1 = time.perf_counter()
        val_metrics, _, _ = evaluate_split(model, data["validation"], device)
        eval_seconds = time.perf_counter() - t1
        score = selection_score(val_metrics)
        improved = score > best_score if cfg.stop_epoch is None else fixed_last
        if improved:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        else:
            stale += 1
        overall = val_metrics["overall"]
        row = {
            "epoch": epoch,
            "step": step,
            "train_loss": float((type_loss_sum + tag_loss_sum).item() / n_batches),
            "train_type_loss": float(type_loss_sum.item() / n_batches),
            "train_tag_loss": float(tag_loss_sum.item() / n_batches),
            "val_score": score,
            "val_type_macro_f1": overall["type"]["macro_f1"],
            "val_type_accuracy": overall["type"]["accuracy"],
            "val_span_f1": overall["target"]["f1"],
            "val_span_exact_match": overall["target"]["exact_match"],
            "lr": scheduler.get_last_lr()[0],
            "epoch_train_sec": epoch_seconds,
            "eval_sec": eval_seconds,
            "wall_sec": time.perf_counter() - started,
            "best": improved,
        }
        log_rows.append(row)
        log.info(
            "epoch %d step %d loss %.4f val_score %.4f (type F1 %s, span F1 %s) %.1fs%s",
            epoch,
            step,
            row["train_loss"],
            score,
            _fmt(row["val_type_macro_f1"]),
            _fmt(row["val_span_f1"]),
            epoch_seconds,
            " *" if improved else "",
        )
        if cfg.max_steps is not None and step >= cfg.max_steps:
            break
        if fixed_last:
            break
        if cfg.stop_epoch is None and stale >= cfg.patience:
            log.info("early stopping: no improvement for %d epochs", cfg.patience)
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    val_metrics, _, _ = evaluate_split(model, data["validation"], device)
    test_metrics, test_types, test_spans = evaluate_split(model, data["test"], device)
    wall = time.perf_counter() - started

    write_jsonl(out / "train_log.jsonl", log_rows, overwrite=cfg.overwrite)
    _write_json(
        out / "metrics.json",
        {"validation": val_metrics, "test": test_metrics, "truncation": truncation},
        cfg.overwrite,
    )
    write_jsonl(
        out / "predictions_test.jsonl",
        (
            {
                "id": r["id"],
                "text": r["text"],
                "gold_type": r["type"],
                "pred_type": t,
                "gold_span": r["target"],
                "pred_span": s,
            }
            for r, t, s in zip(data["test"].records, test_types, test_spans, strict=True)
        ),
        overwrite=cfg.overwrite,
    )

    checkpoint_path = None
    if cfg.save_weights:
        checkpoint_path = str(
            save_checkpoint(
                model,
                tokenizer,
                weights_dir(cfg),
                {
                    "best_epoch": best_epoch,
                    "val_score": best_score,
                    "seed": cfg.seed,
                    "lr": cfg.lr,
                    "max_length": cfg.max_length,
                    **(
                        {"stop_epoch": cfg.stop_epoch, "selection": "none (fixed epoch)"}
                        if cfg.stop_epoch is not None
                        else {}
                    ),
                },
            )
        )
    summary = {
        "model": cfg.model,
        "best_epoch": best_epoch,
        "best_val_score": best_score,
        "test_type_macro_f1": test_metrics["overall"]["type"]["macro_f1"],
        "test_span_f1": test_metrics["overall"]["target"]["f1"],
        "param_count": param_count,
        "epochs_run": epochs_run,
        "steps": step,
        "device": str(device),
        "wall_time_sec": wall,
        "train_time_sec": train_seconds,
        "sec_per_epoch": train_seconds / epochs_run,
        "train_examples_per_sec": trained_examples / train_seconds if train_seconds else None,
        "checkpoint_path": checkpoint_path,
    }
    if cfg.stop_epoch is not None:
        summary |= {"stop_epoch": cfg.stop_epoch, "selection": "none (fixed epoch)"}
    _write_json(out / "checkpoint.json", summary, cfg.overwrite)
    return summary


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def train_grid(
    model: str,
    lrs: Sequence[float],
    seeds: Sequence[int],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Run every (lr, seed) pair for ``model`` sequentially."""
    summaries = []
    for lr in lrs:
        for seed in seeds:
            cfg = TrainConfig(model=model, lr=lr, seed=seed, **kwargs)
            log.info("run %s %s", model_slug(model), run_name(lr, seed))
            summaries.append(train_run(cfg))
    return summaries
