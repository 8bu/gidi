"""Train the 3-head value-span model (type + target BIO + value BIO), one run per seed.

The model is compression-v3 (``gidi-finance-v1``) plus one head: the same pretrained BamiBERT
4x768 encoder (layers 0-based 2/5/8/11), position table cut to 34 rows, vocabulary B-rank-8000,
FFN 2048 (combined neuron map), built by exactly the code path of v1
(``gidi.distillation.train_student.initial_model``), so the encoder and the seeded type/target
heads of seed ``s`` are bit-identical to v1's initial model for ``s``. It does **not** start from
the fine-tuned v1 checkpoint. The value head is added by ``GidiValueModel.from_v1`` from its own
seeded generator.

Recipe (compression-v3 supervised arm, unchanged): AdamW, encoder lr 5e-5, head lr 1e-3 (the
three heads), batch 8, 40 epochs, linear warmup 0.1 / decay over all steps, weight decay 0.01
(none on biases / layer norms), dropout 0.1, max grad norm 1.0, shuffles
``randperm(seed + epoch)``, LAST epoch kept, no selection on any held-out set, no KD.

Loss = CE(type) + CE(target tags) + CE(value tags), equal weights. Each CE is the v1 form: mean
over the batch (type) or over the positions whose label is not ``IGNORE_INDEX`` (tags). Value
positions of records that are not ``value_status == complete`` carry ``IGNORE_INDEX``, so such
records train only the type and target heads; a batch with no value supervision contributes a
clean zero. Every component is logged separately.

Writes ``<out_dir>/seed<N>/`` (``config.json``, ``train_log.jsonl``, ``metrics.json``,
``predictions_<split>.jsonl``, ``checkpoint.json``) and ``<weights_dir>/seed<N>/`` (a
``save_value_checkpoint`` directory). ``--validation`` / ``--test`` are optional evaluation files;
nothing is selected on them.
"""

from __future__ import annotations

import logging
import platform
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import transformers
from torch.nn import functional as F
from transformers import get_linear_schedule_with_warmup

from gidi.annotation.schema import load_config
from gidi.compression.vocab import load_vocab_spec
from gidi.corpus.jsonl import read_jsonl, write_jsonl
from gidi.device import resolve_device
from gidi.distillation.guard import FROZEN_TEST
from gidi.distillation.student import INIT_METHODS, TEACHER_TOKENIZER
from gidi.distillation.targets import sha256_file
from gidi.distillation.train_student import StudentTrainConfig, initial_model
from gidi.evaluation.value_eval import evaluate_records, value_span_categories
from gidi.evaluation.value_metrics import evaluate_value
from gidi.evaluation.value_predict import decode_predictions, torch_logits
from gidi.modeling.preprocessing import IGNORE_INDEX, TYPES
from gidi.modeling.value import (
    ANNOTATION_VERSION,
    VALUE_TAGS,
    CRFValueHead,
    GidiValueModel,
    ValuePrepared,
    prepare_value,
    save_value_checkpoint,
    value_crf,
)
from gidi.training.train import TrainConfig, _fmt, _optimizer, _sync, _write_json

log = logging.getLogger(__name__)

DEFAULT_OUT_DIR = "experiments/value-span-v1/runs"
DEFAULT_WEIGHTS_DIR = "models/value-span-v1"
DEFAULT_VOCAB_SPEC = "models/compression-v1/vocab/B-rank-8000"
DEFAULT_FFN_MAP = "models/compression-v3/ffn/combined-K2048"
STUDENT = "student-4x768"
LOSS_KEYS = ("total", "type", "target", "value")
FORBIDDEN_ID_PREFIX = "probe-"


@dataclass
class ValueTrainConfig:
    seed: int
    train_file: str
    validation_file: str | None = None  # evaluation only, logged per epoch
    test_file: str | None = None  # evaluation only, scored once after training
    lr: float = 5e-5
    head_lr: float = 1e-3
    epochs: int = 40
    batch_size: int = 8
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    dropout: float = 0.1
    max_grad_norm: float = 1.0
    max_length: int = 32
    tokenizer_name: str = TEACHER_TOKENIZER
    vocab_spec: str = DEFAULT_VOCAB_SPEC
    ffn_map: str = DEFAULT_FFN_MAP
    out_dir: str = DEFAULT_OUT_DIR
    weights_dir: str = DEFAULT_WEIGHTS_DIR
    max_steps: int | None = None  # smoke runs: stop after this many optimizer steps
    device: str = "mps"
    overwrite: bool = False


def run_dir(cfg: ValueTrainConfig) -> Path:
    return Path(cfg.out_dir) / f"seed{cfg.seed}"


def weights_dir(cfg: ValueTrainConfig) -> Path:
    return Path(cfg.weights_dir) / f"seed{cfg.seed}"


def v1_config(cfg: ValueTrainConfig) -> StudentTrainConfig:
    """The v1 (compression-v3 supervised) config whose initial model v2 starts from."""
    return StudentTrainConfig(
        arm="supervised",
        seed=cfg.seed,
        student=STUDENT,
        init="pretrained",
        dropout=cfg.dropout,
        max_length=cfg.max_length,
        tokenizer_name=cfg.tokenizer_name,
        truncate_positions=True,
        vocab_spec=cfg.vocab_spec,
        ffn_map=cfg.ffn_map,
    )


def initial_value_model(
    cfg: ValueTrainConfig, kept_old_ids: list[int] | None
) -> tuple[GidiValueModel, dict[str, Any] | None, dict[str, Any] | None]:
    """The seeded initial 3-head model: the v1 initial model of this seed plus a value head."""
    base, init_report, compression = initial_model(v1_config(cfg), kept_old_ids)
    return GidiValueModel.from_v1(base, cfg.seed), init_report, compression


# --------------------------------------------------------------------------- data


def load_value_records(path: str | Path) -> list[dict[str, Any]]:
    """Read and validate a merged annotation-v2 records file; refuses probe records."""
    records = read_jsonl(path)
    for r in records:
        if str(r.get("id", "")).startswith(FORBIDDEN_ID_PREFIX):
            raise ValueError(f"{path}: {r['id']!r} is a diagnostic probe record")
        missing = [k for k in ("id", "text", "type", "target") if k not in r]
        if missing:
            raise ValueError(f"{path}: record {r.get('id')!r} lacks {missing}")
        if r["type"] not in TYPES:
            raise ValueError(f"{path}: record {r['id']!r} has unknown type {r['type']!r}")
    return records  # the value block is validated by ``prepare_value``


def assert_not_test(records: list[dict[str, Any]], name: str, root: Path = Path(".")) -> None:
    """Training/validation files must not contain a frozen-test record."""
    frozen = root / FROZEN_TEST
    if not frozen.exists():
        return
    test_ids = {r["id"] for r in read_jsonl(frozen)}
    bad = sorted(r["id"] for r in records if r["id"] in test_ids)
    if bad:
        raise ValueError(f"{name} contains frozen test records: {bad[:5]} ({len(bad)} total)")


def _batches(
    data: ValuePrepared, batch_size: int, order: torch.Tensor | None = None
) -> Iterator[tuple[torch.Tensor, ...]]:
    """Yield ``(input_ids, mask, tag_labels, value_labels, type_ids)`` trimmed to max length."""
    n = len(data)
    order = torch.arange(n) if order is None else order
    enc = data.encoded
    value_labels = data.value_encoded.value_labels
    for i in range(0, n, batch_size):
        idx = order[i : i + batch_size]
        mask = enc.attention_mask[idx]
        width = int(mask.sum(dim=1).max())
        yield (
            enc.input_ids[idx, :width],
            mask[:, :width],
            enc.tag_labels[idx, :width],
            value_labels[idx, :width],
            data.type_ids[idx],
        )


# --------------------------------------------------------------------------- loss


def _masked_ce(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Mean CE over positions with ``label != IGNORE_INDEX`` (clean zero when there is none)."""
    keep = labels != IGNORE_INDEX
    n = int(keep.sum())
    if n == 0:
        return logits.sum() * 0.0, 0
    return F.cross_entropy(logits[keep], labels[keep]), n


def multitask_loss(
    type_logits: torch.Tensor,
    tag_logits: torch.Tensor,
    value_logits: torch.Tensor,
    type_ids: torch.Tensor,
    tag_labels: torch.Tensor,
    value_labels: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """``total = type + target + value`` (equal weights); the others are detached copies."""
    type_loss = F.cross_entropy(type_logits, type_ids)
    target_loss, _ = _masked_ce(tag_logits, tag_labels)
    value_loss, _ = _masked_ce(value_logits, value_labels)
    return {
        "total": type_loss + target_loss + value_loss,
        "type": type_loss.detach(),
        "target": target_loss.detach(),
        "value": value_loss.detach(),
    }


@torch.no_grad()
def eval_losses(model: GidiValueModel, data: ValuePrepared, device: torch.device) -> dict[str, Any]:
    """Loss components on a split (eval mode): type mean over notes, tags mean over positions.

    For an ``mlp-crf`` value head the value loss is the CRF NLL, mean over supervised notes.
    """
    crf = model.value_head.crf if isinstance(model.value_head, CRFValueHead) else None
    model.eval()
    sums = {"type": 0.0, "target": 0.0, "value": 0.0}
    counts = {"type": 0, "target": 0, "value": 0}
    for ids, mask, tags, values, types in _batches(data, 64):
        type_l, tag_l, value_l = model(ids.to(device), mask.to(device))
        types, tags, values = types.to(device), tags.to(device), values.to(device)
        sums["type"] += float(F.cross_entropy(type_l, types, reduction="sum"))
        counts["type"] += len(types)
        for name, logits, labels in (("target", tag_l, tags), ("value", value_l, values)):
            keep = labels != IGNORE_INDEX
            if not keep.any():
                continue
            if name == "value" and crf is not None:
                n_seq = int(keep.any(dim=1).sum())
                sums[name] += float(crf.nll(logits, labels, keep)) * n_seq
                counts[name] += n_seq
            else:
                sums[name] += float(F.cross_entropy(logits[keep], labels[keep], reduction="sum"))
                counts[name] += int(keep.sum())
    out: dict[str, Any] = {
        k: (sums[k] / counts[k] if counts[k] else None) for k in ("type", "target", "value")
    }
    out["total"] = sum(v for v in out.values() if v is not None)
    return out


# --------------------------------------------------------------------------- run


def _split_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [r for r in records if r["value_status"] == "complete"]
    return {
        "n": len(records),
        "value_complete": len(complete),
        "value_uncertain": len(records) - len(complete),
        "value_present": sum(r["value"] is not None for r in complete),
        "value_no_amount": sum(r["value"] is None for r in complete),
        "value_provenance": {
            p: sum(r["value_provenance"] == p for r in records) for p in ("human", "rule")
        },
    }


def _nonempty(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def train_value(cfg: ValueTrainConfig) -> dict[str, Any]:
    """Train one seed end to end; returns the ``checkpoint.json`` payload."""
    out, wdir = run_dir(cfg), weights_dir(cfg)
    if not cfg.overwrite:
        for path in (out, wdir):
            if _nonempty(path):
                raise FileExistsError(f"directory exists (pass overwrite): {path}")
    if load_config().types != TYPES:
        raise RuntimeError("TYPES does not match configs/annotation-v1.yaml")

    device = resolve_device(cfg.device)
    if device.type == "mps":
        log.warning("MPS kernels are not bit-reproducible; repeated runs of a seed may differ.")

    files = {
        "train": cfg.train_file,
        "validation": cfg.validation_file,
        "test": cfg.test_file,
    }
    records = {name: load_value_records(p) for name, p in files.items() if p}
    assert_not_test(records["train"], "the training file")
    if "validation" in records:
        assert_not_test(records["validation"], "the validation file")
    tokenizer, kept_old_ids = load_vocab_spec(cfg.vocab_spec)
    data = {
        name: prepare_value(tokenizer, recs, cfg.max_length, strict=name == "train")
        for name, recs in records.items()
    }
    train = data["train"]
    truncation = {
        name: {
            "n": len(d),
            "truncated": d.encoded.n_truncated,
            "target_span_truncated": d.encoded.n_span_truncated,
            "value_span_truncated": d.value_encoded.n_value_span_truncated,
            "value_boundary_mismatch": d.value_encoded.n_value_boundary_mismatch,
        }
        for name, d in data.items()
    }

    # --- model: v1's seeded initial model + value head ---
    model, init_report, compression = initial_value_model(cfg, kept_old_ids)
    param_count = model.param_count()
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    model.to(device)
    optimizer = _optimizer(
        model,
        TrainConfig(
            model="gidi-value-span-v1",
            lr=cfg.lr,
            seed=cfg.seed,
            head_lr=cfg.head_lr,
            weight_decay=cfg.weight_decay,
        ),
    )
    steps_per_epoch = -(-len(train) // cfg.batch_size)
    total_steps = cfg.epochs * steps_per_epoch
    if cfg.max_steps is not None:
        total_steps = min(total_steps, cfg.max_steps)
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
            "annotation_version": ANNOTATION_VERSION,
            "recipe": "compression-v3 supervised, plus value head (see gidi.training.train_value)",
            "student_config": model.encoder.config.to_dict(),
            "init_method": INIT_METHODS["pretrained"],
            "init_report": init_report,
            "compression": compression,
            "value_tags": list(VALUE_TAGS),
            "loss": "CE(type) + CE(target tags) + CE(value tags), equal weights, no KD",
            "resolved_device": str(device),
            "param_count": param_count,
            "param_bytes_fp32": param_bytes,
            "files_sha256": {n: sha256_file(Path(p)) for n, p in files.items() if p},
            "splits": {name: _split_summary(recs) for name, recs in records.items()},
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
    step = 0
    train_seconds = 0.0
    trained_examples = 0
    epochs_run = 0
    started = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = torch.randperm(
            len(train), generator=torch.Generator().manual_seed(cfg.seed + epoch)
        )
        sums = {k: torch.zeros((), device=device) for k in LOSS_KEYS}
        n_batches = 0
        _sync(device)
        t0 = time.perf_counter()
        for ids, mask, tags, values, types in _batches(train, cfg.batch_size, order):
            batch = (ids, mask, tags, values, types)
            ids, mask, tags, values, types = (x.to(device) for x in batch)
            type_l, tag_l, value_l = model(ids, mask)
            losses = multitask_loss(type_l, tag_l, value_l, types, tags, values)
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            for k in LOSS_KEYS:
                sums[k] += losses[k].detach()
            n_batches += 1
            trained_examples += len(types)
            step += 1
            if cfg.max_steps is not None and step >= cfg.max_steps:
                break
        _sync(device)
        epoch_seconds = time.perf_counter() - t0
        train_seconds += epoch_seconds
        epochs_run = epoch

        row: dict[str, Any] = {
            "epoch": epoch,
            "step": step,
            "train_loss": float(sums["total"].item() / n_batches),
            "train_type_loss": float(sums["type"].item() / n_batches),
            "train_target_loss": float(sums["target"].item() / n_batches),
            "train_value_loss": float(sums["value"].item() / n_batches),
            "lr": scheduler.get_last_lr()[0],
            "epoch_train_sec": epoch_seconds,
        }
        if "validation" in data:
            row |= _validation_row(model, data["validation"], device)
        row["wall_sec"] = time.perf_counter() - started
        log_rows.append(row)
        log.info(
            "[value seed%d] epoch %d step %d loss %.4f (type %.4f target %.4f value %.4f)%s %.1fs",
            cfg.seed,
            epoch,
            step,
            row["train_loss"],
            row["train_type_loss"],
            row["train_target_loss"],
            row["train_value_loss"],
            (
                f" val: value F1 {_fmt(row['val_value_span_f1'])} "
                f"exact {_fmt(row['val_value_exact'])}"
                if "validation" in data
                else ""
            ),
            epoch_seconds,
        )
        if cfg.max_steps is not None and step >= cfg.max_steps:
            break

    # The last epoch's weights are final: nothing is selected on any held-out set.
    wall_seconds = time.perf_counter() - started
    categories = value_span_categories(required=False)
    train_records = records["train"]
    metrics: dict[str, Any] = {"truncation": truncation, "losses": {}}
    for name, recs in records.items():
        split_metrics, rows, _ = evaluate_records(
            recs,
            lambda d: torch_logits(model, d, device),
            tokenizer,
            cfg.max_length,
            train_records=train_records,
            categories=categories,
        )
        metrics[name] = split_metrics
        metrics["losses"][name] = eval_losses(model, data[name], device)
        write_jsonl(out / f"predictions_{name}.jsonl", rows, overwrite=cfg.overwrite)
    write_jsonl(out / "train_log.jsonl", log_rows, overwrite=cfg.overwrite)
    _write_json(out / "metrics.json", metrics, cfg.overwrite)

    model.to("cpu")
    checkpoint_path = save_value_checkpoint(
        model,
        tokenizer,
        wdir,
        {
            "arm": "supervised-multitask",
            "student": STUDENT,
            "seed": cfg.seed,
            "epochs": epochs_run,
            "init": INIT_METHODS["pretrained"],
            "compression": compression,
            "lr": cfg.lr,
            "head_lr": cfg.head_lr,
            "max_length": cfg.max_length,
            "train_file_sha256": sha256_file(Path(cfg.train_file)),
        },
    )
    summary = {
        "seed": cfg.seed,
        "final_epoch": epochs_run,
        "final_train_losses": {
            k: log_rows[-1][f"train_{k}_loss"] for k in ("type", "target", "value")
        },
        "param_count": param_count,
        "param_bytes_fp32": param_bytes,
        "steps": step,
        "device": str(device),
        "wall_time_sec": wall_seconds,
        "train_time_sec": train_seconds,
        "sec_per_epoch": train_seconds / epochs_run,
        "train_examples_per_sec": trained_examples / train_seconds if train_seconds else None,
        "checkpoint_path": str(checkpoint_path),
    }
    _write_json(out / "checkpoint.json", summary, cfg.overwrite)
    return summary


def _validation_row(
    model: GidiValueModel, data: ValuePrepared, device: torch.device
) -> dict[str, Any]:
    """Validation losses (separately) and headline metrics, prefixed ``val_``."""
    t1 = time.perf_counter()
    losses = eval_losses(model, data, device)
    preds = decode_predictions(data, torch_logits(model, data, device), value_crf(model))
    m = evaluate_value(data.records, preds, offsets=data.encoded.offsets, slices=())
    overall, value = m["overall"], m["value"]
    return {
        "val_loss": losses["total"],
        "val_type_loss": losses["type"],
        "val_target_loss": losses["target"],
        "val_value_loss": losses["value"],
        "val_type_accuracy": overall["type"]["accuracy"],
        "val_type_macro_f1": overall["type"]["macro_f1"],
        "val_target_f1": overall["target"]["f1"],
        "val_target_exact": overall["target"]["exact_match"],
        "val_value_span_f1": value["span"]["f1"],
        "val_value_token_f1": value["token"]["f1"] if value["token"] else None,
        "val_value_exact": value["exact_match"],
        "val_full_joint": m["full_joint"]["accuracy"],
        "eval_sec": time.perf_counter() - t1,
    }


__all__ = [
    "ValueTrainConfig",
    "eval_losses",
    "initial_value_model",
    "load_value_records",
    "multitask_loss",
    "train_value",
]
