"""Train ONLY a new value BIO head on the frozen gidi-finance-v1 model.

Used by value-span-v3-frozen-v1 (``linear`` head) and value-span-v4-nonlinear-head (``mlp``
head: Linear(768, 256) -> GELU -> Dropout -> Linear(256, 3)). The base checkpoint (encoder, type
head, target BIO head) is loaded as trained, wrapped with ``GidiValueModel.from_v1`` (a seeded
fresh value head) and frozen except for the value head. Everything outside the value head runs
in eval mode; during training the value head itself is in train mode (only the ``mlp`` head has
a dropout). The loss is the value BIO cross-entropy only, with uncertain values masked.

Three programmatic checks guard the freezing; each raises on failure and is recorded in
``metrics.json``: the optimizer holds exactly the ``value_head.*`` parameters; after the first
backward every frozen parameter has grad ``None`` or all zero; the sha256 of the frozen state
equals the base checkpoint's before training and after the last epoch.
"""

from __future__ import annotations

import hashlib
import logging
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import get_linear_schedule_with_warmup

from gidi.corpus.jsonl import write_jsonl
from gidi.device import resolve_device
from gidi.distillation.targets import sha256_file
from gidi.modeling.checkpoint import WEIGHTS_FILE, load_checkpoint
from gidi.modeling.preprocessing import IGNORE_INDEX
from gidi.modeling.value import (
    CRFValueHead,
    GidiValueModel,
    ValuePrepared,
    prepare_value,
    save_value_checkpoint,
)
from gidi.training.train import _fmt, _sync, _write_json
from gidi.training.train_value import (
    _batches,
    _masked_ce,
    _nonempty,
    _split_summary,
    _validation_row,
    assert_not_test,
    load_value_records,
)

log = logging.getLogger(__name__)

EXPERIMENT = "value-span-v3-frozen-v1"
DEFAULT_BASE = (
    "models/compression-v3/supervised/"
    "student-4x768-pretrained-pos32-vocab-B-rank-8000-ffn-combined-K2048/seed1"
)
DEFAULT_OUT_DIR = f"models/{EXPERIMENT}"
VALUE_PREFIX = "value_head."


@dataclass
class ValueHeadConfig:
    seed: int
    train_file: str
    experiment: str = EXPERIMENT
    head_arch: str = "linear"
    validation_file: str | None = None  # logged per epoch only, never selected on
    base: str = DEFAULT_BASE
    lr: float = 1e-3
    adapter_lr: float | None = None  # value adapter block (``adapter-mlp-crf``); None: ``lr``
    encoder_lr: float | None = None  # cloned value encoder (``encoder-mlp-crf``); None: ``lr``
    weight_decay: float = 0.01
    epochs: int = 40
    batch_size: int = 8
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    out_dir: str = DEFAULT_OUT_DIR
    max_steps: int | None = None  # smoke runs: stop after this many optimizer steps
    device: str = "auto"
    overwrite: bool = False


def run_dir(cfg: ValueHeadConfig) -> Path:
    return Path(cfg.out_dir) / f"seed{cfg.seed}"


# --------------------------------------------------------------------------- freezing


def frozen_state_sha256(model: torch.nn.Module) -> str:
    """sha256 over every non-value-head ``state_dict`` tensor, sorted by key (CPU bytes)."""
    digest = hashlib.sha256()
    state = model.state_dict()
    for key in sorted(k for k in state if not k.startswith("value_head.")):
        flat = state[key].detach().cpu().contiguous().reshape(-1)
        digest.update(flat.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def value_params(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    return {n: p for n, p in model.named_parameters() if n.startswith(VALUE_PREFIX)}


def freeze_all_but_value_head(model: GidiValueModel) -> None:
    """``requires_grad`` only on the value head; the whole model goes to eval mode."""
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(VALUE_PREFIX)
    model.eval()


def build_optimizer(model: GidiValueModel, cfg: ValueHeadConfig) -> torch.optim.Optimizer:
    """AdamW over the value head; only weight matrices (Linear / attention / embeddings) are
    decayed (not biases or LayerNorms, as in the v1 optimizer groups, and not CRF transition
    scores). The value adapter block (``value_head.adapter.*``) uses ``adapter_lr`` and the
    cloned value encoder (``value_head.encoder.*``) ``encoder_lr`` when set; the rest ``lr``."""
    params = value_params(model)
    part_lr = {
        "adapter": cfg.lr if cfg.adapter_lr is None else cfg.adapter_lr,
        "encoder": cfg.lr if cfg.encoder_lr is None else cfg.encoder_lr,
        "head": cfg.lr,
    }

    def part(name: str) -> str:
        for body in ("adapter", "encoder"):
            if name.startswith(f"{VALUE_PREFIX}{body}."):
                return body
        return "head"

    def decayed(name: str, p: torch.nn.Parameter) -> bool:
        return p.ndim >= 2 and ".crf." not in name

    groups = []
    for name_part in ("head", "adapter", "encoder"):
        for decay in (True, False):
            chosen = [
                p for n, p in params.items() if part(n) == name_part and decayed(n, p) == decay
            ]
            if chosen:
                weight_decay = cfg.weight_decay if decay else 0.0
                groups.append(
                    {"params": chosen, "weight_decay": weight_decay, "lr": part_lr[name_part]}
                )
    return torch.optim.AdamW(groups, lr=cfg.lr)


def check_value_encoder_clone(model: GidiValueModel) -> dict[str, Any] | None:
    """``encoder-mlp-crf``: the value encoder starts bit-identical to the frozen v1 encoder but
    shares no storage with it. ``None`` for heads without a value encoder."""
    value_encoder = getattr(model.value_head, "encoder", None)
    if value_encoder is None:
        return None
    original, clone = model.encoder.state_dict(), value_encoder.state_dict()
    if list(original) != list(clone):
        raise RuntimeError("value encoder keys differ from the v1 encoder's")
    for key, tensor in original.items():
        if not torch.equal(tensor, clone[key]):
            raise RuntimeError(f"value encoder {key} differs from the v1 encoder at init")
        if tensor.untyped_storage().data_ptr() == clone[key].untyped_storage().data_ptr():
            raise RuntimeError(f"value encoder {key} shares storage with the v1 encoder")
    return {"ok": True, "n_tensors": len(original), "identical_to_v1_encoder": True}


def value_loss(model: GidiValueModel, value_logits: torch.Tensor, values: torch.Tensor):
    """Value BIO CE (``linear`` / ``mlp``) or CRF NLL (``mlp-crf``, mean over supervised notes);
    uncertain values (``IGNORE_INDEX`` everywhere) are masked either way."""
    if isinstance(model.value_head, CRFValueHead):
        return model.value_head.crf.nll(value_logits, values, values != IGNORE_INDEX)
    return _masked_ce(value_logits, values)[0]


def check_optimizer_params(optimizer: torch.optim.Optimizer, model: GidiValueModel) -> dict:
    """The optimizer's parameters (by identity) are exactly the value head's parameters."""
    held = [p for group in optimizer.param_groups for p in group["params"]]
    expected = {id(p): n for n, p in value_params(model).items()}
    if len(held) != len(expected) or {id(p) for p in held} != set(expected):
        raise RuntimeError(f"optimizer holds {len(held)} tensors, not exactly {expected.values()}")
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if sorted(trainable) != sorted(expected.values()):
        raise RuntimeError(f"requires_grad parameters are {trainable}, not the value head")
    return {
        "ok": True,
        "optimizer_params": sorted(expected.values()),
        "trainable_parameters": trainable,
        "n_trainable": sum(p.numel() for p in held),
    }


def check_frozen_grads(model: GidiValueModel) -> dict:
    """After a backward pass every frozen parameter has grad ``None`` or all zero."""
    n_frozen = n_none = 0
    for name, param in model.named_parameters():
        if name.startswith(VALUE_PREFIX):
            if param.grad is None:
                raise RuntimeError(f"{name} received no gradient")
            continue
        n_frozen += 1
        if param.grad is None:
            n_none += 1
        elif bool((param.grad != 0).any()):
            raise RuntimeError(f"frozen parameter {name} has a non-zero gradient")
    return {"ok": True, "n_frozen_params": n_frozen, "n_grad_none": n_none}


# --------------------------------------------------------------------------- training


def fit_value_head(
    model: GidiValueModel,
    train: ValuePrepared,
    cfg: ValueHeadConfig,
    device: torch.device,
    validation: ValuePrepared | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Train the value head in place; returns ``(log_rows, checks)``. ``model`` is on ``device``.

    Raises if the freezing checks fail (optimizer params, frozen grads, frozen-state hash).
    """
    sha_before = frozen_state_sha256(model)
    freeze_all_but_value_head(model)
    optimizer = build_optimizer(model, cfg)
    checks: dict[str, Any] = {"optimizer": check_optimizer_params(optimizer, model)}

    steps_per_epoch = -(-len(train) // cfg.batch_size)
    total_steps = cfg.epochs * steps_per_epoch
    if cfg.max_steps is not None:
        total_steps = min(total_steps, cfg.max_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    log_rows: list[dict[str, Any]] = []
    step = 0
    started = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.eval()  # the head trains on exactly the deployed encoder features
        model.value_head.train()  # only the value head's own dropout (mlp) is active
        order = torch.randperm(
            len(train), generator=torch.Generator().manual_seed(cfg.seed + epoch)
        )
        loss_sum = torch.zeros((), device=device)
        n_batches = 0
        _sync(device)
        t0 = time.perf_counter()
        for ids, mask, _tags, values, _types in _batches(train, cfg.batch_size, order):
            ids, mask, values = ids.to(device), mask.to(device), values.to(device)
            _type_l, _tag_l, value_l = model(ids, mask)
            loss = value_loss(model, value_l, values)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if step == 0:
                checks["first_backward_frozen_grads"] = check_frozen_grads(model)
            torch.nn.utils.clip_grad_norm_(model.value_head.parameters(), cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            loss_sum += loss.detach()
            n_batches += 1
            step += 1
            if cfg.max_steps is not None and step >= cfg.max_steps:
                break
        _sync(device)
        epoch_seconds = time.perf_counter() - t0
        row: dict[str, Any] = {
            "epoch": epoch,
            "step": step,
            "train_value_loss": float(loss_sum.item() / n_batches),
            "lr": scheduler.get_last_lr()[0],
            "epoch_train_sec": epoch_seconds,
        }
        if validation is not None:
            row |= _validation_row(model, validation, device)
        row["wall_sec"] = time.perf_counter() - started
        log_rows.append(row)
        log.info(
            "[value-head seed%d] epoch %d step %d value loss %.4f%s %.1fs",
            cfg.seed,
            epoch,
            step,
            row["train_value_loss"],
            (
                f" val: loss {_fmt(row['val_value_loss'])} exact {_fmt(row['val_value_exact'])}"
                if validation is not None
                else ""
            ),
            epoch_seconds,
        )
        if cfg.max_steps is not None and step >= cfg.max_steps:
            break
    model.eval()  # leave the value head's dropout off for evaluation and saving

    sha_after = frozen_state_sha256(model)
    if sha_after != sha_before:
        raise RuntimeError(f"frozen state changed during training: {sha_before} -> {sha_after}")
    checks["frozen_state_sha256_before"] = sha_before
    checks["frozen_state_sha256_after"] = sha_after
    checks["frozen_state_unchanged"] = True
    checks["steps"] = step
    return log_rows, checks


def train_value_head(cfg: ValueHeadConfig) -> dict[str, Any]:
    """Train one value-head seed end to end; returns the ``checkpoint.json`` payload."""
    out = run_dir(cfg)
    if not cfg.overwrite and _nonempty(out):
        raise FileExistsError(f"directory exists (pass overwrite): {out}")
    device = resolve_device(cfg.device)
    if device.type == "mps":
        log.warning("MPS kernels are not bit-reproducible; repeated runs of a seed may differ.")

    base_dir = Path(cfg.base)
    base, tokenizer, base_meta = load_checkpoint(base_dir)
    max_length = int(base_meta["max_length"])
    base_sha = frozen_state_sha256(base)  # the freshly loaded base, before wrapping

    files = {"train": cfg.train_file, "validation": cfg.validation_file}
    records = {name: load_value_records(p) for name, p in files.items() if p}
    for name, recs in records.items():
        assert_not_test(recs, f"the {name} file")
    data = {
        name: prepare_value(tokenizer, recs, max_length, strict=name == "train")
        for name, recs in records.items()
    }

    model = GidiValueModel.from_v1(base, cfg.seed, value_head_arch=cfg.head_arch)
    wrapped_sha = frozen_state_sha256(model)
    if wrapped_sha != base_sha:
        raise RuntimeError(f"wrapping changed the frozen state: {base_sha} -> {wrapped_sha}")
    clone_check = check_value_encoder_clone(model)
    # the dropout stream (value-branch dropouts only; everything else is in eval mode)
    torch.manual_seed(cfg.seed)
    model.to(device)
    trainable_names = list(value_params(model))
    trainable = sum(p.numel() for p in value_params(model).values())

    started = time.perf_counter()
    log_rows, checks = fit_value_head(model, data["train"], cfg, device, data.get("validation"))
    wall_seconds = time.perf_counter() - started
    model.to("cpu")
    # also after the device round trip: this is the hash the checkpoint is saved under
    final_sha = frozen_state_sha256(model)
    if final_sha != base_sha:
        raise RuntimeError(f"saved frozen state differs from the base: {base_sha} -> {final_sha}")
    checks["frozen_state_sha256_base"] = base_sha
    checks["frozen_state_equals_base"] = True
    if clone_check is not None:
        checks["value_encoder_clone_at_init"] = clone_check

    out.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {
        "experiment": cfg.experiment,
        "seed": cfg.seed,
        "checks": checks,
        "config": {
            **asdict(cfg),
            "max_length": max_length,
            "resolved_device": str(device),
            "loss": (
                "value CRF NLL only (mean over supervised notes; uncertain values masked)"
                if cfg.head_arch.endswith("crf")
                else "value BIO CE only (uncertain values masked)"
            )
            + "; all but the value head in eval",
            "trainable_parameter_count": trainable,
            "files_sha256": {n: sha256_file(Path(p)) for n, p in files.items() if p},
            "splits": {name: _split_summary(recs) for name, recs in records.items()},
            "env": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "platform": platform.platform(),
            },
        },
        "final_epoch": log_rows[-1],
    }
    write_jsonl(out / "train_log.jsonl", log_rows, overwrite=cfg.overwrite)
    _write_json(out / "metrics.json", metrics, cfg.overwrite)
    checkpoint_path = save_value_checkpoint(
        model,
        tokenizer,
        out,
        {
            "experiment": cfg.experiment,
            "arm": f"frozen-v1-value-head-{cfg.head_arch}",
            "student": base_meta.get("student"),
            "seed": cfg.seed,
            "epochs": log_rows[-1]["epoch"],
            "lr": cfg.lr,
            "max_length": max_length,
            "heads": ["type", "target", "value"],
            "frozen_base": {
                "path": str(base_dir),
                "model_safetensors_sha256": sha256_file(base_dir / WEIGHTS_FILE),
            },
            "frozen_state_sha256": final_sha,
            "trainable_parameters": trainable_names,
            "train_file_sha256": sha256_file(Path(cfg.train_file)),
        },
    )
    summary = {
        "seed": cfg.seed,
        "final_epoch": log_rows[-1]["epoch"],
        "final_train_value_loss": log_rows[-1]["train_value_loss"],
        "steps": checks["steps"],
        "device": str(device),
        "wall_time_sec": wall_seconds,
        "frozen_state_sha256": final_sha,
        "checkpoint_path": str(checkpoint_path),
    }
    _write_json(out / "checkpoint.json", summary, cfg.overwrite)
    return summary


__all__ = [
    "DEFAULT_BASE",
    "DEFAULT_OUT_DIR",
    "EXPERIMENT",
    "VALUE_PREFIX",
    "ValueHeadConfig",
    "build_optimizer",
    "check_frozen_grads",
    "check_optimizer_params",
    "fit_value_head",
    "freeze_all_but_value_head",
    "frozen_state_sha256",
    "run_dir",
    "train_value_head",
]
