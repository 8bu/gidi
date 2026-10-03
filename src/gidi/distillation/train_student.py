"""Train a distillation student, supervised-only or distilled from cached teacher logits.

One call to :func:`train_student` trains one (arm, student, init, seed) run and writes::

    <out_dir>/<arm>/<variant>/seed<N>/
        config.json            resolved config: student architecture, KD config, arm, init, env
        train_log.jsonl        one row per epoch (loss components, in-sample validation metrics)
        metrics.json           final-epoch validation + frozen-test metrics
        predictions_test.jsonl id, text, gold/pred type, gold/pred span
        checkpoint.json        summary: param count, throughput, weights path
    <weights_dir>/<arm>/<variant>/seed<N>/   ``save_checkpoint`` layout

``variant`` is the student name (``student-4x256``, ``student-4x768``) for random init and
``<student>-pretrained`` for pretrained init (see :mod:`gidi.distillation.student`), suffixed
``-pos<max_length>`` when the position table is truncated and ``-vocab-<policy>`` when the
vocabulary is pruned (compression-v1, :mod:`gidi.compression`), and ``-ffn-<map>`` when each
layer's FFN keeps only the neurons of an ``ffn_map.json`` (compression-v3). Compression is applied
to the initial model; with a pruned vocabulary the cached input ids are remapped to the new ids
and re-verified against the pruned tokenizer.

All arms train on the same cached tensors (inputs, hard labels), start from identical weights
for a given (student, init, seed), and use the same shuffles; supervised and distilled differ
only in ``KDConfig.alpha_hard``. The schedule is a fixed epoch budget with linear warmup/decay
over all epochs, no early stopping, and the LAST epoch's weights are kept: nothing is selected
on a held-out set. The training data already contains the validation split, so the per-epoch
validation metrics are in-sample diagnostics only. The frozen test split is evaluated once,
after training, and never trained on.
"""

from __future__ import annotations

import json
import logging
import platform
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
import transformers
from huggingface_hub import try_to_load_from_cache
from transformers import get_linear_schedule_with_warmup

from gidi.annotation.schema import load_config
from gidi.compression.ffn import prune_ffn
from gidi.compression.positions import truncate_positions
from gidi.compression.vocab import load_vocab_spec, prune_vocab, remap_ids
from gidi.corpus.jsonl import write_jsonl
from gidi.device import resolve_device
from gidi.distillation.guard import assert_distillation_trainable
from gidi.distillation.loss import KDConfig, kd_loss
from gidi.distillation.student import (
    DEFAULT_STUDENT,
    INIT_METHODS,
    INITS,
    STUDENTS,
    TEACHER_TOKENIZER,
    build_student,
    copy_pretrained_encoder,
)
from gidi.distillation.targets import MANIFEST_FILE, load_targets, sha256_file, verify_encoding
from gidi.modeling.checkpoint import save_checkpoint
from gidi.modeling.model import GidiMultiTaskModel
from gidi.modeling.preprocessing import TYPES
from gidi.modeling.tokenization import load_tokenizer
from gidi.training.train import (
    Prepared,
    TrainConfig,
    _batches,
    _fmt,
    _optimizer,
    _sync,
    _write_json,
    evaluate_split,
    load_split,
    prepare,
    selection_score,
)

log = logging.getLogger(__name__)

ARMS = ("supervised", "distilled")
DEFAULT_TARGETS_DIR = "experiments/distillation-v1/teacher-targets"
DEFAULT_SPLITS_DIR = "datasets/annotation-v1/distillation-v1"
DEFAULT_OUT_DIR = "experiments/distillation-v1/runs"
DEFAULT_WEIGHTS_DIR = "models/distillation-v1"
LOSS_KEYS = ("total", "type_loss", "span_loss", "type_hard", "tag_hard", "type_kl", "tag_kl")


@dataclass
class StudentTrainConfig:
    arm: str
    seed: int
    student: str = DEFAULT_STUDENT
    init: str = "random"
    kd: KDConfig = field(default_factory=KDConfig)
    lr: float = 5e-4
    head_lr: float = 1e-3
    epochs: int = 40
    batch_size: int = 8
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    dropout: float = 0.1
    max_grad_norm: float = 1.0
    max_length: int = 32
    tokenizer_name: str = TEACHER_TOKENIZER
    targets_dir: str = DEFAULT_TARGETS_DIR
    splits_dir: str = DEFAULT_SPLITS_DIR
    out_dir: str = DEFAULT_OUT_DIR
    weights_dir: str = DEFAULT_WEIGHTS_DIR
    max_steps: int | None = None  # smoke runs: stop after this many optimizer steps
    device: str = "mps"
    overwrite: bool = False
    # compression-v1: applied to the initial model, before training (see _compress)
    truncate_positions: bool = False
    vocab_spec: str | None = None  # models/compression-v1/vocab/<policy>/
    ffn_map: str | None = None  # compression-v3: models/compression-v3/ffn/<policy>/


def effective_kd(cfg: StudentTrainConfig) -> KDConfig:
    """The KD config actually used: the supervised arm forces ``alpha_hard = 1``."""
    if cfg.arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {cfg.arm!r}")
    if cfg.arm == "supervised":
        return replace(cfg.kd, alpha_hard=1.0)
    if not cfg.kd.uses_teacher:
        raise ValueError("the distilled arm needs alpha_hard < 1 (it would be supervised)")
    return cfg.kd


def variant(cfg: StudentTrainConfig) -> str:
    name = cfg.student if cfg.init == "random" else f"{cfg.student}-{cfg.init}"
    if cfg.truncate_positions:
        name += f"-pos{cfg.max_length}"
    if cfg.vocab_spec:
        name += f"-vocab-{Path(cfg.vocab_spec).name}"
    if cfg.ffn_map:
        name += f"-ffn-{Path(cfg.ffn_map).name}"
    return name


def run_dir(cfg: StudentTrainConfig) -> Path:
    return Path(cfg.out_dir) / cfg.arm / variant(cfg) / f"seed{cfg.seed}"


def weights_dir(cfg: StudentTrainConfig) -> Path:
    return Path(cfg.weights_dir) / cfg.arm / variant(cfg) / f"seed{cfg.seed}"


def _pretrained_init(model: GidiMultiTaskModel, name: str) -> dict[str, Any]:
    """Copy the pretrained ``name`` encoder into ``model.encoder``; returns provenance."""
    source = GidiMultiTaskModel.from_encoder(name).encoder
    report = copy_pretrained_encoder(model.encoder, source)
    cached = try_to_load_from_cache(name, "model.safetensors")
    return {
        "source": name,
        "source_commit": getattr(source.config, "_commit_hash", None),
        "source_weights_sha256": sha256_file(Path(cached)) if isinstance(cached, str) else None,
        **report,
    }


def _compress(
    model: GidiMultiTaskModel, cfg: StudentTrainConfig, kept_old_ids: list[int] | None
) -> tuple[GidiMultiTaskModel, dict[str, Any] | None]:
    """Truncate the position table, prune the vocabulary and/or the FFNs of the initial model.

    Every transform copies the kept rows/neurons exactly and leaves the heads untouched. Position
    truncation and vocab pruning keep the function (on in-vocabulary inputs) of the uncompressed
    run of the same seed; FFN pruning removes whole neurons.
    """
    if not cfg.truncate_positions and kept_old_ids is None and not cfg.ffn_map:
        return model, None
    report: dict[str, Any] = {}
    if cfg.truncate_positions:
        rows = model.encoder.config.max_position_embeddings
        model = truncate_positions(model, cfg.max_length)
        report["positions"] = {
            "rows_before": rows,
            "rows_after": model.encoder.config.max_position_embeddings,
        }
    if kept_old_ids is not None:
        rows = model.encoder.config.vocab_size
        model = prune_vocab(model, kept_old_ids)
        report["vocab"] = {
            "spec": cfg.vocab_spec,
            "map_sha256": sha256_file(Path(cfg.vocab_spec) / "vocab_map.json"),
            "rows_before": rows,
            "rows_after": model.encoder.config.vocab_size,
        }
    if cfg.ffn_map:
        path = Path(cfg.ffn_map) / "ffn_map.json"
        spec = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "student": cfg.student,
            "init": cfg.init,
            "max_length": cfg.max_length,
            "truncate_positions": cfg.truncate_positions,
            "vocab_map_sha256": report.get("vocab", {}).get("map_sha256"),
        }
        if spec["base"] != expected:
            raise ValueError(f"{path} was built for {spec['base']}, this run is {expected}")
        size = model.encoder.config.intermediate_size
        model = prune_ffn(model, spec["keep"])
        report["ffn"] = {
            "map": cfg.ffn_map,
            "map_sha256": sha256_file(path),
            "criterion": spec["criterion"],
            "intermediate_before": size,
            "intermediate_after": model.encoder.config.intermediate_size,
        }
    return model, report


def initial_model(
    cfg: StudentTrainConfig, kept_old_ids: list[int] | None
) -> tuple[GidiMultiTaskModel, dict[str, Any] | None, dict[str, Any] | None]:
    """The seeded initial model of ``cfg``: built, pretrained-initialized, then compressed."""
    model = build_student(
        cfg.seed,
        student=cfg.student,
        tokenizer_name=cfg.tokenizer_name,
        max_length=cfg.max_length,
        dropout=cfg.dropout,
    )
    init_report = _pretrained_init(model, cfg.tokenizer_name) if cfg.init == "pretrained" else None
    model, compression = _compress(model, cfg, kept_old_ids)
    return model, init_report, compression


def _nonempty(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def train_student(cfg: StudentTrainConfig) -> dict[str, Any]:
    """Train one (arm, seed) run end to end; returns the ``checkpoint.json`` payload."""
    kd = effective_kd(cfg)
    if cfg.student not in STUDENTS or cfg.init not in INITS:
        raise ValueError(f"unknown student/init: {cfg.student!r}/{cfg.init!r}")
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

    # --- data: the cache is the single source of inputs and hard labels for both arms ---
    targets = load_targets(cfg.targets_dir)
    splits_dir = Path(cfg.splits_dir)
    train_records = load_split(splits_dir / "train.jsonl")
    limit = targets.manifest["data"]["limit"]
    if limit is None:
        cache_data_sha = targets.manifest["data"]["sha256"]
        if sha256_file(splits_dir / "train.jsonl") != cache_data_sha:
            raise ValueError(
                f"{splits_dir / 'train.jsonl'} is not the file the targets were built on"
            )
    else:
        train_records = train_records[:limit]
    val_records = load_split(splits_dir / "validation.jsonl")
    test_records = load_split(splits_dir / "test.jsonl")  # evaluation only, not guarded
    assert_distillation_trainable([*train_records, *val_records])
    if cfg.vocab_spec:
        # Pruned vocabulary: the cached old ids are remapped; verify_encoding then proves the
        # pruned tokenizer encodes every training note to exactly the remapped ids.
        tokenizer, kept_old_ids = load_vocab_spec(cfg.vocab_spec)
        targets = replace(targets, input_ids=remap_ids(targets.input_ids, kept_old_ids))
    else:
        tokenizer, kept_old_ids = load_tokenizer(cfg.tokenizer_name), None
    reencoded = verify_encoding(targets, train_records, tokenizer, cfg.max_length)
    train = Prepared(
        train_records,
        replace(
            reencoded.encoded,
            input_ids=targets.input_ids,
            attention_mask=targets.attention_mask,
            tag_labels=targets.tag_labels,
        ),
        targets.type_ids,
    )
    data = {
        "train": train,
        "validation": prepare(tokenizer, val_records, cfg.max_length),
        "test": prepare(tokenizer, test_records, cfg.max_length),
    }
    truncation = {
        name: {
            "n": len(d),
            "truncated": d.encoded.n_truncated,
            "span_truncated": d.encoded.n_span_truncated,
        }
        for name, d in data.items()
    }

    # --- model: identical initial weights for every arm of a (student, init, seed) ---
    model, init_report, compression = initial_model(cfg, kept_old_ids)
    param_count = model.param_count()
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    model.to(device)
    opt_cfg = TrainConfig(
        model=variant(cfg),
        lr=cfg.lr,
        seed=cfg.seed,
        head_lr=cfg.head_lr,
        weight_decay=cfg.weight_decay,
    )
    optimizer = _optimizer(model, opt_cfg)
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
    cache_manifest_sha = sha256_file(Path(cfg.targets_dir) / MANIFEST_FILE)
    _write_json(
        out / "config.json",
        {
            **{k: v for k, v in asdict(cfg).items() if k != "kd"},
            "kd_requested": asdict(cfg.kd),
            "kd": asdict(kd),
            "student_config": model.encoder.config.to_dict(),
            "init_method": INIT_METHODS[cfg.init],
            "init_report": init_report,
            "compression": compression,
            "resolved_device": str(device),
            "param_count": param_count,
            "param_bytes_fp32": param_bytes,
            "targets_manifest_sha256": cache_manifest_sha,
            "targets": {
                k: targets.manifest[k] for k in ("teacher", "data", "n", "teacher_vs_gold")
            },
            "split_sizes": {name: len(d) for name, d in data.items()},
            "validation_is_in_sample": True,
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

    teacher_type = targets.teacher_type_logits
    teacher_tag = targets.teacher_tag_logits
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
        for idx, input_ids, mask, tags, types in _batches(train, cfg.batch_size, order):
            input_ids, mask, tags, types = (x.to(device) for x in (input_ids, mask, tags, types))
            type_logits, tag_logits = model(input_ids, mask)
            if kd.uses_teacher:
                width = input_ids.shape[1]
                t_type = teacher_type[idx].to(device)
                t_tag = teacher_tag[idx, :width].to(device)
            else:
                t_type = t_tag = None
            losses = kd_loss(type_logits, tag_logits, types, tags, kd, t_type, t_tag)
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

        t1 = time.perf_counter()
        val_metrics, _, _ = evaluate_split(model, data["validation"], device)
        eval_seconds = time.perf_counter() - t1
        score = selection_score(val_metrics)
        overall = val_metrics["overall"]
        row = {
            "epoch": epoch,
            "step": step,
            "train_loss": float(sums["total"].item() / n_batches),
            **{f"train_{k}": float(sums[k].item() / n_batches) for k in LOSS_KEYS[1:]},
            "val_score": score,
            "val_type_macro_f1": overall["type"]["macro_f1"],
            "val_type_accuracy": overall["type"]["accuracy"],
            "val_span_f1": overall["target"]["f1"],
            "val_span_exact_match": overall["target"]["exact_match"],
            "lr": scheduler.get_last_lr()[0],
            "epoch_train_sec": epoch_seconds,
            "eval_sec": eval_seconds,
            "wall_sec": time.perf_counter() - started,
        }
        log_rows.append(row)
        log.info(
            "[%s %s seed%d] epoch %d step %d loss %.4f (in-sample val: type F1 %s, span F1 %s) "
            "%.1fs",
            cfg.arm,
            variant(cfg),
            cfg.seed,
            epoch,
            step,
            row["train_loss"],
            _fmt(row["val_type_macro_f1"]),
            _fmt(row["val_span_f1"]),
            epoch_seconds,
        )
        if cfg.max_steps is not None and step >= cfg.max_steps:
            break

    # The last epoch's weights are final: no selection on any held-out set.
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
    checkpoint_path = save_checkpoint(
        model,
        tokenizer,
        wdir,
        {
            "arm": cfg.arm,
            "student": cfg.student,
            "kd": asdict(kd),
            "seed": cfg.seed,
            "epochs": epochs_run,
            "init": INIT_METHODS[cfg.init],
            "init_layer_map": init_report["layer_map"] if init_report else None,
            "compression": compression,
            "lr": cfg.lr,
            "max_length": cfg.max_length,
            "targets_manifest_sha256": cache_manifest_sha,
        },
    )
    summary = {
        "arm": cfg.arm,
        "variant": variant(cfg),
        "seed": cfg.seed,
        "kd": asdict(kd),
        "final_epoch": epochs_run,
        "test_type_macro_f1": test_metrics["overall"]["type"]["macro_f1"],
        "test_span_f1": test_metrics["overall"]["target"]["f1"],
        "param_count": param_count,
        "param_bytes_fp32": param_bytes,
        "steps": step,
        "device": str(device),
        "wall_time_sec": wall,
        "train_time_sec": train_seconds,
        "sec_per_epoch": train_seconds / epochs_run,
        "train_examples_per_sec": trained_examples / train_seconds if train_seconds else None,
        "checkpoint_path": str(checkpoint_path),
    }
    _write_json(out / "checkpoint.json", summary, cfg.overwrite)
    return summary
