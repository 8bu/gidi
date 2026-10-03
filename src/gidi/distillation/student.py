"""Distillation students: small RoBERTa encoders on BamiBERT's vocabulary.

Three architectures (:data:`STUDENTS`), all with BamiBERT's vocabulary, special-token ids and
tokenizer (20,481 tokens, pad id 1), so teacher and student see byte-identical ``input_ids`` and
the teacher's per-token logits line up one-to-one with the student's. All use gelu, dropout 0.1,
layer-norm eps 1e-5, ``type_vocab_size`` 1, absolute positions, and the same two linear heads.

* ``student-4x256`` (distillation-v1): 4 layers, hidden 256, 4 heads of 64, FFN 1024.
  ``max_position_embeddings`` is ``max_length + pad_token_id + 1`` (34 for max_length 32):
  RoBERTa position ids start at ``pad_token_id + 1``.
* ``student-4x768`` (distillation-v2, a diagnostic, not a deployment shape): 4 layers, hidden
  768, 12 heads, FFN 3072, i.e. BamiBERT's layer shape. ``max_position_embeddings`` stays
  BamiBERT's (2,050) so the position table can be copied whole instead of truncated; only the
  first ``max_length + 2`` rows are ever used.
* ``student-3x768`` (compression-v2): ``student-4x768`` with 3 layers.

Initialization (``init``):

* ``random`` (option C of distillation-v1): HF ``_init_weights`` (normal, ``initializer_range``
  0.02) for the encoder and default ``nn.Linear`` init for the two heads, under
  ``set_seed(seed)``. Every arm of a seed starts from bit-identical weights.
* ``pretrained``: the model is first built exactly as ``random`` (so the heads of a seed are
  bit-identical across inits: **fresh seeded heads**, never the teacher's), then every encoder
  tensor is overwritten by :func:`copy_pretrained_encoder` from the pretrained BamiBERT encoder
  (the checkpoint the teacher was fine-tuned from, not the fine-tuned teacher): word, position
  and token-type embeddings, the embedding layer norm, and whole transformer blocks chosen by
  :func:`layer_map`. Every copied tensor must be shape-identical; nothing is projected,
  truncated or averaged, and every student encoder tensor must be covered. Only the 768-wide
  students are shape-compatible with BamiBERT.

The 4x256 student has no pretrained option: no pretrained 4x256 encoder shares BamiBERT's
vocabulary, and the teacher's 768-wide layers cannot be copied into 256-wide ones without an
approximation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoConfig, PretrainedConfig, RobertaConfig

from gidi.modeling.model import GidiMultiTaskModel
from gidi.training.train import set_seed

TEACHER_TOKENIZER = "Qualcomm-AI-Research/BamiBERT"
DROPOUT = 0.1
INITS = ("random", "pretrained")
INIT_METHODS = {
    "random": "random-hf-init-seeded",
    "pretrained": "pretrained-bamibert-evenly-spaced-layers+seeded-heads",
}


@dataclass(frozen=True)
class StudentSpec:
    name: str
    num_layers: int
    hidden_size: int
    num_heads: int
    intermediate_size: int
    # Keep the base model's position table size (needed for a shape-identical copy).
    base_positions: bool


STUDENTS = {
    spec.name: spec
    for spec in (
        StudentSpec("student-4x256", 4, 256, 4, 1024, base_positions=False),
        StudentSpec("student-4x768", 4, 768, 12, 3072, base_positions=True),
        StudentSpec("student-3x768", 3, 768, 12, 3072, base_positions=True),
    )
}
DEFAULT_STUDENT = "student-4x256"

_LAYER_KEY = re.compile(r"^encoder\.layer\.(\d+)\.(.+)$")


def student_config(
    tokenizer_name: str = TEACHER_TOKENIZER,
    max_length: int = 32,
    *,
    student: str = DEFAULT_STUDENT,
    base_config: PretrainedConfig | None = None,
) -> RobertaConfig:
    """Encoder config of ``student``.

    Vocabulary size and pad/bos/eos ids come from ``base_config`` (the BamiBERT config of
    ``tokenizer_name`` when not given), so the student shares the teacher's token ids.
    """
    spec = STUDENTS[student]
    base = base_config if base_config is not None else AutoConfig.from_pretrained(tokenizer_name)
    pad_id = int(base.pad_token_id)
    positions = (
        int(base.max_position_embeddings) if spec.base_positions else max_length + pad_id + 1
    )
    return RobertaConfig(
        vocab_size=int(base.vocab_size),
        pad_token_id=pad_id,
        bos_token_id=int(base.bos_token_id),
        eos_token_id=int(base.eos_token_id),
        hidden_size=spec.hidden_size,
        num_hidden_layers=spec.num_layers,
        num_attention_heads=spec.num_heads,
        intermediate_size=spec.intermediate_size,
        hidden_act="gelu",
        hidden_dropout_prob=DROPOUT,
        attention_probs_dropout_prob=DROPOUT,
        layer_norm_eps=1e-5,
        type_vocab_size=1,
        max_position_embeddings=positions,
        initializer_range=0.02,
        position_embedding_type="absolute",
    )


def build_student(
    seed: int,
    *,
    student: str = DEFAULT_STUDENT,
    tokenizer_name: str = TEACHER_TOKENIZER,
    max_length: int = 32,
    dropout: float = DROPOUT,
    base_config: PretrainedConfig | None = None,
) -> GidiMultiTaskModel:
    """Randomly initialized student; identical weights for every call with the same ``seed``."""
    config = student_config(tokenizer_name, max_length, student=student, base_config=base_config)
    config.hidden_dropout_prob = dropout
    config.attention_probs_dropout_prob = dropout
    set_seed(seed)
    return GidiMultiTaskModel.from_config(config, dropout=dropout, encoder_name=student)


def layer_map(n_source: int, n_student: int) -> list[int]:
    """Source layer copied into each student layer: the top layer of each equal-depth block.

    Student layer ``k`` (0-based) takes source layer ``(k + 1) * n_source // n_student - 1``;
    for 12 -> 4 that is ``[2, 5, 8, 11]`` (1-based 3, 6, 9, 12), so the student keeps the
    source's final layer and evenly spaced intermediate depths.
    """
    if not 1 <= n_student <= n_source:
        raise ValueError(f"cannot map {n_source} source layers onto {n_student}")
    return [(k + 1) * n_source // n_student - 1 for k in range(n_student)]


def copy_pretrained_encoder(student: nn.Module, source: nn.Module) -> dict:
    """Overwrite every tensor of ``student`` (an encoder) with its ``source`` counterpart.

    Non-layer tensors (embeddings, embedding layer norm) map by name; ``encoder.layer.k.*`` maps
    to ``encoder.layer.{layer_map(...)[k]}.*``. Raises if any student tensor has no source
    tensor or a different shape. Returns a provenance report.
    """
    layers = layer_map(int(source.config.num_hidden_layers), int(student.config.num_hidden_layers))
    src = source.state_dict()
    dst = student.state_dict()
    plan: dict[str, str] = {}
    for key in dst:
        match = _LAYER_KEY.match(key)
        plan[key] = f"encoder.layer.{layers[int(match[1])]}.{match[2]}" if match else key
    missing = sorted(k for k, s in plan.items() if s not in src)
    if missing:
        raise ValueError(f"no source tensor for {missing}")
    mismatched = sorted(
        f"{k}: {tuple(dst[k].shape)} vs {tuple(src[s].shape)}"
        for k, s in plan.items()
        if dst[k].shape != src[s].shape
    )
    if mismatched:
        raise ValueError(f"shape mismatch (no projection/truncation allowed): {mismatched}")
    with torch.no_grad():
        for key, src_key in plan.items():
            dst[key].copy_(src[src_key])
    return {
        "layer_map": {str(k): v for k, v in enumerate(layers)},
        "n_tensors": len(plan),
        "n_values": sum(dst[k].numel() for k in plan),
        "tensors": plan,
    }
