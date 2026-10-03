"""Distillation loss for the type head and the token BIO tag head.

With ``a = alpha_hard``, ``T = temperature``::

    L = type_weight * [ a * CE(s_type, y_type) + (1 - a) * T^2 * KL_type ]
      + span_weight * [ a * CE_tok(s_tag, y_tag) + (1 - a) * T^2 * KL_tok ]

* ``CE`` is the usual mean cross-entropy over the batch; ``CE_tok`` is the mean over the
  positions with ``y_tag != -100`` (real tokens: no padding, no special tokens).
* ``KL_type = KL(softmax(t_type / T) || softmax(s_type / T))``, batch mean.
* ``KL_tok`` is the same per-token KL, averaged over exactly the positions with
  ``y_tag != -100`` (the positions of the hard CE). Teacher and student logits at ignored
  positions therefore never influence the loss or its gradient.
* ``T^2`` keeps the soft-term gradient magnitude comparable across temperatures (Hinton et al.).
* ``t_*`` are the cached raw teacher logits; the temperature is applied here, never in the cache.

Supervised-only training is ``alpha_hard = 1``: the KL terms are not computed and no teacher
tensors are needed, and ``L`` equals ``CE(type) + CE(tags)`` for unit weights, the loss of
``gidi.training.train.train_run``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from gidi.modeling.preprocessing import IGNORE_INDEX


@dataclass
class KDConfig:
    temperature: float = 2.0
    alpha_hard: float = 0.5
    type_weight: float = 1.0
    span_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if not 0.0 <= self.alpha_hard <= 1.0:
            raise ValueError(f"alpha_hard must be in [0, 1], got {self.alpha_hard}")

    @property
    def uses_teacher(self) -> bool:
        return self.alpha_hard < 1.0


def _soft_kl(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float
) -> torch.Tensor:
    """``KL(softmax(teacher / T) || softmax(student / T))`` averaged over the rows of ``[P, C]``."""
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.log_softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
        log_target=True,
    )


def kd_loss(
    type_logits: torch.Tensor,
    tag_logits: torch.Tensor,
    type_ids: torch.Tensor,
    tag_labels: torch.Tensor,
    cfg: KDConfig,
    teacher_type_logits: torch.Tensor | None = None,
    teacher_tag_logits: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Combined loss. ``total`` carries the gradient; every other entry is a detached copy.

    Shapes: ``type_logits [B, C]``, ``tag_logits [B, L, K]``, ``type_ids [B]``,
    ``tag_labels [B, L]`` (``-100`` off real tokens); teacher logits match the student's.
    Logged components (all before the ``alpha``/``T^2``/weight scaling): ``type_hard``,
    ``tag_hard``, ``type_kl``, ``tag_kl`` (zero when ``alpha_hard == 1``); ``type_loss`` and
    ``span_loss`` are the two bracketed, alpha/T^2-scaled terms before ``type_weight`` /
    ``span_weight``.
    """
    a, temp = cfg.alpha_hard, cfg.temperature
    keep = tag_labels != IGNORE_INDEX
    type_hard = F.cross_entropy(type_logits, type_ids)
    # Boolean selection (not ignore_index) so the empty case is a clean zero, not NaN.
    student_tokens = tag_logits[keep]
    tag_hard = (
        F.cross_entropy(student_tokens, tag_labels[keep])
        if student_tokens.shape[0]
        else tag_logits.sum() * 0.0
    )
    zero = type_hard.new_zeros(())
    type_kl = tag_kl = zero

    type_loss = a * type_hard
    span_loss = a * tag_hard
    if cfg.uses_teacher:
        if teacher_type_logits is None or teacher_tag_logits is None:
            raise ValueError("alpha_hard < 1 needs teacher_type_logits and teacher_tag_logits")
        type_kl = _soft_kl(type_logits, teacher_type_logits, temp)
        if student_tokens.shape[0]:
            tag_kl = _soft_kl(student_tokens, teacher_tag_logits[keep], temp)
        else:
            tag_kl = tag_logits.sum() * 0.0
        type_loss = type_loss + (1.0 - a) * temp**2 * type_kl
        span_loss = span_loss + (1.0 - a) * temp**2 * tag_kl

    total = cfg.type_weight * type_loss + cfg.span_weight * span_loss
    return {
        "total": total,
        "type_loss": type_loss.detach(),
        "span_loss": span_loss.detach(),
        "type_hard": type_hard.detach(),
        "tag_hard": tag_hard.detach(),
        "type_kl": type_kl.detach(),
        "tag_kl": tag_kl.detach(),
    }
