"""Shared-encoder multi-task model: transaction type (sequence) + BIO target tags (token)."""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel, PretrainedConfig

from gidi.modeling.preprocessing import TAGS, TYPES


def _load_encoder(name_or_path: str) -> nn.Module:
    """Load a bare encoder; the pooler is unused here so drop it when the class allows it."""
    try:
        return AutoModel.from_pretrained(name_or_path, add_pooling_layer=False)
    except TypeError:
        return AutoModel.from_pretrained(name_or_path)


def _build_encoder(config: PretrainedConfig) -> nn.Module:
    try:
        return AutoModel.from_config(config, add_pooling_layer=False)
    except TypeError:
        return AutoModel.from_config(config)


class GidiMultiTaskModel(nn.Module):
    """Encoder with two linear heads sharing dropout.

    The type head reads the attention-masked mean of the token hidden states (it trains much
    faster than the ``<s>``/CLS state on ~500 examples, see experiments/baseline-v1/report.md);
    the tag head reads every token. ``forward`` returns a plain ``(type_logits, tag_logits)``
    tuple so it traces to ONNX.
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_types: int = len(TYPES),
        num_tags: int = len(TAGS),
        dropout: float = 0.1,
        encoder_name: str = "custom",
    ) -> None:
        super().__init__()
        hidden = int(encoder.config.hidden_size)
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.type_head = nn.Linear(hidden, num_types)
        self.tag_head = nn.Linear(hidden, num_tags)
        self.num_types = num_types
        self.num_tags = num_tags
        self.dropout_p = dropout
        self.encoder_name = encoder_name

    @classmethod
    def from_encoder(
        cls,
        name_or_path: str,
        num_types: int = len(TYPES),
        num_tags: int = len(TAGS),
        dropout: float = 0.1,
    ) -> GidiMultiTaskModel:
        """Build from pretrained encoder weights (hub id or local directory)."""
        encoder = _load_encoder(name_or_path)
        return cls(encoder, num_types, num_tags, dropout, encoder_name=name_or_path)

    @classmethod
    def from_config(
        cls,
        config: PretrainedConfig,
        num_types: int = len(TYPES),
        num_tags: int = len(TAGS),
        dropout: float = 0.1,
        encoder_name: str = "custom",
    ) -> GidiMultiTaskModel:
        """Build with a randomly initialized encoder (tests, checkpoint loading)."""
        return cls(_build_encoder(config), num_types, num_tags, dropout, encoder_name)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden = self.dropout(hidden)
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.type_head(pooled), self.tag_head(hidden)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
