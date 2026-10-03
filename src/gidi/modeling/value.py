"""annotation-v2 value span: record contract, BIO alignment, the 3-head model, checkpoints.

The value span is the exact substring of the note that is the monetary amount (``cơm tấm 100``
-> ``100``). It is span selection only: nothing here parses or stores a number.

Merged training record: every v1 field plus ``value`` (``{"text", "start", "end"}`` or ``None``),
``value_status`` (``"complete"`` | ``"uncertain"``) and ``value_provenance`` (``"human"`` |
``"rule"``). Only ``complete`` records supervise the value head: for any other record every
value tag position is ``IGNORE_INDEX`` (the type and target heads still train on it). A
``complete`` record with ``value = None`` is a ``no_amount`` note: every real token is ``O``.

Alignment is the target head's, unchanged: ``gidi.modeling.preprocessing.encode`` is called with
the value spans in place of the target spans, so tokens overlapping the span (after trimming
whitespace off the text they cover) are ``B-VALUE`` then ``I-VALUE``, a span that lost tokens to
truncation keeps its surviving tags, and a token straddling a span edge raises (``strict``) or
is counted.

``GidiValueModel`` is :class:`GidiMultiTaskModel` plus ``value_head``, a ``Linear(hidden, 3)``
reading the same dropout-ed token states as the target head. ``from_v1`` builds it from a model
produced by the v1 recipe so the encoder and the type/target heads are exactly v1's for that
seed, and the value head comes from its own seeded generator.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, PreTrainedTokenizerBase

from gidi.modeling.checkpoint import META_FILE, WEIGHTS_FILE, save_checkpoint
from gidi.modeling.crf import CRFTransitions, LinearChainCRF
from gidi.modeling.model import GidiMultiTaskModel, _build_encoder
from gidi.modeling.preprocessing import (
    IGNORE_INDEX,
    TYPES,
    Encoded,
    encode,
)
from gidi.modeling.tokenization import load_tokenizer

VALUE_TAGS: tuple[str, ...] = ("O", "B-VALUE", "I-VALUE")
NUM_VALUE_TAGS = len(VALUE_TAGS)
ANNOTATION_VERSION = "annotation-v2"
VALUE_STATUSES = ("complete", "uncertain")
VALUE_PROVENANCES = ("human", "rule")
# The value head's generator seed is ``VALUE_HEAD_SEED_BASE + seed``: independent of the global
# RNG stream that the encoder and the type/target heads of the v1 recipe are drawn from.
VALUE_HEAD_SEED_BASE = 1_000_003


# --------------------------------------------------------------------------- records


def validate_value_record(record: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` unless ``record`` carries a well-formed annotation-v2 value block."""
    rid = record.get("id")
    for key in ("value", "value_status", "value_provenance"):
        if key not in record:
            raise ValueError(f"record {rid!r} lacks {key!r}")
    status, provenance = record["value_status"], record["value_provenance"]
    if status not in VALUE_STATUSES:
        raise ValueError(
            f"record {rid!r}: value_status must be in {VALUE_STATUSES}, got {status!r}"
        )
    if provenance not in VALUE_PROVENANCES:
        raise ValueError(
            f"record {rid!r}: value_provenance must be in {VALUE_PROVENANCES}, got {provenance!r}"
        )
    value = record["value"]
    if value is None:
        return
    text = record["text"]
    start, end = value.get("start"), value.get("end")
    if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text)):
        raise ValueError(f"record {rid!r}: bad value offsets {value!r} for {text!r}")
    if text[start:end] != value.get("text"):
        raise ValueError(f"record {rid!r}: value offsets do not reproduce its text: {value!r}")
    if value["text"] != value["text"].strip():
        raise ValueError(f"record {rid!r}: value has leading/trailing whitespace: {value!r}")


def value_supervised(record: Mapping[str, Any]) -> bool:
    """Whether the value head trains/evaluates on ``record`` (``value_status == complete``)."""
    return record.get("value_status") == "complete"


# --------------------------------------------------------------------------- encoding


@dataclass(frozen=True)
class ValueEncoded:
    """``Encoded`` (inputs + target tags) plus the value tag labels of the same batch.

    ``value_labels`` has the shape of ``encoded.tag_labels``: ``0/1/2`` on real tokens of
    ``complete`` records, ``IGNORE_INDEX`` everywhere else (padding, special tokens, and every
    position of a record that is not ``complete``).
    """

    encoded: Encoded
    value_labels: torch.Tensor
    supervised: list[bool]
    value_span_truncated: list[bool]
    value_boundary_mismatch: list[bool]

    def __len__(self) -> int:
        return len(self.encoded)

    @property
    def n_value_span_truncated(self) -> int:
        return sum(self.value_span_truncated)

    @property
    def n_value_boundary_mismatch(self) -> int:
        return sum(self.value_boundary_mismatch)


def encode_value(
    tokenizer: PreTrainedTokenizerBase,
    records: Sequence[Mapping[str, Any]],
    max_length: int,
    *,
    strict: bool = True,
) -> ValueEncoded:
    """Tokenize ``records`` once for the target and once for the value alignment.

    The target alignment is ``encode`` exactly as v1 calls it. ``strict`` applies to the value
    alignment (the target keeps v1's strictness): a token straddling a value edge raises
    ``ValueError`` when true, otherwise it is counted in ``value_boundary_mismatch``.
    """
    texts = [r["text"] for r in records]
    supervised = [value_supervised(r) for r in records]
    encoded = encode(tokenizer, texts, [r["target"] for r in records], max_length)
    values = [r["value"] if sup else None for r, sup in zip(records, supervised, strict=True)]
    venc = encode(tokenizer, texts, values, max_length, strict=strict)
    if not torch.equal(venc.input_ids, encoded.input_ids):
        raise RuntimeError("value and target alignments tokenized the batch differently")
    labels = venc.tag_labels.clone()
    for i, sup in enumerate(supervised):
        if not sup:
            labels[i, :] = IGNORE_INDEX
    return ValueEncoded(
        encoded=encoded,
        value_labels=labels,
        supervised=supervised,
        value_span_truncated=venc.span_truncated,
        value_boundary_mismatch=venc.boundary_mismatch,
    )


@dataclass
class ValuePrepared:
    """One split, encoded once: records, v1 tensors, value labels, type ids."""

    records: list[dict[str, Any]]
    value_encoded: ValueEncoded
    type_ids: torch.Tensor

    @property
    def encoded(self) -> Encoded:
        return self.value_encoded.encoded

    def __len__(self) -> int:
        return len(self.records)


def prepare_value(
    tokenizer: PreTrainedTokenizerBase,
    records: list[dict[str, Any]],
    max_length: int,
    *,
    strict: bool = True,
) -> ValuePrepared:
    for r in records:
        validate_value_record(r)
    venc = encode_value(tokenizer, records, max_length, strict=strict)
    type_ids = torch.tensor([TYPES.index(r["type"]) for r in records], dtype=torch.long)
    return ValuePrepared(records, venc, type_ids)


# --------------------------------------------------------------------------- model


def _seeded_linear(in_features: int, out_features: int, seed: int) -> nn.Linear:
    """``nn.Linear`` with PyTorch's default init, drawn from a private generator.

    The global RNG is left exactly as it was found.
    """
    with torch.random.fork_rng(devices=[]):
        layer = nn.Linear(in_features, out_features)
    generator = torch.Generator().manual_seed(seed)
    bound = 1.0 / math.sqrt(in_features)  # kaiming_uniform(a=sqrt(5)) and the bias bound
    with torch.no_grad():
        for param in (layer.weight, layer.bias):
            param.copy_(torch.empty_like(param).uniform_(-bound, bound, generator=generator))
    return layer


VALUE_HEAD_ARCHS = ("linear", "mlp", "mlp-crf", "adapter-mlp-crf", "encoder-mlp-crf")
MLP_VALUE_HIDDEN = 256
# The second MLP layer's generator seed is offset so the two layers draw independent streams.
MLP_SECOND_LAYER_SEED_OFFSET = 1_000_000
# ... and the CRF transitions (``mlp-crf``) draw from a third one ...
CRF_SEED_OFFSET = 2_000_000
# ... and the value adapter block (``adapter-mlp-crf``) from a fourth.
ADAPTER_SEED_OFFSET = 3_000_000
# The value adapter: one post-LN Transformer encoder block (the RoBERTa layer layout).
ADAPTER_HEADS = 12
ADAPTER_FFN = 1024
ADAPTER_LAYER_NORM_EPS = 1e-5


def build_value_adapter(hidden: int, dropout: float, seed: int | None = None) -> nn.Module:
    """One ``nn.TransformerEncoderLayer``: 12 heads, FFN 1024, GELU, ``dropout`` on attention
    weights, FFN activation and both residual branches, post-LN
    (``x = LN(x + Drop(MHA(x)))``, ``x = LN(x + Drop(FFN(x)))``).

    PyTorch's default init; with ``seed`` drawn from ``torch.manual_seed(seed)`` inside a forked
    RNG, so the global RNG is left exactly as it was found."""
    with torch.random.fork_rng(devices=[]):
        if seed is not None:
            torch.manual_seed(seed)
        return nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=ADAPTER_HEADS,
            dim_feedforward=ADAPTER_FFN,
            dropout=dropout,
            activation="gelu",
            layer_norm_eps=ADAPTER_LAYER_NORM_EPS,
            batch_first=True,
            norm_first=False,
        )


class CRFValueHead(nn.Module):
    """``mlp`` emission scores plus a linear-chain CRF over them. ``adapter-mlp-crf`` first runs
    the token states through a value-only Transformer block (:func:`build_value_adapter`);
    ``encoder-mlp-crf`` ignores the shared states and runs its own value encoder (a clone of the
    v1 encoder) on the same inputs, followed by dropout as in the v1 heads.

    ``forward`` returns the emission scores ``[batch, seq, tags]`` (the model's ``value_logits``);
    training uses ``crf.nll`` and decoding Viterbi over the CRF transitions. Neither the adapter
    nor the value encoder modifies the shared encoder states the type/target heads read.
    """

    def __init__(
        self,
        emissions: nn.Module,
        crf: LinearChainCRF,
        adapter: nn.Module | None = None,
        encoder: nn.Module | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.adapter = adapter
        self.encoder = encoder
        self.encoder_dropout = nn.Dropout(dropout) if encoder is not None else None
        self.emissions = emissions
        self.crf = crf

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.encoder is not None:
            if input_ids is None or attention_mask is None:
                raise ValueError("the value encoder needs input_ids and the attention mask")
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden = self.encoder_dropout(out.last_hidden_state)
        if self.adapter is not None:
            if attention_mask is None:
                raise ValueError("the value adapter needs the attention mask")
            hidden = self.adapter(hidden, src_key_padding_mask=attention_mask == 0)
        return self.emissions(hidden)


def build_value_head(
    arch: str,
    hidden: int,
    num_value_tags: int,
    dropout: float,
    seed: int | None = None,
    encoder: nn.Module | None = None,
) -> nn.Module:
    """``linear``: ``Linear(hidden, tags)``. ``mlp``: ``Linear(hidden, 256) -> GELU -> Dropout
    -> Linear(256, tags)``. ``mlp-crf``: the ``mlp`` emissions plus a :class:`LinearChainCRF`.
    ``adapter-mlp-crf``: a value-only Transformer block before the ``mlp-crf`` head.
    ``encoder-mlp-crf``: a deep copy of ``encoder`` (same weights) feeding the ``mlp-crf`` head.
    With ``seed`` each Linear is drawn by :func:`_seeded_linear`."""
    if arch == "linear":
        if seed is None:
            return nn.Linear(hidden, num_value_tags)
        return _seeded_linear(hidden, num_value_tags, seed)
    if arch in ("mlp", "mlp-crf", "adapter-mlp-crf", "encoder-mlp-crf"):
        if seed is None:
            first, second = (
                nn.Linear(hidden, MLP_VALUE_HIDDEN),
                nn.Linear(MLP_VALUE_HIDDEN, num_value_tags),
            )
        else:
            first = _seeded_linear(hidden, MLP_VALUE_HIDDEN, seed)
            second = _seeded_linear(
                MLP_VALUE_HIDDEN, num_value_tags, seed + MLP_SECOND_LAYER_SEED_OFFSET
            )
        mlp = nn.Sequential(first, nn.GELU(), nn.Dropout(dropout), second)
        if arch == "mlp":
            return mlp
        crf_seed = None if seed is None else seed + CRF_SEED_OFFSET
        crf = LinearChainCRF(num_value_tags, crf_seed)
        if arch == "mlp-crf":
            return CRFValueHead(mlp, crf)
        if arch == "encoder-mlp-crf":
            if encoder is None:
                raise ValueError("encoder-mlp-crf needs the encoder to clone")
            return CRFValueHead(mlp, crf, encoder=copy.deepcopy(encoder), dropout=dropout)
        adapter_seed = None if seed is None else seed + ADAPTER_SEED_OFFSET
        return CRFValueHead(mlp, crf, build_value_adapter(hidden, dropout, adapter_seed))
    raise ValueError(f"unknown value head arch {arch!r}; expected one of {VALUE_HEAD_ARCHS}")


def value_crf(model: nn.Module) -> CRFTransitions | None:
    """The CRF transitions of a CRF value head (for Viterbi decoding), else ``None``."""
    head = getattr(model, "value_head", None)
    return head.crf.numpy_transitions() if isinstance(head, CRFValueHead) else None


class GidiValueModel(GidiMultiTaskModel):
    """Type head + target BIO head + value BIO head over one shared encoder.

    ``forward`` returns a plain ``(type_logits, tag_logits, value_logits)`` tuple (traces to
    ONNX); ``tag_logits`` is the target head, ``value_logits`` is ``[batch, seq, 3]`` in the
    order of :data:`VALUE_TAGS`.
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_types: int = len(TYPES),
        num_tags: int = 3,
        dropout: float = 0.1,
        encoder_name: str = "custom",
        num_value_tags: int = NUM_VALUE_TAGS,
        value_head_arch: str = "linear",
    ) -> None:
        super().__init__(encoder, num_types, num_tags, dropout, encoder_name)
        self.value_head = build_value_head(
            value_head_arch,
            int(encoder.config.hidden_size),
            num_value_tags,
            dropout,
            encoder=encoder,
        )
        self.num_value_tags = num_value_tags
        self.value_head_arch = value_head_arch

    @classmethod
    def from_v1(
        cls, base: GidiMultiTaskModel, seed: int, value_head_arch: str = "linear"
    ) -> GidiValueModel:
        """Wrap a v1-recipe model: its encoder and type/target heads move over unchanged.

        The value head is initialised from ``VALUE_HEAD_SEED_BASE + seed`` with PyTorch's default
        ``Linear`` init. Neither the global RNG state nor any v1 tensor is touched, so a v2 run
        of seed ``s`` starts from v1's weights for seed ``s`` (heads bit-identical) and sees
        the same dropout stream.
        """
        hidden = int(base.encoder.config.hidden_size)
        with torch.random.fork_rng(devices=[]):
            model = cls(
                base.encoder,
                base.num_types,
                base.num_tags,
                base.dropout_p,
                base.encoder_name,
                value_head_arch=value_head_arch,
            )
        model.type_head.load_state_dict(base.type_head.state_dict())
        model.tag_head.load_state_dict(base.tag_head.state_dict())
        model.value_head = build_value_head(
            value_head_arch,
            hidden,
            model.num_value_tags,
            base.dropout_p,
            VALUE_HEAD_SEED_BASE + seed,
            encoder=base.encoder,  # encoder-mlp-crf: cloned with the trained v1 weights
        )
        model.train(base.training)
        return model

    @classmethod
    def from_config(  # type: ignore[override]
        cls,
        config,
        num_types: int = len(TYPES),
        num_tags: int = 3,
        dropout: float = 0.1,
        encoder_name: str = "custom",
        num_value_tags: int = NUM_VALUE_TAGS,
        value_head_arch: str = "linear",
    ) -> GidiValueModel:
        encoder = _build_encoder(config)
        return cls(
            encoder, num_types, num_tags, dropout, encoder_name, num_value_tags, value_head_arch
        )

    def forward(  # type: ignore[override]
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden = self.dropout(hidden)
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        if isinstance(self.value_head, CRFValueHead):
            # the value branch reads the same states (or its own encoder on the same inputs);
            # any adapter / value-encoder output is a separate tensor
            value_logits = self.value_head(hidden, attention_mask, input_ids)
        else:
            value_logits = self.value_head(hidden)
        return self.type_head(pooled), self.tag_head(hidden), value_logits


# --------------------------------------------------------------------------- checkpoints


def save_value_checkpoint(
    model: GidiValueModel,
    tokenizer: PreTrainedTokenizerBase,
    out_dir: str | Path,
    meta: dict[str, Any],
) -> Path:
    """``save_checkpoint`` layout; the metadata also records the value head."""
    return save_checkpoint(
        model,
        tokenizer,
        out_dir,
        {
            "annotation_version": ANNOTATION_VERSION,
            "heads": ["type", "target", "value"],
            "value_tags": list(VALUE_TAGS),
            "num_value_tags": model.num_value_tags,
            # only non-default heads are recorded, so linear-head checkpoints stay byte-identical
            **(
                {}
                if model.value_head_arch == "linear"
                else {"value_head_arch": model.value_head_arch}
            ),
            **meta,
        },
    )


def load_value_checkpoint(
    out_dir: str | Path,
) -> tuple[GidiValueModel, PreTrainedTokenizerBase, dict[str, Any]]:
    """Rebuild ``(model, tokenizer, meta)`` from a ``save_value_checkpoint`` dir (CPU, eval)."""
    from safetensors.torch import load_model

    out = Path(out_dir)
    meta = json.loads((out / META_FILE).read_text(encoding="utf-8"))
    if "value_tags" not in meta:
        raise ValueError(f"{out} has no value head (a v1 checkpoint); use load_checkpoint")
    if list(meta["value_tags"]) != list(VALUE_TAGS):
        raise ValueError(f"{out}: unexpected value tags {meta['value_tags']}")
    model = GidiValueModel.from_config(
        AutoConfig.from_pretrained(out),
        num_types=meta["num_types"],
        num_tags=meta["num_tags"],
        dropout=meta["dropout"],
        encoder_name=meta["encoder"],
        num_value_tags=meta["num_value_tags"],
        value_head_arch=meta.get("value_head_arch", "linear"),
    )
    load_model(model, str(out / WEIGHTS_FILE), strict=True)
    model.eval()
    return model, load_tokenizer(str(out)), meta
