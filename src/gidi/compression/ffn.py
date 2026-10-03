"""Structured FFN pruning: keep ``k`` of each layer's intermediate neurons (compression-v3).

A RoBERTa block's feed-forward network is ``out = W2 · gelu(W1 · x + b1) + b2`` with
``W1: [I, H]`` (``intermediate.dense``) and ``W2: [H, I]`` (``output.dense``). Intermediate
neuron ``j`` owns row ``j`` of ``W1``, entry ``j`` of ``b1`` and column ``j`` of ``W2``; dropping
it removes exactly those ``2H + 1`` values and nothing else (``b2``, attention and layer norms are
untouched). Every kept neuron is copied exactly, in its original order.

Neuron scores (higher = keep), all computed per layer:

* ``magnitude``: ``‖W1[j]‖₂ · ‖W2[:, j]‖₂`` (weights only).
* ``activation``: mean ``|gelu(W1[j] · x + b1[j])|`` over the non-pad tokens of the given
  inputs.
* ``combined``: ``activation · ‖W2[:, j]‖₂``, the mean size of neuron ``j``'s contribution to
  the FFN output.

Selection keeps the ``k`` highest scores; ties go to the lower index. HF stores one
``intermediate_size`` per model, so every layer keeps the same ``k``.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence

import torch
from torch import nn

from gidi.modeling.model import GidiMultiTaskModel

CRITERIA = ("magnitude", "activation", "combined")


def _ffn_pairs(model: GidiMultiTaskModel) -> list[tuple[nn.Linear, nn.Linear]]:
    return [(layer.intermediate.dense, layer.output.dense) for layer in model.encoder.encoder.layer]


def params_per_neuron(hidden_size: int) -> int:
    """Values one intermediate neuron owns in one layer: a W1 row, a b1 entry, a W2 column."""
    return 2 * hidden_size + 1


@torch.no_grad()
def neuron_scores(
    model: GidiMultiTaskModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    batch_size: int = 64,
) -> list[dict[str, torch.Tensor]]:
    """Per-layer ``{criterion: scores[I]}`` on the given (training) inputs; eval mode, CPU."""
    model = model.eval()
    pairs = _ffn_pairs(model)
    sums = [torch.zeros(w1.out_features, dtype=torch.float64) for w1, _ in pairs]
    mask_box: list[torch.Tensor] = []
    n_tokens = 0

    def hook(i: int):
        def fn(_module, _inputs, output):
            keep = mask_box[0].reshape(-1)
            sums[i] += output.reshape(-1, output.shape[-1])[keep].abs().sum(0).double()

        return fn

    handles = [
        layer.intermediate.register_forward_hook(hook(i))
        for i, layer in enumerate(model.encoder.encoder.layer)
    ]
    try:
        for start in range(0, len(input_ids), batch_size):
            ids = input_ids[start : start + batch_size]
            mask = attention_mask[start : start + batch_size]
            mask_box[:] = [mask.bool()]
            n_tokens += int(mask.sum())
            model.encoder(input_ids=ids, attention_mask=mask)
    finally:
        for h in handles:
            h.remove()
    out = []
    for (w1, w2), total in zip(pairs, sums, strict=True):
        activation = (total / n_tokens).float()
        out_norm = w2.weight.norm(dim=0)
        out.append(
            {
                "magnitude": w1.weight.norm(dim=1) * out_norm,
                "activation": activation,
                "combined": activation * out_norm,
            }
        )
    return out


def select_neurons(scores: torch.Tensor, k: int) -> list[int]:
    """Indices of the ``k`` highest scores (ties to the lower index), in ascending order."""
    n = int(scores.numel())
    if not 0 < k <= n:
        raise ValueError(f"k must be in 1..{n}, got {k}")
    order = sorted(range(n), key=lambda j: (-float(scores[j]), j))
    return sorted(order[:k])


def prune_ffn(model: GidiMultiTaskModel, keep: Sequence[Sequence[int]]) -> GidiMultiTaskModel:
    """Return a new model whose layer ``l`` keeps intermediate neurons ``keep[l]`` exactly.

    All layers must keep the same number of distinct, in-range, ascending indices. Every other
    tensor (and both heads) is copied unchanged; ``model`` is not mutated.
    """
    pairs = _ffn_pairs(model)
    if len(keep) != len(pairs):
        raise ValueError(f"{len(keep)} neuron lists for {len(pairs)} layers")
    widths = {len(k) for k in keep}
    if len(widths) != 1:
        raise ValueError(f"every layer must keep the same number of neurons, got {sorted(widths)}")
    size = int(model.encoder.config.intermediate_size)
    for layer, idx in enumerate(keep):
        if list(idx) != sorted(set(idx)) or not idx or idx[0] < 0 or idx[-1] >= size:
            raise ValueError(f"layer {layer}: indices must be ascending, unique and in 0..{size}")

    new_config = copy.deepcopy(model.encoder.config)
    new_config.intermediate_size = widths.pop()
    new = GidiMultiTaskModel.from_config(
        new_config,
        num_types=model.num_types,
        num_tags=model.num_tags,
        dropout=model.dropout_p,
        encoder_name=model.encoder_name,
    )
    state = dict(model.state_dict())
    for layer, idx in enumerate(keep):
        rows = torch.tensor(list(idx), dtype=torch.long)
        prefix = f"encoder.encoder.layer.{layer}."
        for key in ("intermediate.dense.weight", "intermediate.dense.bias"):
            state[prefix + key] = state[prefix + key].index_select(0, rows).clone()
        key = prefix + "output.dense.weight"
        state[key] = state[key].index_select(1, rows).clone()
    new.load_state_dict(state, strict=True)
    new.train(model.training)
    return new
