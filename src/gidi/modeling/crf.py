"""Linear-chain CRF over BIO tags: negative log-likelihood (torch).

Viterbi decoding is pure numpy and lives in ``gidi.inference.crf`` (re-exported here).

Parameters follow the usual ``pytorch-crf`` layout: ``start_transitions [K]``,
``end_transitions [K]`` and ``transitions [K, K]`` (``transitions[i, j]`` scores tag ``i``
followed by tag ``j``). Transitions are learned only; there are no hard BIO constraints.

A sequence is the positions where ``mask`` is true, in order; they need not be contiguous or
start at 0 (the value labels skip ``<s>``/``</s>``). Sequences without any such position are
dropped from the loss.
"""

from __future__ import annotations

import torch
from torch import nn

from gidi.inference.crf import CRFTransitions, viterbi, viterbi_masked

# ``pytorch-crf`` initialises every transition score uniformly in (-0.1, 0.1).
CRF_INIT_BOUND = 0.1


class LinearChainCRF(nn.Module):
    def __init__(self, num_tags: int, seed: int | None = None) -> None:
        super().__init__()
        self.num_tags = num_tags
        generator = None if seed is None else torch.Generator().manual_seed(seed)

        def init(*shape: int) -> nn.Parameter:
            values = torch.empty(*shape).uniform_(
                -CRF_INIT_BOUND, CRF_INIT_BOUND, generator=generator
            )
            return nn.Parameter(values)

        self.start_transitions = init(num_tags)
        self.end_transitions = init(num_tags)
        self.transitions = init(num_tags, num_tags)

    def nll(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean negative log-likelihood over the sequences with at least one masked-in position.

        ``emissions [B, T, K]``, ``tags [B, T]`` (any value where ``mask`` is false),
        ``mask [B, T]`` bool. Returns a scalar; 0 (with a graph) if no sequence is supervised.
        """
        lengths = mask.sum(dim=1)
        keep = lengths > 0
        if not bool(keep.any()):
            return emissions.sum() * 0.0
        emissions, tags, mask, lengths = emissions[keep], tags[keep], mask[keep], lengths[keep]
        # left-align the masked-in positions of every row (stable: order is preserved)
        order = torch.argsort((~mask).to(torch.int8), dim=1, stable=True)
        emissions = emissions.gather(1, order.unsqueeze(-1).expand_as(emissions))
        tags = tags.gather(1, order).clamp(min=0)
        width = int(lengths.max())
        emissions, tags = emissions[:, :width], tags[:, :width]
        steps = torch.arange(width, device=emissions.device)
        valid = steps.unsqueeze(0) < lengths.unsqueeze(1)  # [B, W]
        return (self._log_partition(emissions, valid) - self._score(emissions, tags, valid)).mean()

    def _score(self, emissions: torch.Tensor, tags: torch.Tensor, valid: torch.Tensor):
        batch = torch.arange(emissions.shape[0], device=emissions.device)
        emit = emissions.gather(2, tags.unsqueeze(-1)).squeeze(-1)  # [B, W]
        # ``where`` rather than multiplying by the mask: padded scores may be non-finite
        zero = emit.new_zeros(())
        score = self.start_transitions[tags[:, 0]] + torch.where(valid, emit, zero).sum(dim=1)
        trans = self.transitions[tags[:, :-1], tags[:, 1:]]  # [B, W-1]
        score = score + torch.where(valid[:, 1:], trans, zero).sum(dim=1)
        last = tags[batch, valid.sum(dim=1) - 1]
        return score + self.end_transitions[last]

    def _log_partition(self, emissions: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        alpha = self.start_transitions + emissions[:, 0]  # [B, K]
        for t in range(1, emissions.shape[1]):
            nxt = torch.logsumexp(
                alpha.unsqueeze(2) + self.transitions + emissions[:, t].unsqueeze(1), dim=1
            )
            alpha = torch.where(valid[:, t].unsqueeze(1), nxt, alpha)
        return torch.logsumexp(alpha + self.end_transitions, dim=1)

    def numpy_transitions(self) -> CRFTransitions:
        return CRFTransitions(
            self.start_transitions.detach().float().cpu().numpy(),
            self.end_transitions.detach().float().cpu().numpy(),
            self.transitions.detach().float().cpu().numpy(),
        )


__all__ = ["CRFTransitions", "LinearChainCRF", "viterbi", "viterbi_masked"]
