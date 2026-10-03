"""The linear-chain CRF against brute-force enumeration over every tag sequence."""

from __future__ import annotations

import itertools

import numpy as np
import torch

from gidi.modeling.crf import LinearChainCRF, viterbi, viterbi_masked

K = 3


def path_score(crf: LinearChainCRF, emissions: torch.Tensor, path: tuple[int, ...]) -> float:
    s = crf.start_transitions[path[0]] + crf.end_transitions[path[-1]]
    s = s + sum(emissions[t, k] for t, k in enumerate(path))
    s = s + sum(crf.transitions[a, b] for a, b in itertools.pairwise(path))
    return float(s.detach())


def random_crf(seed: int) -> LinearChainCRF:
    crf = LinearChainCRF(K, seed=seed)
    with torch.no_grad():
        crf.transitions.normal_(generator=torch.Generator().manual_seed(seed + 1))
    return crf


def test_nll_equals_brute_force_with_scattered_masks_and_skips_empty_rows():
    crf = random_crf(0)
    emissions = torch.randn(3, 6, K, generator=torch.Generator().manual_seed(1))
    tags = torch.tensor([[-100, 1, 2, 0, -100, -100], [0, 0, 1, 2, 2, 0], [-100] * 6])
    mask = tags != -100  # row 0: positions 1-3 (not starting at 0), row 2: no position
    want = []
    for row in range(2):
        idx = mask[row].nonzero().flatten()
        e, gold = emissions[row, idx], tuple(int(t) for t in tags[row, idx])
        paths = itertools.product(range(K), repeat=len(idx))
        log_z = torch.logsumexp(torch.tensor([path_score(crf, e, p) for p in paths]), dim=0)
        want.append(float(log_z) - path_score(crf, e, gold))
    got = crf.nll(emissions, tags, mask)
    assert abs(float(got.detach()) - sum(want) / 2) < 1e-4


def test_viterbi_finds_the_best_path_and_masked_positions_get_fill():
    crf = random_crf(2)
    emissions = torch.randn(5, K, generator=torch.Generator().manual_seed(3))
    best = max(itertools.product(range(K), repeat=5), key=lambda p: path_score(crf, emissions, p))
    assert tuple(viterbi(emissions.numpy(), crf.numpy_transitions())) == best

    padded = np.concatenate([np.full((1, K), 9.0), emissions.numpy(), np.full((1, K), 9.0)])
    real = [False, True, True, True, True, True, False]
    tags = viterbi_masked(padded, real, crf.numpy_transitions(), fill=0)
    assert tags[0] == tags[-1] == 0 and tuple(tags[1:-1]) == best


def test_transitions_override_the_token_argmax():
    crf = LinearChainCRF(K, seed=0)
    with torch.no_grad():
        crf.transitions.zero_()
        crf.transitions[0, 2] = -100.0  # O -> I forbidden in practice
    # argmax alone would give O I (a span starting at I); the CRF must pick O O or B I
    emissions = np.array([[2.0, 0.0, 0.0], [0.0, 1.5, 2.0]])
    assert viterbi(emissions, crf.numpy_transitions()) == [0, 1]
