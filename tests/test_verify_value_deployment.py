"""``verify_value_deployment.py``: the independent reference implementations the value-head
hardening checks compare the runtime against."""

from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

from gidi.inference.crf import CRFTransitions, viterbi

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def vv():
    spec = importlib.util.spec_from_file_location(
        "verify_value_deployment_under_test", ROOT / "scripts" / "verify_value_deployment.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # @dataclass looks the module up while it executes
    spec.loader.exec_module(module)
    return module


def test_reference_viterbi_is_optimal_and_matches_the_runtime(vv):
    rng = np.random.default_rng(7)
    for _ in range(200):
        t = int(rng.integers(1, 6))
        em = rng.normal(size=(t, 3)) * rng.choice([0.0, 1.0, 10.0])  # 0.0: all tied
        crf = CRFTransitions(*(rng.normal(size=s) for s in ((3,), (3,), (3, 3))))
        path = vv.reference_viterbi(em, crf.start, crf.end, crf.transitions)
        best = max(
            vv.path_score(em, list(p), crf.start, crf.end, crf.transitions)
            for p in itertools.product(range(3), repeat=t)
        )
        assert vv.path_score(em, path, crf.start, crf.end, crf.transitions) == pytest.approx(best)
        assert path == viterbi(em, crf)  # same tie-breaking as the deployed implementation


def test_bio_spans_follow_the_target_head_rule(vv):
    text = "ab 12 cd 34"
    offsets = [(0, 0), (0, 2), (3, 5), (6, 8), (9, 11), (0, 0)]
    special = [1, 0, 0, 0, 0, 1]
    # I after O starts a span; a B ends the previous span and starts the next
    spans = vv.bio_spans([0, 0, 2, 0, 1, 0], offsets, special, text)
    assert [(s["start"], s["end"]) for s in spans] == [(3, 5), (9, 11)]
    spans = vv.bio_spans([0, 1, 2, 1, 2, 0], offsets, special, text)
    assert [(s["start"], s["end"]) for s in spans] == [(0, 5), (6, 11)]


def test_spans_map_back_into_the_callers_nfd_string(vv):
    import unicodedata

    nfd = unicodedata.normalize("NFD", "nợ 2tr")
    pred = {"target": None, "target_span": None, "value_text": "2tr", "value_span": [3, 6]}
    mapped = vv.to_original(nfd, pred)
    start, end = mapped["value_span"]
    assert nfd[start:end] == mapped["value_text"] == "2tr"
    assert mapped["value_span"] != pred["value_span"]  # NFD is longer than NFC
