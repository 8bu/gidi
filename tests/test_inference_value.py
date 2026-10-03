"""Runtime with a value head, on a throwaway bundle built from the v1 one (see ``conftest.py``).

The fixture's ``value_logits`` are ``tag_logits`` shifted one token left, so the expected value
span is computed here independently from the v1 model's raw tag logits.
"""

from __future__ import annotations

import itertools
import json
import math
import shutil
import unicodedata
from pathlib import Path

import numpy as np
import pytest

from gidi.inference import BundleError, GidiPredictor, verify_bundle
from gidi.inference.bundle import CONFIG_FILE, MANIFEST_FILE, load_config
from gidi.inference.crf import CRFTransitions, viterbi, viterbi_masked
from gidi.inference.decode import decode_all_spans, decode_first_span, decode_value_crf
from gidi.inference.text import normalize_nfc
from gidi.modeling.value_decode import decode_value

ROOT = Path(__file__).resolve().parents[1]
V1_BUNDLE = ROOT / "models" / "gidi-finance-v1"

ZERO_CRF = CRFTransitions(np.zeros(3), np.zeros(3), np.zeros((3, 3)))  # Viterbi == argmax

V2_KEYS = [
    "type",
    "type_confidence",
    "target",
    "target_span",
    "target_confidence",
    "value_text",
    "value_span",
    "value_confidence",
    "truncated",
    "model_version",
]
V1_KEYS = [k for k in V2_KEYS if not k.startswith("value_")]

NOTES = [
    "trả nợ chị Mai 500k",
    "😀 trả nợ chị Mai 500k",
    "🎉😀 cho Tuấn vay 5 triệu",
    unicodedata.normalize("NFD", "😀 Tuấn cho vay 5 triệu"),
    "ăn phở 45k",
    "Tuấn \ud800 cho vay 5 triệu",
]

pytestmark = pytest.mark.skipif(not V1_BUNDLE.is_dir(), reason="deployment bundle not built")


@pytest.fixture(scope="module")
def v1() -> GidiPredictor:
    return GidiPredictor.from_bundle(V1_BUNDLE)


@pytest.fixture(scope="module")
def v2(v2_bundle) -> GidiPredictor:
    return GidiPredictor.from_bundle(v2_bundle)


def expected_value_span(v1: GidiPredictor, text: str) -> tuple[int, int] | None:
    """The value span the fixture graph encodes, decoded by the *training-side* value decoder."""
    raw = v1.run(text)
    shifted = np.vstack([raw.tag_logits[1:], np.zeros((1, raw.tag_logits.shape[1]))])
    real = [not special for special in raw.special_tokens_mask]
    span, _ = decode_value(shifted, raw.normalized_offsets, raw.normalized_text, real)
    if span is None:
        return None
    return normalize_nfc(text).span_to_original(span.start, span.end)


def test_v2_bundle_is_detected_and_verifies(v2_bundle, v2):
    verify_bundle(v2_bundle)
    config = load_config(v2_bundle)
    assert config.value_labels == ("O", "B-VALUE", "I-VALUE")
    assert config.output_names == ("type_logits", "tag_logits", "value_logits")
    assert v2.has_value_head and not GidiPredictor.from_bundle(V1_BUNDLE).has_value_head


def test_v2_output_keys_and_value_slicing_from_the_callers_string(v1, v2):
    found_value = False
    for text in NOTES:
        out = v2.predict(text).to_dict()
        assert list(out) == V2_KEYS
        assert out["model_version"] == "gidi-finance-v2-test"
        span = out["value_span"]
        assert (span is None) == (out["value_text"] is None)
        assert 0.0 < out["value_confidence"] <= 1.0
        if span is not None:
            found_value = True
            s, e = span
            assert text[s:e] == out["value_text"] == out["value_text"].strip()
        assert (None if span is None else tuple(span)) == expected_value_span(v1, text)
    assert found_value


def test_value_offsets_survive_astral_prefix_and_decomposed_input(v2):
    out = v2.predict("😀 trả nợ chị Mai 500k").to_dict()
    assert (out["value_text"], out["value_span"]) == ("chị", [9, 12])  # code points, not UTF-16
    assert out["target_span"] == [0, 1]  # v1 target, untouched by the value head

    nfd = unicodedata.normalize("NFD", "trả nợ chị Mai 500k")
    out = v2.predict(nfd).to_dict()
    s, e = out["value_span"]
    assert nfd[s:e] == out["value_text"]
    assert unicodedata.normalize("NFC", out["value_text"]) == "chị"


def test_v2_adds_value_without_changing_the_v1_fields(v1, v2):
    for text in NOTES:
        old, new = v1.predict(text).to_dict(), v2.predict(text).to_dict()
        assert list(old) == V1_KEYS
        assert {k: new[k] for k in V1_KEYS if k != "model_version"} == {
            k: old[k] for k in V1_KEYS if k != "model_version"
        }


def test_no_value_span_reports_o_confidence(v2):
    out = v2.predict("ăn phở 45k").to_dict()
    assert out["value_span"] is None and out["value_text"] is None
    assert 0.0 < out["value_confidence"] <= 1.0


def test_raw_output_exposes_value_logits_only_with_a_value_head(v1, v2):
    raw = v2.run("trả nợ chị Mai 500k")
    assert raw.value_logits.shape == (len(raw.input_ids), 3)
    assert v1.run("trả nợ chị Mai 500k").value_logits is None


def copy_bundle(src: Path, dest: Path) -> Path:
    shutil.copytree(src, dest)
    return dest


def test_config_value_labels_must_match_the_onnx_outputs(v2_bundle, tmp_path):
    bad = copy_bundle(v2_bundle, tmp_path / "no-head-in-onnx")
    shutil.copyfile(V1_BUNDLE / "model.int8.onnx", bad / "model.int8.onnx")
    with pytest.raises(ValueError, match="value_logits"):
        GidiPredictor.from_bundle(bad)

    hidden = copy_bundle(v2_bundle, tmp_path / "onnx-has-head-config-does-not")
    config = json.loads((hidden / CONFIG_FILE).read_text(encoding="utf-8"))
    del config["value_labels"]
    config["onnx"]["outputs"] = config["onnx"]["outputs"][:2]
    (hidden / CONFIG_FILE).write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="value_logits"):
        GidiPredictor.from_bundle(hidden)


SWAPPED_OUTPUTS = ["type_logits", "value_logits", "tag_logits"]


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda c: c.update(value_labels=["O", "B-VALUE"]), "value label set"),
        (lambda c: c.update(value_labels=["O", "B-AMOUNT", "I-AMOUNT"]), "value label set"),
        (lambda c: c["onnx"].update(outputs=SWAPPED_OUTPUTS), "third"),
    ],
)
def test_malformed_value_config_is_rejected(v2_bundle, tmp_path, edit, message):
    bad = copy_bundle(v2_bundle, tmp_path / "bad")
    config = json.loads((bad / CONFIG_FILE).read_text(encoding="utf-8"))
    edit(config)
    (bad / CONFIG_FILE).write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(BundleError, match=message):
        load_config(bad)


def test_manifest_and_config_must_agree_on_value_labels(v2_bundle, tmp_path):
    bad = copy_bundle(v2_bundle, tmp_path / "manifest-without-labels")
    manifest = json.loads((bad / MANIFEST_FILE).read_text(encoding="utf-8"))
    del manifest["value_labels"]
    (bad / MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BundleError, match="value_labels differ"):
        verify_bundle(bad)


def _logits(tags: list[int], p: list[float]) -> np.ndarray:
    """Log-probabilities putting probability ``p[i]`` on ``tags[i]`` (rest split evenly)."""
    out = np.zeros((len(tags), 3))
    for i, (tag, prob) in enumerate(zip(tags, p, strict=True)):
        out[i] = np.log((1 - prob) / 2)
        out[i, tag] = np.log(prob)
    return out


# "ăn 2 tô phở 70": the quantity "2" and the amount "70" are both predicted as value spans.
OFFSETS = [(0, 0), (0, 2), (3, 4), (5, 7), (8, 11), (12, 14), (0, 0)]
TEXT = "ăn 2 tô phở 70"
REAL = [False, True, True, True, True, True, False]


def test_value_head_keeps_the_most_confident_span_not_the_first():
    logits = _logits([0, 0, 1, 0, 0, 1, 0], [0.99, 0.99, 0.6, 0.99, 0.99, 0.95, 0.99])
    span, confidence = decode_value_crf(logits, OFFSETS, TEXT, REAL, ZERO_CRF)
    assert (TEXT[span.start : span.end], span.members) == ("70", (5,))
    assert confidence == pytest.approx(0.95)
    first = decode_first_span(OFFSETS, np.argmax(logits, axis=-1), TEXT)
    assert TEXT[first.start : first.end] == "2"  # the target-style rule would pick the quantity


def test_equal_confidence_spans_resolve_to_the_earlier_one():
    logits = _logits([0, 0, 1, 0, 0, 1, 0], [0.99, 0.99, 0.9, 0.99, 0.99, 0.9, 0.99])
    span, _ = decode_value_crf(logits, OFFSETS, TEXT, REAL, ZERO_CRF)
    assert TEXT[span.start : span.end] == "2"


def test_no_value_span_confidence_is_the_weakest_o_probability():
    logits = _logits([0, 0, 0, 0, 0, 0, 0], [0.99, 0.99, 0.9, 0.99, 0.99, 0.8, 0.99])
    span, confidence = decode_value_crf(logits, OFFSETS, TEXT, REAL, ZERO_CRF)
    assert span is None and confidence == pytest.approx(0.8)


def random_crf(rng: np.random.Generator, scale: float = 2.0) -> CRFTransitions:
    return CRFTransitions(
        rng.normal(size=3) * scale, rng.normal(size=3) * scale, rng.normal(size=(3, 3)) * scale
    )


def test_viterbi_equals_brute_force_over_every_path():
    rng = np.random.default_rng(0)
    for _ in range(200):
        length = int(rng.integers(1, 6))
        emissions, crf = rng.normal(size=(length, 3)) * 3, random_crf(rng)

        def score(path, emissions=emissions, crf=crf):
            total = crf.start[path[0]] + crf.end[path[-1]]
            total += sum(emissions[t, k] for t, k in enumerate(path))
            return total + sum(crf.transitions[a, b] for a, b in itertools.pairwise(path))

        best = max(itertools.product(range(3), repeat=length), key=score)
        assert tuple(viterbi(emissions, crf)) == best


def test_viterbi_handles_no_tokens_huge_emissions_and_any_bio_sequence():
    rng = np.random.default_rng(1)
    assert viterbi(np.zeros((0, 3)), ZERO_CRF) == []
    assert viterbi_masked(np.zeros((4, 3)), [False] * 4, ZERO_CRF) == [0, 0, 0, 0]
    for _ in range(100):
        length = int(rng.integers(1, 9))
        emissions = rng.normal(size=(length, 3)) * 1e30
        crf = random_crf(rng, scale=1e30)
        tags = viterbi(emissions, crf)
        assert len(tags) == length and set(tags) <= {0, 1, 2}

    # an I right after O (and a leading I) is a legal span start; spans are never empty
    offsets = [(0, 3), (4, 7), (8, 11)]
    for tags in itertools.product(range(3), repeat=3):
        for span in decode_all_spans(offsets, tags, "abc def ghi"):
            assert span.start < span.end
    (only,) = decode_all_spans(offsets, [0, 2, 2], "abc def ghi")
    assert (only.start, only.end) == (4, 11)


def test_forbidding_o_to_i_changes_the_decode_versus_argmax():
    # argmax alone reads "O I" (a span); with O->I forbidden the best path is "O O" (no span)
    logits = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.5, 0.0, 2.0], [3.0, 0.0, 0.0]])
    offsets, text, real = [(0, 0), (0, 3), (4, 7), (0, 0)], "abc def", [False, True, True, False]
    argmax_span, _ = decode_value_crf(logits, offsets, text, real, ZERO_CRF)
    forbid = CRFTransitions(np.zeros(3), np.zeros(3), np.array([[0, 0, -100.0], [0] * 3, [0] * 3]))
    forbid_span, _ = decode_value_crf(logits, offsets, text, real, forbid)
    assert argmax_span is not None and argmax_span.members == (2,)
    assert forbid_span is None


def test_empty_real_tokens_give_no_span_and_unit_confidence():
    logits = np.random.default_rng(2).normal(size=(2, 3))
    span, confidence = decode_value_crf(
        logits, [(0, 0), (0, 0)], "", [False, False], random_crf(np.random.default_rng(3))
    )
    assert span is None and confidence == 1.0


def test_runtime_crf_decoder_equals_the_training_side_one_on_random_inputs():
    rng = np.random.default_rng(7)
    for _ in range(300):
        logits = rng.normal(size=(len(OFFSETS), 3)) * 2
        real = [bool(x) for x in rng.random(len(OFFSETS)) < 0.8]
        crf = random_crf(rng)
        got = decode_value_crf(logits, OFFSETS, TEXT, real, crf)
        want = decode_value(logits, OFFSETS, TEXT, real, crf=crf)
        assert got == want
        assert math.isfinite(got[1]) and 0.0 <= got[1] <= 1.0


def with_value_decoding(v2_bundle: Path, dest: Path, edit) -> Path:
    bad = copy_bundle(v2_bundle, dest)
    config = json.loads((bad / CONFIG_FILE).read_text(encoding="utf-8"))
    edit(config["value_decoding"])
    (bad / CONFIG_FILE).write_text(json.dumps(config), encoding="utf-8")
    return bad


@pytest.mark.parametrize(
    "edit",
    [
        lambda d: d.update(method="argmax"),
        lambda d: d.pop("method"),
        lambda d: d.update(span_selection="first"),
        lambda d: d.pop("start_transitions"),
        lambda d: d.update(start_transitions=[0.0, 0.0]),
        lambda d: d.update(end_transitions=[0.0, 0.0, 0.0, 0.0]),
        lambda d: d.update(end_transitions=[0.0, "x", 0.0]),
        lambda d: d.update(end_transitions=[0.0, True, 0.0]),
        lambda d: d.update(start_transitions=[0.0, float("nan"), 0.0]),
        lambda d: d.update(start_transitions=[0.0, float("inf"), 0.0]),
        lambda d: d.update(transitions=[[0.0] * 3] * 2),
        lambda d: d.update(transitions=[[0.0] * 3, [0.0] * 3, [0.0] * 2]),
        lambda d: d.update(transitions=[0.0] * 9),
    ],
)
def test_malformed_value_decoding_is_rejected(v2_bundle, tmp_path, edit):
    with pytest.raises(BundleError, match="value_decoding"):
        load_config(with_value_decoding(v2_bundle, tmp_path / "bad", edit))


def test_value_labels_without_value_decoding_are_rejected(v2_bundle, tmp_path):
    bad = copy_bundle(v2_bundle, tmp_path / "bad")
    config = json.loads((bad / CONFIG_FILE).read_text(encoding="utf-8"))
    del config["value_decoding"]
    (bad / CONFIG_FILE).write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(BundleError, match="value_decoding"):
        load_config(bad)


def test_predictor_decodes_value_with_the_bundles_crf(v1, v2_bundle, tmp_path):
    # O -> I forbidden, I -> I free: the fixture's logits put "I" on a token after "O" often
    def forbid(d):
        d["transitions"] = [[0.0, 0.0, -1e4], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    crf_bundle = with_value_decoding(v2_bundle, tmp_path / "crf", forbid)
    shutil.copyfile(v2_bundle / MANIFEST_FILE, crf_bundle / MANIFEST_FILE)
    crf = GidiPredictor.from_bundle(crf_bundle)
    plain = GidiPredictor.from_bundle(v2_bundle)
    for text in NOTES:
        raw = crf.run(text)
        assert len(raw.value_tags) == len(raw.input_ids)
        assert all(
            t == 0 for t, s in zip(raw.value_tags, raw.special_tokens_mask, strict=True) if s
        )
        # no span ever starts with I after O under the forbidding CRF
        tags = raw.value_tags
        assert all(not (a == 0 and b == 2) for a, b in itertools.pairwise(tags))
        out = crf.predict(text).to_dict()
        assert list(out) == V2_KEYS
        assert math.isfinite(out["value_confidence"])
        if out["value_span"] is not None:
            start, end = out["value_span"]
            assert start < end and out["value_text"] == text[start:end]
        assert plain.run(text).value_tags == tuple(
            int(t) if not s else 0
            for t, s in zip(
                np.argmax(raw.value_logits, axis=-1), raw.special_tokens_mask, strict=True
            )
        )
