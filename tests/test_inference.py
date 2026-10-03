"""Deployment runtime: parity with the training-side tokenizer/decoder, offsets, edge inputs."""

from __future__ import annotations

import json
import math
import random
import subprocess
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pytest

from gidi.inference import EmptyInputError, GidiPredictor, verify_bundle
from gidi.inference.decode import decode_first_span, target_confidence
from gidi.inference.text import normalize_nfc
from gidi.inference.tokenizer import BundleTokenizer
from gidi.modeling.preprocessing import TYPES, encode, spans_from_tags
from gidi.modeling.tokenization import load_tokenizer

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "models" / "gidi-finance-v1"
PROTOCOL = ROOT / "experiments" / "deployment-v1" / "protocol.json"
CHECKPOINT = ROOT / json.loads(PROTOCOL.read_text(encoding="utf-8"))["source_checkpoint"]

needs_bundle = pytest.mark.skipif(not BUNDLE.is_dir(), reason="deployment bundle not built")
needs_checkpoint = pytest.mark.skipif(
    not (BUNDLE.is_dir() and CHECKPOINT.is_dir()), reason="bundle or source checkpoint missing"
)

NOTES = [
    "Tuấn cho vay 5 triệu",
    "ăn phở 45k",
    "  lương tháng 10 về 15tr  ",
    "Trả nợ cho Thảo 2tr5",
    "chuyển khoản cho mẹ 3 triệu",
    "hoàn tiền shopee 120.000đ",
    "mượn anh Hùng 500k, hẹn cuối tháng",
    "😀 cà phê với Linh 😀 35k",
    "line one\nline two\ttabbed 10k",
    "x",
    "5",
    "...!!!",
    "ＡＢＣ fullwidth １２３",
    "\u200bzero\u200bwidth\u200b",
    "Nguyễn Văn A trả tôi 1.500.000 đồng tiền nợ tháng trước, cảm ơn nhiều lắm nhé",
    "word " * 40,
    "Tuấn " * 30 + "cho vay 5 triệu",
]

_ALPHABET = list("abcdeăâêôơưđ ĐÁẤỆỢ0123456789.,!?…-_/\t\n\u00a0\u200b😀🎉") + [
    "\u1ee3",
    "o\u031b\u0323",
    "e\u0302\u0301",
    "\u1100\u1161\u11a8",
]


def _random_texts(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    texts = []
    for _ in range(n):
        text = "".join(rng.choice(_ALPHABET) for _ in range(rng.randint(1, 120)))
        texts.append(unicodedata.normalize("NFC", text))
    return texts


@pytest.fixture(scope="module")
def predictor() -> GidiPredictor:
    return GidiPredictor.from_bundle(BUNDLE)


@needs_checkpoint
def test_tokenizer_matches_training_tokenizer_for_any_input():
    reference = load_tokenizer(str(CHECKPOINT))
    runtime = BundleTokenizer(BUNDLE / "tokenizer.json", max_length=32)
    for text in NOTES + _random_texts(300, seed=7):
        text = unicodedata.normalize("NFC", text)
        if not text.strip():
            continue
        expected = encode(reference, [text], [None])
        got = runtime.encode(text)
        assert list(got.ids) == expected.input_ids[0].tolist(), text
        assert list(got.offsets) == [tuple(o) for o in expected.offsets[0]], text
        assert got.truncated == expected.truncated[0], text
        assert len(got.ids) <= 32


@needs_checkpoint
def test_tokenizer_truncation_drops_tail_and_keeps_specials():
    runtime = BundleTokenizer(BUNDLE / "tokenizer.json", max_length=32)
    long = runtime.encode("word " * 100)
    assert len(long.ids) == 32 and long.truncated
    assert long.special_tokens_mask[0] == 1 and long.special_tokens_mask[-1] == 1
    assert long.ids[0] == 0 and long.ids[-1] == 2
    short = runtime.encode("word " * 3)
    assert not short.truncated and len(short.ids) < 32


def test_decoder_equals_training_decoder_on_random_tag_sequences():
    rng = random.Random(11)
    for _ in range(2000):
        text = "".join(rng.choice("ab cd\t,é") for _ in range(rng.randint(1, 24)))
        cuts = sorted(
            {0, len(text), *(rng.randint(0, len(text)) for _ in range(rng.randint(0, 8)))}
        )
        offsets = [(0, 0)]  # <s>
        offsets += list(zip(cuts, cuts[1:], strict=False))  # byte-level style: spaces inside
        offsets.append((0, 0))  # </s>
        tags = [rng.choice((0, 0, 1, 2, 2)) for _ in offsets]
        if rng.random() < 0.2:
            tags = tags[: rng.randint(0, len(tags))]
        expected = spans_from_tags(offsets, tags, text)
        got = decode_first_span(offsets, tags, text)
        if expected is None:
            assert got is None, (text, offsets, tags)
        else:
            assert got is not None, (text, offsets, tags)
            assert (got.start, got.end) == (expected["start"], expected["end"])
            assert text[got.start : got.end] == expected["text"]


def test_confidences():
    logits = np.log(
        np.array(
            [
                [0.2, 0.7, 0.1],  # <s>
                [0.1, 0.8, 0.1],  # B
                [0.05, 0.05, 0.9],  # I
                [0.6, 0.3, 0.1],  # O
                [0.9, 0.05, 0.05],  # </s>
            ]
        )
    )
    tags = np.argmax(logits, axis=-1)
    offsets = [(0, 0), (0, 3), (3, 6), (6, 9), (0, 0)]
    span = decode_first_span(offsets, tags, "abcdefghi")
    assert span is not None and span.members == (1, 2)
    assert target_confidence(logits, tags, span, [False, True, True, True, False]) == pytest.approx(
        math.sqrt(0.8 * 0.9)
    )
    # no span: weakest P(O) over real tokens only (the <s>/</s> rows are ignored)
    quiet = np.log(np.array([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1], [0.5, 0.3, 0.2], [0.1, 0.8, 0.1]]))
    assert target_confidence(quiet, np.zeros(4, int), None, [False, True, True, False]) == (
        pytest.approx(0.5)
    )


def test_nfc_offset_map_round_trips_word_boundaries():
    rng = random.Random(3)
    words = ["nợ", "Tuấn", "triệu", "Thảo", "Việt", "5tr", "😀", "한글", "ế"]
    for _ in range(300):
        chosen = [rng.choice(words) for _ in range(rng.randint(1, 6))]
        parts = [
            unicodedata.normalize(rng.choice(("NFC", "NFD")), w) for w in chosen
        ]  # mixed composed/decomposed
        original = " ".join(parts)
        normalized = normalize_nfc(original)
        assert normalized.text == unicodedata.normalize("NFC", original)
        nfc_cursor = 0
        orig_cursor = 0
        for word, part in zip(chosen, parts, strict=True):
            nfc_word = unicodedata.normalize("NFC", word)
            start, end = normalized.span_to_original(nfc_cursor, nfc_cursor + len(nfc_word))
            assert (start, end) == (orig_cursor, orig_cursor + len(part))
            assert original[start:end] == part
            nfc_cursor += len(nfc_word) + 1
            orig_cursor += len(part) + 1


def test_nfc_offset_map_is_monotonic_and_exact_after_arbitrary_marks():
    rng = random.Random(9)
    pool = [chr(c) for c in (*range(0x61, 0x6B), *range(0x300, 0x330), *range(0x1100, 0x1113))]
    pool += [chr(c) for c in (*range(0x1161, 0x1176), *range(0x11A8, 0x11C3), 0xAC00, 0x1EE3)]
    for _ in range(3000):
        head = "".join(rng.choice(pool) for _ in range(rng.randint(1, 10)))
        original = head + " tail 😀"
        n = normalize_nfc(original)
        assert n.text == unicodedata.normalize("NFC", original)
        if n.start_map is None:
            continue
        assert len(n.start_map) == len(n.end_map) == len(n.text) + 1
        assert list(n.start_map) == sorted(n.start_map) and list(n.end_map) == sorted(n.end_map)
        # text after a space is outside any composition: its span maps exactly
        tail_start = n.text.index(" tail")
        start, end = n.span_to_original(tail_start + 1, len(n.text))
        assert original[start:end] == "tail 😀"
        assert n.span_to_original(0, len(n.text)) == (0, len(original))


def test_nfc_offset_map_is_identity_for_nfc_text_and_whole_string_covers_everything():
    text = "Tuấn nợ 😀"
    n = normalize_nfc(text)
    assert n.text == text and n.start_map is None and n.span_to_original(2, 5) == (2, 5)
    decomposed = unicodedata.normalize("NFD", text)
    d = normalize_nfc(decomposed)
    assert d.span_to_original(0, len(d.text)) == (0, len(decomposed))


@needs_bundle
def test_decomposed_input_returns_offsets_into_callers_string(predictor):
    composed = "Tuấn nợ 2 triệu"
    decomposed = unicodedata.normalize("NFD", composed)
    assert decomposed != composed and len(decomposed) > len(composed)
    a, b = predictor.predict(composed), predictor.predict(decomposed)
    assert (a.type, a.type_confidence, a.target_confidence) == (
        b.type,
        b.type_confidence,
        b.target_confidence,
    )
    assert a.target is not None and b.target is not None
    assert unicodedata.normalize("NFC", b.target) == a.target
    start, end = b.target_span
    assert decomposed[start:end] == b.target

    # the exact example from the spec: 'nợ' typed as n + o + U+031B + U+0323
    raw = predictor.run("n\u006f\u031b\u0323 Tuấn")
    assert raw.normalized_text == "n\u1ee3 Tuấn"
    assert raw.offsets[1][0] == 0 and max(e for _, e in raw.offsets) == len(
        "n\u006f\u031b\u0323 Tuấn"
    )


@needs_bundle
def test_astral_characters_keep_code_point_offsets(predictor):
    text = "😀😀 Tuấn cho vay 5 triệu 🎉"
    pred = predictor.predict(text)
    if pred.target is not None:
        start, end = pred.target_span
        assert text[start:end] == pred.target
    raw = predictor.run(text)
    assert all(0 <= s <= e <= len(text) for s, e in raw.offsets)


@needs_bundle
def test_spans_always_slice_back_to_target(predictor):
    for text in NOTES + _random_texts(60, seed=5):
        if not text.strip():
            continue
        pred = predictor.predict(text)
        assert pred.type in TYPES
        if pred.target is None:
            assert pred.target_span is None
        else:
            start, end = pred.target_span
            assert 0 <= start < end <= len(text)
            assert text[start:end] == pred.target == pred.target.strip()


@needs_bundle
@pytest.mark.parametrize("text", ["", "  \n", "\t", "\u00a0\u2003"])
def test_empty_input_raises(predictor, text):
    with pytest.raises(EmptyInputError):
        predictor.predict(text)
    with pytest.raises(EmptyInputError):
        predictor.run(text)
    assert issubclass(EmptyInputError, ValueError)


@needs_bundle
@pytest.mark.parametrize("value", [None, b"abc", 5, ["a"]])
def test_non_string_raises_type_error(predictor, value):
    with pytest.raises(TypeError):
        predictor.predict(value)


@needs_bundle
def test_overlength_text_is_flagged_not_fatal(predictor):
    text = "Tuấn cho vay 5 triệu. " * 300
    pred = predictor.predict(text)
    assert pred.truncated
    raw = predictor.run(text)
    assert len(raw.input_ids) == 32 and raw.tag_logits.shape == (32, 3) and raw.truncated
    assert not predictor.predict("Tuấn cho vay 5 triệu").truncated


@needs_bundle
def test_lone_surrogate_does_not_crash_and_keeps_alignment(predictor):
    text = "Tuấn \ud800 cho vay 5 triệu"
    pred = predictor.predict(text)
    if pred.target is not None:
        start, end = pred.target_span
        assert text[start:end] == pred.target


@needs_bundle
def test_prediction_dict_shape_and_determinism(predictor):
    text = "Tuấn cho vay 5 triệu"
    first = predictor.predict(text)
    assert list(first.to_dict()) == [
        "type",
        "type_confidence",
        "target",
        "target_span",
        "target_confidence",
        "truncated",
        "model_version",
    ]
    payload = first.to_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert isinstance(payload["target_span"], list)
    assert type(payload["type_confidence"]) is float
    assert payload["model_version"] == predictor.model_version == "gidi-finance-v1"
    assert predictor.predict(text) == first
    fresh = GidiPredictor.from_bundle(BUNDLE)
    assert fresh.predict(text) == first


@needs_bundle
def test_confidences_follow_the_documented_formulas(predictor):
    raw = predictor.run("Tuấn cho vay 5 triệu")
    pred = predictor.predict("Tuấn cho vay 5 triệu")
    x = raw.type_logits.astype(np.float64)
    p = np.exp(x - x.max())
    p /= p.sum()
    assert pred.type == TYPES[int(np.argmax(x))]
    assert pred.type_confidence == pytest.approx(p.max())
    assert 0.0 < pred.target_confidence <= 1.0


@needs_bundle
def test_bundle_files_match_manifest():
    verify_bundle(BUNDLE)


@needs_bundle
def test_runtime_never_imports_training_stack():
    code = (
        "import sys\n"
        "from gidi.inference import GidiPredictor\n"
        f"p = GidiPredictor.from_bundle({str(BUNDLE)!r})\n"
        "p.predict('Tuấn cho vay 5 triệu')\n"
        "bad = [m for m in sys.modules if m.split('.')[0] in "
        "{'torch', 'transformers', 'onnx', 'safetensors'} "
        "or m in {'gidi.modeling', 'gidi.export'}]\n"
        "assert not bad, bad\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, check=False
    )
    assert result.returncode == 0, result.stderr


def test_runtime_package_imports_without_bundle():
    code = (
        "import sys\n"
        "import gidi.inference\n"
        "assert 'torch' not in sys.modules and 'transformers' not in sys.modules\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, check=False
    )
    assert result.returncode == 0, result.stderr
