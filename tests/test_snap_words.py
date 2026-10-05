"""Word snap of the decoded target span: whole-word extension, off by default in the predictor."""

from __future__ import annotations

from pathlib import Path

import pytest

from gidi.inference import GidiPredictor
from gidi.inference.decode import DecodedSpan, snap_span_to_words, snap_to_words

BUNDLE = Path(__file__).resolve().parents[1] / "models" / "gidi-finance-v1"
needs_bundle = pytest.mark.skipif(not BUNDLE.is_dir(), reason="deployment bundle not built")


def snap(text: str, piece: str) -> str:
    start = text.index(piece)
    lo, hi = snap_to_words(text, start, start + len(piece))
    return text[lo:hi]


@pytest.mark.parametrize(
    ("text", "piece", "expected"),
    [
        # the fragments the retrain-v3 encoder produced on human-value-01
        ("rut tien vpbank 1tr", "pbank", "vpbank"),
        ("thanh toán thẻ hsbc 7,2tr", "sbc", "hsbc"),
        ("pizza 4p tối t7 389.000đ", "piz", "pizza"),
        ("sua tuoi banh keo bach hoa xanh 96k", "ch hoa xanh", "bach hoa xanh"),
        ("nạp ví momo trả tiền điện 700k", "m", "momo"),
        # Vietnamese code points: the word is counted in code points, not bytes
        ("chuyển cho thảo nguyên 2tr", "ảo", "thảo"),
        ("trả nợ chị dâu 1tr", "dâ", "dâu"),
        ("mua quà cho mẹ 3tr", "ẹ", "mẹ"),
    ],
)
def test_fragment_extends_to_the_whole_word(text, piece, expected):
    assert snap(text, piece) == expected


@pytest.mark.parametrize(
    ("text", "piece"),
    [
        ("no tien nha chu tro 1tr5 chua tra", "chu tro"),
        ("go! big c siêu thị 1tr1", "go! big c"),
        ("mượn chị hai 3 triệu", "chị hai"),
        ("vay chị dâu 5tr sửa nhà", "chị dâu"),
        ("rut tien vpbank 1tr", "vpbank"),
    ],
)
def test_whole_words_and_multi_word_spans_are_kept(text, piece):
    assert snap(text, piece) == piece


def test_only_the_two_ends_move_and_only_outward():
    text = "taxi xanh sm ra san bay 210k"
    start = text.index("anh sm ra")
    lo, hi = snap_to_words(text, start, start + len("anh sm ra"))
    assert text[lo:hi] == "xanh sm ra"
    assert lo <= start and hi >= start + len("anh sm ra")


def test_edge_punctuation_of_the_extension_is_trimmed():
    assert snap("chuyển (vpbank), 1tr", "pban") == "vpbank"
    assert snap('gap "Vy"! 300k', "V") == "Vy"
    # punctuation inside the word stays; the word is the maximal run of non-whitespace
    assert snap("chuyen cho a-b 1tr", "-") == "a-b"


def test_empty_span_and_no_span_stay_unchanged():
    assert snap_to_words("rut tien vpbank 1tr", 5, 5) == (5, 5)
    assert snap_span_to_words(None, "rut tien vpbank 1tr") is None


def test_snap_keeps_member_tokens():
    span = DecodedSpan(10, 15, (4, 5))
    snapped = snap_span_to_words(span, "rut tien vpbank 1tr")
    assert snapped == DecodedSpan(9, 15, (4, 5))


@needs_bundle
def test_predictor_snap_is_off_by_default_and_never_shrinks():
    plain = GidiPredictor.from_bundle(BUNDLE)
    snapped = GidiPredictor.from_bundle(BUNDLE, snap_words=True)
    notes = [
        "rut tien vpbank 1tr",
        "thanh toán thẻ hsbc 7,2tr",
        "pizza 4p tối t7 389.000đ",
        "cho ban Quang muon 2tr",
        "ăn phở 45k",
        "1tr",
    ]
    for note in notes:
        a, b = plain.predict(note), snapped.predict(note)
        assert a.type == b.type
        if a.target_span is None:
            assert b.target_span is None
            continue
        (s0, e0), (s1, e1) = a.target_span, b.target_span
        assert s1 <= s0 and e1 >= e0
        assert s1 == 0 or note[s1 - 1].isspace()
        assert e1 == len(note) or note[e1].isspace()
        assert b.target == note[s1:e1]
