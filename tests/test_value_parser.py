"""Deterministic value parser: spans in the caller's string, competing numbers, null."""

from __future__ import annotations

import time
import unicodedata

import pytest

from gidi.value_parser import find_candidates, parse_value
from gidi.value_parser.parser import fold


def value(text: str) -> str | None:
    span = parse_value(text)
    return None if span is None else span.text


# --------------------------------------------------------------------------- money forms


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("cơm trưa 45k", "45k"),
        ("cafe 35 K", "35 K"),
        ("lương part time 2tr4", "2tr4"),
        ("đóng học phí 1tr150", "1tr150"),
        ("mua laptop 15,5 triệu", "15,5 triệu"),
        ("vay Thảo 1 triệu 2 mua gạo", "1 triệu 2"),
        ("tiền xe 2 triệu rưỡi", "2 triệu rưỡi"),
        ("bán nhà 1 tỷ 2", "1 tỷ 2"),
        ("mượn bạn 3 củ", "3 củ"),
        ("bảo hiểm chi trả 3 củ 5", "3 củ 5"),
        ("cho vay 5 xị", "5 xị"),
        ("nhờ mẹ 2 củ rưỡi", "2 củ rưỡi"),
        ("trả nợ 500 ngàn", "500 ngàn"),
        ("tiền gas 480.000", "480.000"),
        ("thanh toán 12.345.678 vnđ", "12.345.678 vnđ"),
        ("ăn tối 150.000đ", "150.000đ"),
        ("chuyển 1 500 000 vào tk", "1 500 000"),
        ("mượn mẹ 5000000 đóng tiền", "5000000"),
        ("tiền nhà bốn triệu", "bốn triệu"),
        ("quà hai trăm nghìn", "hai trăm nghìn"),
        ("mượn chị nửa củ", "nửa củ"),
        ("cơm tấm 100", "100"),
    ],
)
def test_money_forms(text: str, expected: str) -> None:
    assert value(text) == expected


@pytest.mark.parametrize(
    ("accented", "plain", "expected_plain"),
    [
        ("mượn bạn 3 củ", "muon ban 3 cu", "3 cu"),
        ("vay 1 triệu 2", "vay 1 trieu 2", "1 trieu 2"),
        ("trả 500 nghìn", "tra 500 nghin", "500 nghin"),
        ("cho vay 2 tỷ rưỡi", "cho vay 2 ty ruoi", "2 ty ruoi"),
    ],
)
def test_unaccented_note_gives_the_same_span_shape(
    accented: str, plain: str, expected_plain: str
) -> None:
    accented_value = value(accented)
    assert accented_value is not None
    assert fold(accented_value) == value(plain) == expected_plain


# --------------------------------------------------------------------------- competing numbers


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("cho a Nam vay 1 triệu 20/10", "1 triệu"),  # a date is not the "1 triệu 2" tail
        ("20/10 mua quà mẹ 500", "500"),
        ("ăn 2 tô phở 70", "70"),
        ("mua 3 vé 150k.", "150k"),
        ("đổ 5 lít xăng 120k", "120k"),
        ("cho anh Tu vay 3 lít sửa xe", "3 lít"),  # sửa (repair) is not sữa (milk)
        ("vay Hà 2 củ trả nợ", "2 củ"),  # trả (pay) is not trà (tea)
        ("mua 2 chai bia 90k", "90k"),
        ("tiền điện tháng 10 hết 612k", "612k"),
        ("trả góp kỳ 3 1tr5", "1tr5"),
        ("trả nợ Huy kỳ 3 2tr", "2tr"),
        ("nap dien thoai 0912345678 100", "100"),
        ("nạp 50k vào 0987 654 321", "50k"),
        ("tiền lãi tiết kiệm 5.5%/năm về 420", "420"),
        ("hoàn thuế thu nhập năm 2023 3.350.000", "3.350.000"),
        ("gửi sổ kỳ hạn 6 tháng 2000", "2000"),
        ("mua gạo 10kg 190k", "190k"),
        ("lẩu vs team 4 người chia 180k", "180k"),
        ("cho Lan mượn 300", "300"),  # Lan is a name, not lần
        ("trả nợ lần 2 500k", "500k"),
        ("sáng mùng 1 lì xì bà nội 200k lấy hên", "200k"),
        ("họp lúc 10h30 chi 400k", "400k"),
        ("hôm thứ 6 chi 70", "70"),
        ("tiền nhà t11 4tr2", "4tr2"),
    ],
)
def test_competing_numbers(text: str, expected: str) -> None:
    assert value(text) == expected


def test_several_explicit_amounts_pick_the_leading_one_and_bare_numbers_the_last() -> None:
    assert value("grabfood tối 89k ship 15k") == "89k"
    assert value("cho vay 3 100") == "100"


def test_explicit_unit_beats_bare_number_and_slang_beats_bare() -> None:
    assert value("mua 5 100k") == "100k"
    assert value("mượn 100 5 củ") == "5 củ"


# --------------------------------------------------------------------------- no amount


@pytest.mark.parametrize(
    "text",
    [
        "nạp momo",
        "shopee hoàn tiền rồi",
        "thưởng tết 2 tháng lương",
        "trích 20% lương để dành",
        "đóng tiền net.",
        "mua 2 chai bia",
        "họp lúc 10h30",
        "gọi 0912345678",
        "",
        "   ",
    ],
)
def test_no_amount_is_none(text: str) -> None:
    assert parse_value(text) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("mượn mẹ 3tr đóng học kỳ 2", "3tr"),  # đóng = to pay, not đồng
        ("muon me 3tr dong hoc phi", "3tr"),
        ("tiền gửi xe 350.000 đồng", "350.000 đồng"),
        ("cho Hoa muon 250.000 dong", "250.000 dong"),
    ],
)
def test_dong_is_a_currency_only_when_it_is_dong(text: str, expected: str) -> None:
    assert value(text) == expected


# --------------------------------------------------------------------------- caller's offsets


def test_span_is_a_substring_of_the_original_with_exact_offsets() -> None:
    text = "  ship đồ ăn baemin 67k,  tối nay"
    span = parse_value(text)
    assert span is not None
    assert text[span.start : span.end] == span.text == "67k"


def test_decomposed_input_returns_offsets_in_the_decomposed_string() -> None:
    text = unicodedata.normalize("NFD", "mượn Lộc 6 xị đóng tiền học")
    assert text != unicodedata.normalize("NFC", text)
    span = parse_value(text)
    assert span is not None
    assert text[span.start : span.end] == span.text
    assert unicodedata.normalize("NFC", span.text) == "6 xị"
    assert span.end == len(unicodedata.normalize("NFD", "mượn Lộc 6 xị"))


def test_decomposed_unit_is_kept_whole_not_cut_inside_a_character() -> None:
    text = unicodedata.normalize("NFD", "cho vay 3 củ")
    span = parse_value(text)
    assert span is not None
    assert span.text == text[text.index("3") :]


def test_astral_characters_before_the_amount_shift_offsets_by_code_points() -> None:
    text = "🍜 ăn phở 70k"
    span = parse_value(text)
    assert span is not None
    assert (span.start, span.end) == (text.index("70k"), len(text))


def test_span_never_includes_trailing_punctuation_or_surrounding_space() -> None:
    for text, expected in [("trả 5 củ, hẹn t10", "5 củ"), ("ăn 1tr.", "1tr"), ("(500k)", "500k")]:
        assert value(text) == expected


# --------------------------------------------------------------------------- robustness


def test_candidates_are_ordered_and_disjoint() -> None:
    cands = find_candidates("a 5k b 3 củ c 1tr5 d 100")
    assert [c.start for c in cands] == sorted(c.start for c in cands)
    assert all(a.end <= b.start for a, b in zip(cands, cands[1:], strict=False))


@pytest.mark.parametrize(
    "text",
    [
        "1 dong " * 20_000,
        "123 " * 30_000,
        "hai " * 30_000,
        "0912 345 678 " * 8_000,
        "1." * 50_000,
        "5 củ " * 20_000,
    ],
)
def test_adversarial_inputs_run_in_linear_time_and_never_raise(text: str) -> None:
    start = time.perf_counter()
    first = parse_value(text)
    assert time.perf_counter() - start < 3.0
    assert parse_value(text) == first
    if first is not None:
        assert text[first.start : first.end] == first.text
