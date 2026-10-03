"""Word lists of the deterministic value parser.

Every word is accent-folded lowercase (``tỷ`` -> ``ty``, ``đồng`` -> ``dong``): the parser
matches on its folded shadow string, so accented and unaccented notes hit the same entries.
"""

from __future__ import annotations

# --- money units ---------------------------------------------------------------------------
THOUSAND_LETTER = frozenset({"k"})
THOUSAND_WORDS = frozenset({"nghin", "ngan"})
MILLION = frozenset({"tr", "trieu"})
BILLION = frozenset({"ty"})
# Slang for 100k (xi, chai, lit) and a million (cu). Ambiguous with quantities, see SLANG_QTY.
SLANG = frozenset({"cu", "xi", "chai", "lit"})
CURRENCY_WORDS = frozenset({"d", "vnd", "dong"})
CURRENCY_SIGNS = frozenset({"₫"})
# Units after which a half ("rưỡi") or a trailing digit can continue the amount (2 củ rưỡi).
CONTINUABLE = MILLION | BILLION | SLANG | THOUSAND_WORDS
WORD_UNITS = THOUSAND_WORDS | frozenset({"trieu", "ty", "cu", "chai", "lit", "xi"})
HALF = frozenset({"ruoi"})

# Slang unit followed by one of these is a quantity of goods, not money (5 lít xăng, 2 chai bia).
SLANG_QTY = frozenset(
    {"xang", "dau", "nhot", "nuoc", "bia", "ruou", "cafe", "coca", "pepsi", "gas", "khoai"}
)
# Same, but the folded form is ambiguous with another word, so the accented spelling is required
# (``sữa`` milk vs ``sửa`` repair, ``trà`` tea vs ``trả`` pay, ``hành`` onion vs ``Hạnh``).
SLANG_QTY_ACCENTED = frozenset({"sữa", "trà", "hành", "tỏi", "gừng", "cải"})

# --- number words (for "bốn triệu", "hai trăm nghìn", "nửa củ") ------------------------------
DIGIT_WORDS = frozenset({"mot", "hai", "ba", "bon", "nam", "sau", "bay", "tam", "chin", "muoi"})
NUMBER_WORDS = DIGIT_WORDS | frozenset({"tram", "linh", "le", "lam", "muoi"})
HALF_PREFIX = frozenset({"nua"})

# --- context that explains a number as something other than money ----------------------------
# Word BEFORE the number: month, instalment, period, week, day, ...
PERIOD_BEFORE = frozenset(
    {"thang", "ky", "dot", "lan", "quy", "nam", "tuan", "ngay", "mung", "thu", "lop", "tang",
     "phong", "chia", "so", "hang"}
)  # fmt: skip
# Of those, words that are also common given names: only an excluding context when lowercase.
NAME_LIKE = frozenset({"lan", "thu", "nam", "quy", "tang", "phong", "dot"})
# Word AFTER the number: time units and counted things.
QUANTITY_AFTER = frozenset(
    {
        # time and periods
        "thang", "ngay", "tuan", "nam", "quy", "gio", "phut", "giay", "buoi", "hom", "dem",
        "lan", "dot", "ky", "tieng", "tuoi", "t", "h", "p",
        # counted things
        "nguoi", "em", "be", "to", "ly", "coc", "ve", "phan", "suat", "cuon", "quyen", "mon",
        "cai", "chiec", "con", "cay", "bo", "doi", "cap", "goi", "hop", "thung", "lon", "lang",
        "chi", "phong", "tang", "tiet", "bai", "trang", "chuyen", "cuoc", "vien", "mieng",
        "trai", "bat", "dia", "tui", "cuc", "thanh", "sp",
        # measures
        "kg", "kilo", "gam", "gram", "g", "mg", "can", "yen", "tan", "m", "km", "cm", "mm",
        "ml", "cc", "l", "m2", "m3", "inch", "gb", "tb", "mb",
    }
)  # fmt: skip
WEEKDAY_WORD = "thu"  # "thứ 6": a single digit 2-8 after it is a weekday
