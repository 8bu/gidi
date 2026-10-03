"""Value-span proposer and validators for annotation-v2 (contract: ``configs/annotation-v2.yaml``).

annotation-v2 adds one thing to annotation-v1: the **value span**, the exact substring of a note
that is the monetary amount (``"cơm tấm 100"`` -> ``"100"``, ``"mượn chú hai 5 xị"`` -> ``"5 xị"``).
It is span selection only. Nothing here converts a span into a number, and no numeric value is
ever stored.

``propose_value`` is a deterministic, rule-based *proposer*. It finds money-looking expressions,
discards numbers that context explains as something else (months, dates, quantities, ...), and
either picks the single remaining amount or refuses to choose. It never forces a choice: more than
one plausible amount gives ``chosen=None``. Its output is advisory (Quet proposals) or, for the
high-confidence subset, a label with ``provenance: rule``; it is never a human label.

The module also holds the validators for the value label files and for merged training records,
which mirror the Quet label rules plus the Gidi rules (``uncertain`` needs a note, ``no_amount``
has a null span, a complete ``amount`` has a span).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

RULE_VERSION = "value-span-rules-v1"
SEED = "value-span-v1"
DEFAULT_VALUE_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "annotation-v2.yaml"

VALUE_TAGS: tuple[str, ...] = ("O", "B-VALUE", "I-VALUE")
SPAN_FIELDS: tuple[str, ...] = ("text", "start", "end")

# Review reasons, highest priority first (this is also the review-queue order).
REVIEW_PRIORITY: tuple[str, ...] = (
    "multiple_money_candidates",
    "no_candidate",
    "multi_number",
    "compound_amount",
    "slang",
    "bare_number",
    "unusual_punctuation",
)

# ---------------------------------------------------------------------------------------------
# Text preparation
# ---------------------------------------------------------------------------------------------


def _lower_char(c: str) -> str:
    low = c.lower()
    return low if len(low) == 1 else c


def _fold_char(c: str) -> str:
    """Lowercase, diacritics stripped, đ -> d; always exactly one character."""
    if c in "đĐ":
        return "d"
    base = "".join(x for x in unicodedata.normalize("NFD", c) if unicodedata.category(x) != "Mn")
    base = base.lower()
    return base if len(base) == 1 else _lower_char(c)


def fold(text: str) -> str:
    """Lowercase ASCII-ish view of ``text`` (same length for NFC text)."""
    return "".join(_fold_char(c) for c in text)


def fold_keep_case(text: str) -> str:
    """Strip diacritics (đ -> d) but keep letter case."""
    out = []
    for c in text:
        if c in "đĐ":
            out.append("d" if c == "đ" else "D")
            continue
        decomposed = unicodedata.normalize("NFD", c)
        base = "".join(x for x in decomposed if unicodedata.category(x) != "Mn")
        out.append(base if len(base) == 1 else c)
    return "".join(out)


@dataclass(frozen=True)
class _Prepared:
    """NFC view of the caller's string, with a map back to the caller's code-point offsets."""

    nfc: str
    lo: str  # lowercase NFC (accents kept)
    fo: str  # lowercase, accents stripped
    starts: tuple[int, ...] | None  # NFC index -> start offset in the original string
    ends: tuple[int, ...] | None  # NFC index -> end offset in the original string

    def to_original(self, start: int, end: int) -> tuple[int, int]:
        if self.starts is None or self.ends is None:
            return start, end
        return self.starts[start], self.ends[end - 1]


def _prepare(text: str) -> _Prepared:
    if unicodedata.is_normalized("NFC", text):
        nfc, starts, ends = text, None, None
    else:
        chars: list[str] = []
        s_map: list[int] = []
        e_map: list[int] = []
        i, n = 0, len(text)
        while i < n:
            j = i + 1
            while j < n and unicodedata.combining(text[j]):
                j += 1
            for ch in unicodedata.normalize("NFC", text[i:j]):
                chars.append(ch)
                s_map.append(i)
                e_map.append(j)
            i = j
        nfc, starts, ends = "".join(chars), tuple(s_map), tuple(e_map)
    lo = "".join(_lower_char(c) for c in nfc)
    return _Prepared(nfc, lo, fold(nfc), starts, ends)


# ---------------------------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """A money-looking expression; offsets are code points into the caller's string."""

    start: int
    end: int
    text: str
    kind: str  # explicit_unit | slang | number | bare_number
    tags: tuple[str, ...]

    def as_span(self) -> dict[str, Any]:
        return {"text": self.text, "start": self.start, "end": self.end}


@dataclass(frozen=True)
class Exclusion:
    """A number the proposer decided is not money, with the reason."""

    start: int
    end: int
    text: str
    reason: str


@dataclass(frozen=True)
class ValueProposal:
    text: str
    candidates: tuple[Candidate, ...]
    chosen: Candidate | None
    confidence: float
    categories: tuple[str, ...]
    needs_review: bool
    review_reasons: tuple[str, ...]
    reason: str
    excluded: tuple[Exclusion, ...]

    @property
    def span(self) -> dict[str, Any] | None:
        return self.chosen.as_span() if self.chosen else None

    @property
    def auto_accept(self) -> bool:
        """A rule label may be written without human review."""
        return self.chosen is not None and not self.needs_review


# ---------------------------------------------------------------------------------------------
# Recognition rules
# ---------------------------------------------------------------------------------------------

_NUM_RE = re.compile(
    r"(?<![\w/])(?<!\d[.,])(?:\d{1,3}(?:[.,]\d{3})+(?!\d)|\d+(?:[.,]\d+)?)",
)
_SEP3_RE = re.compile(r"\d{1,3}(?:[.,]\d{3})+")

_CUR = r"(?:\s?(?:đồng|vnđ|vnd|đ)(?!\w))?"
_HALF = r"\s?(?:rưỡi|ruoi)"
_UNITS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("k", re.compile(rf"\s?k(?P<tail>\d{{1,3}})?{_CUR}(?!\w)")),
    (
        "tr",
        re.compile(rf"\s?(?:triệu|trieu|tr)(?:(?P<tail>\d{{1,3}})|(?P<half>{_HALF}))?{_CUR}(?!\w)"),
    ),
    (
        "nghin",
        re.compile(rf"\s?(?:nghìn|nghin|ngàn|ngan(?!\s?h[aà]ng))(?P<half>{_HALF})?{_CUR}(?!\w)"),
    ),
    (
        "ty",
        re.compile(rf"\s?(?:tỷ|tỉ|ty)(?:(?P<tail>\d{{1,3}})|(?P<half>{_HALF}))?{_CUR}(?!\w)"),
    ),
    (
        "slang",
        re.compile(rf"\s?(?P<w>củ|cu|xị|xi|chai|lít|lit)(?P<half>{_HALF})?{_CUR}(?!\w)"),
    ),
)
_COMPOUND_UNITS = frozenset({"tr", "nghin", "ty"})
_COMPOUND_RE = re.compile(r"\s(?P<n>\d{1,3})(?![\w]|[.,]\d|[/:%])")
_CUR_ONLY = re.compile(r"\s?(?P<w>đồng|dong|đ|vnđ|vnd|d)(?!\w)")
_CUR_TAIL = re.compile(_CUR)

_DIGIT_WORD = r"(?:mot|hai|ba|bon|tu|nam|sau|bay|tam|chin|lam|nham)"
_NUMWORD_FOLDED = re.compile(
    rf"(?:nua\s+)?(?:{_DIGIT_WORD}\s+(?:tram|chuc)\s+)?(?:(?:le|linh)\s+)?"
    rf"(?:(?:{_DIGIT_WORD}\s+)?muoi\s+)?(?:{_DIGIT_WORD}\s+)?"
)
_WORDNUM_RE = re.compile(
    r"(?<!\w)(?P<words>(?:(?:nửa|nua|một|mot|hai|ba|bốn|bon|tư|tu|năm|nam|sáu|sau|bảy|bay|tám|tam"
    r"|chín|chin|mười|muoi|mươi|trăm|tram|chục|chuc|lẻ|le|linh|lăm|lam|nhăm|nham)\s+){1,4})"
    rf"(?P<unit>triệu|trieu|nghìn|nghin|ngàn|tỷ|củ|cu)(?P<half>{_HALF})?(?!\w)"
)
_WORD_RE = re.compile(r"\S+")
_NEXT_WORD_RE = re.compile(r"\s+([a-z]+)")
_NEXT_WORD_LO_RE = re.compile(r"\s+([^\W\d_]+)")

# Nouns after "củ"/"chai"/"lít"/"xị" that make the number a quantity, not money.
_VEG = frozenset(
    [
        "hanh",
        "toi",
        "cai",
        "ca",
        "khoai",
        "san",
        "gung",
        "nghe",
        "rieng",
        "su",
        "sen",
        "den",
        "dau",
        "cam",
        "bap",
    ]
)
# Accent-sensitive where the folded form collides with another word ("sữa" milk vs "sửa" repair).
_GOODS = frozenset(
    [
        "nước",
        "nuoc",
        "dầu",
        "xăng",
        "xang",
        "nhớt",
        "nhot",
        "sữa",
        "bia",
        "rượu",
        "ruou",
        "tương",
        "mắm",
        "mam",
        "giấm",
        "coca",
        "pepsi",
    ]
)
_AMBIGUOUS_SLANG = frozenset({"chai", "lit", "xi"})  # folded; usually a quantity

_QTY_RE = re.compile(
    r"\s?(?P<w>to|ly|ve|cai|chiec|kg|g|gr|gam|gb|mb|tb|km|m|cm|mm|lang|chi|phan|phut|gio|nguoi"
    r"|ng|suat|hop|lon|goi|thung|cuon|quyen|bo|con|trai|mieng|doi|lan|tuan|ngay|tieng|nam|thang"
    r"|buoi|bua|mon|loai|tiet|chuyen|xe|cay|vien|set|hat|thanh|chen|tui|bich|tuoi|diem|giay|dem"
    r"|ky|dot|lit|chai|cu)(?!\w)"
)
_ORD_BEFORE_RE = re.compile(
    r"(?<!\w)(?P<w>hoc ky|thang|quy|ky|ki|lan|dot|tuan|ngay|mung|mong|giai|top|lop|cap|trang"
    r"|vong|chia|iphone|galaxy|samsung|ps|ver|size|pixel)\s?$"
)
_SHORT_ORDINALS = frozenset({"cap", "chia", "top", "lop"})  # only 1-2 digit numbers
_WEEKDAY_BEFORE_RE = re.compile(r"(?<!\w)thu\s?$")
_YEAR_BEFORE_RE = re.compile(r"(?<!\w)nam\s?$")
_ID_BEFORE_RE = re.compile(r"(?<!\w)(?:stk|sdt|cccd|cmnd|mst|id|ma don|ma so)\s?:?\s?$")
_DATE_BEFORE_RE = re.compile(r"\d\s?[/:]\s?$")
_AFTER_PCT_RE = re.compile(r"\s?%")
_AFTER_DATE_RE = re.compile(r"\s?[/:]\s?\d")
_LETTER_ADJ_RE = re.compile(r"[^\W\d_]")
_CUR_WORD_STRONG = frozenset({"đ", "đồng", "vnđ", "vnd"})
_CUR_TRAILING_RE = re.compile(r"(?:đồng|vnđ|vnd|đ)$")
_ODD_ADJACENT = frozenset("~=*()[]{}|^_@#<>-–—×+&:;/")
_ODD_CATEGORIES = frozenset({"So", "Sk", "Cf", "Co", "Cs", "Cn"})
_ODD_WHITESPACE = frozenset("\u00a0\t\n\r\u2009\u202f")


def _first_upper(prep: _Prepared, index: int) -> bool:
    return prep.nfc[index].isupper()


# Quantity words whose folded form collides with a common non-quantity word ("về", "bố", ...).
_QTY_SPELLINGS = {
    "ve": {"vé", "ve"},
    "to": {"tô", "tờ", "to"},
    "chi": {"chỉ", "chi"},
    "bo": {"bó", "bộ", "bo"},
    "doi": {"đôi", "doi"},
    "phan": {"phân", "phần", "phan"},
    "lan": {"lần", "lan"},
    "tuan": {"tuần", "tuan"},
    "nam": {"năm", "nam"},
}


def _qty_after(prep: _Prepared, pos: int) -> bool:
    m = _QTY_RE.match(prep.fo, pos)
    if not m or _first_upper(prep, m.start("w")):
        return False
    allowed = _QTY_SPELLINGS.get(m.group("w"))
    return allowed is None or prep.lo[m.start("w") : m.end("w")] in allowed


def _next_word(prep: _Prepared, pos: int) -> str:
    m = _NEXT_WORD_RE.match(prep.fo, pos)
    return m.group(1) if m else ""


def _context_exclusion(prep: _Prepared, a: int, b: int, numtext: str) -> str | None:
    """Why the unit-less number at ``[a, b)`` is not money, or ``None``."""
    fo = prep.fo
    plain = numtext.isdigit()
    digits = len(numtext)
    if _AFTER_PCT_RE.match(fo, b):
        return "percent"
    if _AFTER_DATE_RE.match(fo, b) or _DATE_BEFORE_RE.search(fo[max(0, a - 4) : a]):
        return "date_or_time"
    if b < len(fo) and _LETTER_ADJ_RE.match(fo, b):
        return "attached_unit"
    if _qty_after(prep, b):
        return "quantity"
    if plain and _ID_BEFORE_RE.search(fo, 0, a):
        return "identifier"
    if plain and digits == 4 and numtext[:2] in ("19", "20") and _YEAR_BEFORE_RE.search(fo, 0, a):
        return "year"
    if plain and digits == 1 and "2" <= numtext <= "8":
        m = _WEEKDAY_BEFORE_RE.search(fo, 0, a)
        if m and not _first_upper(prep, m.start()):
            return "weekday"
    if plain and digits <= 3:
        m = _ORD_BEFORE_RE.search(fo, 0, a)
        if m and not _first_upper(prep, m.start("w")):
            if m.group("w") in _SHORT_ORDINALS and digits > 2:
                return None
            return "period_or_ordinal"
    return None


def _match_unit(prep: _Prepared, b: int) -> tuple[int, str, dict[str, bool]] | None:
    """Match a money unit right after a number ending at ``b``: ``(end, unit, flags)``."""
    lo = prep.lo
    for name, pattern in _UNITS:
        m = pattern.match(lo, b)
        if not m:
            continue
        groups = m.groupdict()
        flags = {"compact": bool(groups.get("tail")), "half": bool(groups.get("half"))}
        end = m.end()
        word = ""
        if name == "slang":
            word = prep.fo[m.start("w") : m.end("w")]
            nxt = _next_word(prep, m.end("w"))
            nxt_lo = _NEXT_WORD_LO_RE.match(prep.lo, m.end("w"))
            goods = nxt_lo is not None and nxt_lo.group(1) in _GOODS
            if (word == "cu" and nxt in _VEG) or (word in _AMBIGUOUS_SLANG and goods):
                return None
        compound_unit = name in _COMPOUND_UNITS or word == "cu"
        if compound_unit and not flags["compact"] and not flags["half"]:
            cm = _COMPOUND_RE.match(lo, end)
            if cm and not _qty_after(prep, cm.end("n")) and not _match_unit(prep, cm.end("n")):
                end = cm.end("n")
                tail = _CUR_TAIL.match(lo, end)
                if tail:
                    end = tail.end()
                flags["compound"] = True
        return end, name, flags
    return None


def _wordnum_start(prep: _Prepared, m: re.Match[str]) -> int | None:
    """Start of the spelled-out number in ``m`` after dropping leading names/stray words."""
    words = list(_WORD_RE.finditer(m.group("words")))
    base = m.start("words")
    for i, w in enumerate(words):
        if _first_upper(prep, base + w.start()):
            continue
        folded = " ".join(prep.fo[base + x.start() : base + x.end()] for x in words[i:]) + " "
        if _NUMWORD_FOLDED.fullmatch(folded):
            return base + w.start()
    return None


def _scan(
    prep: _Prepared,
) -> tuple[list[tuple[int, int, str, tuple[str, ...]]], list[tuple[int, int, str]]]:
    """Candidates ``(start, end, kind, tags)`` and exclusions ``(start, end, reason)``."""
    lo = prep.lo
    cands: list[tuple[int, int, str, tuple[str, ...]]] = []
    excl: list[tuple[int, int, str]] = []
    covered: list[tuple[int, int]] = []

    def is_covered(index: int) -> bool:
        return any(s <= index < e for s, e in covered) or any(s <= index < e for s, e, _ in excl)

    # Spelled-out amounts ("hai trăm nghìn", "bốn triệu").
    for m in _WORDNUM_RE.finditer(lo):
        unit = m.group("unit")
        if unit in ("củ", "cu") and _next_word(prep, m.end("unit")) in _VEG:
            continue
        start = _wordnum_start(prep, m)
        # unaccented "cu" is also "cũ"/"cụ": only "nửa cu" (half a million) is taken as money
        if start is None or (unit == "cu" and not prep.fo[start:].startswith("nua")):
            continue
        tail = _CUR_TAIL.match(lo, m.end())
        end = tail.end() if tail else m.end()
        cands.append((start, end, "slang", ("explicit_unit", "slang", "number_words")))
        covered.append((start, end))

    pos = 0
    while True:
        m = _NUM_RE.search(lo, pos)
        if not m:
            break
        a, b = m.span()
        pos = b
        if is_covered(a):
            continue
        numtext = m.group()
        sep3 = bool(_SEP3_RE.fullmatch(numtext))
        decimal = not sep3 and ("." in numtext or "," in numtext)
        long_plain = numtext.isdigit() and len(numtext) >= 4

        unit = _match_unit(prep, b)
        if unit:
            end, name, flags = unit
            kind = "slang" if name == "slang" else "explicit_unit"
            tags = [kind]
            if decimal or sep3:
                tags.append("decimal_unit")
            if flags["compact"]:
                tags.append("compact_unit")
            if flags["half"]:
                tags.append("slang")
            if flags.get("compound"):
                tags.append("compound_amount")
            if _CUR_TRAILING_RE.search(lo[a:end]):
                tags.append("currency_suffix")
            cands.append((a, end, kind, tuple(dict.fromkeys(tags))))
            covered.append((a, end))
            pos = end
            continue

        cur = _CUR_ONLY.match(lo, b)
        if cur and (cur.group("w") in _CUR_WORD_STRONG or sep3 or long_plain):
            end = cur.end()
            covered.append((a, end))
            pos = end
            if numtext.isdigit() and _ID_BEFORE_RE.search(prep.fo, 0, a):
                excl.append((a, end, "identifier"))
                continue
            tags = ["currency_suffix"]
            if sep3:
                tags.append("separator_format")
            elif long_plain and len(numtext) >= 5:
                tags.append("long_digits")
            cands.append((a, end, "number", tuple(tags)))
            continue

        covered.append((a, b))
        reason = _context_exclusion(prep, a, b, numtext)
        if reason:
            excl.append((a, b, reason))
        elif sep3:
            cands.append((a, b, "number", ("separator_format",)))
        elif numtext.isdigit() and len(numtext) >= 5:
            if len(numtext) >= 10 or numtext.startswith("0"):
                excl.append((a, b, "identifier"))
            else:
                cands.append((a, b, "number", ("long_digits",)))
        else:
            cands.append((a, b, "bare_number", ("bare_number",)))

    # Digit runs the number scan could not start at: dates, times, glued tokens, other.
    for m in re.finditer(r"\d+", lo):
        a, b = m.span()
        if is_covered(a):
            continue
        before = lo[a - 1] if a else ""
        after = lo[b] if b < len(lo) else ""
        dotted = before in (".", ",") and a >= 2 and lo[a - 2].isdigit()
        if before in ("/", ":") or dotted:
            reason = "date_or_time"
        elif (before and (before.isalpha() or before == "_")) or (after and after.isalpha()):
            reason = "attached_unit"
        else:
            reason = "unexplained"
        excl.append((a, b, reason))
    cands.sort()
    excl.sort()
    return cands, excl


def _unusual_punctuation(prep: _Prepared, spans: Sequence[tuple[int, int]]) -> bool:
    for c in prep.nfc:
        if c in _ODD_WHITESPACE or unicodedata.category(c) in _ODD_CATEGORIES:
            return True
    text = prep.nfc
    for a, b in spans:
        before = text[a - 1] if a else ""
        if before == " " and a >= 2:
            before = text[a - 2]
        after = text[b] if b < len(text) else ""
        if after == " " and b + 1 < len(text):
            after = text[b + 1]
        if before in _ODD_ADJACENT or after in _ODD_ADJACENT:
            return True
    return False


# Joiners that make two digit runs one expression: dates and times ("20/10", "8:30", "1.2.3",
# "8h30"). An amount ("1tr5", "1.500.000", "2 triệu rưỡi", "12,5tr") is already one span.
_JOIN_SLASH_RE = re.compile(r"\s?[/:]\s?")


def _numeric_expression_count(
    lo: str,
    cands: Sequence[tuple[int, int, str, tuple[str, ...]]],
    excl: Sequence[tuple[int, int, str]],
) -> int:
    """Distinct numeric expressions in the note (spelled-out number words do not count)."""
    spans = [(a, b, "") for a, b, _, tags in cands if "number_words" not in tags]
    spans += [(a, b, reason) for a, b, reason in excl]
    spans.sort()
    count = 0
    prev_end: int | None = None
    for a, b, reason in spans:
        if prev_end is not None:
            gap = lo[prev_end:a]
            if (
                _JOIN_SLASH_RE.fullmatch(gap)
                or (gap in (".", ",") and reason == "date_or_time")
                or (gap in ("h", "g") and prev_end > 0 and lo[prev_end - 1].isdigit())
            ):
                prev_end = b
                continue
        count += 1
        prev_end = b
    return count


_CATEGORY_ORDER: tuple[str, ...] = (
    "explicit_unit",
    "decimal_unit",
    "compact_unit",
    "compound_amount",
    "currency_suffix",
    "separator_format",
    "long_digits",
    "bare_number",
    "slang",
    "number_words",
    "multi_number",
    "multiple_money_candidates",
    "no_candidate",
    "unusual_punctuation",
    "context_excluded",
    "unaccented",
)
_CONTEXT_REASONS = frozenset(
    {"percent", "date_or_time", "attached_unit", "quantity", "identifier", "year", "weekday"}
    | {"period_or_ordinal", "slang_unit_quantity"}
)


def propose_value(text: str) -> ValueProposal:
    """Propose the value span of ``text``; ``chosen`` is ``None`` when no single amount is clear.

    Offsets are code points into ``text`` as given (decomposed accents are matched as their NFC
    form and mapped back, so ``text[start:end] == span.text`` always holds).
    """
    prep = _prepare(text)
    raw_cands, raw_excl = _scan(prep)
    to_orig = prep.to_original

    def make_c(a: int, b: int, kind: str, tags: tuple[str, ...]) -> Candidate:
        s, e = to_orig(a, b)
        return Candidate(s, e, text[s:e], kind, tags)

    def make_x(a: int, b: int, reason: str) -> Exclusion:
        s, e = to_orig(a, b)
        return Exclusion(s, e, text[s:e], reason)

    cands = [make_c(*c) for c in raw_cands]
    excluded = [make_x(*x) for x in raw_excl]

    strong = [c for c in cands if c.kind != "bare_number"]
    bare = [c for c in cands if c.kind == "bare_number"]
    # chai / lít / xị are usually quantities: beside another money expression they are dropped.
    firm = [c for c in strong if not (c.kind == "slang" and _is_ambiguous_slang(c.text))]
    if firm and len(firm) < len(strong):
        for c in strong:
            if c not in firm:
                excluded.append(Exclusion(c.start, c.end, c.text, "slang_unit_quantity"))
        strong = firm
    pool = strong
    if strong:
        for c in bare:
            excluded.append(Exclusion(c.start, c.end, c.text, "bare_beside_money"))
    else:
        pool = bare

    chosen = pool[0] if len(pool) == 1 else None
    multi_number = _numeric_expression_count(prep.lo, raw_cands, raw_excl) >= 2

    cats: set[str] = set()
    for c in [chosen] if chosen else pool:
        cats.update(c.tags)
    if multi_number:
        cats.add("multi_number")
    if len(pool) >= 2:
        cats.add("multiple_money_candidates")
    if not pool:
        cats.add("no_candidate")
    if any(x.reason in _CONTEXT_REASONS for x in excluded):
        cats.add("context_excluded")
    if not any(ord(ch) > 127 for ch in text):
        cats.add("unaccented")
    if _unusual_punctuation(prep, [(a, b) for a, b, _, _ in raw_cands]):
        cats.add("unusual_punctuation")
    categories = tuple(c for c in _CATEGORY_ORDER if c in cats)

    reasons = tuple(r for r in REVIEW_PRIORITY if r in cats)
    needs_review = bool(reasons)
    return ValueProposal(
        text=text,
        candidates=tuple(pool),
        chosen=chosen,
        confidence=_confidence(chosen, cats),
        categories=categories,
        needs_review=needs_review,
        review_reasons=reasons,
        reason=_reason(chosen, pool, reasons, excluded),
        excluded=tuple(excluded),
    )


def _is_ambiguous_slang(span: str) -> bool:
    words = re.findall(r"[^\W\d_]+", fold(span))
    return bool(words) and words[-1] in _AMBIGUOUS_SLANG


def _confidence(chosen: Candidate | None, cats: set[str]) -> float:
    if "no_candidate" in cats:
        score = 0.25
    elif chosen is None:
        score = 0.3
    elif "bare_number" in cats:
        score = 0.55
    elif "number_words" in cats:
        score = 0.65
    elif "compound_amount" in cats:
        score = 0.6
    elif "slang" in cats:
        score = 0.7
    elif "decimal_unit" in cats or "compact_unit" in cats:
        score = 0.95
    else:
        score = 0.97
    if chosen is not None:
        if "multi_number" in cats:
            score = min(score, 0.6)
        if "unusual_punctuation" in cats:
            score = min(score, 0.5)
        if "unaccented" in cats:
            score -= 0.02
    return round(max(score, 0.0), 2)


def _reason(
    chosen: Candidate | None,
    pool: Sequence[Candidate],
    reasons: Sequence[str],
    excluded: Sequence[Exclusion],
) -> str:
    if not pool:
        base = "no money-like expression found"
    elif chosen is None:
        base = "several plausible amounts: " + ", ".join(repr(c.text) for c in pool)
    else:
        base = f"{chosen.kind} {chosen.text!r}"
    notes = [f"{x.text!r} is {x.reason}" for x in excluded if x.reason in _CONTEXT_REASONS]
    parts = [base] + notes
    if reasons:
        parts.append("review: " + ", ".join(reasons))
    return "; ".join(parts)


# ---------------------------------------------------------------------------------------------
# Surface patterns
# ---------------------------------------------------------------------------------------------

_SIG_NUM = re.compile(r"\d{1,3}(?:[.,]\d{3})+(?!\d)|\d+(?:[.,]\d+)?")
_SIG_WORDS = re.compile(
    r"^(?:(?:mot|hai|ba|bon|tu|nam|sau|bay|tam|chin|muoi|tram|chuc|le|linh|lam|nham)\s+)+"
    r"(?=trieu|nghin|ngan|ty|cu)"
)


def pattern_signature(span_text: str) -> str:
    """Surface pattern of a span with digits masked and accents stripped (case/spacing kept).

    ``50k`` -> ``<num>k``; ``50 K`` -> ``<num> K``; ``1tr5`` -> ``<num>tr<num>``;
    ``1.5tr`` -> ``<dec.>tr``; ``1,5tr`` -> ``<dec,>tr``; ``1.500.000`` -> ``<dot3>``;
    ``612,000`` -> ``<com3>``; ``1500000`` -> ``<long>``; ``2 triệu`` -> ``<num> trieu``;
    ``80.000đ`` -> ``<dot3>đ``.
    """

    def repl(m: re.Match[str]) -> str:
        t = m.group()
        if _SEP3_RE.fullmatch(t):
            return "<dot3>" if "." in t else "<com3>"
        if "." in t:
            return "<dec.>"
        if "," in t:
            return "<dec,>"
        return "<long>" if len(t) >= 5 else "<num>"

    sig = _SIG_NUM.sub(repl, fold_keep_case(span_text))
    sig = re.sub(r"(?<=[>\s])(?:dong|vnd|d)$", "đ", sig)
    sig = _SIG_WORDS.sub("<words> ", sig)
    return re.sub(r"\s+", " ", sig).strip()


# ---------------------------------------------------------------------------------------------
# Quet serialisation
# ---------------------------------------------------------------------------------------------


def to_quet_proposal(record_id: str, proposal: ValueProposal) -> dict[str, Any]:
    """Advisory Quet proposal (never a label): ``p`` in Quet accepts exactly this."""
    if proposal.chosen is not None:
        out: dict[str, Any] = {
            "id": record_id,
            "annotation_status": "complete",
            "type": "amount",
            "target": proposal.chosen.as_span(),
        }
    elif "multiple_money_candidates" in proposal.categories:
        shown = ", ".join(c.text for c in proposal.candidates)
        out = {
            "id": record_id,
            "annotation_status": "uncertain",
            "type": "amount",
            "target": None,
            "note": f"several amounts ({shown}); the transaction amount is unclear",
        }
    else:
        out = {
            "id": record_id,
            "annotation_status": "complete",
            "type": "no_amount",
            "target": None,
        }
    out["confidence"] = proposal.confidence
    out["reason"] = proposal.reason
    return out


def to_rule_label(record_id: str, proposal: ValueProposal) -> dict[str, Any]:
    """A rule-provenance label (only for ``proposal.auto_accept``); never a human label."""
    if not proposal.auto_accept or proposal.chosen is None:
        raise ValueError(f"{record_id}: proposal is not auto-acceptable")
    return {
        "id": record_id,
        "annotation_status": "complete",
        "type": "amount",
        "target": proposal.chosen.as_span(),
        "provenance": "rule",
        "confidence": proposal.confidence,
        "rule_version": RULE_VERSION,
    }


# ---------------------------------------------------------------------------------------------
# Config and validators
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ValueConfig:
    """The ``value:`` section of ``configs/annotation-v2.yaml``."""

    version: str
    types: tuple[str, ...]
    statuses: tuple[str, ...]
    null_target_types: frozenset[str]
    null_label_statuses: frozenset[str]
    note_required_statuses: frozenset[str]
    trainable_statuses: frozenset[str]
    provenances: tuple[str, ...]
    tags: tuple[str, ...]
    seed: str


def load_value_config(path: str | Path = DEFAULT_VALUE_CONFIG) -> ValueConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    v = raw["value"]
    return ValueConfig(
        version=str(raw["version"]),
        types=tuple(v["types"]),
        statuses=tuple(v["statuses"]),
        null_target_types=frozenset(v.get("null_target_types", ())),
        null_label_statuses=frozenset(v.get("null_label_statuses", ())),
        note_required_statuses=frozenset(v.get("note_required_statuses", ())),
        trainable_statuses=frozenset(v["trainable_statuses"]),
        provenances=tuple(v["provenances"]),
        tags=tuple(v["tags"]),
        seed=str(v["rules"]["seed"]),
    )


LABEL_FIELDS: tuple[str, ...] = ("id", "annotation_status", "type", "target")
LABEL_OPTIONAL: tuple[str, ...] = ("note",)
RULE_LABEL_EXTRA: tuple[str, ...] = ("provenance", "confidence", "rule_version")


def validate_span(span: Any, text: str | None, field: str = "target") -> list[str]:
    """Span mechanics, mirroring Quet: shape, code-point offsets, slice equality, no edge blanks."""
    if not isinstance(span, dict):
        return [f"{field}: expected an object or null"]
    problems = [f"{field}: missing field {k!r}" for k in SPAN_FIELDS if k not in span]
    problems += [f"{field}: unknown field {k!r}" for k in span if k not in SPAN_FIELDS]
    if problems:
        return problems
    value, start, end = span["text"], span["start"], span["end"]
    if not isinstance(value, str) or not value:
        problems.append(f"{field}.text: expected a non-empty string")
    elif value != value.strip():
        problems.append(f"{field}.text: has leading or trailing whitespace")
    elif value != unicodedata.normalize("NFC", value):
        problems.append(f"{field}.text: is not NFC-normalized")
    for name, number in (("start", start), ("end", end)):
        if not isinstance(number, int) or isinstance(number, bool) or number < 0:
            problems.append(f"{field}.{name}: expected a non-negative integer")
    if problems:
        return problems
    if start >= end:
        return [f"{field}: start {start} must be less than end {end}"]
    if text is None:
        return []
    if end > len(text):
        return [f"{field}: end {end} is beyond the text length {len(text)}"]
    if text[start:end] != value:
        return [f"{field}: text[{start}:{end}] is {text[start:end]!r}, not {value!r}"]
    return []


def span_shape_warnings(span_text: str) -> list[str]:
    """Soft checks of the span convention (reported, never blocking)."""
    warnings = []
    folded = fold(span_text)
    if not re.search(r"\d", span_text) and not _WORDNUM_RE.search(span_text.lower()):
        warnings.append("no digit or spelled-out number")
    if re.search(r"/\s?(thang|tuan|ngay|nam|lan|nguoi)", folded):
        warnings.append("includes a per-unit qualifier")
    if re.match(r"(tam|khoang|hon|gan|ngot|chung|~)", folded):
        warnings.append("includes an approximator")
    if re.search(r"(thang|quy|ky|tuan|ngay)\s?\d", folded):
        warnings.append("includes a month/period")
    return warnings


def validate_value_label(
    record: Any,
    text: str | None,
    config: ValueConfig,
    *,
    extra_fields: Sequence[str] = (),
) -> list[str]:
    """Contract violations of one value label (Quet label format); empty means valid."""
    if not isinstance(record, dict):
        return ["record: expected a JSON object"]
    problems = [f"missing field {k!r}" for k in LABEL_FIELDS if k not in record]
    allowed = set(LABEL_FIELDS) | set(LABEL_OPTIONAL) | set(extra_fields)
    problems += [f"unknown field {k!r}" for k in record if k not in allowed]
    rid = record.get("id")
    if "id" in record and (not isinstance(rid, str) or not rid):
        problems.append("id: expected a non-empty string")
    if "note" in record and not isinstance(record["note"], str):
        problems.append("note: expected a string")
    status = record.get("annotation_status")
    if "annotation_status" in record and status not in config.statuses:
        problems.append(f"annotation_status: {status!r} is not one of {list(config.statuses)}")
    note = record.get("note")
    if status in config.note_required_statuses and not (isinstance(note, str) and note.strip()):
        problems.append(f"note: required (non-empty) when annotation_status is {status!r}")
    if status in config.null_label_statuses:
        for f in ("type", "target"):
            if record.get(f) is not None:
                problems.append(f"{f}: must be null when annotation_status is {status!r}")
    kind = record.get("type")
    if kind is None:
        if status == "complete":
            problems.append("type: required when annotation_status is 'complete'")
    elif kind not in config.types:
        problems.append(f"type: {kind!r} is not one of {list(config.types)}")
    target = record.get("target")
    if target is not None:
        problems += validate_span(target, text, "target")
        if kind in config.null_target_types:
            problems.append(f"target: must be null for type {kind!r}")
    elif status == "complete" and kind is not None and kind not in config.null_target_types:
        problems.append(f"target: required for a complete {kind!r} label")
    if "provenance" in extra_fields and record.get("provenance") != "rule":
        problems.append("provenance: expected 'rule'")
    return problems


def validate_quet_proposal(proposal: Any, text: str | None, config: ValueConfig) -> list[str]:
    """Would Quet accept this proposal for ``text``? (Quet's own checks, not Gidi's extras.)"""
    if not isinstance(proposal, dict):
        return ["proposal: expected a JSON object"]
    problems = []
    if not isinstance(proposal.get("id"), str) or not proposal.get("id"):
        problems.append("id: expected a non-empty string")
    status = proposal.get("annotation_status")
    if not isinstance(status, str):
        problems.append("annotation_status: expected a string")
    elif status not in config.statuses:
        problems.append(f"annotation_status: {status!r} is not declared")
    kind = proposal.get("type")
    if kind is not None and kind not in config.types:
        problems.append(f"type: {kind!r} is not declared")
    if status == "complete" and kind is None:
        problems.append("type: required when complete")
    target = proposal.get("target")
    if target is not None:
        problems += validate_span(target, text, "target")
        if kind in config.null_target_types:
            problems.append(f"target: must be null for type {kind!r}")
    conf = proposal.get("confidence")
    if conf is not None and (
        isinstance(conf, bool) or not isinstance(conf, int | float) or not 0 <= conf <= 1
    ):
        problems.append("confidence: expected a number from 0 to 1")
    for f in ("note", "reason"):
        if f in proposal and not isinstance(proposal[f], str):
            problems.append(f"{f}: expected a string")
    if status in config.note_required_statuses and not str(proposal.get("note", "")).strip():
        problems.append("note: required for an uncertain proposal (Quet saves the proposal's note)")
    return problems


@dataclass
class LabelFileReport:
    path: Path
    labels: dict[str, dict[str, Any]]
    errors: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_label_file(
    path: str | Path,
    texts: Mapping[str, str],
    config: ValueConfig,
    *,
    extra_fields: Sequence[str] = (),
    require_ids_in_queue: bool = True,
) -> LabelFileReport:
    """Validate a value-label JSONL against the queue texts; also rejects duplicate ids."""
    import json

    path = Path(path)
    labels: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"{path}:{lineno}: invalid JSON: {e.msg}")
                continue
            rid = record.get("id") if isinstance(record, dict) else None
            text = texts.get(rid) if isinstance(rid, str) else None
            problems = validate_value_label(record, text, config, extra_fields=extra_fields)
            if isinstance(rid, str) and rid not in texts and require_ids_in_queue:
                problems.append(f"id {rid!r} is not in the queue")
            errors += [f"{path}:{lineno}: {p}" for p in problems]
            if isinstance(rid, str):
                if rid in labels:
                    errors.append(f"{path}:{lineno}: duplicate id {rid!r}")
                labels[rid] = record
    return LabelFileReport(path, labels, errors)


MERGED_VALUE_STATUSES: tuple[str, ...] = ("complete", "uncertain")


def validate_merged_record(record: Any, config: ValueConfig) -> list[str]:
    """Contract violations of a merged training record (v1 fields + value fields)."""
    if not isinstance(record, dict):
        return ["record: expected a JSON object"]
    problems = [
        f"missing field {k!r}"
        for k in ("id", "text", "type", "target", "value", "value_status", "value_provenance")
        if k not in record
    ]
    if problems:
        return problems
    text = record["text"]
    if not isinstance(text, str) or text != unicodedata.normalize("NFC", text):
        problems.append("text: expected an NFC string")
        text = text if isinstance(text, str) else None
    status = record["value_status"]
    if status not in MERGED_VALUE_STATUSES:
        problems.append(f"value_status: {status!r} is not one of {list(MERGED_VALUE_STATUSES)}")
    if record["value_provenance"] not in config.provenances:
        problems.append(
            f"value_provenance: {record['value_provenance']!r} is not one of "
            f"{list(config.provenances)}"
        )
    value, target = record["value"], record["target"]
    if value is not None:
        problems += validate_span(value, text, "value")
        if status != "complete":
            problems.append("value: must be null unless value_status is 'complete'")
        if (
            target is not None
            and not problems
            and value["start"] < target["end"]
            and target["start"] < value["end"]
        ):
            problems.append("value: overlaps the target span")
    return problems


def merged_value_fields(
    human: Mapping[str, Any] | None, rule: Mapping[str, Any] | None
) -> tuple[dict[str, Any] | None, str, str, str]:
    """``(value, value_status, value_provenance, source)`` for one record; human beats rule.

    ``source`` is one of ``human``, ``human-uncertain``, ``human-skipped``, ``rule`` or
    ``unlabelled`` (needs review but has no human label yet: masked).
    """
    if human is not None:
        status = human["annotation_status"]
        if status == "complete" and human["type"] == "amount":
            return dict(human["target"]), "complete", "human", "human"
        if status == "complete":
            return None, "complete", "human", "human"
        source = "human-uncertain" if status == "uncertain" else "human-skipped"
        return None, "uncertain", "human", source
    if rule is not None:
        return dict(rule["target"]), "complete", "rule", "rule"
    return None, "uncertain", "rule", "unlabelled"
