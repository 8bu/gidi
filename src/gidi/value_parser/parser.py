"""Deterministic value-span parser: the monetary amount of a short Vietnamese note, as a span.

``parse_value(text)`` returns the substring of ``text`` that is the amount (``"1 triệu"``,
``"1tr5"``, ``"80.000đ"``, ``"70"``) with ``[start, end)`` offsets in code points of the caller's
string, or ``None`` when the note holds no amount. No number is ever derived from the span.

How it works (one linear pass, no regular expressions on user text, no randomness):

1. NFC-normalize (``gidi.inference.text.normalize_nfc`` keeps the offset map back to the
   caller's string) and build a *folded* shadow string of the same length: lowercase, accents
   removed, ``đ`` -> ``d``. Word lists match on the shadow, spans are cut from the NFC text.
2. Scan for number tokens. A digit token is a digit run with ``.``/``,`` separators between digit
   groups (``80.000``, ``1,5``). A word token is a numeral word run followed by a money unit
   (``bốn triệu``, ``nửa củ``). Each token becomes a ``Candidate`` (a money expression) or is
   dropped as a date, time, percentage, phone number, identifier, period or quantity.
3. A candidate is ``number [unit [half | digits]] [currency]``: ``1tr5``, ``1 triệu 2``,
   ``2 củ rưỡi``, ``250.000 VND``, ``15 K``.
4. One candidate is chosen by tier (explicit money unit or money-shaped number, then slang unit,
   then bare number) and by position inside the tier.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from gidi.inference.text import normalize_nfc
from gidi.value_parser import lexicon as lx

TIER_MONEY = 1  # explicit unit (k, tr, nghìn, ...), currency suffix, thousand-grouped, >= 5 digits
TIER_SLANG = 2  # củ, xị, chai, lít
TIER_BARE = 3  # a bare number that no context explains away

_MAX_PLAIN_DIGITS = 10  # a plain digit run longer than this is an account / id, not an amount
_BARE_PLAIN_MONEY_DIGITS = 5  # a plain digit run of at least this many digits is money-shaped


@dataclass(frozen=True)
class ValueSpan:
    """The amount: ``text == original[start:end]`` in code points of the caller's string."""

    text: str
    start: int
    end: int


@dataclass(frozen=True)
class Candidate:
    """A money expression in NFC coordinates."""

    start: int
    end: int
    tier: int
    kind: str  # unit | currency | grouped | plain | slang | word | bare


# --------------------------------------------------------------------------- folding


def _fold_char(char: str) -> str:
    """One lowercase accent-free character for one NFC character (length-preserving)."""
    if char in "đĐ":
        return "d"
    base = unicodedata.normalize("NFD", char)[0]
    lowered = base.lower()
    return lowered if len(lowered) == 1 else base


def fold(text: str) -> str:
    """Accent-folded lowercase shadow of ``text`` with the same length."""
    return "".join(_fold_char(c) for c in text)


def _is_digit(char: str) -> bool:
    return "0" <= char <= "9"


def _is_letter(char: str) -> bool:
    return char.isalpha()


# --------------------------------------------------------------------------- scanning helpers


def _word_end(shadow: str, pos: int) -> int:
    """End of the letter run starting at ``pos`` (``pos`` when there is none)."""
    end = pos
    while end < len(shadow) and _is_letter(shadow[end]):
        end += 1
    return end


def _prev_word(shadow: str, pos: int) -> tuple[int, int]:
    """``(start, end)`` of the word ending one space before ``pos``; ``(pos, pos)`` if none."""
    if pos < 2 or shadow[pos - 1] != " ":
        return pos, pos
    end = pos - 1
    start = end
    while start > 0 and _is_letter(shadow[start - 1]):
        start -= 1
    return (start, end) if start < end else (pos, pos)


def _next_word(shadow: str, pos: int) -> tuple[int, int]:
    """``(start, end)`` of the word one space after ``pos``; empty ``(pos, pos)`` if none."""
    if pos >= len(shadow) or shadow[pos] != " ":
        return pos, pos
    start = pos + 1
    end = _word_end(shadow, start)
    return (start, end) if end > start else (pos, pos)


def _digit_run(shadow: str, pos: int) -> int:
    end = pos
    while end < len(shadow) and _is_digit(shadow[end]):
        end += 1
    return end


def _number_end(shadow: str, pos: int) -> tuple[int, list[int]]:
    """End of the number starting at ``pos`` and the lengths of its separated digit groups.

    Groups are joined by ``.``/``,`` (``80.000``, ``1,5``) or, for at least two trailing
    three-digit groups, by single spaces (``1 000 000``).
    """
    end = _digit_run(shadow, pos)
    groups = [end - pos]
    while end + 1 < len(shadow) and shadow[end] in ".," and _is_digit(shadow[end + 1]):
        nxt = _digit_run(shadow, end + 1)
        groups.append(nxt - end - 1)
        end = nxt
    if len(groups) == 1 and groups[0] <= 3:
        cursor, spaced = end, 0
        while cursor + 1 < len(shadow) and shadow[cursor] == " " and _is_digit(shadow[cursor + 1]):
            nxt = _digit_run(shadow, cursor + 1)
            if nxt - cursor - 1 != 3:
                break
            spaced, cursor = spaced + 1, nxt
        if spaced >= 2:
            return cursor, [groups[0], *([3] * spaced)]
    return end, groups


def _is_grouped(groups: list[int]) -> bool:
    """Thousand-grouped: ``80.000``, ``1.250.000``, ``12,500,000`` (first group 1-3 digits)."""
    return len(groups) > 1 and 1 <= groups[0] <= 3 and all(g == 3 for g in groups[1:])


def _skip_identifier(shadow: str, pos: int) -> int:
    """End of an alphanumeric token starting at ``pos`` (digits glued to letters)."""
    end = pos
    while end < len(shadow) and (_is_letter(shadow[end]) or _is_digit(shadow[end])):
        end += 1
    return max(end, pos + 1)


def _phone_end(shadow: str, pos: int) -> int | None:
    """End of a phone-like number at ``pos`` (9-11 digits, leading 0 or 84), longest match.

    Digit groups of >= 3 digits may be joined by one space, ``.`` or ``-`` (``0912 345 678``).
    """
    groups: list[tuple[str, int]] = []
    end = pos
    while True:
        run = _digit_run(shadow, end)
        if groups and run - end < 3:
            break
        groups.append((shadow[end:run], run))
        end = run
        if sum(len(g) for g, _ in groups) > 11:
            break
        if end + 1 < len(shadow) and shadow[end] in " .-" and _is_digit(shadow[end + 1]):
            end += 1
            continue
        break
    best: int | None = None
    joined = ""
    for digits, group_end in groups:
        joined += digits
        if 9 <= len(joined) <= 11 and (
            joined.startswith("0") or (joined.startswith("84") and len(joined) >= 10)
        ):
            best = group_end
    return best


# --------------------------------------------------------------------------- context rules


def _digit_text(shadow: str, start: int, end: int) -> str:
    return shadow[start:end].replace(".", "").replace(",", "")


def _explained_before(text: str, shadow: str, start: int, end: int, groups: list[int]) -> bool:
    """Is the number at ``[start, end)`` a month / period / weekday / year by its left word?"""
    ws, we = _prev_word(shadow, start)
    if ws == we:
        return False
    word = shadow[ws:we]
    if word not in lx.PERIOD_BEFORE:
        return False
    if word in lx.NAME_LIKE and text[ws].isupper():
        return False  # "Lan 300": a person, not "lần"
    if len(groups) > 1:
        return False
    digits = end - start
    if word == "nam":
        accented = text[ws:we].lower() == "năm"
        year = digits == 4 and shadow[start : start + 2] in ("19", "20")
        return year or (accented and digits <= 2)
    if word == lx.WEEKDAY_WORD:
        return digits == 1 and shadow[start] in "2345678"
    return digits <= 2


def _quantity_after(shadow: str, end: int) -> bool:
    """Is the number followed by a counted-thing or time word (``2 tô``, ``3 vé``, ``5 tháng``)?"""
    ws, we = _next_word(shadow, end)
    return ws != we and shadow[ws:we] in lx.QUANTITY_AFTER


def _glued_suffix_blocks(shadow: str, end: int) -> bool:
    """Digits glued to a non-money letter run (``10kg``, ``5h``, ``2x``)."""
    return end < len(shadow) and _is_letter(shadow[end])


# --------------------------------------------------------------------------- units


def _is_currency(text: str, shadow: str, start: int, stop: int) -> bool:
    """Is ``shadow[start:stop]`` a currency word? ``dong`` must be ``đồng``, not ``đóng`` (pay)."""
    word = shadow[start:stop]
    if word not in lx.CURRENCY_WORDS:
        return False
    if word != "dong":
        return True
    if "\u1ed3" in text[start:stop].lower():  # ồ
        return True
    if text[start:stop].lower() != "dong":
        return False
    return all(not shadow[i].isalnum() for i in range(stop, len(shadow)))


def _read_unit(
    text: str, shadow: str, pos: int, number_is_big: bool
) -> tuple[str, int, int] | None:
    """The money unit right after a number: ``(word, start, end)`` or ``None``.

    A unit is glued (``50k``, ``2tr``) or after one space (``15 K``, ``5 củ``). Currency
    letters need the number to be grouped or long when spaced, so ``250 d`` stays a bare number.
    """
    glued = pos < len(shadow) and (_is_letter(shadow[pos]) or shadow[pos] in lx.CURRENCY_SIGNS)
    start = pos
    if not glued:
        ws, we = _next_word(shadow, pos)
        if ws == we:
            if (
                pos + 1 < len(shadow)
                and shadow[pos] == " "
                and shadow[pos + 1] in lx.CURRENCY_SIGNS
            ):
                return shadow[pos + 1], pos + 1, pos + 2
            return None
        start = ws
    if shadow[start] in lx.CURRENCY_SIGNS:
        return shadow[start], start, start + 1
    end = _word_end(shadow, start)
    word = shadow[start:end]
    is_money = (
        word in lx.THOUSAND_LETTER
        or word in lx.THOUSAND_WORDS
        or word in lx.MILLION
        or word in lx.BILLION
        or word in lx.SLANG
        or _is_currency(text, shadow, start, end)
    )
    if not is_money:
        return None
    if not glued and word == "d" and not number_is_big:
        return None
    return word, start, end


def _slang_is_quantity(text: str, shadow: str, end: int) -> bool:
    """Does a goods noun follow a slang unit (``5 lít xăng``, ``2 chai bia``)?"""
    ws, we = _next_word(shadow, end)
    if ws == we:
        return False
    word = shadow[ws:we]
    if word in lx.SLANG_QTY:
        return True
    return text[ws:we].lower() in lx.SLANG_QTY_ACCENTED


def _continuation(shadow: str, end: int, word: str) -> int:
    """Extend past a unit: half (``rưỡi``), glued/spaced digits (``1tr5``, ``1 triệu 2``)."""
    if word not in lx.CONTINUABLE:
        return end
    ws, we = _next_word(shadow, end)
    if ws != we and shadow[ws:we] in lx.HALF:
        return we
    glued = end < len(shadow) and _is_digit(shadow[end])
    spaced = (
        not glued and end + 1 < len(shadow) and shadow[end] == " " and _is_digit(shadow[end + 1])
    )
    if not glued and not spaced:
        return end
    if spaced and word == "tr":
        return end  # "5tr 20/10": a spaced digit after a bare "tr" is another token
    ds = end + (1 if spaced else 0)
    de = _digit_run(shadow, ds)
    size = de - ds
    if size > 3 or (spaced and size not in (1, 3)):
        return end
    if de < len(shadow) and (shadow[de] in "/:%" or _is_letter(shadow[de])):
        return end
    if de + 1 < len(shadow) and shadow[de] in ".," and _is_digit(shadow[de + 1]):
        return end
    if spaced and _quantity_after(shadow, de):
        return end
    return de


def _currency_suffix(text: str, shadow: str, end: int) -> int:
    """Extend past a currency marker: ``250.000đ``, ``1.200.000 vnd``, ``350.000 đồng``."""
    if end < len(shadow) and shadow[end] in lx.CURRENCY_SIGNS:
        return end + 1
    glued = end < len(shadow) and _is_letter(shadow[end])
    start = end if glued else _next_word(shadow, end)[0]
    if not glued and start == end:
        if end + 1 < len(shadow) and shadow[end] == " " and shadow[end + 1] in lx.CURRENCY_SIGNS:
            return end + 2
        return end
    stop = _word_end(shadow, start)
    return stop if _is_currency(text, shadow, start, stop) else end


# --------------------------------------------------------------------------- candidates


def _digit_candidate(text: str, shadow: str, pos: int) -> tuple[Candidate | None, int]:
    """Analyse the digit token at ``pos``: ``(candidate or None, scan resume position)``."""
    n = len(shadow)
    if pos > 0 and _is_letter(shadow[pos - 1]):
        return None, _skip_identifier(shadow, pos)  # t10, q4, x2, iphone15
    phone = _phone_end(shadow, pos)
    if phone is not None:
        return None, phone
    end, groups = _number_end(shadow, pos)
    # dates, times, fractions: 20/10, 9/12/2024, 10:30
    if end < n and shadow[end] in "/:" and end + 1 < n and _is_digit(shadow[end + 1]):
        stop = end + 1
        while stop < n and (_is_digit(shadow[stop]) or shadow[stop] in "/:."):
            stop += 1
        return None, stop
    if pos > 0 and shadow[pos - 1] == "/":
        return None, end
    # percentages: 20%, 5.5%/năm
    after = end + 1 if end < n and shadow[end] == " " else end
    if after < n and shadow[after] == "%":
        return None, after + 1

    digits = len(_digit_text(shadow, pos, end))
    grouped = _is_grouped(groups)
    big = grouped or digits >= _BARE_PLAIN_MONEY_DIGITS
    if len(groups) == 1 and digits > _MAX_PLAIN_DIGITS:
        return None, end

    unit = _read_unit(text, shadow, end, big)
    if unit is None:
        if _glued_suffix_blocks(shadow, end):
            return None, _skip_identifier(shadow, end)
        if big:
            kind = "grouped" if grouped else "plain"
            return Candidate(pos, end, TIER_MONEY, kind), end
        if _explained_before(text, shadow, pos, end, groups) or _quantity_after(shadow, end):
            return None, end
        return Candidate(pos, end, TIER_BARE, "bare"), end

    word, _, unit_end = unit
    if word in lx.SLANG:
        if _slang_is_quantity(text, shadow, unit_end):
            return None, unit_end
        stop = _currency_suffix(text, shadow, _continuation(shadow, unit_end, word))
        return Candidate(pos, stop, TIER_SLANG, "slang"), stop
    stop = _continuation(shadow, unit_end, word)
    stop = _currency_suffix(text, shadow, stop) if word not in lx.CURRENCY_WORDS else stop
    kind = "currency" if word in lx.CURRENCY_WORDS or word in lx.CURRENCY_SIGNS else "unit"
    return Candidate(pos, stop, TIER_MONEY, kind), stop


def _word_candidate(text: str, shadow: str, pos: int) -> tuple[Candidate | None, int]:
    """A numeral-word amount at ``pos`` (a word start): ``hai trăm nghìn``, ``nửa củ``."""
    n = len(shadow)
    end = _word_end(shadow, pos)
    word = shadow[pos:end]
    cursor = end
    if word in lx.HALF_PREFIX:
        pass
    elif word in lx.NUMBER_WORDS:
        while True:
            ws, we = _next_word(shadow, cursor)
            if ws != we and shadow[ws:we] in lx.NUMBER_WORDS:
                cursor = we
                continue
            break
    else:
        return None, end
    ws, we = _next_word(shadow, cursor)
    unit = shadow[ws:we] if ws != we else ""
    if unit not in lx.WORD_UNITS:
        return None, cursor  # no suffix of this numeral run can end in a unit either
    if unit in lx.SLANG and _slang_is_quantity(text, shadow, we):
        return None, we
    stop = _continuation(shadow, we, unit)
    stop = _currency_suffix(text, shadow, stop)
    tier = TIER_SLANG if unit in lx.SLANG else TIER_MONEY
    return Candidate(pos, min(stop, n), tier, "word"), stop


def find_candidates(text: str) -> list[Candidate]:
    """All money candidates of NFC ``text``, left to right."""
    shadow = fold(text)
    n = len(shadow)
    out: list[Candidate] = []
    pos = 0
    while pos < n:
        char = shadow[pos]
        if _is_digit(char):
            cand, resume = _digit_candidate(text, shadow, pos)
        elif _is_letter(char) and (
            pos == 0 or not (_is_letter(shadow[pos - 1]) or _is_digit(shadow[pos - 1]))
        ):
            cand, resume = _word_candidate(text, shadow, pos)
        else:
            cand, resume = None, pos + 1
        if cand is not None:
            out.append(cand)
        pos = max(resume, pos + 1)
    return out


def choose(candidates: list[Candidate]) -> Candidate | None:
    """The amount among ``candidates``: best tier; first of the tier, last for bare numbers.

    Several explicit amounts (``89k ship 15k``) are ambiguous by the annotation contract and the
    train labels hold no such complete note, so the tie rule is a prior, not a tuned rule: the
    leading amount is the main one and fees follow. Several bare numbers are mostly quantity then
    price (``3 100``), so the last one wins.
    """
    if not candidates:
        return None
    best = min(c.tier for c in candidates)
    tied = [c for c in candidates if c.tier == best]
    return tied[-1] if best == TIER_BARE else tied[0]


def parse_value(text: str) -> ValueSpan | None:
    """The monetary amount of ``text`` as a span of the caller's string, or ``None``."""
    normalized = normalize_nfc(text)
    chosen = choose(find_candidates(normalized.text))
    if chosen is None:
        return None
    start, end = normalized.span_to_original(chosen.start, chosen.end)
    return ValueSpan(text[start:end], start, end)
