"""Deterministic en-US/UK/IN/SG amount scan: labeler A of the gidi-en value labels (silver).

Scope: ``datasets/annotation-en-v1`` value labelling only. It is not the runtime scanner of the
playground (``playground/src/app/amount.ts``), which stays the thing the learned head is compared
with. Span convention (``docs/annotation-en-v1.md``, "Amount formats"): the amount exactly as
written, currency symbol / prefix code / money word included (``$20``, ``20 bucks``, ``s$30``,
``rs 500``, ``20$``, ``3 grand``, ``twenty quid``), nothing normalised.

:func:`scan` lists every amount-shaped span with a tier; :func:`choose` is labeler A's pick;
:func:`loose_candidates` is the superset shown to labeler B (Clef) so a miss of the strict scan
can still be found.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PREFIX = (
    r"(?:(?:us|s|a|c|nz|hk)?\$|£|€|₹|¥|₱|"
    r"(?:rs|inr|usd|sgd|gbp|eur|aud|cad|rm|php|myr|thb|aed)\.?\s?)"
)
_NUM = r"(?:\d{1,2}(?:,\d{2})+,\d{3}(?:\.\d+)?|\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
_SCALE_GLUED = r"(?:k|m|mm|b|bn|cr|lac|lacs)"
_SCALE_SPACED = r"(?:grand|thousand|million|mil|lakhs?|crores?|billion)"
_MONEY_ALPHA = (
    r"bucks?|bux|dollars?|usd|sgd|gbp|eur|inr|quid|pounds?|euros?|rupees?|rs|cents?|pence"
)
_MONEY_WORD = rf"(?:{_MONEY_ALPHA}|(?:us|s|a|c|nz|hk)\$|\$|€|£)"
_AMOUNT = re.compile(
    rf"(?<![\w.,$£€₹¥₱/:#@-])(?P<pre>{_PREFIX})?(?P<num>{_NUM})"
    rf"(?:(?P<sg>{_SCALE_GLUED})|\s?(?P<ss>{_SCALE_SPACED}))?"
    rf"(?:\s?(?P<suf>{_MONEY_WORD}))?(?![\w%]|[/:]\d|\.\d)",
    re.IGNORECASE,
)
_UNITS = (
    "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
    "sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety"
)
_BIG = "hundred|thousand|million|grand"
_WORD_RUN = rf"(?:(?:{_UNITS}|{_BIG})(?:[\s-]+(?:and\s+)?(?:{_UNITS}|{_BIG}))*)"
_WORD_AMOUNT = re.compile(
    rf"(?<![\w-])(?P<a>a\s+)?(?P<run>{_WORD_RUN}|fiver|tenner)"
    rf"(?:\s+(?P<suf>{_MONEY_ALPHA}))?(?![\w-])",
    re.IGNORECASE,
)
_SMALL = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine"}
_YEAR = re.compile(r"^(?:19|20)\d\d$")
_DIGIT_TOKEN = re.compile(r"\S*\d\S*")
_RETIREMENT = re.compile(r"401k|403b", re.IGNORECASE)
# Numbers that are part of a product or index name, not a sum of money.
_NAMED_BEFORE = re.compile(r"(?:s&p|nifty|trading|nasdaq|dow|ftse|sti|msci)\s?$", re.IGNORECASE)
# A number followed by a duration or rate word is not an amount (``due in 4 weeks``).
_DURATION_AFTER = re.compile(
    r"\s?(?:weeks?|wks?|days?|months?|mos?|years?|yrs?|hours?|hrs?|mins?|minutes?|%|x)\b",
    re.IGNORECASE,
)
_ORDINAL_AFTER = re.compile(r"\s+(?:first|second|third|\w+th)\b", re.IGNORECASE)
_CLOCK_BEFORE = re.compile(r"\b(?:at|by|before|around)\s$", re.IGNORECASE)
_SLANG_PREPOSITION = {"2", "4"}  # ``4`` = for, ``2`` = to between words
_EDGE = "()[]{}<>\"'.,;:!?*_"


@dataclass(frozen=True)
class Match:
    start: int
    end: int
    text: str
    tier: int  # 3 currency-marked, 2 scaled, 1 plain number, 0 year-like / weak word amount
    kind: str  # "digit" | "word" | "loose"


def scan(text: str) -> list[Match]:
    """Every strict amount-shaped span of ``text`` in reading order."""
    found: list[Match] = []
    for m in _AMOUNT.finditer(text):
        if _RETIREMENT.match(text, m.start()):
            continue  # a 401k plan is an account name, not an amount
        if _NAMED_BEFORE.search(text, 0, m.start("num")) or _DURATION_AFTER.match(text, m.end()):
            continue
        marked = bool(m.group("pre") or m.group("suf"))
        scaled = bool(m.group("sg") or m.group("ss"))
        plain = m.group("num")
        if marked:
            tier = 3
        elif scaled:
            tier = 2
        elif _YEAR.match(plain) or (plain in _SLANG_PREPOSITION and m.group(0) == plain):
            tier = 0
        else:
            tier = 1
        found.append(Match(m.start(), m.end(), m.group(0), tier, "digit"))
    for m in _WORD_AMOUNT.finditer(text):
        words = [w for w in re.split(r"[\s-]+", m.group("run").lower()) if w != "and"]
        article_is_numeral = m.group("a") and words[0] in ("hundred", "thousand", "grand")
        start = m.start("a") if article_is_numeral else m.start("run")
        if any(f.start < m.end() and start < f.end for f in found):
            continue
        if _ORDINAL_AFTER.match(text, m.end()) or _CLOCK_BEFORE.search(text, 0, m.start()):
            continue
        marked = bool(m.group("suf"))
        weak = len(words) == 1 and words[0] in _SMALL and not marked
        tier = 3 if marked else 0 if weak else 1
        found.append(Match(start, m.end(), text[start : m.end()], tier, "word"))
    found.sort(key=lambda f: (f.start, f.end))
    return found


def loose_candidates(text: str) -> list[Match]:
    """:func:`scan` plus every whitespace token that carries a digit (edge punctuation trimmed)."""
    found = scan(text)
    for m in _DIGIT_TOKEN.finditer(text):
        start, end = m.start(), m.end()
        while start < end and text[start] in _EDGE:
            start += 1
        while end > start and text[end - 1] in _EDGE:
            end -= 1
        if start >= end or any(f.start <= start and end <= f.end for f in found):
            continue
        found.append(Match(start, end, text[start:end], 0, "loose"))
    found.sort(key=lambda f: (f.start, f.end))
    return found


def choose(matches: list[Match]) -> Match | None:
    """Labeler A's pick: first currency-marked / scaled amount, else the last plain number.

    Notes usually end with the amount (``venmo'd mom 50``) and lead with a quantity
    (``2 tacos 15``) when they hold several bare numbers; a marked amount beats a bare one.
    Year-like numbers and weak number words (``one``) are used only when nothing else exists.
    """
    strong = [m for m in matches if m.kind != "loose"]
    for tier, last in ((3, False), (2, False), (1, True), (0, True)):
        tiered = [m for m in strong if m.tier == tier]
        if tiered:
            return tiered[-1] if last else tiered[0]
    return None


_CORE_STRIP = re.compile(
    r"(?:us|s|a|c|nz|hk)?\$|[£€₹¥₱]|"
    r"(?<![a-z])(?:bucks?|bux|dollars?|usd|sgd|gbp|eur|inr|quid|pounds?|euros?|rupees?|rs|aed|"
    r"php|myr|thb|rm|cents?|pence)(?![a-z])\.?",
    re.IGNORECASE,
)


def amount_core(text: str) -> str:
    """The amount of a value span without currency symbols, codes and money words, lowercase and
    without whitespace (``$20`` / ``20 bucks`` / ``rs 500`` -> ``20`` / ``20`` / ``500``).

    Used only to compare span *content* with the playground scanner, which returns the number
    (and its scale) without the currency marker, unlike the annotation convention.
    """
    return re.sub(r"\s+", "", _CORE_STRIP.sub("", text.lower()))
