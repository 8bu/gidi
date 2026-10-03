"""Measure how a pretrained tokenizer handles short Vietnamese finance notes.

The audit answers practical questions before an encoder is chosen: how many tokens a note
costs, how badly words fragment, how much Vietnamese text collapses into ``[UNK]``, whether
text survives an encode/decode round trip, and how expensive amount expressions
(``45k``, ``45.000đ``, ``1tr5``, ``200 nghìn``, ...) are.

Everything here is pure Python and deterministic: no downloads, no training, no randomness.
The tokenizer is only ever used through the Hugging Face ``PreTrainedTokenizerBase`` interface
(``__call__``, ``convert_ids_to_tokens``, ``decode``), so any pretrained tokenizer works.
"""

from __future__ import annotations

import dataclasses
import math
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gidi.corpus.jsonl import iter_jsonl

# Matches the amount expressions Gidi notes actually use:
#   45k, 50K, 15tr, 1tr5, 1,5tr, 2.5tr, 500đ, 45.000đ, 1.500.000vnđ, 200 nghìn/ngàn, 2 triệu,
#   2 củ (slang for million).
# The lookaround keeps matches from starting or ending inside a longer word.
AMOUNT_PATTERN = re.compile(
    r"""
    (?<![\w,.])
    (?:
        \d{1,3}(?:[.,]\d{3})+(?:\s*(?:đ|₫|vnđ|vnd))?      # 45.000đ, 1.500.000
      | \d+(?:[.,]\d+)?\s*(?:k|tr|m)\d*                    # 45k, 50K, 15tr, 1tr5, 1,5tr
      | \d+(?:[.,]\d+)?\s*(?:nghìn|ngàn|triệu|tỷ|tỉ|củ)    # 200 nghìn, 2 triệu, 2 củ
      | \d+(?:[.,]\d+)?\s*(?:đ|₫|vnđ|vnd)                  # 500đ, 500 vnđ
    )
    (?![\w])
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Shorthand and slang words from the corpus spec, matched case-insensitively as whole words
# (surrounding punctuation stripped). Their mean token cost is the slang fragmentation metric.
SHORTHAND_WORDS = frozenset(
    {
        "ck", "cf", "vs", "ko", "k", "dc", "đc", "mn", "nh", "ib", "nt", "stk", "tk", "thg",
        "ok", "tip", "ship", "bill", "vcb", "tcb", "mb", "momo", "zalopay", "shopeepay",
        "grab", "be", "shopee", "shope", "refund", "refun",
    }
)  # fmt: skip

# Fixed finance/slang forms whose exact tokenization is reported per tokenizer: amounts,
# units, shorthand, unaccented phrases, misspelled brands, and English loanwords.
DOMAIN_PROBES = (
    "1tr5",
    "2tr",
    "2 củ",
    "5 lít",
    "45k",
    "45.000đ",
    "ck",
    "thg",
    "tien nha",
    "an trua",
    "shope",
    "refund",
    "refun",
    "grab",
    "khach tra not",
)


@dataclass(frozen=True)
class TokenizerAuditReport:
    """Result of :func:`audit_tokenizer`. ``*_rate`` fields are fractions, ``pct_*`` percents."""

    name: str
    vocab_size: int
    n_texts: int
    n_words: int
    mean_tokens_per_text: float
    p50_tokens_per_text: float
    p95_tokens_per_text: float
    max_tokens_per_text: int
    fertility: float
    pct_words_split: float
    unk_token: str | None
    unk_rate: float
    roundtrip_exact_match_rate: float
    roundtrip_failures: tuple[dict[str, str], ...]
    unk_characters: tuple[str, ...]
    n_amounts: int
    mean_amount_tokens: float
    max_amount_tokens: int
    n_shorthand: int
    mean_shorthand_tokens: float
    pct_shorthand_split: float
    probes: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view (tuples become lists, field order preserved)."""
        out: dict[str, Any] = {}
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            out[field.name] = list(value) if isinstance(value, tuple) else value
        return out


def load_texts(paths: Iterable[str | Path]) -> list[str]:
    """Read the ``text`` field of every record in the given JSONL files, in file order."""
    texts: list[str] = []
    for path in paths:
        path = Path(path)
        for index, record in enumerate(iter_jsonl(path), start=1):
            text = record.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path}: record {index} has no usable 'text' field")
            texts.append(text)
    return texts


def audit_tokenizer(
    tokenizer: Any,
    texts: Sequence[str],
    *,
    max_examples: int = 20,
    probes: Sequence[str] = DOMAIN_PROBES,
) -> TokenizerAuditReport:
    """Audit ``tokenizer`` on ``texts``.

    ``tokenizer`` is any Hugging Face ``PreTrainedTokenizerBase``. ``max_examples`` caps the
    number of round-trip failure examples kept. Word-level metrics tokenize each
    whitespace-separated word in isolation, so they are independent of the tokenizer's own
    pre-tokenization.
    """
    texts = list(texts)
    words = [word for text in texts for word in text.split()]
    unk_token = getattr(tokenizer, "unk_token", None)
    if unk_token is not None:
        unk_token = str(unk_token)

    token_counts: list[int] = []
    n_unk = 0
    n_tokens = 0
    n_split_words = 0
    roundtrip_matches = 0
    failures: list[dict[str, str]] = []
    amount_token_counts: list[int] = []
    shorthand_token_counts: list[int] = []

    for raw_text in texts:
        text = unicodedata.normalize("NFC", raw_text)
        ids = _encode_ids(tokenizer, text)
        tokens = list(tokenizer.convert_ids_to_tokens(ids))
        token_counts.append(len(ids))
        n_tokens += len(tokens)
        if unk_token is not None:
            n_unk += sum(1 for token in tokens if token == unk_token)

        for word in text.split():
            n_word_tokens = len(_encode_ids(tokenizer, word))
            if n_word_tokens > 1:
                n_split_words += 1
            bare = word.strip(".,!?:;()\"'").lower()
            if bare in SHORTHAND_WORDS:
                shorthand_token_counts.append(len(_encode_ids(tokenizer, bare)))

        decoded = unicodedata.normalize("NFC", _decode(tokenizer, ids).strip())
        if decoded == text:
            roundtrip_matches += 1
        elif len(failures) < max_examples:
            failures.append({"text": text, "roundtrip": decoded})

        for match in AMOUNT_PATTERN.finditer(text):
            amount_token_counts.append(len(_encode_ids(tokenizer, match.group(0).strip())))

    sorted_counts = sorted(token_counts)
    n_words = len(words)
    return TokenizerAuditReport(
        name=_tokenizer_name(tokenizer),
        vocab_size=int(tokenizer.vocab_size),
        n_texts=len(texts),
        n_words=n_words,
        mean_tokens_per_text=_mean(token_counts),
        p50_tokens_per_text=_percentile(sorted_counts, 50.0),
        p95_tokens_per_text=_percentile(sorted_counts, 95.0),
        max_tokens_per_text=max(token_counts, default=0),
        fertility=(n_tokens / n_words) if n_words else 0.0,
        pct_words_split=(100.0 * n_split_words / n_words) if n_words else 0.0,
        unk_token=unk_token,
        unk_rate=(n_unk / n_tokens) if (unk_token is not None and n_tokens) else 0.0,
        roundtrip_exact_match_rate=(roundtrip_matches / len(texts)) if texts else 0.0,
        roundtrip_failures=tuple(failures),
        unk_characters=tuple(sorted(_unk_characters(tokenizer, texts, unk_token))),
        n_amounts=len(amount_token_counts),
        mean_amount_tokens=_mean(amount_token_counts),
        max_amount_tokens=max(amount_token_counts, default=0),
        n_shorthand=len(shorthand_token_counts),
        mean_shorthand_tokens=_mean(shorthand_token_counts),
        pct_shorthand_split=(
            100.0
            * sum(1 for count in shorthand_token_counts if count > 1)
            / len(shorthand_token_counts)
            if shorthand_token_counts
            else 0.0
        ),
        probes=tuple(probe_tokenizer(tokenizer, probes, unk_token)),
    )


def probe_tokenizer(
    tokenizer: Any, probes: Sequence[str], unk_token: str | None
) -> list[dict[str, Any]]:
    """Token sequence, token count, and UNK presence for each probe string.

    ``tokens`` are raw vocabulary entries; ``pieces`` decodes each token on its own, which
    makes byte-level BPE entries (``Ġcá»§``) readable (`` củ``).
    """
    results = []
    for probe in probes:
        text = unicodedata.normalize("NFC", probe)
        ids = _encode_ids(tokenizer, text)
        tokens = [str(token) for token in tokenizer.convert_ids_to_tokens(ids)]
        results.append(
            {
                "text": text,
                "tokens": tokens,
                "pieces": [_decode(tokenizer, [token_id]) for token_id in ids],
                "n_tokens": len(tokens),
                "has_unk": unk_token is not None and unk_token in tokens,
            }
        )
    return results


def _tokenizer_name(tokenizer: Any) -> str:
    name = getattr(tokenizer, "name_or_path", "") or ""
    return str(name) or type(tokenizer).__name__


def _encode_ids(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def _decode(tokenizer: Any, ids: Sequence[int]) -> str:
    return tokenizer.decode(list(ids), skip_special_tokens=True, clean_up_tokenization_spaces=False)


def _unk_characters(tokenizer: Any, texts: Sequence[str], unk_token: str | None) -> set[str]:
    """Characters of ``texts`` that encode entirely as ``unk_token`` (diacritic coverage)."""
    if unk_token is None:
        return set()
    unk_id = getattr(tokenizer, "unk_token_id", None)
    characters = {char for text in texts for char in text if not char.isspace()}
    missing: set[str] = set()
    for char in characters:
        ids = _encode_ids(tokenizer, char)
        if not ids:
            continue
        if unk_id is not None:
            is_unk = all(token_id == unk_id for token_id in ids)
        else:
            is_unk = all(token == unk_token for token in tokenizer.convert_ids_to_tokens(ids))
        if is_unk:
            missing.add(char)
    return missing


def _mean(values: Sequence[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    """Linearly interpolated percentile (numpy's default method) of a sorted sequence."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (len(sorted_values) - 1) * percentile / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(sorted_values[low])
    return float(sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (rank - low))
