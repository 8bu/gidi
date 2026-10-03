"""Bundle tokenizer: the ``tokenizers`` library alone, configured like the training path.

Training used ``transformers`` with ``truncation=True, max_length=32`` and no padding per note.
That is ``Tokenizer.enable_truncation(max_length=32)`` (strategy ``longest_first``, right side,
stride 0; the template post-processor reserves room for ``<s>`` and ``</s>``), so tokens beyond
the limit are dropped from the end. The tokenizer has no normalizer: NFC is applied by the
caller, offsets are code points of the string given to ``encode``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from tokenizers import Tokenizer

_LONE_SURROGATE = re.compile("[\ud800-\udfff]")


@dataclass(frozen=True)
class TokenizedText:
    ids: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]
    special_tokens_mask: tuple[int, ...]
    truncated: bool


class BundleTokenizer:
    def __init__(self, tokenizer_file: str | Path, max_length: int):
        self.max_length = max_length
        self._tokenizer = Tokenizer.from_file(str(tokenizer_file))
        self._tokenizer.no_padding()
        self._tokenizer.enable_truncation(max_length=max_length)

    def encode(self, text: str) -> TokenizedText:
        # A lone surrogate cannot cross the str -> Rust boundary. U+FFFD is also one code point,
        # so offsets stay aligned with the caller's string.
        text = _LONE_SURROGATE.sub("\ufffd", text)
        enc = self._tokenizer.encode(text)
        return TokenizedText(
            ids=tuple(enc.ids),
            offsets=tuple((int(s), int(e)) for s, e in enc.offsets),
            special_tokens_mask=tuple(enc.special_tokens_mask),
            truncated=len(enc.overflowing) > 0,
        )
