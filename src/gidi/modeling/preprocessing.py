"""Turn (text, character span) pairs into token tensors with BIO tags, and back.

Offsets in the annotation contract are Unicode code points into the NFC text. Fast tokenizers
report per-token character offsets into the string they were given, so feeding them the NFC text
keeps both in the same coordinate system. Byte-level BPE tokenizers (BamiBERT) fold the leading
space into the token (``" Thảo"`` covers ``(3, 8)``) while SentencePiece tokenizers (MiniLM) do
not; both are handled by trimming whitespace off every token's covered text before comparing it
with the span.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

# Class ids follow the order in configs/annotation-v1.yaml (a test pins this).
TYPES: tuple[str, ...] = (
    "expense",
    "income",
    "borrow",
    "lend",
    "repayment_in",
    "repayment_out",
    "transfer",
    "refund",
)
TAGS: tuple[str, ...] = ("O", "B-TARGET", "I-TARGET")
TAG_O, TAG_B, TAG_I = 0, 1, 2
IGNORE_INDEX = -100
DEFAULT_MAX_LENGTH = 32

Offsets = list[tuple[int, int]]


@dataclass(frozen=True)
class Encoded:
    """Batch of encoded examples, padded to the longest example (right padding).

    ``offsets[i]`` has one ``(start, end)`` per position, aligned with the tensors; special and
    padding positions carry ``(0, 0)``. The offsets are the tokenizer's raw ones (a byte-level
    token may include its leading space); ``spans_from_tags`` trims them.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    tag_labels: torch.Tensor
    offsets: list[Offsets]
    truncated: list[bool]  # the text needed more than max_length tokens
    span_truncated: list[bool]  # ... and the target span lost tokens because of it
    boundary_mismatch: list[bool]  # tokens did not align with the span (only if strict=False)

    @property
    def n_truncated(self) -> int:
        return sum(self.truncated)

    @property
    def n_span_truncated(self) -> int:
        return sum(self.span_truncated)

    @property
    def n_boundary_mismatch(self) -> int:
        return sum(self.boundary_mismatch)

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])


def nfc_text(text: str) -> str:
    """Return ``text`` unchanged, asserting it is already NFC (span offsets assume NFC)."""
    normalized = unicodedata.normalize("NFC", text)
    if normalized != text:
        raise ValueError(f"text is not NFC-normalized: {text!r} -> {normalized!r}")
    return text


def trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Shrink ``[start, end)`` so it neither starts nor ends on whitespace (may become empty)."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _tags_for_example(
    text: str,
    offsets: Sequence[tuple[int, int]],
    real: Sequence[bool],
    target: Mapping[str, Any] | None,
) -> tuple[list[int], tuple[int, int] | None]:
    """BIO ids (``IGNORE_INDEX`` off real tokens) and the reconstructed span, if any."""
    tags = [TAG_O if r else IGNORE_INDEX for r in real]
    if target is None:
        return tags, None
    t_start, t_end = int(target["start"]), int(target["end"])
    inside: list[tuple[int, int]] = []
    seen_first = False
    for i, (is_real, (s, e)) in enumerate(zip(real, offsets, strict=True)):
        if not is_real:
            continue
        s, e = trim_span(text, s, e)
        if s < e and s < t_end and e > t_start:
            tags[i] = TAG_I if seen_first else TAG_B
            seen_first = True
            inside.append((s, e))
    if not inside:
        return tags, None
    return tags, (min(s for s, _ in inside), max(e for _, e in inside))


def encode(
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
    targets: Sequence[Mapping[str, Any] | None],
    max_length: int = DEFAULT_MAX_LENGTH,
    *,
    strict: bool = True,
) -> Encoded:
    """Tokenize ``texts`` and derive BIO tag labels from the character spans in ``targets``.

    A token is inside the span when its offsets, after trimming whitespace from the text they
    cover, overlap ``[start, end)``; the first such token is ``B-TARGET`` and the rest
    ``I-TARGET``. ``None`` targets tag every real token ``O``. Special and padding tokens get
    ``IGNORE_INDEX``.

    The span rebuilt from the tags (min start to max end of the tagged tokens) must equal the
    target's offsets and text. A span that lost tokens to truncation at ``max_length`` is
    exempt: surviving tokens keep their tags and the example is counted in ``span_truncated``.
    Any other mismatch (a token straddling a span edge) raises ``ValueError`` when ``strict``,
    otherwise it is counted in ``boundary_mismatch``.
    """
    if len(texts) != len(targets):
        raise ValueError(f"{len(texts)} texts but {len(targets)} targets")
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("a fast tokenizer with offset mapping is required")
    texts = [nfc_text(t) for t in texts]
    for text, target in zip(texts, targets, strict=True):
        if target is not None and text[target["start"] : target["end"]] != target["text"]:
            raise ValueError(f"target offsets do not reproduce its text in {text!r}: {target!r}")

    enc = tokenizer(
        list(texts),
        padding="longest",
        truncation=True,
        max_length=max_length,
        padding_side="right",
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        return_tensors="pt",
    )
    full_lengths = [len(ids) for ids in tokenizer(list(texts), truncation=False)["input_ids"]]

    input_ids: torch.Tensor = enc["input_ids"]
    attention_mask: torch.Tensor = enc["attention_mask"]
    all_offsets: list[Offsets] = []
    tag_rows: list[list[int]] = []
    truncated: list[bool] = []
    span_truncated: list[bool] = []
    mismatch: list[bool] = []

    for i, (text, target) in enumerate(zip(texts, targets, strict=True)):
        special = enc["special_tokens_mask"][i].tolist()
        mask = attention_mask[i].tolist()
        real = [bool(m) and not s for m, s in zip(mask, special, strict=True)]
        raw = [(int(s), int(e)) for s, e in enc["offset_mapping"][i].tolist()]
        offsets = [o if r else (0, 0) for o, r in zip(raw, real, strict=True)]
        tags, rebuilt = _tags_for_example(text, offsets, real, target)

        was_truncated = full_lengths[i] > max_length
        lost = False
        bad = False
        if target is not None:
            last_end = max(
                (trim_span(text, s, e)[1] for (s, e), r in zip(offsets, real, strict=True) if r),
                default=0,
            )
            lost = was_truncated and target["end"] > last_end
            expected = (int(target["start"]), int(target["end"]))
            if not lost and (rebuilt != expected or text[slice(*rebuilt)] != target["text"]):
                bad = True
                if strict:
                    raise ValueError(
                        f"tokens do not align with target {target['text']!r} "
                        f"{expected} in {text!r}: rebuilt span {rebuilt}"
                    )
        all_offsets.append(offsets)
        tag_rows.append(tags)
        truncated.append(was_truncated)
        span_truncated.append(lost)
        mismatch.append(bad)

    return Encoded(
        input_ids=input_ids,
        attention_mask=attention_mask,
        tag_labels=torch.tensor(tag_rows, dtype=torch.long),
        offsets=all_offsets,
        truncated=truncated,
        span_truncated=span_truncated,
        boundary_mismatch=mismatch,
    )


def spans_from_tags(
    offsets: Sequence[tuple[int, int]], tag_ids: Sequence[int], text: str
) -> dict[str, Any] | None:
    """Decode the first predicted span: ``{"text", "start", "end"}`` or ``None``.

    A span starts at the first ``B-TARGET`` (or an ``I-TARGET`` that follows a non-span token)
    and continues over consecutive ``I-TARGET`` tokens; a later ``B-TARGET`` or ``O`` ends it.
    Its bounds are the min start and max end of the member tokens after trimming whitespace, so
    byte-level offsets with a leading space decode to the same span as SentencePiece ones.
    Tokens whose trimmed offsets are empty (special, padding, lone ``▁``) are skipped.
    """
    start = end = None
    for (s, e), tag in zip(offsets, tag_ids, strict=False):
        s, e = trim_span(text, int(s), int(e))
        if s >= e:
            continue
        tag = int(tag)
        if tag == TAG_B:
            if start is not None:
                break
            start, end = s, e
        elif tag == TAG_I:
            if start is None:
                start, end = s, e
            else:
                start, end = min(start, s), max(end, e)
        elif start is not None:
            break
    if start is None:
        return None
    return {"text": text[start:end], "start": start, "end": end}
