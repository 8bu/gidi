"""Prune the byte-level BPE vocabulary of a trained Gidi model (compression-v1).

BamiBERT ships a byte-level BPE (``tokenizer.json``: vocab + ranked merges). Most of its 20,481
rows are never used by short Vietnamese finance notes, but each costs 768 parameters. This module
shrinks the vocabulary without touching anything else:

* :class:`BpeVocab` parses ``tokenizer.json`` (token strings, ranked merges, merge parents).
* :func:`closure` makes a keep-set *merge-closed*: every kept merged token keeps the parents of
  every merge that produces it, recursively, down to single bytes.
* :func:`prune_tokenizer_json` builds the pruned tokenizer: vocab renumbered contiguously in old
  id order (``<s>``=0, ``<pad>``=1, ``</s>``=2, ``<unk>``=3 keep their ids, ``<mask>`` becomes the
  last id), merges filtered to those whose two parts and result are kept (rank order preserved).
* :func:`prune_vocab` returns a NEW model whose embedding keeps exactly the kept rows.

Exactness of a closed keep-set. Take a word whose original segmentation uses only kept tokens.
The original BPE run applies a chain of merges; every token created along the way is either a
final piece (kept) or the part of a later merge that (transitively) builds a final piece, so
closure keeps it. Hence every merge applied has all parts and its result kept and survives the
filter, and since the filter only deletes merges and preserves relative rank, the lowest-rank
applicable pair at each step is the same pair: the pruned run is identical. Any other word loses
some merges and falls back to the smaller pieces still reachable, ultimately single bytes.
BamiBERT's 256-byte alphabet is incomplete (46 bytes such as control characters and 0xF2..0xFF
have no token, so they already map to ``<unk>`` before pruning); pruning never makes that worse.
"""

from __future__ import annotations

import copy
import hashlib
import json
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from gidi.modeling.model import GidiMultiTaskModel
from gidi.modeling.tokenization import load_tokenizer

VOCAB_MAP_FILE = "vocab_map.json"
WORD_EMBEDDINGS_KEY = "encoder.embeddings.word_embeddings.weight"
# Ids that survive pruning in place: ``<s>``, ``<pad>``, ``</s>``, ``<unk>``.
LEADING_SPECIAL_IDS = (0, 1, 2, 3)


@cache
def byte_to_char() -> dict[int, str]:
    """GPT-2 byte-level alphabet: every byte value mapped to one printable character."""
    printable = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    chars = printable[:]
    extra = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            chars.append(256 + extra)
            extra += 1
    return {b: chr(c) for b, c in zip(printable, chars, strict=True)}


@cache
def char_to_byte() -> dict[str, int]:
    return {c: b for b, c in byte_to_char().items()}


def token_bytes(token: str) -> bytes:
    """Raw bytes a byte-level token string stands for (specials are not byte-level)."""
    table = char_to_byte()
    try:
        return bytes(table[c] for c in token)
    except KeyError as exc:
        raise ValueError(f"{token!r} is not a byte-level token string") from exc


def decode_token(token: str) -> str:
    """Human-readable text of a token; partial UTF-8 sequences show as ``\\xNN`` escapes."""
    try:
        return token_bytes(token).decode("utf-8", errors="backslashreplace")
    except ValueError:
        return token


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------------------------
# BPE structure
# --------------------------------------------------------------------------------------------


def _merge_pair(entry: str | Sequence[str]) -> tuple[str, str]:
    if isinstance(entry, str):
        left, right = entry.split(" ")
        return left, right
    left, right = entry
    return left, right


@dataclass(frozen=True)
class BpeVocab:
    """Parsed byte-level BPE.

    ``tokens[i]`` is the string of old id ``i`` (added tokens such as ``<mask>`` included),
    ``merges[r]`` the ``(left_id, right_id, result_id)`` of rank ``r`` (lower = earlier = more
    frequent in the tokenizer's training corpus).
    """

    tokens: tuple[str, ...]
    merges: tuple[tuple[int, int, int], ...]
    special_ids: frozenset[int]
    base_ids: frozenset[int]

    @classmethod
    def from_json(cls, tok: dict[str, Any]) -> BpeVocab:
        model = tok["model"]
        if model["type"] != "BPE":
            raise ValueError(f"expected a BPE tokenizer, got {model['type']!r}")
        vocab: dict[str, int] = dict(model["vocab"])
        added = {int(a["id"]): a["content"] for a in tok.get("added_tokens", [])}
        specials = {int(a["id"]) for a in tok.get("added_tokens", []) if a.get("special")}
        size = max([*vocab.values(), *added]) + 1
        tokens: list[str | None] = [None] * size
        for token, i in vocab.items():
            tokens[i] = token
        for i, content in added.items():
            if tokens[i] not in (None, content):
                raise ValueError(f"id {i} is both {tokens[i]!r} and added token {content!r}")
            tokens[i] = content
        if any(t is None for t in tokens):
            raise ValueError("token ids are not contiguous")
        ids = {t: i for i, t in enumerate(tokens) if t is not None}
        merges = []
        for entry in model["merges"]:
            left, right = _merge_pair(entry)
            merges.append((ids[left], ids[right], ids[left + right]))
        table = char_to_byte()
        base = {i for i, t in enumerate(tokens) if i not in specials and len(t or "") == 1}
        if any(tokens[i] not in table for i in base):
            raise ValueError("single-character token outside the byte alphabet")
        return cls(
            tuple(t for t in tokens if t is not None),
            tuple(merges),
            frozenset(specials),
            frozenset(base),
        )

    @property
    def size(self) -> int:
        return len(self.tokens)

    @property
    def mask_id(self) -> int:
        return self.size - 1

    def merge_rank(self) -> dict[int, int]:
        """Lowest merge rank producing each merged token (base and special tokens absent)."""
        rank: dict[int, int] = {}
        for r, (_, _, result) in enumerate(self.merges):
            rank.setdefault(result, r)
        return rank

    def parents(self) -> dict[int, list[tuple[int, int]]]:
        out: dict[int, list[tuple[int, int]]] = {}
        for left, right, result in self.merges:
            out.setdefault(result, []).append((left, right))
        return out

    def mandatory_ids(self) -> set[int]:
        """Specials plus every single-byte base token that exists in the vocabulary."""
        return set(self.special_ids) | set(self.base_ids)

    def orphans(self) -> list[int]:
        """Vocabulary tokens that no merge produces and that are neither special nor a byte.

        BPE can never emit such a token for a multi-byte string, so it is dead weight.
        """
        produced = {result for _, _, result in self.merges}
        return [
            i
            for i in range(self.size)
            if i not in produced and i not in self.special_ids and i not in self.base_ids
        ]

    def missing_bytes(self) -> list[int]:
        """Byte values with no base token (they tokenize to ``<unk>`` before and after pruning)."""
        present = {self.tokens[i] for i in self.base_ids}
        return [b for b, c in byte_to_char().items() if c not in present]


def closure(bpe: BpeVocab, ids: Iterable[int]) -> set[int]:
    """Smallest superset of ``ids`` closed under merge parents (of every producing merge)."""
    parents = bpe.parents()
    keep = set(ids)
    stack = list(keep)
    while stack:
        token = stack.pop()
        for left, right in parents.get(token, ()):
            for parent in (left, right):
                if parent not in keep:
                    keep.add(parent)
                    stack.append(parent)
    return keep


def is_closed(bpe: BpeVocab, ids: Iterable[int]) -> bool:
    keep = set(ids)
    return closure(bpe, keep) == keep


# --------------------------------------------------------------------------------------------
# Token script categories (decoded text)
# --------------------------------------------------------------------------------------------

# Latin letters Vietnamese text can use: Latin-1 Supplement letters through Latin Extended-B,
# Latin Extended Additional (all Vietnamese precomposed letters) and combining diacritics.
_LATIN_RANGES = ((0x00C0, 0x024F), (0x1E00, 0x1EFF), (0x0300, 0x036F))
# Typographic punctuation and currency that appear in Vietnamese finance text.
_EXTRA_ALLOWED = frozenset("\u2013\u2014\u2018\u2019\u201c\u201d\u2026\u2022\u00b7\u00b0\u20ab")


def _is_latin_range(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _LATIN_RANGES)


def _char_allowed(ch: str) -> bool:
    """Vietnamese-compatible character: ASCII printable, Vietnamese Latin range, typography."""
    cp = ord(ch)
    return 0x20 <= cp <= 0x7E or _is_latin_range(ch) or ch in _EXTRA_ALLOWED


def _script(ch: str) -> str:
    try:
        return unicodedata.name(ch).split(" ")[0]
    except ValueError:
        return "UNKNOWN"


@cache
def _partial_sets() -> tuple[frozenset[bytes], frozenset[bytes]]:
    """Proper prefixes / continuation-byte suffixes of every allowed non-ASCII character."""
    prefixes: set[bytes] = set()
    suffixes: set[bytes] = set()
    for lo, hi in _LATIN_RANGES:
        for cp in range(lo, hi + 1):
            raw = chr(cp).encode("utf-8")
            for k in range(1, len(raw)):
                prefixes.add(raw[:k])
                suffixes.add(raw[k:])
    for ch in _EXTRA_ALLOWED:
        raw = ch.encode("utf-8")
        for k in range(1, len(raw)):
            prefixes.add(raw[:k])
            suffixes.add(raw[k:])
    return frozenset(prefixes), frozenset(suffixes)


def _utf8_length(lead: int) -> int:
    if lead < 0x80:
        return 1
    if 0xC2 <= lead <= 0xDF:
        return 2
    if 0xE0 <= lead <= 0xEF:
        return 3
    if 0xF0 <= lead <= 0xF4:
        return 4
    return 0  # continuation or invalid lead byte


def partial_is_vietnamese(raw: bytes) -> bool:
    """Whether non-decodable ``raw`` is built only from pieces of Vietnamese-compatible chars.

    Accepted shape: optional continuation-byte tail of an allowed character, then complete
    allowed characters, then optional incomplete head of an allowed character.
    """
    prefixes, suffixes = _partial_sets()
    n = len(raw)
    i = 0
    while i < n and 0x80 <= raw[i] <= 0xBF:
        i += 1
    if i and raw[:i] not in suffixes:
        return False
    while i < n:
        size = _utf8_length(raw[i])
        if size == 0:
            return False
        if i + size > n:
            return raw[i:] in prefixes
        try:
            ch = raw[i : i + size].decode("utf-8")
        except UnicodeDecodeError:
            return False
        if not _char_allowed(ch):
            return False
        i += size
    return True


CATEGORIES = (
    "special",
    "base_byte",
    "partial_byte",
    "ascii_letters",
    "vietnamese_latin",
    "digits_numeric",
    "ascii_letters_with_punct",
    "ascii_punct_symbols",
    "other_symbols",
    "non_latin_script",
)


def token_category(token: str, *, special: bool = False, base: bool = False) -> str:
    """Category of one token string (see :data:`CATEGORIES`; priority is the listed order)."""
    if special:
        return "special"
    if base:
        return "base_byte"
    raw = token_bytes(token)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return "partial_byte"
    non_ascii_letters = [ch for ch in text if ch.isalpha() and not ch.isascii()]
    if any(_script(ch) != "LATIN" for ch in non_ascii_letters):
        return "non_latin_script"
    allowed = all(_char_allowed(ch) for ch in text)
    if any(ch.isascii() and ch.isdigit() for ch in text):
        return "digits_numeric" if allowed else "other_symbols"
    if any(_is_latin_range(ch) for ch in text):
        return "vietnamese_latin" if allowed else "other_symbols"
    if not text.isascii():
        return "other_symbols"
    if any(ch.isalpha() for ch in text):
        return "ascii_letters" if text.lstrip(" ").isalpha() else "ascii_letters_with_punct"
    return "ascii_punct_symbols"


def is_vietnamese_compatible(token: str) -> bool:
    """Decoded text uses only Vietnamese-compatible characters (partial bytes: see above)."""
    raw = token_bytes(token)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return partial_is_vietnamese(raw)
    return all(_char_allowed(ch) for ch in text)


def vietnamese_compatible_ids(bpe: BpeVocab) -> set[int]:
    """Non-special ids whose decoded text is Vietnamese-compatible."""
    return {
        i
        for i, tok in enumerate(bpe.tokens)
        if i not in bpe.special_ids and is_vietnamese_compatible(tok)
    }


# --------------------------------------------------------------------------------------------
# Pruned tokenizer and vocabulary spec
# --------------------------------------------------------------------------------------------


def _validate_kept(kept_old_ids: Sequence[int], old_size: int) -> list[int]:
    kept = [int(i) for i in kept_old_ids]
    if any(b <= a for a, b in zip(kept, kept[1:], strict=False)):
        raise ValueError("kept_old_ids must be strictly ascending")
    if not kept or kept[0] < 0 or kept[-1] >= old_size:
        raise ValueError(f"kept_old_ids must lie in [0, {old_size})")
    missing = [i for i in (*LEADING_SPECIAL_IDS, old_size - 1) if i not in set(kept)]
    if missing:
        raise ValueError(f"kept_old_ids must contain the special ids {missing}")
    return kept


def prune_tokenizer_json(tok: dict[str, Any], kept_old_ids: Sequence[int]) -> dict[str, Any]:
    """Pruned copy of a byte-level BPE ``tokenizer.json`` dict (input is not mutated)."""
    bpe = BpeVocab.from_json(tok)
    kept = _validate_kept(kept_old_ids, bpe.size)
    if not is_closed(bpe, kept):
        raise ValueError("kept_old_ids is not closed under merge parents; use closure() first")
    old_to_new = {old: new for new, old in enumerate(kept)}
    out = copy.deepcopy(tok)
    model_vocab = tok["model"]["vocab"]
    new_vocab = {t: old_to_new[i] for t, i in model_vocab.items() if i in old_to_new}
    if sorted(new_vocab.values()) != list(range(len(new_vocab))):
        raise ValueError("pruned vocab ids are not contiguous (non-vocab ids must be last)")
    out["model"]["vocab"] = dict(sorted(new_vocab.items(), key=lambda kv: kv[1]))
    out["model"]["merges"] = [
        entry
        for entry in tok["model"]["merges"]
        if all(part in new_vocab for part in (*_merge_pair(entry), "".join(_merge_pair(entry))))
    ]
    for added in out.get("added_tokens", []):
        added["id"] = old_to_new[int(added["id"])]
    post = out.get("post_processor") or {}
    for special in post.get("special_tokens", {}).values():
        special["ids"] = [old_to_new[int(i)] for i in special["ids"]]
    return out


def write_vocab_spec(
    out_dir: str | Path,
    source_tokenizer_dir: str | Path,
    kept_old_ids: Sequence[int],
    policy: str,
) -> Path:
    """Write the vocabulary spec directory (tokenizer files + ``vocab_map.json``).

    ``source_tokenizer_dir`` holds the baseline ``tokenizer.json`` and ``tokenizer_config.json``.
    """
    src = Path(source_tokenizer_dir)
    tok = json.loads((src / "tokenizer.json").read_text(encoding="utf-8"))
    pruned = prune_tokenizer_json(tok, kept_old_ids)
    kept = [int(i) for i in kept_old_ids]
    bpe = BpeVocab.from_json(tok)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tokenizer.json").write_text(
        json.dumps(pruned, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    config = json.loads((src / "tokenizer_config.json").read_text(encoding="utf-8"))
    (out / "tokenizer_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    specials = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "sep_token": "</s>",
        "pad_token": "<pad>",
        "cls_token": "<s>",
        "mask_token": "<mask>",
    }
    (out / "special_tokens_map.json").write_text(
        json.dumps(specials, indent=2) + "\n", encoding="utf-8"
    )
    new_id = {t: i for i, t in enumerate(bpe.tokens[i] for i in kept)}
    vocab_map = {
        "policy": policy,
        "source_tokenizer_sha256": sha256_file(src / "tokenizer.json"),
        "kept_old_ids": kept,
        "n_kept": len(kept),
        "removed_old_ids_count": bpe.size - len(kept),
        "special_tokens": {t: new_id[t] for t in ("<s>", "<pad>", "</s>", "<unk>", "<mask>")},
    }
    (out / VOCAB_MAP_FILE).write_text(
        json.dumps(vocab_map, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return out


def load_vocab_spec(spec_dir: str | Path) -> tuple[PreTrainedTokenizerBase, list[int]]:
    """``(tokenizer, kept_old_ids)`` from a spec directory written by :func:`write_vocab_spec`."""
    spec = Path(spec_dir)
    vocab_map = json.loads((spec / VOCAB_MAP_FILE).read_text(encoding="utf-8"))
    kept = [int(i) for i in vocab_map["kept_old_ids"]]
    tokenizer = load_tokenizer(str(spec))
    if len(tokenizer) != len(kept):
        raise ValueError(f"{spec}: tokenizer has {len(tokenizer)} ids but {len(kept)} are kept")
    return tokenizer, kept


def remap_ids(input_ids: torch.Tensor, kept_old_ids: Sequence[int]) -> torch.Tensor:
    """Map old-vocabulary ids to pruned ids; raises ``ValueError`` on any removed id."""
    kept = [int(i) for i in kept_old_ids]
    lookup = torch.full((kept[-1] + 1,), -1, dtype=torch.long)
    lookup[torch.tensor(kept, dtype=torch.long)] = torch.arange(len(kept), dtype=torch.long)
    flat = input_ids.long()
    bad = (flat < 0) | (flat >= lookup.numel())
    mapped = lookup[flat.clamp(0, lookup.numel() - 1)]
    bad |= mapped < 0
    if bool(bad.any()):
        removed = sorted({int(i) for i in flat[bad].tolist()})
        raise ValueError(f"ids not in the pruned vocabulary: {removed[:10]}")
    return mapped.to(input_ids.dtype)


# --------------------------------------------------------------------------------------------
# Model transform
# --------------------------------------------------------------------------------------------


def prune_vocab(model: GidiMultiTaskModel, kept_old_ids: Sequence[int]) -> GidiMultiTaskModel:
    """New model whose word-embedding row ``i`` is old row ``kept_old_ids[i]``.

    Everything else (positions, layers, heads) is copied exactly; ``model`` is not mutated.
    ``kept_old_ids`` is strictly ascending and holds ids 0..3 and the last old id (``<mask>``).
    """
    old_rows = int(model.encoder.config.vocab_size)
    kept = _validate_kept(kept_old_ids, old_rows)
    config = copy.deepcopy(model.encoder.config)
    config.vocab_size = len(kept)
    new = GidiMultiTaskModel.from_config(
        config,
        num_types=model.num_types,
        num_tags=model.num_tags,
        dropout=model.dropout_p,
        encoder_name=model.encoder_name,
    )
    state = dict(model.state_dict())
    weight = state[WORD_EMBEDDINGS_KEY]
    state[WORD_EMBEDDINGS_KEY] = weight[torch.tensor(kept, dtype=torch.long)].clone()
    new.load_state_dict(state, strict=True)
    new.to(weight.dtype)
    new.train(model.training)
    return new
