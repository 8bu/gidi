"""Load a fast tokenizer with offset mapping for any encoder candidate."""

from __future__ import annotations

from transformers import AutoTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast


def load_tokenizer(name: str) -> PreTrainedTokenizerBase:
    """Load the fast tokenizer for hub id or local directory ``name``.

    ``AutoTokenizer`` is tried first. BamiBERT ships only ``tokenizer.json`` with a class
    ``AutoTokenizer`` cannot resolve (``TypeError``), so that case falls back to
    ``PreTrainedTokenizerFast``. Tokenizers without offset mapping (the slow PhoBERT
    tokenizer) are rejected: span labels cannot be aligned without them.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(name)
    except TypeError:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(name)
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(
            f"{name!r} loaded a slow tokenizer ({type(tokenizer).__name__}) without offset "
            "mapping; span tagging needs a fast tokenizer"
        )
    return tokenizer
