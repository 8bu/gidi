#!/usr/bin/env python
"""Build the gidi-en student: pretrained roberta-base, 4 layers, pruned vocab, pos32, FFN K2048.

Usage:
    uv run python scripts/build_en_student.py \
        --out models/gidi-en-backbone/student-4x768-roberta-pos32-vocab8000-ffn2048 \
        [--train-notes datasets/annotation-en-v1/splits/train.jsonl] \
        [--calib-notes <same train notes>] [--max-rows 9000] [--ffn-k 2048]

The English equivalent of the Vietnamese compression-v1..v3 recipe, reusing its library code:

* Student: ``student-4x768`` initialised from roberta-base layers 2/5/8/11 (``layer_map``); fresh
  seeded heads (the weights are NOT trained here).
* Position table truncated to ``max_length + pad + 1`` rows (``truncate_positions``).
* Vocabulary (``prune_vocab``): specials + the 256 byte tokens, plus the closure of (a) every
  token used by the ``--train-notes`` (gidi-en TRAIN split only) and (b) every English-compatible
  token whose BPE merge rank is below a rank limit (merge rank = pretraining-frequency proxy,
  as policy B-rank-N in Vietnamese; ``--variants`` also counts space/case variants, which the
  vocab-policy probe measured as worse). The limit is the largest one
  keeping at most ``--max-rows`` rows (about 0.77 MB of int8 per 1000 rows). No dev/test/probe
  note is read.
* FFN (``prune_ffn``): the ``--ffn-k`` neurons per layer with the highest ``combined`` score
  (mean |activation| x ||W2 column||) on ``--calib-notes`` (use the TRAIN notes).

Writes a ``save_checkpoint`` directory (loadable by scripts/export_onnx.py), the vocab spec and
``build_report.json`` into ``--out``. With no ``--train-notes`` the vocabulary is rank-only and the
calibration defaults to the tokenizer-audit sample (feasibility runs only).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from gidi.compression.ffn import neuron_scores, prune_ffn, select_neurons
from gidi.compression.positions import truncate_positions
from gidi.compression.vocab import (
    BpeVocab,
    closure,
    decode_token,
    load_vocab_spec,
    prune_vocab,
    token_bytes,
    write_vocab_spec,
)
from gidi.corpus.jsonl import read_jsonl
from gidi.distillation.student import build_student, copy_pretrained_encoder
from gidi.modeling.checkpoint import save_checkpoint
from gidi.modeling.model import GidiMultiTaskModel
from gidi.modeling.tokenization import load_tokenizer

SOURCE = "roberta-base"
MAX_LENGTH = 32
VI_SPLIT = r" ?[^(\s|[.,!?…。，、।۔،])]+"
DEFAULT_CALIB = Path("experiments/gidi-en-backbone/audit-notes.jsonl")
# Typographic punctuation and currency signs that occur in English finance notes.
EXTRA_ALLOWED = frozenset(
    "\u2013\u2014\u2018\u2019\u201c\u201d\u2026\u2022\u00b7\u00b0\u00a3\u20ac"
)


def english_compatible_ids(bpe: BpeVocab) -> set[int]:
    """Non-special ids whose decoded text is printable ASCII or an allowed typographic char.

    Tokens that are partial UTF-8 sequences are dropped (the byte tokens that remain mandatory
    still spell any character, just in more pieces).
    """
    ok: set[int] = set()
    for i, token in enumerate(bpe.tokens):
        if i in bpe.special_ids or i in bpe.base_ids:
            continue
        try:
            text = token_bytes(token).decode("utf-8")
        except UnicodeDecodeError:
            continue
        if all(0x20 <= ord(c) <= 0x7E or c in EXTRA_ALLOWED for c in text):
            ok.add(i)
    return ok


def _forms(token: str) -> set[str]:
    """Space/no-space and lower/Capitalised variants of a byte-level word token."""
    core = token[1:] if token.startswith("Ġ") else token
    out = {core, "Ġ" + core}
    if core.isalpha():
        out |= {core.lower(), "Ġ" + core.lower(), core.capitalize(), "Ġ" + core.capitalize()}
    return out


def _is_amount_piece(token: str) -> bool:
    core = token[1:] if token.startswith("Ġ") else token
    return core.isdigit() and (len(core) <= 2 or (len(core) == 3 and core.endswith("00")))


def keep_set_at(
    bpe: BpeVocab, train_ids: set[int], rank_limit: int, variants: bool = False
) -> list[int]:
    """Keep-set for one rank limit.

    A token is kept when its merge rank is below ``rank_limit``; with ``variants`` the best rank of
    any space/case variant counts (experimental: measured worse than plain rank at equal rows,
    see experiments/gidi-en-backbone/vocab_policy_probe.json).
    Amount pieces (1-2 digit numbers and 100..900) are always kept. Everything is closed under
    merge parents.
    """
    rank = bpe.merge_rank()
    ids = {t: i for i, t in enumerate(bpe.tokens)}
    compat = english_compatible_ids(bpe)
    chosen = set()
    for i in compat:
        token = bpe.tokens[i]
        best = min(
            (rank[ids[f]] for f in (_forms(token) if variants else {token}) if ids.get(f) in rank),
            default=None,
        )
        if best is not None and best < rank_limit or _is_amount_piece(token):
            chosen.add(i)
    return sorted(closure(bpe, bpe.mandatory_ids() | train_ids | chosen))


def keep_set(
    bpe: BpeVocab, train_ids: set[int], max_rows: int, variants: bool = False
) -> tuple[list[int], int]:
    """Largest rank limit whose keep-set has at most ``max_rows`` rows (binary search).

    Returns ``(kept_ids, rank_limit)``. The train-note tokens are always kept, so ``max_rows``
    must exceed their closure.
    """
    lo, hi = 0, len(bpe.merges)
    if len(keep_set_at(bpe, train_ids, 0, variants)) > max_rows:
        raise ValueError(f"mandatory + train-note tokens alone exceed {max_rows} rows")
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(keep_set_at(bpe, train_ids, mid, variants)) <= max_rows:
            lo = mid
        else:
            hi = mid - 1
    return keep_set_at(bpe, train_ids, lo, variants), lo


def read_texts(path: Path) -> list[str]:
    return [r["text"] for r in read_jsonl(path)]


def write_source_tokenizer(hub_dir: Path, out_dir: Path) -> Path:
    """roberta-base ``tokenizer.json`` rewritten into the format ``gidi.compression.vocab`` reads.

    Only the container changes (``model.type`` added, ``RobertaProcessing`` replaced by the
    equivalent ``TemplateProcessing`` ``<s> A </s>``, GPT-2 regex swapped for the Vietnamese
    bundle's Split pre-tokenizer); vocab and merges are untouched.
    """
    tok = json.loads((hub_dir / "tokenizer.json").read_text(encoding="utf-8"))
    tok["model"]["type"] = "BPE"
    # Same pre-tokenizer as the Vietnamese bundle (Split + byte-level without regex), so the web
    # runtime tokenizer needs no change; ids are identical to roberta's GPT-2 regex on 763 notes
    # (experiments/gidi-en-backbone/verify.json).
    tok["pre_tokenizer"] = {
        "type": "Sequence",
        "pretokenizers": [
            {
                "type": "Split",
                "pattern": {"Regex": VI_SPLIT},
                "behavior": "Isolated",
                "invert": False,
            },
            {
                "type": "ByteLevel",
                "add_prefix_space": False,
                "trim_offsets": True,
                "use_regex": False,
            },
        ],
    }
    tok["model"]["unk_token"] = "<unk>"
    tok["post_processor"] = {
        "type": "TemplateProcessing",
        "single": [
            {"SpecialToken": {"id": "<s>", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}},
            {"SpecialToken": {"id": "</s>", "type_id": 0}},
        ],
        "pair": [
            {"SpecialToken": {"id": "<s>", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}},
            {"SpecialToken": {"id": "</s>", "type_id": 0}},
            {"SpecialToken": {"id": "</s>", "type_id": 0}},
            {"Sequence": {"id": "B", "type_id": 0}},
            {"SpecialToken": {"id": "</s>", "type_id": 0}},
        ],
        "special_tokens": {
            "<s>": {"id": "<s>", "ids": [0], "tokens": ["<s>"]},
            "</s>": {"id": "</s>", "ids": [2], "tokens": ["</s>"]},
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tokenizer.json").write_text(json.dumps(tok, ensure_ascii=False), encoding="utf-8")
    config = {
        "backend": "tokenizers",
        "bos_token": "<s>",
        "cls_token": "<s>",
        "eos_token": "</s>",
        "mask_token": "<mask>",
        "pad_token": "<pad>",
        "sep_token": "</s>",
        "unk_token": "<unk>",
        "model_max_length": 512,
        "tokenizer_class": "TokenizersBackend",
    }
    (out_dir / "tokenizer_config.json").write_text(json.dumps(config, indent=2) + "\n")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-notes", type=Path, default=None)
    ap.add_argument("--calib-notes", type=Path, default=None)
    ap.add_argument("--max-rows", type=int, default=9000)
    ap.add_argument("--variants", action="store_true")
    ap.add_argument("--ffn-k", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)

    hub_dir = Path(snapshot_download(SOURCE, allow_patterns=["tokenizer*", "config.json"]))
    src_dir = write_source_tokenizer(hub_dir, args.out / "source-tokenizer")
    tok_json = json.loads((src_dir / "tokenizer.json").read_text(encoding="utf-8"))
    bpe = BpeVocab.from_json(tok_json)
    source_tok = load_tokenizer(SOURCE)

    train_ids: set[int] = set()
    if args.train_notes is not None:
        for text in read_texts(args.train_notes):
            train_ids.update(source_tok(text, add_special_tokens=False)["input_ids"])
    kept, rank_limit = keep_set(bpe, train_ids, args.max_rows, args.variants)
    policy = f"B-rank-{rank_limit}-max{args.max_rows}" + ("-variants" if args.variants else "")
    policy += "+train" if train_ids else ""
    spec_dir = write_vocab_spec(args.out / "vocab", src_dir, kept, policy)
    tokenizer, _ = load_vocab_spec(spec_dir)

    model = build_student(
        args.seed, student="student-4x768", tokenizer_name=SOURCE, max_length=MAX_LENGTH
    )
    init = copy_pretrained_encoder(model.encoder, GidiMultiTaskModel.from_encoder(SOURCE).encoder)
    model = truncate_positions(model, MAX_LENGTH)
    model = prune_vocab(model, kept)

    calib_path = args.calib_notes or args.train_notes or DEFAULT_CALIB
    enc = tokenizer(
        read_texts(calib_path),
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    scores = neuron_scores(model, enc["input_ids"], enc["attention_mask"])
    keep_ffn = [select_neurons(layer["combined"], args.ffn_k) for layer in scores]
    model = prune_ffn(model, keep_ffn)

    save_checkpoint(
        model,
        tokenizer,
        args.out,
        {
            "encoder": "student-4x768-roberta-base-pos32-vocab-ffn",
            "max_length": MAX_LENGTH,
            "trained": False,
            "note": "random heads; architecture/size feasibility only",
        },
    )
    (args.out / "ffn_map.json").write_text(
        json.dumps({"criterion": "combined", "k": args.ffn_k, "keep": keep_ffn}) + "\n"
    )
    cfg = model.encoder.config
    report = {
        "source": SOURCE,
        "layer_map": init["layer_map"],
        "vocab_policy": policy,
        "vocab_rows": len(kept),
        "rank_limit": rank_limit,
        "train_notes": str(args.train_notes) if args.train_notes else None,
        "train_token_ids": len(train_ids),
        "calibration_notes": str(calib_path),
        "positions": cfg.max_position_embeddings,
        "intermediate": cfg.intermediate_size,
        "params": model.param_count(),
        "sample_kept_tokens": [decode_token(bpe.tokens[i]) for i in kept[256:296]],
    }
    (args.out / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "sample_kept_tokens"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
