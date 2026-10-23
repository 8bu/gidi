#!/usr/bin/env python
"""Build the init checkpoint of one bake-off candidate (pretrained layers, pruned, random heads).

    uv run python experiments/gidi-en-backbone-bakeoff/build_candidate.py \
        --candidate distilroberta --out models/gidi-en-bakeoff/init/distilroberta

The widths are the fixed rule of ``protocol.json`` (``width_rule``); nothing here reads an
evaluation set. Vocabulary selection and FFN calibration use the gidi-en TRAIN split only.
Same order of operations as ``scripts/build_en_student.py``: copy the pretrained layers
(``copy_pretrained_encoder`` with the evenly spaced ``layer_map``), truncate the position table,
prune the vocabulary, prune the FFN.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, BertTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import build_en_student as roberta_recipe  # noqa: E402

from gidi.compression.ffn import neuron_scores, prune_ffn, select_neurons  # noqa: E402
from gidi.compression.positions import truncate_positions  # noqa: E402
from gidi.compression.vocab import (  # noqa: E402
    VOCAB_MAP_FILE,
    BpeVocab,
    load_vocab_spec,
    prune_vocab,
    write_vocab_spec,
)
from gidi.distillation.student import copy_pretrained_encoder  # noqa: E402
from gidi.modeling.checkpoint import save_checkpoint  # noqa: E402
from gidi.modeling.model import GidiMultiTaskModel  # noqa: E402
from gidi.training.train import set_seed  # noqa: E402

TRAIN_NOTES = Path("datasets/annotation-en-v1/splits/train.jsonl")
AUDIT_NOTES = Path("experiments/gidi-en-backbone/audit-notes.jsonl")
MAX_LENGTH = 32
SPECIALS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")

# kind: bpe | wordpiece; layers: None = keep every source layer; rows: vocabulary row target
CANDIDATES = {
    "distilroberta": {
        "source": "distilbert/distilroberta-base",
        "kind": "bpe",
        "layers": None,
        "ffn_k": 832,
        "rows": 9000,
    },
    "electra": {
        "source": "google/electra-base-discriminator",
        "kind": "wordpiece",
        "layers": 4,
        "ffn_k": 2048,
        "rows": 9000,
    },
    "minilm": {
        "source": "microsoft/MiniLM-L12-H384-uncased",
        "kind": "wordpiece",
        "layers": None,
        "ffn_k": None,
        "rows": 20500,
    },
}


def wordpiece_compatible(token: str) -> bool:
    core = token[2:] if token.startswith("##") else token
    return bool(core) and all(
        0x21 <= ord(c) <= 0x7E or c in roberta_recipe.EXTRA_ALLOWED for c in core
    )


def wordpiece_keep_set(
    tokens: list[str], train_ids: set[int], rows: int
) -> tuple[list[int], dict[str, int]]:
    """Old ids to keep (ascending), by the ``vocabulary`` rule of protocol.json."""
    index = {t: i for i, t in enumerate(tokens)}
    mandatory = {index[t] for t in SPECIALS}
    alphabet = {
        i
        for i, t in enumerate(tokens)
        if wordpiece_compatible(t) and len(t[2:] if t.startswith("##") else t) == 1
    }
    kept = mandatory | alphabet | train_ids
    if len(kept) > rows:
        raise SystemExit(f"mandatory + train tokens need {len(kept)} rows, more than {rows}")
    stats = {
        "specials": len(mandatory),
        "alphabet": len(alphabet),
        "train_tokens": len(train_ids),
        "closed_before_fill": len(kept),
    }
    for i, t in enumerate(tokens):
        if len(kept) >= rows:
            break
        if i not in kept and not t.startswith("[unused") and wordpiece_compatible(t):
            kept.add(i)
    stats["fill_max_old_id"] = max(kept)
    return sorted(kept), stats


def write_wordpiece_tokenizer(
    hub_dir: Path, tokens: list[str], kept: list[int], out_dir: Path
) -> BertTokenizerFast:
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = BertTokenizerFast(
        vocab={tokens[old]: new for new, old in enumerate(kept)},
        do_lower_case=True,
        tokenize_chinese_chars=True,
        model_max_length=512,
    )
    tok.save_pretrained(out_dir)
    (out_dir / VOCAB_MAP_FILE).write_text(
        json.dumps(
            {
                "kept_old_ids": kept,
                "n_kept": len(kept),
                "source_vocab_sha256": hashlib.sha256(
                    (hub_dir / "vocab.txt").read_bytes()
                ).hexdigest(),
            }
        )
        + "\n"
    )
    return tok


def prune_word_rows(model: GidiMultiTaskModel, kept: list[int]) -> GidiMultiTaskModel:
    """New model whose word-embedding row i is old row kept[i] (everything else copied)."""
    import copy

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
    key = "encoder.embeddings.word_embeddings.weight"
    state[key] = state[key][torch.tensor(kept, dtype=torch.long)].clone()
    new.load_state_dict(state, strict=True)
    return new


def agreement(full_tok, pruned_tok, texts: list[str]) -> dict:
    """Notes tokenized identically by the full and the pruned vocabulary, and the [UNK] count."""
    same = unk = pieces_full = pieces_pruned = 0
    for t in texts:
        a = full_tok.convert_ids_to_tokens(full_tok(t)["input_ids"])
        b = pruned_tok.convert_ids_to_tokens(pruned_tok(t)["input_ids"])
        same += a == b
        unk += b.count("[UNK]")
        pieces_full += len(a)
        pieces_pruned += len(b)
    return {
        "notes": len(texts),
        "identical_pieces": same,
        "unk_tokens": unk,
        "pieces_per_note_full": pieces_full / len(texts),
        "pieces_per_note_pruned": pieces_pruned / len(texts),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--candidate", choices=sorted(CANDIDATES), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    spec = CANDIDATES[args.candidate]
    source = spec["source"]
    train_texts = roberta_recipe.read_texts(TRAIN_NOTES)
    audit_texts = roberta_recipe.read_texts(AUDIT_NOTES)

    base = AutoConfig.from_pretrained(source)
    n_source = int(base.num_hidden_layers)
    base.num_hidden_layers = spec["layers"] or n_source
    set_seed(args.seed)
    model = GidiMultiTaskModel.from_config(base, encoder_name=args.candidate)
    init = copy_pretrained_encoder(model.encoder, GidiMultiTaskModel.from_encoder(source).encoder)
    model = truncate_positions(model, MAX_LENGTH)

    if spec["kind"] == "bpe":
        # distilroberta-base ships the roberta-base byte-level BPE: same keep-set as the control
        hub = Path(snapshot_download("roberta-base", allow_patterns=["tokenizer*", "config.json"]))
        distil_hub = Path(snapshot_download(source, allow_patterns=["tokenizer.json"]))
        a = json.loads((hub / "tokenizer.json").read_text())["model"]
        b = json.loads((distil_hub / "tokenizer.json").read_text())["model"]
        if a["vocab"] != b["vocab"] or a["merges"] != b["merges"]:
            raise SystemExit("distilroberta-base BPE differs from roberta-base")
        src_dir = roberta_recipe.write_source_tokenizer(hub, args.out / "source-tokenizer")
        bpe = BpeVocab.from_json(json.loads((src_dir / "tokenizer.json").read_text()))
        source_tok = roberta_recipe.load_tokenizer("roberta-base")
        train_ids: set[int] = set()
        for text in train_texts:
            train_ids.update(source_tok(text, add_special_tokens=False)["input_ids"])
        kept, rank_limit = roberta_recipe.keep_set(bpe, train_ids, spec["rows"])
        policy = f"B-rank-{rank_limit}-max{spec['rows']}+train"
        tokenizer, _ = load_vocab_spec(write_vocab_spec(args.out / "vocab", src_dir, kept, policy))
        model = prune_vocab(model, kept)
        vocab_report = {
            "policy": policy,
            "rank_limit": rank_limit,
            "train_token_ids": len(train_ids),
        }
    else:
        hub = Path(snapshot_download(source, allow_patterns=["vocab.txt", "config.json"]))
        tokens = (hub / "vocab.txt").read_text(encoding="utf-8").splitlines()
        full_tok = BertTokenizerFast(vocab={t: i for i, t in enumerate(tokens)}, do_lower_case=True)
        train_ids = set()
        for text in train_texts:
            train_ids.update(full_tok(text)["input_ids"])
        kept, stats = wordpiece_keep_set(tokens, train_ids, spec["rows"])
        tokenizer = write_wordpiece_tokenizer(hub, tokens, kept, args.out / "vocab")
        model = prune_word_rows(model, kept)
        vocab_report = {
            "policy": f"wordpiece-id-order-{spec['rows']}+train",
            **stats,
            "train_notes_check": agreement(full_tok, tokenizer, train_texts),
            "audit_notes_check": agreement(full_tok, tokenizer, audit_texts),
        }
        cfg = model.encoder.config
        cfg.pad_token_id = tokenizer.pad_token_id
        assert tokenizer.pad_token_id == 0, "padding_idx must stay 0 after pruning"

    if spec["ffn_k"] is not None:
        enc = tokenizer(
            train_texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        )
        scores = neuron_scores(model, enc["input_ids"], enc["attention_mask"])
        keep_ffn = [select_neurons(layer["combined"], spec["ffn_k"]) for layer in scores]
        model = prune_ffn(model, keep_ffn)
        (args.out / "ffn_map.json").write_text(
            json.dumps({"criterion": "combined", "k": spec["ffn_k"], "keep": keep_ffn}) + "\n"
        )

    save_checkpoint(
        model,
        tokenizer,
        args.out,
        {
            "encoder": f"{args.candidate}-bakeoff",
            "max_length": MAX_LENGTH,
            "trained": False,
            "note": "bake-off init: pretrained layers, random seeded heads",
        },
    )
    cfg = model.encoder.config
    report = {
        "candidate": args.candidate,
        "source": source,
        "layer_map": init["layer_map"],
        "n_source_layers": n_source,
        "vocab_rows": int(cfg.vocab_size),
        "positions": int(cfg.max_position_embeddings),
        "hidden": int(cfg.hidden_size),
        "intermediate": int(cfg.intermediate_size),
        "params": model.param_count(),
        "train_notes": str(TRAIN_NOTES),
        "vocab": vocab_report,
    }
    (args.out / "build_report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
