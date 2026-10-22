#!/usr/bin/env python
"""Load-test the feasibility student with plain onnxruntime and measure its tokenizer.

Everything here is a measurement on the 63 tokenizer-audit notes (not training data, not an eval
set). Writes verify.json next to this file.

Usage: uv run python experiments/gidi-en-backbone/verify_student.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from tokenizers import Tokenizer, pre_tokenizers

from gidi.compression.vocab import load_vocab_spec
from gidi.modeling.tokenization import load_tokenizer

HERE = Path(__file__).parent
STUDENT = Path("models/gidi-en-backbone/feasibility-roberta-4x768")
ONNX = Path("models/gidi-en-backbone/feasibility-roberta-4x768-onnx/model.int8.onnx")
NOTES = [json.loads(line)["text"] for line in (HERE / "audit-notes.jsonl").open()]


def main() -> None:
    tok, kept = load_vocab_spec(STUDENT / "vocab")
    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    enc = tok(NOTES, padding=True, truncation=True, max_length=32, return_tensors="np")
    ids, mask = enc["input_ids"].astype(np.int64), enc["attention_mask"].astype(np.int64)
    type_logits, tag_logits = sess.run(None, {"input_ids": ids, "attention_mask": mask})
    n = int(mask[0].sum())
    single = sess.run(None, {"input_ids": ids[:1, :n], "attention_mask": mask[:1, :n]})
    out: dict = {
        "onnxruntime": {
            "version": ort.__version__,
            "file_bytes": ONNX.stat().st_size,
            "inputs": [(i.name, i.shape) for i in sess.get_inputs()],
            "outputs": [(o.name, o.shape) for o in sess.get_outputs()],
            "batch_shapes": [list(type_logits.shape), list(tag_logits.shape)],
            "batch1_shapes": [list(a.shape) for a in single],
            "finite": bool(np.isfinite(type_logits).all() and np.isfinite(tag_logits).all()),
            "ops": sorted({n.op_type for n in onnx.load(str(ONNX)).graph.node}),
        }
    }

    full = load_tokenizer("roberta-base")
    remap = {old: new for new, old in enumerate(kept)}
    same = pieces_pruned = pieces_full = 0
    for text in NOTES:
        f = full(text, add_special_tokens=False)["input_ids"]
        p = tok(text, add_special_tokens=False)["input_ids"]
        pieces_full += len(f)
        pieces_pruned += len(p)
        same += int(all(i in remap for i in f) and [remap[i] for i in f] == p)
    out["tokenizer_on_audit_notes"] = {
        "rows": len(kept),
        "identical_to_full_vocab": f"{same}/{len(NOTES)}",
        "tokens_per_note_full_vocab": round(pieces_full / len(NOTES), 3),
        "tokens_per_note_pruned": round(pieces_pruned / len(NOTES), 3),
    }

    # The student tokenizer uses the Vietnamese bundle's Split pre-tokenizer (web runtime
    # unchanged); roberta-base was pretrained with the GPT-2 regex. Compare ids on more notes.
    proxy = [json.loads(line)["text"] for line in (HERE / "proxy-train-notes.jsonl").open()]
    texts = NOTES + proxy
    split_tok = Tokenizer.from_file(str(STUDENT / "vocab" / "tokenizer.json"))
    gpt2_tok = Tokenizer.from_file(str(STUDENT / "vocab" / "tokenizer.json"))
    gpt2_tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    a_ids = [split_tok.encode(t, add_special_tokens=False).ids for t in texts]
    b_ids = [gpt2_tok.encode(t, add_special_tokens=False).ids for t in texts]
    differ = [t for t, a, b in zip(texts, a_ids, b_ids, strict=True) if a != b]
    out["pretokenizer_split_vs_gpt2_regex"] = {
        "notes_with_different_ids": f"{len(differ)}/{len(texts)} (audit + proxy notes)",
        "examples": differ[:8],
    }
    (HERE / "verify.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
