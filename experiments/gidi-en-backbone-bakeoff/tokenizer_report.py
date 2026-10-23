#!/usr/bin/env python
"""Diagnostic (reporting only): pieces per note and [UNK] rate of each candidate's pruned tokenizer.

Reads the eval notes only to count pieces; nothing here feeds back into any model.
"""

from __future__ import annotations

import json
from pathlib import Path

from gidi.modeling.tokenization import load_tokenizer
from gidi.training.english import load_labeled

INIT = Path("models/gidi-en-bakeoff/init")
CHECKPOINTS = {
    "roberta-base": Path("models/gidi-en-v1-init"),
    "distilroberta": INIT / "distilroberta",
    "electra": INIT / "electra",
    "minilm": INIT / "minilm",
}


def main() -> int:
    from gidi.corpus.jsonl import read_jsonl

    sets = {
        "test": [r["text"] for r in load_labeled("test")],
        "slang_probe": [r["text"] for r in read_jsonl(Path("datasets/probe-en-slang/notes.jsonl"))]
        if Path("datasets/probe-en-slang/notes.jsonl").exists()
        else [],
    }
    out = {}
    for name, path in CHECKPOINTS.items():
        tok = load_tokenizer(str(path))
        unk_id = tok.unk_token_id
        row = {}
        for set_name, texts in sets.items():
            if not texts:
                continue
            ids = tok(texts, add_special_tokens=False)["input_ids"]
            n = sum(len(i) for i in ids)
            row[set_name] = {
                "notes": len(texts),
                "pieces_per_note": n / len(texts),
                "unk": sum(i.count(unk_id) for i in ids),
                "notes_with_unk": sum(unk_id in i for i in ids),
            }
        out[name] = row
    Path("experiments/gidi-en-backbone-bakeoff/tokenizer_report.json").write_text(
        json.dumps(out, indent=1) + "\n"
    )
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
