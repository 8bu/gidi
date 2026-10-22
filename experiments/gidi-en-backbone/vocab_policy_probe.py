#!/usr/bin/env python
"""Compare vocabulary-selection policies for the pruned roberta-base vocabulary.

Selection inputs: roberta-base merge ranks and (optionally) a proxy train pool, the snapshot
``proxy-train-notes.jsonl`` (700 raw generated notes, no labels, nothing held out). The policies are
*measured* on the independently written ``audit-notes.jsonl``, which are never used for selection.

Usage: uv run python experiments/gidi-en-backbone/vocab_policy_probe.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from build_en_student import keep_set, write_source_tokenizer  # noqa: E402

from gidi.compression.vocab import BpeVocab, load_vocab_spec, write_vocab_spec  # noqa: E402
from gidi.modeling.tokenization import load_tokenizer  # noqa: E402

HERE = Path(__file__).parent
audit = [json.loads(line)["text"] for line in (HERE / "audit-notes.jsonl").open()]
proxy = [json.loads(line)["text"] for line in (HERE / "proxy-train-notes.jsonl").open()]


def main() -> None:
    full = load_tokenizer("roberta-base")
    full_ids = [full(t, add_special_tokens=False)["input_ids"] for t in audit]
    train_ids = {i for t in proxy for i in full(t, add_special_tokens=False)["input_ids"]}
    hub = Path(snapshot_download("roberta-base", allow_patterns=["tokenizer*", "config.json"]))
    result = {"proxy_train_notes": len(proxy), "proxy_distinct_token_ids": len(train_ids)}
    with tempfile.TemporaryDirectory() as t:
        src = write_source_tokenizer(hub, Path(t) / "src")
        bpe = BpeVocab.from_json(json.loads((src / "tokenizer.json").read_text()))
        for variants in (False, True):
            for use_train in (False, True):
                for max_rows in (8222, 9000, 10000):
                    kept, limit = keep_set(
                        bpe, train_ids if use_train else set(), max_rows, variants
                    )
                    name = f"{'variants' if variants else 'plain'}|train={use_train}|max{max_rows}"
                    spec = write_vocab_spec(Path(t) / "s", src, kept, name)
                    ptok, _ = load_vocab_spec(spec)
                    remap = {old: new for new, old in enumerate(kept)}
                    same = n_pruned = n_full = 0
                    for text, f in zip(audit, full_ids, strict=True):
                        p = ptok(text, add_special_tokens=False)["input_ids"]
                        n_pruned += len(p)
                        n_full += len(f)
                        same += int(all(i in remap for i in f) and [remap[i] for i in f] == p)
                    result[name] = {
                        "rows": len(kept),
                        "rank_limit": limit,
                        "audit_notes_identical": f"{same}/{len(audit)}",
                        "audit_tokens_per_note": round(n_pruned / len(audit), 3),
                        "full_vocab_tokens_per_note": round(n_full / len(audit), 3),
                    }
                    print(name, result[name], flush=True)
    (HERE / "vocab_policy_probe.json").write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
