#!/usr/bin/env python
"""Analyze how Gidi's data uses BamiBERT's 20,481-token vocabulary (compression-v1, vocab slice).

Usage:
    uv run python scripts/analyze_vocab.py \\
        [--tokenizer-dir models/distillation-v2/supervised/student-4x768-pretrained/seed1] \\
        [--out experiments/compression-v1/vocab/analysis.json]

Tokenizes train (A, 723), frozen test (B, 105) and probe (C, 81) with the baseline tokenizer and
reports vocabulary usage, composition by decoded-script category, per-slice fragmentation, number
shorthand segmentation and counterparty-target segmentation. Analysis only: nothing here feeds a
construction decision (``prune_vocab.py`` uses train and the vocab itself).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from evaluate_probe import load_probe
from transformers import PreTrainedTokenizerBase

from gidi.annotation.split import accented as text_is_accented
from gidi.compression.vocab import BpeVocab, decode_token, token_category
from gidi.modeling.preprocessing import trim_span
from gidi.modeling.tokenization import load_tokenizer
from gidi.tokenizer.audit import AMOUNT_PATTERN
from gidi.training.train import load_split

TOKENIZER_DIR = Path("models/distillation-v2/supervised/student-4x768-pretrained/seed1")
TRAIN = Path("datasets/annotation-v1/distillation-v1/train.jsonl")
TEST = Path("datasets/annotation-v1/distillation-v1/test.jsonl")
PROBE = Path("datasets/probe-v1")
MAX_LENGTH = 32
SPLITS = ("train", "test", "probe")
BUCKETS = (("1", 1, 1), ("2-4", 2, 4), ("5-9", 5, 9), ("10-99", 10, 99), ("100+", 100, 10**9))
TITLES = (
    "anh",
    "chị",
    "chi",
    "em",
    "cô",
    "co",
    "chú",
    "chu",
    "bác",
    "bac",
    "dì",
    "di",
    "ông",
    "ong",
    "bà",
    "ba",
    "bạn",
    "ban",
    "mợ",
    "mo",
    "cậu",
    "cau",
    "dượng",
    "duong",
)
MERCHANT_TYPES = ("expense", "refund")


def load_splits() -> dict[str, list[dict[str, Any]]]:
    return {"train": load_split(TRAIN), "test": load_split(TEST), "probe": load_probe(PROBE)}


def tokenize(tokenizer: PreTrainedTokenizerBase, text: str, max_length: int | None):
    """Fast-tokenizer encoding with offsets; ``max_length`` truncates (None = full)."""
    return tokenizer(
        text,
        truncation=max_length is not None,
        max_length=max_length,
        return_offsets_mapping=True,
    )


def pieces_in_span(tokenizer, text: str, start: int, end: int, max_length: int | None = None):
    """Token strings whose trimmed offsets overlap ``[start, end)`` (the BIO-tagged tokens)."""
    enc = tokenize(tokenizer, text, max_length)
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])
    out = []
    for token, (s, e), special in zip(
        tokens,
        enc["offset_mapping"],
        tokenizer.get_special_tokens_mask(enc["input_ids"], already_has_special_tokens=True),
        strict=True,
    ):
        if special:
            continue
        s, e = trim_span(text, int(s), int(e))
        if s < e and s < end and e > start:
            out.append(token)
    return out


def show(pieces: Sequence[str]) -> list[str]:
    """Readable form of byte-level token strings (leading space kept, partial bytes escaped)."""
    return [decode_token(p) for p in pieces]


def words_with_offsets(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in re.finditer(r"\S+", text)]


def word_piece_counts(tokenizer, text: str) -> list[int]:
    """Number of tokens per whitespace-separated word (token assigned by its trimmed start)."""
    enc = tokenize(tokenizer, text, None)
    starts = []
    for (s, e), special in zip(
        enc["offset_mapping"],
        tokenizer.get_special_tokens_mask(enc["input_ids"], already_has_special_tokens=True),
        strict=True,
    ):
        if not special:
            s, e = trim_span(text, int(s), int(e))
            if s < e:
                starts.append(s)
    return [sum(ws <= s < we for s in starts) for ws, we in words_with_offsets(text)]


def is_person_target(record: dict[str, Any]) -> bool:
    """Heuristic person-name test (documented in the report ``heuristic`` field)."""
    target = record["target"]
    text = record["text"]
    words = target["text"].split()
    before = text[: target["start"]].split()
    prev = before[-1].lower() if before else ""
    if prev in TITLES or (words and words[0].lower() in TITLES):
        return True
    capitalized = (
        all(w[:1].isupper() and not w.isupper() for w in words)
        and len(words) <= 4
        and not any(ch.isdigit() for ch in target["text"])
    )
    return capitalized and record["type"] not in MERCHANT_TYPES


def bucket_counts(counts: Counter) -> dict[str, int]:
    return {name: sum(lo <= c <= hi for c in counts.values()) for name, lo, hi in BUCKETS}


def decoded(bpe: BpeVocab, i: int) -> str:
    return decode_token(bpe.tokens[i])


def usage_report(
    tokenizer, bpe: BpeVocab, splits: dict[str, list[dict[str, Any]]], max_length: int | None
) -> dict[str, Any]:
    """Distinct ids, frequency buckets and top tokens per split/union for one truncation view."""
    counts: dict[str, Counter] = {}
    note_ids: dict[str, list[set[int]]] = {}
    for name, records in splits.items():
        counts[name] = Counter()
        note_ids[name] = []
        for r in records:
            ids = tokenize(tokenizer, r["text"], max_length)["input_ids"]
            counts[name].update(ids)
            note_ids[name].append(set(ids))
    union: Counter = Counter()
    for c in counts.values():
        union.update(c)
    vocab_size = bpe.size
    out: dict[str, Any] = {
        "vocab_size": vocab_size,
        "distinct_ids": {n: len(c) for n, c in counts.items()} | {"union": len(union)},
        "unused_share_of_vocab": {
            n: 1 - len(c) / vocab_size for n, c in (counts | {"union": union}).items()
        },
        "count_histogram_buckets": {
            n: bucket_counts(c) for n, c in (counts | {"union": union}).items()
        },
        "tokens_total": {n: sum(c.values()) for n, c in counts.items()},
    }
    for name, c in (("train", counts["train"]), ("union", union)):
        out[f"top30_{name}"] = [
            {"id": i, "token": decoded(bpe, i), "count": n} for i, n in c.most_common(30)
        ]
    seen = set(counts["train"])
    unseen: dict[str, Any] = {}
    for name in ("test", "probe"):
        missing = {i for i in counts[name] if i not in seen}
        notes_with = sum(bool(ids & missing) for ids in note_ids[name])
        unseen[name] = {
            "distinct_tokens": len(missing),
            "occurrences": sum(counts[name][i] for i in missing),
            "token_share_of_split": sum(counts[name][i] for i in missing)
            / sum(counts[name].values()),
            "notes_with_unseen_token": notes_with,
            "notes": len(note_ids[name]),
            "tokens": sorted(
                ({"id": i, "token": decoded(bpe, i), "count": counts[name][i]} for i in missing),
                key=lambda d: (-d["count"], d["id"]),
            ),
        }
    both = {i for i in counts["test"] if i not in seen} & {
        i for i in counts["probe"] if i not in seen
    }
    unseen["in_both_test_and_probe"] = sorted(decoded(bpe, i) for i in both)
    out["unseen_in_train"] = unseen
    out["_counts"] = counts
    return out


def category_report(bpe: BpeVocab, counts: dict[str, Counter]) -> dict[str, Any]:
    cat_of = [
        token_category(t, special=i in bpe.special_ids, base=i in bpe.base_ids)
        for i, t in enumerate(bpe.tokens)
    ]
    out: dict[str, Any] = {}
    union: set[int] = set().union(*(set(c) for c in counts.values()))
    for cat in sorted(set(cat_of)):
        ids = [i for i, c in enumerate(cat_of) if c == cat]
        entry: dict[str, Any] = {
            "vocab_tokens": len(ids),
            "examples": [decoded(bpe, i) for i in ids[:: max(1, len(ids) // 8)][:8]],
            "distinct_used": {n: sum(i in c for i in ids) for n, c in counts.items()}
            | {"union": sum(i in union for i in ids)},
            "occurrences": {n: sum(c[i] for i in ids) for n, c in counts.items()},
        }
        out[cat] = entry
    out["_cat_of"] = cat_of
    return out


def slice_report(tokenizer, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Fragmentation of the notes (untruncated view) split by accented/unaccented."""

    def block(recs: list[dict[str, Any]]) -> dict[str, Any]:
        if not recs:
            return {"notes": 0}
        n_tok = []
        n_trunc = 0
        words = 0
        multi = 0
        pieces = 0
        chars = 0
        for r in recs:
            ids = tokenize(tokenizer, r["text"], None)["input_ids"]
            n_tok.append(len(ids) - 2)
            n_trunc += len(ids) > MAX_LENGTH
            chars += len(r["text"])
            counts = word_piece_counts(tokenizer, r["text"])
            words += len(counts)
            pieces += sum(counts)
            multi += sum(c >= 2 for c in counts)
        return {
            "notes": len(recs),
            "mean_content_tokens_per_note": sum(n_tok) / len(recs),
            "max_content_tokens": max(n_tok),
            "notes_truncated_at_32": n_trunc,
            "words": words,
            "tokens_per_word": pieces / words,
            "fraction_words_split_into_2plus_pieces": multi / words,
            "chars_per_token": chars / sum(n_tok),
        }

    accented = [r for r in records if text_is_accented(r["text"])]
    plain = [r for r in records if not text_is_accented(r["text"])]
    return {"all": block(records), "accented": block(accented), "unaccented": block(plain)}


NUMBER_CLASSES = (
    ("Nk (45k, 500k)", re.compile(r"^\d+(?:[.,]\d+)?[kK]$")),
    ("NtrM (1tr2, 3tr4, 15tr)", re.compile(r"^\d+(?:[.,]\d+)?(?:tr|TR|Tr)\d*$")),
    ("grouped (89,000, 1.500.000)", re.compile(r"^\d{1,3}(?:[.,]\d{3})+(?:\s*(?:đ|₫|vnđ|vnd))?$")),
    ("N <word> (2 triệu, 200 nghìn, 2 củ)", re.compile(r"^\d+(?:[.,]\d+)?\s+\S+$")),
    ("N<đ|vnd>", re.compile(r"^\d+\s*(?:đ|₫|vnđ|vnd)$")),
)
SHORTHAND_PROBES = (
    "500k",
    "1tr2",
    "3tr4",
    "89,000",
    "1.5tr",
    "2 triệu",
    "45k",
    "1tr5",
    "200 nghìn",
    "2 củ",
    "1,5tr",
    "12tr",
    "1.500.000",
    "50K",
    "15tr",
    "500đ",
)


def number_report(tokenizer, splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    by_class: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)
    for records in splits.values():
        for r in records:
            for m in AMOUNT_PATTERN.finditer(r["text"]):
                pieces = pieces_in_span(tokenizer, r["text"], m.start(), m.end())
                label = next((c for c, rx in NUMBER_CLASSES if rx.match(m.group())), "other")
                by_class[label].append((m.group(), show(pieces)))
    classes = {}
    for label, items in sorted(by_class.items()):
        shapes: dict[str, list[str]] = {}
        for text, pieces in items:
            shapes.setdefault(text, pieces)
        classes[label] = {
            "occurrences": len(items),
            "mean_pieces": sum(len(p) for _, p in items) / len(items),
            "max_pieces": max(len(p) for _, p in items),
            "examples": [
                {"text": t, "pieces": p}
                for t, p in sorted(shapes.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:6]
            ],
        }
    # Requested shorthand strings, tokenized inside a neutral finance note.
    probes = {}
    for amount in SHORTHAND_PROBES:
        text = f"chuyển khoản {amount}"
        start = len("chuyển khoản ")
        probes[amount] = show(pieces_in_span(tokenizer, text, start, start + len(amount)))
    in_data = {}
    for amount in ("500k", "1tr2", "3tr4", "89,000", "1.5tr", "2 triệu"):
        for name, records in splits.items():
            for r in records:
                at = r["text"].find(amount)
                if at >= 0 and amount not in in_data:
                    in_data[amount] = {
                        "split": name,
                        "note": r["text"],
                        "pieces": show(pieces_in_span(tokenizer, r["text"], at, at + len(amount))),
                    }
    return {"by_class": classes, "probe_in_neutral_note": probes, "first_example_in_data": in_data}


def target_report(
    tokenizer, bpe: BpeVocab, splits: dict[str, list[dict[str, Any]]], train_counts: Counter
) -> dict[str, Any]:
    train_targets = {r["target"]["text"] for r in splits["train"] if r["target"]}
    train_targets_lower = {t.lower() for t in train_targets}
    out: dict[str, Any] = {
        "heuristic": (
            "person = the word before the target (or the target's first word) is a title/kinship "
            f"word {list(TITLES)}, OR every target word is capitalized (not ALL-CAPS), <=4 words, "
            "no digits, and the note type is not expense/refund; everything else = "
            "merchant_brand_other. Types are used only to label analysis rows."
        ),
        "splits": {},
    }
    train_ids = set(train_counts)
    for name, records in splits.items():
        rows = []
        for r in records:
            t = r["target"]
            if t is None:
                continue
            pieces = pieces_in_span(tokenizer, r["text"], t["start"], t["end"])
            ids = tokenizer.convert_tokens_to_ids(pieces)
            rows.append(
                {
                    "text": t["text"],
                    "person": is_person_target(r),
                    "pieces": show(pieces),
                    "n_pieces": len(pieces),
                    "unseen_string": t["text"] not in train_targets,
                    "unseen_string_ci": t["text"].lower() not in train_targets_lower,
                    "has_token_unseen_in_train": any(i not in train_ids for i in ids),
                }
            )
        block: dict[str, Any] = {"n_notes": len(records), "n_targets": len(rows)}
        for label, sel in (
            ("all", rows),
            ("person", [x for x in rows if x["person"]]),
            ("merchant_brand_other", [x for x in rows if not x["person"]]),
        ):
            block[label] = {
                "n": len(sel),
                "mean_pieces": sum(x["n_pieces"] for x in sel) / len(sel) if sel else None,
                "max_pieces": max((x["n_pieces"] for x in sel), default=None),
                "mean_pieces_per_word": (
                    sum(x["n_pieces"] for x in sel) / sum(len(x["text"].split()) for x in sel)
                    if sel
                    else None
                ),
                "piece_histogram": dict(sorted(Counter(x["n_pieces"] for x in sel).items())),
            }
        if name != "train":
            unseen = [x for x in rows if x["unseen_string"]]
            block["unseen_targets"] = {
                "n": len(unseen),
                "share_of_targets": len(unseen) / len(rows),
                "n_case_insensitive": sum(x["unseen_string_ci"] for x in rows),
                "mean_pieces": sum(x["n_pieces"] for x in unseen) / len(unseen),
                "n_with_token_unseen_in_train": sum(x["has_token_unseen_in_train"] for x in unseen),
                "n_person": sum(x["person"] for x in unseen),
                "examples": sorted(
                    unseen[:: max(1, len(unseen) // 15)][:15],
                    key=lambda x: (-x["n_pieces"], x["text"]),
                ),
            }
        out["splits"][name] = block
    return out


def build_report(tokenizer_dir: Path) -> dict[str, Any]:
    tokenizer = load_tokenizer(str(tokenizer_dir))
    tok_json = json.loads((tokenizer_dir / "tokenizer.json").read_text(encoding="utf-8"))
    bpe = BpeVocab.from_json(tok_json)
    splits = load_splits()
    report: dict[str, Any] = {
        "tokenizer_dir": str(tokenizer_dir),
        "notes": {n: len(r) for n, r in splits.items()},
        "max_length": MAX_LENGTH,
        "vocab": {
            "size": bpe.size,
            "merges": len(bpe.merges),
            "single_byte_base_tokens": len(bpe.base_ids),
            "missing_byte_values": len(bpe.missing_bytes()),
            "orphans_not_produced_by_a_merge": len(bpe.orphans()),
        },
    }
    for view, max_length in (("truncated_at_32", MAX_LENGTH), ("untruncated", None)):
        usage = usage_report(tokenizer, bpe, splits, max_length)
        counts = usage.pop("_counts")
        report[f"usage_{view}"] = usage
        if max_length is not None:
            cats = category_report(bpe, counts)
            cats.pop("_cat_of")
            report["categories"] = cats
            train_counts = counts["train"]
    report["slices"] = {name: slice_report(tokenizer, recs) for name, recs in splits.items()}
    report["numbers"] = number_report(tokenizer, splits)
    report["targets"] = target_report(tokenizer, bpe, splits, train_counts)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokenizer-dir", type=Path, default=TOKENIZER_DIR)
    parser.add_argument(
        "--out", type=Path, default=Path("experiments/compression-v1/vocab/analysis.json")
    )
    args = parser.parse_args(argv)
    report = build_report(args.tokenizer_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    used = report["usage_truncated_at_32"]
    print(f"distinct ids used: {used['distinct_ids']}")
    print(f"unused share of vocab (union): {used['unused_share_of_vocab']['union']:.3f}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
