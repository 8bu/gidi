"""Data loading for gidi-en (annotation-en-v1): note texts joined with their labels.

Splits live in ``datasets/annotation-en-v1/splits/<name>.jsonl`` (id, text); labels in
``labels-<name>.jsonl`` (test/dev/seed) or ``silver-train.jsonl`` (train). Records come out in the
shape ``gidi.training.train.prepare`` expects: ``id, text, type, target`` with only the
``complete`` notes (the only trainable status). Target spans are exact substrings, so the token
level BIO tags come from ``gidi.modeling.preprocessing.encode`` unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gidi.corpus.jsonl import read_jsonl
from gidi.modeling.preprocessing import TYPES

EN_DIR = Path("datasets/annotation-en-v1")
PRIMARY_SOURCE = "consensus"


def _texts(root: Path, split: str) -> dict[str, str]:
    return {r["id"]: r["text"] for r in read_jsonl(root / "splits" / f"{split}.jsonl")}


def _join(texts: dict[str, str], labels: list[dict[str, Any]], origin: str) -> list[dict[str, Any]]:
    out = []
    for label in labels:
        if label["annotation_status"] != "complete":
            continue
        if label["type"] not in TYPES:
            raise ValueError(f"{origin}: {label['id']!r} has unknown type {label['type']!r}")
        text = texts[label["id"]]
        target = label["target"]
        if target is not None and text[target["start"] : target["end"]] != target["text"]:
            raise ValueError(f"{origin}: {label['id']!r} target is not an exact substring")
        out.append({"id": label["id"], "text": text, "type": label["type"], "target": target})
    return out


def load_labeled(split: str, root: Path = EN_DIR) -> list[dict[str, Any]]:
    """Complete notes of ``test``, ``dev`` or ``seed`` with their final labels."""
    if split not in ("test", "dev", "seed"):
        raise ValueError(f"no label file for split {split!r}")
    labels = read_jsonl(root / f"labels-{split}.jsonl")
    return _join(_texts(root, split), labels, f"labels-{split}")


def load_train(root: Path = EN_DIR) -> list[dict[str, Any]]:
    """Training notes: silver-train (Clef teacher) plus the labelled seed notes. Never test/dev."""
    silver = _join(_texts(root, "train"), read_jsonl(root / "silver-train.jsonl"), "silver-train")
    return silver + load_labeled("seed", root)


def load_slang_silver(root: Path = EN_DIR) -> list[dict[str, Any]]:
    """Self-consistent complete notes of ``silver-slang-train.jsonl`` (v2 slang/Gen Z notes).

    Silver labels from the Clef kNN teacher (pool = seed). Notes where the two teacher passes
    disagree are dropped. Texts are stored in the file itself.
    """
    rows = [r for r in read_jsonl(root / "silver-slang-train.jsonl") if r["consistent"]]
    texts = {r["id"]: r["text"] for r in rows}
    return _join(texts, rows, "silver-slang-train")


def load_train_v2(root: Path = EN_DIR) -> list[dict[str, Any]]:
    """v2 training notes: ``load_train`` plus the consistent slang silver notes. Never eval sets."""
    return load_train(root) + load_slang_silver(root)


def sources(root: Path = EN_DIR) -> dict[str, str]:
    """Label provenance per note id (``consensus`` or ``claude-adjudicated``)."""
    return {r["id"]: r["source"] for r in read_jsonl(root / "provenance.jsonl")}


def primary_ids(split: str, root: Path = EN_DIR) -> set[str]:
    """Ids of the primary acceptance subset: complete notes whose labels are ``consensus``."""
    src = sources(root)
    return {r["id"] for r in load_labeled(split, root) if src[r["id"]] == PRIMARY_SOURCE}
