"""Overlap checks of candidate training notes against evaluation sets, training data, each other.

Normalisation follows ``gidi.annotation.split.normalize``: lowercase, diacritics stripped, amounts
masked. The *skeleton* additionally masks the counterparty and capitalised name tokens, so
entity and amount swaps collide.

A candidate is **rejected** when it has:

* against evaluation sets (validation, test, ``datasets/probe-*``): the same normalised text, the
  same bag of words, the same skeleton, or a similarity of at least ``EVAL_THRESHOLD``;
* against training data (annotation-v1 records outside validation/test): the same normalised
  text or a similarity of at least ``NEAR_DUP_THRESHOLD``;
* within the batch: the same normalised text, or a similarity of at least ``NEAR_DUP_THRESHOLD``
  with a note from a *different* group (minimal pairs share a group on purpose).

It is **warned** when its skeleton is used by more than ``MAX_SKELETON_USES`` groups in the batch
(template collapse) or matches a training skeleton.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from gidi.annotation.split import STOPWORDS, normalize
from gidi.corpus.jsonl import read_jsonl

EVAL_THRESHOLD = 0.85
NEAR_DUP_THRESHOLD = 0.90
MAX_SKELETON_USES = 2
SPLITS_DIR = Path("datasets/annotation-v1/splits")
COMBINED_DIR = Path("datasets/annotation-v1/combined")
PROBE_GLOB = "datasets/probe-*"
EVAL_REJECTS = frozenset({"eval_norm_match", "eval_bag_match", "eval_skel_match", "eval_near"})

_NAME = re.compile(r"(?<!\w)[A-ZÀ-Ỹ][\wÀ-ỹ]*")


def skeleton(text: str, target: str | None) -> str:
    text = unicodedata.normalize("NFC", text)
    if target and target in text:
        text = text.replace(target, " X ", 1)
    return normalize(_NAME.sub(" X ", text))


def bag(norm: str) -> tuple[str, ...]:
    return tuple(sorted(t for t in norm.split() if t not in STOPWORDS))


def reference(text: str, target: str | None, source: str) -> dict[str, Any]:
    norm = normalize(text)
    return {
        "text": text,
        "norm": norm,
        "bag": bag(norm),
        "skel": skeleton(text, target),
        "source": source,
    }


def _target_text(record: dict) -> str | None:
    target = record.get("target")
    return target["text"] if isinstance(target, dict) else None


def load_references(root: Path) -> tuple[list[dict], list[dict]]:
    """Evaluation references (validation, test, probes) and training references under ``root``."""
    eval_refs, train_refs = [], []
    held_out = set()
    for split in ("validation", "test"):
        for r in read_jsonl(root / SPLITS_DIR / f"{split}.jsonl"):
            held_out.add(r["id"])
            eval_refs.append(reference(r["text"], _target_text(r), split))
    for probe_dir in sorted(root.glob(PROBE_GLOB)):
        labels = {lab["id"]: lab for lab in read_jsonl(probe_dir / "labels.jsonl")}
        for r in read_jsonl(probe_dir / "queue.jsonl"):
            target = _target_text(labels.get(r["id"], {}))
            eval_refs.append(reference(r["text"], target, probe_dir.name))
    labels = {lab["id"]: lab for lab in read_jsonl(root / COMBINED_DIR / "labels.jsonl")}
    for r in read_jsonl(root / COMBINED_DIR / "queue.jsonl"):
        if r["id"] not in held_out:
            train_refs.append(reference(r["text"], _target_text(labels.get(r["id"], {})), "train"))
    return eval_refs, train_refs


def nearest(norm: str, refs: list[dict], skip: int | None = None) -> tuple[float, int | None]:
    """Highest SequenceMatcher ratio of ``norm`` against ``refs`` and its index."""
    best, best_i = 0.0, None
    for i, ref in enumerate(refs):
        if i == skip:
            continue
        m = SequenceMatcher(None, norm, ref["norm"])
        if m.real_quick_ratio() <= best or m.quick_ratio() <= best:
            continue
        ratio = m.ratio()
        if ratio > best:
            best, best_i = ratio, i
    return best, best_i


def check(
    candidates: list[dict], eval_refs: list[dict], train_refs: list[dict], show_eval: bool = False
) -> list[dict[str, Any]]:
    """Per-candidate reject/warn reasons and nearest neighbours.

    Each candidate is ``{"text", "intended_target"?, "group"?}``; ``intended_target`` only masks
    the counterparty for skeleton matching. Evaluation neighbour texts are returned only with
    ``show_eval``.
    """
    cands = [reference(c["text"], c.get("intended_target"), "batch") for c in candidates]
    eval_keys = {
        "norm": {r["norm"] for r in eval_refs},
        "bag": {r["bag"] for r in eval_refs if r["bag"]},
        "skel": {r["skel"] for r in eval_refs},
    }
    train_norms = {r["norm"] for r in train_refs}
    train_skels = {r["skel"] for r in train_refs}
    norm_counts = Counter(c["norm"] for c in cands)
    skel_groups: dict[str, set] = {}
    for cand, raw in zip(cands, candidates, strict=True):
        skel_groups.setdefault(cand["skel"], set()).add(raw.get("group"))

    results = []
    for i, (cand, raw) in enumerate(zip(cands, candidates, strict=True)):
        reject, warn = [], []
        for key in ("norm", "bag", "skel"):
            if cand[key] and cand[key] in eval_keys[key]:
                reject.append(f"eval_{key}_match")
        eval_sim, eval_i = nearest(cand["norm"], eval_refs)
        if eval_sim >= EVAL_THRESHOLD:
            reject.append("eval_near")
        if cand["norm"] in train_norms:
            reject.append("train_duplicate")
        train_sim, train_i = nearest(cand["norm"], train_refs)
        if train_sim >= NEAR_DUP_THRESHOLD:
            reject.append("train_near")
        if cand["skel"] in train_skels:
            warn.append("train_skeleton")
        if norm_counts[cand["norm"]] > 1:
            reject.append("batch_duplicate")
        others = [
            c if candidates[j].get("group") != raw.get("group") else {"norm": ""}
            for j, c in enumerate(cands)
        ]
        batch_sim, batch_i = nearest(cand["norm"], others, skip=i)
        if batch_sim >= NEAR_DUP_THRESHOLD:
            reject.append("batch_near_other_group")
        if len(skel_groups[cand["skel"]]) > MAX_SKELETON_USES:
            warn.append("template_collapse")
        eval_ref = eval_refs[eval_i] if eval_i is not None else None
        results.append(
            {
                "text": raw["text"],
                "group": raw.get("group"),
                "reject": reject,
                "warn": warn,
                "eval_sim": round(eval_sim, 3),
                "eval_source": eval_ref["source"] if eval_ref else None,
                "eval_neighbour": eval_ref["text"] if eval_ref and show_eval else None,
                "train_sim": round(train_sim, 3),
                "train_neighbour": train_refs[train_i]["text"] if train_i is not None else None,
                "batch_sim": round(batch_sim, 3),
                "batch_neighbour": cands[batch_i]["text"] if batch_i is not None else None,
            }
        )
    return results
