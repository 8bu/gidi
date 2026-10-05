"""Invariants of the committed annotation-v3 ``human-value-02`` test-set queue."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "datasets/annotation-v3/human-value-02"


def _load():
    path = REPO_ROOT / "scripts" / "build_human_value_02.py"
    spec = importlib.util.spec_from_file_location("build_human_value_02", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text("utf-8").splitlines() if x.strip()]


def test_queue_is_independent_of_the_held_out_and_training_sets() -> None:
    mod = _load()
    hv = mod.hv
    queue = _rows(OUT_DIR / "review-queue.jsonl")
    for row in queue:
        assert row["id"] == "hv02-" + hashlib.sha256(row["text"].encode()).hexdigest()[:12]
        assert row["text"] == unicodedata.normalize("NFC", row["text"]).strip()
        assert row["review_group"] == "primary"
        assert set(row) == {"id", "text", "strata", "review_group"}
    texts, _ = mod.load_reference_groups(REPO_ROOT, mod.DEFAULT_OUT_DIR)
    kept, dropped, _ = mod.leakage_filter([("weak", r["text"]) for r in queue], texts)
    assert not dropped
    assert len(kept) == len(queue) == len({r["id"] for r in queue})
    inside = [hv.Reference(r["text"]) for r in queue]
    for i, ref in enumerate(inside):
        assert all(hv.rule_between(ref, other) is None for other in inside[:i])


def test_test_set_has_no_proposals_or_labels_by_the_builder() -> None:
    names = {p.name for p in OUT_DIR.iterdir()}
    assert not any("proposal" in n for n in names)
    manifest = json.loads((OUT_DIR / "manifest.json").read_text("utf-8"))
    assert manifest["proposals"] is None
    assert manifest["never_train"] is True
