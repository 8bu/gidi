"""Invariants of the committed annotation-v3 ``debt-01`` batch (frozen-set exclusion, validity)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "build_debt_relabel_queue", REPO_ROOT / "scripts" / "build_debt_relabel_queue.py"
)
bd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bd)

BATCH = REPO_ROOT / bd.DEFAULT_OUT_DIR


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def test_committed_batch_is_reproducible():
    assert bd.main(["--root", str(REPO_ROOT), "--check"]) == 0


def test_queue_never_contains_a_frozen_held_out_note():
    queue = read_jsonl(BATCH / bd.QUEUE_FILE)
    held_out = read_jsonl(REPO_ROOT / bd.HUMAN_QUEUE)
    held_ids = {r["id"] for r in held_out}
    exact_key = bd._load_hv().exact_key
    held_texts = {exact_key(r["text"]) for r in held_out}
    manifest = json.loads((BATCH / bd.MANIFEST_FILE).read_text("utf-8"))
    assert not held_ids & {q["id"] for q in queue}
    assert not held_texts & {exact_key(q["text"]) for q in queue}
    assert len(manifest["excluded_human_value_01_ids"]) == 9
    assert set(manifest["excluded_human_value_01_ids"]) <= held_ids


def test_proposals_cover_the_queue_and_name_the_llm():
    queue = read_jsonl(BATCH / bd.QUEUE_FILE)
    proposals = read_jsonl(BATCH / bd.PROPOSALS_FILE)
    assert [p["id"] for p in proposals] == [q["id"] for q in queue]
    assert all(p["reason"].startswith(bd.REASON_PREFIX) for p in proposals)
    # a debt-only note is never proposed as skipped; unclear direction is uncertain, not forced
    for p in proposals:
        if p["annotation_status"] == "complete":
            assert p["type"] in {"borrow", "lend"}
        else:
            assert p["annotation_status"] == "uncertain" and p["note"]
