"""Tests for ``gidi.annotation.leakage`` on a tiny fixture tree (never the real data)."""

from __future__ import annotations

import json
from pathlib import Path

from gidi.annotation import leakage as cns


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")


def span(text: str, target: str) -> dict:
    start = text.index(target)
    return {"text": target, "start": start, "end": start + len(target)}


def test_rejects_eval_swaps_and_mirrors_but_keeps_same_group_pairs(tmp_path: Path) -> None:
    test_text = "cho chị Diệu vay 3tr sửa nhà"
    probe_text = "Khánh mượn 2tr đóng học"
    write_jsonl(
        tmp_path / cns.SPLITS_DIR / "test.jsonl",
        [{"id": "t1", "text": test_text, "type": "lend", "target": span(test_text, "Diệu")}],
    )
    write_jsonl(tmp_path / cns.SPLITS_DIR / "validation.jsonl", [])
    write_jsonl(tmp_path / "datasets/probe-x/queue.jsonl", [{"id": "p1", "text": probe_text}])
    write_jsonl(tmp_path / "datasets/probe-x/labels.jsonl", [])
    write_jsonl(tmp_path / cns.COMBINED_DIR / "queue.jsonl", [{"id": "t1", "text": test_text}])
    write_jsonl(tmp_path / cns.COMBINED_DIR / "labels.jsonl", [])

    candidates = [
        # entity + amount swap of a test note: same skeleton
        {"text": "cho chị Hoa vay 7tr sửa nhà", "intended_target": "Hoa", "group": "g1"},
        # word-order mirror of a probe note: same bag of words
        {"text": "mượn Khánh 2tr đóng học", "intended_target": "Khánh", "group": "g2"},
        # deliberate minimal pair: near-identical, same group -> allowed
        {
            "text": "bà ngoại lì xì mình 300k sáng mùng hai",
            "intended_target": "bà ngoại",
            "group": "g3",
        },
        {"text": "lì xì bà ngoại 300k sáng mùng hai", "intended_target": "bà ngoại", "group": "g3"},
        # near-identical to g3 but a different group -> batch near-duplicate
        {
            "text": "bà nội lì xì mình 300k sáng mùng hai",
            "intended_target": "bà nội",
            "group": "g4",
        },
    ]
    eval_refs, train_refs = cns.load_references(tmp_path)
    results = cns.check(candidates, eval_refs, train_refs, show_eval=False)
    rejects = [set(r["reject"]) for r in results]

    assert "eval_skel_match" in rejects[0]
    assert results[0]["eval_neighbour"] is None  # evaluation text stays hidden
    assert "eval_bag_match" in rejects[1]
    assert rejects[2] == {"batch_near_other_group"}  # only its g4 near-copy, not its partner
    assert rejects[3] == set()
    assert "batch_near_other_group" in rejects[4]
