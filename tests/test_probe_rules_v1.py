"""Consistency checks for the evaluation-only rule probe datasets/probe-rules-v1."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "datasets" / "probe-rules-v1"
TYPES = {
    "expense",
    "income",
    "borrow",
    "lend",
    "repayment_in",
    "repayment_out",
    "transfer",
    "refund",
}


def _rows(name: str) -> list[dict]:
    lines = (ROOT / name).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_probe_rules_v1_consistent() -> None:
    queue, labels, prov = (_rows(n) for n in ("queue.jsonl", "labels.jsonl", "provenance.jsonl"))
    ids = [r["id"] for r in queue]
    assert len(ids) == 64 == len(set(ids))
    assert ids == [r["id"] for r in labels] == [r["id"] for r in prov]
    assert len({r["text"] for r in queue}) == 64
    for q, lab in zip(queue, labels, strict=True):
        assert lab["annotation_status"] == "complete"
        assert lab["type"] in TYPES
        tgt = lab["target"]
        if tgt is not None:
            assert q["text"][tgt["start"] : tgt["end"]] == tgt["text"]
    assert all(p["annotator"] == "llm" for p in prov)
    for rule in ("R1", "R2", "R3", "R4"):
        assert sum(p["rule"] == rule for p in prov) == 16
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["human_labels"] is False
