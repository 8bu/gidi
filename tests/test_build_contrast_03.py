"""Invariants of the LLM-composed `contrast-03` batch, `training-v4` and the hv02 scoring guard."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bc = load("build_contrast_03")
bt = load("build_annotation_v3_training_v4")
score = load("score_human_value_02")
BATCH = REPO_ROOT / bc.DEFAULT_OUT_DIR
TRAIN = REPO_ROOT / bt.OUT


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def test_committed_batch_and_training_set_are_reproducible_and_leak_free():
    # --check rebuilds in memory (the leakage gates, human-value-02 included, run inside).
    assert bc.main(["--root", str(REPO_ROOT), "--check"]) == 0
    assert bt.main(["--root", str(REPO_ROOT), "--check"]) == 0


def test_batch_is_llm_labelled_complete_and_covers_every_family():
    queue = read_jsonl(BATCH / bc.QUEUE_FILE)
    labels = read_jsonl(BATCH / bc.LABELS_FILE)
    provenance = read_jsonl(BATCH / bc.PROVENANCE_FILE)
    assert [r["id"] for r in labels] == [q["id"] for q in queue]
    assert {p["annotator"] for p in provenance} == {"llm"}
    assert {label["annotation_status"] for label in labels} == {"complete"}
    families = [p["family"] for p in provenance]
    assert set(families) == set(bc.QUOTA)
    assert all(families.count(f) <= quota for f, quota in bc.QUOTA.items())
    assert len(queue) >= 75


def test_label_spans_slice_the_note_text_and_banks_used_as_channel_have_no_target():
    texts = {q["id"]: q["text"] for q in read_jsonl(BATCH / bc.QUEUE_FILE)}
    provenance = {p["id"]: p for p in read_jsonl(BATCH / bc.PROVENANCE_FILE)}
    for label in read_jsonl(BATCH / bc.LABELS_FILE):
        for field in ("target", "value"):
            span = label[field]
            if span is not None:
                assert texts[label["id"]][span["start"] : span["end"]] == span["text"]
        if label["type"] == "transfer":  # annotation-v1: transfers never have a target
            assert label["target"] is None
        if provenance[label["id"]]["family"] in {"gift", "item"} and label["target"]:
            # a receiver or a product brand is never the target; only a named shop can be
            assert label["target"]["text"].lower() not in {"nike", "sony", "iphone", "adidas"}


def test_human_value_02_is_part_of_the_gate():
    manifest = json.loads((BATCH / bc.MANIFEST_FILE).read_text("utf-8"))
    gate = manifest["leakage"]
    queue = REPO_ROOT / bc.HV02_QUEUE
    assert gate["human_value_02"]["gated"] is True
    assert gate["reference_texts"]["human-value-02"] == len(read_jsonl(queue))
    assert gate["max_char3_kept"]["human-value-02"] < bc.HV_CHAR3_LIMIT
    # the train set never holds a human-value-02 text or id (exact, NFC + lowercase)
    held = read_jsonl(queue)
    held_ids = {r["id"] for r in held}
    held_texts = {bt.v1.exact_key(r["text"]) for r in held}
    for record in read_jsonl(TRAIN / "train.jsonl"):
        assert record["id"] not in held_ids
        assert bt.v1.exact_key(record["text"]) not in held_texts


def test_builders_refuse_without_human_value_02(monkeypatch):
    monkeypatch.setattr(bc, "HV02_QUEUE", Path("datasets/annotation-v3/does-not-exist.jsonl"))
    with pytest.raises(ValueError, match="human-value-02"):
        bc.build(REPO_ROOT, BATCH)
    monkeypatch.setattr(bt, "HV02_QUEUE", Path("datasets/annotation-v3/does-not-exist.jsonl"))
    with pytest.raises(ValueError, match="human-value-02"):
        bt.build(REPO_ROOT)


def test_training_v4_is_v2_plus_the_kept_contrast_notes():
    train = read_jsonl(TRAIN / "train.jsonl")
    manifest = json.loads((TRAIN / "manifest.json").read_text("utf-8"))
    base_bytes = (REPO_ROOT / bt.BASE / "train.jsonl").read_bytes()
    assert (TRAIN / "train.jsonl").read_bytes().startswith(base_bytes)
    removed = {r["id"] for r in manifest["contrast_02_removed"]}
    ids = {r["id"] for r in train}
    assert removed and not removed & ids
    assert {r["reason"] for r in manifest["contrast_02_removed"]} <= set(bt.REMOVE_RULES)
    by_batch = {}
    for record in train:
        by_batch[record["source_batch"]] = by_batch.get(record["source_batch"], 0) + 1
    assert by_batch["contrast-03"] == len(read_jsonl(BATCH / bc.QUEUE_FILE))
    assert by_batch["contrast-02"] == 100 - len(removed)
    assert manifest["counts"]["train"] == len(train)


def test_removal_rules_hit_the_suspect_notes_only():
    def rec(type_, target, text="x"):
        return {"type": type_, "target": {"text": target} if target else None, "text": text}

    reason = bt.removal_reason
    assert reason(rec("income", "grab"), []) == "brand-income"
    assert reason(rec("income", "Chợ Tốt"), []) == "brand-income"
    assert reason(rec("lend", "dong nghiep"), []) == "generic-peer-noun"
    assert reason(rec("lend", "bạn"), []) == "generic-peer-noun"
    # a brand as the shop of an expense, a named person and a null target stay in
    assert reason(rec("expense", "grab"), []) is None
    assert reason(rec("lend", "Đạt"), []) is None
    assert reason(rec("income", None), []) is None
    held = [bt.hv.Reference("tra no anh Dung 2tr")]
    assert reason(rec("repayment_out", "Dung", "tra no anh Dung 3tr"), held) == (
        "char3-human-value-02"
    )


def test_hv02_scoring_refuses_until_labels_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(score, "HV02", tmp_path)
    with pytest.raises(SystemExit, match="does not exist yet"):
        score.main(["--out", str(tmp_path / "out.json")])
    assert not (tmp_path / "out.json").exists()
