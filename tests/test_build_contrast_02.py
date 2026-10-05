"""Invariants of the LLM-composed `contrast-02` batch, `training-v3`, and the soft-vote ensemble."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bc = load("build_contrast_02")
bt = load("build_annotation_v3_training_v3")
ev = load("evaluate_encoder_retrain_v3")
BATCH = REPO_ROOT / bc.DEFAULT_OUT_DIR


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def test_committed_batch_and_training_set_are_reproducible_and_leak_free():
    # --check rebuilds in memory (the leakage gates run inside) and compares the files on disk.
    assert bc.main(["--root", str(REPO_ROOT), "--check"]) == 0
    assert bt.main(["--root", str(REPO_ROOT), "--check"]) == 0


def test_batch_is_llm_labelled_complete_and_covers_the_quota():
    queue = read_jsonl(BATCH / bc.QUEUE_FILE)
    labels = read_jsonl(BATCH / bc.LABELS_FILE)
    provenance = read_jsonl(BATCH / bc.PROVENANCE_FILE)
    assert [r["id"] for r in labels] == [q["id"] for q in queue]
    assert {p["annotator"] for p in provenance} == {"llm"}
    assert {label["annotation_status"] for label in labels} == {"complete"}
    families = [p["family"] for p in provenance]
    assert {f: families.count(f) for f in bc.QUOTA} == bc.QUOTA
    debt_types = {
        label["type"] for label, p in zip(labels, provenance, strict=True) if p["family"] == "debt"
    }
    assert debt_types == {"borrow", "lend"}  # debt-only notes in both directions


def test_label_spans_slice_the_note_text():
    texts = {q["id"]: q["text"] for q in read_jsonl(BATCH / bc.QUEUE_FILE)}
    for label in read_jsonl(BATCH / bc.LABELS_FILE):
        for field in ("target", "value"):
            span = label[field]
            if span is not None:
                assert texts[label["id"]][span["start"] : span["end"]] == span["text"]


def test_no_note_is_close_to_a_human_value_note():
    hv = bc.debt._load_hv()
    held = [r["text"] for r in read_jsonl(REPO_ROOT / bt.v1.HUMAN / "review-queue.jsonl")]
    refs = [hv.Reference(t) for t in held]
    for q in read_jsonl(BATCH / bc.QUEUE_FILE):
        assert bc.max_char3(hv, q["text"], refs) < bc.HV_CHAR3_LIMIT, q["text"]


def test_char3_bound_refuses_a_close_note():
    human = [{"id": "h", "text": "tra no anh Dung 2tr"}]
    bt.assert_char3_below_limit([{"id": "a", "text": "mua cơm tấm 45k"}], human)
    with pytest.raises(ValueError, match="char3 bound"):
        bt.assert_char3_below_limit([{"id": "b", "text": "tra no anh Dung 3tr"}], human)


class _Member:
    """A fixed-logit stand-in for GidiPredictor.run (3 tokens: <s>, one word, </s>)."""

    def __init__(self, type_logits: list[float], tag_logits: list[list[float]]):
        self.type_logits = np.asarray(type_logits, dtype=np.float32)
        self.tag_logits = np.asarray(tag_logits, dtype=np.float32)

    def run(self, text: str):
        return ev.SimpleNamespace(
            type_logits=self.type_logits,
            tag_logits=self.tag_logits,
            input_ids=(0, 5, 2),
            normalized_text=text,
            normalized_offsets=((0, 0), (0, len(text)), (0, 0)),
        )


def test_soft_vote_follows_the_confident_member_not_the_plurality():
    o = [3.0, 0.0, 0.0]
    # two members barely prefer type 0 (expense), one is sure of type 1 (income)
    weak0 = [0.6, 0.5] + [0.0] * 6
    sure1 = [0.0, 6.0] + [0.0] * 6
    tags_o = [[5.0, 0, 0], o, [5.0, 0, 0]]
    tags_b = [[5.0, 0, 0], [0, 4.0, 0], [5.0, 0, 0]]
    members = [_Member(weak0, tags_o), _Member(weak0, tags_o), _Member(sure1, tags_b)]
    pred = ev.SoftVoteEnsemble(members).predict("khach")
    assert pred.type == "income"
    # the word is tagged O by two members and a confident B by one: mean tag prob says O
    assert pred.target_span is None
