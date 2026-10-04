"""Invariants of the LLM-composed `contrast-01` batch and the `training-v2` set built from it."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bc = load("build_contrast_01")
bt = load("build_annotation_v3_training_v2")
BATCH = REPO_ROOT / bc.DEFAULT_OUT_DIR


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def test_committed_batch_and_training_set_are_reproducible_and_leak_free():
    # --check rebuilds in memory (the leakage gates run inside) and compares the files on disk.
    assert bc.main(["--root", str(REPO_ROOT), "--check"]) == 0
    assert bt.main(["--root", str(REPO_ROOT), "--check"]) == 0


def test_contrast_notes_are_llm_labelled_and_never_a_regression_note():
    queue = read_jsonl(BATCH / bc.QUEUE_FILE)
    labels = read_jsonl(BATCH / bc.LABELS_FILE)
    provenance = read_jsonl(BATCH / bc.PROVENANCE_FILE)
    assert [r["id"] for r in labels] == [q["id"] for q in queue]
    assert {p["annotator"] for p in provenance} == {"llm"}
    assert {label["annotation_status"] for label in labels} == {"complete"}
    hv = bc.debt._load_hv()
    refs = bc.regression_texts(REPO_ROOT)
    assert max(bc.regression_ratio(hv, q["text"], refs) for q in queue) < bc.REGRESSION_RATIO


def test_label_spans_slice_the_note_text():
    texts = {q["id"]: q["text"] for q in read_jsonl(BATCH / bc.QUEUE_FILE)}
    for label in read_jsonl(BATCH / bc.LABELS_FILE):
        for field in ("target", "value"):
            span = label[field]
            if span is not None:
                assert texts[label["id"]][span["start"] : span["end"]] == span["text"]


def test_span_requires_a_word_boundary():
    assert bc.span("trả nợ Hung 2tr", "Hung", "target")["start"] == 7
    with pytest.raises(ValueError, match="word boundary"):
        bc.span("trả nợ Hung 2tr", "Hun", "target")
    with pytest.raises(ValueError, match="not found"):
        bc.span("trả nợ Hung 2tr", "Lan", "target")


def test_near_duplicate_of_a_held_out_note_is_refused():
    held = {"frozen test": [{"id": "t", "text": "tra lai chi Mai 2tr"}]}
    clean = {"id": "a", "text": "mua cơm 30k"}
    bt.assert_no_near_duplicates([clean], held)
    with pytest.raises(ValueError, match="near duplicate"):
        bt.assert_no_near_duplicates([{"id": "b", "text": "Trả lại chị Mai 2tr"}], held)
