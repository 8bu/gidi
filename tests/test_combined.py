"""Tests for ``scripts/build_combined.py`` on tiny fixture trees (never the real dataset)."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_combined.py"


def dumps(obj: dict, *, compact: bool = False) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":") if compact else None)


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{ln}\n" for ln in lines), encoding="utf-8")


def queue_row(id_: str, text: str, position: int, corpus: str) -> str:
    return dumps({"id": id_, "text": text, "corpus": corpus, "position": position})


def label(id_: str, status: str, type_: str | None, target: str | None, text: str = "") -> str:
    span = None
    if target is not None:
        start = text.index(target)
        span = {"text": target, "start": start, "end": start + len(target)}
    record = {"id": id_, "annotation_status": status, "type": type_, "target": span}
    if status == "uncertain":
        record["note"] = "ambiguous"
    return dumps(record, compact=True)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """Repository skeleton: real config/docs, a 3-record baseline, a 2-record targeted batch."""
    for rel in ("configs/annotation-v1.yaml", "docs/annotation-v1.md"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO_ROOT / rel, tmp_path / rel)
    base = tmp_path / "datasets/annotation-v1"
    write_lines(
        base / "queue.jsonl",
        [
            queue_row("b-1", "Hùng trả nợ 500k", 1, "corpus/b.jsonl"),
            queue_row("b-2", "ca phe highlands 45k", 2, "corpus/b.jsonl"),
            queue_row("b-3", "còn nợ Nam 300k", 3, "corpus/b.jsonl"),
        ],
    )
    # Quet saves labels in annotation order, not queue order
    write_lines(
        base / "labels.jsonl",
        [
            label("b-2", "complete", "expense", "highlands", "ca phe highlands 45k"),
            label("b-3", "skipped", None, None),
            label("b-1", "complete", "repayment_in", "Hùng", "Hùng trả nợ 500k"),
        ],
    )
    write_lines(
        base / "provenance.jsonl",
        [dumps({"id": i, "annotator": "human"}, compact=True) for i in ("b-1", "b-2", "b-3")],
    )
    targeted = base / "targeted-01"
    write_lines(
        targeted / "queue.jsonl",
        [
            queue_row("t-1", "chị Thảo trả lại 1tr", 1, "corpus/t.jsonl"),
            queue_row("t-2", "chuyen tien cho me 2tr", 2, "corpus/t.jsonl"),
        ],
    )
    write_lines(
        targeted / "labels.jsonl",
        [
            label("t-1", "complete", "repayment_in", "Thảo", "chị Thảo trả lại 1tr"),
            label("t-2", "uncertain", None, None),
        ],
    )
    prov = {
        "annotator": "ai",
        "origin": "synthetic-targeted",
        "source_batch": "targeted-annotation-v1-01",
    }
    write_lines(
        targeted / "provenance.jsonl",
        [dumps({"id": i, **prov}, compact=True) for i in ("t-1", "t-2")],
    )
    return tmp_path


def run(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def combined(root: Path) -> Path:
    return root / "datasets/annotation-v1/combined"


@pytest.fixture
def built(root: Path) -> Path:
    result = run(root)
    assert result.returncode == 0, result.stderr
    return root


def test_build_merges_batches_and_keeps_labels_verbatim(built: Path):
    out = combined(built)
    queue = [
        json.loads(ln) for ln in (out / "queue.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [q["id"] for q in queue] == ["b-1", "b-2", "b-3", "t-1", "t-2"]
    assert [q["source_batch"] for q in queue] == ["baseline-01"] * 3 + [
        "targeted-annotation-v1-01"
    ] * 2

    base = built / "datasets/annotation-v1"
    expected_labels = (base / "labels.jsonl").read_text(encoding="utf-8") + (
        base / "targeted-01/labels.jsonl"
    ).read_text(encoding="utf-8")
    assert (out / "labels.jsonl").read_text(encoding="utf-8") == expected_labels

    prov = [
        json.loads(ln) for ln in (out / "provenance.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    # provenance is line-aligned with labels, and source_batch is added when missing
    assert [p["id"] for p in prov] == ["b-2", "b-3", "b-1", "t-1", "t-2"]
    assert [p["source_batch"] for p in prov][:3] == ["baseline-01"] * 3


def test_manifest_counts_and_hashes(built: Path):
    manifest = json.loads((combined(built) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["annotation_version"] == "annotation-v1"
    assert manifest["frozen"] is True
    assert manifest["records"] == {
        "total": 5,
        "by_status": {"complete": 3, "skipped": 1, "uncertain": 1},
        "trainable": 3,
    }
    assert manifest["types"]["expense"] == 1
    assert manifest["types"]["repayment_in"] == 2
    assert manifest["accent"]["overall"] == {"accented": 2, "unaccented": 1}
    assert manifest["accent"]["by_type"]["repayment_in"] == {"accented": 2, "unaccented": 0}
    assert manifest["annotators"] == {"human_reviewed": 3, "ai_accepted": 2}
    baseline = manifest["source_batches"]["baseline-01"]
    assert (baseline["records"], baseline["trainable"], baseline["human_reviewed"]) == (3, 2, 3)
    assert manifest["source_batches"]["targeted-annotation-v1-01"]["ai_accepted"] == 2
    labels_sha = hashlib.sha256((combined(built) / "labels.jsonl").read_bytes()).hexdigest()
    assert manifest["sha256"]["combined"]["combined/labels.jsonl"] == labels_sha


def test_build_is_deterministic(built: Path):
    first = {p.name: p.read_bytes() for p in combined(built).iterdir()}
    assert set(first) == {"queue.jsonl", "labels.jsonl", "provenance.jsonl", "manifest.json"}
    assert run(built, "--overwrite").returncode == 0
    assert {p.name: p.read_bytes() for p in combined(built).iterdir()} == first
    assert run(built, "--out-dir", "elsewhere").returncode == 0
    assert {p.name: p.read_bytes() for p in (built / "elsewhere").iterdir()} == first


def test_refuses_to_overwrite_without_flag(built: Path):
    before = (combined(built) / "labels.jsonl").read_bytes()
    result = run(built)
    assert result.returncode != 0
    assert "--overwrite" in result.stderr
    assert (combined(built) / "labels.jsonl").read_bytes() == before


def test_check_passes_when_frozen_and_fails_after_any_drift(built: Path):
    assert run(built, "--check").returncode == 0

    labels = built / "datasets/annotation-v1/labels.jsonl"
    original = labels.read_text(encoding="utf-8")
    labels.write_text(original.replace('"type":"expense"', '"type":"income"'), encoding="utf-8")
    drifted_source = run(built, "--check")
    assert drifted_source.returncode != 0
    assert "freeze check FAILED" in drifted_source.stderr
    labels.write_text(original, encoding="utf-8")
    assert run(built, "--check").returncode == 0

    queue = combined(built) / "queue.jsonl"
    queue.write_text(queue.read_text(encoding="utf-8").replace("45k", "46k"), encoding="utf-8")
    result = run(built, "--check")
    assert result.returncode != 0
    assert "queue.jsonl" in result.stderr


def test_check_fails_when_combined_files_are_missing(root: Path):
    assert run(root, "--check").returncode != 0


def test_duplicate_ids_across_batches_fail(root: Path):
    targeted = root / "datasets/annotation-v1/targeted-01"
    for name in ("queue", "labels", "provenance"):
        path = targeted / f"{name}.jsonl"
        path.write_text(path.read_text(encoding="utf-8").replace("t-1", "b-1"), encoding="utf-8")
    result = run(root)
    assert result.returncode != 0
    assert "duplicate id b-1" in result.stderr
    assert not combined(root).exists()


def test_id_mismatch_between_queue_labels_and_provenance_fails(root: Path):
    prov = root / "datasets/annotation-v1/provenance.jsonl"
    prov.write_text(prov.read_text(encoding="utf-8").replace("b-3", "b-9"), encoding="utf-8")
    result = run(root)
    assert result.returncode != 0
    assert "provenance ids differ" in result.stderr
    assert not combined(root).exists()


def test_validator_errors_fail_the_build(root: Path):
    labels = root / "datasets/annotation-v1/labels.jsonl"
    labels.write_text(
        labels.read_text(encoding="utf-8").replace('"start":0,"end":4', '"start":1,"end":5'),
        encoding="utf-8",
    )
    result = run(root)
    assert result.returncode != 0
    assert "baseline-01" in result.stderr
    assert not combined(root).exists()
