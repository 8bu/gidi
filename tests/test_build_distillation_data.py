"""Tests for ``scripts/build_distillation_data.py`` on tiny fixture trees (never the real data)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bt = load_script("build_training")
bd = load_script("build_distillation_data")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")


def split_row(id_: str, text: str, type_: str, target: str | None, batch: str = "baseline-01"):
    span = None
    if target:
        start = text.index(target)
        span = {"text": target, "start": start, "end": start + len(target)}
    return {
        "id": id_,
        "text": text,
        "type": type_,
        "target": span,
        "source_batch": batch,
        "accented": True,
    }


def make_root(tmp_path: Path, validation_id: str = "v1") -> Path:
    """A root where training-v2 is built and splits/manifest.json holds the frozen hashes."""
    splits = tmp_path / bt.SPLITS
    write_jsonl(splits / "train.jsonl", [split_row("tr1", "ăn phở 45k", "expense", None)])
    write_jsonl(
        splits / "validation.jsonl",
        [split_row(validation_id, "lương tháng 9 15tr", "income", None)],
    )
    write_jsonl(
        splits / "test.jsonl", [split_row("t1", "cho chị Diệu vay 3tr sửa nhà", "lend", "Diệu")]
    )
    manifest = {
        "sha256": {
            f"{name}.jsonl": hashlib.sha256((splits / f"{name}.jsonl").read_bytes()).hexdigest()
            for name in ("train", "validation", "test")
        }
    }
    (splits / "manifest.json").write_text(json.dumps(manifest), "utf-8")
    write_jsonl(tmp_path / "datasets/annotation-v1/combined/queue.jsonl", [])
    write_jsonl(tmp_path / "datasets/annotation-v1/combined/labels.jsonl", [])
    write_jsonl(tmp_path / "datasets/probe-x/queue.jsonl", [{"id": "p1", "text": "trúng số 2tr"}])
    write_jsonl(tmp_path / "datasets/probe-x/labels.jsonl", [])
    aug = tmp_path / bt.ANNOTATION / "targeted-02"
    label = split_row(
        "a1",
        "anh Khôi cho mình mượn 2tr đóng tiền trọ",
        "borrow",
        "Khôi",
        "targeted-annotation-v1-02",
    )
    write_jsonl(aug / "queue.jsonl", [{"id": "a1", "text": label["text"]}])
    write_jsonl(aug / "labels.jsonl", [{**label, "annotation_status": "complete"}])
    write_jsonl(aug / "notes.jsonl", [{"id": "a1", "group": "t02-A-01", "patterns": ["x"]}])
    write_jsonl(aug / "provenance.jsonl", [{"id": "a1", "annotator": "ai", "review_state": "auto"}])
    assert bt.main(["--version", "v2", "--root", str(tmp_path)]) == 0
    return tmp_path


def run(root: Path, *flags: str) -> int:
    return bd.main(["--root", str(root), *flags])


def test_train_is_training_v2_bytes_then_validation_bytes(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    assert run(root) == 0
    out = root / bd.OUT
    assert (out / "train.jsonl").read_bytes() == (
        root / bd.TRAINING_V2 / "train.jsonl"
    ).read_bytes() + (root / bd.SPLITS / "validation.jsonl").read_bytes()
    for name in ("validation", "test"):
        assert (out / f"{name}.jsonl").resolve() == (root / bd.SPLITS / f"{name}.jsonl").resolve()
    manifest = json.loads((out / "manifest.json").read_text("utf-8"))
    assert manifest["counts"] == {
        "original_train": 1,
        "targeted_02": 1,
        "frozen_validation": 1,
        "train": 3,
    }
    assert manifest["type_counts"]["income"] == 1
    assert (
        manifest["files"]["train.jsonl"]
        == hashlib.sha256((out / "train.jsonl").read_bytes()).hexdigest()
    )
    assert {"datasets/probe-x/queue.jsonl", "datasets/annotation-v1/splits/test.jsonl"} <= set(
        manifest["evaluation_files"]
    )
    assert run(root, "--check") == 0


def test_check_detects_changed_train_file(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    assert run(root) == 0
    with (root / bd.OUT / "train.jsonl").open("a", encoding="utf-8") as f:
        f.write('{"id": "extra"}\n')
    assert run(root, "--check") == 1


def test_frozen_hash_mismatch_fails_and_writes_nothing(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    with (root / bd.SPLITS / "test.jsonl").open("a", encoding="utf-8") as f:
        f.write("\n")
    assert run(root) == 1
    assert not (root / bd.OUT).exists()


def test_tampered_training_v2_fails_and_writes_nothing(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    with (root / bd.TRAINING_V2 / "train.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(split_row("extra", "ăn bún 30k", "expense", None)) + "\n")
    assert run(root) == 1
    assert not (root / bd.OUT).exists()


def test_validation_record_that_is_a_test_id_fails_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = make_root(tmp_path, validation_id="t1")
    assert run(root) == 1
    assert "frozen test record" in capsys.readouterr().err
    assert not (root / bd.OUT).exists()
