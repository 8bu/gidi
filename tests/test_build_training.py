"""Tests for ``scripts/build_training.py`` on tiny fixture trees (never the real dataset)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_training", REPO_ROOT / "scripts" / "build_training.py"
)
bt = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bt)

V2_OUT = bt.VERSIONS["v2"]["out"]
V3_OUT = bt.VERSIONS["v3"]["out"]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")


def split_row(id_: str, text: str, type_: str, target: str | None) -> dict:
    span = None
    if target:
        start = text.index(target)
        span = {"text": target, "start": start, "end": start + len(target)}
    return {
        "id": id_,
        "text": text,
        "type": type_,
        "target": span,
        "source_batch": "baseline-01",
        "accented": True,
    }


def write_batch(
    root: Path, batch: str, id_: str, text: str, target: str, group: str, type_: str = "borrow"
) -> None:
    aug = root / bt.ANNOTATION / batch
    label = split_row(id_, text, type_, target)
    write_jsonl(aug / "queue.jsonl", [{"id": id_, "text": text}])
    write_jsonl(aug / "labels.jsonl", [{**label, "annotation_status": "complete"}])
    write_jsonl(aug / "notes.jsonl", [{"id": id_, "group": group, "patterns": ["lender_first"]}])
    write_jsonl(aug / "provenance.jsonl", [{"id": id_, "annotator": "ai", "review_state": "auto"}])


def make_root(tmp_path: Path, augment_text: str, augment_target: str) -> Path:
    splits = tmp_path / bt.SPLITS
    write_jsonl(splits / "train.jsonl", [split_row("tr1", "ăn phở 45k", "expense", None)])
    write_jsonl(
        splits / "validation.jsonl", [split_row("v1", "lương tháng 9 15tr", "income", None)]
    )
    write_jsonl(
        splits / "test.jsonl", [split_row("t1", "cho chị Diệu vay 3tr sửa nhà", "lend", "Diệu")]
    )
    write_jsonl(tmp_path / "datasets/annotation-v1/combined/queue.jsonl", [])
    write_jsonl(tmp_path / "datasets/annotation-v1/combined/labels.jsonl", [])
    write_jsonl(tmp_path / "datasets/probe-x/queue.jsonl", [{"id": "p1", "text": "trúng số 2tr"}])
    write_jsonl(tmp_path / "datasets/probe-x/labels.jsonl", [])
    write_batch(tmp_path, "targeted-02", "a1", augment_text, augment_target, "t02-A-01")
    return tmp_path


def test_builds_frozen_train_prefix_and_links_eval_splits(tmp_path: Path) -> None:
    root = make_root(tmp_path, "anh Khôi cho mình mượn 2tr đóng tiền trọ", "Khôi")
    assert bt.main(["--version", "v2", "--root", str(root)]) == 0
    out = root / V2_OUT
    frozen = (root / bt.SPLITS / "train.jsonl").read_bytes()
    body = (out / "train.jsonl").read_bytes()
    assert body.startswith(frozen)
    added = json.loads(body[len(frozen) :])
    assert (added["id"], added["group"], added["type"]) == ("a1", "t02-A-01", "borrow")
    assert (out / "test.jsonl").resolve() == (root / bt.SPLITS / "test.jsonl").resolve()
    assert bt.main(["--version", "v2", "--root", str(root), "--check"]) == 0
    manifest = json.loads((out / "manifest.json").read_text("utf-8"))
    assert manifest["name"] == "training-v2"
    assert manifest["counts"] == {"original_train": 1, "targeted_02_trainable": 1, "train": 2}
    assert "targeted_03_trainable" not in manifest["counts"]


def test_template_leakage_against_test_fails_and_writes_nothing(tmp_path: Path) -> None:
    # entity + amount swap of the frozen test note
    root = make_root(tmp_path, "cho chị Hoa vay 7tr sửa nhà", "Hoa")
    with pytest.raises(SystemExit) as exc:
        bt.main(["--version", "v2", "--root", str(root)])
    assert exc.value.code == 1
    assert not (root / V2_OUT).exists()


def test_v3_appends_batches_in_order_with_per_batch_manifest_keys(tmp_path: Path) -> None:
    root = make_root(tmp_path, "anh Khôi cho mình mượn 2tr đóng tiền trọ", "Khôi")
    write_batch(
        root, "targeted-03", "b1", "bạn Nam trả mình 500k tiền cơm", "Nam", "t03-A-01", "repay"
    )
    assert bt.main(["--version", "v3", "--root", str(root)]) == 0
    out = root / V3_OUT
    frozen = (root / bt.SPLITS / "train.jsonl").read_bytes()
    added = [
        json.loads(line) for line in (out / "train.jsonl").read_bytes()[len(frozen) :].splitlines()
    ]
    assert [r["id"] for r in added] == ["a1", "b1"]
    assert [r["source_batch"] for r in added] == [
        "targeted-annotation-v1-02",
        "targeted-annotation-v1-03",
    ]
    manifest = json.loads((out / "manifest.json").read_text("utf-8"))
    assert manifest["name"] == "training-v3"
    assert manifest["counts"] == {
        "original_train": 1,
        "targeted_02_trainable": 1,
        "targeted_03_trainable": 1,
        "train": 3,
    }
    for key in ("type_counts_targeted_03", "targeted_03_patterns", "targeted_03_groups"):
        assert key in manifest
    assert bt.main(["--version", "v3", "--root", str(root), "--check"]) == 0


def test_v3_duplicate_of_earlier_batch_fails_and_writes_nothing(tmp_path: Path) -> None:
    root = make_root(tmp_path, "anh Khôi cho mình mượn 2tr đóng tiền trọ", "Khôi")
    write_batch(
        root, "targeted-03", "b1", "anh Khôi cho mình mượn 2tr đóng tiền trọ", "Khôi", "t03-A-01"
    )
    with pytest.raises(SystemExit) as exc:
        bt.main(["--version", "v3", "--root", str(root)])
    assert exc.value.code == 1
    assert not (root / V3_OUT).exists()
    # the same fixture is a valid v2 build: only the cross-batch gate rejects it
    assert bt.main(["--version", "v2", "--root", str(root)]) == 0


def test_v3_id_shared_between_batches_fails_and_writes_nothing(tmp_path: Path) -> None:
    root = make_root(tmp_path, "anh Khôi cho mình mượn 2tr đóng tiền trọ", "Khôi")
    write_batch(
        root, "targeted-03", "a1", "bạn Nam trả mình 500k tiền cơm", "Nam", "t03-A-01", "repay"
    )
    with pytest.raises(SystemExit) as exc:
        bt.main(["--version", "v3", "--root", str(root)])
    assert exc.value.code == 1
    assert not (root / V3_OUT).exists()
