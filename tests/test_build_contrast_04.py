"""Gift-receiver rule: human-value-01 amendment, relabel map, `contrast-04` and `training-v5`."""

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
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


amend = load("amend_human_value_01")
relabel = load("build_gift_relabel_01")
bc = load("build_contrast_04")
bt = load("build_annotation_v3_training_v5")
BATCH = REPO_ROOT / bc.DEFAULT_OUT_DIR
TRAIN = REPO_ROOT / bt.OUT


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def test_committed_files_are_reproducible_and_leak_free():
    # --check rebuilds in memory; the leakage gates (human-value-02 batches a and b) run inside.
    assert amend.main(["--root", str(REPO_ROOT), "--check"]) == 0
    assert relabel.main(["--root", str(REPO_ROOT), "--check"]) == 0
    assert bc.main(["--root", str(REPO_ROOT), "--check"]) == 0
    assert bt.main(["--root", str(REPO_ROOT), "--check"]) == 0


def test_every_shop_the_contrast_03_quota_dropped_is_in_the_batch():
    manifest = json.loads((BATCH / bc.MANIFEST_FILE).read_text("utf-8"))
    by_name = manifest["counts"]["shop_notes_by_name"]
    for name in bc.REQUIRED_SHOPS:
        assert 1 <= by_name[name] <= bc.PER_CUE_CAP["shop"], name


def test_per_cue_cap_gives_every_name_a_turn():
    # the contrast-03 bug: a plain round-robin over coarse cues starved the later names
    rows = {
        f"{name} {i}": {"family": "shop", "cue": name}
        for name in ("a", "b", "c", "d")
        for i in range(5)
    }
    chosen = bc.pick_by_cue(list(rows), rows)
    names = [rows[t]["cue"] for t in chosen if rows[t]["family"] == "shop"]
    assert len(names) == min(bc.QUOTA["shop"], 4 * bc.PER_CUE_CAP["shop"])
    assert {n: names.count(n) for n in "abcd"} == {n: 3 for n in "abcd"}


def test_gift_notes_have_the_receiver_as_target_and_unnamed_gifts_none():
    texts = {q["id"]: q["text"] for q in read_jsonl(BATCH / bc.QUEUE_FILE)}
    family = {p["id"]: p["family"] for p in read_jsonl(BATCH / bc.PROVENANCE_FILE)}
    labels = read_jsonl(BATCH / bc.LABELS_FILE)
    gifts = [x for x in labels if family[x["id"]] == "gift"]
    assert len(gifts) >= 12
    for label in gifts:
        target = label["target"]
        text = texts[label["id"]]
        assert target is not None and text[target["start"] : target["end"]] == target["text"]
        # a kinship prefix before a proper name is dropped (`anh Dũng` -> `Dũng`); a birth-order
        # phrase or a kinship term that is the only identifier stays whole (`bác Tư`, `ông nội`)
        first = target["text"].split()[0]
        assert first not in {"anh", "chị", "em", "bé", "cháu", "bạn", "chau", "be"}
        assert " " not in target["text"] or target["text"] in {
            "bác Tư",
            "cô Năm",
            "ông Tám",
            "bac Hai",
            "ông bà ngoại",
            "ông nội",
            "ba noi",
        }
    for label in labels:
        if family[label["id"]] == "gift_null":
            assert label["target"] is None


def test_training_v5_applies_the_relabel_map_and_contrast_04():
    train = {r["id"]: r for r in read_jsonl(TRAIN / "train.jsonl")}
    for row in read_jsonl(REPO_ROOT / relabel.OUT / "map.jsonl"):
        if row["action"] == "drop":
            assert row["id"] not in train
            continue
        record = train[row["id"]]
        assert record["target"]["text"] == row["new"]
        assert record["text"][record["target"]["start"] : record["target"]["end"]] == row["new"]
        assert record["provenance"]["relabeled"] == "gift-relabel-01"
    queue = {q["id"] for q in read_jsonl(BATCH / bc.QUEUE_FILE)}
    assert queue <= set(train)
    assert (TRAIN / "validation.jsonl").is_symlink() and (TRAIN / "test.jsonl").is_symlink()


def test_relabel_map_refuses_a_row_that_does_not_match_the_record():
    record = {
        "id": "x",
        "text": "quà sinh nhật Duc 350k",
        "target": None,
        "provenance": {"annotator": "llm"},
    }
    row = {"id": "x", "text": record["text"], "old": "Duc", "new": "Duc", "action": "relabel"}
    with pytest.raises(ValueError, match="does not match"):
        bt.apply_relabel([record], [row])
    row["old"] = None
    out, dropped = bt.apply_relabel([record], [row])
    assert out[0]["target"]["text"] == "Duc" and not dropped
    record["provenance"]["annotator"] = "human"
    with pytest.raises(ValueError, match="only LLM"):
        bt.apply_relabel([record], [row])


def test_human_value_01_amendment_is_the_only_change_and_validates():
    amendment = json.loads((REPO_ROOT / amend.HV / amend.AMENDMENT).read_text("utf-8"))
    assert [c["new_target"] for c in amendment["changes"]] == ["Vy"]
    labels = {x["id"]: x for x in read_jsonl(REPO_ROOT / amend.HV / amend.LABELS)}
    change = amendment["changes"][0]
    assert labels[change["id"]]["target"] == change["new_span"]
    assert change["text"][change["new_span"]["start"] : change["new_span"]["end"]] == "Vy"
