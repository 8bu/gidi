"""Tests for ``scripts/build_human_value_queue.py`` on a tiny fixture tree (never the real data)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "build_human_value_queue", REPO_ROOT / "scripts" / "build_human_value_queue.py"
)
hv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hv)

OUT_DIR = Path("datasets/annotation-v2/human-value-01")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")


def make_tree(root: Path, pool: list[tuple[str, str, str]], train: list[str], probe: list[str]):
    write_jsonl(
        root / hv.POOL,
        [
            {"id": rid, "text": text, "source": "claude", "quet": {"status": status}}
            for rid, text, status in pool
        ],
    )
    for rel in (*hv.HARD_FILES, *hv.SEEN_FILES, *hv.UNAPPROVED_RAW):
        write_jsonl(root / rel, [])
    write_jsonl(root / hv.HARD_FILES[0], [{"id": f"t{i}", "text": t} for i, t in enumerate(train)])
    write_jsonl(
        root / "datasets/probe-v1/notes.jsonl",
        [{"id": f"p{i}", "text": t} for i, t in enumerate(probe)],
    )
    write_jsonl(root / hv.ANNOTATION_V1_QUEUE, [])
    write_jsonl(root / hv.ANNOTATION_V1_LABELS, [])
    (root / "datasets/annotation-v2/targeted-value-01").mkdir(parents=True, exist_ok=True)


@pytest.fixture
def small_limits(monkeypatch):
    monkeypatch.setattr(hv, "MIN_TOTAL", 1)
    monkeypatch.setattr(hv, "TARGET_TOTAL", 50)


def test_folded_rule_ignores_case_accents_digits_whitespace():
    a = hv.Reference("Trả  nợ chị Hạnh 1tr5")
    b = hv.Reference("tra no chi hanh 2tr6")
    assert hv.rule_between(a, b) == "folded"
    assert hv.rule_between(hv.Reference("mua giày 450k"), hv.Reference("bia 1 thùng 340k")) is None


def test_exclusion_dedup_and_approval(tmp_path, small_limits):
    pool = [
        ("a", "cho Nam mượn 500k", "approved"),  # exact in V8 rule-design input
        ("b", "CHO nam  muon 900k", "approved"),  # folded duplicate of 'a' train note
        ("c", "gửi tiết kiệm vcb 10tr kỳ hạn 3th", "approved"),  # near duplicate of a probe note
        ("d", "mua 3 vé 150k", "approved"),
        ("e", "mua 3 vé 200k", "approved"),  # folded duplicate of 'd' inside the pool
        ("f", "tiền điện tháng 10 hết 742,000", "needs-review"),  # not approved
        ("g", "thưởng tết 2 tháng lương", "approved"),
        ("h", "nhận lương t10 18tr2", "approved"),
    ]
    make_tree(
        tmp_path,
        pool,
        train=["cho Nam mượn 500k"],
        probe=["gửi tiết kiệm vcb 10tr kỳ hạn 3t"],
    )
    files = hv.build(tmp_path, OUT_DIR)
    queue = [json.loads(line) for line in files[hv.QUEUE_FILE].decode().splitlines()]
    manifest = json.loads(files[hv.MANIFEST_FILE])

    assert {r["id"] for r in queue} == {"d", "g", "h"}
    assert manifest["counts"]["not_approved"] == 1
    assert manifest["counts"]["dropped_hard_exclusion"]["exact"] == 1
    assert manifest["counts"]["dropped_hard_exclusion"]["folded"] == 1
    assert manifest["counts"]["dropped_hard_exclusion"]["sequence"] == 1
    assert manifest["counts"]["dropped_pool_duplicate"]["folded"] == 1
    # the queue carries no value-derived field
    assert all(set(r) == {"id", "text", "strata", "review_group"} for r in queue)
    assert all(r["review_group"] == "primary" for r in queue)
    assert hv.build(tmp_path, OUT_DIR) == files


def test_surface_strata_separate_context_numbers_from_amounts():
    strata, _ = hv.assign_strata("ăn 2 tô phở 70")
    assert "multi_number" in strata and "quantity_amount" in strata
    assert "bare_number" not in hv.assign_strata("nhận lương t10 18tr2")[0]
    assert "bare_number" in hv.assign_strata("tra bot no ban Duy 500")[0]
    assert "date_month_year_amount" in hv.assign_strata("mượn mẹ 1tr5 hôm 12/11")[0]
    assert "null_no_number_ambiguous" in hv.assign_strata("thưởng tết 2 tháng lương")[0]
    # a number straight after a money unit continues the expression ("1 triệu 2")
    assert hv.surface_features("gói 1 triệu 2")["n_expr"] == 1
