"""``build_golden_suite.py``: annotation-v2 records keep their value gold; v1 mode ignores it."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "build_golden_suite", ROOT / "scripts" / "build_golden_suite.py"
)
golden = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = golden  # @dataclass looks the module up while it executes
_SPEC.loader.exec_module(golden)


def record(rid, text, value, status="complete", target=None, kind="expense"):
    out = {
        "id": rid,
        "text": text,
        "type": kind,
        "target": target,
        "accented": text != golden._strip_accents(text),
        "provenance": {"annotator": "human"},
        "value": None,
        "value_status": status,
    }
    if value is not None:
        start = text.index(value)
        out["value"] = {"text": value, "start": start, "end": start + len(value)}
    return out


RECORDS = [
    record("a1", "cơm tấm 100", "100"),
    record("a2", "ăn 2 tô phở 70", "70"),
    record("a3", "trả góp kỳ 3 1tr5", "1tr5"),
    record("a4", "mua 3 vé 150k", "150k"),
    record("a5", "lương 1.500.000đ", "1.500.000đ"),
    record("a6", "rút tiền atm", None),
    record("a7", "mua 2 cái 50k 70k", "50k", status="uncertain"),
]


def write_dataset(root: Path) -> Path:
    (root / "splits").mkdir(parents=True)
    for split, rows in (("train", RECORDS[:5]), ("validation", RECORDS[5:])):
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        (root / "splits" / f"{split}.jsonl").write_text(text, encoding="utf-8")
    for name in ("labels", "queue", "provenance"):
        (root / f"{name}.jsonl").write_text("", encoding="utf-8")
    return root


def test_value_mode_keeps_value_gold_and_adds_value_categories(tmp_path):
    cases, summary = golden.build_golden(write_dataset(tmp_path), with_value=True)
    by_text = {c["text"]: c for c in cases}
    assert len(cases) == len(RECORDS)
    gold = by_text["ăn 2 tô phở 70"]["gold"]
    assert (gold["value"]["text"], gold["value_status"]) == ("70", "complete")
    assert "value_not_first_number" in by_text["ăn 2 tô phở 70"]["categories"]
    assert "value_unit_or_slang" in by_text["trả góp kỳ 3 1tr5"]["categories"]
    assert "value_separator" in by_text["lương 1.500.000đ"]["categories"]
    assert "value_currency_marker" in by_text["lương 1.500.000đ"]["categories"]
    assert "value_bare_number" in by_text["cơm tấm 100"]["categories"]
    assert "value_no_amount" in by_text["rút tiền atm"]["categories"]
    uncertain = by_text["mua 2 cái 50k 70k"]
    assert uncertain["gold"] == {
        "type": "expense",
        "target": None,
        "value": None,
        "value_status": "uncertain",
    }
    assert "value_uncertain" in uncertain["categories"]
    assert summary["category_counts"]["value_present"] == 5


def test_v1_mode_ignores_value_fields(tmp_path):
    cases, _ = golden.build_golden(write_dataset(tmp_path))
    assert all(set(c["gold"]) == {"type", "target"} for c in cases)
    assert not any(cat.startswith("value_") for c in cases for cat in c["categories"])


def test_flat_training_directory_has_no_skipped_records_and_dedupes_validation_in_train(tmp_path):
    # training-v1 layout: {train,validation}.jsonl directly, validation ids repeated in train
    for split, rows in (("train", RECORDS), ("validation", RECORDS[5:])):
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        (tmp_path / f"{split}.jsonl").write_text(text, encoding="utf-8")
    cases, _ = golden.build_golden(tmp_path, with_value=True)
    texts = [c["text"] for c in cases]
    assert sorted(texts) == sorted(r["text"] for r in RECORDS)  # each record exactly once
    assert all(c["gold"]["type"] is not None for c in cases)  # no skipped (type-less) records


def test_hardening_cases_put_the_value_before_across_and_after_the_cut():
    def count(text):  # one token per word plus <s> and </s>
        return len(text.split()) + 2

    cases = {c["case"]: c for c in golden.hardening_cases(count)}
    assert {c["cut"] for c in cases.values() if "cut" in c} == {
        "before",
        "at-end",
        "across",
        "after",
    }
    assert all(c["expect_empty_error"] == (not c["text"].strip()) for c in cases.values())
    assert len(cases) == len({c["case"] for c in cases.values()})
