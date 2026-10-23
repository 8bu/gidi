"""English value labeling (scanner A), value loaders and the value-eval helpers."""

from __future__ import annotations

import pytest

from gidi.corpus.jsonl import read_jsonl
from gidi.modeling.value import validate_value_record
from gidi.training.english import (
    EN_DIR,
    VALUE_FILES,
    load_labeled_values,
    load_train_values_v2,
)
from gidi.value_parser.english import amount_core, choose, loose_candidates, scan


def pick(text: str) -> str | None:
    match = choose(scan(text))
    return match.text if match else None


@pytest.mark.parametrize(
    ("note", "amount"),
    [
        ("venmo'd mom 50", "50"),
        ("owe Daryl $20", "$20"),
        ("coffee 4.50", "4.50"),
        ("paid josh 15 bucks 4 the haircut", "15 bucks"),
        ("got a loan from my cousin 2 grand", "2 grand"),
        ("grab to office 12s$", "12s$"),
        ("sold old car 1,65,000", "1,65,000"),
        ("lent Desmond fifty dollars", "fifty dollars"),
        ("iou buddy a tenner", "tenner"),
        ("need a hundred for rent", "a hundred"),
        ("rent 800 for oct 2024", "800"),
        ("lunch 2 tacos 15", "15"),
        ("sam paid 25 plus 3 weeks late", "25"),
        ("moved 250 into the vanguard s&p 500 fund", "250"),
        ("401k contribution 400", "400"),
        ("lunch on card 4 sam", "4"),
    ],
)
def test_scan_picks_the_amount(note: str, amount: str) -> None:
    assert pick(note) == amount


@pytest.mark.parametrize(
    "note",
    [
        "got paid today",
        "dinner at 7:30",
        "paid on 3/5",
        "the 2nd round",
        "due in 4 weeks",
        "at three",
    ],
)
def test_scan_finds_no_amount(note: str) -> None:
    assert pick(note) is None


def test_loose_candidates_keep_odd_digit_tokens() -> None:
    texts = [c.text for c in loose_candidates("paid 2hrs parking 8")]
    assert "8" in texts and "2hrs" in texts


@pytest.mark.parametrize(
    ("span", "core"),
    [("$20", "20"), ("20 bucks", "20"), ("rs 500", "500"), ("150s$", "150"), ("2.5k", "2.5k")],
)
def test_amount_core(span: str, core: str) -> None:
    assert amount_core(span) == core


def test_value_label_files_cover_every_note_and_are_exact() -> None:
    for name, file in VALUE_FILES.items():
        rows = read_jsonl(EN_DIR / file)
        assert rows, name
        ids = [r["id"] for r in rows]
        assert len(ids) == len(set(ids)), name
        for r in rows:
            assert r["value_provenance"].startswith("silver-"), "value labels are never human"


def test_train_values_exclude_held_out_notes_and_follow_v2_order() -> None:
    from gidi.training.english import load_train_v2

    train = load_train_values_v2()
    base = load_train_v2()
    assert [r["id"] for r in train] == [r["id"] for r in base]
    assert all(not r["id"].startswith("probe-") for r in train)
    for r in train:
        validate_value_record(r)


def test_eval_values_attach_to_complete_test_notes() -> None:
    records = load_labeled_values("test")
    assert records and all(r["value_status"] in ("complete", "uncertain") for r in records)
    for r in records:
        validate_value_record(r)
