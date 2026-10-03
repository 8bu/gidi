"""Tests for the value-span proposer, its validators and the training-record merge."""

from __future__ import annotations

import importlib.util
import json
import shutil
import unicodedata
from pathlib import Path

import pytest

from gidi.annotation import review_gate as gate
from gidi.annotation import value_review as vr
from gidi.annotation.value_span import (
    RULE_LABEL_EXTRA,
    load_value_config,
    merged_value_fields,
    pattern_signature,
    propose_value,
    to_quet_proposal,
    to_rule_label,
    validate_label_file,
    validate_merged_record,
    validate_quet_proposal,
    validate_span,
    validate_value_label,
)
from gidi.corpus.jsonl import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_value_config(ROOT / "configs" / "annotation-v2.yaml")
V2 = "datasets/annotation-v2"


def span_of(text: str) -> str | None:
    chosen = propose_value(text).chosen
    return chosen.text if chosen else None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("cơm tấm 100", "100"),
        ("mượn chú hai 5 xị", "5 xị"),
        ("ăn 2 tô phở 70", "70"),
        ("tiền điện tháng 10 hết 612", "612"),
        ("trả góp kỳ 3 1tr5", "1tr5"),
        ("20/10 mua quà mẹ 500", "500"),
        ("mua 3 vé 150k", "150k"),
        ("đóng kỳ 2 khoản vay 3tr", "3tr"),
    ],
)
def test_goal_examples(text: str, expected: str) -> None:
    p = propose_value(text)
    assert p.chosen is not None
    assert p.chosen.text == expected
    assert text[p.chosen.start : p.chosen.end] == expected


@pytest.mark.parametrize(
    ("text", "expected", "signature"),
    [
        ("cf 50k", "50k", "<num>k"),
        ("cf 50 K", "50 K", "<num> K"),
        ("cf 50K", "50K", "<num>K"),
        ("góp 1tr", "1tr", "<num>tr"),
        ("góp 1tr5", "1tr5", "<num>tr<num>"),
        ("góp 1tr150", "1tr150", "<num>tr<num>"),
        ("vay 4tr664", "4tr664", "<num>tr<num>"),
        ("mua 1.5tr", "1.5tr", "<dec.>tr"),
        ("mua 1,5tr", "1,5tr", "<dec,>tr"),
        ("lương 2 triệu", "2 triệu", "<num> trieu"),
        ("luong 2 trieu", "2 trieu", "<num> trieu"),
        ("thưởng 2 củ", "2 củ", "<num> cu"),
        ("mượn 5 xị", "5 xị", "<num> xi"),
        ("vay 5 chai", "5 chai", "<num> chai"),
        ("mượn 5 lít", "5 lít", "<num> lit"),
        ("cho 500 nghìn", "500 nghìn", "<num> nghin"),
        ("cho 500 ngàn", "500 ngàn", "<num> ngan"),
        ("mua xe 1.500.000", "1.500.000", "<dot3>"),
        ("lương về 12,500,000", "12,500,000", "<com3>"),
        ("chuyển 1500000", "1500000", "<long>"),
        ("đổ xăng 80.000đ", "80.000đ", "<dot3>đ"),
        ("trả 80.000 đồng", "80.000 đồng", "<dot3> đ"),
        ("cho 200 vnđ", "200 vnđ", "<num> đ"),
        ("mua nhà 2 tỷ", "2 tỷ", "<num> ty"),
    ],
)
def test_coverage_forms(text: str, expected: str, signature: str) -> None:
    assert span_of(text) == expected
    assert pattern_signature(expected) == signature


def test_only_the_slang_spans_need_review_among_unit_forms() -> None:
    assert propose_value("mua 1.5tr").auto_accept
    assert propose_value("trả 80.000đ").auto_accept
    for text in ("mượn 5 xị", "thưởng 2 củ", "vay 5 chai", "mua 2 triệu rưỡi"):
        p = propose_value(text)
        assert p.chosen is not None and "slang" in p.review_reasons and not p.auto_accept


def test_unaccented_single_explicit_unit_is_not_forced_to_review() -> None:
    p = propose_value("tra no cho Hung 2 trieu")
    assert p.chosen is not None and p.chosen.text == "2 trieu"
    assert "unaccented" in p.categories
    assert not p.needs_review


@pytest.mark.parametrize(
    "text",
    [
        "grabfood tối 89k ship 15k",
        "ăn trưa 35k cà phê 25k",
        "mua 100k, 200k",
        "nhận 500k trả 2tr",
    ],
)
def test_several_money_candidates_are_never_forced(text: str) -> None:
    p = propose_value(text)
    assert p.chosen is None
    assert len(p.candidates) >= 2
    assert {"multiple_money_candidates", "multi_number"} <= set(p.categories)
    assert p.needs_review
    proposal = to_quet_proposal("x", p)
    assert proposal["annotation_status"] == "uncertain"
    assert proposal["target"] is None and proposal["note"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("tiền điện tháng 10 612k", "612k"),
        ("tra gop dt thang 3 1tr450", "1tr450"),
        ("hoc phi hoc ky 2 9tr5", "9tr5"),
        ("lương quý 3 15tr", "15tr"),
        ("thu nợ đợt 1 2tr", "2tr"),
        ("cf muối 29k sáng t4", "29k"),
        ("lau vs team 20/10 chia moi nguoi 320k", "320k"),
        ("họp lúc 8:30 mua 100k", "100k"),
        ("nạp thẻ 8h30 50k", "50k"),
        ("tăng 5% lãi 10tr", "10tr"),
        ("cho vay 3tr lai 2%", "3tr"),
        ("mua gạo 10kg 190k", "190k"),
        ("cf highlands vs Tuấn 2 ly 118k", "118k"),
        ("mua 1 chỉ vàng 6tr8", "6tr8"),
        ("lẩu vs team 4 người chia 180k", "180k"),
        ("icloud 50gb 19k", "19k"),
        ("gia han goi 4g 120k", "120k"),
        ("xổ số trúng giải 8 100k", "100k"),
        ("stk 123456789 chuyển 2tr", "2tr"),
        ("năm 2024 mua xe 500tr", "500tr"),
        ("đổ 5 lít xăng 120k", "120k"),
        ("mua 2 củ hành 20k", "20k"),
        ("lương tháng 10 về 18tr5", "18tr5"),
    ],
)
def test_non_money_numbers_are_excluded_but_the_note_is_multi_number(
    text: str, expected: str
) -> None:
    p = propose_value(text)
    assert p.chosen is not None and p.chosen.text == expected
    assert "context_excluded" in p.categories
    # Two numeric expressions: the proposal keeps its span but a human must confirm it.
    assert "multi_number" in p.categories and "multi_number" in p.review_reasons
    assert p.needs_review and not p.auto_accept


def test_notes_with_only_non_money_numbers_have_no_candidate() -> None:
    for text in ("thuong tet 2 thang luong", "trich 20% luong de danh", "mua 2 củ cà rốt"):
        p = propose_value(text)
        assert p.chosen is None and p.needs_review
        assert "no_candidate" in p.categories and "context_excluded" in p.categories
        proposal = to_quet_proposal("x", p)
        assert proposal["type"] == "no_amount" and proposal["target"] is None


def test_a_name_that_looks_like_context_does_not_hide_the_amount() -> None:
    # "Lan" / "Nam" / "Tuấn" are names, not "lần" / "năm" / "tuần".
    assert span_of("cho Lan mượn 500k") == "500k"
    p = propose_value("Tuấn trả 300")
    assert p.chosen is not None and p.chosen.text == "300"
    assert p.review_reasons == ("bare_number",)


def test_bare_number_is_proposed_but_needs_review() -> None:
    p = propose_value("cơm tấm 100")
    assert p.review_reasons == ("bare_number",)
    assert p.confidence < 0.7


def test_unexplained_extra_number_forces_review() -> None:
    p = propose_value("ship 3 đơn 100k")
    assert p.chosen is not None and p.chosen.text == "100k"
    assert "multi_number" in p.review_reasons


def test_compound_amount_is_one_span_and_reviewed() -> None:
    p = propose_value("mua đồ 1 triệu 2")
    assert p.chosen is not None and p.chosen.text == "1 triệu 2"
    assert p.review_reasons == ("compound_amount",)
    assert "multi_number" not in p.categories  # one amount expression
    # a date next to the amount is a second numeric expression, and is never a compound tail
    d = propose_value("vay 2 triệu 20/10")
    assert d.chosen is not None and d.chosen.text == "2 triệu"
    assert "multi_number" in d.categories


@pytest.mark.parametrize(
    "text",
    [
        "tien dien thang 10 620k",
        "ăn 2 tô phở 70",
        "mua 3 vé 150k",
        "trả góp kỳ 3 1tr5",
    ],
)
def test_extra_numbers_make_a_multi_number_note_that_needs_review(text: str) -> None:
    p = propose_value(text)
    assert p.chosen is not None and 0 < p.confidence < 1
    assert "multi_number" in p.categories
    assert p.review_reasons[0] == "multi_number"
    assert p.needs_review and not p.auto_accept


@pytest.mark.parametrize(
    "text",
    [
        "ăn trưa 45k",
        "cf 1tr5",
        "mua 1.500.000",
        "lương 12,5tr",
        "vay 2 triệu rưỡi",
        "họp 20/10",
        "họp 8:30",
        "ship bốn triệu hai trăm",
    ],
)
def test_one_numeric_expression_is_not_multi_number(text: str) -> None:
    # Amounts (1tr5, 1.500.000, 2 triệu rưỡi, 12,5tr), dates and spelled-out numbers count once.
    assert "multi_number" not in propose_value(text).categories


def test_spelled_out_amounts_are_slang_and_skip_names() -> None:
    assert span_of("tiền nhà tháng 10 bốn triệu") == "bốn triệu"
    assert span_of("hai tram nghin mua hoa") == "hai tram nghin"
    p = propose_value("Nam trả ba triệu")
    assert p.chosen is not None and p.chosen.text == "ba triệu"
    assert {"slang", "number_words"} <= set(p.categories)


def test_unusual_punctuation_and_symbols_force_review() -> None:
    for text in ("cơm 50k/tháng", "cơm (45k)", "cơm 😀 45k"):
        p = propose_value(text)
        assert p.chosen is not None
        assert p.review_reasons == ("unusual_punctuation",), text
    assert propose_value("ăn sáng 45k.").auto_accept


@pytest.mark.parametrize("form", ["NFC", "NFD"])
def test_offsets_with_emoji_and_decomposed_accents(form: str) -> None:
    for text, expected in (
        ("😀 mượn chú hai 5 xị 🎉", "5 xị"),
        ("👨‍👩‍👧 tiền điện tháng 10 hết 612", "612"),
        ("trả góp kỳ 3 1tr5 👍", "1tr5"),
    ):
        s = unicodedata.normalize(form, text)
        c = propose_value(s).chosen
        assert c is not None
        assert s[c.start : c.end] == c.text
        assert unicodedata.normalize("NFC", c.text) == expected
        assert not c.text[0].isspace() and not c.text[-1].isspace()


def test_proposal_and_rule_label_pass_the_validators() -> None:
    text = "mượn chú hai 5 xị"
    p = propose_value(text)
    assert validate_quet_proposal(to_quet_proposal("a", p), text, CONFIG) == []
    ok = propose_value("cf 50k")
    label = to_rule_label("a", ok)
    assert label["provenance"] == "rule"
    assert validate_value_label(label, "cf 50k", CONFIG, extra_fields=RULE_LABEL_EXTRA) == []
    with pytest.raises(ValueError):
        to_rule_label("a", p)  # slang needs a human


# ---------------------------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------------------------

TEXT = "cơm 50k"


def label(**changes) -> dict:
    base = {
        "id": "r1",
        "annotation_status": "complete",
        "type": "amount",
        "target": {"text": "50k", "start": 4, "end": 7},
    }
    base.update(changes)
    return base


def test_valid_labels_pass() -> None:
    assert validate_value_label(label(), TEXT, CONFIG) == []
    assert validate_value_label(label(type="no_amount", target=None), TEXT, CONFIG) == []
    uncertain = label(annotation_status="uncertain", target=None, note="two amounts")
    assert validate_value_label(uncertain, TEXT, CONFIG) == []
    skipped = label(annotation_status="skipped", type=None, target=None)
    assert validate_value_label(skipped, TEXT, CONFIG) == []


@pytest.mark.parametrize(
    ("changes", "fragment"),
    [
        ({"target": {"text": "50k", "start": 3, "end": 6}}, "not '50k'"),
        ({"target": {"text": " 50k", "start": 3, "end": 7}}, "whitespace"),
        ({"target": {"text": "50k ", "start": 4, "end": 8}}, "whitespace"),
        ({"target": {"text": "50k", "start": 7, "end": 4}}, "must be less"),
        ({"target": {"text": "50k", "start": 4, "end": 99}}, "beyond the text length"),
        ({"target": {"text": "50k", "start": -1, "end": 2}}, "non-negative"),
        ({"target": {"text": "", "start": 4, "end": 4}}, "non-empty"),
        ({"target": {"text": "50k", "start": 4}}, "missing field 'end'"),
        ({"target": None}, "required for a complete 'amount'"),
        ({"type": "no_amount"}, "must be null for type 'no_amount'"),
        ({"type": "income"}, "is not one of"),
        ({"type": None}, "type: required"),
        ({"annotation_status": "uncertain"}, "note: required"),
        ({"annotation_status": "done"}, "not one of"),
        ({"annotation_status": "skipped", "type": "amount"}, "must be null"),
        ({"extra": 1}, "unknown field"),
    ],
)
def test_validator_rejects_bad_labels(changes: dict, fragment: str) -> None:
    problems = validate_value_label(label(**changes), TEXT, CONFIG)
    assert any(fragment in p for p in problems), problems


def test_validate_span_checks_nfc_and_types() -> None:
    assert validate_span({"text": "50k", "start": 4, "end": 7}, TEXT, "value") == []
    decomposed = unicodedata.normalize("NFD", "đồng")
    problems = validate_span({"text": decomposed, "start": 0, "end": len(decomposed)}, None)
    assert any("NFC" in p for p in problems)
    problems = validate_span({"text": "50k", "start": True, "end": 7}, TEXT)
    assert any("non-negative integer" in p for p in problems)


def record(**changes) -> dict:
    base = {
        "id": "r1",
        "text": TEXT,
        "type": "expense",
        "target": None,
        "value": {"text": "50k", "start": 4, "end": 7},
        "value_status": "complete",
        "value_provenance": "rule",
    }
    base.update(changes)
    return base


def test_merged_record_validation() -> None:
    assert validate_merged_record(record(), CONFIG) == []
    assert validate_merged_record(record(value=None), CONFIG) == []  # complete no_amount
    masked = record(value=None, value_status="uncertain")
    assert validate_merged_record(masked, CONFIG) == []
    bad = {
        "value with status uncertain": record(value_status="uncertain"),
        "bad status": record(value_status="maybe"),
        "bad provenance": record(value_provenance="ai"),
        "wrong offsets": record(value={"text": "50k", "start": 0, "end": 3}),
        "overlaps target": record(target={"text": "m 50", "start": 2, "end": 6}, text=TEXT),
        "missing field": {k: v for k, v in record().items() if k != "value_provenance"},
    }
    for name, rec in bad.items():
        assert validate_merged_record(rec, CONFIG), name


def test_human_label_beats_rule_and_unlabelled_is_masked() -> None:
    rule = {
        "annotation_status": "complete",
        "type": "amount",
        "target": {"text": "50k", "start": 4, "end": 7},
    }
    human = {
        "annotation_status": "complete",
        "type": "amount",
        "target": {"text": "7", "start": 1, "end": 2},
    }
    assert merged_value_fields(human, rule)[0]["text"] == "7"
    assert merged_value_fields(human, rule)[2] == "human"
    assert merged_value_fields(None, rule) == (rule["target"], "complete", "rule", "rule")
    assert merged_value_fields(None, None) == (None, "uncertain", "rule", "unlabelled")
    no_amount = {"annotation_status": "complete", "type": "no_amount", "target": None}
    assert merged_value_fields(no_amount, rule) == (None, "complete", "human", "human")
    unsure = {"annotation_status": "uncertain", "type": None, "target": None, "note": "?"}
    assert merged_value_fields(unsure, rule)[:3] == (None, "uncertain", "human")


def test_label_file_reports_duplicates_unknown_ids_and_bad_spans(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    rows = [
        label(),
        label(),  # duplicate
        label(id="ghost"),  # not in the queue
        label(id="r2", target={"text": "x", "start": 0, "end": 1}),  # wrong slice
    ]
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")
    report = validate_label_file(path, {"r1": TEXT, "r2": TEXT}, CONFIG)
    text = "\n".join(report.errors)
    assert not report.ok
    assert "duplicate id" in text and "not in the queue" in text and "not 'x'" in text


# ---------------------------------------------------------------------------------------------
# scripts/build_training_v2.py on a tiny fixture tree
# ---------------------------------------------------------------------------------------------

SPEC = importlib.util.spec_from_file_location(
    "build_training_v2", ROOT / "scripts" / "build_training_v2.py"
)
bt = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bt)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")


def v1_row(rid: str, text: str, type_: str = "expense") -> dict:
    return {
        "id": rid,
        "text": text,
        "type": type_,
        "target": None,
        "source_batch": "b",
        "accented": any(ord(c) > 127 for c in text),
        "provenance": {"annotator": "human"},
    }


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    (tmp_path / "configs").mkdir()
    shutil.copy(ROOT / "configs" / "annotation-v2.yaml", tmp_path / "configs")
    v1 = tmp_path / "datasets" / "annotation-v1" / "distillation-v1"
    rows = {
        "train": [
            v1_row("t1", "cf 50k"),
            v1_row("t2", "mượn chú hai 5 xị"),
            v1_row("t3", "cơm tấm 100"),
        ],
        "validation": [v1_row("v1", "bún 45k")],
        "test": [v1_row("e1", "grab 30k")],
    }
    rows["train"].append(rows["validation"][0])
    for split, data in rows.items():
        write_jsonl(v1 / f"{split}.jsonl", data)
    probe = tmp_path / "datasets" / "probe-v1"
    write_jsonl(probe / "notes.jsonl", [{"id": "probe-v1-a", "text": "nạp 2tr", "pattern": "x"}])
    write_jsonl(
        probe / "labels.jsonl",
        [{"id": "probe-v1-a", "annotation_status": "complete", "type": "expense", "target": None}],
    )
    write_jsonl(probe / "provenance.jsonl", [{"id": "probe-v1-a", "annotator": "ai"}])

    all_rows = (
        rows["train"][:3]
        + rows["validation"]
        + rows["test"]
        + [{"id": "probe-v1-a", "text": "nạp 2tr"}]
    )
    v2 = tmp_path / V2
    write_jsonl(v2 / "value-queue.jsonl", [{"id": r["id"], "text": r["text"]} for r in all_rows])
    write_jsonl(
        v2 / "value-auto-labels.jsonl",
        [
            to_rule_label(r["id"], p)
            for r in all_rows
            if (p := propose_value(r["text"])).auto_accept
        ],
    )
    write_review(tmp_path)
    score_tree(tmp_path)
    return tmp_path


def write_review(tree: Path, unsampled: tuple[str, ...] = ()) -> None:
    """Review files of the tree: flagged records are must-review, except ``unsampled`` ones,
    which stay outside the queue in the ``slang_xi`` stratum with their rule proposal."""
    v2 = tree / V2
    queue, provenance, proposals = [], [], []
    for r in read_jsonl(v2 / "value-queue.jsonl"):
        p = propose_value(r["text"])
        proposals.append(to_quet_proposal(r["id"], p))
        if p.auto_accept or r["id"] in unsampled:
            stratum = "clean" if p.auto_accept else "slang_xi"
            row = {"review_group": "auto_accept", "stratum": stratum, "provenance": "rule"}
            provenance.append({"id": r["id"], **row, "status": vr.STATUS_PENDING})
            continue
        reasons = list(p.review_reasons)
        row = {"review_group": "must_review", "stratum": reasons[0], "reasons": reasons}
        queue.append({"id": r["id"], "text": r["text"], **row, "split": "train", "source": "x"})
        provenance.append(
            {"id": r["id"], **{k: row[k] for k in ("review_group", "stratum")}} | {"status": "q"}
        )
    write_jsonl(v2 / "value-proposals.jsonl", proposals)
    write_jsonl(v2 / "value-review-queue.jsonl", queue)
    write_jsonl(v2 / "value-review-provenance.jsonl", provenance)


def score_tree(tree: Path, verdicts: dict[str, str] | None = None) -> None:
    """A score entry (as ``scripts/score_value_review.py`` writes it) for the current files."""
    v2 = tree / V2
    names = ("value-review-queue", "value-review-provenance", "value-proposals", "value-labels")
    inputs = {
        f"{V2}/{n}.jsonl": gate.sha256_file(v2 / f"{n}.jsonl")
        for n in names
        if (v2 / f"{n}.jsonl").exists()
    }
    entry = {"strata_status": verdicts or {"clean": "passed"}, "inputs": inputs}
    path = tree / gate.SCORE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"datasets": {vr.NAME: entry}}), encoding="utf-8")


def test_training_build_masks_unreviewed_and_keeps_v1_ids(tree: Path) -> None:
    args = bt.argparse.Namespace(extra=None, extra_name=None, extra_fractions="0.7,0.15,0.15")
    contents, manifest = bt.build(tree, args)
    train = [json.loads(line) for line in contents["train.jsonl"].decode().splitlines()]
    assert [r["id"] for r in train] == ["t1", "t2", "t3", "v1"]
    by_id = {r["id"]: r for r in train}
    assert by_id["t1"]["value"]["text"] == "50k" and by_id["t1"]["value_provenance"] == "rule"
    assert by_id["t2"]["value"] is None and by_id["t2"]["value_status"] == "uncertain"
    assert by_id["t3"]["value_status"] == "uncertain"
    assert manifest["awaiting_human_review_masked"]["count"] == 2
    assert "probe-v1-a" in contents["probe-v1-eval-only.jsonl"].decode()
    assert "probe-v1-a" not in contents["train.jsonl"].decode()


def test_training_build_applies_human_labels(tree: Path) -> None:
    human = [
        {
            "id": "t2",
            "annotation_status": "complete",
            "type": "amount",
            "target": {"text": "5 xị", "start": 13, "end": 17},
        },
        {
            "id": "t3",
            "annotation_status": "complete",
            "type": "amount",
            "target": {"text": "100", "start": 8, "end": 11},
        },
    ]
    write_jsonl(tree / V2 / "value-labels.jsonl", human)
    score_tree(tree)
    args = bt.argparse.Namespace(extra=None, extra_name=None, extra_fractions="0.7,0.15,0.15")
    contents, manifest = bt.build(tree, args)
    train = {
        json.loads(line)["id"]: json.loads(line)
        for line in contents["train.jsonl"].decode().splitlines()
    }
    assert train["t2"]["value"]["text"] == "5 xị" and train["t2"]["value_provenance"] == "human"
    assert train["t3"]["value_status"] == "complete"
    assert manifest["awaiting_human_review_masked"]["count"] == 0
    assert manifest["human_value_labels"]["labels"] == 2


def test_training_build_refuses_invalid_human_labels(tree: Path) -> None:
    bad = [
        {
            "id": "t2",
            "annotation_status": "complete",
            "type": "amount",
            "target": {"text": "5 xị", "start": 0, "end": 4},
        }
    ]
    write_jsonl(tree / V2 / "value-labels.jsonl", bad)
    score_tree(tree)
    args = bt.argparse.Namespace(extra=None, extra_name=None, extra_fractions="0.7,0.15,0.15")
    with pytest.raises(SystemExit):
        bt.build(tree, args)


def build_args():
    return bt.argparse.Namespace(extra=None, extra_name=None, extra_fractions="0.7,0.15,0.15")


def test_training_build_refuses_without_a_fresh_score(tree: Path, capsys) -> None:
    (tree / gate.SCORE_PATH).unlink()
    with pytest.raises(SystemExit):
        bt.build(tree, build_args())
    assert "is missing" in capsys.readouterr().err
    labels = tree / V2 / "value-labels.jsonl"
    write_jsonl(labels, [])
    score_tree(tree)
    bt.build(tree, build_args())  # fresh
    late = {
        "id": "t3",
        "annotation_status": "complete",
        "type": "amount",
        "target": {"text": "100", "start": 8, "end": 11},
    }
    write_jsonl(labels, [late])  # a human label arrived after scoring
    with pytest.raises(SystemExit):
        bt.build(tree, build_args())
    assert "stale" in capsys.readouterr().err


def rows_by_id(contents: dict[str, bytes]) -> dict[str, dict]:
    rows = (json.loads(line) for line in contents["train.jsonl"].decode().splitlines())
    return {r["id"]: r for r in rows}


def test_unsampled_flagged_record_trains_from_its_proposal_only_if_its_stratum_passed(
    tree: Path,
) -> None:
    write_review(tree, unsampled=("t2",))  # t2 "mượn chú hai 5 xị": flagged, outside the queue
    score_tree(tree, {"clean": "passed", "slang_xi": "passed"})
    contents, manifest = bt.build(tree, build_args())
    t2 = rows_by_id(contents)["t2"]
    assert t2["value"]["text"] == "5 xị" and t2["value_provenance"] == "rule"
    assert t2["value_status"] == "complete"
    assert manifest["stratum_unverified_masked"]["count"] == 0
    assert manifest["awaiting_human_review_masked"]["count"] == 1  # t3 is still queued

    score_tree(tree, {"clean": "passed", "slang_xi": "failed"})
    contents, manifest = bt.build(tree, build_args())
    t2 = rows_by_id(contents)["t2"]
    assert t2["value"] is None and t2["value_status"] == "uncertain"
    assert t2["type"] == "expense"  # type/target still train; only the value loss is masked
    assert manifest["stratum_unverified_masked"]["by_stratum"] == {"slang_xi": 1}
    assert rows_by_id(contents)["t1"]["value_provenance"] == "rule"

    score_tree(tree, {"clean": "passed"})  # a stratum the score does not mention is not trusted
    _, manifest = bt.build(tree, build_args())
    assert manifest["stratum_unverified_masked"]["by_stratum"] == {"slang_xi": 1}


def test_a_failed_clean_audit_masks_the_clean_records_and_a_human_label_still_wins(
    tree: Path,
) -> None:
    score_tree(tree, {"clean": "failed"})
    contents, manifest = bt.build(tree, build_args())
    train = rows_by_id(contents)
    assert train["t1"]["value"] is None and train["t1"]["value_status"] == "uncertain"
    assert manifest["stratum_unverified_masked"]["by_stratum"] == {"clean": 5}
    human = {
        "id": "t1",
        "annotation_status": "complete",
        "type": "amount",
        "target": {"text": "50k", "start": 3, "end": 6},
    }
    write_jsonl(tree / V2 / "value-labels.jsonl", [human])
    score_tree(tree, {"clean": "failed"})
    contents, _ = bt.build(tree, build_args())
    t1 = rows_by_id(contents)["t1"]
    assert t1["value_provenance"] == "human" and t1["value"]["text"] == "50k"


def test_group_split_never_separates_a_group() -> None:
    groups = [f"g{i % 20}" for i in range(100)]
    assignment = bt.split_groups(groups, (0.7, 0.15, 0.15))
    assert set(assignment) == set(groups)
    assert sorted(set(assignment.values())) == ["test", "train", "validation"]
    assert assignment == bt.split_groups(list(reversed(groups)), (0.7, 0.15, 0.15))
