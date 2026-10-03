import importlib.util
import json
import subprocess
import sys
import unicodedata
from pathlib import Path

import pytest

from gidi.corpus.schema import (
    DuplicateGroup,
    Issue,
    render_report,
    validate_file,
    validate_record,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "validate_corpus.py"


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("validate_corpus_script", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_lines(path: Path, lines: list[str]) -> Path:
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return path


def rec(record_id: str, text: str, source: str = "synthetic:test", **extra) -> str:
    return json.dumps(
        {"id": record_id, "text": text, "source": source, **extra}, ensure_ascii=False
    )


def test_valid_record_has_no_problems():
    record = {
        "id": "r1",
        "text": unicodedata.normalize("NFC", "ăn phở 45k"),
        "source": "synthetic:test",
        "prompt_id": "raw-finance-notes-v1",
        "extra_key": "preserved and ignored",
    }
    assert validate_record(record) == []


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        ({"text": "ăn phở", "source": "manual"}, "id: missing"),
        ({"id": "r1", "source": "manual"}, "text: missing"),
        ({"id": "r1", "text": "ăn phở"}, "source: missing"),
        ({"id": "", "text": "ăn phở", "source": "manual"}, "id: must not be empty"),
        ({"id": "r1", "text": "", "source": "manual"}, "text: must not be empty"),
        ({"id": "r1", "text": "ăn phở", "source": ""}, "source: must not be empty"),
        ({"id": 1, "text": "ăn phở", "source": "manual"}, "id: must be a string"),
        ({"id": "r1", "text": 45, "source": "manual"}, "text: must be a string"),
        ({"id": "r1", "text": "ăn phở", "source": None}, "source: must be a string"),
        (
            {"id": "r1", "text": "ăn phở", "source": "manual", "prompt_id": 7},
            "prompt_id: must be a string",
        ),
        ({"id": "r1", "text": "   ", "source": "manual"}, "leading or trailing whitespace"),
        ({"id": "r1", "text": " ăn phở 45k", "source": "manual"}, "leading or trailing"),
        ({"id": "r1", "text": "ăn phở 45k ", "source": "manual"}, "leading or trailing"),
        ({"id": "r1", "text": "ăn phở\n45k", "source": "manual"}, "must not contain newlines"),
        ({"id": "r1", "text": "ăn phở\r45k", "source": "manual"}, "must not contain newlines"),
        (
            {
                "id": "r1",
                "text": unicodedata.normalize("NFD", "ăn phở 45k"),
                "source": "manual",
            },
            "must be Unicode NFC",
        ),
        ("not an object", "record: expected a JSON object"),
    ],
)
def test_contract_violations(record, expected):
    problems = validate_record(record)
    assert any(expected in problem for problem in problems), problems


def test_nfd_and_nfc_text_differ_only_by_normalization():
    nfd = unicodedata.normalize("NFD", "ăn phở 45k")
    nfc = unicodedata.normalize("NFC", "ăn phở 45k")
    assert nfd != nfc
    assert validate_record({"id": "r", "text": nfc, "source": "manual"}) == []
    assert validate_record({"id": "r", "text": nfd, "source": "manual"}) == [
        "text: must be Unicode NFC"
    ]


def test_all_problems_reported_together():
    problems = validate_record({"text": "", "prompt_id": 3})
    assert problems == [
        "id: missing",
        "text: must not be empty",
        "source: missing",
        "prompt_id: must be a string",
    ]


def test_clean_file_reports_no_findings(tmp_path):
    path = write_lines(
        tmp_path / "clean.jsonl", [rec("r1", "ăn phở 45k"), rec("r2", "nhận lương 15tr")]
    )
    report = validate_file(path)

    assert report.n_records == 2
    assert report.errors == []
    assert report.warnings == []
    assert report.duplicate_ids == []
    assert report.duplicate_texts == []
    assert report.ok


def test_duplicate_id_is_an_error_with_both_lines(tmp_path):
    path = write_lines(
        tmp_path / "dupes.jsonl",
        [rec("r1", "ăn phở 45k"), rec("r2", "gửi xe 5k"), rec("r1", "cf 30k")],
    )
    report = validate_file(path)

    assert not report.ok
    assert report.errors == [Issue(3, "r1", "duplicate id 'r1' (first seen at line 1)")]
    assert report.errors[0] == (3, "r1", "duplicate id 'r1' (first seen at line 1)")
    assert report.duplicate_ids == [DuplicateGroup("r1", (1, 3), ("r1", "r1"))]
    assert report.warnings == []


def test_duplicate_text_is_a_warning_not_an_error(tmp_path):
    path = write_lines(tmp_path / "same_text.jsonl", [rec("r1", "cf 30k"), rec("r2", "cf 30k")])
    report = validate_file(path)

    assert report.ok
    assert report.errors == []
    assert len(report.warnings) == 1
    line, record_id, message = report.warnings[0]
    assert (line, record_id) == (2, "r2")
    assert "'r1'" in message and "line 1" in message and "cf 30k" in message
    assert report.duplicate_texts == [DuplicateGroup("cf 30k", (1, 2), ("r1", "r2"))]


def test_blank_lines_skipped_but_line_numbers_are_physical(tmp_path):
    path = write_lines(
        tmp_path / "gaps.jsonl",
        [rec("r1", "ăn phở 45k"), "", "   ", json.dumps({"id": "r2", "text": "gửi xe 5k"}), ""],
    )
    report = validate_file(path)

    assert report.n_records == 2
    assert report.errors == [Issue(4, "r2", "source: missing")]


def test_bad_json_reports_line_number_instead_of_raising(tmp_path):
    path = write_lines(
        tmp_path / "broken.jsonl",
        [rec("r1", "ăn phở 45k"), "", "{not json", "[1, 2]"],
    )
    report = validate_file(path)

    assert report.n_records == 1
    assert report.errors[0].line == 3
    assert report.errors[0].record_id is None
    assert "invalid JSON" in report.errors[0].message
    assert report.errors[1] == Issue(4, None, "record: expected a JSON object")
    assert not report.ok


def test_findings_are_ordered_by_line(tmp_path):
    path = write_lines(
        tmp_path / "many.jsonl",
        [
            rec("r1", "text with trailing space "),
            rec("r2", "ok 5k"),
            rec("r2", "also bad\nline"),
        ],
    )
    report = validate_file(path)

    assert [issue.line for issue in report.errors] == [1, 3, 3]
    assert [issue.line for issue in report.warnings] == []
    assert report.errors[0] == Issue(1, "r1", "text: must not have leading or trailing whitespace")
    assert {issue.message for issue in report.errors[1:]} == {
        "duplicate id 'r2' (first seen at line 2)",
        "text: must not contain newlines",
    }


def test_render_report_lists_summary_and_findings(tmp_path):
    path = write_lines(
        tmp_path / "mixed.jsonl", [rec("r1", "cf 30k"), rec("r2", "cf 30k"), rec("r3", "")]
    )
    text = render_report(validate_file(path))

    assert f"{path}: 3 record(s), 1 error(s), 1 warning(s)" in text
    assert "error line 3 [r3]: text: must not be empty" in text
    assert "warning line 2 [r2]: duplicate text 'cf 30k'" in text


def test_main_exits_zero_for_clean_file(tmp_path, capsys, cli):
    path = write_lines(tmp_path / "clean.jsonl", [rec("r1", "ăn phở 45k")])

    assert cli.main([str(path)]) == 0
    out = capsys.readouterr().out
    assert "1 record(s), 0 error(s), 0 warning(s)" in out


def test_main_exits_zero_for_warnings_only(tmp_path, capsys, cli):
    path = write_lines(tmp_path / "warn.jsonl", [rec("r1", "cf 30k"), rec("r2", "cf 30k")])

    assert cli.main([str(path)]) == 0
    assert "1 warning(s)" in capsys.readouterr().out


def test_main_exits_one_for_errors_and_covers_every_path(tmp_path, capsys, cli):
    good = write_lines(tmp_path / "good.jsonl", [rec("r1", "ăn phở 45k")])
    bad = write_lines(tmp_path / "bad.jsonl", [rec("r2", "ăn phở 45k"), "oops"])

    assert cli.main([str(good), str(bad)]) == 1
    out = capsys.readouterr().out
    assert str(good) in out and str(bad) in out
    assert "error line 2 [-]: invalid JSON" in out


def test_main_reports_unreadable_path(tmp_path, capsys, cli):
    missing = tmp_path / "nope.jsonl"

    assert cli.main([str(missing)]) == 1
    captured = capsys.readouterr()
    assert "cannot read" in captured.err
    assert captured.out == ""


def test_script_runs_as_subprocess(tmp_path):
    good = write_lines(tmp_path / "good.jsonl", [rec("r1", "ăn phở 45k")])
    bad = write_lines(tmp_path / "bad.jsonl", [rec("r2", "ăn phở 45k"), "{oops"])

    ok = subprocess.run(
        [sys.executable, str(SCRIPT), str(good)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert ok.returncode == 0
    assert "0 error(s)" in ok.stdout

    failed = subprocess.run(
        [sys.executable, str(SCRIPT), str(bad)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert failed.returncode == 1
    assert "error line 2" in failed.stdout
