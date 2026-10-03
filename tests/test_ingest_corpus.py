import importlib.util
import json
import subprocess
import sys
import unicodedata
from pathlib import Path

import pytest

from gidi.corpus.ingest import (
    derive_id,
    ingest_file,
    merge_reports,
    render_report,
)
from gidi.corpus.jsonl import read_jsonl
from gidi.corpus.schema import DuplicateGroup, validate_file

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "ingest_generated_corpus.py"

SOURCE = "claude"
PROMPT = "baseline-01"


@pytest.fixture(scope="module")
def cli():
    spec = importlib.util.spec_from_file_location("ingest_generated_corpus_script", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def jline(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def write_text(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def write_lines(path: Path, lines: list[str]) -> Path:
    return write_text(path, "".join(f"{line}\n" for line in lines))


def write_bytes(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


def test_derive_id_is_deterministic_and_content_addressed():
    first = derive_id(source=SOURCE, prompt_id=PROMPT, text="ăn phở 45k")
    assert first == derive_id(source=SOURCE, prompt_id=PROMPT, text="ăn phở 45k")
    assert first.startswith(f"{PROMPT}-")
    assert len(first) == len(f"{PROMPT}-") + 12
    assert derive_id(source="other", prompt_id=PROMPT, text="ăn phở 45k") != first
    assert derive_id(source=SOURCE, prompt_id="baseline-02", text="ăn phở 45k") != first
    assert derive_id(source=SOURCE, prompt_id=PROMPT, text="cf 30k") != first
    assert (
        derive_id(source=SOURCE, prompt_id=PROMPT, text="ăn phở 45k", occurrence=2) == f"{first}-d2"
    )
    assert derive_id(source=SOURCE, prompt_id=PROMPT, text="x", occurrence=7).endswith("-d7")


def test_duplicate_texts_get_unique_ids_and_input_fields_are_dropped(tmp_path):
    path = write_lines(
        tmp_path / "gen.jsonl",
        [
            jline({"id": "provided", "text": "cf 30k", "amount": 30000}),
            jline({"text": "cf 30k", "tags": ["x"], "source": "model"}),
            jline({"text": "ăn phở 45k"}),
        ],
    )
    report = ingest_file(path, source=SOURCE, prompt_id=PROMPT)

    assert report.ok
    base = derive_id(source=SOURCE, prompt_id=PROMPT, text="cf 30k")
    ids = [record["id"] for record in report.records]
    assert ids == [
        base,
        f"{base}-d2",
        derive_id(source=SOURCE, prompt_id=PROMPT, text="ăn phở 45k"),
    ]
    assert len(set(ids)) == 3
    assert [record["text"] for record in report.records] == ["cf 30k", "cf 30k", "ăn phở 45k"]

    for record in report.records:
        assert list(record) == ["id", "text", "source", "prompt_id"]
        assert record["source"] == SOURCE
        assert record["prompt_id"] == PROMPT

    assert report.dropped_id_lines == [1]
    assert report.dropped_keys == {"amount": 1, "tags": 1, "source": 1}
    assert report.duplicate_texts == [
        DuplicateGroup("cf 30k", (1, 2), (base, f"{base}-d2")),
    ]
    warnings = [issue for issue in report.warnings if "duplicate text" in issue.message]
    assert len(warnings) == 1
    assert warnings[0].line == 1
    assert base in warnings[0].message
    assert f"{base}-d2" in warnings[0].message


def test_nfd_text_is_normalized_to_nfc_with_a_warning(tmp_path):
    nfd = unicodedata.normalize("NFD", "ăn phở 45k")
    assert nfd != unicodedata.normalize("NFC", nfd)
    path = write_lines(tmp_path / "gen.jsonl", [jline({"text": nfd})])

    report = ingest_file(path, source=SOURCE, prompt_id=PROMPT)

    assert report.ok
    final = unicodedata.normalize("NFC", nfd)
    assert report.records[0]["text"] == final
    assert report.normalized_lines == [1]
    assert report.records[0]["id"] == derive_id(source=SOURCE, prompt_id=PROMPT, text=final)
    assert any("normalized to NFC" in issue.message for issue in report.warnings)


def test_jsonl_padded_text_is_an_error_with_no_record(tmp_path):
    path = write_lines(tmp_path / "gen.jsonl", [jline({"text": " cf 30k"})])
    report = ingest_file(path, source=SOURCE, prompt_id=PROMPT)
    assert not report.ok
    assert report.records == []
    assert "leading or trailing whitespace" in report.errors[0].message
    assert "' cf 30k'" in report.errors[0].message


def test_text_format_padded_line_is_an_error_with_no_record(tmp_path):
    path = write_bytes(tmp_path / "gen.txt", b" cf 30k\n")
    report = ingest_file(path, source=SOURCE, prompt_id=PROMPT, input_format="text")
    assert not report.ok
    assert report.records == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", "text: must not be empty"),
        ("a\nb", "text must not contain '\\n'"),
        ("a\tb", "text must not contain '\\t'"),
    ],
)
def test_jsonl_empty_or_multiline_text_is_an_error(tmp_path, text, expected):
    path = write_lines(tmp_path / "gen.jsonl", [jline({"text": text})])
    report = ingest_file(path, source=SOURCE, prompt_id=PROMPT)
    assert not report.ok
    assert report.records == []
    assert any(expected in issue.message for issue in report.errors)


def test_jsonl_missing_and_non_string_text_are_errors(tmp_path):
    path = write_lines(
        tmp_path / "gen.jsonl",
        [jline({"prompt_id": "p"}), jline({"text": 42})],
    )
    report = ingest_file(path, source=SOURCE, prompt_id=PROMPT)
    assert [issue.message for issue in report.errors] == ["text: missing", "text: must be a string"]


def test_invalid_utf8_line_is_reported_with_byte_offset(tmp_path):
    path = write_bytes(tmp_path / "gen.jsonl", b'{"text": "\xff bad"}\n{"text": "ok"}\n')
    report = ingest_file(path, source=SOURCE, prompt_id=PROMPT)
    assert not report.ok
    assert report.n_lines == 2
    assert report.errors[0].line == 1
    assert "invalid UTF-8 at byte offset 10" in report.errors[0].message
    assert [record["text"] for record in report.records] == ["ok"]


def test_jsonl_malformed_and_non_object_lines_report_their_numbers(tmp_path):
    path = write_lines(
        tmp_path / "gen.jsonl",
        [jline({"text": "a"}), "{oops", "[1, 2]", "42"],
    )
    report = ingest_file(path, source=SOURCE, prompt_id=PROMPT)
    assert [(issue.line, issue.message) for issue in report.errors] == [
        (2, "invalid JSON: Expecting property name enclosed in double quotes"),
        (3, "record: expected a JSON object"),
        (4, "record: expected a JSON object"),
    ]


def test_text_format_skips_blanks_and_warns_on_whitespace_only_line(tmp_path):
    path = write_bytes(tmp_path / "gen.txt", "ăn phở 45k\n\n   \ncf 30k\r\n".encode())
    report = ingest_file(path, source=SOURCE, prompt_id=PROMPT, input_format="text")

    assert report.ok
    assert report.input_format == "text"
    assert report.n_lines == 4
    assert [record["text"] for record in report.records] == ["ăn phở 45k", "cf 30k"]
    whitespace = [issue for issue in report.warnings if "whitespace-only" in issue.message]
    assert [issue.line for issue in whitespace] == [3]


def test_auto_format_detection(tmp_path):
    as_jsonl = write_lines(tmp_path / "a.jsonl", ["", jline({"text": "cf 30k"})])
    as_text = write_bytes(tmp_path / "b.txt", "ăn phở 45k\n".encode())

    jsonl_report = ingest_file(as_jsonl, source=SOURCE, prompt_id=PROMPT)
    text_report = ingest_file(as_text, source=SOURCE, prompt_id=PROMPT)

    assert jsonl_report.input_format == "jsonl"
    assert jsonl_report.ok
    assert text_report.input_format == "text"
    assert text_report.ok
    assert jsonl_report.records[0]["text"] == "cf 30k"
    assert text_report.records[0]["text"] == "ăn phở 45k"


def test_render_report_includes_findings_and_dropped_keys(tmp_path):
    path = write_lines(
        tmp_path / "gen.jsonl",
        [
            jline({"text": "cf 30k", "amount": 30000}),
            jline({"text": "cf 30k", "amount": 30000, "tags": ["a"]}),
        ],
    )
    report = ingest_file(path, source=SOURCE, prompt_id=PROMPT)
    assert report.dropped_keys == {"amount": 2, "tags": 1}

    text = render_report(report)
    assert "jsonl input, 2 line(s), 2 record(s), 0 error(s), 4 warning(s)" in text
    assert "warning line 1" in text and "duplicate text 'cf 30k' at lines 1, 2" in text
    assert "dropped input keys: amount x2, tags x1" in text


def test_merge_reports_concatenates_records_and_dropped_keys(tmp_path):
    one = ingest_file(
        write_lines(tmp_path / "one.jsonl", [jline({"text": "a", "amount": 1})]),
        source=SOURCE,
        prompt_id=PROMPT,
    )
    two = ingest_file(
        write_lines(tmp_path / "two.jsonl", [jline({"text": "b", "amount": 2})]),
        source=SOURCE,
        prompt_id=PROMPT,
    )
    merged = merge_reports([one, two])
    assert merged.ok
    assert [record["text"] for record in merged.records] == ["a", "b"]
    assert merged.n_lines == 2
    assert merged.n_records == 2
    assert merged.dropped_keys == {"amount": 2}
    assert merged.sources == [SOURCE]


def test_cli_end_to_end_writes_valid_corpus_and_is_stable(tmp_path, capsys, cli):
    src = write_lines(
        tmp_path / "gen.jsonl",
        [
            jline({"id": "x", "text": "ăn phở 45k", "amount": 45000}),
            jline({"text": "cf 30k"}),
            jline({"text": "cf 30k"}),
        ],
    )
    out_first = tmp_path / "raw" / "baseline-01.jsonl"
    out_second = tmp_path / "raw" / "again.jsonl"
    base_argv = [str(src), "--source", SOURCE, "--prompt-id", PROMPT]

    code = cli.main([*base_argv, "--out", str(out_first)])
    captured = capsys.readouterr()
    assert code == 0
    assert "wrote 3 record(s)" in captured.out
    assert "validate with: uv run python scripts/validate_corpus.py" in captured.out
    assert validate_file(out_first).ok

    assert cli.main([*base_argv, "--out", str(out_second)]) == 0
    assert read_jsonl(out_first) == read_jsonl(out_second)


def test_cli_error_writes_no_output_file(tmp_path, capsys, cli):
    src = write_lines(
        tmp_path / "gen.jsonl",
        [jline({"text": "ok"}), jline({"text": " padded "})],
    )
    out = tmp_path / "out.jsonl"

    code = cli.main([str(src), "--source", SOURCE, "--prompt-id", PROMPT, "--out", str(out)])

    captured = capsys.readouterr()
    assert code == 1
    assert not out.exists()
    assert "no output written" in captured.err


def test_cli_refuses_existing_out_then_allows_overwrite(tmp_path, capsys, cli):
    src = write_lines(tmp_path / "gen.jsonl", [jline({"text": "cf 30k"})])
    out = write_text(tmp_path / "out.jsonl", "existing\n")
    argv = [str(src), "--source", SOURCE, "--prompt-id", PROMPT, "--out", str(out)]

    assert cli.main(argv) == 1
    assert "refusing to overwrite existing file" in capsys.readouterr().err
    assert out.read_text(encoding="utf-8") == "existing\n"

    assert cli.main([*argv, "--overwrite"]) == 0
    assert validate_file(out).ok
    assert [record["text"] for record in read_jsonl(out)] == ["cf 30k"]


def test_cli_refuses_out_equal_to_input(tmp_path, capsys, cli):
    src = write_lines(tmp_path / "gen.jsonl", [jline({"text": "cf 30k"})])
    code = cli.main(
        [str(src), "--source", SOURCE, "--prompt-id", PROMPT, "--out", str(src), "--overwrite"]
    )
    assert code == 1
    assert "refusing" in capsys.readouterr().err
    assert [record["text"] for record in read_jsonl(src)] == ["cf 30k"]


@pytest.mark.parametrize(
    ("flag", "value"),
    [("--source", "bad slug"), ("--source", ".leading"), ("--prompt-id", "x/y")],
)
def test_cli_rejects_invalid_slug(tmp_path, capsys, cli, flag, value):
    src = write_lines(tmp_path / "gen.jsonl", [jline({"text": "cf 30k"})])
    argv = [
        str(src),
        "--source",
        SOURCE,
        "--prompt-id",
        PROMPT,
        "--out",
        str(tmp_path / "out.jsonl"),
    ]
    argv[argv.index(flag) + 1] = value

    with pytest.raises(SystemExit) as excinfo:
        cli.main(argv)
    assert excinfo.value.code == 2
    assert "invalid slug" in capsys.readouterr().err


def test_script_runs_as_subprocess(tmp_path):
    src = write_lines(tmp_path / "gen.jsonl", [jline({"text": "ăn phở 45k"})])
    out = tmp_path / "raw" / "baseline-01.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(src),
            "--source",
            SOURCE,
            "--prompt-id",
            PROMPT,
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )

    assert result.returncode == 0
    assert "wrote 1 record(s)" in result.stdout
    assert validate_file(out).ok
