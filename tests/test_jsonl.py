import pytest

from gidi.corpus.jsonl import read_jsonl, write_jsonl


def test_roundtrip_preserves_vietnamese_unescaped(tmp_path):
    path = tmp_path / "raw" / "notes.jsonl"
    records = [{"text": "ăn phở 45k"}, {"text": "nhận lương 15tr", "source": "synthetic"}]
    assert write_jsonl(path, records) == 2
    assert read_jsonl(path) == records
    assert "phở" in path.read_text(encoding="utf-8")


def test_refuses_to_overwrite_existing_file(tmp_path):
    path = tmp_path / "notes.jsonl"
    write_jsonl(path, [{"text": "gốc"}])
    with pytest.raises(FileExistsError):
        write_jsonl(path, [{"text": "mới"}])
    assert read_jsonl(path) == [{"text": "gốc"}]
    assert list(tmp_path.iterdir()) == [path]


def test_explicit_overwrite_replaces(tmp_path):
    path = tmp_path / "notes.jsonl"
    write_jsonl(path, [{"text": "gốc"}])
    write_jsonl(path, [{"text": "mới"}], overwrite=True)
    assert read_jsonl(path) == [{"text": "mới"}]


def test_skips_blank_lines(tmp_path):
    path = tmp_path / "notes.jsonl"
    path.write_text('{"text": "a"}\n\n  \n{"text": "b"}\n', encoding="utf-8")
    assert read_jsonl(path) == [{"text": "a"}, {"text": "b"}]


@pytest.mark.parametrize("bad_line", ["{not json", "[1, 2]"])
def test_bad_line_reports_line_number(tmp_path, bad_line):
    path = tmp_path / "notes.jsonl"
    path.write_text('{"text": "a"}\n' + bad_line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"notes\.jsonl:2:"):
        read_jsonl(path)
