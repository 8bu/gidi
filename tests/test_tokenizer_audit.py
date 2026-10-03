"""Offline tests for the tokenizer audit.

Every tokenizer here is built in memory with the ``tokenizers`` library, so the suite needs no
network access and no downloads. Metrics are asserted against hand-computed values.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from gidi.corpus.jsonl import write_jsonl
from gidi.tokenizer.audit import AMOUNT_PATTERN, audit_tokenizer, load_texts

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_tokenizers.py"

WORDLEVEL_VOCAB = {"[UNK]": 0, "ăn": 1, "phở": 2, "45k": 3}
WORDPIECE_VOCAB = {
    "[UNK]": 0,
    "ăn": 1,
    "ph": 2,
    "##ở": 3,
    "45k": 4,
    "nhận": 5,
    "lương": 6,
    "15tr": 7,
}


def wordlevel_tokenizer(vocab: dict[str, int], name: str = "tiny-wordlevel"):
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", name_or_path=name)


def wordpiece_tokenizer(vocab: dict[str, int], name: str = "tiny-wordpiece"):
    """WordPiece with an explicit vocabulary, so token counts are fully determined."""
    tokenizer = Tokenizer(models.WordPiece(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.decoder = decoders.WordPiece(prefix="##")
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", name_or_path=name)


def test_wordlevel_metrics_are_hand_computable():
    tokenizer = wordlevel_tokenizer(WORDLEVEL_VOCAB)
    # "ăn bún 45k": "bún" is unknown, so "[UNK]" and the word is dropped on decode.
    report = audit_tokenizer(tokenizer, ["ăn phở 45k", "ăn bún 45k"])

    assert report.name == "tiny-wordlevel"
    assert report.vocab_size == 4
    assert report.n_texts == 2
    assert report.n_words == 6
    assert report.mean_tokens_per_text == 3.0
    assert report.p50_tokens_per_text == 3.0
    assert report.p95_tokens_per_text == 3.0
    assert report.max_tokens_per_text == 3
    assert report.fertility == 1.0
    assert report.pct_words_split == 0.0
    assert report.unk_token == "[UNK]"
    assert report.unk_rate == pytest.approx(1 / 6)
    assert report.roundtrip_exact_match_rate == 0.5
    assert report.roundtrip_failures == ({"text": "ăn bún 45k", "roundtrip": "ăn 45k"},)
    assert report.n_amounts == 2
    assert report.mean_amount_tokens == 1.0
    assert report.max_amount_tokens == 1


def test_to_dict_is_json_serializable_with_lists():
    report = audit_tokenizer(wordlevel_tokenizer(WORDLEVEL_VOCAB), ["ăn bún 45k"])
    payload = json.loads(json.dumps(report.to_dict(), ensure_ascii=False))

    assert payload["roundtrip_failures"] == [{"text": "ăn bún 45k", "roundtrip": "ăn 45k"}]
    assert payload["name"] == "tiny-wordlevel"
    # Whole-word vocabulary: every single character is unknown, and tuples become lists.
    assert isinstance(payload["unk_characters"], list)
    assert set(payload["unk_characters"]) == {"ă", "n", "b", "ú", "4", "5", "k"}


def test_wordpiece_fragmentation_fertility_and_clean_roundtrip():
    tokenizer = wordpiece_tokenizer(WORDPIECE_VOCAB)
    # "phở" is the only word that fragments: ph + ##ở -> 4 and 3 tokens per text.
    report = audit_tokenizer(tokenizer, ["ăn phở 45k", "nhận lương 15tr"])

    assert report.vocab_size == 8
    assert report.n_words == 6
    assert report.mean_tokens_per_text == 3.5
    assert report.p50_tokens_per_text == 3.5
    assert report.p95_tokens_per_text == pytest.approx(3.95)
    assert report.max_tokens_per_text == 4
    assert report.fertility == pytest.approx(7 / 6)
    assert report.pct_words_split == pytest.approx(100 / 6)
    assert report.unk_rate == 0.0
    assert report.roundtrip_exact_match_rate == 1.0
    assert report.roundtrip_failures == ()
    assert report.n_amounts == 2
    assert report.mean_amount_tokens == 1.0


def test_shorthand_fragmentation_counts_case_and_punctuation_insensitive_matches():
    vocab = {**WORDPIECE_VOCAB, "c": 8, "##k": 9, "grab": 10}
    # "CK," and "ck" both match shorthand "ck" (c + ##k); "grab" is one token.
    report = audit_tokenizer(wordpiece_tokenizer(vocab), ["CK, 45k", "ck grab 45k"])

    assert report.n_shorthand == 3
    assert report.mean_shorthand_tokens == pytest.approx(5 / 3)
    assert report.pct_shorthand_split == pytest.approx(200 / 3)


def test_probes_report_tokens_decoded_pieces_count_and_unk():
    report = audit_tokenizer(
        wordpiece_tokenizer(WORDPIECE_VOCAB), ["ăn phở 45k"], probes=["phở", "bún 45k"]
    )

    assert report.probes == (
        {"text": "phở", "tokens": ["ph", "##ở"], "pieces": ["ph", "##ở"], "n_tokens": 2,
         "has_unk": False},
        {"text": "bún 45k", "tokens": ["[UNK]", "45k"], "pieces": ["", "45k"], "n_tokens": 2,
         "has_unk": True},
    )  # fmt: skip


def test_amount_regex_matches_cu_slang_for_million():
    assert [m.group(0) for m in AMOUNT_PATTERN.finditer("ck mẹ 2 củ")] == ["2 củ"]


def test_roundtrip_failures_capped_by_max_examples():
    tokenizer = wordpiece_tokenizer(WORDPIECE_VOCAB)
    texts = ["ăn phở 45k", "ăn phở 45k bún", "nhận lương 15tr bún"]
    report = audit_tokenizer(tokenizer, texts, max_examples=1)

    assert report.roundtrip_exact_match_rate == pytest.approx(1 / 3)
    assert report.roundtrip_failures == ({"text": "ăn phở 45k bún", "roundtrip": "ăn phở 45k"},)
    # max_examples=0 keeps no examples but still scores the round trips.
    assert audit_tokenizer(tokenizer, texts, max_examples=0).roundtrip_failures == ()


def test_unk_characters_report_uncovered_diacritics():
    vocab = {"[UNK]": 0, "ơ": 1, "n": 2, "4": 3, "5": 4, "k": 5, "ơn": 6, "45k": 7}
    report = audit_tokenizer(wordlevel_tokenizer(vocab), ["ơn 45k", "ắ"])

    assert report.unk_characters == ("ắ",)
    assert report.unk_rate == pytest.approx(1 / 3)


def test_covering_tokenizer_reports_no_unk_characters():
    vocab = {"[UNK]": 0, "ơ": 1, "n": 2, "4": 3, "5": 4, "k": 5, "ơn": 6, "45k": 7, "ắ": 8}
    report = audit_tokenizer(wordlevel_tokenizer(vocab), ["ơn 45k", "ắ"])

    assert report.unk_characters == ()
    assert report.unk_rate == 0.0


def test_empty_corpus_yields_zeroed_metrics():
    report = audit_tokenizer(wordlevel_tokenizer(WORDLEVEL_VOCAB), [])

    assert report.n_texts == 0
    assert report.n_words == 0
    assert report.mean_tokens_per_text == 0.0
    assert report.p95_tokens_per_text == 0.0
    assert report.max_tokens_per_text == 0
    assert report.fertility == 0.0
    assert report.pct_words_split == 0.0
    assert report.roundtrip_exact_match_rate == 0.0
    assert report.n_amounts == 0
    assert report.mean_amount_tokens == 0.0


AMOUNT_FORMS = [
    "45k",
    "50K",
    "15tr",
    "1tr5",
    "1,5tr",
    "2.5tr",
    "45.000đ",
    "500đ",
    "1.500.000vnđ",
    "200 nghìn",
    "200 ngàn",
    "2 triệu",
]


@pytest.mark.parametrize("amount", AMOUNT_FORMS)
def test_amount_regex_matches_canonical_forms(amount):
    matches = [match.group(0) for match in AMOUNT_PATTERN.finditer(f"ăn {amount} nhé")]
    assert matches == [amount]


@pytest.mark.parametrize("text", ["ăn 1.5 nhé", "năm 2026 có 12 tháng", "5 người", "abc45k"])
def test_amount_regex_ignores_plain_numbers(text):
    assert [match.group(0) for match in AMOUNT_PATTERN.finditer(text)] == []


def test_amount_token_statistics_count_every_amount_match():
    vocab = {
        "[UNK]": 0,
        "ăn": 1,
        "45k": 2,
        "45.000đ": 3,
        "1tr5": 4,
        "1,5tr": 5,
        "200": 6,
        "nghìn": 7,
        "2": 8,
        "triệu": 9,
        "500đ": 10,
        "50K": 11,
    }
    texts = [
        "ăn 45k",
        "tiền nhà 45.000đ",
        "vay 1tr5",
        "trả 1,5tr",
        "mua sữa 200 nghìn",
        "thuê 2 triệu",
        "trà đá 500đ",
        "cf vs Lan 50K",
    ]
    report = audit_tokenizer(wordlevel_tokenizer(vocab), texts)

    # "200 nghìn" and "2 triệu" cost two tokens each; the other six cost one.
    assert report.n_amounts == 8
    assert report.mean_amount_tokens == pytest.approx(10 / 8)
    assert report.max_amount_tokens == 2


def test_load_texts_reads_text_field_in_file_then_record_order(tmp_path):
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    write_jsonl(
        first,
        [
            {"id": "1", "text": "ăn phở 45k", "source": "synthetic:g1"},
            {"id": "2", "text": "nhận lương 15tr", "source": "manual", "prompt_id": "p1"},
        ],
    )
    write_jsonl(second, [{"id": "3", "text": "cf vs Lan 60k", "source": "manual"}])

    assert load_texts([first, second]) == ["ăn phở 45k", "nhận lương 15tr", "cf vs Lan 60k"]
    assert load_texts([second]) == ["cf vs Lan 60k"]


@pytest.mark.parametrize(
    "record",
    [
        {"id": "1", "source": "manual"},
        {"id": "1", "text": "", "source": "manual"},
        {"id": "1", "text": "   ", "source": "manual"},
        {"id": "1", "text": 42, "source": "manual"},
    ],
)
def test_load_texts_rejects_missing_or_blank_text(tmp_path, record):
    path = tmp_path / "bad.jsonl"
    write_jsonl(path, [record])

    with pytest.raises(ValueError, match=r"bad\.jsonl: record 1"):
        load_texts([path])


CORPUS_TEXTS = ["ăn phở 45k", "ăn bún 45k", "nhận lương 15tr", "cf vs Lan 60k"]
SCRIPT_VOCAB = {
    "[UNK]": 0,
    "ăn": 1,
    "phở": 2,
    "45k": 3,
    "nhận": 4,
    "lương": 5,
    "15tr": 6,
    "cf": 7,
    "vs": 8,
    "Lan": 9,
    "60k": 10,
}


@pytest.fixture(scope="module")
def tiny_tokenizer_dir(tmp_path_factory) -> Path:
    """A locally saved tokenizer, so the script runs without touching the Hub."""
    path = tmp_path_factory.mktemp("tokenizers") / "tiny-wordlevel"
    wordlevel_tokenizer(SCRIPT_VOCAB).save_pretrained(path)
    return path


def run_script(
    corpus: Path, tokenizer: Path, out: Path, *extra: str
) -> subprocess.CompletedProcess:
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--corpus",
            str(corpus),
            "--tokenizer",
            str(tokenizer),
            "--out",
            str(out),
            *extra,
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=SCRIPT.parents[1],
    )


def test_script_prints_table_and_writes_json_report(tmp_path, tiny_tokenizer_dir):
    corpus = tmp_path / "notes.jsonl"
    records = [
        {"id": str(i), "text": text, "source": "manual"} for i, text in enumerate(CORPUS_TEXTS)
    ]
    write_jsonl(corpus, records)
    out = tmp_path / "report.json"

    result = run_script(corpus, tiny_tokenizer_dir, out)

    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    header_index = next(i for i, line in enumerate(lines) if line.startswith("tokenizer"))
    assert lines[header_index].split() == [
        "tokenizer",
        "vocab",
        "texts",
        "words",
        "tok/txt",
        "p50",
        "p95",
        "max",
        "fert",
        "%split",
        "unk%",
        "rt%",
        "amt",
        "amt/txt",
        "slang/w",
        "%slsplit",
    ]
    assert [cell.strip("-") for cell in lines[header_index + 1].split()] == [""] * 16

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["corpus"] == [str(corpus)]
    (report,) = payload["reports"]
    assert Path(report["name"]).name == "tiny-wordlevel"
    assert report["n_texts"] == 4
    assert report["n_words"] == 13
    assert report["unk_rate"] == pytest.approx(1 / 13)
    assert report["roundtrip_exact_match_rate"] == 0.75
    assert report["roundtrip_failures"] == [{"text": "ăn bún 45k", "roundtrip": "ăn 45k"}]
    assert report["n_amounts"] == 4


def test_script_refuses_to_overwrite_out_unless_asked(tmp_path, tiny_tokenizer_dir):
    corpus = tmp_path / "notes.jsonl"
    write_jsonl(corpus, [{"id": "1", "text": "ăn phở 45k", "source": "manual"}])
    out = tmp_path / "report.json"
    out.write_text("{}", encoding="utf-8")

    refused = run_script(corpus, tiny_tokenizer_dir, out)

    assert refused.returncode != 0
    assert "refusing to overwrite" in refused.stderr
    assert out.read_text(encoding="utf-8") == "{}"

    replaced = run_script(corpus, tiny_tokenizer_dir, out, "--overwrite")

    assert replaced.returncode == 0, replaced.stderr
    assert json.loads(out.read_text(encoding="utf-8"))["reports"][0]["n_texts"] == 1


def test_script_reports_empty_corpus(tmp_path, tiny_tokenizer_dir):
    corpus = tmp_path / "empty.jsonl"
    corpus.write_text("", encoding="utf-8")

    result = run_script(corpus, tiny_tokenizer_dir, tmp_path / "report.json")

    assert result.returncode != 0
    assert "no texts found" in result.stderr
