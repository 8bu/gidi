import unicodedata

import pytest
import torch

from gidi.annotation.schema import load_config
from gidi.modeling.preprocessing import (
    IGNORE_INDEX,
    TAG_B,
    TAG_I,
    TAG_O,
    TAGS,
    TYPES,
    encode,
    spans_from_tags,
)
from gidi.modeling.tokenization import load_tokenizer

BAMIBERT = "Qualcomm-AI-Research/BamiBERT"
MINILM = "microsoft/Multilingual-MiniLM-L12-H384"


@pytest.fixture(scope="module", params=[BAMIBERT, MINILM], ids=["bamibert", "minilm"])
def tokenizer(request):
    try:
        return load_tokenizer(request.param)
    except Exception as e:  # not cached and no network
        pytest.skip(f"{request.param} not loadable offline: {e}")


def target_of(text: str, span: str) -> dict:
    start = text.index(span)
    return {"text": span, "start": start, "end": start + len(span)}


def tag_names(tags: torch.Tensor) -> list[str]:
    return ["-" if t == IGNORE_INDEX else TAGS[t] for t in tags.tolist()]


def test_label_order_matches_annotation_config():
    assert load_config().types == TYPES
    assert TAGS[TAG_O] == "O" and TAGS[TAG_B] == "B-TARGET" and TAGS[TAG_I] == "I-TARGET"


def test_single_token_span_tags_and_decodes(tokenizer):
    text = "Chị Thảo trả lại 1tr"
    enc = encode(tokenizer, [text], [target_of(text, "Thảo")])
    tags = tag_names(enc.tag_labels[0])
    assert tags[0] == "-" and tags[-1] == "-"  # <s> ... </s>
    assert tags.count("B-TARGET") == 1 and "I-TARGET" not in tags
    assert tags[2] == "B-TARGET"  # second real token, after "Chị"
    decoded = spans_from_tags(enc.offsets[0], enc.tag_labels[0].clamp(min=0).tolist(), text)
    assert decoded == {"text": "Thảo", "start": 4, "end": 8}


@pytest.mark.parametrize(
    ("name", "expected"), [(BAMIBERT, (3, 8)), (MINILM, (4, 8))], ids=["bamibert", "minilm"]
)
def test_raw_offsets_differ_by_tokenizer_but_tags_agree(name, expected):
    try:
        tok = load_tokenizer(name)
    except Exception as e:
        pytest.skip(f"{name} not loadable offline: {e}")
    text = "Chị Thảo trả lại 1tr"
    enc = encode(tok, [text], [target_of(text, "Thảo")])
    # byte-level offsets include the leading space, SentencePiece ones do not
    assert enc.offsets[0][2] == expected
    assert tag_names(enc.tag_labels[0])[2] == "B-TARGET"


def test_multi_token_span_round_trips(tokenizer):
    text = "nhận tiền từ bà ngoại cho 500k"
    target = target_of(text, "bà ngoại")
    enc = encode(tokenizer, [text], [target])
    tags = tag_names(enc.tag_labels[0])
    assert tags.count("B-TARGET") == 1 and tags.count("I-TARGET") >= 1
    b = tags.index("B-TARGET")
    assert all(t == "I-TARGET" for t in tags[b + 1 : b + tags.count("I-TARGET") + 1])
    decoded = spans_from_tags(enc.offsets[0], enc.tag_labels[0].clamp(min=0).tolist(), text)
    assert decoded == target


def test_null_target_tags_every_real_token_o(tokenizer):
    text = "mua ccq quỹ đầu tư 2tr"
    enc = encode(tokenizer, [text], [None])
    tags = tag_names(enc.tag_labels[0])
    assert tags[0] == "-" and tags[-1] == "-"
    assert set(tags[1:-1]) == {"O"}
    assert spans_from_tags(enc.offsets[0], enc.tag_labels[0].clamp(min=0).tolist(), text) is None


def test_special_and_padding_tokens_are_ignored(tokenizer):
    texts = ["Nam", "chuyển khoản cho anh Hộ tiền nhà tháng mười"]
    targets = [target_of(texts[0], "Nam"), target_of(texts[1], "Hộ")]
    enc = encode(tokenizer, texts, targets)
    mask = enc.attention_mask
    assert mask[0].sum() < mask[1].sum() == mask.shape[1]
    labels = enc.tag_labels
    assert (labels[mask == 0] == IGNORE_INDEX).all()  # padding
    assert labels[0, 0] == IGNORE_INDEX and labels[1, 0] == IGNORE_INDEX  # <s>
    last = int(mask[0].sum()) - 1
    assert labels[0, last] == IGNORE_INDEX and labels[1, -1] == IGNORE_INDEX  # </s>
    assert (labels[mask == 1][labels[mask == 1] != IGNORE_INDEX] >= 0).all()
    assert enc.offsets[0][0] == (0, 0) and enc.offsets[0][-1] == (0, 0)
    assert all(len(o) == mask.shape[1] for o in enc.offsets)


@pytest.mark.parametrize(
    ("text", "span"),
    [
        ("Hộ chuyển 3tr", "Hộ"),  # ộ: one code point in NFC
        ("trả lại anh Nguyễn Hữu Tài 2tr", "Nguyễn Hữu Tài"),
        ("cho chị Thương mượn 1tr", "chị Thương"),
    ],
)
def test_vietnamese_nfc_offsets_are_code_points(tokenizer, text, span):
    text = unicodedata.normalize("NFC", text)
    target = target_of(text, span)
    enc = encode(tokenizer, [text], [target])
    decoded = spans_from_tags(enc.offsets[0], enc.tag_labels[0].clamp(min=0).tolist(), text)
    assert decoded == target
    assert text[decoded["start"] : decoded["end"]] == span


def test_decomposed_nfd_input_is_rejected_then_works_after_nfc(tokenizer):
    nfc = "trả lại anh Hữu 2tr"
    nfd = unicodedata.normalize("NFD", nfc)
    assert nfd != nfc and len(nfd) > len(nfc)
    with pytest.raises(ValueError, match="NFC"):
        encode(tokenizer, [nfd], [None])
    target = target_of(nfc, "Hữu")
    enc = encode(tokenizer, [unicodedata.normalize("NFC", nfd)], [target])
    decoded = spans_from_tags(enc.offsets[0], enc.tag_labels[0].clamp(min=0).tolist(), nfc)
    assert decoded == target


def test_target_offsets_must_reproduce_target_text(tokenizer):
    with pytest.raises(ValueError, match="offsets"):
        encode(tokenizer, ["Chị Thảo trả lại"], [{"text": "Thảo", "start": 0, "end": 4}])


def test_truncation_that_cuts_the_span_is_counted_not_fatal(tokenizer):
    text = "chuyển khoản tiền nhà tháng này cho anh Nguyễn Hữu Tài nhé"
    target = target_of(text, "Nguyễn Hữu Tài")
    enc = encode(tokenizer, [text], [target], max_length=12)
    assert enc.input_ids.shape[1] == 12
    assert enc.truncated == [True]
    assert enc.span_truncated == [True]
    assert enc.n_truncated == 1 and enc.n_span_truncated == 1
    tags = tag_names(enc.tag_labels[0])
    assert tags[0] == "-" and tags[-1] == "-"
    surviving = spans_from_tags(enc.offsets[0], enc.tag_labels[0].clamp(min=0).tolist(), text)
    if surviving is not None:  # any surviving tagged tokens must lie inside the gold span
        assert target["start"] <= surviving["start"] < surviving["end"] <= target["end"]


def test_truncation_after_the_span_leaves_it_intact(tokenizer):
    text = "Nam trả lại tiền ăn trưa hôm qua ở quán cơm tấm sườn bì chả gần nhà"
    target = target_of(text, "Nam")
    enc = encode(tokenizer, [text], [target], max_length=10)
    assert enc.truncated == [True]
    assert enc.span_truncated == [False]
    decoded = spans_from_tags(enc.offsets[0], enc.tag_labels[0].clamp(min=0).tolist(), text)
    assert decoded == target


def test_token_straddling_span_edge_raises_when_strict(tokenizer):
    text = "Thảo trả lại 1tr"
    target = target_of(text, "Th")  # the token "Thảo" is not aligned with this span
    with pytest.raises(ValueError, match="align"):
        encode(tokenizer, [text], [target])
    enc = encode(tokenizer, [text], [target], strict=False)
    assert enc.boundary_mismatch == [True] and enc.n_boundary_mismatch == 1


def test_spans_from_tags_hand_built():
    text = "Chị bà ngoại Thảo"
    # byte-level style: leading spaces belong to the token; (0, 0) marks <s> and </s>
    offsets = [(0, 0), (0, 3), (3, 6), (6, 12), (12, 17), (0, 0)]
    tags = [TAG_O, TAG_O, TAG_B, TAG_I, TAG_B, TAG_O]
    assert spans_from_tags(offsets, tags, text) == {"text": "bà ngoại", "start": 4, "end": 12}
    # I without a B starts a span; later B/I after it is ignored
    assert spans_from_tags(offsets, [0, 0, TAG_I, TAG_I, TAG_O, 0], text)["text"] == "bà ngoại"
    assert spans_from_tags(offsets, [0, TAG_B, TAG_O, TAG_B, TAG_I, 0], text) == {
        "text": "Chị",
        "start": 0,
        "end": 3,
    }
    assert spans_from_tags(offsets, [TAG_O] * 6, text) is None
    # special positions (0, 0) are never part of a span even if tagged
    assert spans_from_tags(offsets, [TAG_B, 0, 0, 0, 0, TAG_I], text) is None
