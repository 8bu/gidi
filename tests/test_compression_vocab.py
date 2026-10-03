"""Vocabulary pruning: closure, exact segmentation of kept words, byte fallback, model rows."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from transformers import PreTrainedTokenizerFast, RobertaConfig

from gidi.compression.vocab import (
    BpeVocab,
    closure,
    is_closed,
    load_vocab_spec,
    prune_tokenizer_json,
    prune_vocab,
    remap_ids,
    token_category,
    write_vocab_spec,
)
from gidi.modeling.model import GidiMultiTaskModel

CORPUS = [
    "chuyển khoản tiền nhà tháng này 3tr5",
    "mua cà phê highlands coffee 45k",
    "trả nợ anh Thảo 500k",
    "grab đi làm 89,000",
    "cho em Linh mượn 1tr2",
    "nhận lương công ty tháng mười",
    "thanh toán hóa đơn điện nước 650k",
    "bách hóa xanh mua đồ ăn 230k",
    "momo nạp điện thoại 100k",
    "tiền ăn trưa với bạn 75k",
] * 6
SPECIALS = ["<s>", "<pad>", "</s>", "<unk>"]


def _train_tokenizer(vocab_size: int = 420) -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(CORPUS, trainer)
    tok.add_special_tokens(["<mask>"])
    tok.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[("<s>", tok.token_to_id("<s>")), ("</s>", tok.token_to_id("</s>"))],
    )
    return tok


@pytest.fixture(scope="module")
def tok_json() -> dict:
    return json.loads(_train_tokenizer().to_str())


@pytest.fixture(scope="module")
def bpe(tok_json: dict) -> BpeVocab:
    return BpeVocab.from_json(tok_json)


def _tokenizer_from(tok_json: dict) -> Tokenizer:
    return Tokenizer.from_str(json.dumps(tok_json))


def _word_ids(tokenizer: Tokenizer, word: str) -> list[int]:
    return tokenizer.encode(word, add_special_tokens=False).ids


def _keep_from_words(bpe: BpeVocab, tokenizer: Tokenizer, words: list[str]) -> list[int]:
    seen = {i for w in words for i in _word_ids(tokenizer, w)}
    return sorted(closure(bpe, seen | bpe.mandatory_ids()))


UNSEEN = [
    "khoản",
    "nhà trọ",
    "Highlands",
    "Phúc Long",
    "tiền điện 1tr25",
    "ghép phòng",
    "cà phê",
    "nguyễn văn a",
    "zzz qqq xyz",
    "漢字 ❤",
]


def test_synthetic_layout_matches_bamibert(bpe: BpeVocab) -> None:
    assert bpe.tokens[:4] == tuple(SPECIALS)
    assert bpe.tokens[-1] == "<mask>" and bpe.mask_id == bpe.size - 1
    assert len(bpe.base_ids) == 256
    assert bpe.orphans() == []


def test_closure_keeps_merge_parents(bpe: BpeVocab) -> None:
    rank = bpe.merge_rank()
    deepest = max(rank, key=lambda i: len(bpe.tokens[i]))
    keep = closure(bpe, {deepest})
    assert deepest in keep and is_closed(bpe, keep)
    for left, right in bpe.parents()[deepest]:
        assert left in keep and right in keep
    # a bare token without its parents is not closed
    assert not is_closed(bpe, {deepest})
    # closure only ever adds lower-rank tokens (parents precede children)
    assert all(rank.get(i, -1) <= rank[deepest] for i in keep)


def test_kept_words_segment_identically_after_pruning(tok_json: dict, bpe: BpeVocab) -> None:
    original = _tokenizer_from(tok_json)
    words = [w for line in CORPUS[:5] for w in line.split()]
    kept = _keep_from_words(bpe, original, words)
    assert 0 < len(kept) < bpe.size
    pruned = _tokenizer_from(prune_tokenizer_json(tok_json, kept))
    new_of = {old: new for new, old in enumerate(kept)}
    checked = 0
    for text in [*CORPUS, *UNSEEN]:
        for word in text.split():
            ids = _word_ids(original, word)
            if all(i in new_of for i in ids):
                assert _word_ids(pruned, word) == [new_of[i] for i in ids], word
                checked += 1
    assert checked > 30


def test_random_closed_keep_sets_are_exact_for_kept_words(tok_json: dict, bpe: BpeVocab) -> None:
    original = _tokenizer_from(tok_json)
    rng = random.Random(0)
    merged = sorted(bpe.merge_rank())
    texts = [*CORPUS, *UNSEEN]
    for _ in range(8):
        sample = rng.sample(merged, k=len(merged) // 3)
        kept = sorted(closure(bpe, set(sample) | bpe.mandatory_ids() | bpe.special_ids))
        pruned = _tokenizer_from(prune_tokenizer_json(tok_json, kept))
        new_of = {old: new for new, old in enumerate(kept)}
        for text in texts:
            for word in text.split():
                ids = _word_ids(original, word)
                if all(i in new_of for i in ids):
                    assert _word_ids(pruned, word) == [new_of[i] for i in ids]


def test_unkept_words_fall_back_to_pieces_without_unk(tok_json: dict, bpe: BpeVocab) -> None:
    original = _tokenizer_from(tok_json)
    kept = _keep_from_words(bpe, original, ["mua", "45k", "anh"])
    pruned = _tokenizer_from(prune_tokenizer_json(tok_json, kept))
    new_of = {old: new for new, old in enumerate(kept)}
    fell_back = 0
    for text in [*CORPUS, *UNSEEN]:
        for word in text.split():
            before = _word_ids(original, word)
            after = _word_ids(pruned, word)
            assert 3 not in after  # byte-level fallback: never <unk>
            assert pruned.decode(after, skip_special_tokens=False) == word
            if not all(i in new_of for i in before):
                fell_back += 1
                assert len(after) > len(before)
    assert fell_back > 10


def test_unclosed_keep_set_is_rejected(tok_json: dict, bpe: BpeVocab) -> None:
    rank = bpe.merge_rank()
    deepest = max(rank, key=lambda i: len(bpe.tokens[i]))
    bad = sorted({*bpe.mandatory_ids(), deepest})
    with pytest.raises(ValueError, match="closed"):
        prune_tokenizer_json(tok_json, bad)


def test_orphan_tokens_are_detected(tok_json: dict) -> None:
    broken = copy.deepcopy(tok_json)
    dropped = broken["model"]["merges"].pop()
    orphans = BpeVocab.from_json(broken).orphans()
    assert [BpeVocab.from_json(broken).tokens[i] for i in orphans] == ["".join(dropped)]


def test_spec_directory_roundtrip_keeps_special_ids(
    tmp_path: Path, tok_json: dict, bpe: BpeVocab
) -> None:
    original = _tokenizer_from(tok_json)
    src = tmp_path / "src"
    PreTrainedTokenizerFast(
        tokenizer_object=original,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        sep_token="</s>",
        pad_token="<pad>",
        cls_token="<s>",
        mask_token="<mask>",
    ).save_pretrained(src)
    kept = _keep_from_words(bpe, original, CORPUS[:4])
    write_vocab_spec(tmp_path / "spec", src, kept, policy="unit")
    tokenizer, loaded = load_vocab_spec(tmp_path / "spec")
    spec = json.loads((tmp_path / "spec" / "vocab_map.json").read_text(encoding="utf-8"))
    assert loaded == kept
    assert spec["n_kept"] == len(kept) == len(tokenizer)
    assert spec["removed_old_ids_count"] == bpe.size - len(kept)
    assert spec["special_tokens"] == {
        "<s>": 0,
        "<pad>": 1,
        "</s>": 2,
        "<unk>": 3,
        "<mask>": len(kept) - 1,
    }
    assert tokenizer.pad_token_id == 1 and tokenizer.mask_token_id == len(kept) - 1
    note = CORPUS[1]
    enc = tokenizer(note)["input_ids"]
    assert enc[0] == 0 and enc[-1] == 2
    old = original.encode(note).ids
    assert enc == [kept.index(i) for i in old]  # all pieces kept -> identical mapped ids
    assert tokenizer.decode(enc, skip_special_tokens=True) == note


def _tiny_model(vocab: int) -> GidiMultiTaskModel:
    config = RobertaConfig(
        vocab_size=vocab,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=40,
        type_vocab_size=1,
        pad_token_id=1,
    )
    torch.manual_seed(0)
    return GidiMultiTaskModel.from_config(config, encoder_name="tiny").eval()


def test_prune_vocab_copies_rows_and_everything_else_exactly(bpe: BpeVocab) -> None:
    model = _tiny_model(bpe.size)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    kept = sorted(closure(bpe, set(range(0, 300, 3)) | bpe.mandatory_ids()))
    pruned = prune_vocab(model, kept)
    # input untouched
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert model.encoder.config.vocab_size == bpe.size
    # new rows are the old rows
    word = "encoder.embeddings.word_embeddings.weight"
    new_state = pruned.state_dict()
    assert new_state[word].shape == (len(kept), 16)
    assert torch.equal(new_state[word], before[word][torch.tensor(kept)])
    assert pruned.encoder.config.vocab_size == len(kept)
    for key, tensor in before.items():
        if key != word:
            assert torch.equal(new_state[key], tensor), key
    assert pruned.num_types == model.num_types and pruned.encoder_name == "tiny"
    # a note whose ids are all kept gets identical logits
    old_ids = torch.tensor([[0, kept[5], kept[9], kept[40], 2]])
    mask = torch.ones_like(old_ids)
    new_ids = remap_ids(old_ids, kept)
    with torch.no_grad():
        for a, b in zip(model(old_ids, mask), pruned(new_ids, mask), strict=True):
            assert torch.equal(a, b)


def test_prune_vocab_requires_leading_specials_and_mask(bpe: BpeVocab) -> None:
    model = _tiny_model(bpe.size)
    with pytest.raises(ValueError, match="special"):
        prune_vocab(model, [0, 1, 2, 5, bpe.mask_id])  # <unk> (3) dropped
    with pytest.raises(ValueError, match="special"):
        prune_vocab(model, [0, 1, 2, 3, 7])  # <mask> dropped
    with pytest.raises(ValueError, match="ascending"):
        prune_vocab(model, [0, 1, 3, 2, bpe.mask_id])


def test_remap_ids_maps_and_raises_on_removed_ids() -> None:
    kept = [0, 1, 2, 3, 10, 11, 20]
    ids = torch.tensor([[0, 10, 20, 2], [11, 1, 1, 1]])
    assert remap_ids(ids, kept).tolist() == [[0, 4, 6, 2], [5, 1, 1, 1]]
    with pytest.raises(ValueError, match="not in the pruned"):
        remap_ids(torch.tensor([[0, 12]]), kept)
    with pytest.raises(ValueError, match="not in the pruned"):
        remap_ids(torch.tensor([[0, 99]]), kept)


def test_token_category_classifies_decoded_text(bpe: BpeVocab) -> None:
    cats = {
        i: token_category(t, special=i in bpe.special_ids, base=i in bpe.base_ids)
        for i, t in enumerate(bpe.tokens)
    }
    assert cats[0] == cats[bpe.mask_id] == "special"
    assert set(cats[i] for i in bpe.base_ids) == {"base_byte"}
    from gidi.compression.vocab import byte_to_char

    to_char = byte_to_char()

    def category(raw: bytes) -> str:
        return token_category("".join(to_char[b] for b in raw))

    assert category(" chuyển".encode()) == "vietnamese_latin"
    assert category(b" mua") == "ascii_letters"
    assert category(b" 45k") == "digits_numeric"
    assert category(b", ") == "ascii_punct_symbols"
    assert category("李明".encode()) == "non_latin_script"
    assert category("ạ".encode()[:2]) == "partial_byte"
