import json

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from torch.nn import functional as F
from transformers import BertConfig, PreTrainedTokenizerFast

from gidi.modeling.checkpoint import load_checkpoint, save_checkpoint
from gidi.modeling.model import GidiMultiTaskModel
from gidi.modeling.preprocessing import IGNORE_INDEX, TAGS, TYPES, encode

VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "chị", "thảo", "trả", "lại", "1tr", "nam"]


def tiny_config() -> BertConfig:
    return BertConfig(
        vocab_size=len(VOCAB),
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
    )


def tiny_tokenizer() -> PreTrainedTokenizerFast:
    tok = Tokenizer(models.WordLevel({t: i for i, t in enumerate(VOCAB)}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )


@pytest.fixture
def model() -> GidiMultiTaskModel:
    torch.manual_seed(0)
    return GidiMultiTaskModel.from_config(tiny_config())


def test_forward_shapes_and_tuple(model):
    ids = torch.randint(4, len(VOCAB), (3, 7))
    mask = torch.ones_like(ids)
    out = model(ids, mask)
    assert isinstance(out, tuple) and len(out) == 2
    type_logits, tag_logits = out
    assert type_logits.shape == (3, len(TYPES))
    assert tag_logits.shape == (3, 7, len(TAGS))


def test_padding_does_not_change_real_token_logits(model):
    model.eval()
    ids = torch.tensor([[2, 4, 5, 3]])
    padded = torch.tensor([[2, 4, 5, 3, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]])
    a_type, a_tags = model(ids, torch.ones_like(ids))
    b_type, b_tags = model(padded, mask)
    assert torch.allclose(a_type, b_type, atol=1e-5)
    assert torch.allclose(a_tags, b_tags[:, :4], atol=1e-5)


def test_joint_loss_backprops_into_encoder_and_both_heads(model):
    tokenizer = tiny_tokenizer()
    texts = ["chị thảo trả lại 1tr", "nam"]
    targets = [{"text": "thảo", "start": 4, "end": 8}, None]
    enc = encode(tokenizer, texts, targets, max_length=16)
    types = torch.tensor([TYPES.index("repayment_in"), TYPES.index("expense")])
    type_logits, tag_logits = model(enc.input_ids, enc.attention_mask)
    type_loss = F.cross_entropy(type_logits, types)
    tag_loss = F.cross_entropy(
        tag_logits.reshape(-1, len(TAGS)), enc.tag_labels.reshape(-1), ignore_index=IGNORE_INDEX
    )
    (type_loss + tag_loss).backward()
    assert torch.isfinite(type_loss) and torch.isfinite(tag_loss)
    for name in (
        "type_head.weight",
        "tag_head.weight",
        "encoder.embeddings.word_embeddings.weight",
    ):
        grad = dict(model.named_parameters())[name].grad
        assert grad is not None and grad.abs().sum() > 0, name


def test_checkpoint_round_trip(model, tmp_path):
    tokenizer = tiny_tokenizer()
    out = save_checkpoint(model, tokenizer, tmp_path / "ckpt", {"best_epoch": 3, "seed": 7})
    assert (out / "model.safetensors").is_file()
    meta = json.loads((out / "gidi_checkpoint.json").read_text(encoding="utf-8"))
    assert meta["types"] == list(TYPES) and meta["tags"] == list(TAGS)
    assert meta["max_length"] == 32 and meta["best_epoch"] == 3 and meta["seed"] == 7
    assert "encoder" in meta

    loaded, loaded_tokenizer, loaded_meta = load_checkpoint(out)
    assert loaded_meta == meta
    model.eval()
    ids = torch.tensor([[2, 4, 5, 3], [2, 6, 3, 0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    expected = model(ids, mask)
    actual = loaded(ids, mask)
    assert torch.equal(expected[0], actual[0]) and torch.equal(expected[1], actual[1])
    assert loaded_tokenizer("chị thảo")["input_ids"] == tokenizer("chị thảo")["input_ids"]
    assert not loaded.training
