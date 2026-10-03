"""Frozen-v1 value-head training: only the value head trains; all else stays bit-identical."""

from __future__ import annotations

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import BertConfig, PreTrainedTokenizerFast

from gidi.modeling.model import GidiMultiTaskModel
from gidi.modeling.value import GidiValueModel, prepare_value
from gidi.training.train_value_head import (
    VALUE_PREFIX,
    ValueHeadConfig,
    build_optimizer,
    check_frozen_grads,
    check_optimizer_params,
    check_value_encoder_clone,
    fit_value_head,
    freeze_all_but_value_head,
    frozen_state_sha256,
)

WORDS = ["cơm", "tấm", "100", "mượn", "chú", "hai", "5", "xị", "ăn", "2", "tô", "phở", "70"]
VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", *WORDS]


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


H = 24  # tiny hidden size; divisible by the value adapter's 12 heads


def tiny_model(arch: str = "linear") -> GidiValueModel:
    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=len(VOCAB),
        hidden_size=H,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=2 * H,
        max_position_embeddings=64,
    )
    base = GidiMultiTaskModel.from_config(config)
    model = GidiValueModel.from_v1(base, seed=1, value_head_arch=arch)
    model.train()  # fit_value_head must switch the whole model to eval mode itself
    return model


def rec(rid: str, text: str, piece: str, status: str = "complete", type_: str = "expense") -> dict:
    start = text.index(piece)
    return {
        "id": rid,
        "text": text,
        "type": type_,
        "target": None,
        "value": {"text": piece, "start": start, "end": start + len(piece)},
        "value_status": status,
        "value_provenance": "human",
    }


def tiny_data():
    records = [
        rec("a", "ăn 2 tô phở 70", "70"),
        rec("b", "cơm tấm 100", "100"),
        rec("c", "mượn chú hai 5 xị", "5 xị", type_="borrow"),
        rec("d", "ăn phở 70", "70"),
        rec("e", "mượn chú hai 5 xị", "5 xị", status="uncertain", type_="borrow"),
    ]
    return prepare_value(tiny_tokenizer(), records, 32)


def test_value_encoder_starts_as_v1_clone_and_trains_without_touching_v1():
    model = tiny_model("encoder-mlp-crf")
    clone_check = check_value_encoder_clone(model)
    assert clone_check is not None and clone_check["ok"]
    before = {k: v.clone() for k, v in model.state_dict().items()}
    sha_before = frozen_state_sha256(model)
    cfg = ValueHeadConfig(
        seed=1,
        train_file="",
        epochs=3,
        batch_size=2,
        lr=1e-2,
        encoder_lr=1e-3,
        head_arch="encoder-mlp-crf",
    )

    rows, checks = fit_value_head(model, tiny_data(), cfg, torch.device("cpu"))

    optimizer_params = checks["optimizer"]["optimizer_params"]
    assert all(n.startswith(VALUE_PREFIX) for n in optimizer_params)
    assert any(n.startswith(VALUE_PREFIX + "encoder.") for n in optimizer_params)
    assert checks["first_backward_frozen_grads"]["n_frozen_params"] == len(
        [n for n, _ in model.named_parameters() if not n.startswith(VALUE_PREFIX)]
    )
    assert frozen_state_sha256(model) == sha_before
    state = model.state_dict()
    for key, tensor in state.items():
        if not key.startswith(VALUE_PREFIX):
            assert torch.equal(tensor, before[key]), key
    enc_keys = [k for k in state if k.startswith(VALUE_PREFIX + "encoder.")]
    assert any(not torch.equal(state[k], before[k]) for k in enc_keys)
    # the v1 encoder is untouched, so its clone has now diverged from it
    v1 = model.encoder.state_dict()
    assert any(
        not torch.equal(v1[k.removeprefix(VALUE_PREFIX + "encoder.")], state[k]) for k in enc_keys
    )


@pytest.mark.parametrize(
    ("arch", "n_trainable", "n_tensors"),
    [
        ("linear", H * 3 + 3, 2),
        ("mlp", H * 256 + 256 + 256 * 3 + 3, 4),
        ("mlp-crf", H * 256 + 256 + 256 * 3 + 3 + 3 + 3 + 9, 7),
        (
            "adapter-mlp-crf",
            H * 256
            + 256
            + 256 * 3
            + 3
            + 15  # mlp + crf
            + 4 * H * H
            + 4 * H  # attention in/out projections
            + 2 * H * 1024
            + 1024
            + H  # FFN
            + 4 * H,  # two LayerNorms
            19,
        ),
    ],
)
def test_only_value_head_trains_and_everything_else_is_bit_identical(arch, n_trainable, n_tensors):
    model = tiny_model(arch)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    sha_before = frozen_state_sha256(model)
    cfg = ValueHeadConfig(seed=1, train_file="", epochs=3, batch_size=2, lr=1e-2, head_arch=arch)

    rows, checks = fit_value_head(model, tiny_data(), cfg, torch.device("cpu"))

    value_keys = [k for k in model.state_dict() if k.startswith(VALUE_PREFIX)]
    assert len(value_keys) == n_tensors
    assert checks["optimizer"]["optimizer_params"] == sorted(value_keys)
    assert checks["optimizer"]["n_trainable"] == n_trainable
    assert checks["frozen_state_sha256_before"] == checks["frozen_state_sha256_after"] == sha_before
    assert not any(m.training for m in model.modules())
    assert len(rows) == 3
    for key, tensor in model.state_dict().items():
        if key.startswith(VALUE_PREFIX):
            assert not torch.equal(tensor, before[key]), key
        else:
            assert torch.equal(tensor, before[key]), key
    assert frozen_state_sha256(model) == sha_before


def test_optimizer_holds_exactly_the_value_head():
    model = tiny_model()
    freeze_all_but_value_head(model)
    optimizer = build_optimizer(model, ValueHeadConfig(seed=1, train_file=""))
    held = [p for g in optimizer.param_groups for p in g["params"]]
    assert {id(p) for p in held} == {id(model.value_head.weight), id(model.value_head.bias)}
    assert check_optimizer_params(optimizer, model)["ok"]

    leaky = torch.optim.AdamW(model.parameters())  # includes every frozen tensor
    with pytest.raises(RuntimeError, match="optimizer holds"):
        check_optimizer_params(leaky, model)


def test_grad_check_catches_an_unfrozen_encoder():
    model = tiny_model()
    freeze_all_but_value_head(model)
    data = tiny_data()
    ids, mask = data.encoded.input_ids, data.encoded.attention_mask
    model(ids, mask)[2].sum().backward()
    assert check_frozen_grads(model)["ok"]

    model.zero_grad(set_to_none=True)
    for p in model.encoder.parameters():
        p.requires_grad = True
    model(ids, mask)[2].sum().backward()
    with pytest.raises(RuntimeError, match="non-zero gradient"):
        check_frozen_grads(model)


def test_frozen_hash_ignores_the_value_head_but_sees_every_other_tensor():
    model = tiny_model()
    sha = frozen_state_sha256(model)
    with torch.no_grad():
        model.value_head.weight.add_(1.0)
    assert frozen_state_sha256(model) == sha
    with torch.no_grad():
        model.tag_head.bias.add_(1e-3)
    assert frozen_state_sha256(model) != sha
